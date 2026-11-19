from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p2_build_panel as P2  # noqa: E402
import p4_features as F  # noqa: E402
import p4_pipeline as PP  # noqa: E402
import p4_screen as S  # noqa: E402

RES: list[dict] = []
FORBIDDEN = {"ACTIVE", "CHANGEC1", "ENDEFYMD", "BKCLASS", "COST", "FAILDATE", "y", "CERT", "REPDTE", "YEAR", "q_idx", "STNAME"}
CUTOFFS = [date(1990, 12, 31), date(2008, 12, 31), date(2019, 12, 31)]


def check(name: str, ok: bool, detail: str = "") -> None:
    RES.append({"test": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def gold() -> pl.LazyFrame:
    return pl.scan_parquet(C.DATA_GOLD / "features" / "**" / "*.parquet", hive_partitioning=True, hive_schema={C.PARTITION_KEY: pl.Int16})


def differing_cells(a: pl.DataFrame, b: pl.DataFrame, cols: list[str]) -> dict[str, int]:
    j = a.join(b, on=["CERT", "REPDTE"], how="inner", suffix="__b")
    out = {}
    for c in cols:
        x, z = pl.col(c).cast(pl.Float64).fill_nan(None), pl.col(f"{c}__b").cast(pl.Float64).fill_nan(None)
        d = (x - z).abs()
        out[c] = j.select(((x.is_null() != z.is_null()) | (d > 1e-12 * pl.max_horizontal(x.abs(), z.abs(), pl.lit(1.0))).fill_null(False)).sum()).item()
    return out


def main() -> None:
    fin, fail, inst = P2.load_bronze()
    safe = F.safe_columns()
    g = gold().collect(engine="streaming")
    silver = F.scan_silver().select("CERT", "REPDTE", "y").collect(engine="streaming")

    # ---- the table ----
    check("key (CERT, REPDTE) is unique", g.select(pl.struct("CERT", "REPDTE").n_unique()).item() == g.height, f"{g.height:,} rows")
    j = g.select("CERT", "REPDTE", "y").join(silver, on=["CERT", "REPDTE"], how="full", suffix="__s", coalesce=True)
    check("gold has exactly the silver rows, and the same y", j.height == silver.height == g.height
          and j.select(pl.col("y").ne_missing(pl.col("y__s")).sum()).item() == 0, f"y=1 rows {int(g['y'].sum()):,}")
    sc = PP.schema()
    used = sc["numeric"] + sc["flags"]
    check("no forbidden field and no _ column among the features", not (set(F.FEATURES) & FORBIDDEN) and not any(c.startswith("_") for c in F.FEATURES),
          f"{len(F.FEATURES)} candidates, {len(used)} selected")
    check("selected features are all in the registry", set(used) <= set(F.FEATURES))
    check("selected features exclude Institutions-snapshot fields", not ({"ACTIVE", "CHANGEC1", "BKCLASS"} & set(safe)))

    # ---- the label cannot reach the features ----
    silver_full = F.scan_silver()
    no_label = silver_full.drop("y", "_fail_date", "_exit_class", "_last_report", "_quarters_to_fail", "_days_to_fail", "_other_exit_4q", "_near_miss")
    a = F.build_features(no_label, fail, safe).collect(engine="streaming")
    d = differing_cells(a, g, F.FEATURES)
    check("features are identical when every label-side column is removed first", sum(d.values()) == 0, f"{a.height:,} rows, {len(F.FEATURES)} columns")

    # ---- truncation, end to end from bronze, with controls ----
    leaky = {"leak_lead": pl.col("NETINC_Q").shift(-1).over("CERT"),                                # next quarter's value
             "leak_centred": pl.col("EQ").rolling_mean(5, center=True, min_samples=1).over("CERT"),  # window that looks both ways
             "leak_all_time": pl.col("ASSET").rank("average") / pl.col("ASSET").count(),             # rank across all quarters
             "leak_bank_max": pl.col("ASSET").max().over("CERT")}                                    # peak over the bank's whole life
    def rebuild(cutoff: date | None) -> pl.DataFrame:
        panel, _ = P2.build_panel(fin, fail, inst, cutoff=cutoff)
        f2 = fail if cutoff is None else fail.filter(pl.col("FAILDATE") <= cutoff)
        return F.build_features(panel.lazy(), f2, safe, extra=leaky).collect(engine="streaming")

    full = rebuild(None)
    d0 = differing_cells(full.select("CERT", "REPDTE", *F.FEATURES), g, F.FEATURES)
    check("rebuild from bronze gives the stored feature table", sum(d0.values()) == 0)
    caught = {k: 0 for k in leaky}
    for cut in CUTOFFS:
        t = rebuild(cut)
        ref = full.filter(pl.col("REPDTE") <= cut)
        d = differing_cells(t, ref, F.FEATURES)
        bad = {k: v for k, v in d.items() if v}
        check(f"truncated at {cut}: {len(F.FEATURES)} features unchanged", t.height == ref.height and not bad, f"{t.height:,} rows {bad or ''}")
        for k, v in differing_cells(t, ref, list(leaky)).items():
            caught[k] += v
    check("control: all four leaky features are caught by truncation", all(v > 0 for v in caught.values()), str(caught))

    # ---- time split ----
    q = lambda s: P2.q_index(pl.lit(date.fromisoformat(s)))  # noqa: E731
    dev_q, test_q = pl.select(q(C.DEV_END)).item(), pl.select(q(C.TEST_START)).item()
    check("gap: 4 unused quarters between development and test", test_q - dev_q - 1 >= 4, f"{test_q - dev_q - 1} quarters")
    check("the last development label ends before the first test quarter", dev_q + C.LABEL_HORIZON_Q < test_q)
    counts = {s: g.filter(pl.col("_split") == s, pl.col("y") == 1)["CERT"].n_unique() for s in ("dev", "gap", "test")}
    check("test contains at least 50 failures", counts["test"] >= 50, str(counts))
    in_dev = set(g.filter(pl.col("_split") == "dev", pl.col("y") == 1)["CERT"].to_list())
    in_test = set(g.filter(pl.col("_split") == "test", pl.col("y") == 1)["CERT"].to_list())
    check("no bank's failure is labelled in both development and test", not (in_dev & in_test), f"{len(in_dev):,} and {len(in_test)} banks")
    check("each row has one split; unlabelled rows only after 2025Q2", g.filter(pl.col("y").is_null())["_split"].unique().to_list() == ["unlabelled"]
          and g.filter(pl.col("_split") == "test")["REPDTE"].max() == date.fromisoformat(C.LAST_MODEL_Q))
    pv_gap = pl.select(q(C.PIPE_VAL_START)).item() - pl.select(q(C.PIPE_TRAIN_END)).item() - 1
    check("Part 4 train and validation also have a 4-quarter gap", pv_gap >= 4, f"{pv_gap} quarters")
    check("screening reads nothing after its training window", S.load_train()["REPDTE"].max() == date.fromisoformat(C.PIPE_TRAIN_END))

    # ---- preprocessing ----
    Xtr, ytr, _ = PP.load_xy(C.PIPE_TRAIN_START, C.PIPE_TRAIN_END)
    Xva, _, _ = PP.load_xy(C.PIPE_VAL_START, C.PIPE_VAL_END)
    try:
        PP.load_xy(C.TEST_START, C.TEST_END)
        check("load_xy refuses the test period by default", False)
    except PermissionError:
        check("load_xy refuses the test period by default", True)

    pre = PP.build_preprocessor(sc["numeric"], sc["flags"]).fit(Xtr)
    num = pre.named_transformers_["num"]
    win, imp, sca = (num.named_steps[k] for k in ("winsor", "impute", "scale"))
    A = Xtr[sc["numeric"]].to_numpy(dtype=float)
    lo, up = np.nanquantile(A, C.WINSOR_LOWER, axis=0), np.nanquantile(A, C.WINSOR_UPPER, axis=0)
    capped = np.clip(A, lo, up)
    med = np.nanmedian(capped, axis=0)
    filled = np.where(np.isnan(capped), med, capped)
    check("fitted parameters equal an independent computation on training rows",
          np.allclose(win.lower_, lo) and np.allclose(win.upper_, up) and np.allclose(imp.statistics_, med)
          and np.allclose(sca.mean_, filled.mean(axis=0)) and np.allclose(sca.scale_, filled.std(axis=0)), f"{len(Xtr):,} training rows")
    pre2 = PP.build_preprocessor(sc["numeric"], sc["flags"]).fit(pd.concat([Xtr, Xva]))
    w2 = pre2.named_transformers_["num"].named_steps["winsor"]
    check("control: fitting with validation rows gives different parameters", not np.allclose(win.upper_, w2.upper_),
          f"{int((~np.isclose(win.upper_, w2.upper_)).sum())} of {len(win.upper_)} upper caps change")
    before = {k: np.array(v, copy=True) for k, v in (("lo", win.lower_), ("up", win.upper_), ("med", imp.statistics_), ("mean", sca.mean_), ("sd", sca.scale_))}
    corrupted = Xva.copy()
    corrupted[sc["numeric"]] = corrupted[sc["numeric"]] * 1e6
    Z_clean = pre.transform(Xva.iloc[:1000])
    pre.transform(corrupted)
    check("transforming extreme rows does not change what was fitted",
          all(np.array_equal(before[k], v) for k, v in (("lo", win.lower_), ("up", win.upper_), ("med", imp.statistics_), ("mean", sca.mean_), ("sd", sca.scale_))))
    Z_again = pre.transform(pd.concat([Xva.iloc[:1000], corrupted.iloc[1000:2000]]))[:1000]
    check("a row's output does not depend on other rows", np.array_equal(Z_clean, Z_again))
    Z = pre.transform(Xva)
    check("no nulls remain after preprocessing", not np.isnan(Z).any(), f"shape {Z.shape}")
    Zc = Z[:, : len(sc["numeric"])]
    check("capping holds on validation rows", (Zc <= (win.upper_ - sca.mean_) / sca.scale_ + 1e-9).all() and (Zc >= (win.lower_ - sca.mean_) / sca.scale_ - 1e-9).all())

    # ---- formulas, recomputed separately in pandas ----
    p = F.scan_silver().select(
        "CERT", "REPDTE", "q_idx", "STNAME", "ASSET", "EQ", "ORE", "NCLNLS", "LNATRES", "LNLSGR", "NETINC_Q", "NTLNLS_Q").collect(engine="streaming").to_pandas()
    p = p.sort_values(["CERT", "REPDTE"]).reset_index(drop=True)
    g4 = p.groupby("CERT")
    runs = (p["q_idx"] - g4["q_idx"].shift(3)) == 3
    p["eq_ta"] = (p["EQ"] / p["ASSET"]).where(p["ASSET"] > 0)
    p["npl"] = (p["NCLNLS"] / p["LNLSGR"]).where(p["LNLSGR"] > 0)
    npa, cushion = p["NCLNLS"] + p["ORE"], (p["EQ"] + p["LNATRES"]).clip(lower=0)
    p["texas"] = (npa / (npa + cushion)).where(npa + cushion > 0)
    p["roa_4q"] = (g4["NETINC_Q"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True) / p["ASSET"]).where(runs & (p["ASSET"] > 0))
    p["nco_4q"] = (g4["NTLNLS_Q"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True) / p["LNLSGR"]).where(runs & (p["LNLSGR"] > 0))
    p["loss_q4"] = (p["NETINC_Q"].lt(0).where(p["NETINC_Q"].notna()).astype(float).groupby(p["CERT"]).rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)).where(runs)
    p["size_pct"] = p.groupby("REPDTE")["ASSET"].rank(method="average") / p.groupby("REPDTE")["ASSET"].transform("count")
    alt = pl.from_pandas(p[["CERT", "REPDTE", "eq_ta", "npl", "texas", "roa_4q", "nco_4q", "loss_q4", "size_pct"]]).with_columns(pl.col("REPDTE").cast(pl.Date), pl.col("CERT").cast(pl.Int32))
    j = g.select("CERT", "REPDTE", "eq_ta", "npl", "texas", "roa_4q", "nco_4q", "loss_q4", "size_pct").join(alt, on=["CERT", "REPDTE"], suffix="__b")
    bad = {}
    for c in ["eq_ta", "npl", "texas", "roa_4q", "nco_4q", "loss_q4", "size_pct"]:
        x, z = pl.col(c).fill_nan(None), pl.col(f"{c}__b").fill_nan(None)
        n = j.select(((x.is_null() != z.is_null()) | ((x - z).abs() > 1e-9).fill_null(False)).sum()).item()
        if n:
            bad[c] = n
    check("7 features match an independent pandas computation", not bad, f"{j.height:,} rows {bad or ''}")
    # st_fail_4q from first principles for one state and quarter.
    home = F.scan_silver().group_by("CERT").agg(pl.col("STNAME").sort_by("REPDTE").last()).collect(engine="streaming")
    evs = P2.failure_events(fail).join(home, on="CERT")
    tx = evs.filter(pl.col("STNAME") == "TEXAS", pl.col("_fail_date").is_between(date(1989, 1, 1), date(1989, 12, 31))).height
    tx_certs = F.scan_silver().filter(pl.col("STNAME") == "TEXAS", pl.col("REPDTE") == date(1989, 12, 31)).select("CERT").collect()
    txn = tx_certs.height
    val = g.filter(pl.col("REPDTE") == date(1989, 12, 31), pl.col("CERT").is_in(tx_certs["CERT"].to_list()))["st_fail_4q"].unique().to_list()
    val = val[0] if len(val) == 1 else float("nan")
    check("st_fail_4q for Texas 1989Q4 equals failures in 1989 / banks", abs(val - tx / txn) < 1e-12, f"{tx} / {txn} = {val:.4f}")

    # ---- windows respect gaps and bank boundaries ----
    gap_rows = g.filter(pl.col("f_gap_before"))
    check("a reporting gap gives null, not a wrong total", gap_rows["roa_4q"].null_count() == gap_rows.height and gap_rows["eq_ta_d4"].null_count() == gap_rows.height, f"{gap_rows.height} gap rows")
    first = g.sort("CERT", "REPDTE").group_by("CERT", maintain_order=True).head(3)
    check("a bank's first 3 quarters have no 4-quarter total", first["roa_4q"].null_count() == first.height, f"{first.height:,} rows")

    # ---- known cases ----
    def bank(cert: int, d: date, cols: list[str]) -> dict:
        return g.filter(pl.col("CERT") == cert, pl.col("REPDTE") == d).select(*cols, "y").row(0, named=True)

    svb = bank(24735, date(2022, 12, 31), ["eq_ta", "npl"])
    check("known case: SVB 2022Q4 is y=1 and not in distress on these ratios", svb["y"] == 1 and svb["npl"] < 0.01 and svb["eq_ta"] > 0.05, str({k: round(v, 4) for k, v in svb.items()}))
    wm = bank(32633, date(2008, 6, 30), ["eq_ta", "npl"])
    check("known case: Washington Mutual 2008Q2 is y=1", wm["y"] == 1, str({k: round(v, 4) for k, v in wm.items()}))

    t = pl.DataFrame(RES)
    t.write_csv(C.REPORTS / "p4_checks.csv")
    fails = t.filter(pl.col("result") == "FAIL").height
    print(f"{t.height - fails}/{t.height} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
