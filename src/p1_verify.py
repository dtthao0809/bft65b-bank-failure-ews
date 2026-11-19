from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import fdic_download_starter as starter  # noqa: E402  (reuses its retrying HTTP helper)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_keys(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path, columns=["CERT", "REPDTE"]).with_columns(
        pl.col("CERT").cast(pl.Int64), pl.col("REPDTE").cast(pl.String))


# ----------------------------------------------------------------------------
# A + B: financials coverage and keys
# ----------------------------------------------------------------------------
def check_financials(raw: Path) -> tuple[dict, pl.DataFrame]:
    quarters = starter.quarter_ends(date.fromisoformat(C.START_Q), date.fromisoformat(C.END_Q))
    manifest = json.loads((raw / "manifest_financials.json").read_text())
    rows, wrong_name = [], []
    keys = []
    for q in quarters:
        f = raw / "financials" / f"REPDTE={q}.parquet"
        if not f.exists():
            continue
        k = raw_keys(f)
        keys.append(k)
        if k["REPDTE"].n_unique() != 1 or k["REPDTE"][0] != q:
            wrong_name.append(q)
        rows.append({"quarter": q, "rows_on_disk": k.height,
                     "api_total": manifest.get(q, {}).get("api_total")})
    per_q = pl.DataFrame(rows)
    panel = pl.concat(keys)
    res = {
        "quarters_expected": len(quarters),
        "quarters_on_disk": per_q.height,
        "quarters_missing": sorted(set(quarters) - set(per_q["quarter"])),
        "quarters_rows_differ_from_api": per_q.filter(
            pl.col("rows_on_disk") != pl.col("api_total"))["quarter"].to_list(),
        "quarters_wrong_repdte_inside": wrong_name,
        "panel_rows": panel.height,
        "distinct_banks": panel["CERT"].n_unique(),
        "null_keys": panel["CERT"].null_count() + panel["REPDTE"].null_count(),
        "duplicate_cert_repdte": panel.height - panel.unique().height,
    }
    return res, panel


# ----------------------------------------------------------------------------
# C: small tables
# ----------------------------------------------------------------------------
def check_small_tables(raw: Path, offline: bool) -> dict:
    res = {}
    for name in ("failures", "institutions"):
        df = pl.read_parquet(raw / f"{name}.parquet")
        info = {"rows": df.height, "exact_duplicate_rows": df.height - df.unique().height,
                "null_cert": df["CERT"].null_count()}
        if not offline:
            meta = starter.get_json(name, {"fields": "CERT", "limit": 1, "format": "json"})["meta"]
            info |= {"api_total": meta["total"], "api_index": meta["index"]["name"],
                     "api_index_created": meta["index"]["createTimestamp"]}
        res[name] = info
    return res


# ----------------------------------------------------------------------------
# D: external control total
# ----------------------------------------------------------------------------
def fin_year_ends(raw: Path) -> pl.DataFrame:
    """Year-end panel totals, split by whether the bank has a record in Institutions."""
    inst = pl.read_parquet(raw / "institutions.parquet", columns=["CERT"]).with_columns(
        pl.col("CERT").cast(pl.Int64))
    rows = []
    for f in sorted((raw / "financials").glob("REPDTE=*1231.parquet")):
        d = pl.read_parquet(f, columns=["CERT", "ASSET"]).with_columns(
            pl.col("CERT").cast(pl.Int64), pl.col("ASSET").cast(pl.Int64))
        un = d.join(inst, on="CERT", how="anti")
        rows.append({"YEAR": int(f.stem[7:11]), "panel_banks": d.height, "panel_asset": d["ASSET"].sum(),
                     "not_in_institutions_banks": un.height, "not_in_institutions_asset": un["ASSET"].sum()})
    return pl.DataFrame(rows)


def control_totals(raw: Path) -> pl.DataFrame:
    """Year-end bank count and total assets: panel vs /summary (national row, CB + SI).

    /summary reports TOTAL = 0 for savings institutions (SI) before 2019 and holds their count
    in COUNT, so the larger of the two fields is used. Panel banks with no Institutions record
    (for example non-insured trust companies) are removed before comparing; the numbers kept in
    the output show how much of the gap they explain.
    """
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        js = starter.get_json("summary", {
            "fields": "YEAR,CB_SI,TOTAL,COUNT,ASSET", "limit": 1000, "offset": offset, "format": "json",
            "filters": 'STNAME:"All States and Territories" AND YEAR:[1984 TO 2026]'})
        total = js["meta"]["total"]
        page = [r["data"] for r in js["data"]]
        if not page:
            break
        rows += page
        offset += len(page)
    summ = (pl.DataFrame(rows, infer_schema_length=None)
            .with_columns(pl.col("YEAR").cast(pl.Int64),
                          pl.max_horizontal("TOTAL", "COUNT").alias("n"))
            .group_by("YEAR").agg(pl.col("n").sum().alias("fdic_banks"),
                                  pl.col("ASSET").sum().alias("fdic_asset")))
    matched_n = pl.col("panel_banks") - pl.col("not_in_institutions_banks")
    matched_a = pl.col("panel_asset") - pl.col("not_in_institutions_asset")
    return (fin_year_ends(raw).join(summ, on="YEAR", how="left").sort("YEAR")
            .with_columns((matched_n - pl.col("fdic_banks")).alias("banks_diff"),
                          ((matched_a / pl.col("fdic_asset") - 1) * 100).round(4).alias("asset_diff_pct")))


# ----------------------------------------------------------------------------
# E: cross-table coverage
# ----------------------------------------------------------------------------
def cross_table(raw: Path, panel: pl.DataFrame) -> pl.DataFrame:
    fail = pl.read_parquet(raw / "failures.parquet").with_columns(
        pl.col("CERT").cast(pl.Int64), pl.col("FAILDATE").str.to_date("%m/%d/%Y"))
    banks = panel.select("CERT").unique().with_columns(pl.lit(True).alias("in_panel"))
    start, end = date.fromisoformat("1984-01-01"), date.fromisoformat(C.END_Q)
    return (fail.join(banks, on="CERT", how="left")
            .with_columns(pl.when(pl.col("FAILDATE") < start).then(pl.lit("before 1984"))
                          .when(pl.col("FAILDATE") > end).then(pl.lit("after 2026Q2"))
                          .otherwise(pl.lit("in scope")).alias("period"))
            .group_by("period", "RESTYPE")
            .agg(pl.len().alias("rows"), pl.col("CERT").null_count().alias("null_cert"),
                 pl.col("CERT").n_unique().alias("distinct_cert"),
                 pl.col("in_panel").sum().alias("rows_with_panel_history"),
                 pl.col("FAILDATE").min().alias("first"), pl.col("FAILDATE").max().alias("last"))
            .sort("period", "RESTYPE"))


# ----------------------------------------------------------------------------
# F: second, independent download
# ----------------------------------------------------------------------------
def compare_downloads(raw: Path, other: Path) -> dict:
    """Cell-by-cell comparison on the columns both downloads share."""
    diff_cells: Counter = Counter()
    quarters, cells, cols = 0, 0, set()
    for f in sorted((raw / "financials").glob("REPDTE=*.parquet")):
        g = other / "financials" / f.name
        if not g.exists():
            continue
        a, b = pl.read_parquet(f), pl.read_parquet(g)
        shared = [c for c in a.columns if c in b.columns and c != "REPDTE"]
        cols |= set(shared)
        key = "CERT"

        def norm(df: pl.DataFrame) -> pl.DataFrame:
            return df.select(shared).with_columns(
                [pl.col(c).cast(pl.Float64) for c in shared if c != "STNAME"]).sort(key)

        a, b = norm(a), norm(b)
        if a.height != b.height:
            raise ValueError(f"{f.name}: {a.height} vs {b.height} rows")
        for c in shared:
            diff_cells[c] += a.height - int(a[c].eq_missing(b[c]).sum())
        quarters += 1
        cells += a.height * len(shared)
    return {"other_download": str(other), "quarters_compared": quarters,
            "shared_columns": len(cols), "cells_compared": cells,
            "cells_different": sum(diff_cells.values()),
            "columns_with_differences": {k: v for k, v in diff_cells.items() if v}}


# ----------------------------------------------------------------------------
# G: evidence files
# ----------------------------------------------------------------------------
def schema_drift(raw: Path) -> pl.DataFrame:
    """Physical type of each column across the quarterly files, read from Parquet footers."""
    types: dict[str, Counter] = defaultdict(Counter)
    files = sorted((raw / "financials").glob("REPDTE=*.parquet"))
    for f in files:
        for field in pq.read_schema(f):
            if not field.name.startswith("__"):
                types[field.name][str(field.type)] += 1
    return pl.DataFrame([
        {"column": k, "n_physical_types": len(v), "quarters_absent": len(files) - sum(v.values()),
         "physical_types": ", ".join(f"{t}: {n}" for t, n in v.most_common())}
        for k, v in types.items()
    ]).sort(["n_physical_types", "quarters_absent", "column"], descending=[True, True, False])


def file_hashes(raw: Path) -> pl.DataFrame:
    files = sorted((raw / "financials").glob("*.parquet")) + sorted(raw.glob("*.parquet"))
    return pl.DataFrame([{"file": str(f.relative_to(raw)).replace("\\", "/"),
                          "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files])


def financial_vintage() -> dict:
    meta = starter.get_json("financials", {"fields": "CERT", "limit": 1, "format": "json"})["meta"]
    return {"api_index": meta["index"]["name"], "api_index_created": meta["index"]["createTimestamp"]}


def plot_coverage(panel: pl.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_q = (panel.group_by("REPDTE").len().sort("REPDTE")
             .with_columns(pl.col("REPDTE").str.to_date("%Y%m%d")))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(per_q["REPDTE"], per_q["len"], color="#2463eb", linewidth=2)
    ax.set_ylim(0, per_q["len"].max() * 1.08)
    ax.set_ylabel("Reporting institutions", color="#57606a")
    ax.grid(axis="y", color="#d8dee4", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#57606a", length=0)
    ax.set_title("Institutions reporting per quarter, 1984Q1 to 2026Q2", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description='Part 1a - Verify that the raw download is complete and record its provenance.')
    ap.add_argument("--compare", type=Path, help="folder of a second, independent download")
    ap.add_argument("--offline", action="store_true", help="skip the checks that call the API")
    a = ap.parse_args()
    raw, rep = C.DATA_RAW, C.REPORTS
    rep.mkdir(parents=True, exist_ok=True)

    fields_path = raw / "fields.json"
    fields_ok = None
    if fields_path.exists():
        fields_ok = set(json.loads(fields_path.read_text())["financials"]) == set(C.FIN_FIELDS)

    fin, panel = check_financials(raw)
    small = check_small_tables(raw, a.offline)
    res = {"raw_folder": str(raw), "financials": fin, **small,
           "field_list_matches_config": fields_ok}

    log = raw / "download_log.json"
    if log.exists():
        res["download_log"] = json.loads(log.read_text())

    cross = cross_table(raw, panel)
    cross.write_csv(rep / "p1_failures_coverage.csv")
    panel_banks = panel.select("CERT").unique()
    inst = pl.read_parquet(raw / "institutions.parquet", columns=["CERT"]).with_columns(
        pl.col("CERT").cast(pl.Int64))
    res["panel_banks_missing_from_institutions"] = panel_banks.join(inst, on="CERT", how="anti").height

    if not a.offline:
        res["financials"] |= financial_vintage()
        ctl = control_totals(raw)
        ctl.write_csv(rep / "p1_control_totals.csv")
        compared = ctl.drop_nulls("fdic_banks")
        res["control_totals"] = {
            "year_ends_compared": compared.height,
            "year_ends_bank_count_exact": compared.filter(pl.col("banks_diff") == 0).height,
            "max_abs_bank_diff": int(compared["banks_diff"].abs().max()),
            "max_abs_asset_diff_pct": float(compared["asset_diff_pct"].abs().max()),
        }
    if a.compare:
        res["independent_download"] = compare_downloads(raw, a.compare)

    schema_drift(raw).write_csv(rep / "p1_schema_drift.csv")
    file_hashes(raw).write_csv(rep / "p1_raw_sha256.csv")
    plot_coverage(panel, rep / "fig_p1_coverage.png")

    hard = [not fin["quarters_missing"], not fin["quarters_rows_differ_from_api"],
            not fin["quarters_wrong_repdte_inside"], fin["null_keys"] == 0,
            fin["duplicate_cert_repdte"] == 0, fin["quarters_on_disk"] == fin["quarters_expected"],
            fields_ok is not False]
    for name in ("failures", "institutions"):
        hard.append(small[name]["exact_duplicate_rows"] == 0)
        if not a.offline:
            hard.append(small[name]["rows"] == small[name]["api_total"])
    res["ok"] = all(hard)
    (rep / "p1_completeness.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=160):
        print(cross)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
