"""
Part 4b - Which candidate features go forward? Decided from the training window of development only.

The test period is never read: load_train() filters by date before anything else is computed. The
validation window of development (2008-2011) is also kept out of every y-driven rule here, so Part 5 can
validate on it without having helped choose the features.

Rules, set before looking at the results:
  S1 null share at most 20% in every training era (a field that is often null would mostly be imputed)
  S2 signal: mean within-era AUC at least 0.02 away from 0.5
  S3 the signal has the same direction in every era with at least 30 failed banks
  S4 of a pair with |Spearman| above 0.90, keep the one with the stronger within-era signal; a feature
     over 5% null gives up 0.01 of strength (this tie-break was added after the first run, see below)
  L1 a single-feature AUC above 0.98 stops the run: that is more likely a leak than a discovery

AUC is computed inside eras as well as pooled. A pooled AUC can be high simply because a feature drifted
over time and failures cluster in a few years. The gap between the two measures that effect, and it is
also how missing-value indicators are judged.

Usage:  python src/p4_screen.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p4_features as F  # noqa: E402

ERAS = [("1986-89", 1986, 1989), ("1990-94", 1990, 1994), ("1995-2006", 1995, 2006)]
MIN_FAILED_FOR_SIGN = 30


def load_train() -> pl.DataFrame:
    start, end = date.fromisoformat(C.PIPE_TRAIN_START), date.fromisoformat(C.PIPE_TRAIN_END)
    return (pl.scan_parquet(C.DATA_GOLD / "features" / "**" / "*.parquet", hive_partitioning=True, hive_schema={C.PARTITION_KEY: pl.Int16})
            .filter(pl.col("REPDTE").is_between(start, end))
            .select("CERT", "REPDTE", C.PARTITION_KEY, *F.FEATURES, "y").collect(engine="streaming"))


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC with ties averaged. Rows where x is null are ignored."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    pos = int(y.sum())
    neg = y.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = pl.Series(x).rank("average").to_numpy()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> None:
    d = load_train()
    y = d["y"].to_numpy().astype(int)
    year = d[C.PARTITION_KEY].to_numpy()
    masks = {name: (year >= a) & (year <= b) for name, a, b in ERAS}
    failed = {name: d.filter(pl.col(C.PARTITION_KEY).is_between(a, b), pl.col("y") == 1)["CERT"].n_unique() for name, a, b in ERAS}
    meta = {r[0]: r for r in F.REGISTRY}

    rows = []
    for f in F.FEATURES:
        x = d[f].cast(pl.Float64).to_numpy()
        era_auc = {n: auc(x[m], y[m]) for n, m in masks.items()}
        era_null = {n: float(np.isnan(x[m]).mean() * 100) for n, m in masks.items()}
        wauc = float(np.nanmean(list(era_auc.values())))
        signs = {n: np.sign(a - 0.5) for n, a in era_auc.items() if failed[n] >= MIN_FAILED_FOR_SIGN}
        isnull = np.isnan(x).astype(float)
        null_wauc = float(np.nanmean([auc(isnull[m], y[m]) for m in masks.values()])) if isnull.any() else float("nan")
        rows.append({"feature": f, "pillar": meta[f][1], "kind": meta[f][2], "expected_sign": meta[f][3],
                     "auc_pooled": auc(x, y), "auc_within_era": wauc, **{f"auc_{n}": v for n, v in era_auc.items()},
                     "null_max_era_pct": max(era_null.values()), **{f"null_{n}": v for n, v in era_null.items()},
                     "null_auc_pooled": auc(isnull, y) if isnull.any() else None, "null_auc_within_era": null_wauc,
                     "sign_stable": len(set(signs.values())) == 1})
    t = pl.DataFrame(rows, infer_schema_length=None)

    leak = t.filter(pl.max_horizontal(pl.col("auc_pooled"), 1 - pl.col("auc_pooled")) > 0.98)
    if leak.height:
        print(leak)
        raise SystemExit("L1: a single feature separates failures almost perfectly. Investigate before going on.")

    t = t.with_columns(
        (pl.col("null_max_era_pct") <= 20).alias("s1_complete"),
        ((pl.col("auc_within_era") - 0.5).abs() >= 0.02).alias("s2_signal"),
        pl.col("sign_stable").alias("s3_stable"),
        (pl.col("auc_within_era") - 0.5).abs().alias("_strength"))

    # S4 only among features that pass S1-S3. Spearman on a fixed sample: ranks of up to 200k rows.
    # Tie-break, added after the first run: that run dropped eq_ta (never null) for tier1_lev (18% null in
    # 1986-89) over an AUC gap of 0.006, and an imputed tier1_lev gives a failing thrift median capital.
    # A feature over 5% null therefore gives up 0.01 of strength in this ordering.
    cand = (t.filter("s1_complete", "s2_signal", "s3_stable")
            .with_columns((pl.col("_strength") - pl.when(pl.col("null_max_era_pct") > 5).then(0.01).otherwise(0.0)).alias("_order"))
            .sort("_order", descending=True)["feature"].to_list())
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(d.height, size=min(200_000, d.height), replace=False)
    sample = d[idx].select(cand).cast(pl.Float64)
    ranks = sample.with_columns(pl.all().rank("average"))
    corr = {}
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            ok = ranks.select(a, b).drop_nulls()
            corr[(a, b)] = float(np.corrcoef(ok[a].to_numpy(), ok[b].to_numpy())[0, 1])
    kept, dropped = [], {}
    for f in cand:                               # strongest first, so the stronger of a pair stays
        clash = [k for k in kept if abs(corr.get((k, f), corr.get((f, k), 0.0))) > 0.90]
        if clash:
            dropped[f] = clash[0]
        else:
            kept.append(f)

    def verdict(r: dict) -> str:
        if not r["s1_complete"]:
            return "out: S1 often null"
        if not r["s2_signal"]:
            return "out: S2 no signal"
        if not r["s3_stable"]:
            return "out: S3 direction changes"
        if r["feature"] in dropped:
            return f"out: S4 duplicates {dropped[r['feature']]}"
        return "kept"

    t = t.with_columns(pl.Series("decision", [verdict(r) for r in t.iter_rows(named=True)])).drop("_strength").sort("auc_within_era")
    t.write_csv(C.REPORTS / "p4_feature_screen.csv", float_precision=4)

    numeric = [f for f in F.NUMERIC if f in kept]
    flags = [f for f in F.FLAGS if f in kept]
    # A missing-value indicator is justified only if missingness carries risk within eras. Where the
    # pooled AUC is much higher than within eras, missingness identifies a period or charter instead.
    ind = (t.filter(pl.col("decision") == "kept", pl.col("kind") == "num", pl.col("null_max_era_pct") > 0.1)
           .select("feature", "null_max_era_pct", "null_auc_pooled", "null_auc_within_era"))
    use_ind = ind.filter((pl.col("null_auc_within_era") - 0.5).abs() >= 0.02,
                         (pl.col("null_auc_pooled") - pl.col("null_auc_within_era")).abs() < 0.02)["feature"].to_list()
    ind.write_csv(C.REPORTS / "p4_missing_indicators.csv", float_precision=4)
    (C.DATA_GOLD / "_schema_features.json").write_text(json.dumps({
        "numeric": numeric, "flags": flags, "missing_indicators": use_ind, "label": "y",
        "keys": ["CERT", "REPDTE"], "split_column": "_split", "failed_banks_per_era": failed,
        "screened_on": [C.PIPE_TRAIN_START, C.PIPE_TRAIN_END]}, indent=1))
    (C.REPORTS / "p4_selected_features.json").write_text((C.DATA_GOLD / "_schema_features.json").read_text())
    with pl.Config(tbl_rows=50, tbl_cols=12, tbl_width_chars=175, fmt_str_lengths=34, float_precision=3):
        print(failed)
        print(t.select("feature", "auc_pooled", "auc_within_era", "auc_1986-89", "auc_1990-94", "auc_1995-2006", "null_max_era_pct", "decision"))
        print(ind)
    print(f"kept {len(numeric)} numeric + {len(flags)} flags; indicators {use_ind}")


if __name__ == "__main__":
    main()
