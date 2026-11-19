"""
Part 3b - Do the three engines give the same answer?

Polars is the reference. Every row and every column of the pandas and Spark outputs is compared with
it, and all three are also compared with the silver panel of Part 2, which was built by different
code (eager, more rules). A count of matching totals would not be enough: errors can cancel out.

Rules of the comparison
  * rows must match exactly on (CERT, REPDTE), with no extra or missing rows;
  * integer-valued columns and y must match exactly;
  * floats match within a relative tolerance of 1e-9, because rolling windows sum in a different
    order in each engine. The largest difference found is reported, so the tolerance is not hiding
    anything;
  * null must match null. pandas stores null as NaN, so NaN in any engine counts as null.

Exit code is non-zero if anything fails.
Usage:  python src/p3_compare.py [--dir data/_p3/out]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p3_core as K  # noqa: E402

ENGINES = {"pandas_cols": "pandas", "spark": "PySpark", "polars_stream": "Polars streaming"}
EXACT = {"q_idx", "y", "ASSET", "EQ", "LIAB", "DEP", "CHBAL", "LNLSGR", "NCLNLS", *K.QFLOWS, "YEAR"}


def load(folder: Path) -> pl.DataFrame:
    df = pl.read_parquet(folder / "*.parquet")
    return df.with_columns(pl.col("REPDTE").cast(pl.Date), pl.col("_fail_date").cast(pl.Date)).sort("CERT", "REPDTE")


def compare(ref: pl.DataFrame, other: pl.DataFrame, name: str) -> list[dict]:
    rows = [{"engine": name, "column": "(rows)", "kind": "key", "rows_ref": ref.height, "rows_other": other.height,
             "null_mismatch": ref.join(other, on=["CERT", "REPDTE"], how="anti").height
             + other.join(ref, on=["CERT", "REPDTE"], how="anti").height,
             "value_mismatch": 0, "max_abs_diff": 0.0, "max_rel_diff": 0.0}]
    j = ref.select(K.OUTPUT).join(other.select(K.OUTPUT), on=["CERT", "REPDTE"], how="inner", suffix="__o")
    for c in K.OUTPUT:
        if c in ("CERT", "REPDTE"):
            continue
        a, b = pl.col(c), pl.col(f"{c}__o")
        if j.schema[c] in (pl.String, pl.Date):
            nm = j.select((a.is_null() != b.is_null()).sum()).item()
            vm = j.select(((a != b) & a.is_not_null() & b.is_not_null()).sum()).item()
            mad = mrd = 0.0
            kind = "exact"
        else:
            fa, fb = a.cast(pl.Float64).fill_nan(None), b.cast(pl.Float64).fill_nan(None)
            d = (fa - fb).abs()
            r = d / pl.max_horizontal(fa.abs(), fb.abs(), pl.lit(1e-300))
            tol = 0.0 if c in EXACT else K.FLOAT_TOL
            nm = j.select((fa.is_null() != fb.is_null()).sum()).item()
            # Rolling differences near zero are compared on an absolute scale: a ratio of two values
            # around 1e-17 is meaningless.
            vm = j.select(((r > tol) & (d > tol)).sum()).item()
            mad = j.select(d.max()).item() or 0.0
            mrd = j.select(r.max()).item() or 0.0
            kind = "exact" if c in EXACT else "float"
        rows.append({"engine": name, "column": c, "kind": kind, "rows_ref": j.height, "rows_other": j.height,
                     "null_mismatch": nm, "value_mismatch": vm, "max_abs_diff": mad, "max_rel_diff": mrd})
    return rows


def arbitrate_rolling_std(ref: pl.DataFrame, pdf: pl.DataFrame) -> list[dict]:
    """Where pandas and Polars disagree on roa_std8, recompute each window directly from its 8 values."""
    j = ref.select("CERT", "REPDTE", "roa_std8").join(pdf.select("CERT", "REPDTE", pl.col("roa_std8").alias("pd")), on=["CERT", "REPDTE"])
    d = (pl.col("roa_std8") - pl.col("pd")).abs()
    hits = j.filter((d > K.FLOAT_TOL) & (d / pl.max_horizontal(pl.col("roa_std8").abs(), pl.col("pd").abs()) > K.FLOAT_TOL))
    out = []
    for r in hits.iter_rows(named=True):
        win = ref.filter(pl.col("CERT") == r["CERT"], pl.col("REPDTE") <= r["REPDTE"]).tail(8)["roa_q"]
        direct = float(np.std(win.to_numpy(), ddof=1))
        out.append({"CERT": r["CERT"], "REPDTE": str(r["REPDTE"]), "polars": r["roa_std8"], "pandas": r["pd"], "direct": direct,
                    "polars_err": abs(r["roa_std8"] - direct), "pandas_err": abs(r["pd"] - direct),
                    "bank_max_abs_roa_q": float(ref.filter(pl.col("CERT") == r["CERT"], pl.col("REPDTE") <= r["REPDTE"])["roa_q"].abs().max())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=C.DATA / "_p3" / "out")
    a = ap.parse_args()
    ref = load(a.dir / "polars_mem")
    checks: list[dict] = []
    for folder, name in ENGINES.items():
        checks += compare(ref, load(a.dir / folder), name)
    tab = pl.DataFrame(checks)
    tab.write_csv(C.REPORTS / "p3_engine_reconcile.csv")
    arbitration = arbitrate_rolling_std(ref, load(a.dir / "pandas_cols"))

    # The silver panel: built by other code and with other rules for the cells this spec does not
    # clean, so only columns with the same definition are compared.
    silver = (pl.scan_parquet(C.DATA_SILVER / "panel" / "**" / "*.parquet", hive_partitioning=True, hive_schema={"YEAR": pl.Int16})
              .select("CERT", "REPDTE", "q_idx", "ASSET", "EQ", "RBC1AAJ", *K.NONNEG, *K.QFLOWS, "y", "_fail_date").collect())
    s_rows = {"silver_rows": silver.height, "core_rows": ref.height,
              "rows_only_in_one": ref.join(silver, on=["CERT", "REPDTE"], how="anti").height
              + silver.join(ref, on=["CERT", "REPDTE"], how="anti").height}
    j = ref.join(silver, on=["CERT", "REPDTE"], suffix="__s")
    s_diff = {c: j.select(pl.col(c).ne_missing(pl.col(f"{c}__s").cast(j.schema[c])).sum()).item()
              for c in ["q_idx", "ASSET", "EQ", "RBC1AAJ", *K.NONNEG, *K.QFLOWS, "y", "_fail_date"]}

    summary = {
        "rows": ref.height, "banks": ref["CERT"].n_unique(), "cells_compared_per_engine": ref.height * (len(K.OUTPUT) - 2),
        "y1_rows": int(ref["y"].sum()), "stname_nulls": ref["STNAME"].null_count(),
        "by_engine": (tab.group_by("engine").agg(pl.col("null_mismatch").sum(), pl.col("value_mismatch").sum(),
                                                 pl.col("max_abs_diff").max(), pl.col("max_rel_diff").filter(pl.col("kind") == "exact").max().alias("max_rel_diff_exact_cols"))
                      .sort("engine").to_dicts()),
        "against_silver": {**s_rows, "cell_differences": s_diff},
        "pandas_roa_std8_arbitration": arbitration,
    }
    (C.REPORTS / "p3_engine_reconcile.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    with pl.Config(tbl_rows=12, tbl_cols=9, tbl_width_chars=150):
        print(tab.filter((pl.col("null_mismatch") > 0) | (pl.col("value_mismatch") > 0)))
        print(tab.filter(pl.col("kind") == "float").sort("max_abs_diff", descending=True).head(8))
    # A pandas roa_std8 difference is forgiven only if direct recomputation proves Polars right.
    explained = len(arbitration) if all(x["polars_err"] <= K.FLOAT_TOL for x in arbitration) else 0
    bad = (tab.filter((pl.col("null_mismatch") > 0) | (pl.col("value_mismatch") > 0))["value_mismatch"].sum()
           + tab["null_mismatch"].sum() + s_rows["rows_only_in_one"] + sum(s_diff.values()) - explained)
    print(f"explained by direct recomputation: {explained} cells")
    print("PASS" if not bad else f"FAIL: {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
