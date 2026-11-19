"""
Part 3a - The core transformations, written three times: Polars (main), pandas (baseline), PySpark.

All three implement the same specification, from the bronze layer:

  T1 scan      keep only the columns needed
  T2 filter    semi-join Institutions (R1), EQ not null (R2)
  T3 join      first in-scope failure per bank; drop reports dated on or after it (R3)
  T4 clean     zero capital ratio -> null (R4), impossible negative balance -> null (R7)
  T5 window    year-to-date flows -> quarterly flows, by bank
  T6 row       ratios, null when the denominator is not positive
  T7 window    lags and rolling statistics by bank, null unless the quarters are consecutive
  T8 cross     percentile rank within the quarter, and gap to the state median for the quarter
  T9 label     y = 1 if the first failure falls in (q, q + 4 quarters]; null after 2025Q2
  T10 sort     (CERT, REPDTE), write ZSTD Parquet

Every derived column uses only reports dated at or before q (T7 looks backwards, T8 stays inside q).
y and _fail_date are the only columns that look forward; they are label-side columns.

The specification is written so the engines cannot silently disagree. These are the differences
that had to be pinned down, and the report explains them:
  * zero denominators: Polars and pandas give inf/NaN, Spark in ANSI mode raises an error.
    Every ratio is therefore guarded, and "undefined" is null everywhere.
  * rolling windows: a Spark window average ignores nulls, Polars and pandas return null. Spark
    counts non-null values and requires a full window.
  * percentile rank: Spark's percent_rank is (rank - 1) / (n - 1) with ties at the lowest rank. All
    engines here use the average rank divided by the number of non-null values.
  * null keys: pandas merge matches null to null, Polars and Spark joins do not. STNAME is never
    null after T2 and T3 (checked in p3_compare.py), so the difference cannot bite.

Usage (one engine, one run; p3_benchmark.py drives this from fresh processes):
  python src/p3_core.py --engine polars_mem --src data/bronze --out data/_p3/out/polars_mem
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Kept free of imports from config.py at module level so a pandas or Spark run does not load Polars
# before its timer starts.
SCOPE_START, END_Q = date(1984, 1, 1), date(2026, 6, 30)
LAST_MODEL_IDX = 2025 * 4 + 1          # 2025Q2
H = 4                                  # label horizon, quarters

KEYS = ["CERT", "REPDTE", "YEAR", "STNAME"]
FLOWS = ["NETINC", "INTINC", "EINTEXP", "NTLNLS"]
NONNEG = ["LIAB", "DEP", "CHBAL", "LNLSGR", "NCLNLS"]
INPUT = KEYS + ["ASSET", "EQ", "RBC1AAJ"] + NONNEG + FLOWS       # 16 of the 44 columns in bronze
QFLOWS = [f"{c}_Q" for c in FLOWS]
RATIOS = ["eq_ta", "npl", "cash_ta", "loan_dep", "roa_q", "nim_q", "nco_q"]
WINDOWED = ["eq_ta_d4", "asset_g4", "npl_d1", "roa_4q", "roa_std8"]
CROSS = ["eq_ta_pct", "npl_pct", "eq_ta_st_med", "eq_ta_gap"]
OUTPUT = KEYS + ["q_idx", "ASSET", "EQ", "RBC1AAJ"] + NONNEG + QFLOWS + RATIOS + WINDOWED + CROSS + ["y", "_fail_date"]
FLOAT_TOL = 1e-9                       # relative tolerance for the cross-engine comparison


# ------------------------------------------------------------------------------------------------
# Polars: one lazy query. Nothing runs until collect() or sink_parquet().
# ------------------------------------------------------------------------------------------------
def polars_core(src: Path, pre_filter=None, upto: str | None = None):
    """pre_filter and upto exist for p3_plans.py: what filtering before the windows would break, and
    how much time each step adds. A normal run uses neither."""
    import polars as pl

    fin = pl.scan_parquet(src / "financials" / "**" / "*.parquet", hive_partitioning=True,
                          hive_schema={"YEAR": pl.Int16})
    if pre_filter is not None:
        fin = fin.filter(pre_filter)
    inst = pl.scan_parquet(src / "institutions.parquet").select("CERT")
    events = (pl.scan_parquet(src / "failures.parquet")
              .filter(pl.col("RESTYPE") == "FAILURE", pl.col("CERT").is_not_null(),
                      pl.col("FAILDATE").is_between(SCOPE_START, END_Q))
              .group_by("CERT").agg(pl.col("FAILDATE").min().alias("_fail_date")))

    def guard(num: pl.Expr, den: str | pl.Expr) -> pl.Expr:
        d = pl.col(den) if isinstance(den, str) else den
        return pl.when(d > 0).then(num / d)

    def consecutive(n: int) -> pl.Expr:
        return (pl.col("q_idx") - pl.col("q_idx").shift(n).over("CERT")) == n

    def bank_shift(c: str, n: int) -> pl.Expr:
        return pl.col(c).shift(n).over("CERT")

    def pct_rank(c: str) -> pl.Expr:
        return pl.col(c).rank("average").over("REPDTE") / pl.col(c).count().over("REPDTE")

    def stop(step: str) -> bool:
        return upto == step

    p = (fin.select(INPUT)
         .join(inst, on="CERT", how="semi")                                    # T2 R1
         .filter(pl.col("EQ").is_not_null())                                   # T2 R2
         .join(events, on="CERT", how="left")                                  # T3
         .filter(pl.col("_fail_date").is_null() | (pl.col("REPDTE") < pl.col("_fail_date"))))
    if stop("T1-T3 scan, filter, join"):
        return p
    p = p.sort("CERT", "REPDTE")                                               # every window below relies on this
    if stop("sort by bank"):
        return p
    p = (p.with_columns(
             (pl.col("REPDTE").dt.year().cast(pl.Int32) * 4 + pl.col("REPDTE").dt.quarter().cast(pl.Int32) - 1).alias("q_idx"),
             pl.when(pl.col("RBC1AAJ") != 0).then(pl.col("RBC1AAJ")).alias("RBC1AAJ"),          # T4 R4
             *[pl.when(pl.col(c) >= 0).then(pl.col(c)).alias(c) for c in NONNEG])               # T4 R7
         .with_columns(*[                                                                       # T5
             pl.when(pl.col("REPDTE").dt.quarter() == 1).then(pl.col(c))
             .when(consecutive(1)).then(pl.col(c) - bank_shift(c, 1)).alias(f"{c}_Q") for c in FLOWS])
         .with_columns(                                                                         # T6
             guard(pl.col("EQ"), "ASSET").alias("eq_ta"),
             guard(pl.col("NCLNLS"), "LNLSGR").alias("npl"),
             guard(pl.col("CHBAL"), "ASSET").alias("cash_ta"),
             guard(pl.col("LNLSGR"), "DEP").alias("loan_dep"),
             guard(4 * pl.col("NETINC_Q"), "ASSET").alias("roa_q"),
             guard(4 * (pl.col("INTINC_Q") - pl.col("EINTEXP_Q")), "ASSET").alias("nim_q"),
             guard(4 * pl.col("NTLNLS_Q"), "LNLSGR").alias("nco_q")))
    if stop("T4-T6 clean, quarterly flows, ratios"):
        return p
    p = p.with_columns(                                                                         # T7
             pl.when(consecutive(4)).then(pl.col("eq_ta") - bank_shift("eq_ta", 4)).alias("eq_ta_d4"),
             pl.when(consecutive(4) & (bank_shift("ASSET", 4) > 0)).then(pl.col("ASSET") / bank_shift("ASSET", 4) - 1).alias("asset_g4"),
             pl.when(consecutive(1)).then(pl.col("npl") - bank_shift("npl", 1)).alias("npl_d1"),
             pl.when(consecutive(3)).then(pl.col("roa_q").rolling_mean(4, min_samples=4).over("CERT")).alias("roa_4q"),
             pl.when(consecutive(7)).then(pl.col("roa_q").rolling_std(8, min_samples=8).over("CERT")).alias("roa_std8"))
    if stop("T7 bank windows"):
        return p
    p = p.with_columns(pct_rank("eq_ta").alias("eq_ta_pct"), pct_rank("npl").alias("npl_pct"))  # T8
    if stop("T8 quarter ranks"):
        return p
    state_med = p.group_by("REPDTE", "STNAME").agg(pl.col("eq_ta").median().alias("eq_ta_st_med"))
    fail_q = pl.col("_fail_date").dt.year().cast(pl.Int32) * 4 + pl.col("_fail_date").dt.quarter().cast(pl.Int32) - 1
    p = (p.join(state_med, on=["REPDTE", "STNAME"], how="left")
         .with_columns((pl.col("eq_ta") - pl.col("eq_ta_st_med")).alias("eq_ta_gap"),
                       pl.when(pl.col("q_idx") <= LAST_MODEL_IDX)                               # T9
                       .then((fail_q - pl.col("q_idx")).is_between(1, H).fill_null(False).cast(pl.Int8)).alias("y"))
         .select(OUTPUT))
    if stop("T8 state median, T9 label"):
        return p
    return p.sort("CERT", "REPDTE")


def run_polars(src: Path, out: Path, mode: str) -> dict:
    import polars as pl

    out.parent.mkdir(parents=True, exist_ok=True)
    lf = polars_core(src)
    file = out / "part-0.parquet"
    out.mkdir(parents=True, exist_ok=True)
    if mode == "stream":
        lf.sink_parquet(file, compression="zstd", compression_level=3, engine="streaming")
    else:
        # In Polars 2 the default engine of collect() is streaming, so each mode names its engine.
        if mode == "noopt":
            df = lf.collect(engine="streaming", optimizations=pl.QueryOptFlags.none())
        else:
            df = lf.collect(engine="in-memory")
        df.write_parquet(file, compression="zstd", compression_level=3)
    return {}


# ------------------------------------------------------------------------------------------------
# pandas baseline: vectorised, idiomatic, no apply(). Ints with nulls become float64 and null is NaN.
# ------------------------------------------------------------------------------------------------
def run_pandas(src: Path, out: Path, all_columns: bool) -> dict:
    import numpy as np
    import pandas as pd

    fin = pd.read_parquet(src / "financials", columns=None if all_columns else INPUT, engine="pyarrow")
    if all_columns:
        fin = fin[INPUT]
    fin["YEAR"] = fin["YEAR"].astype("int32")
    fin["REPDTE"] = pd.to_datetime(fin["REPDTE"])
    inst = pd.read_parquet(src / "institutions.parquet", columns=["CERT"])["CERT"]
    fail = pd.read_parquet(src / "failures.parquet", columns=["CERT", "FAILDATE", "RESTYPE"])
    fail["FAILDATE"] = pd.to_datetime(fail["FAILDATE"])
    fail = fail[(fail["RESTYPE"] == "FAILURE") & fail["CERT"].notna()
                & fail["FAILDATE"].between(pd.Timestamp(SCOPE_START), pd.Timestamp(END_Q))]
    events = fail.groupby("CERT", as_index=False)["FAILDATE"].min().rename(columns={"FAILDATE": "_fail_date"})

    p = fin[fin["CERT"].isin(inst) & fin["EQ"].notna()]
    p = p.merge(events, on="CERT", how="left")
    p = p[p["_fail_date"].isna() | (p["REPDTE"] < p["_fail_date"])]
    p = p.sort_values(["CERT", "REPDTE"]).reset_index(drop=True)

    p["q_idx"] = p["REPDTE"].dt.year * 4 + p["REPDTE"].dt.quarter - 1
    p["RBC1AAJ"] = p["RBC1AAJ"].where(p["RBC1AAJ"] != 0)
    for c in NONNEG:
        p[c] = p[c].where(p[c] >= 0)

    g = p.groupby("CERT", sort=False)
    gq = g["q_idx"]
    consec = {n: (p["q_idx"] - gq.shift(n)) == n for n in (1, 3, 4, 7)}
    is_q1 = p["REPDTE"].dt.quarter == 1
    for c in FLOWS:
        p[f"{c}_Q"] = np.where(is_q1, p[c], (p[c] - g[c].shift(1)).where(consec[1]))

    def guard(num, den):
        return (num / den).where(den > 0)

    p["eq_ta"] = guard(p["EQ"], p["ASSET"])
    p["npl"] = guard(p["NCLNLS"], p["LNLSGR"])
    p["cash_ta"] = guard(p["CHBAL"], p["ASSET"])
    p["loan_dep"] = guard(p["LNLSGR"], p["DEP"])
    p["roa_q"] = guard(4 * p["NETINC_Q"], p["ASSET"])
    p["nim_q"] = guard(4 * (p["INTINC_Q"] - p["EINTEXP_Q"]), p["ASSET"])
    p["nco_q"] = guard(4 * p["NTLNLS_Q"], p["LNLSGR"])

    g = p.groupby("CERT", sort=False)
    p["eq_ta_d4"] = (p["eq_ta"] - g["eq_ta"].shift(4)).where(consec[4])
    a4 = g["ASSET"].shift(4)
    p["asset_g4"] = (p["ASSET"] / a4 - 1).where(consec[4] & (a4 > 0))
    p["npl_d1"] = (p["npl"] - g["npl"].shift(1)).where(consec[1])
    p["roa_4q"] = g["roa_q"].rolling(4, min_periods=4).mean().reset_index(level=0, drop=True).where(consec[3])
    p["roa_std8"] = g["roa_q"].rolling(8, min_periods=8).std().reset_index(level=0, drop=True).where(consec[7])

    for c, name in (("eq_ta", "eq_ta_pct"), ("npl", "npl_pct")):
        gr = p.groupby("REPDTE", sort=False)[c]
        p[name] = gr.rank(method="average") / gr.transform("count")
    p["eq_ta_st_med"] = p.groupby(["REPDTE", "STNAME"], sort=False)["eq_ta"].transform("median")
    p["eq_ta_gap"] = p["eq_ta"] - p["eq_ta_st_med"]

    fq = p["_fail_date"].dt.year * 4 + p["_fail_date"].dt.quarter - 1
    d = fq - p["q_idx"]
    p["y"] = ((d >= 1) & (d <= H)).astype("float64").where(p["q_idx"] <= LAST_MODEL_IDX)

    out.mkdir(parents=True, exist_ok=True)
    p[OUTPUT].to_parquet(out / "part-0.parquet", compression="zstd", index=False, engine="pyarrow")
    return {}


# ------------------------------------------------------------------------------------------------
# PySpark: the DataFrame API only, no Python UDF, so all the work stays in the JVM.
# ------------------------------------------------------------------------------------------------
def spark_core(spark, src: Path):
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    fin = spark.read.parquet(str(src / "financials")).select(*INPUT)
    inst = spark.read.parquet(str(src / "institutions.parquet")).select("CERT")
    events = (spark.read.parquet(str(src / "failures.parquet"))
              .filter((F.col("RESTYPE") == "FAILURE") & F.col("CERT").isNotNull()
                      & F.col("FAILDATE").between(F.lit(SCOPE_START), F.lit(END_Q)))
              .groupBy("CERT").agg(F.min("FAILDATE").alias("_fail_date")))

    w = Window.partitionBy("CERT").orderBy("REPDTE")

    def guard(num, den):
        return F.when(den > 0, num / den)

    def consec(n):
        return (F.col("q_idx") - F.lag("q_idx", n).over(w)) == n

    def rolling(c, n, fn):
        frame = w.rowsBetween(-(n - 1), 0)
        return F.when(consec(n - 1) & (F.count(c).over(frame) == n), fn(c).over(frame))

    def pct_rank(c):
        # Average rank: rank() is the lowest rank of a tie, a cumulative count is the highest.
        # asc_nulls_last keeps nulls from taking up rank numbers.
        wv = Window.partitionBy("REPDTE").orderBy(F.col(c).asc_nulls_last())
        lo, hi = F.rank().over(wv), F.count(F.lit(1)).over(wv.rangeBetween(Window.unboundedPreceding, Window.currentRow))
        return F.when(F.col(c).isNotNull(), (lo + hi) / 2 / F.count(c).over(Window.partitionBy("REPDTE")))

    p = (fin.join(inst.distinct(), on="CERT", how="left_semi")
         .filter(F.col("EQ").isNotNull())
         .join(F.broadcast(events), on="CERT", how="left")
         .filter(F.col("_fail_date").isNull() | (F.col("REPDTE") < F.col("_fail_date")))
         .withColumns({"q_idx": F.year("REPDTE") * 4 + F.quarter("REPDTE") - 1,
                       "RBC1AAJ": F.when(F.col("RBC1AAJ") != 0, F.col("RBC1AAJ")),
                       **{c: F.when(F.col(c) >= 0, F.col(c)) for c in NONNEG}})
         .withColumns({f"{c}_Q": F.when(F.quarter("REPDTE") == 1, F.col(c))
                       .when(consec(1), F.col(c) - F.lag(c, 1).over(w)) for c in FLOWS})
         .withColumns({"eq_ta": guard(F.col("EQ"), F.col("ASSET")),
                       "npl": guard(F.col("NCLNLS"), F.col("LNLSGR")),
                       "cash_ta": guard(F.col("CHBAL"), F.col("ASSET")),
                       "loan_dep": guard(F.col("LNLSGR"), F.col("DEP")),
                       "roa_q": guard(4 * F.col("NETINC_Q"), F.col("ASSET")),
                       "nim_q": guard(4 * (F.col("INTINC_Q") - F.col("EINTEXP_Q")), F.col("ASSET")),
                       "nco_q": guard(4 * F.col("NTLNLS_Q"), F.col("LNLSGR"))})
         .withColumns({"eq_ta_d4": F.when(consec(4), F.col("eq_ta") - F.lag("eq_ta", 4).over(w)),
                       "asset_g4": F.when(consec(4) & (F.lag("ASSET", 4).over(w) > 0), F.col("ASSET") / F.lag("ASSET", 4).over(w) - 1),
                       "npl_d1": F.when(consec(1), F.col("npl") - F.lag("npl", 1).over(w)),
                       "roa_4q": rolling("roa_q", 4, F.avg),
                       "roa_std8": rolling("roa_q", 8, F.stddev_samp)})
         .withColumn("eq_ta_pct", pct_rank("eq_ta"))
         .withColumn("npl_pct", pct_rank("npl")))
    state_med = p.groupBy("REPDTE", "STNAME").agg(F.median("eq_ta").alias("eq_ta_st_med"))
    fq = F.year("_fail_date") * 4 + F.quarter("_fail_date") - 1
    d = fq - F.col("q_idx")
    return (p.join(state_med, on=["REPDTE", "STNAME"], how="left")
            .withColumns({"eq_ta_gap": F.col("eq_ta") - F.col("eq_ta_st_med"),
                          "y": F.when(F.col("q_idx") <= LAST_MODEL_IDX,
                                      F.coalesce(d.between(1, H), F.lit(False)).cast("tinyint"))})
            .select(*OUTPUT).orderBy("CERT", "REPDTE"))


def spark_stage_totals(spark) -> dict:
    """Shuffle and input bytes of all finished stages, from Spark's own monitoring REST API."""
    import urllib.request

    sc = spark.sparkContext
    url = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/stages?status=complete"
    stages = json.load(urllib.request.urlopen(url, timeout=30))
    return {"stages": len(stages),
            "shuffle_write_bytes": sum(s["shuffleWriteBytes"] for s in stages),
            "spill_disk_bytes": sum(s["diskBytesSpilled"] for s in stages),
            "input_bytes": sum(s["inputBytes"] for s in stages)}


def run_spark(src: Path, out: Path, threads: int, driver_memory: str, shuffle_partitions: int | None) -> dict:
    from p3_spark_session import get_spark

    t0 = time.perf_counter()
    spark = get_spark(threads=threads, driver_memory=driver_memory, shuffle_partitions=shuffle_partitions,
                      extra={"spark.local.dir": str(out.parent / "_spark_tmp")})
    started = time.perf_counter() - t0
    spark_core(spark, src).write.mode("overwrite").option("compression", "zstd").parquet(str(out))
    extra = {"session_start_s": round(started, 2), **spark_stage_totals(spark)}
    spark.stop()
    return extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["pandas_cols", "pandas_all", "polars_mem", "polars_stream", "polars_noopt", "spark"])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=0, help="Spark only; Polars reads POLARS_MAX_THREADS")
    ap.add_argument("--driver-memory", default="8g")
    ap.add_argument("--shuffle-partitions", type=int, default=None)
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)

    t0 = time.perf_counter()
    if a.engine.startswith("polars"):
        extra = run_polars(a.src, a.out, a.engine.split("_")[1])
    elif a.engine.startswith("pandas"):
        extra = run_pandas(a.src, a.out, a.engine == "pandas_all")
    else:
        extra = run_spark(a.src, a.out, a.threads, a.driver_memory, a.shuffle_partitions)
    wall = time.perf_counter() - t0
    print("RESULT " + json.dumps({"wall_s": round(wall, 3), **extra}))


if __name__ == "__main__":
    main()
