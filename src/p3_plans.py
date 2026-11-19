from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p3_core as K  # noqa: E402

SRC = C.DATA_BRONZE


def selective_query() -> pl.LazyFrame:
    """A typical analyst query: one year, one number by state. Safe to push down."""
    return (pl.scan_parquet(SRC / "financials" / "**" / "*.parquet", hive_partitioning=True, hive_schema={"YEAR": pl.Int16})
            .filter(pl.col("YEAR") == 2008)
            .group_by("STNAME").agg(pl.col("ROA").mean()))


def polars_part(summary: dict) -> None:
    core = K.polars_core(SRC)
    text = ["=== CORE PIPELINE: logical plan as written ===", core.explain(optimized=False),
            "\n=== CORE PIPELINE: after optimisation ===", core.explain(optimized=True),
            "\n=== CORE + filter YEAR == 2008 AFTER the windows: after optimisation ===",
            core.filter(pl.col("YEAR") == 2008).explain(optimized=True),
            "\n=== SELECTIVE QUERY: as written ===", selective_query().explain(optimized=False),
            "\n=== SELECTIVE QUERY: after optimisation ===", selective_query().explain(optimized=True)]
    (C.REPORTS / "p3_plan_polars.txt").write_text("\n".join(text), encoding="utf-8")

    dot = core.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    (C.REPORTS / "p3_polars_streaming.dot").write_text(dot, encoding="utf-8")
    labels = re.findall(r'label="([^"\\]+)', dot)
    kinds: dict[str, int] = {}
    for lab in labels:
        k = lab.split("\\n")[0].split("[")[0].strip()
        kinds[k] = kinds.get(k, 0) + 1
    summary["polars_streaming_nodes"] = dict(sorted(kinds.items(), key=lambda kv: -kv[1]))

    # Polars 2 removed LazyFrame.profile(), so the time of each step comes from timing the query up
    # to that step, on both engines. Medians of 5; the steps share work, so these are approximate.
    steps = ["T1-T3 scan, filter, join", "sort by bank", "T4-T6 clean, quarterly flows, ratios", "T7 bank windows",
             "T8 quarter ranks", "T8 state median, T9 label", None]
    rows = []
    for engine in ("in-memory", "streaming"):
        prev = 0.0
        for st in steps:
            q = K.polars_core(SRC, upto=st)
            ts = []
            for _ in range(5):
                t = time.perf_counter()
                q.collect(engine=engine)
                ts.append(time.perf_counter() - t)
            m = sorted(ts)[2]
            rows.append({"engine": engine, "up_to_step": st or "T10 final sort", "cumulative_s": round(m, 3), "added_s": round(m - prev, 3)})
            prev = m
    prof = pl.DataFrame(rows)
    prof.write_csv(C.REPORTS / "p3_polars_profile.csv")
    print(prof)
    res = core.collect()
    summary["polars_rows"] = res.height

    # Optimiser effect, 5 repeats each, median. One process, files already in the page cache.
    flags = {"all rules on": pl.QueryOptFlags(),
             "no projection pushdown": pl.QueryOptFlags(projection_pushdown=False),
             "no predicate pushdown": pl.QueryOptFlags(predicate_pushdown=False),
             "no common sub-plan or sub-expression": pl.QueryOptFlags(comm_subplan_elim=False, comm_subexpr_elim=False),
             "all rules off": pl.QueryOptFlags.none()}
    rows = []
    for qname, q in {"core pipeline": K.polars_core(SRC), "selective query": selective_query()}.items():
        for fname, f in flags.items():
            ts = []
            for _ in range(5):
                t = time.perf_counter()
                q.collect(optimizations=f)
                ts.append(time.perf_counter() - t)
            rows.append({"query": qname, "optimizer": fname, "median_s": round(sorted(ts)[2], 3), "min_s": round(min(ts), 3), "max_s": round(max(ts), 3)})
            print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(C.REPORTS / "p3_optimizer_effect.csv")


def pushdown_safety(summary: dict) -> None:
    """Filtering before a window changes the answer. Engines must not, and do not, push it down."""
    after = K.polars_core(SRC).filter(pl.col("YEAR") == 2008).collect()
    naive = K.polars_core(SRC, pre_filter=pl.col("YEAR") == 2008).collect()
    j = after.join(naive, on=["CERT", "REPDTE"], suffix="__n")
    diffs = {c: j.select(pl.col(c).ne_missing(pl.col(f"{c}__n")).sum()).item()
             for c in ["eq_ta_d4", "asset_g4", "roa_4q", "roa_std8", "NETINC_Q", "npl_d1", "eq_ta_pct", "y"]}
    summary["filter_before_windows_changes"] = {"rows": j.height, "cells_changed": diffs}


def fetch(url: str):
    return json.load(urllib.request.urlopen(url, timeout=60))


def spark_part(summary: dict) -> None:
    from p3_spark_session import get_spark
    from pyspark.sql import functions as F

    spark = get_spark(threads=0, app="bft65b-plans", extra={"spark.local.dir": str(C.DATA / "_p3" / "_spark_tmp")})
    sc = spark.sparkContext
    jvm = sc._jvm
    base = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}"

    def plan(df, mode: str) -> str:
        return jvm.PythonSQLUtils.explainString(df._jdf.queryExecution(), mode)

    df = K.spark_core(spark, SRC)
    before = plan(df, "formatted")
    extended = plan(df, "extended")
    t = time.perf_counter()
    arrow = df.toArrow()
    took = time.perf_counter() - t
    final = plan(df, "formatted")

    sel = (spark.read.parquet(str(SRC / "financials")).filter(F.col("YEAR") == 2008).groupBy("STNAME").agg(F.avg("ROA")))
    sel.collect()
    sel_plan = plan(sel, "formatted")
    core_year = K.spark_core(spark, SRC).filter(F.col("YEAR") == 2008)
    core_year_plan = plan(core_year, "formatted")
    bank = spark.read.parquet(str(SRC / "financials")).filter(F.col("CERT") == 14).select("REPDTE", "ASSET")
    bank_plan = plan(bank, "formatted")

    (C.REPORTS / "p3_plan_spark.txt").write_text("\n\n".join([
        "=== CORE: parsed, analyzed, optimized, physical (before running) ===", extended,
        "=== CORE: formatted physical plan before running (AQE not final) ===", before,
        "=== CORE: formatted physical plan after running (AQE final) ===", final,
        "=== SELECTIVE QUERY: YEAR == 2008, mean ROA by state ===", sel_plan,
        "=== ONE BANK: CERT == 14 ===", bank_plan,
        "=== CORE + filter YEAR == 2008 after the windows (not run) ===", core_year_plan]), encoding="utf-8")

    stages = fetch(f"{base}/stages?status=complete")
    rows = []
    for s in stages:
        rows.append({"stage": s["stageId"], "tasks": s["numTasks"], "input_mb": round(s["inputBytes"] / 1e6, 1),
                     "shuffle_read_mb": round(s["shuffleReadBytes"] / 1e6, 1), "shuffle_write_mb": round(s["shuffleWriteBytes"] / 1e6, 1),
                     "output_records": s["outputRecords"], "spill_disk_mb": round(s["diskBytesSpilled"] / 1e6, 1),
                     "executor_run_s": round(s["executorRunTime"] / 1e3, 1), "jvm_gc_s": round(s.get("jvmGcTime", 0) / 1e3, 1),
                     "name": s["name"][:60]})
    st = pl.DataFrame(rows).sort("stage")
    st.write_csv(C.REPORTS / "p3_spark_stages.csv")

    # The final adaptive plan is the tree above "== Initial Plan =="; details follow it, numbered.
    tree_text = final.split("+- == Initial Plan ==")[0]
    details = final.split("+- == Initial Plan ==")[1]
    ex = []
    for m in re.finditer(r"ShuffleQueryStage \((\d+)\), Statistics\(sizeInBytes=([\d.]+ \w+), rowCount=([\d.E+]+)\)\s*\n[ :|+\-*]*Exchange \((\d+)\)", tree_text):
        arg = re.search(rf"\({m.group(4)}\) Exchange\s*\n.*?Arguments: ([^\n]+)", details, re.S)
        ex.append({"exchange": int(m.group(4)), "size": m.group(2), "rows": m.group(3),
                   "partitioning": re.sub(r"#\d+L?", "", arg.group(1)) if arg else ""})
    pl.DataFrame(ex).write_csv(C.REPORTS / "p3_spark_exchanges.csv")

    def count(pattern: str) -> int:
        return len(re.findall(pattern, tree_text))

    core_stages = st.filter(pl.col("name").str.starts_with("toArrow"))
    summary["spark"] = {
        "toArrow_s": round(took, 1), "rows": arrow.num_rows, "final_flag": "isFinalPlan=true" in final,
        "final_plan_nodes": {k: count(rf"\b{k}\b") for k in ("Exchange", "ReusedExchange", "Sort", "Window", "SortMergeJoin", "BroadcastHashJoin", "HashAggregate", "ObjectHashAggregate", "AQEShuffleRead")},
        "parquet_scans_of_financials": count(r"Scan parquet"),
        "stages": core_stages.height, "shuffle_write_gb": round(core_stages["shuffle_write_mb"].sum() / 1e3, 3),
        "shuffle_read_gb": round(core_stages["shuffle_read_mb"].sum() / 1e3, 3), "spill_mb": core_stages["spill_disk_mb"].sum(),
        "executor_run_s": core_stages["executor_run_s"].sum(), "gc_s": core_stages["jvm_gc_s"].sum(),
        "selective_partition_filter": [ln.strip() for ln in sel_plan.splitlines() if "PartitionFilters" in ln][:2],
        "bank_pushed_filter": [ln.strip() for ln in bank_plan.splitlines() if "PushedFilters" in ln][:2],
        "core_year_partition_filter": [ln.strip() for ln in core_year_plan.splitlines() if "PartitionFilters" in ln][:3],
    }
    spark.stop()


def main() -> None:
    summary: dict = {}
    polars_part(summary)
    pushdown_safety(summary)
    spark_part(summary)
    (C.REPORTS / "p3_plan_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
