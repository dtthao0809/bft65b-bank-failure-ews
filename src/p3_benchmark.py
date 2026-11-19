from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

MEM_LIMIT_GB = 13.5
MIN_FREE_GB = 0.5
START_FREE_GB = 3.0              # wait for this much free RAM before a run starts
SPARK_HEAP = "3g"                # other applications leave about 4.5 GB free on this laptop
WORK = C.DATA / "_p3"
RUNS_CSV = C.REPORTS / "p3_benchmark_runs.csv"
BASE_ROWS, BASE_POS = 1_666_719, 10_860
ONLY = ""                        # --only: restrict a suite to engines whose name contains this


def scaled_source(x: int) -> Path:
    """Copy the bronze layer x times, each copy with its own CERT range. Layout stays the same."""
    if x == 1:
        return C.DATA_BRONZE
    dst = WORK / "scaled" / f"x{x}"
    if (dst / "_DONE").exists():
        return dst
    shutil.rmtree(dst, ignore_errors=True)
    offset = lambda k: pl.col("CERT") + k * 1_000_000  # noqa: E731  CERT stays well below the Int32 limit
    for name in ("failures", "institutions"):
        df = pl.read_parquet(C.DATA_BRONZE / f"{name}.parquet")
        dst.mkdir(parents=True, exist_ok=True)
        pl.concat([df.with_columns(offset(k).cast(df.schema["CERT"]).alias("CERT")) for k in range(x)]).write_parquet(
            dst / f"{name}.parquet", compression="zstd", compression_level=3)
    for part in sorted((C.DATA_BRONZE / "financials").glob("YEAR=*")):
        df = pl.read_parquet(part / "*.parquet", hive_partitioning=False)
        (dst / "financials" / part.name).mkdir(parents=True)
        pl.concat([df.with_columns(offset(k).cast(pl.Int32).alias("CERT")) for k in range(x)]).write_parquet(
            dst / "financials" / part.name / "part-0.parquet", compression="zstd", compression_level=3,
            statistics=True, row_group_size=C.ROW_GROUP_SIZE)
    (dst / "_DONE").write_text("ok")
    return dst


def tree(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.Error:
        return []


def run_once(engine: str, x: int, threads: int = 0, shuffle: int | None = None, timeout: int = 2400,
             out_name: str | None = None, keep: bool = False, heap: str = SPARK_HEAP) -> dict:
    src = scaled_source(x)
    out = WORK / "out" / (out_name or f"{engine}_x{x}")
    cmd = [sys.executable, str(C.ROOT / "src" / "p3_core.py"), "--engine", engine, "--src", str(src), "--out", str(out)]
    env = dict(os.environ)
    if engine == "spark":
        cmd += ["--threads", str(threads), "--driver-memory", heap]
        if shuffle:
            cmd += ["--shuffle-partitions", str(shuffle)]
    elif engine.startswith("polars") and threads:
        env["POLARS_MAX_THREADS"] = str(threads)

    deadline = time.perf_counter() + 300
    while psutil.virtual_memory().available < START_FREE_GB * 2**30 and time.perf_counter() < deadline:
        time.sleep(2)
    avail_start = round(psutil.virtual_memory().available / 2**30, 2)

    log = WORK / "last_child.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=C.ROOT)
        root = psutil.Process(p.pid)
        peak, cpu, status = 0, {}, "ok"
        while p.poll() is None:
            total = 0
            for proc in tree(root):
                try:
                    total += proc.memory_info().rss
                    t = proc.cpu_times()
                    cpu[proc.pid] = t.user + t.system
                except psutil.Error:
                    pass
            peak = max(peak, total)
            if total > MEM_LIMIT_GB * 2**30 or psutil.virtual_memory().available < MIN_FREE_GB * 2**30:
                status = "memory_guard"
            elif time.perf_counter() - t0 > timeout:
                status = "timeout"
            if status != "ok":
                for proc in reversed(tree(root)):
                    try:
                        proc.kill()
                    except psutil.Error:
                        pass
                break
            time.sleep(0.1)
        p.wait()
    wall = time.perf_counter() - t0
    text = log.read_text(encoding="utf-8", errors="replace")
    res = {}
    for line in text.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if status == "ok" and not res:
        status = "crashed"
        (WORK / f"crash_{engine}_x{x}.log").write_text(text[-6000:], encoding="utf-8")
    rows = pos = None
    if status == "ok":
        o = pl.scan_parquet(out / "*.parquet").select(pl.len().alias("n"), pl.col("y").sum().alias("p")).collect()
        rows, pos = o["n"][0], int(o["p"][0])
        if rows != BASE_ROWS * x or pos != BASE_POS * x:
            status = "wrong_answer"
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return {"engine": engine, "scale": x, "threads": threads, "shuffle_partitions": shuffle or 0, "heap": heap if engine == "spark" else "", "status": status,
            "avail_start_gb": avail_start, "wall_s": res.get("wall_s", round(wall, 3)), "process_s": round(wall, 3), "cpu_s": round(sum(cpu.values()), 1),
            "peak_rss_gb": round(peak / 2**30, 3), "rows": rows, "y1": pos, "session_start_s": res.get("session_start_s"),
            "shuffle_write_bytes": res.get("shuffle_write_bytes"), "spill_disk_bytes": res.get("spill_disk_bytes"), "stages": res.get("stages")}


def run_x1(engine: str, **kw) -> dict:
    """At x1 all engines fit, so a guard stop means other applications took the memory: try again."""
    for attempt in range(4):
        r = run_once(engine, 1, **kw)
        if r["status"] != "memory_guard":
            break
        time.sleep(20)
    return {**r, "retries": attempt}


def append(suite: str, r: dict, rep: int) -> None:
    row = {"suite": suite, "rep": rep, "retries": 0, **r}
    df = pl.DataFrame([row], schema_overrides={"heap": pl.String, "avail_start_gb": pl.Float64, "retries": pl.Int64, "session_start_s": pl.Float64, "rows": pl.Int64, "y1": pl.Int64, "shuffle_write_bytes": pl.Int64,
                                               "spill_disk_bytes": pl.Int64, "stages": pl.Int64, "wall_s": pl.Float64, "process_s": pl.Float64, "cpu_s": pl.Float64, "peak_rss_gb": pl.Float64})
    if RUNS_CSV.exists():
        prior = pl.read_csv(RUNS_CSV, infer_schema_length=None)
        if "heap" not in prior.columns:
            prior = prior.with_columns(pl.when(pl.col("engine") == "spark").then(pl.lit(SPARK_HEAP)).otherwise(pl.lit("")).alias("heap"))
        df = pl.concat([prior.select(df.columns).cast(df.schema), df])
    df.write_csv(RUNS_CSV)
    print(f"[{suite}] rep {rep} {r['engine']} x{r['scale']} t{r['threads']} s{r['shuffle_partitions']}: {r['status']} "
          f"{r['wall_s']:.1f}s rss {r['peak_rss_gb']:.2f}GB cpu {r['cpu_s']:.0f}s", flush=True)


def drop_suite(suite: str) -> None:
    if RUNS_CSV.exists():
        d = pl.read_csv(RUNS_CSV, infer_schema_length=None)
        d.filter(~((pl.col("suite") == suite) & pl.col("engine").str.contains(ONLY, literal=True))).write_csv(RUNS_CSV)


def suite_spark_tune(repeat: int) -> None:
    """Interleaved, so a slow period on the machine hits every setting alike."""
    plans = [{}, {"threads": 18}, {"shuffle": 18}, {"shuffle": 54}, {"heap": "5g"}]
    runs = [(kw, i) for kw in plans for i in range(repeat)]
    random.Random(C.SEED).shuffle(runs)
    for kw, i in runs:
        append("spark_tune", run_x1("spark", **kw), i)


def suite_engines(repeat: int, spark_repeat: int) -> None:
    plans = [(e, {}) for e in ("polars_mem", "polars_stream", "polars_noopt", "pandas_cols", "pandas_all")] + \
            [("spark", {}), ("spark", {"shuffle": 18})]
    plans = [p for p in plans if ONLY in p[0]]
    for e, kw in plans:
        run_once(e, 1, out_name=f"warm_{e}", **kw)     # warm-up, discarded
    runs = [(e, kw, i) for e, kw in plans for i in range(spark_repeat if e == "spark" else repeat)]
    random.Random(C.SEED).shuffle(runs)
    for e, kw, i in runs:
        append("engines", run_x1(e, out_name=e if not kw else f"{e}_s{kw['shuffle']}", keep=True, **kw), i)
    shutil.rmtree(WORK / "out" / "spark_s18", ignore_errors=True)
    for name in ("polars_noopt", "pandas_all"):
        shutil.rmtree(WORK / "out" / name, ignore_errors=True)


def suite_threads(repeat: int, spark_repeat: int) -> None:
    runs = [("polars_mem", t, i) for t in (1, 2, 4, 8, 12, 18) for i in range(repeat)] + \
           [("polars_stream", t, i) for t in (1, 2, 4, 8, 12, 18) for i in range(repeat)] + \
           [("spark", t, i) for t in (1, 2, 4, 8, 12, 18) for i in range(spark_repeat)]
    runs = [r for r in runs if ONLY in r[0]]
    random.Random(C.SEED).shuffle(runs)
    for e, t, i in runs:
        append("threads", run_x1(e, threads=t, timeout=1800), i)


def suite_scale(scales: list[int], engines: list[str], spark_heap: str) -> None:
    """Ascending sizes; an engine that failed at a size is not tried on larger ones.

    "polars_mem@4" means 4 threads: fewer threads use less memory. Spark gets a fixed heap, and
    18 shuffle partitions per copy of the data, so partitions stay about the same size.
    """
    failed: set[str] = set()
    for x in scales:
        print(f"preparing x{x}", flush=True)
        scaled_source(x)
        order = [e for e in engines if e not in failed]
        random.Random(C.SEED + x).shuffle(order)
        for spec in order:
            e, _, t = spec.partition("@")
            kw = {"threads": int(t or 0)}
            if e == "spark":
                kw.update(shuffle=18 * x, heap=spark_heap)
            for attempt in range(2):                 # other applications may have taken the memory
                r = run_once(e, x, **kw)
                if r["status"] != "memory_guard":
                    break
                time.sleep(30)
            append("scale", {**r, "retries": attempt}, 0)
            if r["status"] != "ok":
                failed.add(spec)


def write_env() -> None:
    import pandas

    vm = psutil.virtual_memory()
    (C.REPORTS / "p3_bench_env.json").write_text(json.dumps({
        "os": platform.platform(), "cpu_logical": psutil.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(vm.total / 2**30, 1), "python": platform.python_version(), "polars": pl.__version__,
        "pandas": pandas.__version__, "mem_limit_gb": MEM_LIMIT_GB, "disk": "local, same drive for input and output"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["engines", "threads", "scale", "spark_tune"])
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--spark-repeat", type=int, default=3)
    ap.add_argument("--scales", default="2,4,8,16")
    ap.add_argument("--engines", default="polars_mem,polars_stream,pandas_cols,polars_mem@4,polars_stream@4,spark")
    ap.add_argument("--keep-prior", action="store_true", help="add to this suite's earlier rows instead of replacing them")
    ap.add_argument("--only", default="")
    ap.add_argument("--spark-heap", default="2g", help="scale suite only")
    a = ap.parse_args()
    global ONLY
    ONLY = a.only
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    write_env()
    if not a.keep_prior:
        drop_suite(a.suite)
    if a.suite == "engines":
        suite_engines(a.repeat, a.spark_repeat)
    elif a.suite == "spark_tune":
        suite_spark_tune(a.spark_repeat)
    elif a.suite == "threads":
        suite_threads(min(a.repeat, 3), min(a.spark_repeat, 2))
    else:
        suite_scale([int(s) for s in a.scales.split(",")], a.engines.split(","), a.spark_heap)
    d = pl.read_csv(RUNS_CSV, infer_schema_length=None).filter(pl.col("suite") == a.suite)
    print(d.group_by("engine", "scale", "threads", "shuffle_partitions", "status")
          .agg(pl.col("wall_s").median(), pl.col("peak_rss_gb").median(), pl.len()).sort("engine", "scale", "threads"))


if __name__ == "__main__":
    main()
