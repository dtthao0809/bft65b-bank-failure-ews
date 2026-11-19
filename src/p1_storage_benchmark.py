"""
Part 1c - Storage benchmark: measure the bronze layout against CSV and against the alternatives.

Each design dimension is varied on the SAME table:
  format        CSV, CSV+gzip, Parquet, Delta
  compression   none, Snappy, ZSTD-3
  partitioning  one file, YEAR, quarter
  sort          (CERT, REPDTE) vs (REPDTE, CERT)
  row group     4k, 16k, 64k rows
  types         Int64 amounts vs all-Float64

Workloads, chosen because the later parts of the pipeline actually run them:
  W1 full scan, Polars        W2 full scan, pandas
  W3 three columns, all rows  W4 one year, mean ROA by state
  W5 one bank's history       W6 one quarter's cross-section

Fairness:
  * every measurement runs in a fresh process, so no engine reuses caches from an earlier query;
  * runs are in random order (fixed seed), so slow drift on the machine hits all layouts alike;
  * one discarded warm-up per pair puts all files in the OS page cache. This FAVOURS CSV: the
    numbers exclude disk time, and CSV is the layout with the most bytes to read;
  * CSV is given the full schema by hand, so it pays no inference cost in the timings;
  * every layout must return the same answer, or the run fails.

Timings depend on the machine, so a machine-independent measure is reported too: the bytes a
reader must fetch for each query, computed from the Parquet footers (partition values plus
row-group min/max). This is the plan available from metadata, not an I/O trace.

Usage:  python src/p1_storage_benchmark.py [--repeat 5] [--keep]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

BASE = {"compression": "zstd", "level": 3, "rg": 16_384, "sort": ("CERT", "REPDTE")}
LAYOUTS: dict[str, dict] = {
    "csv":              {"kind": "csv"},
    "csv_gz":           {"kind": "csv_gz"},
    "pq_none_1file":    {"kind": "file", **BASE, "compression": "uncompressed", "level": None},
    "pq_snappy_1file":  {"kind": "file", **BASE, "compression": "snappy", "level": None},
    "pq_zstd_1file":    {"kind": "file", **BASE},
    "pq_year":          {"kind": "hive", "key": "YEAR", **BASE},                # chosen
    "pq_year_rg4k":     {"kind": "hive", "key": "YEAR", **BASE, "rg": 4_096},
    "pq_year_rg64k":    {"kind": "hive", "key": "YEAR", **BASE, "rg": 65_536},
    "pq_year_sortdate": {"kind": "hive", "key": "YEAR", **BASE, "sort": ("REPDTE", "CERT")},
    "pq_year_float":    {"kind": "hive", "key": "YEAR", **BASE, "float_amounts": True},
    "pq_quarter":       {"kind": "hive", "key": "RQ", **BASE},
    "delta_year":       {"kind": "delta", **BASE},
}
CHOSEN = "pq_year"
WORKLOADS = {"W1": "full scan, Polars", "W2": "full scan, pandas", "W3": "3 columns, all rows",
             "W4": "1 year, ROA by state", "W5": "1 bank's history", "W6": "1 quarter's cross-section"}
Q_DATE = date(2008, 12, 31)
Q_YEAR = 2008
CSV_SCHEMA = {**{c: t for c, t in C.FIN_SCHEMA.items()}, "REPDTE": pl.String, "YEAR": pl.Int16}


def layout_path(name: str) -> Path:
    return C.BENCH_DIR / name


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------
def write_layout(df: pl.DataFrame, name: str) -> None:
    s, p = LAYOUTS[name], layout_path(name)
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True)
    kind = s["kind"]
    if kind == "csv":
        df.write_csv(p / "fin.csv")
    elif kind == "csv_gz":
        with gzip.open(p / "fin.csv.gz", "wb", compresslevel=6) as fh:
            df.write_csv(fh)
    else:
        d = df.sort(*s["sort"])
        if s.get("float_amounts"):
            d = d.with_columns([pl.col(c).cast(pl.Float64) for c in C.INT_FIELDS])
        opts = {"compression": s["compression"], "statistics": True, "row_group_size": s["rg"]}
        if s["level"] is not None:
            opts["compression_level"] = s["level"]
        if kind == "file":
            d.write_parquet(p / "fin.parquet", **opts)
        elif kind == "hive":
            if s["key"] == "RQ":
                d = d.drop("YEAR").with_columns(
                    pl.col("REPDTE").dt.strftime("%Y%m%d").cast(pl.Int32).alias("RQ"))
            for (k,), part in d.partition_by(s["key"], as_dict=True, include_key=False).items():
                folder = p / f"{s['key']}={k}"
                folder.mkdir()
                part.write_parquet(folder / "part-0.parquet", **opts)
        elif kind == "delta":
            from deltalake import WriterProperties
            shutil.rmtree(p)
            d.write_delta(p, delta_write_options={
                "partition_by": ["YEAR"],
                "writer_properties": WriterProperties(compression="ZSTD", compression_level=s["level"],
                                                       max_row_group_size=s["rg"])})


def disk_stats(p: Path) -> tuple[float, int]:
    files = [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 2**20, len(files)


# ----------------------------------------------------------------------------
# Readers and queries
# ----------------------------------------------------------------------------
def scan(name: str) -> pl.LazyFrame:
    s, p = LAYOUTS[name], layout_path(name)
    if s["kind"] in ("csv", "csv_gz"):
        f = p / ("fin.csv" if s["kind"] == "csv" else "fin.csv.gz")
        return pl.scan_csv(f, schema=CSV_SCHEMA).with_columns(pl.col("REPDTE").str.to_date())
    if s["kind"] == "file":
        return pl.scan_parquet(p / "fin.parquet")
    if s["kind"] == "delta":
        # pl.scan_delta fails when the path contains a space: deltalake returns percent-encoded
        # URIs (Final%20BigData) and Polars does not decode them. So read the Delta log for the
        # file list, then use the same native Parquet reader as the other layouts.
        from urllib.parse import unquote

        from deltalake import DeltaTable
        files = [unquote(u) for u in DeltaTable(str(p)).file_uris()]
        return pl.scan_parquet(files, hive_partitioning=True, hive_schema={"YEAR": pl.Int16})
    dtype = pl.Int16 if s["key"] == "YEAR" else pl.Int32
    return pl.scan_parquet(p / "**" / "*.parquet", hive_partitioning=True, hive_schema={s["key"]: dtype})


def year_pred(name: str) -> pl.Expr:
    if LAYOUTS[name].get("key") == "RQ":
        return pl.col("RQ").is_between(Q_YEAR * 10_000, Q_YEAR * 10_000 + 9_999)
    return pl.col("YEAR") == Q_YEAR


def quarter_pred(name: str) -> pl.Expr:
    if LAYOUTS[name].get("key") == "RQ":
        return pl.col("RQ") == int(Q_DATE.strftime("%Y%m%d"))
    return (pl.col("YEAR") == Q_YEAR) & (pl.col("REPDTE") == Q_DATE)


def query(name: str, w: str, cert: int) -> pl.LazyFrame:
    lf = scan(name)
    if w == "W1":
        return lf
    if w == "W3":
        return lf.select("CERT", "REPDTE", "ASSET")
    if w == "W4":
        return lf.filter(year_pred(name)).group_by("STNAME").agg(pl.col("ROA").mean())
    if w == "W5":
        return lf.filter(pl.col("CERT") == cert)
    if w == "W6":
        return lf.filter(quarter_pred(name)).select("CERT", "ASSET", "ROA")
    raise ValueError(w)


def read_pandas(name: str):
    import pandas as pd
    s, p = LAYOUTS[name], layout_path(name)
    if s["kind"] == "csv":
        return pd.read_csv(p / "fin.csv", parse_dates=["REPDTE"])
    if s["kind"] == "csv_gz":
        return pd.read_csv(p / "fin.csv.gz", parse_dates=["REPDTE"])
    if s["kind"] == "file":
        return pd.read_parquet(p / "fin.parquet")
    if s["kind"] == "delta":
        from deltalake import DeltaTable
        return DeltaTable(str(p)).to_pandas()
    return pd.read_parquet(p)


def peak_mb() -> float:
    m = psutil.Process().memory_info()
    return getattr(m, "peak_wset", m.rss) / 2**20


def worker(name: str, w: str, cert: int) -> None:
    """One measurement, in its own process. Prints timing, peak memory and a result checksum."""
    # Library import time must not count towards any layout.
    if w == "W2":
        import pandas  # noqa: F401
        import pyarrow.dataset  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    if LAYOUTS[name]["kind"] == "delta":
        import deltalake  # noqa: F401
    base = peak_mb()
    t0 = time.perf_counter()
    if w == "W2":
        out = read_pandas(name)
        secs, peak = time.perf_counter() - t0, peak_mb()
        check = [len(out), float(out["ASSET"].sum())]
    else:
        out = query(name, w, cert).collect()
        secs, peak = time.perf_counter() - t0, peak_mb()
        col = "ROA" if w == "W4" else "ASSET"
        check = [out.height, float(out[col].sum())]
    print(json.dumps({"seconds": secs, "peak_mb": peak, "base_mb": base, "check": check}))


# ----------------------------------------------------------------------------
# Machine-independent measure: bytes needed, planned from the Parquet footers
# ----------------------------------------------------------------------------
def planned_bytes(name: str, w: str, cert: int) -> tuple[int, int]:
    """(bytes a reader must fetch, total data bytes) for a query, from metadata only."""
    import pyarrow.parquet as pq
    s, p = LAYOUTS[name], layout_path(name)
    if s["kind"] in ("csv", "csv_gz"):
        size = sum(f.stat().st_size for f in p.iterdir())
        return size, size            # text must be read and parsed from start to end

    cols = {"W1": None, "W2": None, "W3": {"CERT", "REPDTE", "ASSET"},
            "W4": {"STNAME", "ROA", "YEAR"}, "W5": None, "W6": {"CERT", "ASSET", "ROA", "REPDTE", "YEAR"}}[w]
    needed = total = 0
    for f in sorted(p.rglob("*.parquet")):
        part = next((x.split("=", 1) for x in f.parts if x.startswith(("YEAR=", "RQ="))), None)
        keep_file = True
        if part and w in ("W4", "W6"):
            v = int(part[1])
            if part[0] == "YEAR":
                keep_file = v == Q_YEAR
            elif w == "W4":
                keep_file = v // 10_000 == Q_YEAR
            else:
                keep_file = v == int(Q_DATE.strftime("%Y%m%d"))
        md = pq.ParquetFile(f).metadata
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            chunks = {rg.column(j).path_in_schema: rg.column(j) for j in range(rg.num_columns)}
            total += sum(c.total_compressed_size for c in chunks.values())
            if not keep_file or not _rg_may_match(chunks, w, cert):
                continue
            needed += sum(c.total_compressed_size for n, c in chunks.items() if cols is None or n in cols)
    return needed, total


def _rg_may_match(chunks: dict, w: str, cert: int) -> bool:
    def within(col: str, value) -> bool:
        st = chunks[col].statistics if col in chunks else None
        return st is None or not st.has_min_max or st.min <= value <= st.max
    if w == "W5":
        return within("CERT", cert)
    if w == "W4" and "YEAR" in chunks:
        return within("YEAR", Q_YEAR)
    if w == "W6":
        return within("REPDTE", Q_DATE)
    return True


# ----------------------------------------------------------------------------
# Schema fidelity: what does each format hand back with no help from the reader?
# ----------------------------------------------------------------------------
def schema_fidelity(df: pl.DataFrame) -> dict:
    import pandas as pd
    csv = layout_path("csv") / "fin.csv"
    res: dict = {}
    try:
        got = pl.read_csv(csv)
        res["polars_csv_default_inference"] = {
            "read_ok": True,
            "columns_with_wrong_type": sorted(c for c in df.columns if got.schema[c] != df.schema[c])}
    except Exception as exc:
        res["polars_csv_default_inference"] = {"read_ok": False, "error": type(exc).__name__,
                                               "message": str(exc).splitlines()[0][:200]}
    got_pd = pd.read_csv(csv)
    res["pandas_csv_default"] = {
        "repdte_type": str(got_pd["REPDTE"].dtype),
        "int_columns_read_as_float": sorted(c for c in C.INT_FIELDS if got_pd[c].dtype == "float64"),
    }
    # Values survive CSV once the schema is supplied by hand: the problem is types, not precision.
    back = pl.read_csv(csv, schema=CSV_SCHEMA).with_columns(pl.col("REPDTE").str.to_date())
    res["csv_values_equal_with_handwritten_schema"] = back.select(df.columns).equals(df)
    pq_back = scan(CHOSEN).collect().select(df.columns).sort("CERT", "REPDTE")
    res["parquet_types_equal"] = pq_back.schema == df.schema
    res["parquet_values_equal"] = pq_back.equals(df.sort("CERT", "REPDTE"))
    return res


def parquet_evidence(rep: Path) -> None:
    """Per-column size and encodings in the chosen layout, and the optimised query plans."""
    import pyarrow.parquet as pq
    agg: dict[str, dict] = {}
    n_rg = []
    for f in sorted(layout_path(CHOSEN).rglob("*.parquet")):
        md = pq.ParquetFile(f).metadata
        n_rg.append(md.num_row_groups)
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            for j in range(rg.num_columns):
                c = rg.column(j)
                a = agg.setdefault(c.path_in_schema, {"compressed": 0, "uncompressed": 0, "encodings": set()})
                a["compressed"] += c.total_compressed_size
                a["uncompressed"] += c.total_uncompressed_size
                a["encodings"] |= set(c.encodings)
    (pl.DataFrame([{"column": k, "compressed_kb": round(v["compressed"] / 1024, 1),
                    "uncompressed_kb": round(v["uncompressed"] / 1024, 1),
                    "ratio": round(v["uncompressed"] / max(v["compressed"], 1), 2),
                    "encodings": ", ".join(sorted(v["encodings"]))} for k, v in agg.items()])
     .sort("compressed_kb", descending=True).write_csv(rep / "p1_parquet_columns.csv"))
    (rep / "p1_parquet_row_groups.json").write_text(json.dumps(
        {"files": len(n_rg), "row_groups": sum(n_rg), "row_groups_per_file_min": min(n_rg),
         "row_groups_per_file_max": max(n_rg)}, indent=1))


def save_plans(rep: Path, cert: int) -> None:
    text = []
    for w in ("W4", "W5"):
        q = query(CHOSEN, w, cert)
        text += [f"=== {w}: {WORKLOADS[w]} ({CHOSEN}) ===", "--- logical plan, not optimised ---",
                 q.explain(optimized=False), "--- optimised plan ---", q.explain(), ""]
    (rep / "p1_polars_plans.txt").write_text("\n".join(text), encoding="utf-8")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_worker(name: str, w: str, cert: int) -> dict:
    out = subprocess.run([sys.executable, __file__, "--worker", name, w, str(cert)],
                         capture_output=True, text=True, check=False)
    if out.returncode:
        raise RuntimeError(f"{name}/{w} failed:\n{out.stderr[-1500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def fig_summary(tbl: pl.DataFrame, rep: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, BASE_C, HILITE = "#1f2328", "#57606a", "#d8dee4", "#a8b1bb", "#2463eb"
    order = ["csv", "csv_gz", "pq_snappy_1file", "pq_zstd_1file", "pq_quarter", CHOSEN, "delta_year"]
    names = {"csv": "CSV", "csv_gz": "CSV + gzip", "pq_snappy_1file": "Parquet Snappy, 1 file",
             "pq_zstd_1file": "Parquet ZSTD, 1 file", "pq_quarter": "Parquet ZSTD, by quarter",
             CHOSEN: "Parquet ZSTD, by year (chosen)", "delta_year": "Delta ZSTD, by year"}
    rows = {r["layout"]: r for r in tbl.iter_rows(named=True)}
    panels = [("size_mb", "Size on disk (MB)", False), ("W3_s", "3 columns, all rows (s, log)", True),
              ("W4_s", "1 year, ROA by state (s, log)", True), ("W5_s", "1 bank's history (s, log)", True)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    y = list(range(len(order)))
    for ax, (col, title, log) in zip(axes, panels):
        vals = [rows[n][col] for n in order]
        ax.barh(y, vals, height=0.55, color=[HILITE if n == CHOSEN else BASE_C for n in order])
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:,.0f}" if not log else f" {v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        if log:
            ax.set_xscale("log")
            ax.set_xlim(min(vals) / 1.5, max(vals) * 6)
        else:
            ax.set_xlim(0, max(vals) * 1.25)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.minorticks_off()
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
    axes[0].set_yticks(y, [names[n] for n in order], fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(rep / "fig_p1_storage.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--keep", action="store_true", help="keep the scratch copies")
    ap.add_argument("--worker", nargs=3, metavar=("LAYOUT", "WORKLOAD", "CERT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker[0], a.worker[1], int(a.worker[2]))

    rep = C.REPORTS
    rep.mkdir(parents=True, exist_ok=True)
    df = (pl.scan_parquet(C.DATA_BRONZE / "financials" / "**" / "*.parquet", hive_partitioning=True,
                          hive_schema={"YEAR": pl.Int16})
          .collect().sort("YEAR", "CERT", "REPDTE"))
    # The bank with the longest history; the smallest CERT breaks ties so the choice is fixed.
    cert = int(df.group_by("CERT").len().sort(["len", "CERT"], descending=[True, False])["CERT"][0])
    print(f"Table: {df.height:,} rows x {df.width} columns; lookup CERT={cert}", flush=True)

    C.BENCH_DIR.mkdir(parents=True, exist_ok=True)
    layout_rows = {}
    for name in LAYOUTS:
        # gzip takes tens of seconds and shows no variance worth the wait
        n_writes = 1 if name == "csv_gz" else 3
        times = []
        for _ in range(n_writes):
            t0 = time.perf_counter()
            write_layout(df, name)
            times.append(time.perf_counter() - t0)
        size, files = disk_stats(layout_path(name))
        layout_rows[name] = {"layout": name, "size_mb": round(size, 1), "files": files,
                             "write_s": round(statistics.median(times), 3)}
        print(f"  wrote {name:<18} {size:8.1f} MB  {files:>4} files  {layout_rows[name]['write_s']} s", flush=True)

    tasks = [(n, w) for n in LAYOUTS for w in WORKLOADS]
    rng = random.Random(C.SEED)
    warmups = tasks[:]
    rng.shuffle(warmups)
    for n, w in warmups:
        run_worker(n, w, cert)
    print("Warm-up done", flush=True)

    runs = tasks * a.repeat
    rng.shuffle(runs)
    records, answers = [], {}
    t_all = time.perf_counter()
    for i, (n, w) in enumerate(runs, 1):
        r = run_worker(n, w, cert)
        records.append({"layout": n, "workload": w, "seconds": r["seconds"],
                        "peak_mb": r["peak_mb"], "base_mb": r["base_mb"]})
        ref = answers.setdefault(w, (n, r["check"]))
        if ref[1][0] != r["check"][0] or not math.isclose(ref[1][1], r["check"][1], rel_tol=1e-9):
            raise AssertionError(f"{w}: {n} returned {r['check']} but {ref[0]} returned {ref[1]}")
        if i % 60 == 0:
            print(f"  {i}/{len(runs)} runs, {time.perf_counter() - t_all:.0f} s", flush=True)
    long = pl.DataFrame(records)
    long.write_csv(rep / "p1_storage_runs.csv")

    med = long.group_by("layout", "workload").agg(
        pl.col("seconds").median().alias("med"), pl.col("seconds").min().alias("lo"),
        pl.col("seconds").max().alias("hi"), pl.col("peak_mb").median().alias("peak"),
        pl.col("base_mb").median().alias("base"))
    rows = []
    for name, row in layout_rows.items():
        m = med.filter(pl.col("layout") == name)
        for w in WORKLOADS:
            x = m.filter(pl.col("workload") == w).row(0, named=True)
            need, total = planned_bytes(name, w, cert)
            row |= {f"{w}_s": round(x["med"], 4), f"{w}_spread_s": round(x["hi"] - x["lo"], 4),
                    f"{w}_peak_extra_mb": round(x["peak"] - x["base"], 0),
                    f"{w}_needed_mb": round(need / 2**20, 2),
                    f"{w}_needed_pct": round(100 * need / total, 1) if total else None}
        rows.append(row)
    tbl = pl.DataFrame(rows)
    tbl.write_csv(rep / "p1_storage_benchmark.csv")

    base = tbl.filter(pl.col("layout") == "csv").row(0, named=True)
    short = tbl.select(
        "layout", "size_mb", "files", "write_s", "W1_s", "W2_s", "W3_s", "W4_s", "W5_s", "W6_s",
        *[pl.col(f"{w}_needed_pct") for w in ("W3", "W4", "W5", "W6")]
    ).with_columns((pl.col("size_mb") / base["size_mb"]).round(3).alias("size_vs_csv"))
    (rep / "p1_storage_benchmark.md").write_text(
        short.to_pandas().to_markdown(index=False), encoding="utf-8")

    fid = schema_fidelity(df)
    (rep / "p1_schema_fidelity.json").write_text(json.dumps(fid, indent=1))
    parquet_evidence(rep)
    save_plans(rep, cert)
    fig_summary(tbl, rep)
    mem = psutil.virtual_memory().total / 2**30
    (rep / "p1_bench_env.json").write_text(json.dumps({
        "repeats": a.repeat, "lookup_cert": cert, "rows": df.height, "cols": df.width,
        "python": platform.python_version(), "polars": pl.__version__, "os": platform.platform(),
        "cpu_logical": os.cpu_count(), "ram_gb": round(mem, 1),
        "polars_threads": pl.thread_pool_size(), "page_cache": "warm (one discarded run per pair)"},
        indent=1))
    print(short)
    print(json.dumps(fid, indent=1))
    if not a.keep:
        shutil.rmtree(C.BENCH_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
