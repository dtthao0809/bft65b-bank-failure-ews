"""
Part 2c - Automated tests of the silver panel. Exit code 0 only if every test passes.

The two that matter most for the exam's leakage gate:
  truncation  rebuild the panel from data cut at a past date. Every point-in-time column must
              equal the full build. A column that used any later data would change.
  label       recompute y independently, with pandas and plain arithmetic, from the Failures table.

Usage:  python src/p2_checks.py      (run after p2_build_panel.py)
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from p2_build_panel import END_Q, LAST_MODEL_IDX, SCOPE_START, build_panel, failure_events, load_bronze  # noqa: E402

CUTOFFS = [date(1990, 12, 31), date(2008, 12, 31), date(2019, 12, 31)]
FORBIDDEN = {"ACTIVE", "CHANGEC1", "ENDEFYMD", "BKCLASS", "FAILDATE", "COST", "QBFASSET", "QBFDEP", "RESTYPE", "RESTYPE1"}
# (CERT, name fragment, failure date), read from public FDIC records and asserted against the data.
KNOWN = [(24735, "SILICON VALLEY", date(2023, 3, 10)), (59017, "FIRST REPUBLIC", date(2023, 5, 1)),
         (32633, "WASHINGTON MUTUAL", date(2008, 9, 25))]


def run() -> list[dict]:
    fin, fail, inst = load_bronze()
    p = pl.read_parquet(C.DATA_SILVER / "panel" / "**" / "*.parquet", hive_partitioning=True,
                        hive_schema={C.PARTITION_KEY: pl.Int16})
    schema = json.loads((C.DATA_SILVER / "_schema_panel.json").read_text())
    safe = schema["point_in_time_columns"]
    tests: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        tests.append({"test": name, "passed": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    check("key unique", p.height == p.select("CERT", "REPDTE").unique().height, f"{p.height:,} rows")

    fails = p.filter(pl.col("_fail_date").is_not_null())
    check("no report on or after failure", fails.filter(pl.col("REPDTE") >= pl.col("_fail_date")).is_empty())

    # Independent label: pandas and plain integer arithmetic, sharing no code with the build.
    ev = fail.filter(pl.col("RESTYPE") == "FAILURE", pl.col("CERT").is_not_null(),
                     pl.col("FAILDATE").is_between(SCOPE_START, END_Q)).to_pandas()
    ev["FAILDATE"] = pd.to_datetime(ev["FAILDATE"])
    first = ev.groupby("CERT")["FAILDATE"].min()
    first_q = first.dt.year * 4 + first.dt.quarter - 1
    d = p.select("CERT", "REPDTE", "y").to_pandas()
    d["REPDTE"] = pd.to_datetime(d["REPDTE"])
    d["q"] = d["REPDTE"].dt.year * 4 + d["REPDTE"].dt.quarter - 1
    d["fq"] = d["CERT"].map(first_q)
    gap = d["fq"] - d["q"]
    expected = ((gap >= 1) & (gap <= C.LABEL_HORIZON_Q)).astype("float")
    expected[d["q"] > LAST_MODEL_IDX] = float("nan")
    got = d["y"].astype("float")
    differ = ~((expected == got) | (expected.isna() & got.isna()))
    check("y matches independent recomputation", not differ.any(), f"{int(differ.sum())} differences; {int(expected.sum()):,} positives")

    check("y null exactly after last model quarter",
          p.filter((pl.col("q_idx") > LAST_MODEL_IDX) != pl.col("y").is_null()).is_empty(), C.LAST_MODEL_Q)

    pos = p.filter(pl.col("y") == 1)
    check("only failed banks have y = 1", pos["_exit_class"].unique().to_list() == ["failure"])
    check("at most 4 positive rows per bank", int(pos.group_by("CERT").len()["len"].max()) <= C.LABEL_HORIZON_Q)

    exits = pl.read_parquet(C.DATA_SILVER / "bank_exits.parquet")
    events = failure_events(fail)
    unlabelled = events.join(pos.select("CERT").unique(), on="CERT", how="anti")
    had_report = fin.join(unlabelled, on="CERT").filter(pl.col("REPDTE") < pl.col("_fail_date"))["CERT"].n_unique()
    check("failures without a positive row are only those with no earlier report", had_report == 0,
          f"{events.height:,} events, {pos['CERT'].n_unique():,} labelled, {unlabelled.height} with no report before failure")

    removed = fin.join(p.select("CERT").unique(), on="CERT", how="anti").select("CERT").unique()
    no_capital = fin.filter(pl.col("EQ").is_null()).select("CERT").unique()
    no_record = fin.join(inst.select("CERT"), on="CERT", how="anti").select("CERT").unique()
    hit = fail.filter(pl.col("RESTYPE") == "FAILURE").join(pl.concat([removed, no_capital, no_record]).unique(), on="CERT", how="semi")
    check("rows removed by R1 and R2 belong to banks that never failed", hit.is_empty(), f"{hit.height} failure records hit")

    merged = p.filter(pl.col("_exit_class") == "merger")
    check("mergers are never failures", int(merged["y"].sum()) == 0, f"{merged['CERT'].n_unique():,} merged banks")

    assist_only = fail.filter(pl.col("RESTYPE") == "ASSISTANCE").select("CERT").unique().join(exits.filter(pl.col("_fail_date").is_null()), on="CERT")
    check("assisted banks that never failed are y = 0", int(p.join(assist_only, on="CERT", how="semi")["y"].sum()) == 0)

    last = fails.sort("CERT", "REPDTE").group_by("CERT").last().join(
        fail.filter(pl.col("RESTYPE") == "FAILURE").sort("FAILDATE").group_by("CERT").first().select("CERT", "QBFASSET"), on="CERT")
    within = ((last["ASSET"] / last["QBFASSET"] - 1).abs() < 0.05).mean()
    check("last report before failure agrees with FDIC's QBFASSET", within >= 0.95, f"{within:.1%} within 5%")

    complete = (p.filter(pl.col("INTINC_Q").is_not_null()).group_by("CERT", C.PARTITION_KEY)
                .agg(pl.len().alias("n"), pl.col("NETINC_Q").sum(), pl.col("REPDTE").max().alias("m"))
                .filter(pl.col("n") == 4, pl.col("m").dt.month() == 12)
                .join(p.select("CERT", "REPDTE", "NETINC"), left_on=["CERT", "m"], right_on=["CERT", "REPDTE"]))
    check("four quarterly flows sum to the December year-to-date value",
          complete.filter(pl.col("NETINC_Q") != pl.col("NETINC")).is_empty(), f"{complete.height:,} bank-years")

    bad = sorted(set(safe) & FORBIDDEN) + [c for c in safe if c.startswith("_") or c == "y"]
    check("no label-side or snapshot field in the point-in-time list", not bad, str(bad))

    for cert, name, fdate in KNOWN:
        row = fail.filter(pl.col("CERT") == cert, pl.col("FAILDATE") == fdate)
        rows = p.filter(pl.col("CERT") == cert).sort("REPDTE")
        q = fdate.year * 4 + (fdate.month - 1) // 3
        want = rows.with_columns(((q - pl.col("q_idx")).is_between(1, 4)).alias("w"))
        ok = (row.height == 1 and name in row["NAME"][0].upper() and rows["REPDTE"].max() < fdate
              and want.filter(pl.col("w").cast(pl.Int8) != pl.col("y")).is_empty() and int(rows["y"].sum()) == 4)
        check(f"known case: {name.title()}", ok, f"last report {rows['REPDTE'].max()}, failed {fdate}, y = 1 on {int(rows['y'].sum())} rows")

    # Truncation: the core point-in-time proof.
    keys = ["CERT", "REPDTE"]
    cols = keys + safe
    full = p.select(cols)
    for cut in CUTOFFS:
        early, _ = build_panel(fin, fail, inst, cutoff=cut)
        a = full.filter(pl.col("REPDTE") <= cut).sort(keys)
        b = early.select(cols).with_columns([pl.col(c).cast(a.schema[c]) for c in cols]).sort(keys)
        same_rows = a.height == b.height
        diffs = [c for c in safe if not a[c].equals(b[c])] if same_rows else ["row count differs"]
        check(f"truncation at {cut}: point-in-time columns unchanged", same_rows and not diffs,
              f"{a.height:,} rows, {len(safe)} columns compared; differing: {diffs}")

    # The test must be able to fail: label-side columns are expected to differ.
    cut = CUTOFFS[1]
    early, _ = build_panel(fin, fail, inst, cutoff=cut)
    a = p.filter(pl.col("REPDTE") <= cut).sort(keys)
    b = early.sort(keys)
    fake = a["_exit_class"].to_list() != b["_exit_class"].to_list()
    check("control: truncation does change a future-based column", fake, "_exit_class differs, so the test has power")

    return tests


def main() -> None:
    tests = run()
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(tests).write_csv(C.REPORTS / "p2_checks.csv")
    failed = [t["test"] for t in tests if not t["passed"]]
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
