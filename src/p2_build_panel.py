from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

SCOPE_START = date.fromisoformat(C.SCOPE_START)
END_Q = date.fromisoformat(C.END_Q)
LAST_MODEL_Q = date.fromisoformat(C.LAST_MODEL_Q)
H = C.LABEL_HORIZON_Q


def q_index(col: str | pl.Expr) -> pl.Expr:
    """Gap-free quarter counter, so quarter arithmetic never relies on row positions."""
    d = pl.col(col) if isinstance(col, str) else col
    return d.dt.year().cast(pl.Int32) * 4 + d.dt.quarter().cast(pl.Int32) - 1


LAST_MODEL_IDX = LAST_MODEL_Q.year * 4 + (LAST_MODEL_Q.month - 1) // 3


def load_bronze() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    fin = (pl.scan_parquet(C.DATA_BRONZE / "financials" / "**" / "*.parquet", hive_partitioning=True,
                           hive_schema={C.PARTITION_KEY: pl.Int16}).collect())
    return fin, pl.read_parquet(C.DATA_BRONZE / "failures.parquet"), pl.read_parquet(C.DATA_BRONZE / "institutions.parquet")


def failure_events(fail: pl.DataFrame) -> pl.DataFrame:
    """One event per bank: the FIRST in-scope FAILURE.

    The later date of the three banks that fail twice is not modelled: a bank is at risk only
    until its first failure. ASSISTANCE is not a failure: the charter survives (FDIC category 1).
    """
    return (fail.filter(pl.col("RESTYPE") == "FAILURE", pl.col("CERT").is_not_null(),
                        pl.col("FAILDATE").is_between(SCOPE_START, END_Q))
            .group_by("CERT").agg(pl.col("FAILDATE").min().alias("_fail_date")))


def classify_exits(fin: pl.DataFrame, fail: pl.DataFrame, inst: pl.DataFrame) -> pl.DataFrame:
    """One row per bank: how it left the panel. Failure comes from the Failures table only."""
    events = failure_events(fail)
    later = (fail.filter(pl.col("RESTYPE") == "FAILURE", pl.col("FAILDATE") > END_Q)
             .select("CERT").unique().with_columns(pl.lit(True).alias("_later")))
    code_map = pl.DataFrame({"CHANGEC1": list(C.EXIT_CODE_CLASS), "_by_code": list(C.EXIT_CODE_CLASS.values())},
                            schema_overrides={"CHANGEC1": pl.Int32})
    return (fin.group_by("CERT").agg(pl.col("REPDTE").max().alias("_last_report"))
            .join(inst.select("CERT", "ACTIVE", "CHANGEC1"), on="CERT", how="left")
            .join(events, on="CERT", how="left").join(later, on="CERT", how="left")
            .join(code_map, on="CHANGEC1", how="left")
            .with_columns(
                pl.when(pl.col("_fail_date").is_not_null()).then(pl.lit("failure"))
                .when(pl.col("_later").fill_null(False)).then(pl.lit("failure_after_scope"))
                .when(pl.col("_last_report") >= END_Q).then(pl.lit("active_at_end"))
                .when(pl.col("ACTIVE") == 1).then(pl.lit("active_late_report"))
                .when(pl.col("_by_code").is_not_null()).then(pl.col("_by_code"))
                .otherwise(pl.lit("other_or_unknown")).alias("_exit_class"))
            .select("CERT", "_exit_class", "_last_report", "_fail_date", "CHANGEC1", "ACTIVE"))


def build_panel(fin: pl.DataFrame, fail: pl.DataFrame, inst: pl.DataFrame,
                cutoff: date | None = None) -> tuple[pl.DataFrame, list[dict]]:
    """Returns (panel, waterfall). With a cutoff, every table is first cut to what was known then."""
    if cutoff is not None:
        fin = fin.filter(pl.col("REPDTE") <= cutoff)
        fail = fail.filter(pl.col("FAILDATE") <= cutoff)
    exits = classify_exits(fin, fail, inst)
    log: list[dict] = []

    def row_rule(rule: str, what: str, before: pl.DataFrame, after: pl.DataFrame) -> None:
        log.append({"rule": rule, "what": what, "rows_changed": before.height - after.height,
                    "banks_affected": before.join(after.select("CERT", "REPDTE"), on=["CERT", "REPDTE"], how="anti")["CERT"].n_unique(),
                    "rows_left": after.height})

    log.append({"rule": "start", "what": "bronze financials", "rows_changed": 0, "banks_affected": 0, "rows_left": fin.height})
    p = fin.join(inst.select("CERT"), on="CERT", how="semi")
    row_rule("R1", "no Institutions record", fin, p)
    q = p.filter(pl.col("EQ").is_not_null())
    row_rule("R2", "EQ null, no capital reported", p, q)
    q = q.join(failure_events(fail), on="CERT", how="left")
    p = q.filter(pl.col("_fail_date").is_null() | (pl.col("REPDTE") < pl.col("_fail_date")))
    row_rule("R3", "report dated on or after failure", q, p)
    p = p.sort("CERT", "REPDTE").with_columns(q_index("REPDTE").alias("q_idx"))

    def cell_rule(rule: str, what: str, target: str, cond: pl.Expr) -> None:
        nonlocal p
        n = p.select(cond.fill_null(False).sum()).item()
        log.append({"rule": rule, "what": what, "rows_changed": n,
                    "banks_affected": p.filter(cond.fill_null(False))["CERT"].n_unique(), "rows_left": p.height})
        p = p.with_columns(pl.when(cond).then(None).otherwise(pl.col(target)).alias(target))

    # f_cblr must be read before R4 removes the zeros. A CBLR bank reports a leverage ratio but no
    # risk-based ratio, which is itself known at q and says the bank chose the simpler framework.
    p = p.with_columns(((pl.col("RBCRWAJ") == 0) & (pl.col("RBC1AAJ") != 0) & (pl.col("REPDTE") >= date(2020, 1, 1)))
                       .fill_null(False).alias("f_cblr"))
    cell_rule("R4", "RBCRWAJ == 0 means not reported", "RBCRWAJ", pl.col("RBCRWAJ") == 0)
    cell_rule("R4", "RBC1AAJ == 0 means not reported", "RBC1AAJ", pl.col("RBC1AAJ") == 0)
    cell_rule("R5", "LNLSDEPR undefined, DEP == 0", "LNLSDEPR", pl.col("DEP") == 0)
    cell_rule("R5", "NCLNLSR undefined, LNLSGR == 0", "NCLNLSR", pl.col("LNLSGR") == 0)
    cell_rule("R5", "NTLNLSR undefined, LNLSGR == 0", "NTLNLSR", pl.col("LNLSGR") == 0)
    cell_rule("R6", "ROE meaningless, EQ <= 0", "ROE", pl.col("EQ") <= 0)
    n_neg, banks = 0, set()
    for c in C.NONNEG_STOCKS:
        cond = pl.col(c) < 0
        hit = p.filter(cond.fill_null(False))
        n_neg += hit.height
        banks |= set(hit["CERT"].to_list())
        p = p.with_columns(pl.when(cond).then(None).otherwise(pl.col(c)).alias(c))
    log.append({"rule": "R7", "what": "negative balance", "rows_changed": n_neg, "banks_affected": len(banks), "rows_left": p.height})
    cell_rule("R8", "blank STNAME", "STNAME", pl.col("STNAME") == "")

    # R9: quarterly flows. Q1 is the year-to-date value. Later quarters subtract the previous
    # quarter, and stay null when that quarter is missing rather than treating a multi-quarter
    # total as one quarter.
    prev_q = pl.col("q_idx").shift(1).over("CERT")
    consecutive = (pl.col("q_idx") - prev_q) == 1
    is_q1 = pl.col("REPDTE").dt.quarter() == 1
    p = p.with_columns([
        pl.when(is_q1).then(pl.col(c))
        .when(consecutive).then(pl.col(c) - pl.col(c).shift(1).over("CERT"))
        .otherwise(None).alias(f"{c}_Q") for c in C.YTD_FLOWS
    ])

    p = p.with_columns(
        (consecutive.not_() & prev_q.is_not_null()).fill_null(False).alias("f_gap_before"),
        (pl.col("DEP") == 0).fill_null(False).alias("f_no_deposits"),
        (((pl.col("ASSET") - pl.col("LIAB") - pl.col("EQ")).abs() / pl.col("ASSET")) > 0.01).fill_null(False).alias("f_identity_break"),
        (pl.col("INTINC_Q") < 0).fill_null(False).alias("f_ytd_reversal"),
    )

    # Label-side columns.
    assist = (fail.filter(pl.col("RESTYPE") == "ASSISTANCE", pl.col("CERT").is_not_null(),
                          pl.col("FAILDATE").is_between(SCOPE_START, END_Q))
              .select("CERT", q_index("FAILDATE").alias("aq")))
    assisted_rows = (p.select("CERT", "REPDTE", "q_idx").join(assist, on="CERT")
                     .filter((pl.col("aq") - pl.col("q_idx")).is_between(1, H))
                     .select("CERT", "REPDTE").unique().with_columns(pl.lit(True).alias("_assist_4q")))
    fq = q_index("_fail_date")
    p = (p.join(exits.select("CERT", "_exit_class", "_last_report"), on="CERT", how="left")
         .join(assisted_rows, on=["CERT", "REPDTE"], how="left")
         .with_columns(
             pl.when(pl.col("q_idx") > LAST_MODEL_IDX).then(None)
             .otherwise((fq - pl.col("q_idx")).is_between(1, H).fill_null(False).cast(pl.Int8)).alias("y"),
             (fq - pl.col("q_idx")).alias("_quarters_to_fail"),
             (pl.col("_fail_date") - pl.col("REPDTE")).dt.total_days().cast(pl.Int32).alias("_days_to_fail"),
             pl.col("_assist_4q").fill_null(False),
             # A non-failure exit inside the window: y = 0 is right, but the window is not fully observed.
             (pl.col("_exit_class").is_in(["merger", "assisted_merger", "voluntary_closing",
                                           "distress_no_failure_record", "other_or_unknown"])
              & ((q_index("_last_report") - pl.col("q_idx")) < H)).alias("_other_exit_4q"),
             (pl.col("_exit_class").is_in(["assisted_merger", "distress_no_failure_record"])
              & ((q_index("_last_report") - pl.col("q_idx")) < H)).alias("_distress_exit_4q"),
         )
         .with_columns(((pl.col("_assist_4q") | pl.col("_distress_exit_4q")) & (pl.col("y") == 0)).fill_null(False).alias("_near_miss"))
         .drop("_assist_4q", "_distress_exit_4q"))

    first = ["CERT", "REPDTE", C.PARTITION_KEY, "q_idx", "STNAME"]
    meta = [c for c in p.columns if c.startswith("_")]
    middle = [c for c in p.columns if c not in first and c not in meta and c != "y"]
    return p.select(first + middle + ["y"] + meta).sort(C.PARTITION_KEY, "CERT", "REPDTE"), log


def write_silver(panel: pl.DataFrame, exits: pl.DataFrame) -> None:
    C.DATA_SILVER.mkdir(parents=True, exist_ok=True)
    out, tmp = C.DATA_SILVER / "panel", C.DATA_SILVER / "panel_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for (year,), part in panel.partition_by(C.PARTITION_KEY, as_dict=True, include_key=False).items():
        folder = tmp / f"{C.PARTITION_KEY}={year}"
        folder.mkdir(parents=True)
        part.write_parquet(folder / "part-0.parquet", compression=C.COMPRESSION, compression_level=C.COMPRESSION_LEVEL,
                           statistics=True, row_group_size=C.ROW_GROUP_SIZE)
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)
    exits.sort("CERT").write_parquet(C.DATA_SILVER / "bank_exits.parquet", compression=C.COMPRESSION, compression_level=C.COMPRESSION_LEVEL)
    ids = {"CERT", "REPDTE", C.PARTITION_KEY, "q_idx", "y"}
    (C.DATA_SILVER / "_schema_panel.json").write_text(json.dumps({
        "point_in_time_columns": [c for c in panel.columns if c not in ids and not c.startswith("_")],
        "keys": ["CERT", "REPDTE", C.PARTITION_KEY, "q_idx"], "label": "y",
        "label_side_never_features": [c for c in panel.columns if c.startswith("_")],
        "types": {c: str(t) for c, t in panel.schema.items()}}, indent=1))


def main() -> None:
    fin, fail, inst = load_bronze()
    panel, log = build_panel(fin, fail, inst)
    write_silver(panel, classify_exits(fin, fail, inst))
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    wf = pl.DataFrame(log)
    wf.write_csv(C.REPORTS / "p2_waterfall.csv")
    with pl.Config(tbl_rows=30, tbl_width_chars=120, fmt_str_lengths=45):
        print(wf)
    scope = panel.filter(pl.col("y").is_not_null())
    print(f"silver: {panel.height:,} rows, {panel['CERT'].n_unique():,} banks; model scope {scope.height:,} rows, "
          f"y=1 rows {int(scope['y'].sum()):,}, failed banks labelled {scope.filter(pl.col('y') == 1)['CERT'].n_unique():,}")


if __name__ == "__main__":
    main()
