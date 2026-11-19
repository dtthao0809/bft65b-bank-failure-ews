"""
Part 4a - Layer 2 "silver" -> Layer 3 "gold": the leak-free feature table, joined with the label.

Three rules keep the future out:
  1. Features are built from a panel with only the point-in-time columns of _schema_panel.json. y and
     the _ columns are not there, so no expression can reach them. The label is joined at the very end.
  2. Every expression looks at quarter q or earlier: lags, trailing windows, and statistics across banks
     inside the same quarter. There is no lead, no centred window, no statistic across all time.
  3. Nothing is fitted here. Capping, imputing and scaling learn from data, so they live in
     p4_pipeline.py and see the training period only.

Failures enter one feature, st_fail_4q, and only events dated at or before the report date.

Other choices, each one from a Part 2 finding:
  * ratios are recomputed from levels, because FDIC's NCLNLSR disagrees with its own levels in 1985-96;
  * 4-quarter totals of quarterly flows replace year-to-date figures, which would give seasonality;
  * windows need consecutive quarters, so a reporting gap gives null rather than a wrong total;
  * ratios that explode as a denominator nears zero are bounded in [0, 1] instead. A null there would be
    imputed with the median, and would give a bank with no capital left an average score;
  * size is a rank within the quarter. Log assets grows with inflation, so it would proxy calendar time.

Usage:  python src/p4_features.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from p2_build_panel import failure_events, q_index  # noqa: E402

DEV_END = date.fromisoformat(C.DEV_END)
TEST_START = date.fromisoformat(C.TEST_START)
TEST_END = date.fromisoformat(C.TEST_END)

# name, CAMELS pillar, kind, expected direction with failure risk, formula.
REGISTRY: list[tuple[str, str, str, int, str]] = [
    ("eq_ta",       "C capital",   "num",  -1, "EQ / ASSET"),
    ("tier1_lev",   "C capital",   "num",  -1, "RBC1AAJ / 100, reported tier 1 leverage"),
    ("eq_ta_d4",    "C capital",   "num",  -1, "eq_ta minus eq_ta 4 quarters earlier"),
    ("eq_ta_pct",   "C capital",   "num",  -1, "rank of eq_ta among banks in the quarter, 0-1"),
    ("eq_st_gap",   "C capital",   "num",  -1, "eq_ta minus state median in the quarter"),
    ("f_neg_equity", "C capital",  "flag",  1, "EQ <= 0"),
    ("npl",         "A assets",    "num",   1, "NCLNLS / LNLSGR, noncurrent loans"),
    ("npa_ta",      "A assets",    "num",   1, "(NCLNLS + ORE) / ASSET"),
    ("texas",       "A assets",    "num",   1, "NPA / (NPA + max(EQ + LNATRES, 0)), bounded Texas ratio"),
    ("res_cov",     "A assets",    "num",  -1, "LNATRES / (LNATRES + NCLNLS), bounded reserve coverage"),
    ("res_ln",      "A assets",    "num",   1, "LNATRES / LNLSGR"),
    ("nco_4q",      "A assets",    "num",   1, "4-quarter NTLNLS_Q / LNLSGR, net charge-offs"),
    ("prov_4q",     "A assets",    "num",   1, "4-quarter ELNATR_Q / LNLSGR, provisions"),
    ("npl_d4",      "A assets",    "num",   1, "npl minus npl 4 quarters earlier"),
    ("npl_pct",     "A assets",    "num",   1, "rank of npl among banks in the quarter, 0-1"),
    ("npl_st_gap",  "A assets",    "num",   1, "npl minus state median in the quarter"),
    ("re_ln",       "M loan mix",  "num",   1, "LNRE / LNLSGR, real estate share"),
    ("cons_ta",     "M loan mix",  "num",   1, "LNRECONS / ASSET, construction"),
    ("ci_ln",       "M loan mix",  "num",   0, "LNCI / LNLSGR, commercial and industrial share"),
    ("con_ln",      "M loan mix",  "num",   0, "LNCON / LNLSGR, consumer share"),
    ("ln_ta",       "M loan mix",  "num",   1, "LNLSNET / ASSET"),
    ("ln_g4",       "M loan mix",  "num",   0, "LNLSGR growth over 4 quarters"),
    ("roa_4q",      "E earnings",  "num",  -1, "4-quarter NETINC_Q / ASSET"),
    ("roa_q_d4",    "E earnings",  "num",  -1, "annualised quarterly ROA minus the same, 4 quarters earlier"),
    ("nim_4q",      "E earnings",  "num",  -1, "4-quarter (INTINC_Q - EINTEXP_Q) / ASSET"),
    ("cost_inc",    "E earnings",  "num",   1, "NONIX / (NONIX + max(revenue, 0)), bounded, 4 quarters"),
    ("roa_std8",    "E earnings",  "num",   1, "standard deviation of quarterly ROA, 8 quarters"),
    ("loss_q4",     "E earnings",  "num",   1, "quarters with a loss among the last 4"),
    ("roa_pct",     "E earnings",  "num",  -1, "rank of roa_4q among banks in the quarter, 0-1"),
    ("liq_ta",      "L liquidity", "num",  -1, "(CHBAL + SC) / ASSET"),
    ("ln_dep",      "L liquidity", "num",   1, "LNLSGR / DEP"),
    ("bro_dep",     "L liquidity", "num",   1, "BRO / DEP, brokered deposits"),
    ("nondep_ta",   "L liquidity", "num",   1, "(LIAB - DEP) / ASSET, non-deposit funding"),
    ("dep_g4",      "L liquidity", "num",   0, "DEP growth over 4 quarters"),
    ("size_pct",    "S size",      "num",   0, "rank of ASSET among banks in the quarter, 0-1"),
    ("asset_g4",    "S size",      "num",   0, "ASSET growth over 4 quarters"),
    ("st_npl_med",  "X region",    "num",   1, "state median of npl in the quarter"),
    ("st_loss_share", "X region",  "num",   1, "share of the state's banks with a loss in the quarter"),
    ("st_fail_4q",  "X region",    "num",   1, "state failures in the last 4 quarters / state banks"),
    ("f_short_hist", "Q quality",  "flag",  1, "fewer than 8 consecutive quarters of history"),
    ("f_gap_before", "Q quality",  "flag",  0, "the previous quarter is missing"),
    ("f_identity_break", "Q quality", "flag", 0, "|ASSET - LIAB - EQ| > 1% of ASSET"),
]
FEATURES = [r[0] for r in REGISTRY]
NUMERIC = [r[0] for r in REGISTRY if r[2] == "num"]
FLAGS = [r[0] for r in REGISTRY if r[2] == "flag"]

# What was left out on purpose: the report's answer to "why not X?".
EXCLUDED: list[tuple[str, str]] = [
    ("ACTIVE, CHANGEC1, ENDEFYMD, BKCLASS (Institutions)", "Current snapshot: known only today, so using it would be look-ahead. 52 banks are marked inactive yet report in 2026Q2."),
    ("COST (Failures)", "Known only after the failure."),
    ("y and all _ columns", "Built from the future."),
    ("YEAR, REPDTE, q_idx, CERT", "Would let a model memorise crisis years or individual banks. A test period starts after training, so neither would carry over."),
    ("STNAME as a category", "Would memorise Texas in the 1980s. Regional stress enters through point-in-time state statistics instead."),
    ("Number of quarters in the panel", "The panel starts in 1984, so a bank's history is cut off by the data rather than by its age, and the count rises with calendar time."),
    ("log(ASSET)", "Nominal, so it rises with inflation and proxies calendar time. size_pct replaces it."),
    ("National failure rate", "One value per quarter: with few crises it identifies the period rather than any bank's risk. st_fail_4q varies across states."),
    ("VOLIAB, COREDEP", "Median share jumps in one quarter, 2010Q1 (-12.6 and +10.5 points), which looks like a definition change. bro_dep and nondep_ta cover funding risk."),
    ("P3ASSET, OTHBFHLB", "Null before 2001 for most banks: P3ASSET 100% and OTHBFHLB about 85%."),
    ("RBCRWAJ", "Null before 1990Q1 and for 36-41% of banks from 2020 (CBLR), so training and test would mean different things. tier1_lev and eq_ta cover capital."),
    ("f_cblr", "Can be true only from 2020, so it never varies in training and a model could not learn it."),
    ("P9ASSET, NAASSET, NUMEMP", "Null for about 18-19% of banks before 1990, tied to thrift charters, which dominate the 1989 failures. Null would identify thrifts rather than risk. NCLNLS covers noncurrent loans."),
    ("FDIC ROA, ROE, NIMY, NCLNLSR, NTLNLSR, LNLSDEPR", "Recomputed from levels to one definition across time. ROE is also meaningless with negative equity."),
]


def _guard(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when(den > 0).then(num / den)


def _state_failures(panel: pl.LazyFrame, events: pl.DataFrame) -> pl.LazyFrame:
    """Failures per state in the trailing 4 quarters, q included, per bank in the state at q.

    A failure in calendar quarter q is dated at or before that quarter's report date, so it is known.
    The state of a failed bank is its last report's: all its panel rows precede the failure.
    """
    home = panel.group_by("CERT").agg(pl.col("STNAME").sort_by("REPDTE").last())
    fails = (events.lazy().join(home, on="CERT", how="inner").with_columns(q_index("_fail_date").alias("q_idx"))
             .group_by("STNAME", "q_idx").agg(pl.len().cast(pl.Int32).alias("fails")))
    banks = panel.group_by("STNAME", "q_idx").agg(pl.len().cast(pl.Int32).alias("banks"))
    # A full grid, so a rolling window of 4 rows is always 4 quarters even if a state has a gap.
    grid = (banks.select("STNAME").unique().join(banks.select("q_idx").unique(), how="cross")
            .join(banks, on=["STNAME", "q_idx"], how="left").join(fails, on=["STNAME", "q_idx"], how="left")
            .with_columns(pl.col("fails").fill_null(0)).sort("STNAME", "q_idx"))
    return grid.with_columns(
        (pl.col("fails").rolling_sum(4, min_samples=4).over("STNAME") / pl.col("banks")).alias("st_fail_4q")
    ).select("STNAME", "q_idx", "st_fail_4q")


def build_features(panel: pl.LazyFrame, fail: pl.DataFrame, safe_columns: list[str], extra: dict | None = None) -> pl.LazyFrame:
    """Returns keys + features. `panel` must not need y or _ columns: only safe_columns are selected.

    `extra` adds named expressions. Only p4_checks.py uses it, for controls that are leaky on purpose.
    """
    base = panel.select("CERT", "REPDTE", C.PARTITION_KEY, "q_idx", *safe_columns).sort("CERT", "REPDTE")

    def lag(c: str, n: int) -> pl.Expr:
        return pl.col(c).shift(n).over("CERT")

    def consec(n: int) -> pl.Expr:
        return (pl.col("q_idx") - lag("q_idx", n)) == n

    def sum4(e: pl.Expr) -> pl.Expr:
        return pl.when(consec(3)).then(e.rolling_sum(4, min_samples=4).over("CERT"))

    def growth(c: str) -> pl.Expr:
        return pl.when(consec(4) & (lag(c, 4) > 0)).then(pl.col(c) / lag(c, 4) - 1)

    def q_rank(c: str) -> pl.Expr:
        return pl.col(c).rank("average").over("REPDTE") / pl.col(c).count().over("REPDTE")

    nii_q = pl.col("INTINC_Q") - pl.col("EINTEXP_Q")
    p = (base.with_columns(
            sum4(pl.col("NETINC_Q")).alias("_s_net"), sum4(nii_q).alias("_s_nii"), sum4(pl.col("NONII_Q")).alias("_s_nonii"),
            sum4(pl.col("NONIX_Q")).alias("_s_nonix"), sum4(pl.col("NTLNLS_Q")).alias("_s_nco"), sum4(pl.col("ELNATR_Q")).alias("_s_prov"),
            (pl.col("NCLNLS") + pl.col("ORE")).alias("_npa"), (pl.col("EQ") + pl.col("LNATRES")).clip(lower_bound=0).alias("_cushion"))
         .with_columns(
            _guard(pl.col("EQ"), pl.col("ASSET")).alias("eq_ta"),
            (pl.col("RBC1AAJ") / 100).alias("tier1_lev"),
            (pl.col("EQ") <= 0).fill_null(False).alias("f_neg_equity"),
            _guard(pl.col("NCLNLS"), pl.col("LNLSGR")).alias("npl"),
            _guard(pl.col("_npa"), pl.col("ASSET")).alias("npa_ta"),
            _guard(pl.col("_npa"), pl.col("_npa") + pl.col("_cushion")).alias("texas"),
            _guard(pl.col("LNATRES"), pl.col("LNATRES") + pl.col("NCLNLS")).alias("res_cov"),
            _guard(pl.col("LNATRES"), pl.col("LNLSGR")).alias("res_ln"),
            _guard(pl.col("_s_nco"), pl.col("LNLSGR")).alias("nco_4q"),
            _guard(pl.col("_s_prov"), pl.col("LNLSGR")).alias("prov_4q"),
            _guard(pl.col("LNRE"), pl.col("LNLSGR")).alias("re_ln"),
            _guard(pl.col("LNRECONS"), pl.col("ASSET")).alias("cons_ta"),
            _guard(pl.col("LNCI"), pl.col("LNLSGR")).alias("ci_ln"),
            _guard(pl.col("LNCON"), pl.col("LNLSGR")).alias("con_ln"),
            _guard(pl.col("LNLSNET"), pl.col("ASSET")).alias("ln_ta"),
            growth("LNLSGR").alias("ln_g4"),
            _guard(pl.col("_s_net"), pl.col("ASSET")).alias("roa_4q"),
            _guard(4 * pl.col("NETINC_Q"), pl.col("ASSET")).alias("_roa_q"),
            _guard(pl.col("_s_nii"), pl.col("ASSET")).alias("nim_4q"),
            _guard(pl.col("_s_nonix"), pl.col("_s_nonix") + pl.col("_s_nii").add(pl.col("_s_nonii")).clip(lower_bound=0)).alias("cost_inc"),
            pl.when(consec(3)).then((pl.col("NETINC_Q") < 0).cast(pl.Int8).rolling_sum(4, min_samples=4).over("CERT")).cast(pl.Float64).alias("loss_q4"),
            _guard(pl.col("CHBAL") + pl.col("SC"), pl.col("ASSET")).alias("liq_ta"),
            _guard(pl.col("LNLSGR"), pl.col("DEP")).alias("ln_dep"),
            _guard(pl.col("BRO"), pl.col("DEP")).alias("bro_dep"),
            _guard(pl.col("LIAB") - pl.col("DEP"), pl.col("ASSET")).alias("nondep_ta"),
            growth("DEP").alias("dep_g4"),
            growth("ASSET").alias("asset_g4"),
            consec(7).fill_null(False).not_().alias("f_short_hist"))
         .with_columns(
            pl.when(consec(4)).then(pl.col("eq_ta") - lag("eq_ta", 4)).alias("eq_ta_d4"),
            pl.when(consec(4)).then(pl.col("npl") - lag("npl", 4)).alias("npl_d4"),
            pl.when(consec(4)).then(pl.col("_roa_q") - lag("_roa_q", 4)).alias("roa_q_d4"),
            pl.when(consec(7)).then(pl.col("_roa_q").rolling_std(8, min_samples=8).over("CERT")).alias("roa_std8"),
            q_rank("eq_ta").alias("eq_ta_pct"), q_rank("npl").alias("npl_pct"), q_rank("roa_4q").alias("roa_pct"), q_rank("ASSET").alias("size_pct"),
            pl.col("npl").median().over("REPDTE", "STNAME").alias("st_npl_med"),
            pl.col("eq_ta").median().over("REPDTE", "STNAME").alias("_st_eq_med"),
            (pl.col("_roa_q") < 0).mean().over("REPDTE", "STNAME").alias("st_loss_share"))
         .with_columns((pl.col("npl") - pl.col("st_npl_med")).alias("npl_st_gap"), (pl.col("eq_ta") - pl.col("_st_eq_med")).alias("eq_st_gap")))
    if extra:
        p = p.with_columns(**extra)
    events = failure_events(fail)
    p = p.join(_state_failures(base, events), on=["STNAME", "q_idx"], how="left")
    keep = ["CERT", "REPDTE", C.PARTITION_KEY, "q_idx", *FEATURES, *(extra or {})]
    return p.select(keep).sort("CERT", "REPDTE")


def split_label() -> pl.Expr:
    """dev, gap, test, or unlabelled for the quarters after 2025Q2."""
    d = pl.col("REPDTE")
    return (pl.when(d <= DEV_END).then(pl.lit("dev")).when(d < TEST_START).then(pl.lit("gap"))
            .when(d <= TEST_END).then(pl.lit("test")).otherwise(pl.lit("unlabelled")).alias("_split"))


def scan_silver() -> pl.LazyFrame:
    return pl.scan_parquet(C.DATA_SILVER / "panel" / "**" / "*.parquet", hive_partitioning=True, hive_schema={C.PARTITION_KEY: pl.Int16})


def safe_columns() -> list[str]:
    return json.loads((C.DATA_SILVER / "_schema_panel.json").read_text())["point_in_time_columns"]


def main() -> None:
    fail = pl.read_parquet(C.DATA_BRONZE / "failures.parquet")
    silver = scan_silver()
    feats = build_features(silver, fail, safe_columns()).collect(engine="streaming")
    labels = silver.select("CERT", "REPDTE", "y", "_days_to_fail", "_near_miss", "_other_exit_4q", "_fail_date").collect(engine="streaming")
    gold = feats.join(labels, on=["CERT", "REPDTE"], how="left", validate="1:1").with_columns(split_label())

    tmp, out = C.DATA_GOLD / "features_tmp", C.DATA_GOLD / "features"
    shutil.rmtree(tmp, ignore_errors=True)
    for (year,), part in gold.partition_by(C.PARTITION_KEY, as_dict=True, include_key=False).items():
        (tmp / f"{C.PARTITION_KEY}={year}").mkdir(parents=True)
        part.write_parquet(tmp / f"{C.PARTITION_KEY}={year}" / "part-0.parquet", compression=C.COMPRESSION,
                           compression_level=C.COMPRESSION_LEVEL, statistics=True, row_group_size=C.ROW_GROUP_SIZE)
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)

    # Counts only: the exam asks for rows and failures per period, and nothing else about test is read.
    facts = (gold.group_by("_split").agg(
        pl.len().alias("rows"), pl.col("CERT").n_unique().alias("banks"), pl.col("REPDTE").min().alias("first_q"), pl.col("REPDTE").max().alias("last_q"),
        pl.col("y").sum().alias("y1_rows"), pl.col("CERT").filter(pl.col("y") == 1).n_unique().alias("failed_banks")).sort("first_q"))
    facts.write_csv(C.REPORTS / "p4_split.csv")
    pl.DataFrame([{"feature": r[0], "pillar": r[1], "kind": r[2], "expected_sign": r[3], "formula": r[4]} for r in REGISTRY]).write_csv(C.REPORTS / "p4_feature_dictionary.csv")
    pl.DataFrame([{"left_out": a, "why": b} for a, b in EXCLUDED]).write_csv(C.REPORTS / "p4_excluded.csv")
    with pl.Config(tbl_rows=10, tbl_cols=10, tbl_width_chars=130):
        print(facts)
    print(f"gold: {gold.height:,} rows, {len(FEATURES)} candidate features")


if __name__ == "__main__":
    main()
