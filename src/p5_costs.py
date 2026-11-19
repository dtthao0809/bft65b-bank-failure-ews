from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

DEV_LABEL_END = date(2012, 12, 31)     # the last failure a development label can see


def exam_cost_by_year(years: tuple[int, int]) -> pl.DataFrame:
    """Thousand USD per exam. December NONIX is the full year, and NUMEMP is the same report's headcount."""
    p = pl.scan_parquet(C.DATA_SILVER / "panel" / "**" / "*.parquet", hive_partitioning=True, hive_schema={C.PARTITION_KEY: pl.Int16})
    return (p.filter(pl.col(C.PARTITION_KEY).is_between(*years), pl.col("REPDTE").dt.month() == 12, pl.col("NUMEMP") > 0, pl.col("NONIX") > 0)
            .group_by(C.PARTITION_KEY).agg((pl.col("NONIX") / pl.col("NUMEMP")).median().alias("nonix_per_employee"))
            .with_columns((pl.col("nonix_per_employee") / 52 * C.EXAM_PERSON_WEEKS).alias("exam_cost"))
            .rename({C.PARTITION_KEY: "year"}).sort("year").collect(engine="streaming"))


def main() -> None:
    fail = pl.read_parquet(C.DATA_BRONZE / "failures.parquet")
    ev = (fail.filter(pl.col("RESTYPE") == "FAILURE", pl.col("CERT").is_not_null(), pl.col("FAILDATE").is_between(date(1984, 1, 1), DEV_LABEL_END))
          .sort("FAILDATE").group_by("CERT", maintain_order=True).first())
    known = ev.filter(pl.col("COST").is_not_null(), pl.col("QBFASSET") > 0).with_columns((pl.col("COST") / pl.col("QBFASSET")).alias("loss_rate"))
    rate = float(known["loss_rate"].median())

    # Capital at the bank's last report before failing, from development rows.
    gold = pl.scan_parquet(C.DATA_GOLD / "features" / "**" / "*.parquet", hive_partitioning=True, hive_schema={C.PARTITION_KEY: pl.Int16})
    last = (gold.filter(pl.col("_split").is_in(["dev", "gap"]), pl.col("_fail_date").is_not_null()).group_by("CERT")
            .agg(pl.col("eq_ta").sort_by("REPDTE").last()).collect(engine="streaming"))
    j = known.join(last, on="CERT").with_columns(
        pl.when(pl.col("eq_ta") <= 0).then(pl.lit("1 insolvent (EQ/TA <= 0)")).when(pl.col("eq_ta") < 0.02).then(pl.lit("2 under 2%"))
        .when(pl.col("eq_ta") < 0.04).then(pl.lit("3 2-4%")).otherwise(pl.lit("4 4% or more")).alias("capital"))
    by_cap = (j.group_by("capital").agg(pl.len().alias("banks"), pl.col("loss_rate").median().alias("median_loss_rate"), pl.col("COST").median().alias("median_cost_thousand")).sort("capital"))
    by_cap.write_csv(C.REPORTS / "p5_loss_by_capital.csv", float_precision=4)
    r = dict(zip(by_cap["capital"], by_cap["median_loss_rate"]))
    worst, better = r["1 insolvent (EQ/TA <= 0)"], r["3 2-4%"]
    saved = round(1 - better / worst, 2)

    ex = exam_cost_by_year((1986, DEV_LABEL_END.year - 1))
    ex.write_csv(C.REPORTS / "p5_exam_cost_by_year.csv", float_precision=3)
    out = {"loss_rate_median": rate, "loss_rate_failures": known.height, "cost_null_share": float(ev["COST"].is_null().mean()),
           "median_loss_thousand": float(known["COST"].median()), "mean_loss_thousand": float(known["COST"].mean()),
           "share_saved_estimate": saved, "share_saved_range": [0.10, saved, 0.50], "exam_person_weeks": C.EXAM_PERSON_WEEKS,
           "median_exam_cost_thousand": float(ex["exam_cost"].median()),
           "exam_multipliers": [0.5, 1, 2, 4], "ratio_median_loss_to_exam": float(known["COST"].median() / ex["exam_cost"].median()),
           "action_lag_days": C.ACTION_LAG_DAYS}
    (C.REPORTS / "p5_costs.json").write_text(json.dumps(out, indent=1))
    with pl.Config(tbl_rows=10, float_precision=3):
        print(by_cap)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
