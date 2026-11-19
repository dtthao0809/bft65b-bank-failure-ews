"""
Part 5d - The test period, used once.

The script refuses to run unless reports/p5_frozen.json exists and the saved models match its hash, and it
refuses to run again once reports/p5_test_results.json exists. It changes nothing about the models, the
threshold or the features: those were all fixed on development data.

--dry-run    runs the same code on fold F3's validation window, with a model fitted on F3's training window.
             It reads no test row and exists to find bugs before the one real run.
--reproduce  for a fresh clone, where the recorded results already exist. Reruns the evaluation with the
             rebuilt models, writes reports/p5_reproduce_*, and fails unless the numbers equal the recorded
             ones. It cannot overwrite them, and it informs no decision: everything is already fixed.

Confidence intervals resample whole banks, because a bank's rows are not independent.

Usage:  python src/p5_test.py --dry-run
        python src/p5_test.py
        python src/p5_test.py --reproduce
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p4_pipeline as PP  # noqa: E402
import p5_common as M  # noqa: E402
from p5_costs import exam_cost_by_year  # noqa: E402

RESULTS = C.REPORTS / "p5_test_results.json"
LOCK = C.REPORTS / "p5_test_lock.json"
FROZEN = C.REPORTS / "p5_frozen.json"
BOOT = 500


def bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], banks: np.ndarray, top: dict[str, np.ndarray]) -> dict:
    """Percentile intervals for AP, for AP differences against the baselines, and for top-1% recall."""
    rng = np.random.default_rng(C.SEED)
    ids, inv = np.unique(banks, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.searchsorted(inv[order], np.arange(ids.size + 1))
    groups = [order[cuts[i]:cuts[i + 1]] for i in range(ids.size)]
    out: dict[str, list[float]] = {k: [] for k in scores}
    diffs: dict[str, list[float]] = {k: [] for k in scores if k != "chosen"}
    rec: list[float] = []
    for _ in range(BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, ids.size, ids.size)])
        yb = y[idx]
        if yb.sum() == 0:
            continue
        ap = {k: float(average_precision_score(yb, s[idx])) for k, s in scores.items()}
        for k, v in ap.items():
            out[k].append(v)
        for k in diffs:
            diffs[k].append(ap["chosen"] - ap[k])
        rec.append(float(top["chosen"][idx][yb == 1].mean()))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    return {"ap_ci95": {k: q(v) for k, v in out.items()}, "ap_diff_ci95": {k: q(v) for k, v in diffs.items()}, "top_row_recall_ci95": q(rec), "resamples": len(rec), "unit": "bank"}


def differences(a, b, path: str = "") -> list[str]:
    """Paths where two result trees differ. Numbers may differ by 1e-9 relative."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in differences(a.get(k), b.get(k), f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, z) in enumerate(zip(a, b)) for d in differences(x, z, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1) else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a} != {b}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--reproduce", action="store_true")
    a = ap.parse_args()
    if not FROZEN.exists():
        raise SystemExit("No reports/p5_frozen.json: finish development first.")
    frozen = json.loads(FROZEN.read_text())
    path = C.DATA_GOLD / "p5_final_models.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["models_sha256"]:
        raise SystemExit("The saved models do not match the frozen hash.")
    if a.reproduce and not RESULTS.exists():
        raise SystemExit("--reproduce needs the recorded results in reports/p5_test_results.json.")
    if not a.dry_run and not a.reproduce and RESULTS.exists():
        raise SystemExit("The test period has been used, so this evaluation will not run again. On a fresh clone, use --reproduce.")

    if a.dry_run:
        w = C.FOLDS["F3 crisis 2008-11"]
        Xtr, ytr, _ = PP.load_xy(*w["train"], meta=M.META)
        X, y, meta = PP.load_xy(*w["val"], meta=M.META)
        final = {"chosen": M.make_model(frozen["chosen"]).fit(Xtr, ytr),
                 "B1": M.make_model({"kind": "logit", "C": 1.0, "features": ["eq_ta", "npl", "roa_4q"]}).fit(Xtr, ytr),
                 "texas_median": float(np.nanmedian(Xtr["texas"].to_numpy(dtype=float)))}
        years = (2008, 2011)
        prefix = "p5_dryrun"
    else:
        final = joblib.load(path)
        X, y, meta = PP.load_xy(C.TEST_START, C.TEST_END, allow_test=True, meta=M.META)
        years = (2013, 2025)
        prefix = "p5_reproduce" if a.reproduce else "p5_test"

    costs = json.loads((C.REPORTS / "p5_costs.json").read_text())
    fail = pl.read_parquet(C.DATA_BRONZE / "failures.parquet")
    losses = M.loss_table(fail, costs["loss_rate_median"])
    ex = exam_cost_by_year(years)
    exam = dict(zip(ex["year"].to_list(), ex["exam_cost"].to_list()))

    scores = {"chosen": final["chosen"].predict_proba(X)[:, 1], "B1": final["B1"].predict_proba(X)[:, 1], "B0": M.texas_score(X, final["texas_median"])}
    names = {"chosen": frozen["chosen"]["name"], "B1": "B1 logit: capital, NPL, ROA", "B0": "B0 Texas ratio, no fitting"}
    t = frozen["threshold"]
    saved = frozen["share_saved"]

    table, top_flags = [], {}
    for k, s in scores.items():
        d = M.frame(meta, y, s)
        r = {"model": names[k], **M.metrics(d)}
        if k != "B0":
            r.update(M.probability_metrics(y, s))
        table.append(r)
        top_flags[k] = (d.with_columns((pl.col("score").rank("ordinal", descending=True).over("REPDTE") <= (pl.len().over("REPDTE") * C.CAPACITY_SHARE).ceil()).alias("top"))["top"].to_numpy())
    pl.DataFrame(table, infer_schema_length=None).write_csv(C.REPORTS / f"{prefix}_metrics.csv", float_precision=5)

    d = M.frame(meta, y, scores["chosen"])
    at = M.cost_curve(d, losses, exam, saved, thresholds=np.array([t])).row(0, named=True)
    alarmed = d.filter(pl.col("score") >= t)
    no_model = float(d.filter(pl.col("y") == 1).select("CERT").unique().join(losses, on="CERT", how="left")["loss"].fill_null(0).sum())
    hindsight = M.cost_curve(d, losses, exam, saved).sort("total_cost").row(0, named=True)
    positives = d.filter(pl.col("y") == 1)
    any_alarm = positives.filter(pl.col("score") >= t)
    fq = pl.col("_fail_date").dt.year().cast(pl.Int32) * 4 + pl.col("_fail_date").dt.quarter().cast(pl.Int32) - 1
    rq = pl.col("REPDTE").dt.year().cast(pl.Int32) * 4 + pl.col("REPDTE").dt.quarter().cast(pl.Int32) - 1

    # One row per failed bank: what the system would have done.
    named = fail.filter(pl.col("RESTYPE") == "FAILURE").sort("FAILDATE").group_by("CERT", maintain_order=True).first().select("CERT", "NAME", "COST", "QBFASSET")
    per_bank = (positives.with_columns((fq - rq).alias("qtf"), (pl.col("score") >= t).alias("alarm"))
                .group_by("CERT").agg(pl.col("_fail_date").first(), pl.col("score").max().alias("max_score"), pl.col("alarm").any().alias("any_alarm"),
                                      pl.col("qtf").filter("alarm").max().alias("first_alarm_q_before"),
                                      (pl.col("alarm") & (pl.col("_days_to_fail") > C.ACTION_LAG_DAYS)).any().alias("actionable_alarm"),
                                      pl.col("_days_to_fail").min().alias("days_last_report_to_failure"))
                .join(named, on="CERT", how="left").sort("_fail_date", "CERT"))
    per_bank.write_csv(C.REPORTS / f"{prefix}_failed_banks.csv", float_precision=5)

    periods = {}
    for name, (lo, hi) in ({"2013-2019": (2013, 2019), "2020-2025": (2020, 2025)} if not a.dry_run else {"2008-2009": (2008, 2009), "2010-2011": (2010, 2011)}).items():
        q = d.filter(pl.col("REPDTE").dt.year().is_between(lo, hi))
        if q["y"].sum():
            periods[name] = {**M.metrics(q), "alarm_share": float((q["score"] >= t).mean()), "precision_at_threshold": float(q.filter(pl.col("score") >= t)["y"].mean()) if (q["score"] >= t).any() else None}

    res = {"threshold": t, "model": frozen["chosen"]["name"], "rows": d.height, "y1": int(y.sum()), "failed_banks": positives["CERT"].n_unique(),
           "at_threshold": {"alarms": alarmed.height, "alarm_share": alarmed.height / d.height, "banks_alarmed_ever": alarmed["CERT"].n_unique(), "alarms_per_quarter": alarmed.height / d["REPDTE"].n_unique(),
                            "row_precision": float(alarmed["y"].mean()) if alarmed.height else None, "row_recall": float(any_alarm.height / positives.height),
                            "bank_recall_any": float(any_alarm["CERT"].n_unique() / positives["CERT"].n_unique()), "bank_recall_actionable": at["bank_recall"],
                            "false_alarms": int((alarmed["y"] == 0).sum()), "false_alarms_near_miss": int(alarmed.filter(pl.col("y") == 0)["_near_miss"].sum()),
                            "false_alarms_other_exit_within_4q": int(alarmed.filter(pl.col("y") == 0)["_other_exit_4q"].sum()),
                            "median_lead_quarters": float(per_bank["first_alarm_q_before"].median()) if per_bank["first_alarm_q_before"].null_count() < per_bank.height else None},
           "cost_thousand_usd": {"with_system": at["total_cost"], "exam_part": at["exam_cost"], "failure_part": at["failure_cost"], "no_system": no_model, "saving": no_model - at["total_cost"],
                                 "hindsight_best_threshold": hindsight["threshold"], "hindsight_best_cost": hindsight["total_cost"], "hindsight_alarm_share": hindsight["alarm_share"]},
           "periods": periods, "bootstrap": bootstrap(y, scores, meta["CERT"].to_numpy(), top_flags)}

    # Stored, so the figures and any later reading use these numbers without touching the test again.
    meta_out = d.select("CERT", "REPDTE", "y", "_days_to_fail", pl.col("score").alias("chosen")).with_columns(pl.Series("B1", scores["B1"]), pl.Series("B0", scores["B0"]))
    meta_out.write_parquet(C.DATA_GOLD / f"{'p5_dryrun' if a.dry_run else 'p5_test'}_scores.parquet")
    curves = []
    for k, s in scores.items():
        pr, rc, _ = precision_recall_curve(y, s)
        keep = np.unique(np.linspace(0, pr.size - 1, 400).astype(int))
        curves.append(pl.DataFrame({"model": [names[k]] * keep.size, "precision": pr[keep], "recall": rc[keep]}))
    pl.concat(curves).write_csv(C.REPORTS / f"{prefix}_pr_curve.csv", float_precision=5)

    text = json.dumps(res, indent=1, default=str)
    target = RESULTS if prefix == "p5_test" else C.REPORTS / f"{prefix}_results.json"
    target.write_text(text)
    print(text)
    if prefix == "p5_test":
        LOCK.write_text(json.dumps({"threshold": t, "model": frozen["chosen"]["name"], "models_sha256": frozen["models_sha256"],
                                    "results_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(), "evaluated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    with pl.Config(tbl_rows=10, tbl_cols=14, tbl_width_chars=170, fmt_str_lengths=36, float_precision=4):
        print(pl.DataFrame(table, infer_schema_length=None).select("model", "prevalence", "ap", "lift", "roc_auc", "top_precision", "top_row_recall", "top_bank_recall", "top_bank_recall_actionable"))
    if a.reproduce:
        diff = differences(json.loads(RESULTS.read_text()), json.loads(text))
        (C.REPORTS / "p5_reproduce_check.json").write_text(json.dumps({"identical": not diff, "differences": diff}, indent=1))
        print("reproduce:", "identical to the recorded results" if not diff else f"{len(diff)} differences, first: {diff[:5]}")
        sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
