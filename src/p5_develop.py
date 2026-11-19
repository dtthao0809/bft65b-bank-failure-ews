from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p4_pipeline as PP  # noqa: E402
import p5_common as M  # noqa: E402
from p5_costs import exam_cost_by_year  # noqa: E402

BASE3 = ["eq_ta", "npl", "roa_4q"]
LOGIT = [{"name": f"logit C={c}", "kind": "logit", "C": c} for c in (0.01, 0.1, 1.0)]
HGB = [{"name": f"gbm lr={lr} leaves={lv} trees={it}", "kind": "hgb", "lr": lr, "leaves": lv, "iters": it}
       for lr, lv, it in ((0.05, 7, 300), (0.05, 15, 200), (0.05, 31, 200), (0.1, 7, 200), (0.1, 15, 100), (0.03, 15, 400))]


def cv(spec: dict, data: dict, keep_scores: bool = False, row_filter=None, folds=None) -> tuple[list[dict], list[pl.DataFrame]]:
    """Fit on each fold's training window, score its validation window."""
    rows, oof = [], []
    for fold in (folds or C.FOLDS):
        Xtr, ytr, mtr, Xva, yva, mva = data[fold]
        t0 = time.perf_counter()
        if spec["kind"] == "texas":
            med = float(np.nanmedian(Xtr["texas"].to_numpy(dtype=float)))
            score, prob = M.texas_score(Xva, med), None
        else:
            keep = row_filter(mtr, Xtr) if row_filter else np.ones(len(ytr), dtype=bool)
            model = M.make_model(spec).fit(Xtr[keep], ytr[keep])
            prob = model.predict_proba(Xva)[:, 1]
            score = prob
        d = M.frame(mva, yva, score)
        r = {"model": spec["name"], "fold": fold, "fit_s": round(time.perf_counter() - t0, 1), **M.metrics(d)}
        if prob is not None:
            r.update(M.probability_metrics(yva, prob))
        rows.append(r)
        print(f"{spec['name']:44s} {fold:20s} AP {r['ap']:.4f} lift {r['lift']:6.1f} AUC {r['roc_auc']:.4f} top1%rec {r['top_bank_recall']:.3f} {r['fit_s']}s", flush=True)
        if keep_scores:
            oof.append(d.with_columns(pl.lit(fold).alias("fold")))
    return rows, oof


def main() -> None:
    sc = PP.schema()
    data = {}
    for fold, w in C.FOLDS.items():
        Xtr, ytr, mtr = PP.load_xy(*w["train"], meta=M.META)
        Xva, yva, mva = PP.load_xy(*w["val"], meta=M.META)
        data[fold] = (Xtr, ytr, mtr, Xva, yva, mva)

    results: list[dict] = []
    # Baselines.
    for spec in ({"name": "B0 Texas ratio, no fitting", "kind": "texas"}, {"name": "B1 logit: capital, NPL, ROA", "kind": "logit", "C": 1.0, "features": BASE3}):
        results += cv(spec, data)[0]
    # Tuning.
    for spec in LOGIT + HGB:
        results += cv(spec, data)[0]

    def mean_ap(name: str) -> float:
        return float(np.mean([r["ap"] for r in results if r["model"] == name]))

    best_logit = max(LOGIT, key=lambda s: mean_ap(s["name"]))
    best_gbm = max(HGB, key=lambda s: mean_ap(s["name"]))

    # Variants, including the checks Parts 2 and 4 asked for.
    no_state = [f for f in sc["numeric"] + sc["flags"] if f not in M.STATE_FEATURES]
    variants = [
        {**best_gbm, "name": "gbm monotone", "monotone": True},
        {**best_gbm, "name": "gbm no state features", "features": no_state},
        {**best_gbm, "name": "gbm monotone, no state", "monotone": True, "features": no_state},
        {**best_logit, "name": "logit no state features", "features": no_state},
    ]
    for spec in variants:
        results += cv(spec, data)[0]
    # Sensitivities: reported, never candidates.
    near = {**best_gbm, "name": "sens: gbm, near-miss rows out of training"}
    results += cv(near, data, row_filter=lambda m, X: ~m["_near_miss"].to_numpy())[0]
    late = {**best_gbm, "name": "sens: gbm, training starts 1990"}
    results += cv(late, data, row_filter=lambda m, X: (m["REPDTE"].dt.year() >= 1990).to_numpy(), folds=["F2 quiet 1994-2006", "F3 crisis 2008-11"])[0]

    tab = pl.DataFrame(results, infer_schema_length=None)
    tab.write_csv(C.REPORTS / "p5_cv_results.csv", float_precision=5)

    # Selection.
    candidates = [s for s in LOGIT + HGB + variants]
    best_ap = max(mean_ap(s["name"]) for s in candidates)
    simple_order = ["logit no state features", best_logit["name"], "gbm monotone, no state", "gbm monotone", "gbm no state features"]
    spec_by_name = {s["name"]: s for s in candidates}
    chosen_name = next((n for n in simple_order if mean_ap(n) >= 0.99 * best_ap),
                       max(candidates, key=lambda s: mean_ap(s["name"]))["name"])
    chosen = spec_by_name[chosen_name]
    print("best mean AP", round(best_ap, 4), "chosen", chosen_name, round(mean_ap(chosen_name), 4), flush=True)

    # Out-of-fold scores for the chosen model and baselines.
    _, oof = cv(chosen, data, keep_scores=True)
    d = pl.concat(oof)
    d.select("CERT", "REPDTE", "y", "score", "fold", "_days_to_fail", "_near_miss").write_parquet(C.DATA_GOLD / "p5_oof.parquet")

    # Threshold.
    costs = json.loads((C.REPORTS / "p5_costs.json").read_text())
    fail = pl.read_parquet(C.DATA_BRONZE / "failures.parquet")
    losses = M.loss_table(fail, costs["loss_rate_median"])
    ex = exam_cost_by_year((1986, 2011))
    exam = dict(zip(ex["year"].to_list(), ex["exam_cost"].to_list()))
    saved = costs["share_saved_estimate"]
    curve = M.cost_curve(d, losses, exam, saved)
    curve.write_csv(C.REPORTS / "p5_cost_curve_dev.csv", float_precision=6)
    best = curve.sort("total_cost").row(0, named=True)
    threshold = best["threshold"]
    never = float(curve.sort("threshold", descending=True)["failure_cost"][0])

    # How firm is the threshold? Across folds, and across the assumptions.
    by_fold = []
    for fold in C.FOLDS:
        q = d.filter(pl.col("fold") == fold)
        cf = M.cost_curve(q, losses, exam, saved)
        at = M.cost_curve(q, losses, exam, saved, thresholds=np.array([threshold])).row(0, named=True)
        own = cf.sort("total_cost").row(0, named=True)
        by_fold.append({"fold": fold, "own_best_threshold": own["threshold"], "own_best_cost": own["total_cost"], "cost_at_frozen": at["total_cost"],
                        "no_model_cost": q.filter(pl.col("y") == 1).select("CERT").unique().join(losses, on="CERT", how="left")["loss"].fill_null(0).sum(),
                        "alarm_share_at_frozen": at["alarm_share"], "bank_recall_at_frozen": at["bank_recall"],
                        "precision_at_frozen": q.filter(pl.col("score") >= threshold)["y"].mean()})
    pl.DataFrame(by_fold).write_csv(C.REPORTS / "p5_threshold_by_fold.csv", float_precision=6)
    grid = []
    for sv in costs["share_saved_range"]:
        for mult in costs["exam_multipliers"]:
            c2 = M.cost_curve(d, losses, {k: v * mult for k, v in exam.items()}, sv).sort("total_cost").row(0, named=True)
            grid.append({"share_saved": sv, "exam_cost_multiplier": mult, "threshold": c2["threshold"], "alarm_share": c2["alarm_share"], "bank_recall": c2["bank_recall"], "total_cost": c2["total_cost"]})
    pl.DataFrame(grid).write_csv(C.REPORTS / "p5_threshold_sensitivity.csv", float_precision=6)
    lag = []
    for days in (0, 30, 45, 60, 90):
        c3 = M.cost_curve(d, losses, exam, saved, thresholds=np.array([threshold]), lag_days=days).row(0, named=True)
        lag.append({"action_lag_days": days, "bank_recall": c3["bank_recall"], "total_cost": c3["total_cost"]})
    pl.DataFrame(lag).write_csv(C.REPORTS / "p5_action_lag_dev.csv", float_precision=6)

    # Calibration out of fold: predicted against observed by decile of risk within each fold.
    cal = (d.with_columns(pl.col("score").rank("ordinal").over("fold").alias("_r"), pl.len().over("fold").alias("_n"))
           .with_columns(((pl.col("_r") - 1) * 10 // pl.col("_n")).alias("decile")).group_by("fold", "decile").agg(pl.col("score").mean().alias("predicted"), pl.col("y").mean().alias("observed"), pl.len().alias("rows")).sort("fold", "decile"))
    cal.write_csv(C.REPORTS / "p5_calibration_dev.csv", float_precision=6)

    # Importance on the clean fold: how much AP falls when a feature is shuffled.
    Xtr, ytr, _, Xva, yva, _ = data["F3 crisis 2008-11"]
    if chosen["kind"] != "texas":
        model = M.make_model(chosen).fit(Xtr, ytr)
        base = float(M.average_precision_score(yva, model.predict_proba(Xva)[:, 1]))
        rng = np.random.default_rng(C.SEED)
        use = chosen.get("features") or (sc["numeric"] + sc["flags"])
        imp = []
        for f in use:
            drops = []
            saved_col = Xva[f].copy()
            for _ in range(3):
                Xva[f] = rng.permutation(saved_col.to_numpy())
                drops.append(base - float(M.average_precision_score(yva, model.predict_proba(Xva)[:, 1])))
            Xva[f] = saved_col
            imp.append({"feature": f, "ap_drop": float(np.mean(drops)), "ap_drop_sd": float(np.std(drops))})
        pl.DataFrame(imp).sort("ap_drop", descending=True).write_csv(C.REPORTS / "p5_importance_f3.csv", float_precision=5)

    # Fit the frozen models on all of development and record them before any test row is read.
    Xall, yall, _ = PP.load_xy(*C.FINAL_TRAIN, meta=M.META)
    final = {"chosen": M.make_model(chosen).fit(Xall, yall),
             "B1": M.make_model({"kind": "logit", "C": 1.0, "features": BASE3}).fit(Xall, yall),
             "texas_median": float(np.nanmedian(Xall["texas"].to_numpy(dtype=float)))}
    path = C.DATA_GOLD / "p5_final_models.joblib"
    joblib.dump(final, path)
    frozen = {"chosen": {k: v for k, v in chosen.items()}, "threshold": threshold, "capacity_share": C.CAPACITY_SHARE, "action_lag_days": C.ACTION_LAG_DAYS,
              "share_saved": saved, "train_window": list(C.FINAL_TRAIN), "train_rows": len(yall), "train_y1": int(yall.sum()),
              "models_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "selection_mean_ap": mean_ap(chosen_name), "best_candidate_mean_ap": best_ap,
              "dev_oof": {"alarm_share": best["alarm_share"], "bank_recall": best["bank_recall"], "total_cost_thousand": best["total_cost"], "no_model_cost_thousand": never}}
    (C.REPORTS / "p5_frozen.json").write_text(json.dumps(frozen, indent=1))
    print(json.dumps(frozen, indent=1))
    with pl.Config(tbl_rows=40, tbl_cols=9, tbl_width_chars=150, fmt_str_lengths=44, float_precision=4):
        print(tab.group_by("model").agg(pl.len().alias("folds"), pl.col("ap").mean().alias("mean_ap"), pl.col("ap").last().alias("ap_F3"), pl.col("roc_auc").mean().alias("auc"), pl.col("top_bank_recall").mean().alias("top1_rec"), pl.col("brier").mean().alias("brier")).sort("mean_ap", descending=True))
        print(pl.DataFrame(by_fold))
        print(pl.DataFrame(grid))


if __name__ == "__main__":
    main()
