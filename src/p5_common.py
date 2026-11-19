"""
Part 5 - Shared pieces: the models, the metrics and the cost model. Nothing here reads the test period.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")   # enough for these fits, and leaves the laptop usable

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p4_features as F  # noqa: E402
import p4_pipeline as PP  # noqa: E402

META = ("_days_to_fail", "_near_miss", "_other_exit_4q", "_fail_date")
STATE_FEATURES = ["st_npl_med", "st_loss_share", "st_fail_4q"]
SIGN = {r[0]: r[3] for r in F.REGISTRY}


def make_model(spec: dict) -> Pipeline:
    """spec: kind (logit, hgb), features (list or None for all selected), and hyperparameters."""
    sc = PP.schema()
    use = spec.get("features") or (sc["numeric"] + sc["flags"])
    numeric = [f for f in sc["numeric"] if f in use]
    flags = [f for f in sc["flags"] if f in use]
    if spec["kind"] == "logit":
        clf = LogisticRegression(C=spec["C"], max_iter=1000)
        return Pipeline([("pre", PP.build_preprocessor(numeric, flags, scale=True)), ("clf", clf)])
    cst = [SIGN[f] for f in numeric + flags] if spec.get("monotone") else None
    clf = HistGradientBoostingClassifier(learning_rate=spec["lr"], max_leaf_nodes=spec["leaves"], max_iter=spec["iters"], min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=False, random_state=C.SEED, monotonic_cst=cst)
    return Pipeline([("pre", PP.build_preprocessor(numeric, flags, scale=False)), ("clf", clf)])


def texas_score(X, med: float) -> np.ndarray:
    """The rule-based baseline needs no fit except a training median for the few null values."""
    v = X["texas"].to_numpy(dtype=float)
    return np.where(np.isnan(v), med, v)


def frame(meta: pl.DataFrame, y: np.ndarray, score: np.ndarray) -> pl.DataFrame:
    return meta.with_columns(pl.Series("y", y), pl.Series("score", score))


def metrics(d: pl.DataFrame, share: float = C.CAPACITY_SHARE) -> dict:
    """Threshold-free measures, plus the policy of reviewing the riskiest share of banks per quarter."""
    y, s = d["y"].to_numpy(), d["score"].to_numpy()
    r = d.with_columns((pl.col("score").rank("ordinal", descending=True).over("REPDTE") <= (pl.len().over("REPDTE") * share).ceil()).alias("top"))
    top = r.filter("top")
    failed = r.filter(pl.col("y") == 1)
    act = failed.filter(pl.col("_days_to_fail") > C.ACTION_LAG_DAYS)
    banks = failed["CERT"].n_unique()
    return {"rows": d.height, "y1": int(y.sum()), "failed_banks": banks, "prevalence": float(y.mean()),
            "ap": float(average_precision_score(y, s)), "lift": float(average_precision_score(y, s) / y.mean()),
            "roc_auc": float(roc_auc_score(y, s)), "top_precision": float(top["y"].mean()), "top_row_recall": float(top["y"].sum() / y.sum()),
            "top_bank_recall": top.filter(pl.col("y") == 1)["CERT"].n_unique() / banks,
            "top_bank_recall_actionable": act.filter("top")["CERT"].n_unique() / banks}


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "mean_pred": float(p.mean()), "observed": float(y.mean())}


def loss_table(fail: pl.DataFrame, loss_rate: float) -> pl.DataFrame:
    """Loss per failed bank in thousand USD: FDIC's COST for its first failure, else rate x assets."""
    first = (fail.filter(pl.col("RESTYPE") == "FAILURE", pl.col("CERT").is_not_null()).sort("FAILDATE").group_by("CERT", maintain_order=True).first()
             .select("CERT", "COST", "QBFASSET"))
    return first.with_columns(pl.coalesce("COST", pl.col("QBFASSET") * loss_rate).clip(lower_bound=0).alias("loss")).select("CERT", "loss")


def cost_curve(d: pl.DataFrame, losses: pl.DataFrame, exam_by_year: dict[int, float], saved: float, n: int = 400,
               lag_days: int = C.ACTION_LAG_DAYS, thresholds: np.ndarray | None = None) -> pl.DataFrame:
    """Total cost at many score thresholds, in thousand USD.

    cost(t) = exam cost x alarms + sum over failed banks of loss x (1 - saved x caught)
    A failed bank counts once, however many of its reports raise an alarm. It is caught only by an alarm
    on a report filed in time to act. Every alarm costs an exam: a supervisor might not re-examine each
    quarter, so this is on the expensive side.
    """
    w = d.with_columns(pl.col("REPDTE").dt.year().alias("_yr")).with_columns(pl.col("_yr").replace_strict(exam_by_year, return_dtype=pl.Float64).alias("_c"))
    s, c = w["score"].to_numpy(), w["_c"].to_numpy()
    order = np.argsort(-s)
    s_sorted, c_cum = s[order], np.cumsum(c[order])
    pos = w.filter(pl.col("y") == 1)
    bank = (pos.group_by("CERT").agg(pl.col("score").filter(pl.col("_days_to_fail") > lag_days).max().alias("best")).join(losses, on="CERT", how="left")
            .with_columns(pl.col("loss").fill_null(0.0)))
    best, loss = bank["best"].fill_null(-np.inf).to_numpy(), bank["loss"].to_numpy()
    if thresholds is None:
        qs = np.unique(np.concatenate([np.linspace(0, 1, n // 2), 1 - np.logspace(-5, 0, n // 2)]))
        thresholds = np.unique(np.quantile(s, qs))
    rows = []
    for t in thresholds:
        k = int(np.searchsorted(-s_sorted, -t, side="right"))
        caught = best >= t
        exam = float(c_cum[k - 1]) if k else 0.0
        rows.append({"threshold": float(t), "alarms": k, "alarm_share": k / s.size, "exam_cost": exam, "failure_cost": float((loss * (1 - saved * caught)).sum()),
                     "total_cost": exam + float((loss * (1 - saved * caught)).sum()), "banks_caught": int(caught.sum()), "bank_recall": float(caught.mean()) if caught.size else 0.0})
    return pl.DataFrame(rows).sort("threshold")
