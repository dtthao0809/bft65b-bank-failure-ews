from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p5_common as M  # noqa: E402
from p5_costs import exam_cost_by_year  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#57606a", "#d8dee4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def style(ax, axis: str = "y") -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    R = C.REPORTS
    frozen = json.loads((R / "p5_frozen.json").read_text())
    res = json.loads((R / "p5_test_results.json").read_text())
    t = frozen["threshold"]

    # Model comparison across folds.
    cv = pl.read_csv(R / "p5_cv_results.csv", infer_schema_length=None)
    best_logit = cv.filter(pl.col("model").str.starts_with("logit C=")).group_by("model").agg(pl.col("ap").mean()).sort("ap")["model"][-1]
    best_gbm = cv.filter(pl.col("model").str.starts_with("gbm lr=")).group_by("model").agg(pl.col("ap").mean()).sort("ap")["model"][-1]
    show = [("B0 Texas ratio, no fitting", "B0 Texas ratio"), ("B1 logit: capital, NPL, ROA", "B1 logit, 3 ratios"), (best_logit, "Logit, all features"),
            (best_gbm, "GBM, with state features"), (frozen["chosen"]["name"], "GBM, no state (chosen)")]
    summary = []
    for m, lab in show:
        q = cv.filter(pl.col("model") == m)
        summary.append({"model": lab, "source": m, **{f["fold"]: f["ap"] for f in q.iter_rows(named=True)}, "mean_ap": q["ap"].mean()})
    pl.DataFrame(summary).write_csv(R / "p5_cv_summary.csv", float_precision=4)
    folds = list(C.FOLDS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    ys = list(range(len(show)))[::-1]
    for ax, fold in zip(axes, folds):
        vals = [cv.filter(pl.col("model") == m, pl.col("fold") == fold)["ap"][0] for m, _ in show]
        prev = cv.filter(pl.col("fold") == fold)["prevalence"][0]
        ax.barh(ys, vals, height=0.62, color=[ORANGE if lab.endswith("(chosen)") else BLUE for _, lab in show])
        for y, v in zip(ys, vals):
            ax.text(v + max(vals) * 0.02, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, max(vals) * 1.2)
        ax.set_title(f"{fold}\nbase rate {prev * 100:.2f}%", loc="left", fontsize=10, color=INK)
        style(ax, "x")
    axes[0].set_yticks(ys, [lab for _, lab in show], color=INK)
    fig.tight_layout()
    fig.savefig(R / "fig_p5_cv.png", dpi=180)
    plt.close(fig)

    # The threshold on development.
    curve = pl.read_csv(R / "p5_cost_curve_dev.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    style(ax)
    ax.plot(curve["alarm_share"] * 100, curve["total_cost"] / 1e6, color=BLUE, linewidth=2)
    best = curve.sort("total_cost").row(0, named=True)
    ax.scatter([best["alarm_share"] * 100], [best["total_cost"] / 1e6], color=ORANGE, s=60, zorder=3, edgecolor="white")
    ax.annotate(f"chosen: alarm on {best['alarm_share'] * 100:.1f}% of reports\nscore >= {best['threshold']:.4f}, catches {best['bank_recall'] * 100:.0f}% of failed banks in time",
                (best["alarm_share"] * 100, best["total_cost"] / 1e6), xytext=(0.28, 0.62), textcoords="axes fraction", fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel("Share of bank-quarters raising an alarm (%, log scale)", fontsize=9, color=MUTED)
    ax.set_ylabel("Total cost, USD bn", fontsize=9, color=MUTED)
    ax.set_title("Development, out of fold: the threshold that minimises total cost", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(R / "fig_p5_threshold.png", dpi=180)
    plt.close(fig)

    # Test precision-recall.
    met = pl.read_csv(R / "p5_test_metrics.csv", infer_schema_length=None)
    stored = pl.read_parquet(C.DATA_GOLD / "p5_test_scores.parquet")
    y_test = stored["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    style(ax)
    curves = []
    for m, col, key in zip(met["model"], (ORANGE, BLUE, AQUA), ("chosen", "B1", "B0")):
        # One point per recall step, so the sparse high-precision end is drawn as it is.
        prec, rec, _ = precision_recall_curve(y_test, stored[key].to_numpy())
        keep = np.r_[True, rec[1:] != rec[:-1]]
        ap = met.filter(pl.col("model") == m)["ap"][0]
        ax.step(rec[keep], prec[keep], where="post", color=col, linewidth=1.8, label=f"{m}, AP {ap:.3f}")
        curves.append(pl.DataFrame({"model": [m] * int(keep.sum()), "precision": prec[keep], "recall": rec[keep]}))
    pl.concat(curves).write_csv(R / "p5_test_pr_curve.csv", float_precision=5)
    ax.axhline(res["y1"] / res["rows"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.02, res["y1"] / res["rows"] + 0.015, f"base rate {res['y1'] / res['rows'] * 100:.2f}%", fontsize=9, color=MUTED)
    ax.set_xlabel("Recall (y = 1 rows)", fontsize=9, color=MUTED)
    ax.set_ylabel("Precision", fontsize=9, color=MUTED)
    ax.set_title("Test period 2013Q1-2025Q2", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(R / "fig_p5_test_pr.png", dpi=180)
    plt.close(fig)

    # What was caught, by date and size of loss.
    b = pl.read_csv(R / "p5_test_failed_banks.csv", try_parse_dates=True)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    style(ax)
    for flag, col, lab in ((True, BLUE, "alarm in time to act"), (False, ORANGE, "no alarm in time")):
        q = b.filter(pl.col("actionable_alarm") == flag)
        ax.scatter(q["_fail_date"].to_list(), np.maximum(q["COST"].fill_null(0).to_numpy() / 1e3, 0.5), s=46, color=col, edgecolor="white", linewidth=0.8, label=f"{lab} ({q.height})")
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_ylabel("FDIC loss, USD million (log)", fontsize=9, color=MUTED)
    for name, text, dy in (("SILICON VALLEY BANK", "Silicon Valley Bank", 1.5), ("FIRST REPUBLIC BANK", "First Republic", 0.55)):
        r = b.filter(pl.col("NAME") == name).row(0, named=True)
        ax.annotate(text, (r["_fail_date"], r["COST"] / 1e3), xytext=(-8, 0), textcoords="offset points", ha="right", va="center" if dy > 1 else "top", fontsize=9, color=INK)
    ax.set_title("Each failed bank in the test period: the largest losses are the ones the ratios did not show", loc="left", fontsize=10, color=INK)
    ax.text(0.42, 0.02, "Losses below USD 0.5m, and zero, are drawn at 0.5", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.32, 1.0))
    fig.tight_layout()
    fig.savefig(R / "fig_p5_test_banks.png", dpi=180)
    plt.close(fig)

    # Post hoc, descriptive: how much of the test result is two banks?
    costs = json.loads((R / "p5_costs.json").read_text())
    losses = M.loss_table(pl.read_parquet(C.DATA_BRONZE / "failures.parquet"), costs["loss_rate_median"])
    ex = exam_cost_by_year((2013, 2025))
    exam = dict(zip(ex["year"].to_list(), ex["exam_cost"].to_list()))
    d = pl.read_parquet(C.DATA_GOLD / "p5_test_scores.parquet").rename({"chosen": "score"})
    big = b.filter(pl.col("NAME").is_in(["SILICON VALLEY BANK", "FIRST REPUBLIC BANK"]))["CERT"].to_list()
    rest = d.filter(~pl.col("CERT").is_in(big))
    no_model = float(rest.filter(pl.col("y") == 1).select("CERT").unique().join(losses, on="CERT", how="left")["loss"].fill_null(0).sum())
    at = M.cost_curve(rest, losses, exam, frozen["share_saved"], thresholds=np.array([t])).row(0, named=True)
    total = res["cost_thousand_usd"]["no_system"]
    post = {"label": "Post hoc and descriptive. No decision depends on it.",
            "share_of_test_losses_in_two_banks": 1 - no_model / total,
            "without_those_two": {"no_system_thousand": no_model, "with_system_thousand": at["total_cost"], "saving_thousand": no_model - at["total_cost"], "saving_pct": (no_model - at["total_cost"]) / no_model * 100, "bank_recall_actionable": at["bank_recall"]},
            "with_them_saving_pct": res["cost_thousand_usd"]["saving"] / total * 100,
            "calibration": {r["model"]: {"mean_pred": r.get("mean_pred"), "observed": r.get("observed")} for r in met.iter_rows(named=True) if r.get("mean_pred") is not None},
            "dev_alarm_share": frozen["dev_oof"]["alarm_share"], "test_alarm_share": res["at_threshold"]["alarm_share"]}
    (R / "p5_test_posthoc.json").write_text(json.dumps(post, indent=1))
    print(json.dumps(post, indent=1))


if __name__ == "__main__":
    main()
