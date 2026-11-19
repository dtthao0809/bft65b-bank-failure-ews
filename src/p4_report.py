from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#57606a", "#d8dee4"
BLUE, ORANGE, GAP = "#2a78d6", "#eb6834", "#9aa3ad"


def style(ax, axis: str = "y") -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    R = C.REPORTS
    gold = pl.scan_parquet(C.DATA_GOLD / "features" / "**" / "*.parquet", hive_partitioning=True, hive_schema={C.PARTITION_KEY: pl.Int16})

    # Failures per year of failure, coloured by the split of the first report that labels them.
    ev = (gold.filter(pl.col("y") == 1).group_by("CERT").agg(pl.col("_fail_date").first(), pl.col("_split").last())
          .with_columns(pl.col("_fail_date").dt.year().alias("fy")).group_by("fy", "_split").len().sort("fy").collect())
    fig, ax = plt.subplots(figsize=(10, 3.4))
    style(ax)
    colors = {"dev": BLUE, "gap": GAP, "test": ORANGE}
    bottoms: dict[int, int] = {}
    for sp, lab in (("dev", "development"), ("gap", "gap"), ("test", "test")):
        q = ev.filter(pl.col("_split") == sp)
        ax.bar(q["fy"], q["len"], bottom=[bottoms.get(y, 0) for y in q["fy"]], width=0.76, color=colors[sp], label=lab)
        for y, n in zip(q["fy"], q["len"]):
            bottoms[y] = bottoms.get(y, 0) + n
    totals = ev.group_by("_split").agg(pl.col("len").sum())
    t = dict(zip(totals["_split"], totals["len"]))
    ax.set_title(f"Failed banks by year of failure and the period of their last labelled report: development {t['dev']:,}, gap {t.get('gap', 0)}, test {t['test']}",
                 loc="left", fontsize=10, color=INK)
    ax.set_ylabel("Failed banks", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(R / "fig_p4_split.png", dpi=180)
    plt.close(fig)

    # Event study on development rows: how banks look in the quarters before they fail.
    dev = gold.filter(pl.col("_split") == "dev")
    show = [("eq_ta", "Equity / assets"), ("npl", "Noncurrent loans / loans"), ("roa_4q", "ROA, 4 quarters"), ("texas", "Bounded Texas ratio")]
    fq = pl.col("_fail_date").dt.year().cast(pl.Int32) * 4 + pl.col("_fail_date").dt.quarter().cast(pl.Int32) - 1
    failed = dev.filter(pl.col("_fail_date").is_not_null()).with_columns((fq - pl.col("q_idx")).alias("qtf")).filter(pl.col("qtf").is_between(1, 12))
    prof = failed.group_by("qtf").agg([pl.col(c).median() for c, _ in show]).sort("qtf").collect(engine="streaming")
    base = dev.filter(pl.col("_fail_date").is_null()).select([pl.col(c).median() for c, _ in show]).collect(engine="streaming")
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (c, title) in zip(axes, show):
        style(ax)
        ax.plot(prof["qtf"], prof[c], color=ORANGE, linewidth=2, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.8, label="banks that failed")
        ax.axhline(base[c][0], color=BLUE, linewidth=2, label="banks that did not fail, median")
        ax.set_xlim(12.5, 0.5)
        ax.set_xticks([12, 8, 4, 1])
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Quarters before failure", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(R / "fig_p4_event_study.png", dpi=180)
    plt.close(fig)
    prof.write_csv(R / "p4_event_study.csv", float_precision=5)

    # Pooled against within-era AUC: how much of each feature's signal is the calendar.
    s = pl.read_csv(R / "p4_feature_screen.csv").filter(pl.col("decision") == "kept").with_columns(
        (pl.col("auc_within_era") - 0.5).abs().alias("sw")).with_columns(
        pl.when(pl.col("auc_within_era") < 0.5).then(1 - pl.col("auc_within_era")).otherwise(pl.col("auc_within_era")).alias("w"),
        pl.when(pl.col("auc_within_era") < 0.5).then(1 - pl.col("auc_pooled")).otherwise(pl.col("auc_pooled")).alias("p")).sort("w")
    fig, ax = plt.subplots(figsize=(7.4, 6))
    style(ax, "x")
    ys = list(range(s.height))
    ax.hlines(ys, s["w"], s["p"], color=GRID, linewidth=2)
    ax.scatter(s["p"], ys, color=ORANGE, s=44, zorder=3, edgecolor="white", linewidth=0.8, label="pooled over all years")
    ax.scatter(s["w"], ys, color=BLUE, s=44, zorder=3, edgecolor="white", linewidth=0.8, label="within era, mean")
    ax.set_yticks(ys, s["feature"].to_list(), color=INK)
    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.set_xlabel("AUC of the feature alone, direction aligned so that above 0.5 means signal", fontsize=9, color=MUTED)
    fig.suptitle("Selected features, training window 1986-2006\nA wide gap means part of the signal is the calendar", x=0.01, ha="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(R / "fig_p4_signal.png", dpi=180)
    plt.close(fig)
    print("figures written", date.today())


if __name__ == "__main__":
    main()
