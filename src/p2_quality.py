from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from p2_build_panel import END_Q, classify_exits, failure_events, load_bronze, q_index  # noqa: E402

NUM_ORDER = C.INT_FIELDS + C.RATIO_FIELDS
INK, MUTED, GRID, BLUE, ORANGE = "#1f2328", "#57606a", "#d8dee4", "#2a78d6", "#eb6834"


def profile(fin: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for c in NUM_ORDER:
        s = fin[c]
        first = fin.filter(pl.col(c).is_not_null() & (pl.col(c) != 0))["REPDTE"].min()
        rows.append({"column": c, "null_pct": round(100 * s.null_count() / fin.height, 2),
                     "zero_pct": round(100 * float((s == 0).sum()) / fin.height, 2),
                     "negative_pct": round(100 * float((s < 0).sum()) / fin.height, 2),
                     "min": float(s.min()), "p01": float(s.quantile(.01)), "median": float(s.median()),
                     "p99": float(s.quantile(.99)), "max": float(s.max()), "first_nonzero_report": str(first)})
    return pl.DataFrame(rows)


def by_year(fin: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    g = fin.group_by(C.PARTITION_KEY)
    miss = g.agg([(pl.col(c).is_null().mean() * 100).round(1).alias(c) for c in NUM_ORDER]).sort(C.PARTITION_KEY)
    zero = g.agg([((pl.col(c) == 0).mean() * 100).round(1).alias(c) for c in NUM_ORDER]).sort(C.PARTITION_KEY)
    return miss, zero


def issue_register(fin: pl.DataFrame, fail: pl.DataFrame, inst: pl.DataFrame, exits: pl.DataFrame) -> pl.DataFrame:
    def row_issue(dim: str, issue: str, cond: pl.Expr, action: str) -> dict:
        hit = fin.filter(cond.fill_null(False))
        period = f"{hit['REPDTE'].min().year}-{hit['REPDTE'].max().year}" if hit.height else "-"
        return {"dimension": dim, "issue": issue, "rows": hit.height, "banks": hit["CERT"].n_unique(),
                "period": period, "handling": action}

    def plain(dim: str, issue: str, rows: int, banks: int | None, period: str, action: str) -> dict:
        return {"dimension": dim, "issue": issue, "rows": rows, "banks": banks, "period": period, "handling": action}

    a, l, e = pl.col("ASSET"), pl.col("LIAB"), pl.col("EQ")
    events = failure_events(fail)
    post = fin.join(events, on="CERT").filter(pl.col("REPDTE") >= pl.col("_fail_date"))
    no_hist = events.join(fin.select("CERT").unique(), on="CERT", how="anti")
    twice = fail.filter(pl.col("RESTYPE") == "FAILURE", pl.col("CERT").is_not_null(), pl.col("FAILDATE").is_between(date(1984, 1, 1), END_Q)).group_by("CERT").len().filter(pl.col("len") > 1)
    assist = fail.filter(pl.col("RESTYPE") == "ASSISTANCE", pl.col("FAILDATE").is_between(date(1984, 1, 1), END_Q))
    after = fail.filter(pl.col("FAILDATE") > END_Q)
    sort = fin.select("CERT", "REPDTE", "INTINC", C.PARTITION_KEY).sort("CERT", "REPDTE")
    drop = sort.with_columns(pl.col("INTINC").shift(1).over("CERT", C.PARTITION_KEY).alias("prev")).filter(pl.col("INTINC") < pl.col("prev"))
    gaps = fin.select("CERT", q_index("REPDTE").alias("q")).sort("CERT", "q").with_columns((pl.col("q") - pl.col("q").shift(1).over("CERT")).alias("d")).filter(pl.col("d") > 1)
    not_inst = fin.join(inst.select("CERT"), on="CERT", how="anti")
    states = fin.filter(pl.col("STNAME") != "").group_by("CERT").agg(pl.col("STNAME").n_unique()).filter(pl.col("STNAME") > 1)
    inactive_reporting = exits.filter(pl.col("ACTIVE") == 0, pl.col("_last_report") >= END_Q, pl.col("_fail_date").is_null())
    distress = exits.filter(pl.col("_exit_class") == "distress_no_failure_record")
    sentinel = inst.filter(pl.col("ENDEFYMD").dt.year() >= 9999)
    neg = pl.any_horizontal([pl.col(c) < 0 for c in C.NONNEG_STOCKS])

    return pl.DataFrame([
        row_issue("uniqueness", "Duplicate (CERT, REPDTE)", pl.struct("CERT", "REPDTE").is_duplicated(), "Build stops if any appear"),
        plain("uniqueness", "Same CERT with two in-scope FAILURE records", 0, twice.height, "1986-1989", "First date is the event; panel stops there"),
        plain("uniqueness", "Failures rows with no CERT", fail["CERT"].null_count(), None, "before 1984", "Out of scope; panel starts 1984"),
        row_issue("completeness", "P3ASSET null: field starts in 2001", pl.col("P3ASSET").is_null(), "Keep null; Part 4 cannot use it across eras without care"),
        row_issue("completeness", "OTHBFHLB null, mostly commercial banks before 2001", pl.col("OTHBFHLB").is_null(), "Keep null; never impute before fitting"),
        row_issue("completeness", "P9ASSET null, mostly thrifts before 1990", pl.col("P9ASSET").is_null(), "Keep null; missingness tracks charter type"),
        row_issue("completeness", "RBCRWAJ == 0: not reported before 1990Q1", (pl.col("RBCRWAJ") == 0) & (pl.col("REPDTE") < date(1990, 1, 1)), "R4 zero to null"),
        row_issue("completeness", "RBCRWAJ == 0 from 2020 with leverage ratio present (CBLR election)", (pl.col("RBCRWAJ") == 0) & (pl.col("RBC1AAJ") != 0) & (pl.col("REPDTE") >= date(2020, 1, 1)), "R4 zero to null; f_cblr flag"),
        row_issue("completeness", "RBC1AAJ == 0: not reported, mostly thrifts before 1990", pl.col("RBC1AAJ") == 0, "R4 zero to null"),
        row_issue("completeness", "EQ null: no capital (branches of foreign banks and similar)", pl.col("EQ").is_null(), "R2 remove row; none of these banks failed"),
        plain("completeness", "Panel banks with no Institutions record", not_inst.height, not_inst["CERT"].n_unique(), f"{not_inst['REPDTE'].min().year}-{not_inst['REPDTE'].max().year}",
              f"R1 remove; none failed, {100 * float((not_inst['DEP'] == 0).mean()):.0f}% of rows have no deposits"),
        plain("completeness", "In-scope failures with no report before failure", 0, no_hist.height, "1984", "Cannot be labelled; stated as a limitation"),
        plain("completeness", "Reporting gaps inside a bank's history", gaps.height, gaps["CERT"].n_unique(), "1984-2026", "R9 leaves flow null; f_gap_before; no forward fill"),
        row_issue("validity", "Negative balance that cannot be negative", neg, "R7 to null"),
        row_issue("validity", "LNLSDEPR == 0 only because DEP == 0", (pl.col("DEP") == 0) & (pl.col("LNLSDEPR") == 0), "R5 to null; f_no_deposits"),
        row_issue("validity", "NCLNLSR == 0 only because LNLSGR == 0", (pl.col("LNLSGR") == 0) & (pl.col("NCLNLSR") == 0), "R5 to null"),
        row_issue("validity", "ROE with EQ <= 0: sign is meaningless", e <= 0, "R6 to null"),
        row_issue("validity", "Extreme ratio: |ROA| > 100, |ROE| > 1000, RBC1AAJ > 100 or LNLSDEPR > 1000", (pl.col("ROA").abs() > 100) | (pl.col("ROE").abs() > 1000) | (pl.col("RBC1AAJ") > 100) | (pl.col("LNLSDEPR") > 1000), "Keep; capped in Part 4 with limits fitted on training data"),
        row_issue("validity", "ASSET below USD 1 million", a < 1000, "Keep; ratios are unstable, handled by Part 4 capping"),
        row_issue("validity", "Blank STNAME", pl.col("STNAME") == "", "R8 to null"),
        plain("validity", "ENDEFYMD == 9999-12-31 sentinel in Institutions", sentinel.height, sentinel.height, "-", "Institutions dates are never features"),
        row_issue("consistency", "|ASSET - LIAB - EQ| > 1% of assets", ((a - l - e).abs() / a) > 0.01, "Keep; f_identity_break"),
        row_issue("consistency", "DEP > ASSET: consistent with negative equity", pl.col("DEP") > a, "Keep: real distress signal, not an error"),
        row_issue("consistency", "LNRE > LNLSGR", pl.col("LNRE") > pl.col("LNLSGR"), "Keep; excess is small (median under 4%) and mostly before 2000"),
        row_issue("consistency", "Flows are year-to-date: NETINC, INTINC and others rise through each year", pl.col("REPDTE").dt.quarter() > 1, "R9 quarterly <FLOW>_Q"),
        plain("consistency", "INTINC falls inside a year (YTD restated or reset)", drop.height, drop["CERT"].n_unique(), "1984-2026", "Keep; negative quarterly value flagged f_ytd_reversal"),
        plain("consistency", "Bank changes STNAME over time", None, states.height, "1984-2026", "Keep the value reported at q; do not use Institutions' current state"),
        plain("timeliness", "Reports dated on or after failure (thrifts in 1989 above all)", post.height, post["CERT"].n_unique(),
              f"{post['REPDTE'].min().year}-{post['REPDTE'].max().year}", "R3 remove: else labelled y = 0 after the bank has failed"),
        plain("timeliness", "ASSISTANCE: charter survives and bank keeps reporting", assist.height, assist["CERT"].n_unique(), "1984-2009", "Not a failure; rows marked _near_miss for sensitivity"),
        plain("timeliness", "Failures after 2026Q2", after.height, after["CERT"].n_unique(), "2026", "Out of scope; no row in the model period is affected"),
        plain("timeliness", "Institutions says inactive, but bank still reports in 2026Q2", inactive_reporting.height, inactive_reporting.height, "2026", "Proves it is a current snapshot: never a feature"),
        plain("timeliness", "Distress exit code but no Failures record", None, distress.height, "1984-2026", "Exam definition kept: y = 0; rows marked _near_miss for sensitivity"),
    ])


def exit_table(panel: pl.DataFrame, exits: pl.DataFrame) -> pl.DataFrame:
    """Financial health at the last report before exit: the evidence for separating exit types."""
    last = panel.sort("CERT", "REPDTE").group_by("CERT").last()
    d = last.with_columns((pl.col("EQ") / pl.col("ASSET") * 100).alias("eq_ta"), (pl.col("NCLNLS") / pl.col("ASSET") * 100).alias("nc_ta"))
    return (d.group_by("_exit_class").agg(
        pl.len().alias("banks"),
        pl.col("REPDTE").min().dt.year().alias("first_year"), pl.col("REPDTE").max().dt.year().alias("last_year"),
        pl.col("ASSET").median().round(0).alias("median_asset"),
        pl.col("eq_ta").median().round(2).alias("median_eq_ta_pct"),
        ((pl.col("EQ") <= 0).mean() * 100).round(1).alias("eq_nonpositive_pct"),
        pl.col("ROA").median().round(2).alias("median_roa"),
        pl.col("nc_ta").median().round(2).alias("median_nc_ta_pct"),
    ).sort(["banks", "_exit_class"], descending=[True, False]))


def distress_detail(panel: pl.DataFrame, exits: pl.DataFrame) -> pl.DataFrame:
    return (exits.filter(pl.col("_exit_class").is_in(["failure", "distress_no_failure_record", "assisted_merger", "merger", "voluntary_closing", "other_or_unknown"]))
            .group_by("_exit_class", "CHANGEC1").len().sort("_exit_class", "len", descending=[False, True]))


def failure_timing(panel: pl.DataFrame, fail: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    failed = panel.filter(pl.col("_fail_date").is_not_null()).sort("CERT", "REPDTE").group_by("CERT").last()
    gap = failed.group_by("_quarters_to_fail").len().sort("_quarters_to_fail").rename({"_quarters_to_fail": "quarters_last_report_to_failure", "len": "banks"})
    days = pl.DataFrame([{"failure_within_days_of_last_report": n, "banks": failed.filter(pl.col("_days_to_fail") <= n).height,
                          "pct": round(100 * failed.filter(pl.col("_days_to_fail") <= n).height / failed.height, 1)} for n in (30, 45, 60, 90, 120, 180)])
    return gap, days


def structural_breaks(fin: pl.DataFrame, top: int = 15) -> pl.DataFrame:
    """Largest one-quarter jumps in the cross-sectional median, in robust z-scores of that series.
    Flows are excluded: their year-to-date sawtooth would dominate the scan."""
    stocks = [c for c in C.INT_FIELDS if c not in C.YTD_FLOWS and c not in ("ASSET", "NUMEMP")]
    q = (fin.filter(pl.col("ASSET") > 0).group_by("REPDTE").agg(
        [(pl.col(c).cast(pl.Float64) / pl.col("ASSET") * 100).median().alias(c) for c in stocks]
        + [pl.col(c).median() for c in ("ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ", "NCLNLSR", "LNLSDEPR")]).sort("REPDTE"))
    rows = []
    for c in q.columns[1:]:
        s = q.select("REPDTE", pl.col(c).alias("v"), pl.col(c).diff().alias("d")).drop_nulls("d")
        med = s["d"].median()
        mad = (s["d"] - med).abs().median()
        scale = 1.4826 * mad if mad else None
        if scale is None:
            continue
        t = s.with_columns(((pl.col("d") - med).abs() / scale).alias("z")).sort("z", descending=True).row(0, named=True)
        rows.append({"series": c, "quarter": str(t["REPDTE"]), "median_after": round(t["v"], 3), "change": round(t["d"], 3), "robust_z": round(t["z"], 1)})
    return pl.DataFrame(rows).sort("robust_z", descending=True).head(top)


def style(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figures(fin: pl.DataFrame, panel: pl.DataFrame, miss: pl.DataFrame, rep: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Missing share by field and year. One hue, light to dark; fields that are never missing are left out.
    cols = [c for c in NUM_ORDER if miss[c].max() >= 1]
    m = miss.select(cols).to_numpy().T
    years = miss[C.PARTITION_KEY].to_list()
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(cols) + 1.2))
    im = ax.imshow(m, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_yticks(range(len(cols)), cols, fontsize=9, color=INK)
    ticks = [i for i, y in enumerate(years) if y % 5 == 0]
    ax.set_xticks(ticks, [years[i] for i in ticks], fontsize=9, color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    bar.set_label("% of reports with a null value", fontsize=9, color=MUTED)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    ax.set_title("Null values are concentrated in specific fields and eras", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(rep / "fig_p2_missing.png", dpi=200)
    plt.close(fig)

    # Capital ratios stored as zero, by quarter. Two series: legend plus direct labels.
    z = fin.group_by("REPDTE").agg(((pl.col("RBCRWAJ") == 0).mean() * 100).alias("r"), ((pl.col("RBC1AAJ") == 0).mean() * 100).alias("l")).sort("REPDTE")
    fig, ax = plt.subplots(figsize=(8.5, 3))
    ax.plot(z["REPDTE"], z["r"], color=BLUE, linewidth=2, label="RBCRWAJ (total risk-based)")
    ax.plot(z["REPDTE"], z["l"], color=ORANGE, linewidth=2, label="RBC1AAJ (leverage)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of reports equal to 0", color=MUTED, fontsize=9)
    ax.text(date(1991, 6, 30), 88, "Not reported before 1990Q1", fontsize=9, color=INK)
    ax.text(date(2019, 6, 30), 47, "Community bank leverage\nratio election from 2020Q1", fontsize=9, color=INK, ha="right")
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    style(ax)
    ax.set_title("A zero capital ratio usually means \"not reported\"", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(rep / "fig_p2_capital_zero.png", dpi=200)
    plt.close(fig)

    # Year-to-date vs quarterly interest income: aggregate of all banks.
    # Amounts are in thousands of USD, so dividing by 1e6 gives USD billions.
    a = panel.filter(pl.col("YEAR").is_between(2015, 2019)).group_by("REPDTE").agg(pl.col("INTINC").sum() / 1e6, pl.col("INTINC_Q").sum() / 1e6).sort("REPDTE")
    fig, ax = plt.subplots(figsize=(8.5, 3))
    ax.plot(a["REPDTE"], a["INTINC"], color=ORANGE, linewidth=2, marker="o", markersize=4, label="As reported (year-to-date)")
    ax.plot(a["REPDTE"], a["INTINC_Q"], color=BLUE, linewidth=2, marker="o", markersize=4, label="Quarterly (INTINC_Q)")
    ax.set_ylim(0, None)
    ax.set_ylabel("Interest income, USD bn", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=2, labelcolor=INK)
    style(ax)
    ax.set_title("Reported income restarts every Q1; quarterly values do not", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(rep / "fig_p2_ytd.png", dpi=200)
    plt.close(fig)

    # Failure events and positive-label rate. Two scales, so two panels rather than two axes.
    ev = panel.filter(pl.col("_fail_date").is_not_null()).group_by("CERT").first().group_by(pl.col("_fail_date").dt.year().alias("y_")).len().sort("y_")
    rate = panel.filter(pl.col("y").is_not_null()).group_by(C.PARTITION_KEY).agg((pl.col("y").mean() * 100).alias("r")).sort(C.PARTITION_KEY)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True)
    a1.bar(ev["y_"], ev["len"], color=BLUE, width=0.75)
    a1.set_title("Labelled failures per year", loc="left", fontsize=10, color=INK)
    a2.bar(rate[C.PARTITION_KEY], rate["r"], color=BLUE, width=0.75)
    a2.set_title("Share of bank-quarters with y = 1 (%)", loc="left", fontsize=10, color=INK)
    for ax in (a1, a2):
        style(ax)
    fig.tight_layout()
    fig.savefig(rep / "fig_p2_failures.png", dpi=200)
    plt.close(fig)


def md(df: pl.DataFrame) -> str:
    d = df.with_columns(pl.all().cast(pl.String).fill_null("-"))
    return d.to_pandas().to_markdown(index=False, disable_numparse=True)


def main() -> None:
    rep = C.REPORTS
    fin, fail, inst = load_bronze()
    panel = pl.read_parquet(C.DATA_SILVER / "panel" / "**" / "*.parquet", hive_partitioning=True)
    exits = classify_exits(fin, fail, inst)

    prof = profile(fin)
    miss, zero = by_year(fin)
    reg = issue_register(fin, fail, inst, exits)
    ext = exit_table(panel, exits)
    codes = distress_detail(panel, exits)
    gap, days = failure_timing(panel, fail)
    brk = structural_breaks(fin)
    wf = pl.read_csv(rep / "p2_waterfall.csv")

    prof.write_csv(rep / "p2_profile.csv")
    miss.write_csv(rep / "p2_null_pct_by_year.csv")
    zero.write_csv(rep / "p2_zero_pct_by_year.csv")
    reg.write_csv(rep / "p2_issue_register.csv")
    ext.write_csv(rep / "p2_exit_health.csv")
    codes.write_csv(rep / "p2_exit_codes.csv")
    gap.write_csv(rep / "p2_failure_report_gap.csv")
    days.write_csv(rep / "p2_failure_days_after_report.csv")
    brk.write_csv(rep / "p2_structural_breaks.csv")
    figures(fin, panel, miss, rep)

    scope = panel.filter(pl.col("y").is_not_null())
    text = "\n\n".join([
        "# Data-quality report (generated by src/p2_quality.py)",
        f"Bronze: {fin.height:,} rows, {fin['CERT'].n_unique():,} banks. Silver: {panel.height:,} rows, {panel['CERT'].n_unique():,} banks. "
        f"Model scope (to {C.LAST_MODEL_Q}): {scope.height:,} rows, {int(scope['y'].sum()):,} with y = 1, "
        f"{scope.filter(pl.col('y') == 1)['CERT'].n_unique():,} failed banks; "
        f"{int(scope['_near_miss'].sum()):,} y = 0 rows are near misses; {int(scope['_other_exit_4q'].sum()):,} y = 0 rows have a non-failure exit inside the window.",
        "## Issue register", md(reg),
        "## Row and cell waterfall", md(wf),
        "## Exit classes: health at the last report", md(ext),
        "## Exit class by change code", md(codes),
        "## Quarters from last report to failure", md(gap),
        "## Failure soon after the last report", md(days),
        "## Largest one-quarter jumps in cross-sectional medians", md(brk),
    ])
    (rep / "p2_dq_report.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
