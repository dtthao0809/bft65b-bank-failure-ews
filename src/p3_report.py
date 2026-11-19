from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

INK, MUTED, GRID = "#1f2328", "#57606a", "#d8dee4"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BASE_ROWS = 1_666_719
LABEL = {"polars_stream": "Polars streaming", "polars_mem": "Polars in-memory", "polars_noopt": "Polars, optimiser off",
         "pandas_cols": "pandas, 16 columns", "pandas_all": "pandas, all columns", "spark": "PySpark"}


def style(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def medians(d: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    return (d.filter(pl.col("status") == "ok").group_by(by)
            .agg(pl.len().alias("runs"), pl.col("wall_s").median(), pl.col("wall_s").min().alias("min_s"), pl.col("wall_s").max().alias("max_s"),
                 pl.col("cpu_s").median(), pl.col("peak_rss_gb").median(), pl.col("avail_start_gb").median(),
                 (pl.col("shuffle_write_bytes").median() / 1e9).alias("shuffle_gb"), (pl.col("spill_disk_bytes").median() / 1e9).alias("spill_gb"))
            .sort(by))


def main() -> None:
    d = pl.read_csv(C.REPORTS / "p3_benchmark_runs.csv", infer_schema_length=None)
    R = C.REPORTS

    # Engines. Spark settings are compared only inside the interleaved spark_tune suite: suites ran
    # at different times, and the machine's other load differed between them.
    e = medians(d.filter(pl.col("suite") == "engines"), ["engine", "shuffle_partitions"])
    ref = e.filter(pl.col("engine") == "polars_stream")
    e = e.with_columns((pl.col("wall_s") / ref["wall_s"][0]).alias("x_slower_than_polars_stream"),
                       (pl.col("cpu_s") / ref["cpu_s"][0]).alias("x_cpu"),
                       (pl.col("peak_rss_gb") / ref["peak_rss_gb"][0]).alias("x_memory")).sort("wall_s")
    e.write_csv(R / "p3_engines.csv", float_precision=3)
    tune = medians(d.filter(pl.col("suite") == "spark_tune"), ["threads", "shuffle_partitions", "heap"]).sort("wall_s")
    tune.write_csv(R / "p3_spark_settings.csv", float_precision=3)

    # Strong scaling.
    t = medians(d.filter(pl.col("suite") == "threads"), ["engine", "threads"])
    t = t.join(t.filter(pl.col("threads") == 1).select("engine", pl.col("wall_s").alias("t1")), on="engine").with_columns(
        (pl.col("t1") / pl.col("wall_s")).alias("speedup")).with_columns((pl.col("speedup") / pl.col("threads")).alias("efficiency")).drop("t1")
    t.write_csv(R / "p3_threads.csv", float_precision=3)

    # Data volume. x1 comes from the engines suite for the same setting.
    sc = d.filter(pl.col("suite") == "scale").with_columns(
        pl.when(pl.col("threads") > 0).then(pl.col("engine") + "@" + pl.col("threads").cast(pl.String)).otherwise(pl.col("engine")).alias("spec"))
    x1 = d.filter(pl.col("status") == "ok", ((pl.col("suite") == "engines") & pl.col("engine").is_in(["polars_stream", "polars_mem", "pandas_cols"]))
                  | ((pl.col("suite") == "threads") & pl.col("engine").str.starts_with("polars") & (pl.col("threads") == 4))).with_columns(
        pl.when(pl.col("threads") > 0).then(pl.col("engine") + "@" + pl.col("threads").cast(pl.String)).otherwise(pl.col("engine")).alias("spec"))
    x1m = x1.group_by("spec").agg(pl.lit(1).cast(pl.Int64).alias("scale"), pl.lit("ok").alias("status"), pl.col("wall_s").median(), pl.col("cpu_s").median(), pl.col("peak_rss_gb").median(), pl.col("avail_start_gb").median())
    sp = sc.with_columns(pl.col("scale").cast(pl.Int64)).select("spec", "scale", "status", "wall_s", "cpu_s", "peak_rss_gb", "avail_start_gb", "heap", "shuffle_partitions",
                   (pl.col("shuffle_write_bytes") / 1e9).alias("shuffle_gb"), (pl.col("spill_disk_bytes") / 1e9).alias("spill_gb"), "retries")
    all_sc = pl.concat([x1m, sp], how="diagonal").with_columns(pl.col("scale").cast(pl.Int64)).with_columns((pl.col("scale") * BASE_ROWS).alias("rows")).with_columns(
        pl.when(pl.col("status") == "ok").then(pl.col("rows") / pl.col("wall_s") / 1e6).alias("m_rows_per_s")).sort("spec", "scale", "heap")
    all_sc.write_csv(R / "p3_scale.csv", float_precision=3)

    # The case for a cluster.
    silver = pl.scan_parquet(C.DATA_SILVER / "panel" / "**" / "*.parquet", hive_partitioning=True, hive_schema={"YEAR": pl.Int16})
    per_year = silver.group_by("YEAR").agg(pl.len().alias("n")).sort("YEAR").collect()
    last_full = per_year.filter(pl.col("YEAR") == 2025)["n"][0]

    def slope(spec: str) -> dict | None:
        ok = all_sc.filter(pl.col("spec") == spec, pl.col("status") == "ok").sort("scale")
        if ok.height < 2:
            return None
        a, b = ok.row(0, named=True), ok.row(-1, named=True)
        gb_per_x = (b["peak_rss_gb"] - a["peak_rss_gb"]) / (b["scale"] - a["scale"])
        out = {"from_scale": a["scale"], "to_scale": b["scale"], "gb_per_copy": round(gb_per_x, 3),
               "largest_ok_scale_measured": b["scale"]}
        for budget in (12.0, 64.0):
            x = (budget - a["peak_rss_gb"]) / gb_per_x + a["scale"]
            out[f"extrapolated_scale_at_{int(budget)}gb"] = round(x, 1)
            out[f"extrapolated_m_rows_at_{int(budget)}gb"] = round(x * BASE_ROWS / 1e6, 1)
        return out

    spark_row = e.filter(pl.col("engine") == "spark", pl.col("shuffle_partitions") == 0).row(0, named=True)
    spark_best = tune.row(0, named=True)
    pol = ref.row(0, named=True)
    pol_1t = t.filter(pl.col("engine") == "polars_stream", pl.col("threads") == 1).row(0, named=True)
    spk_18 = t.filter(pl.col("engine") == "spark").sort("threads").row(-1, named=True)
    spk_1t = t.filter(pl.col("engine") == "spark", pl.col("threads") == 1).row(0, named=True)
    pl_best = t.filter(pl.col("engine") == "polars_stream").sort("speedup").row(-1, named=True)
    spk_best = t.filter(pl.col("engine") == "spark").sort("speedup").row(-1, named=True)
    plan = json.loads((R / "p3_plan_summary.json").read_text(encoding="utf-8")) if (R / "p3_plan_summary.json").exists() else {}
    shuffle = plan.get("spark", {}).get("shuffle_write_gb")
    case = {
        "data": {"rows_panel": BASE_ROWS, "rows_added_in_2025": last_full,
                 "years_of_growth_to_double": round(BASE_ROWS / last_full, 1)},
        "polars_streaming": {"wall_s": pol["wall_s"], "cpu_s": pol["cpu_s"], "peak_gb": pol["peak_rss_gb"]},
        "spark_default": {"wall_s": spark_row["wall_s"], "cpu_s": spark_row["cpu_s"], "peak_gb": spark_row["peak_rss_gb"]},
        "spark_best_setting": {k: spark_best[k] for k in ("threads", "shuffle_partitions", "heap", "wall_s", "cpu_s", "peak_rss_gb")},
        "cpu_ratio_spark_to_polars": round(spark_best["cpu_s"] / pol["cpu_s"], 1),
        "cpu_ratio_spark_to_polars_1_thread": round(spk_1t["cpu_s"] / pol_1t["cpu_s"], 1),
        "cores_spark_needs_to_match_polars_at_perfect_scaling": round(spk_1t["wall_s"] / pol["wall_s"]),
        "speedup": {"polars_stream_best": {"threads": pl_best["threads"], "speedup": round(pl_best["speedup"], 2)},
                    "spark_best": {"threads": spk_best["threads"], "speedup": round(spk_best["speedup"], 2)}, "spark_at_18": round(spk_18["speedup"], 2)},
        "network_lower_bound": {
            "assumption": "10 Gb/s link, 1.25 GB/s, every shuffled byte crosses it once, no contention. Assumed, not measured.",
            "shuffle_gb_per_run_x1": shuffle, "seconds_x1": round(shuffle / 1.25, 2) if shuffle else None},
        "memory_extrapolation_not_measured": {s: slope(s) for s in ("polars_stream", "polars_mem", "pandas_cols")},
        "extrapolation_warning": "Two points only (x1, x2). x4 could not finish on this machine, so treat these as rough. "
                                 "The 4-thread settings are left out: their x4 runs had already passed the line through x1 and x2.",
    }
    (R / "p3_cluster_case.json").write_text(json.dumps(case, indent=1), encoding="utf-8")
    print(json.dumps(case, indent=1))

    # ---- figures ----
    order = [("polars_stream", 0, "Polars streaming"), ("polars_mem", 0, "Polars in-memory"), ("polars_noopt", 0, "Polars, optimiser off"),
             ("pandas_cols", 0, "pandas, 16 columns"), ("pandas_all", 0, "pandas, all columns"), ("spark", 18, "PySpark, 18 partitions"), ("spark", 0, "PySpark, 200 partitions")]
    rows = [(lab, e.filter(pl.col("engine") == en, pl.col("shuffle_partitions") == sp_).row(0, named=True)) for en, sp_, lab in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    ypos = list(range(len(rows)))[::-1]
    for ax, key, title, unit in ((axes[0], "wall_s", "Time (median, s)", "s"), (axes[1], "peak_rss_gb", "Peak memory (median, GB)", "GB")):
        vals = [r[1][key] for r in rows]
        ax.barh(ypos, vals, height=0.62, color=[BLUE if "Polars" in r[0] else ORANGE if "PySpark" in r[0] else AQUA for r in rows])
        for y, v in zip(ypos, vals):
            ax.text(v + max(vals) * 0.012, y, f"{v:.1f}" if v >= 10 else f"{v:.2f}", va="center", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(vals) * 1.14)
    axes[0].set_yticks(ypos, [r[0] for r in rows], color=INK)
    fig.suptitle(f"Same task, same answer: {BASE_ROWS / 1e6:.2f} million bank-quarters from bronze to Parquet", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(R / "fig_p3_engines.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    style(ax)
    xs = [1, 2, 4, 8, 12, 18]
    ax.plot(xs, xs, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(7.2, 11.3, "perfect scaling", color=MUTED, fontsize=9)
    for en, col, lab in (("polars_stream", BLUE, "Polars streaming"), ("spark", ORANGE, "PySpark")):
        q = t.filter(pl.col("engine") == en).sort("threads")
        ax.plot(q["threads"], q["speedup"], color=col, linewidth=2, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1)
        ax.text(q["threads"][-1] + 0.3, q["speedup"][-1], f"{lab}\n{q['speedup'][-1]:.1f}x", color=INK, fontsize=9, va="center")
    ax.set_xlim(0, 22.5)
    ax.set_ylim(0, 14)
    ax.set_xticks(xs)
    ax.set_xlabel("Threads", fontsize=9, color=MUTED)
    ax.set_ylabel("Speed-up over 1 thread", fontsize=9, color=MUTED)
    ax.set_title("Neither engine uses 18 threads well on a pipeline this size", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(R / "fig_p3_threads.png", dpi=180)
    plt.close(fig)

    series = [("polars_stream", BLUE, "Polars streaming"), ("polars_stream@4", AQUA, "Polars streaming, 4 threads"), ("pandas_cols", YELLOW, "pandas"), ("spark@6", ORANGE, "PySpark, 2 GB heap, 6 threads")]
    spk = all_sc
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, key, title in ((axes[0], "wall_s", "Time (s, log scale)"), (axes[1], "peak_rss_gb", "Peak memory (GB)")):
        style(ax)
        for spec, col, lab in series:
            ok = spk.filter(pl.col("spec") == spec, pl.col("status") == "ok").sort("scale")
            bad = spk.filter(pl.col("spec") == spec, pl.col("status") != "ok").sort("scale")
            if ok.height:
                ax.plot(ok["scale"], ok[key], color=col, linewidth=2, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1, label=lab)
            if bad.height and key == "peak_rss_gb":
                ax.scatter(bad["scale"], bad[key], marker="x", color=col, s=60, linewidths=2, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["x1", "x2", "x4", "x8", "x16"])
        ax.minorticks_off()
        ax.set_xlabel("Copies of the data (x1 = 1.67 million rows)", fontsize=9, color=MUTED)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].set_yscale("log")
    axes[0].set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    axes[0].set_ylim(1, 1000)
    axes[0].minorticks_off()
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    axes[1].set_ylim(0, 4.3)
    free = d.filter(pl.col("suite") == "scale")["avail_start_gb"].median()
    axes[1].axhline(free, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    axes[1].text(1, free + 0.08, f"typical free memory at start: {free:.1f} GB", color=MUTED, fontsize=9)
    axes[1].text(0.99, 0.03, "x = stopped by the memory guard", transform=axes[1].transAxes, ha="right", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "fig_p3_scale.png", dpi=180)
    plt.close(fig)

    with pl.Config(tbl_rows=40, tbl_cols=14, tbl_width_chars=170, float_precision=2):
        print(e.select("engine", "shuffle_partitions", "runs", "wall_s", "min_s", "max_s", "cpu_s", "peak_rss_gb", "x_slower_than_polars_stream", "x_cpu", "x_memory"))
        print(tune.select("threads", "shuffle_partitions", "heap", "runs", "wall_s", "min_s", "max_s", "cpu_s", "peak_rss_gb", "spill_gb"))
        print(t.select("engine", "threads", "runs", "wall_s", "speedup", "efficiency", "peak_rss_gb"))
        print(all_sc.select("spec", "scale", "heap", "status", "wall_s", "cpu_s", "peak_rss_gb", "avail_start_gb", "shuffle_gb", "spill_gb", "m_rows_per_s", "retries"))


if __name__ == "__main__":
    main()
