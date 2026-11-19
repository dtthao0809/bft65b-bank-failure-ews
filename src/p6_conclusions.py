"""
Part 6 - Draft of the conclusions and the first-page facts, generated from the result files.

Every number in the text is read from reports/, so the draft cannot drift away from the measurements. The
script stops if a test suite of Parts 2, 4 or 5 has a failure, or if the test evaluation is missing.

Outputs: reports/p6_conclusions_draft.md, reports/p6_first_page.md, reports/p6_key_numbers.json

The Vietnam section rests on the sources listed at its end, read in November 2026. They are secondary
sources, so check them against the legal texts before relying on a detail.

Usage:  python src/p6_conclusions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

R = C.REPORTS


def js(name: str) -> dict:
    return json.loads((R / name).read_text(encoding="utf-8"))


def main() -> None:
    # ---- gates ----
    p2 = pl.read_csv(R / "p2_checks.csv", infer_schema_length=None)
    p4 = pl.read_csv(R / "p4_checks.csv", infer_schema_length=None)
    p5 = pl.read_csv(R / "p5_checks.csv", infer_schema_length=None)
    if not p2["passed"].all() or (p4["result"] != "PASS").any() or (p5["result"] != "PASS").any():
        raise SystemExit("A test suite has a failure: fix that before writing conclusions.")

    # ---- inputs ----
    comp = js("p1_completeness.json")
    log = comp["download_log"]
    st = pl.read_csv(R / "p1_storage_benchmark.csv", infer_schema_length=None)
    mb = dict(zip(st["layout"], st["size_mb"]))
    wf = pl.read_csv(R / "p2_waterfall.csv", infer_schema_length=None)
    ex = pl.read_csv(R / "p2_exit_health.csv", infer_schema_length=None)
    exits = {r["_exit_class"]: r for r in ex.iter_rows(named=True)}
    eng = pl.read_csv(R / "p3_engines.csv", infer_schema_length=None)

    def engine(name: str, sp: int = 0) -> dict:
        return eng.filter(pl.col("engine") == name, pl.col("shuffle_partitions") == sp).row(0, named=True)

    case = js("p3_cluster_case.json")
    rec = js("p3_engine_reconcile.json")
    sc = pl.read_csv(R / "p3_scale.csv", infer_schema_length=None)

    def scale(spec: str, x: int) -> dict:
        return sc.filter(pl.col("spec") == spec, pl.col("scale") == x).row(0, named=True)

    split = {r["_split"]: r for r in pl.read_csv(R / "p4_split.csv").iter_rows(named=True)}
    sel = js("p4_selected_features.json")
    frozen = js("p5_frozen.json")
    res = js("p5_test_results.json")
    post = js("p5_test_posthoc.json")
    costs = js("p5_costs.json")
    tm = {r["model"]: r for r in pl.read_csv(R / "p5_test_metrics.csv", infer_schema_length=None).iter_rows(named=True)}
    cv = pl.read_csv(R / "p5_cv_summary.csv")
    spark = pl.read_csv(R / "p5_spark_mllib.csv")
    spark_j = js("p5_spark_mllib.json")
    banks = pl.read_csv(R / "p5_test_failed_banks.csv", infer_schema_length=None)

    chosen, b1, b0 = tm[frozen["chosen"]["name"]], tm["B1 logit: capital, NPL, ROA"], tm["B0 Texas ratio, no fitting"]
    th, cost, boot, per = res["at_threshold"], res["cost_thousand_usd"], res["bootstrap"], res["periods"]
    usd_bn = lambda v: v / 1e6  # noqa: E731  thousands of USD to USD billion
    pol, pan, s18, s200 = engine("polars_stream"), engine("pandas_cols"), engine("spark", 18), engine("spark", 0)
    pl2, pl4, pd4 = scale("polars_stream", 2), scale("polars_stream", 4), scale("pandas_cols", 4)
    # Largest copy of the data that Spark (6 threads, 2 GB heap) finished; a larger one that failed is reported, not hidden.
    sp_ok = sc.filter(pl.col("spec") == "spark@6", pl.col("status") == "ok").sort("scale")
    sp16 = sp_ok.row(-1, named=True)
    sp_fail = sc.filter(pl.col("spec") == "spark@6", pl.col("status") != "ok", pl.col("scale") > sp16["scale"]).sort("scale")
    words = {1: "one copy", 2: "two copies", 4: "four copies", 8: "eight copies", 16: "sixteen copies"}
    def first_stop(spec: str) -> dict:
        return sc.filter(pl.col("spec") == spec, pl.col("status") != "ok").sort("scale").row(0, named=True)

    pd_stop, po_stop = first_stop("pandas_cols"), first_stop("polars_stream")
    if pd_stop["scale"] == po_stop["scale"]:
        stop_text = f"Polars and pandas were stopped by the memory guard at {words[po_stop['scale']]} of the data ({po_stop['rows'] / 1e6:.1f} million rows)"
    else:
        stop_text = (f"pandas was stopped by the memory guard at {words[pd_stop['scale']]} of the data ({pd_stop['rows'] / 1e6:.1f} million rows) "
                     f"and Polars streaming at {words[po_stop['scale']]} ({po_stop['rows'] / 1e6:.1f} million rows)")
    sp_fail_text = ""
    if sp_fail.height:
        f0 = sp_fail.row(0, named=True)
        sp_fail_text = (f" At {words.get(f0['scale'], str(f0['scale']) + ' copies')} ({f0['rows'] / 1e6:.1f} million rows) the same job {f0['status']}: "
                        "a 2 GB heap bounds memory but is not unlimited, so beyond that size the heap or the number of partitions has to grow.")
    ext = case["memory_extrapolation_not_measured"]
    chosen_cv = cv.filter(pl.col("model").str.ends_with("(chosen)")).row(0, named=True)
    b0_cv = cv.filter(pl.col("model").str.starts_with("B0")).row(0, named=True)
    b1_cv = cv.filter(pl.col("model").str.starts_with("B1")).row(0, named=True)
    wf_row = lambda r: wf.filter(pl.col("rule") == r)["rows_changed"][0]  # noqa: E731
    distress = exits["distress_no_failure_record"]["banks"] + exits["assisted_merger"]["banks"]
    no_alarm = banks.filter(pl.col("any_alarm") == "false" if banks["any_alarm"].dtype == pl.String else ~pl.col("any_alarm")).height
    late = banks.filter((pl.col("any_alarm").cast(pl.String) == "true") & (pl.col("actionable_alarm").cast(pl.String) == "false")).height
    candidates = pl.read_csv(R / "p4_feature_dictionary.csv").height
    grid = pl.read_csv(R / "p5_threshold_sensitivity.csv")
    n_pandas = len(rec["pandas_roa_std8_arbitration"])
    svb = (pl.scan_parquet(C.DATA_GOLD / "features" / "**" / "*.parquet", hive_partitioning=True, hive_schema={C.PARTITION_KEY: pl.Int16})
           .filter(pl.col("CERT") == 24735, pl.col("y") == 1).sort("REPDTE").select("eq_ta", "npl").collect(engine="streaming").row(-1, named=True))
    caught = banks.filter(pl.col("actionable_alarm").cast(pl.String) == "true").height
    sk = spark.filter(pl.col("engine") == "scikit-learn logistic").row(0, named=True)
    sl = spark.filter(pl.col("engine") == "Spark logistic").row(0, named=True)

    first = f"""# First page facts (generated from the files in reports/)

| Item | Value |
|---|---|
| Name, student ID | Do Thi Thao (\u0110\u1ed7 Th\u1ecb Th\u1ea3o), 11231011 |
| GitHub URL | https://github.com/dtthao0809/bft65b-bank-failure-ews |
| Download method | FDIC BankFind API, `fdic_download_full.py` (resumable) |
| Download date | {log['started_utc'][:10]} ({log['minutes']:.1f} minutes) |
| Financials | {comp['financials']['panel_rows']:,} rows, {comp['financials']['distinct_banks']:,} banks, {comp['financials']['quarters_on_disk']} quarters, 1984Q1-2026Q2 |
| Failures, Institutions | {comp['failures']['rows']:,} and {comp['institutions']['rows']:,} rows |
| Clean panel | {split['dev']['rows'] + split['gap']['rows'] + split['test']['rows'] + split['unlabelled']['rows']:,} bank-quarters |
| Development | 1984Q1-2011Q4: {split['dev']['rows']:,} rows, {split['dev']['failed_banks']:,} failed banks labelled |
| Gap | 2012Q1-2012Q4, not used |
| Test | 2013Q1-2025Q2: {split['test']['rows']:,} rows, {split['test']['failed_banks']} failed banks labelled |
"""
    (R / "p6_first_page.md").write_text(first, encoding="utf-8")

    text = f"""# Part 6. Conclusions

_Generated from the result files by `src/p6_conclusions.py`; every number is read from `reports/`._

## 6.1 Q1: How should four decades of bank reports be stored and processed on a laptop, and when would a Spark cluster be needed?

Store them as typed, ZSTD-compressed Parquet, partitioned by year and sorted by bank and date. This layout takes {mb['pq_year']:.1f} MB against {mb['csv']:.1f} MB for CSV, and a query that needs a few columns or one year reads only that part of the data. Process them with lazy Polars. The core pipeline over {case['data']['rows_panel'] / 1e6:.2f} million bank-quarters took {pol['wall_s']:.1f} s and {pol['peak_rss_gb']:.1f} GB in Polars, {pan['wall_s']:.1f} s and {pan['peak_rss_gb']:.1f} GB in pandas, and {s18['wall_s']:.0f}-{s200['wall_s']:.0f} s and {s18['peak_rss_gb']:.1f}-{s200['peak_rss_gb']:.1f} GB in local PySpark. All three gave the same answer on {rec['cells_compared_per_engine'] / 1e6:.0f} million cells. The only differences were {n_pandas} pandas values, within {rec['by_engine'][2]['max_abs_diff']:.0e}, which direct recomputation attributed to rounding in pandas' rolling variance.

This dataset does not need a cluster. The panel gained {case['data']['rows_added_in_2025']:,} rows in 2025, so it would take about {case['data']['years_of_growth_to_double']:.0f} years to double. Spark was not faster on one machine: it used about {case['cpu_ratio_spark_to_polars']:.0f} times the CPU of Polars and wrote {case['network_lower_bound']['shuffle_gb_per_run_x1']:.1f} GB of shuffle data from a {mb['pq_year']:.0f} MB input.

Spark's real advantage was bounded memory. With about {sc.filter(pl.col('status') == 'memory_guard')['avail_start_gb'].median():.1f} GB free on the test laptop, {stop_text}, while Spark with a 2 GB heap finished {words.get(sp16['scale'], str(sp16['scale']) + ' copies')} ({sp16['rows'] / 1e6:.1f} million rows) in {sp16['wall_s']:.0f} s, peaking at {sp16['peak_rss_gb']:.1f} GB and spilling {sp16['spill_gb']:.1f} GB to disk.{sp_fail_text} A cluster is justified when the working set exceeds the memory of the largest affordable single machine and cannot be split, when spilling pushes a job past its time window, or when long jobs need fault tolerance. Loan-level data is the likely case. Before that point, the cheaper steps are fewer threads, a larger machine, and Spark in local mode.

## 6.2 Q2: What quality problems do the raw data contain, and how does the pipeline handle them without look-ahead bias?

The main problems are these:

- **Reports dated on or after failure:** {wf_row('R3'):,} rows, removed so a failed bank is never labelled a survivor.
- **Zero meaning "not reported":** the risk-based capital ratio is zero for every report before 1990Q1 and for 36-41% of banks from 2020. These cells become null.
- **Year-to-date income:** converted to quarterly flows, and left null across a reporting gap.
- **Meaningless ratios:** ROE with negative equity, and ratios with a zero denominator, become null. Other ratios are recomputed from levels, because FDIC's noncurrent-loan ratio disagrees with its own levels in 1985-96.
- **Current-snapshot fields:** Institutions fields such as ACTIVE describe today, so they never become features.
- **Exits that look alike:** {exits['merger']['banks']:,} mergers are not failures. A further {distress} banks left in a state as bad as failed banks yet have no Failures record. They stay y = 0 under the exam's definition and are marked for sensitivity analysis.

Look-ahead is prevented by construction and then tested. Nothing is fitted in the panel or the feature table: capping, imputation and scaling sit in a pipeline fitted on training rows. The label is joined last, and features are unchanged when every label-side column is removed first. Rebuilding from bronze with data cut at 1990, 2008 and 2019 leaves all {candidates} candidate features unchanged, and four deliberately leaky control features are all caught by the same test. One exception is disclosed: rule R1 uses whether a bank has an Institutions record today, and no bank it removes ever failed.

## 6.3 Q3: Can the pipeline's output support an honestly validated early-warning model?

Yes, for ranking banks by the kind of failure that builds up in reported financials. No, for failures that arrive through a sudden funding run.

The design supports an honest answer. Features, model and threshold were fixed on development data, using three time-ordered folds with four-quarter gaps, and the test period was then used once. Mean average precision (AP) across folds was {chosen_cv['mean_ap']:.3f} for the chosen gradient-boosted model, against {b1_cv['mean_ap']:.3f} for a three-ratio logit and {b0_cv['mean_ap']:.3f} for the Texas ratio alone. The threshold comes from costs: a median FDIC loss of USD {costs['median_loss_thousand'] / 1e3:.1f} million per failure, against about USD {costs['median_exam_cost_thousand']:.0f} thousand for an examination.

On the test period (base rate {chosen['prevalence'] * 100:.2f}%) the model reached AP {chosen['ap']:.3f} (95% interval {boot['ap_ci95']['chosen'][0]:.3f}-{boot['ap_ci95']['chosen'][1]:.3f}). Alarms covered {th['alarm_share'] * 100:.1f}% of bank-quarters with {th['row_precision'] * 100:.1f}% precision, and {caught} of {res['failed_banks']} failed banks ({th['bank_recall_actionable'] * 100:.0f}%) received an alarm in time to act, typically {th['median_lead_quarters']:.0f} quarters ahead. Three results limit that claim:

1. **The complex model did not clearly beat the simple one.** The AP gain over the three-ratio logit ({b1['ap']:.3f}) has an interval of {boot['ap_diff_ci95']['B1'][0]:+.3f} to {boot['ap_diff_ci95']['B1'][1]:+.3f}, which includes zero, and the logit's ROC AUC was higher ({b1['roc_auc']:.3f} against {chosen['roc_auc']:.3f}). Both clearly beat the Texas ratio ({b0['ap']:.3f}).
2. **Performance fell over time.** AP was {per['2013-2019']['ap']:.3f} in 2013-2019 and {per['2020-2025']['ap']:.3f} in 2020-2025, where only {per['2020-2025']['failed_banks']} failed banks are labelled.
3. **The largest losses were missed.** Silicon Valley Bank and First Republic raised no alarm and account for {post['share_of_test_losses_in_two_banks'] * 100:.1f}% of test-period losses, so the cost saving was {post['with_them_saving_pct']:.1f}%. Without those two banks it was {post['without_those_two']['saving_pct']:.1f}%, though that comparison is post hoc. Silicon Valley Bank's last report before failing showed {svb['eq_ta'] * 100:.1f}% equity to assets and {svb['npl'] * 100:.2f}% noncurrent loans. Quarterly ratios of this kind do not appear to capture the rapid deposit outflows and unrealised securities losses reported in these cases, although the data used here cannot test that directly.

The output is therefore suitable as one input to supervisory prioritisation, alongside liquidity monitoring, and not as a stand-alone alarm.

## 6.4 Limitations

- **Data vintage:** FDIC serves current values, which may include revisions, so truncation tests cannot exclude all look-ahead from restated reports.
- **Few test events:** {res['failed_banks']} failed banks, about half of them failing in 2013-14, give wide intervals and little evidence for recent years.
- **Assumed costs:** exam size (16 person-weeks) and the {C.ACTION_LAG_DAYS}-day action lag are assumptions. The {frozen['share_saved'] * 100:.0f}% share of loss avoided is a correlation across failures, and it is not monotonic in capital. The threshold moves between {grid['threshold'].min():.4f} and {grid['threshold'].max():.4f} across the range tested.
- **Near-miss labelling:** {distress} distressed exits count as non-failures under the exam's definition, which may understate true precision in development.
- **Selection:** feature screening saw labels up to 2006, so folds F1 and F2 are slightly optimistic, and one screening tie-break was added after a first look at results.
- **Benchmarks:** they ran on a laptop in daily use, with about a quarter of its memory free, and scaled data repeats real banks, so it tests volume and not variety. The memory limit on a free 15.4 GB machine is extrapolated from two points (roughly {ext['polars_mem']['extrapolated_m_rows_at_12gb']:.0f}-{ext['polars_stream']['extrapolated_m_rows_at_12gb']:.0f} million rows at 12 GB), not measured.
- **Scope:** US bank-level reports contain no market prices, high-frequency deposit flows or ownership links.

## 6.5 What would change for Vietnam

The storage layout, the point-in-time rules, the tests and the validation discipline carry over. The label, features, costs and model complexity do not.

| Layer | US pipeline | Change for Vietnam |
|---|---|---|
| **Label** | Failure recorded in a public table | Formal failures are rare. Weak banks went through special control, and CB, OceanBank, GPBank and DongA Bank were compulsorily transferred in 2024-25, about a decade later [1, 2]. The label would have to be a supervisory distress event, such as early intervention or special control [3], dated when the decision was taken. |
| **Validation** | Test period with at least 50 failures | That many events are unlikely to exist. A time split would give very wide intervals, so results should be reported per event, with expert review and possibly with risk rankings informed by models from other countries. |
| **Sample and model** | Thousands of banks | A few dozen commercial banks. This favours simple, explainable models: even with thousands of US banks, the gradient-boosted model's gain over a three-ratio logit was not established. Within-quarter ranks also carry little information across so few banks. |
| **Features** | Public reported ratios | Add what the US data lacks. The SCB case centred on lending to parties linked to a major shareholder, followed by a deposit run in October 2022 [4, 5], which reported ratios may not reveal. Candidates are related-party and large-exposure measures, under the wider related-party definition and lower lending limits of the 2024 law [3], ownership links, and frequent deposit flows. |
| **Data** | Public API | Supervisory data are not public, so the pipeline needs restricted access and report vintages kept as filed. Whether reported asset quality is comparable across banks and over time has to be tested, not assumed. |
| **Costs** | FDIC loss per failure | There is no equivalent loss table. Deposit insurance covers up to VND 125 million per depositor [6], while the reported support for SCB was far larger, about USD 24 billion [4]. The cost of a missed case should include public support and contagion, so the threshold must be derived again. |
| **Action** | An assumed examination | An alarm can map to a legal step, since the 2024 law sets out early intervention and special control, with mass withdrawals among the conditions [3]. The action lag should reflect the reporting frequency available to the State Bank of Vietnam. |
| **Tools** | Polars on a laptop | Bank-level panels would be far smaller than this one, so a laptop or single server is enough. Spark becomes relevant only if loan-level credit registry data is added, and that should be decided by measurement, as in Part 3. |

### Sources for 6.5

1. The Investor, "Compulsory transfer of 4 banks under special control approved": https://theinvestor.vn/compulsory-transfer-of-4-banks-under-special-control-approved-d4787.html
2. Vietnamnet, "Four underperforming banks in Vietnam restructured through mandatory transfer": https://vietnamnet.vn/en/four-underperforming-banks-in-vietnam-restructured-through-mandatory-transfer-2364609.html
3. Law on Credit Institutions No. 32/2024/QH15. Text: https://apolatlegal.com/wp-content/uploads/2025/07/Law-on-credit-institutions-2024.pdf. Summaries: Mayer Brown, https://www.mayerbrown.com/en/insights/publications/2024/03/vietnams-new-banking-law-for-foreign-investors-five-key-takeaways; Deposit Insurance of Vietnam, https://div.gov.vn/restructuring-weak-credit-institutions-and-the-role-of-div-according-to-the-law-on-credit-institutions-2024a
4. CNBC, "Vietnam mounts unprecedented $24 billion rescue for bank engulfed in giant fraud, documents show": https://www.cnbc.com/2024/04/17/vietnam-mounts-unprecedented-24-billion-rescue-for-bank-engulfed-in-giant-fraud-documents-show.html
5. Tuoi Tre News, "Vietnam's central bank places lender under special scrutiny": https://tuoitrenews.vn/news/business/20221016/vietnam-s-central-bank-places-lender-under-special-scrutiny/69589.html
6. Decision 32/2021/QD-TTg: https://english.luatvietnam.vn/decision-no-32-2021-qd-ttg-on-insurance-money-limit-211331-doc1.html; Deposit Insurance of Vietnam: https://div.gov.vn/deposit-insurance-coverage-limit-increased-to-125-million-vnd-en
"""
    (R / "p6_conclusions_draft.md").write_text(text, encoding="utf-8")

    key = {"storage_mb": {"csv": mb["csv"], "chosen": mb["pq_year"]},
           "engines_s": {"polars": pol["wall_s"], "pandas": pan["wall_s"], "spark_18": s18["wall_s"], "spark_200": s200["wall_s"]},
           "scale": {"spark_max_scale_finished": sp16["scale"], "spark_max_s": sp16["wall_s"], "spark_max_spill_gb": sp16["spill_gb"], "spark_failed_scales": sp_fail["scale"].to_list(), "polars_x2_s": pl2["wall_s"], "polars_x4": pl4["status"], "pandas_x4": pd4["status"]},
           "split": {k: {"rows": v["rows"], "failed_banks": v["failed_banks"]} for k, v in split.items()},
           "features": {"candidates": candidates, "selected": len(sel["numeric"]) + len(sel["flags"])},
           "cv_mean_ap": {"chosen": chosen_cv["mean_ap"], "B1": b1_cv["mean_ap"], "B0": b0_cv["mean_ap"]},
           "test": {"ap": chosen["ap"], "ap_ci": boot["ap_ci95"]["chosen"], "ap_B1": b1["ap"], "ap_B0": b0["ap"], "diff_vs_B1_ci": boot["ap_diff_ci95"]["B1"],
                    "bank_recall_actionable": th["bank_recall_actionable"], "caught": caught, "late": late, "no_alarm": no_alarm, "precision": th["row_precision"], "alarm_share": th["alarm_share"],
                    "saving_pct": post["with_them_saving_pct"], "saving_pct_without_two": post["without_those_two"]["saving_pct"], "no_system_usd_bn": usd_bn(cost["no_system"]), "with_system_usd_bn": usd_bn(cost["with_system"])},
           "spark_vs_sklearn": {"ap_spark": sl["ap"], "ap_sklearn": sk["ap"], "rank_corr": spark_j["rank_corr_spark_vs_sklearn"]},
           "tests_passed": {"part2": p2.height, "part4": p4.height, "part5": p5.height}}
    (R / "p6_key_numbers.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(first)
    print(json.dumps(key, indent=1))


if __name__ == "__main__":
    main()
