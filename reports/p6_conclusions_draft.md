# Part 6. Conclusions

_Draft generated from the result files by `src/p6_conclusions.py`. Rewrite it in your own words and check it against your own understanding before submitting._

## 6.1 Q1: How should four decades of bank reports be stored and processed on a laptop, and when would a Spark cluster be needed?

Store them as typed, ZSTD-compressed Parquet, partitioned by year and sorted by bank and date. This layout takes 206.3 MB against 490.2 MB for CSV, and a query that needs a few columns or one year reads only that part of the data. Process them with lazy Polars. The core pipeline over 1.67 million bank-quarters took 1.8 s and 1.9 GB in Polars, 14.5 s and 1.3 GB in pandas, and 78-106 s and 3.1-3.3 GB in local PySpark. All three gave the same answer on 55 million cells. The only differences were 5 pandas values, within 4e-09, which direct recomputation attributed to rounding in pandas' rolling variance.

This dataset does not need a cluster. The panel gained 17,612 rows in 2025, so it would take about 95 years to double. Spark was not faster on one machine: it used about 27 times the CPU of Polars and wrote 1.1 GB of shuffle data from a 206 MB input.

Spark's real advantage was bounded memory. With about 3.6 GB free on the test laptop, Polars and pandas were stopped by the memory guard at four copies of the data (6.7 million rows), while Spark with a 2 GB heap finished sixteen copies (26.7 million rows) in 346 s, peaking at 2.7 GB and spilling 14.4 GB to disk. A cluster is justified when the working set exceeds the memory of the largest affordable single machine and cannot be split, when spilling pushes a job past its time window, or when long jobs need fault tolerance. Loan-level data is the likely case. Before that point, the cheaper steps are fewer threads, a larger machine, and Spark in local mode.

## 6.2 Q2: What quality problems do the raw data contain, and how does the pipeline handle them without look-ahead bias?

The main problems are these:

- **Reports dated on or after failure:** 1,095 rows, removed so a failed bank is never labelled a survivor.
- **Zero meaning "not reported":** the risk-based capital ratio is zero for every report before 1990Q1 and for 36-41% of banks from 2020. These cells become null.
- **Year-to-date income:** converted to quarterly flows, and left null across a reporting gap.
- **Meaningless ratios:** ROE with negative equity, and ratios with a zero denominator, become null. Other ratios are recomputed from levels, because FDIC's noncurrent-loan ratio disagrees with its own levels in 1985-96.
- **Current-snapshot fields:** Institutions fields such as ACTIVE describe today, so they never become features.
- **Exits that look alike:** 15,339 mergers are not failures. A further 865 banks left in a state as bad as failed banks yet have no Failures record. They stay y = 0 under the exam's definition and are marked for sensitivity analysis.

Look-ahead is prevented by construction and then tested. Nothing is fitted in the panel or the feature table: capping, imputation and scaling sit in a pipeline fitted on training rows. The label is joined last, and features are unchanged when every label-side column is removed first. Rebuilding from bronze with data cut at 1990, 2008 and 2019 leaves all 42 candidate features unchanged, and four deliberately leaky control features are all caught by the same test. One exception is disclosed: rule R1 uses whether a bank has an Institutions record today, and no bank it removes ever failed.

## 6.3 Q3: Can the pipeline's output support an honestly validated early-warning model?

Yes, for ranking banks by the kind of failure that builds up in reported financials. No, for failures that arrive through a sudden funding run.

The design supports an honest answer. Features, model and threshold were fixed on development data, using three time-ordered folds with four-quarter gaps, and the test period was then used once. Mean average precision (AP) across folds was 0.453 for the chosen gradient-boosted model, against 0.332 for a three-ratio logit and 0.296 for the Texas ratio alone. The threshold comes from costs: a median FDIC loss of USD 11.7 million per failure, against about USD 22 thousand for an examination.

On the test period (base rate 0.09%) the model reached AP 0.345 (95% interval 0.251-0.429). Alarms covered 1.1% of bank-quarters with 7.2% precision, and 59 of 78 failed banks (76%) received an alarm in time to act, typically 4 quarters ahead. Three results limit that claim:

1. **The complex model did not clearly beat the simple one.** The AP gain over the three-ratio logit (0.318) has an interval of -0.042 to +0.086, which includes zero, and the logit's ROC AUC was higher (0.960 against 0.938). Both clearly beat the Texas ratio (0.096).
2. **Performance fell over time.** AP was 0.424 in 2013-2019 and 0.091 in 2020-2025, where only 14 failed banks are labelled.
3. **The largest losses were missed.** Silicon Valley Bank and First Republic raised no alarm and account for 88.5% of test-period losses, so the cost saving was 2.1%. Without those two banks it was 18.1%, though that comparison is post hoc. Silicon Valley Bank's last report before failing showed 7.4% equity to assets and 0.19% noncurrent loans. Quarterly ratios of this kind do not appear to capture the rapid deposit outflows and unrealised securities losses reported in these cases, although the data used here cannot test that directly.

The output is therefore suitable as one input to supervisory prioritisation, alongside liquidity monitoring, and not as a stand-alone alarm.

## 6.4 Limitations

- **Data vintage:** FDIC serves current values, which may include revisions, so truncation tests cannot exclude all look-ahead from restated reports.
- **Few test events:** 78 failed banks, about half of them failing in 2013-14, give wide intervals and little evidence for recent years.
- **Assumed costs:** exam size (16 person-weeks) and the 45-day action lag are assumptions. The 23% share of loss avoided is a correlation across failures, and it is not monotonic in capital. The threshold moves between 0.0046 and 0.0407 across the range tested.
- **Near-miss labelling:** 865 distressed exits count as non-failures under the exam's definition, which may understate true precision in development.
- **Selection:** feature screening saw labels up to 2006, so folds F1 and F2 are slightly optimistic, and one screening tie-break was added after a first look at results.
- **Benchmarks:** they ran on a laptop in daily use, with about a quarter of its memory free, and scaled data repeats real banks, so it tests volume and not variety. The memory limit on a free 15.4 GB machine is extrapolated from two points (roughly 16-23 million rows at 12 GB), not measured.
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
