# A Big-Data Pipeline for Bank Failure Early Warning (BFT65B)

Author: Do Thi Thao (Đỗ Thị Thảo), student ID 11231011, Financial Technology, Institute of Banking and Finance

Course: BFT65B Big Data Analytics in Finance, National Economics University

Repository: https://github.com/dtthao0809/bft65b-bank-failure-ews

Scope: FDIC Financials 1984Q1 to 2026Q2, plus the complete Failures and Institutions tables.

## Pipeline layers

| Layer | Folder | Content | Script |
|---|---|---|---|
| 0 Raw | `data_raw/fdic_full/` | API output as received, one Parquet file per quarter, with a manifest | `fdic_download_full.py` |
| 1 Bronze | `data/bronze/` | Typed, schema-enforced, ZSTD-3 Parquet, partitioned by `YEAR`, sorted by `(CERT, REPDTE)` | `src/p1_build_bronze.py` |
| 2 Silver | `data/silver/` | Clean point-in-time bank-quarter panel, exits classified, label `y` | `src/p2_build_panel.py` |
| 3 Gold | `data/gold/` | Leak-free features joined with the label; time split | `src/p4_features.py` |

## Reproduce

Run every command from the repository root. Windows PowerShell shown; on Linux or macOS activate with
`source .venv/bin/activate`.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Part 1
python fdic_download_full.py                 # about 15-60 min; resumable: rerun if it stops
python src/p1_verify.py                      # completeness, control totals, SHA-256, schema drift
python src/p1_build_bronze.py                # typed bronze layer, reconciled against raw
python src/p1_storage_benchmark.py           # layouts compared; uses about 3 GB of scratch space

# Part 2
python src/p2_build_panel.py                 # clean panel, exit classes, label y
python src/p2_quality.py                     # automated data-quality report
python src/p2_checks.py                      # point-in-time and label tests; exits non-zero on failure

# Part 3 (needs a JDK for Spark, see "Java for Spark" below)
python src/p3_benchmark.py --suite engines   # all engines on the real data, about 25 min
python src/p3_compare.py                     # every cell of pandas and Spark against Polars and silver
python src/p3_benchmark.py --suite threads   # 1 to 18 threads, about 25 min
python src/p3_benchmark.py --suite spark_tune                  # Spark settings, about 20 min
python src/p3_benchmark.py --suite scale                       # data copied x2 to x16; uses about 6 GB of disk
python src/p3_benchmark.py --suite scale --keep-prior --scales 1,2,4,8,16 --engines spark@6
python src/p3_plans.py                       # execution plans of Polars and Spark
python src/p3_report.py                      # tables and figures of Part 3

# Part 4
python src/p4_features.py                    # gold feature table, joined with the label
python src/p4_screen.py                      # chooses features from the training window only
python src/p4_pipeline.py                    # preprocessing fitted on training rows; drift on validation
python src/p4_checks.py                      # leakage tests with deliberate controls; exits non-zero on failure
python src/p4_report.py                      # figures of Part 4

# Part 5
python src/p5_costs.py                       # false-alarm and missed-failure costs, development data only
python src/p5_develop.py                     # folds, tuning, threshold; writes reports/p5_frozen.json
python src/p5_spark.py                       # short Spark MLlib demo on one fold, compared with scikit-learn
python src/p5_test.py --dry-run              # the test script on a development window
python src/p5_test.py --reproduce            # on a clone: reruns the frozen evaluation, must equal the recorded one
python src/p5_report.py                      # figures, from stored scores
python src/p5_checks.py                      # validation design tests; exits non-zero on failure
```

`python src/p6_conclusions.py` then writes the first-page facts and a draft of the conclusions from the
result files, and stops if any test suite has a failure.
`python report/build_report.py . report/BFT65B_report.docx` then builds the report from the files in `reports/`
(needs `pip install python-docx==1.2.0`); the PDF is exported from that DOCX.

### Checking a rerun

`python src/repro_compare.py <folder of a clean rerun>` compares every report and the bronze, silver and gold
Parquet files of a rerun with the committed ones (`reports/repro_check.csv`). Deterministic results must match
to 1e-9; timings, execution plans and machine state are listed but expected to differ.

This was done on 2026-11-08 in VS Code, in a clean clone of the project with a new environment from
`requirements.txt` and the same raw download (hashes as in `reports/p1_raw_sha256.csv`). Every step completed, and
`--reproduce` gave the recorded test results exactly. All 61 deterministic results matched, including every bronze,
silver and gold Parquet file byte for byte. The benchmarks ran with fewer repeats. One measurement did not repeat:
Spark with a 2 GB heap ran out of execution memory at sixteen copies of the data (it finished sixteen in the recorded
run and eight in both reruns), so `p6_conclusions.py` reports the largest scale Spark completed. On an idle laptop the
rerun took about 3 hours; the first run, on a busy one, took about 10 hours, mostly `p5_develop.py`.

The published repository was checked again on 2026-11-19, on a fresh clone under Linux: the pinned requirements
installed into a new environment, every script compiled, and `reports/p5_test_results.json` matched the hash in its
lock. This check found that the lock failed on checkouts with LF line endings, because the hash had been taken on
Windows (CRLF); `.gitattributes` now checks that file out with CRLF on every system, and the results were not touched.
The code edited after the 2026-11-08 rerun was compared statement by statement (Python syntax trees, ignoring comments
and docstrings) with the version that was rerun. No step that builds data, features, models, the threshold or the test
results changed; the only code changes are the Part 6 wording about the largest scale Spark finished, an optional
comparison in the Part 1 notebook, and command-line help text.
On the same day the fast checks were rerun on the laptop, on the existing data and at commit e53b803:
`p1_verify --offline`, `p2_checks`, `p3_compare`, `p4_checks`, `p5_test --reproduce`, `p5_checks` and
`p6_conclusions`. All of them passed (Part 2 20/20, Part 4 31/31, Part 5 16/16; Polars, Spark and pandas agreed as
before), and `--reproduce` gave the recorded test results exactly. They changed no committed file except
`reports/p1_completeness.json`, which `--offline` rewrites without the API comparison; the committed version was
restored. The long steps (download, storage and engine benchmarks, model development) were not rerun on 2026-11-19.
Afterwards `src/p6_conclusions.py` was changed to write the author's name, student ID and repository URL into
`reports/p6_first_page.md` and to drop a leftover note to the author from the head of `reports/p6_conclusions_draft.md`;
it was rerun, and no number in either file changed. `notebooks/06_part6_conclusions.ipynb` was executed again with
`python notebooks/_make_notebooks.py part6_conclusions` so that it shows the new first page; it now carries 2026-11-19
timestamps, the other five notebooks keep those of 2026-11-08.

### The test period

The test period was evaluated once, by `python src/p5_test.py`, after `reports/p5_frozen.json` was written. The
results are committed in `reports/p5_test_results.json` and tied by hash to `reports/p5_test_lock.json`.
That command now refuses to run. `--reproduce` reruns the same evaluation on a clone and fails unless the
numbers equal the recorded ones; it cannot overwrite them and informs no decision.

Results depend on the data vintage, because FDIC updates its tables. To match the recorded numbers exactly,
use a download with the hashes in `reports/p1_raw_sha256.csv`. With a later download, `--reproduce` will
report differences, which is expected.

## Walkthrough notebooks and VS Code

`notebooks/01_...` to `notebooks/06_...` show each Part with its outputs saved: they rerun the fast
verification scripts live (`p1_verify`, `p2_checks`, `p3_compare`, `p4_checks`, `p5_test --reproduce`,
`p5_checks`, `p6_conclusions`) and display the tables and figures that the long steps wrote to `reports/`.
They do not repeat the multi-hour benchmarks or model development.

```bash
pip install -r requirements-notebooks.txt
python notebooks/_make_notebooks.py          # rebuilds and executes all six, about 6 minutes
```

In VS Code, open the repository folder with the Python and Jupyter extensions, choose the `.venv` interpreter,
and run the README commands in the integrated terminal from the repository root.

## Java for Spark

Part 3 and `src/p5_spark.py` need a JDK 17 or 21 (for example Eclipse Temurin). Either set `JAVA_HOME`, or unpack
the JDK into `tools/jdk-*/` (git-ignored), where `src/p3_spark_session.py` finds it. On Windows a full JDK is
required, not only a JRE: the session compiles `src/spark_win/NoNativeLocalFS.java` with `javac` once, so that Spark
can list local folders without `hadoop.dll`. On Linux, macOS and Colab that class is not used.

## Data source

FDIC BankFind Suite API, https://api.fdic.gov/banks (no key needed). Endpoints: `/financials`, `/failures`,
`/institutions`, and `/summary` for control totals. Field dictionary: https://api.fdic.gov/banks/docs/.

## References and acknowledgements

The report's reference list, numbered as in the report:

1. FDIC (2026). BankFind Suite API and field dictionary. https://api.fdic.gov/banks/docs/ (accessed 2026-11-07).
2. BFT65B course team (2026). Starter script `fdic_download_starter.py`, reused by `fdic_download_full.py`.
3. Cole, R. A., and White, L. J. (2012). Déjà vu all over again: The causes of U.S. commercial bank failures this
   time around. *Journal of Financial Services Research*, 42, 5-29.
4. Software documentation: Polars (https://docs.pola.rs), Apache Arrow and Parquet, pandas, Apache Spark
   (https://spark.apache.org/docs/latest/) and delta-rs.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
6. Saito, T., and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when
   evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
7. The Investor. Compulsory transfer of 4 banks under special control approved.
   https://theinvestor.vn/compulsory-transfer-of-4-banks-under-special-control-approved-d4787.html
8. Vietnamnet. Four underperforming banks in Vietnam restructured through mandatory transfer.
   https://vietnamnet.vn/en/four-underperforming-banks-in-vietnam-restructured-through-mandatory-transfer-2364609.html
9. Law on Credit Institutions No. 32/2024/QH15; summaries by Mayer Brown (2024) and Deposit Insurance of Vietnam.
   https://div.gov.vn/restructuring-weak-credit-institutions-and-the-role-of-div-according-to-the-law-on-credit-institutions-2024a
10. CNBC (2024, April 17). Vietnam mounts unprecedented $24 billion rescue for bank engulfed in giant fraud, documents show.
    https://www.cnbc.com/2024/04/17/vietnam-mounts-unprecedented-24-billion-rescue-for-bank-engulfed-in-giant-fraud-documents-show.html
11. Tuoi Tre News (2022, October 16). Vietnam's central bank places lender under special scrutiny.
    https://tuoitrenews.vn/news/business/20221016/vietnam-s-central-bank-places-lender-under-special-scrutiny/69589.html
12. Decision No. 32/2021/QD-TTg on the deposit insurance limit.
    https://english.luatvietnam.vn/decision-no-32-2021-qd-ttg-on-insurance-money-limit-211331-doc1.html

AI assistance: Claude (Anthropic), through Claude Code and the Claude app, was used for design discussion, writing
and debugging code, and drafting text. The author reviewed the code, reran the pipeline from this README in VS Code
on 2026-11-08, checked the published repository again on 2026-11-19, and is responsible for all text in the report.
