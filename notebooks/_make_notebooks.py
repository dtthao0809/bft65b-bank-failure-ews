"""
Builds the six walkthrough notebooks (one per Part) and executes them, so they are committed with outputs.

The notebooks do not repeat the heavy work. They rerun the fast verification scripts live, show the
result files the pipeline wrote to reports/, and run a few small Polars queries on the stored layers.
Run from the project root:  python notebooks/_make_notebooks.py   (about 6 minutes)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = '''import json, os, subprocess, sys
from pathlib import Path

import pandas as pd
import polars as pl
from IPython.display import Image, Markdown, display

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
import config as C

pd.set_option("display.max_columns", 30, "display.width", 200, "display.max_colwidth", 80)
R = ROOT / "reports"
print("project root:", ROOT, "| python", sys.version.split()[0], "| polars", pl.__version__, "| seed", C.SEED)


def run(script, *args, tail=25):
    """Run one pipeline script exactly as the README does, and show the end of its output."""
    t = subprocess.run([sys.executable, script, *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUTF8": "1"})
    lines = (t.stdout + t.stderr).strip().splitlines()
    print("\\n".join(lines[-tail:]))
    print(f"\\n>>> {script} {' '.join(args)} exited with {t.returncode}")
    assert t.returncode == 0, "step failed"


def csv(name, n=None):
    df = pd.read_csv(R / name)
    display(df if n is None else df.head(n))


def fig(name, width=820):
    display(Image(filename=str(R / name), width=width))'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


PARTS = {
"part1_acquisition_storage": ("Part 1 - Acquisition and storage", [
    md("""
**Goal.** Download the full FDIC scope reproducibly, prove it is complete, and store it in a layout that
beats raw CSV on measured size and query time.

| Step | Script | Run here? |
|---|---|---|
| Download (resumable, 15-60 min) | `fdic_download_full.py` | no, already done |
| Completeness, control totals, SHA-256, schema drift | `src/p1_verify.py` | **yes, live** |
| Typed bronze layer, reconciled against raw | `src/p1_build_bronze.py` | **yes, live** |
| Storage layouts benchmark (about 4 min, 3 GB scratch) | `src/p1_storage_benchmark.py` | no, results shown |
"""),
    md("The second, independent download (`data_raw/fdic`, starter field list) is compared when it is present."),
    code("run('src/p1_verify.py', *(['--compare', 'data_raw/fdic'] if (ROOT / 'data_raw' / 'fdic').exists() else []))"),
    code("run('src/p1_build_bronze.py')"),
    md("### Completeness against the API control totals"),
    code("print(json.dumps(json.loads((R / 'p1_completeness.json').read_text()), indent=1)[:1500])"),
    code("csv('p1_control_totals.csv', 10)"),
    md("### The bronze layer, read lazily with Polars (only the needed columns and partitions are touched)"),
    code("""
lf = pl.scan_parquet(C.DATA_BRONZE / "financials" / "**/*.parquet", hive_partitioning=True)
print(lf.select(pl.len()).collect())
lf.group_by("YEAR").agg(pl.len().alias("rows"), pl.col("CERT").n_unique().alias("banks")).sort("YEAR").collect().head(8)
"""),
    md("### Storage benchmark: CSV against the chosen ZSTD-3 Parquet layout"),
    code("display(Markdown((R / 'p1_storage_benchmark.md').read_text(encoding='utf-8')))"),
    code("fig('fig_p1_storage.png'); fig('fig_p1_coverage.png')"),
]),
"part2_quality_panel": ("Part 2 - Data quality and the bank-quarter panel", [
    md("""
**Goal.** An automated data-quality report, a rule for every problem found, and a clean panel that
separates failures from other exits and is point-in-time correct.
"""),
    code("run('src/p2_build_panel.py')"),
    code("run('src/p2_quality.py', tail=12)"),
    md("### Cleaning waterfall: what each rule changed"),
    code("csv('p2_waterfall.csv')"),
    md("### Issue register (problem, evidence, handling)"),
    code("csv('p2_issue_register.csv')"),
    md("### Exit classes: failure against merger and other exits"),
    code("csv('p2_exit_health.csv')"),
    md("### Automated tests of the panel and the label (the script exits non-zero on any failure)"),
    code("run('src/p2_checks.py')"),
    code("csv('p2_checks.csv')"),
    code("fig('fig_p2_missing.png'); fig('fig_p2_ytd.png'); fig('fig_p2_capital_zero.png'); fig('fig_p2_failures.png')"),
]),
"part3_scalable_processing": ("Part 3 - Scalable processing: Polars, pandas and Spark", [
    md("""
**Goal.** The core transformations in Polars, how the engine runs them, a fair benchmark against pandas,
the same steps in PySpark with matching results, and an argument from measurements about when a cluster
is needed. The benchmarks took about an hour in total, so they are not rerun here; the comparison of
results is.
"""),
    md("### Engines on the full panel (median wall time, CPU and peak memory)"),
    code("csv('p3_engines.csv')"),
    code("fig('fig_p3_engines.png')"),
    md("### Do pandas and Spark give the same numbers as Polars? (live)"),
    code("run('src/p3_compare.py')"),
    code("print(json.dumps(json.loads((R / 'p3_engine_reconcile.json').read_text()), indent=1)[:1500])"),
    md("### How Polars executes it: the optimised plan of a small query (live)"),
    code("""

lf = (pl.scan_parquet(C.DATA_SILVER / "panel" / "**/*.parquet", hive_partitioning=True)
        .filter(pl.col("REPDTE") >= pl.date(2008, 1, 1))
        .group_by("STNAME").agg(pl.col("ROA").mean()))
print(lf.explain())
"""),
    code("print((R / 'p3_plan_polars.txt').read_text(encoding='utf-8')[:3000])"),
    md("### Spark plan summary and stages"),
    code("print(json.dumps(json.loads((R / 'p3_plan_summary.json').read_text()), indent=1))"),
    code("csv('p3_spark_stages.csv')"),
    md("### Threads and data scale: where a laptop stops"),
    code("csv('p3_threads.csv'); csv('p3_scale.csv')"),
    code("fig('fig_p3_threads.png'); fig('fig_p3_scale.png')"),
    code("print(json.dumps(json.loads((R / 'p3_cluster_case.json').read_text()), indent=1))"),
]),
"part4_feature_pipeline": ("Part 4 - Leak-free feature pipeline", [
    md("""
**Goal.** Features that proxy CAMELS risk, built only from information available at each quarter, joined
with the label, and preprocessing fitted on training rows only.
"""),
    code("run('src/p4_features.py')"),
    md("### Time split (development, gap, test)"),
    code("csv('p4_split.csv')"),
    code("fig('fig_p4_split.png')"),
    md("### Feature dictionary and screening on the training window only"),
    code("csv('p4_feature_dictionary.csv')"),
    code("print(json.dumps(json.loads((R / 'p4_selected_features.json').read_text()), indent=1))"),
    code("csv('p4_feature_screen.csv', 15)"),
    md("### Leakage tests, with deliberate controls (live, about 1 min)"),
    code("run('src/p4_checks.py')"),
    code("csv('p4_checks.csv')"),
    code("fig('fig_p4_signal.png'); fig('fig_p4_event_study.png')"),
]),
"part5_modelling_validation": ("Part 5 - Modelling and validation", [
    md("""
**Goal.** Baselines against our model under rolling time-ordered validation, metrics for rare events,
a threshold from estimated costs, a short Spark MLlib demo, and a single evaluation on the test period.

Model development (`p5_develop.py`, about 6 h) and the dry run (about 2.6 h) are not rerun here. The test
period was evaluated once; `p5_test.py --reproduce` reruns it and fails unless it equals the record.
"""),
    md("### Cost of a false alarm and of a missed failure, from development data"),
    code("print(json.dumps(json.loads((R / 'p5_costs.json').read_text()), indent=1))"),
    md("### Rolling validation: average precision per fold"),
    code("csv('p5_cv_summary.csv')"),
    code("fig('fig_p5_cv.png'); fig('fig_p5_threshold.png')"),
    md("### What was frozen before the test"),
    code("print(json.dumps(json.loads((R / 'p5_frozen.json').read_text()), indent=1)[:1200])"),
    md("### Spark MLlib demo against scikit-learn on the same fold"),
    code("csv('p5_spark_mllib.csv')"),
    md("### The single test evaluation, reproduced and compared with the record (live, about 3 min)"),
    code("run('src/p5_test.py', '--reproduce')"),
    code("csv('p5_test_metrics.csv')"),
    code("fig('fig_p5_test_pr.png'); fig('fig_p5_test_banks.png')"),
    md("### Validation design tests (live)"),
    code("run('src/p5_checks.py')"),
    code("csv('p5_checks.csv')"),
]),
"part6_conclusions": ("Part 6 - Conclusions and reproducibility", [
    md("""
**Goal.** Answer Q1-Q3 from the measured results, state the limitations, and show that the pipeline reruns.
`p6_conclusions.py` stops if any test suite has a failure.
"""),
    code("run('src/p6_conclusions.py')"),
    code("print(json.dumps(json.loads((R / 'p6_key_numbers.json').read_text()), indent=1))"),
    code("display(Markdown((R / 'p6_first_page.md').read_text(encoding='utf-8')))"),
    md("### Clean rerun: exit codes and time of every step"),
    code("csv('repro_status.csv')"),
    code("df = pd.read_csv(R / 'repro_check.csv'); display(df.groupby(['kind', 'same']).size().rename('files').reset_index()); df.head(15)"),
]),
}


def build(stem, title, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}
    nb.cells = [md(f"# {title}\n\nBFT65B final report, walkthrough notebook. Every number below is read from "
                   "`reports/` or recomputed live by the same scripts the README lists. Seed: `config.SEED = 42`."),
                code(SETUP), *cells]
    return nb


if __name__ == "__main__":
    only = sys.argv[1:]
    for i, (stem, (title, cells)) in enumerate(PARTS.items(), 1):
        if only and stem not in only:
            continue
        path = NB / f"{i:02d}_{stem}.ipynb"
        nb = build(stem, title, cells)
        print(f"executing {path.name} ...", flush=True)
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.write(nb, path)
    print("done")
