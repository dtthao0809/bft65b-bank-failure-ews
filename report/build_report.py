# Optional: needs `pip install python-docx` (not in requirements.txt). Usage: python report/build_report.py . report/BFT65B_report.docx
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(sys.argv[1])
R = ROOT / "reports"
OUT = Path(sys.argv[2])

BODY, SMALL, TINY = 10, 8.5, 8
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
MONO = "Consolas"


def J(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def C(name):
    with open(R / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def n0(x):
    return f"{float(x):,.0f}"


doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
TEXT_W = 17.0

st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(BODY)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.08
for lvl, size, before in ((1, 13.5, 10), (2, 11, 7), (3, 10, 5)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.bold = "Calibri", Pt(size), True
    h.font.color.rgb = ACCENT
    h.element.rPr.rFonts.set(qn("w:asciiTheme"), "")
    rf = h.element.rPr.rFonts
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), "Calibri")
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(before), Pt(3)
    h.paragraph_format.keep_with_next = True

# Footer: page number
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER


def field(par, code):
    r = par.add_run()
    for tag, text in (("begin", None), (None, code), ("separate", None), (None, "1"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag); r._r.append(e)
        elif text == code:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = code; r._r.append(e)
        else:
            e = OxmlElement("w:t"); e.text = text; r._r.append(e)
    r.font.size = Pt(SMALL)


fr = fp.add_run("BFT65B Final Report  ·  page "); fr.font.size = Pt(SMALL)
field(fp, "PAGE")

TOKEN = re.compile(r"(\*\*.+?\*\*|`.+?`|(?<![\w`])_[^_`]+_(?![\w]))")


def CODE(lines, size=6.8):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = t.rows[0].cells[0]
    cell.width = Cm(TEXT_W)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    shade(cell, "F3F5F7")
    par = cell.paragraphs[0]
    par.paragraph_format.space_after = Pt(0)
    par.paragraph_format.line_spacing = 1.0
    for i, line in enumerate(lines):
        r = par.add_run(line)
        r.font.name, r.font.size = MONO, Pt(size)
        r._r.rPr.rFonts.set(qn("w:hAnsi"), MONO)
        if line.startswith("--"):
            r.bold = True
        if i < len(lines) - 1:
            r.add_break()
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(0); sp.paragraph_format.line_spacing = 0.6


def runs(par, text, size=None, color=None):
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = MONO; r.font.size = Pt((size or BODY) - 1)
            r._r.rPr.rFonts.set(qn("w:hAnsi"), MONO)
        elif part.startswith("_") and len(part) > 2:
            r = par.add_run(part[1:-1]); r.italic = True
        else:
            r = par.add_run(part)
        if size and not part.startswith("`"):
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    return par


def P(text, size=None, align=None, after=None, keep=False):
    p = runs(doc.add_paragraph(), text, size)
    if align == "j":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def H(text, lvl=1):
    return doc.add_heading(text, level=lvl)


def B(items, size=None):
    for it in items:
        p = runs(doc.add_paragraph(style="List Bullet"), it, size)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.left_indent = Cm(0.55)


def N(items, size=None):
    for it in items:
        p = runs(doc.add_paragraph(style="List Number"), it, size)
        p.paragraph_format.space_after = Pt(1.5)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd"); s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), hexcolor)
    tcPr.append(s)


def borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge, val, sz in (("top", "single", 8), ("bottom", "single", 8), ("insideH", "single", 2), ("left", "nil", 0), ("right", "nil", 0), ("insideV", "nil", 0)):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), val)
        if sz:
            e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "8C99A6")
        b.append(e)
    tblPr.append(b)


TCOUNT = [0]
FCOUNT = [0]


def caption(kind, text):
    cnt = TCOUNT if kind == "Table" else FCOUNT
    cnt[0] += 1
    p = doc.add_paragraph()
    r = p.add_run(f"{kind} {cnt[0]}. "); r.bold = True; r.font.size = Pt(SMALL)
    runs(p, text, SMALL)
    p.paragraph_format.space_after = Pt(6 if kind == "Figure" else 2)
    p.paragraph_format.space_before = Pt(1 if kind == "Figure" else 4)
    if kind == "Table":
        p.paragraph_format.keep_with_next = True
    return cnt[0]


def T(header, rows, widths, cap=None, size=SMALL, align=None):
    if cap:
        caption("Table", cap)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders(t)
    for i, w in enumerate(widths):
        for row in t.rows:
            row.cells[i].width = Cm(w)
    # Write the widths into the table grid and fix the layout, so that Word and LibreOffice
    # render the same column widths (LibreOffice ignores cell widths without a grid).
    for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(Cm(w).twips)))
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(lay)
    for r_i, vals in enumerate([header] + rows):
        row = t.rows[r_i]
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        for c_i, v in enumerate(vals):
            cell = row.cells[c_i]
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
            runs(cell.paragraphs[0], str(v) if str(v) != "" else "–", size)
            if r_i == 0:
                for rr in cell.paragraphs[0].runs:
                    rr.bold = True
                shade(cell, "E8EDF2")
            a = (align or "l" * len(header))[c_i]
            if a == "r":
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif a == "c":
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in t.rows[:-1]:
        for c in row.cells:
            c.paragraphs[0].paragraph_format.keep_with_next = True
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(2)
    sp.paragraph_format.line_spacing = 0.6
    return t


def F(name, width, cap):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(R / name), width=Cm(width))
    caption("Figure", cap)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- data
comp = J("p1_completeness.json")
split = {r["_split"]: r for r in C("p4_split.csv")}
costs = J("p5_costs.json")
frozen = J("p5_frozen.json")
test = J("p5_test_results.json")
post = J("p5_test_posthoc.json")
cluster = J("p3_cluster_case.json")
recon = J("p3_engine_reconcile.json")
plan = J("p3_plan_summary.json")
mllib = J("p5_spark_mllib.json")
pfacts = J("p4_pipeline_facts.json")
sel = J("p4_selected_features.json")
at = test["at_threshold"]
boot = test["bootstrap"]

# ---------------------------------------------------------------- title page
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
runs(p, "NATIONAL ECONOMICS UNIVERSITY  ·  BFT65B BIG DATA ANALYTICS IN FINANCE", SMALL, ACCENT)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(2)
r = p.add_run("A Big-Data Pipeline for Bank Failure Early Warning"); r.bold = True; r.font.size = Pt(18); r.font.color.rgb = ACCENT
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
runs(p, "Final examination, individual research report", 11)

T(["Item", "Value"], [
    ["Name, student ID", "**Do Thi Thao (Đỗ Thị Thảo), 11231011**"],
    ["GitHub repository", "**https://github.com/dtthao0809/bft65b-bank-failure-ews**"],
    ["Download method", "FDIC BankFind Suite API (https://api.fdic.gov/banks) [1], `fdic_download_full.py`, which reuses the course starter's paging, retry and per-quarter resume logic [2]"],
    ["Download date", f"2026-11-07, {comp['download_log']['minutes']} minutes; 170 quarterly Parquet files plus a manifest with SHA-256 hashes"],
    ["Financials", f"{comp['financials']['panel_rows']:,} rows, {comp['financials']['distinct_banks']:,} banks, 170 quarters (1984Q1 to 2026Q2), 43 fields"],
    ["Failures, Institutions", f"{comp['failures']['rows']:,} rows and {comp['institutions']['rows']:,} rows, every field (both equal the API totals)"],
    ["Clean panel (silver)", "1,666,719 bank-quarters, 23,788 banks; 10,860 rows with y = 1"],
    ["Development period", f"1984Q1 to 2011Q4: {int(split['dev']['rows']):,} bank-quarters, **{int(split['dev']['failed_banks']):,} failed banks** labelled"],
    ["Gap (unused)", f"2012Q1 to 2012Q4: {int(split['gap']['rows']):,} bank-quarters"],
    ["Test period (used once)", f"2013Q1 to 2025Q2: {int(split['test']['rows']):,} bank-quarters, **{int(split['test']['failed_banks'])} failed banks** labelled"],
], [4.0, 13.0], size=9)

H("Summary", 2)
P(f"This report builds a reproducible pipeline from the FDIC Call Reports to a point-in-time panel, a leak-free "
  f"feature table and an honestly validated early-warning model, and uses it to answer the three questions of the brief.", align="j")
B([
    "**Q1 (storage and processing).** Typed, ZSTD-compressed Parquet partitioned by year and sorted by bank takes 206 MB instead of 490 MB "
    "as CSV and reads one year or one bank in milliseconds. Polars runs the whole core pipeline in 1.8 s; pandas needs 14.5 s and local PySpark "
    "78 to 106 s, with matching results on 55 million cells. A cluster is not needed for these data. Spark's one advantage measured here, bounded memory, "
    "matters only when the working set outgrows the largest available machine, as loan-level data would.",
    "**Q2 (data quality).** An automated audit registers 32 issues, among them zeros that mean \"not reported\", year-to-date income, "
    "reports dated after failure and 865 distressed exits without a failure record. Each has a stated rule. Look-ahead is excluded by construction and "
    "tested: features do not change when the data are truncated at three dates, and four deliberately leaky control features are caught.",
    f"**Q3 (early warning).** Yes, with limits. On the single test evaluation the chosen model reached average precision "
    f"{float(C('p5_test_metrics.csv')[0]['ap']):.3f} (base rate 0.09%), and {round(at['bank_recall_actionable']*78)} of 78 failed banks received an alarm in time to act. It did not clearly "
    f"beat a three-ratio logit, its performance fell after 2020, and it missed Silicon Valley Bank and First Republic, which carry 89% of test losses.",
], size=9.5)
page_break()

# ---------------------------------------------------------------- 1 architecture
H("1. Pipeline overview")
P("The pipeline follows a layered (medallion) design. Each layer is written by one script, read only by later layers, "
  "and checked by its own tests; every step reruns from the README.", align="j")
T(["Layer", "Content", "Written by", "Checked by"], [
    ["0 Raw", "API output as received, one Parquet file per quarter, manifest, hashes", "`fdic_download_full.py`", "`p1_verify.py`"],
    ["1 Bronze", "Typed, schema-enforced, ZSTD-3 Parquet, partitioned by `YEAR`, sorted by `(CERT, REPDTE)`", "`p1_build_bronze.py`", "reconciliation with raw"],
    ["2 Silver", "Point-in-time bank-quarter panel, exit classes, label `y`", "`p2_build_panel.py`", "`p2_checks.py` (20 tests)"],
    ["3 Gold", "42 leak-free candidate features joined with `y`, split column", "`p4_features.py`", "`p4_checks.py` (31 tests)"],
    ["Model", "Folds, costs, threshold, frozen model, single test", "`p5_develop.py`, `p5_test.py`", "`p5_checks.py` (16 tests)"],
], [1.8, 7.6, 4.0, 3.6], cap="Pipeline layers.")

# ---------------------------------------------------------------- Part 1
H("2. Part 1: Acquisition and storage")
H("2.1 Acquisition and verification", 2)
P("The download reuses the course starter's API client [2] (paging, retries, a resumable loop over quarters) and changes only the "
  "field list and output folder. A quarter is written atomically to its own Parquet file, so an interrupted run resumes at the first missing quarter. "
  "The 43 Financials fields cover the CAMELS dimensions used by supervisors and in the bank-failure literature [3]: "
  "capital (`EQ`, `RBC1AAJ`, `RBCRWAJ`), asset quality (`NCLNLS`, `NTLNLS`, `ORE`, `LNATRES`, `ELNATR`), loan mix (`LNRE`, `LNRECONS`, `LNCI`, `LNCON`), "
  "earnings (`NETINC`, `INTINC`, `EINTEXP`, `NONII`, `NONIX`), liquidity and funding (`DEP`, `COREDEP`, `BRO`, `CHBAL`, `SC`, `VOLIAB`), and size. "
  "FDIC's own ratios are downloaded only to be checked; features are recomputed from levels.", align="j")
P("`p1_verify.py` checks completeness against the API itself, not against expectations:", keep=True)
B([
    f"All 170 quarters are present, and every quarter's row count equals the API's total; there are no null or duplicate `(CERT, REPDTE)` keys. "
    f"Failures and Institutions equal their API totals ({comp['failures']['rows']:,} and {comp['institutions']['rows']:,}).",
    "Bank counts and total assets were reconciled with FDIC's `/summary` endpoint at 42 year-ends. Differences reach 303 banks and 2.1% of assets in 1989-93, "
    "the savings-and-loan resolution years, and are at most 47 banks and 1.2% of assets from 1994 on. The cause was not identified and is listed as a limitation.",
    f"A second, independent download with the starter's own field list agreed on all {comp['independent_download']['cells_compared']/1e6:.1f} million shared cells (0 differences).",
    "SHA-256 hashes of all raw files are stored in `reports/p1_raw_sha256.csv`, so a rerun can prove it used the same data vintage.",
])

H("2.2 Storage layout", 2)
P("Raw CSV loses types: Polars' default inference misreads `CERT`, `REPDTE` and three other columns, and pandas reads 30 integer amounts as floats. "
  "The bronze layer therefore enforces a schema contract (`CERT` Int32, `REPDTE` Date, amounts Int64, ratios Float64) and stops if an integer field holds a "
  "fraction. Rows, null counts and column totals equal the raw files. Twelve layouts were compared on six workloads, each timed in fresh processes "
  "(medians of repeated runs), and the share of row groups each query had to read was recorded. Table 2 and Figure 1 show the eight most informative; all twelve are in `reports/p1_storage_benchmark.csv`.", align="j")
sb = {r["layout"]: r for r in C("p1_storage_benchmark.csv")}
lay = [("csv", "CSV"), ("csv_gz", "CSV, gzip"), ("pq_zstd_1file", "Parquet ZSTD, one file"), ("pq_quarter", "ZSTD, by quarter (170 files)"),
       ("pq_year_rg64k", "ZSTD, by year, 64k-row groups"), ("pq_year_sortdate", "ZSTD, by year, sorted by date"), ("pq_year", "**ZSTD, by year, by bank, 16k groups (chosen)**"),
       ("delta_year", "Delta Lake, by year")]
rows = []
for k, name in lay:
    s = sb[k]
    rows.append([name, s["size_mb"], s["W1_s"], s["W3_s"], s["W4_s"], s["W5_s"], s["W6_s"], f"{s['W4_needed_pct']} / {s['W5_needed_pct']}"])
T(["Layout", "MB", "Full scan s", "3 cols s", "1 year s", "1 bank s", "1 quarter s", "% read: year / bank"], rows,
  [5.3, 1.1, 1.6, 1.4, 1.4, 1.4, 1.6, 3.2], cap="Storage layouts: size and query time (Polars, median seconds). The last column is the share of data the "
  "one-year and one-bank queries had to read.", size=8, align="lrrrrrrr")
F("fig_p1_storage.png", 17.0, "Size and query time by layout; the chosen layout in blue.")
P("The chosen layout takes 206 MB (42% of CSV). Partitioning by year lets a one-year query read 0.1% of the data instead of 6.3% for a single file, while 43 "
  "files avoid the small-file overhead of 170 quarterly ones (one-bank query 0.052 s against 0.154 s). Sorting by bank inside each year with 16,384-row groups "
  "keeps one bank's history in few row groups: the one-bank query reads 43% of row groups, against 91% when sorted by date and 99% with 64k groups. "
  "Delta Lake added 21% in size and a transaction log that an append-only quarterly table does not need.", align="j")

# ---------------------------------------------------------------- Part 2
H("3. Part 2: Data quality and the point-in-time panel")
H("3.1 Automated audit", 2)
P("`p2_quality.py` writes the data-quality report (`reports/p2_dq_report.md`) without manual steps: a profile of every column, null and zero shares "
  "by year, a robust-z scan for structural breaks in cross-sectional medians, and an issue register with 32 entries in five dimensions "
  "(uniqueness, completeness, validity, consistency, timeliness). Table 3 lists the issues that change the panel; the full register is in Appendix A.", align="j")
T(["Issue (rows affected)", "Evidence", "Handling"], [
    ["Reports dated on or after failure (1,095)", "268 failed banks keep filing", "R3: drop, so a failed bank is never a survivor"],
    ["No Institutions record (10,996; 235 banks)", "none failed; 84% of their rows show no deposits", "R1: drop (see 7.2)"],
    ["Zero risk-based capital ratio (462,176)", "100% zero before 1990Q1; over a third of banks from 2020 (CBLR)", "R4: zero to null, never imputed before fitting"],
    ["Year-to-date income and expenses", "flows restart every Q1 (Figure B5)", "R9: quarterly flows by difference; null across a gap"],
    ["Meaningless ratios (ROE with EQ <= 0: 14,604)", "sign flips with negative equity", "R5-R6: null; ratios recomputed from levels"],
    ["FDIC noncurrent-loan ratio vs. its own levels", "disagree in 1985-96", "features use `NCLNLS / LNLSGR`"],
    ["Impossible negative balances (793)", "stocks below zero", "R7: null"],
    ["Breaks in 2010Q1 (`VOLIAB`, `COREDEP`)", "median share moves 12.6 and 10.5 points in one quarter", "excluded from features"],
    ["Current-snapshot fields (`ACTIVE`, `CHANGEC1`)", "describe today, not quarter q", "never features; used only to classify exits"],
], [5.0, 5.8, 6.2], cap="Main data-quality issues and their handling.", size=8)
F("fig_p2_capital_zero.png", 13.5, "A zero total risk-based capital ratio means \"not reported\": before 1990Q1, and for banks using the community bank leverage ratio from 2020.")

H("3.2 Exits and the label", 2)
P("A bank stops reporting for many reasons. Institutions' `CHANGEC1` code and the Failures table classify every exit (Table 4). Health at the last report "
  "justifies the classes: merged banks look like survivors, whereas assisted mergers and distress closures without a failure record look like failures. "
  "The exam's definition is kept: only a FAILURE record in the Failures table makes `y = 1`; the 865 distressed exits stay `y = 0` and are marked `_near_miss` "
  "for sensitivity analysis.", align="j")
eh = C("p2_exit_health.csv")
T(["Exit class", "Banks", "Years", "Median equity/assets %", "Equity <= 0, %", "Median ROA"],
  [[r["_exit_class"].replace("_", " "), n0(r["banks"]), f"{r['first_year']}-{r['last_year']}", r["median_eq_ta_pct"], r["eq_nonpositive_pct"], r["median_roa"]] for r in eh],
  [5.2, 1.6, 2.2, 3.2, 2.4, 2.4], cap="Exit classes and financial health at the last report.", size=8, align="lrcrrr")
P(f"The label is `y(i,q) = 1` if bank i's first in-scope failure falls in (q, q + 4 quarters]. Of 2,858 failure events in scope, 2,837 are labelled; 21 have "
  f"no report before failing. The label is joined last and is null after 2025Q2, where four quarters of future are not yet observed.", align="j")
H("3.3 Point-in-time correctness", 2)
P("Every derived column uses only reports dated at or before q; rolling and lagged values require consecutive quarters, so a reporting gap gives null "
  "rather than a wrong total. `p2_checks.py` runs 20 tests, all passing, including:", keep=True)
B([
    "**Truncation:** rebuilding the panel from data cut at 1990, 2008 and 2019 leaves all 53 point-in-time columns unchanged; a control column that "
    "uses the future (`_exit_class`) does change, so the test has power.",
    "**Independent recomputation:** `y` recomputed separately gives 0 differences on 10,860 positives; quarterly flows sum to the December year-to-date value "
    "for 404,636 bank-years; the last pre-failure report agrees with FDIC's `QBFASSET` within 5% for 98.9% of failures.",
    "**Known cases:** Silicon Valley Bank, First Republic and Washington Mutual each have `y = 1` on exactly their last four reports.",
])

# ---------------------------------------------------------------- Part 3
H("4. Part 3: Scalable processing")
H("4.1 Core transformations in Polars", 2)
P("The core pipeline (bronze to the panel) is specified once and implemented three times, in Polars (main), pandas (baseline) and PySpark: "
  "T1 column projection; T2 semi-join with Institutions and null-equity filter; T3 join of the first failure and removal of later reports; "
  "T4 cleaning rules; T5 year-to-date to quarterly flows by bank; T6 guarded ratios; T7 lags and rolling statistics by bank; T8 cross-sectional ranks and "
  "state medians by quarter; T9 the label; T10 sort and write. Engine differences that could silently change results were pinned down in the "
  "specification: zero denominators (inf in Polars and pandas, an error in ANSI Spark) become null everywhere; Spark's window mean ignores nulls, so a full "
  "window is required explicitly; percentile ranks use one definition in all engines.", align="j")
P("Polars builds a lazy query plan and optimises it before execution [4]. Table 5 isolates the optimiser: on a selective query (one year, few columns), "
  "projection and predicate pushdown into the Parquet scan cut time from 0.25 s to 0.004 s; on the full pipeline, which needs every row, turning all rules off "
  "costs 77% more time and 48% more memory. The streaming engine processes the plan in morsels across threads; profiling by stage shows "
  "no single dominant step (scan and join 0.15 s, bank windows 0.23 s, the final sort 0.31 s).", align="j")
oe = C("p3_optimizer_effect.csv")
T(["Optimiser setting", "Core pipeline, s", "Selective query, s"],
  [[a["optimizer"], a["median_s"], b["median_s"]] for a, b in zip(oe[:5], oe[5:])],
  [6.0, 3.5, 3.5], cap="Effect of the Polars optimiser (median seconds).", size=8, align="lrr")
P("Listing 1 shows the optimised plans, read from the bottom up. In the selective query the year filter has become a choice of file (one of 43) and "
  "only 3 of 44 columns are read. In the core pipeline the scan reads 16 of 44 columns with the null-equity filter applied inside the scan; `(ASSET > 0)` "
  "is computed once and reused by every ratio (common subexpression elimination); and the panel feeding the state median is materialised once (`CACHE`) "
  "and used by both sides of the final join. The optimiser also knows what it must not do: when a `YEAR == 2008` filter is written after the windows, it "
  "stays above them, because filtering first would change lags and rolling values for 33,614 rows. This is why the order of operations in the "
  "specification matters, and why the truncation tests of Part 2 rebuild from bronze rather than filter the finished panel.", align="j")
CODE([
    "-- Selective query (mean ROA by state in 2008), after optimisation",
    "AGGREGATE [col(\"ROA\").mean()] BY [col(\"STNAME\")]",
    "  simple π 2/2 [\"STNAME\", \"ROA\"]",
    "    Parquet SCAN [.../financials/YEAR=2008/part-0.parquet]          <- 1 of 43 files: partition pruning",
    "    PROJECT 3/44 COLUMNS                                             <- projection pushdown",
    "-- Core pipeline, after optimisation (abridged)",
    "SORT BY [CERT, REPDTE]                                               T10",
    "  LEFT JOIN ON [REPDTE, STNAME]  <- AGGREGATE median(eq_ta) BY [REPDTE, STNAME] FROM CACHE    T8",
    "    CACHE[id: abbb0777...]                                           <- computed once, read twice",
    "      WITH_COLUMNS [eq_ta.shift(4).over([CERT]), roa_q.rolling_std().over([CERT]), eq_ta.rank().over([REPDTE]) ...]   T7-T8",
    "        WITH_COLUMNS [when(__POLARS_CSER_..).then(EQ / ASSET) ...]   T6",
    "          WITH_COLUMNS [(ASSET > 0) AS __POLARS_CSER_0x..1b]        <- common subexpression, computed once",
    "            WITH_COLUMNS [NETINC - NETINC.shift(1).over([CERT]) ...]   T5",
    "              SORT BY [CERT, REPDTE]",
    "                FILTER _fail_date.is_null() | (REPDTE < _fail_date)   T3, rule R3",
    "                  LEFT JOIN ON [CERT] <- AGGREGATE min(FAILDATE) BY [CERT] <- Parquet SCAN failures, PROJECT 3/11",
    "                    SEMI JOIN ON [CERT] <- Parquet SCAN institutions, PROJECT 1/11      T2, rule R1",
    "                      Parquet SCAN [financials, 43 files]",
    "                      PROJECT 16/44 COLUMNS                          <- T1 projection pushdown",
    "                      SELECTION: col(\"EQ\").is_not_null()           <- T2 rule R2, predicate pushdown",
])
p = P("**Listing 1.** Optimised Polars plans (`reports/p3_plan_polars.txt`, abridged; arrows are annotations).", size=SMALL, after=6)

H("4.2 Benchmark against pandas and PySpark", 2)
P("Each engine ran from a fresh process on the same input with the same output; time is the median of 3 to 7 runs, memory is the peak resident set of the "
  "whole process tree (including the JVM), and the free memory at start is recorded because the laptop (18 threads, 15.4 GB) was in daily use.", align="j")
F("fig_p3_engines.png", 15.5, "Core pipeline on 1.67 million bank-quarters: median time and peak memory.")
P(f"All engines produced the same panel: {recon['cells_compared_per_engine']/1e6:.0f} million cells per engine compared with Polars and with the silver layer. "
  f"PySpark differed by at most 5.7e-14. pandas differed in 5 cells of the 8-quarter ROA volatility, by up to 4e-9; recomputing those cells directly showed "
  f"Polars exact and pandas' rolling variance off by rounding. Polars is 8 times faster than pandas. pandas used 14.3 CPU seconds in 14.5 s of wall time, so it "
  f"ran essentially on one core, while Polars spread 11.6 CPU seconds over 18 threads.", align="j")

H("4.3 Reading the Spark plan", 2)
sp = plan["spark"]
P(f"The physical plan [4] explains why local Spark is slow on this job. The query runs in {sp['stages']} stages separated by 7 exchanges (shuffles) plus 2 reused "
  f"ones. Each change of grouping key forces a full repartition of 1.67 million rows: by `CERT` for the bank windows (259 MiB), by `REPDTE` for quarterly "
  f"ranks (411 MiB), by `(REPDTE, STNAME)` for state medians (437 MiB), and a range partition for the final sort (475 MiB). In total "
  f"{sp['shuffle_write_gb']:.2f} GB is shuffled from a 206 MB input. The small tables are broadcast (4 `BroadcastHashJoin`), and adaptive query execution "
  f"coalesces partitions (7 `AQEShuffleRead`). Partition pruning works when the query allows it: a one-year query shows `PartitionFilters: YEAR = 2008`, a "
  f"one-bank query pushes `CERT = 14` into the scan; the core pipeline needs every year, so it has no partition filter. Unlike Polars' `CACHE`, Spark "
  f"recomputes the branch that feeds the state median: the financials are scanned and joined a second time (Listing 2). Polars runs the same windows over "
  f"in-memory columns without serialising rows between stages. Tuning confirmed the diagnosis: 18 shuffle partitions instead of the default 200 cut time "
  f"from 100 s to 57 s, consistent with 200 small tasks per shuffle costing more in scheduling than they add in parallelism on 18 threads.", align="j")
CODE([
    "AdaptiveSparkPlan",
    "+- Sort [CERT, REPDTE]",
    "   +- Exchange rangepartitioning(CERT, REPDTE, 200)              <- shuffle: final sort (T10)",
    "      +- SortMergeJoin [REPDTE, STNAME] LeftOuter                <- join with the state median (T8)",
    "         :- Exchange hashpartitioning(REPDTE, STNAME, 200)        <- shuffle",
    "         :  +- Window rank(npl), rank(eq_ta) PARTITION BY REPDTE  <- quarter ranks (T8)",
    "         :     +- Exchange hashpartitioning(REPDTE, 200)          <- shuffle",
    "         :        +- Window lag, avg, stddev_samp PARTITION BY CERT ORDER BY REPDTE   (T5, T7)",
    "         :           +- Exchange hashpartitioning(CERT, 200)      <- shuffle",
    "         :              +- Filter (isnull(_fail_date) OR REPDTE < _fail_date)         (T3)",
    "         :                 +- BroadcastHashJoin [CERT] LeftOuter <- first failure, broadcast",
    "         :                    +- BroadcastHashJoin [CERT] LeftSemi <- Institutions, broadcast",
    "         :                       +- FileScan parquet financials  PushedFilters: [IsNotNull(EQ), IsNotNull(CERT)]",
    "         :                                                         PartitionFilters: []",
    "         +- ObjectHashAggregate percentile(eq_ta, 0.5) BY [REPDTE, STNAME]",
    "            +- Exchange hashpartitioning(REPDTE, STNAME, 200)     <- shuffle",
    "               +- ... BroadcastHashJoin x2 ... FileScan parquet financials   <- second scan of the same data",
])
P("**Listing 2.** Spark physical plan of the core pipeline before adaptive execution (`reports/p3_plan_spark.txt`, abridged).", size=SMALL, after=6)

H("4.4 Threads, scale and when a cluster is justified", 2)
F("fig_p3_scale.png", 16.0, "Data copied 2 to 16 times. Polars and pandas were stopped by the memory guard at four copies; Spark with a 2 GB heap finished sixteen in the recorded run.")
P("More threads help little at this size (Appendix B, Figure B3): Polars streaming reaches a 4.4x speed-up at 18 threads (24% efficiency), Spark 2.0x at 12. "
  "On larger copies of the data, Polars and pandas grew in memory until the guard stopped them at 6.7 million rows with about 3.5 GB free. Spark with "
  "6 threads and a 2 GB heap finished 26.7 million rows in 346 s at 2.7 GB peak by spilling 14.4 GB to disk; with 18 threads it ran out of memory at eight copies. In two later reruns of the same job (Section 8) Spark finished eight copies but ran out of execution memory at sixteen, so 26.7 million rows is at the edge of a 2 GB heap: Spark keeps memory bounded, but beyond that size the heap or the number of partitions has to grow.", align="j")
P("**When would a cluster be justified?** Not for these data. The panel grew by 17,612 rows in 2025 and would take about 95 years to double. Spark used "
  f"{cluster['cpu_ratio_spark_to_polars']} times the CPU of Polars; even with perfect scaling it would need about {cluster['cores_spark_needs_to_match_polars_at_perfect_scaling']} "
  "cores to match one laptop. A cluster pays off when (1) the working set exceeds the memory of the largest affordable single machine and cannot be split by "
  "year or bank, for example loan-level credit-registry data; (2) spilling pushes a job past its time window; or (3) long jobs need fault tolerance and shared "
  "access. Before that, the cheaper steps are fewer threads, Polars streaming, a larger machine, and Spark in local mode. The memory limit on a free 15.4 GB "
  "machine is extrapolated, not measured: roughly 16 to 23 million rows at 12 GB.", align="j")

# ---------------------------------------------------------------- Part 4
H("5. Part 4: Feature pipeline")
H("5.1 Features and the split", 2)
P("The 42 candidate features follow the CAMELS framework plus regional and data-quality signals (full list in Appendix A, Table A2): capital (equity to assets, "
  "its 4-quarter change, rank and gap to the state median), asset quality (noncurrent loans, a bounded Texas ratio, reserve coverage, charge-offs, provisions), "
  "loan mix, earnings (4-quarter ROA, its change and 8-quarter volatility, loss quarters, cost to income), liquidity and funding (liquid assets, brokered "
  "deposits, non-deposit funding), size, region (state median NPL, share of loss-making banks, recent state failures) and flags. All are ratios, ranks or "
  "changes, so they are comparable across four decades. Fields known only today or after failure (`ACTIVE`, `CHANGEC1`, `COST`), identifiers, calendar "
  "variables and fields with era-dependent coverage or definition breaks are excluded (Appendix A, Table A3). Figure 5 shows why the core ratios are "
  "informative: failing banks diverge from survivors well before failure, with equity falling and noncurrent loans rising over three years.", align="j")
F("fig_p4_event_study.png", 16.0, "Median ratios of failed banks over the 12 quarters before failure, against the median of banks that did not fail.")
P("The split puts the test period after development with a four-quarter gap (Figure 6). Development, 1984Q1 to 2011Q4, contains two crises and a quiet period. "
  "The gap, 2012, ensures that the last development label, which looks to 2012Q4, never sees a test failure. The test period, 2013Q1 to 2025Q2, contains "
  "78 failed banks (at least 50 are required) and the 2023 failures, the case a supervisor most needs to understand. Testing on the 2008-14 crisis instead "
  "would leave the model trained on the 1980s alone and leave the recent era unexamined.", align="j")
F("fig_p4_split.png", 16.0, "Failed banks by year, coloured by the period of their last labelled report (a bank labelled in two periods is counted once).")

H("5.2 Screening and preprocessing", 2)
P("Screening used labels from the training window 1986-2006 only, under rules fixed in advance: at most 20% nulls in every era; a within-era AUC at least "
  "0.02 from 0.5; the same direction in every era; of a pair with |Spearman| > 0.90, keep the stronger; and stop if any single feature exceeds AUC 0.98, "
  "which would more likely signal a leak. The within-era AUC matters because pooled AUC rewards features that merely drift with the calendar (Figure 7). "
  f"{len(sel['numeric']) + len(sel['flags'])} features passed; no missing-value indicator added signal beyond its feature.", align="j")
F("fig_p4_signal.png", 9.5, "Single-feature AUC of the selected features, pooled and within eras. The state features (st_*) lose most of their signal within eras.")
P(f"All preprocessing is a scikit-learn [5] `Pipeline` fitted on training rows only: winsorising at the 0.5th and 99.5th percentiles, median imputation, and "
  f"standard scaling. `load_xy` refuses to return test rows unless explicitly asked, and only `p5_test.py` asks (a test checks this). On the Part 4 split "
  f"(train 1986-2006, validation 2008-11), the fitted parameters equal an independent computation on {pfacts['train']['rows']:,} training rows; fitting with "
  f"validation rows would have changed 19 of 24 upper caps, which shows what the pipeline prevents. Six features drift strongly into the crisis "
  f"(PSI > 0.25), a reason to validate across eras rather than on one split.", align="j")
H("5.3 Leakage tests", 2)
P("`p4_checks.py` runs 31 tests, all passing. Features are identical when every label-side column is removed before building them; rebuilding from bronze "
  "with data truncated at 1990, 2008 and 2019 leaves all 42 candidates unchanged; and four deliberately leaky control features (a lead, a centred window, "
  "an all-time and a whole-history bank statistic) are each caught by the same truncation test. Seven features match an independent pandas computation, and "
  "known values are checked by hand, for example the share of Texas banks failing in 1989 (223 of 1,468).", align="j")

# ---------------------------------------------------------------- Part 5
H("6. Part 5: Modelling and validation")
H("6.1 Validation design and metrics", 2)
P("Validation uses three rolling-origin folds inside development, each with an expanding training window and a four-quarter gap, and each refitting "
  "preprocessing and model (Table 6). F1 is a crisis the training data has barely seen, F2 a quiet period like most of the test, and F3 a later crisis and the "
  "only fold that did not inform feature screening. Random cross-validation was rejected: it would mix quarters of the same crisis into training and "
  "validation.", align="j")
T(["Fold", "Training", "Validation", "Failed banks in validation"], [
    ["F1 crisis", "1986-1988", "1990-1991", "696"], ["F2 quiet", "1986-1992", "1994-2006", "71"], ["F3 crisis", "1986-2006", "2008-2011", "463"],
    ["Final model", "1986-2011", "(test 2013Q1-2025Q2, once)", "78"],
], [3.0, 3.5, 5.0, 5.0], cap="Time-ordered folds (each followed by a four-quarter gap).", size=8.5)
P("With a base rate near 0.1%, accuracy is meaningless and ROC AUC looks excellent even for weak models, because true negatives dominate it [6]."
  " The primary metric is therefore **average precision (AP)**, the area under the precision-recall curve, compared with the base rate. "
  "Supporting metrics are the share of failed banks caught in time, precision at the threshold, and Brier score for calibration. Test uncertainty is a "
  "bootstrap that resamples banks, not rows, because a bank's quarters are correlated.", align="j")

H("6.2 Models and selection", 2)
P("Two baselines are compared with two model families: **B0**, the Texas ratio with no fitting; **B1**, a logistic regression on three ratios (equity to assets, "
  "noncurrent loans, ROA); a regularised logistic regression on all features; and histogram gradient boosting (GBM, learning rate 0.03, 15 leaves, 400 trees, "
  "the best of six settings; monotone-constrained variants were also tried), with and without the three state features. The rule, fixed before results: highest mean AP, with a simpler model preferred within 1%.", align="j")
F("fig_p5_cv.png", 16.5, "Average precision by fold. The chosen model is the GBM without state features (mean AP 0.453 against 0.332 for B1 and 0.296 for B0).")
P("The GBM without state features had the highest mean AP (0.453), ahead of the version with them (0.447): the state features helped in F1 but hurt in "
  "the other two folds, which is consistent with their weak within-era signal. Permutation importance on F3 is led by equity to assets (AP drop 0.196), ROA "
  "volatility and the Texas ratio.", align="j")

H("6.3 Costs and the alarm threshold", 2)
P(f"The threshold comes from the costs of the two errors, estimated from development data. A **missed failure** costs FDIC's estimated loss (`COST`): median "
  f"USD {costs['median_loss_thousand']/1000:.1f} million across {costs['loss_rate_failures']:,} failures, or {costs['loss_rate_median']*100:.1f}% of assets. A **false alarm** costs one extra examination: "
  f"16 person-weeks priced at each year's median non-interest expense per employee, a median of USD {costs['median_exam_cost_thousand']:.0f} thousand. "
  f"An alarm cannot avoid the whole loss: banks closed with 2-4% equity cost FDIC 18.1% of assets against 23.6% for insolvent ones, so an alarm is assumed to "
  f"save 23% of the loss (range 10-50% tested). An alarm counts only if it comes from a report filed at least 45 days before failure.", align="j")
F("fig_p5_threshold.png", 11.5, "Total cost against alarm share on out-of-fold development predictions; the minimum sets the threshold.")
P(f"The threshold minimising total cost is a score of {frozen['threshold']:.4f}: on development it raises alarms on 3.1% of bank-quarters and catches 92% of "
  f"failed banks in time. Each fold's own optimum lies between 0.0067 and 0.0091, close to the pooled value; across the sensitivity grid (share saved 10-50%, "
  f"exam cost 0.5-4 times) it ranges from 0.0046 to 0.0407. The model, threshold and a SHA-256 hash of the fitted model were then frozen.", align="j")

H("6.4 Spark MLlib demonstration", 2)
ml = C("p5_spark_mllib.csv")
T(["Engine and model", "AP", "ROC AUC", "Fit time, s"], [[r["engine"], f"{float(r['ap']):.3f}", f"{float(r['roc_auc']):.3f}", f"{float(r['fit_s']):.1f}"] for r in ml],
  [7.0, 2.0, 2.5, 2.5], cap="Fold F3 (train 1,041,927 rows, validate 127,181) in Spark MLlib and scikit-learn, same features and preprocessing.", size=8.5, align="lrrr")
P(f"Quantile caps and medians were fitted on the training window, as in Part 4, followed by an MLlib pipeline (`Imputer`, `VectorAssembler`, "
  f"`StandardScaler`, `LogisticRegression`). It reproduced the scikit-learn logistic regression: AP 0.591 against "
  f"0.589 and a rank correlation of scores of {mllib['rank_corr_spark_vs_sklearn']:.3f}, although individual probabilities differ by up to {mllib['max_abs_prob_diff']:.2f} (the penalty was mapped as lambda = 1 / (C n), but the solvers differ). A 30-tree MLlib GBT reached AP 0.633. Spark took about five times longer to fit, as in Part 3.", align="j")

H("6.5 The single test evaluation", 2)
P("`p5_test.py` refuses to run without the frozen file and verifies the model hash. It was run once; its results are tied by hash to a lock file, and a second "
  "run is refused (tested). Table 8 reports all three models on the test period.", align="j")
tm = C("p5_test_metrics.csv")
T(["Model", "AP", "95% CI (bank bootstrap)", "ROC AUC", "Failed banks in top 1%"],
  [[tm[0]["model"] + " (chosen)", f"{float(tm[0]['ap']):.3f}", f"{boot['ap_ci95']['chosen'][0]:.3f}-{boot['ap_ci95']['chosen'][1]:.3f}", f"{float(tm[0]['roc_auc']):.3f}", f"{float(tm[0]['top_bank_recall'])*100:.0f}%"],
   [tm[1]["model"], f"{float(tm[1]['ap']):.3f}", f"{boot['ap_ci95']['B1'][0]:.3f}-{boot['ap_ci95']['B1'][1]:.3f}", f"{float(tm[1]['roc_auc']):.3f}", f"{float(tm[1]['top_bank_recall'])*100:.0f}%"],
   [tm[2]["model"], f"{float(tm[2]['ap']):.3f}", f"{boot['ap_ci95']['B0'][0]:.3f}-{boot['ap_ci95']['B0'][1]:.3f}", f"{float(tm[2]['roc_auc']):.3f}", f"{float(tm[2]['top_bank_recall'])*100:.0f}%"]],
  [6.6, 1.5, 3.6, 2.0, 3.3], cap="Test period 2013Q1-2025Q2: 274,032 bank-quarters, 254 with y = 1 (base rate 0.09%), 78 failed banks.", size=8.5, align="lrcrr")
pr = test["periods"]
P(f"At the frozen threshold the model raised {at['alarms']:,} alarms, {at['alarm_share']*100:.1f}% of bank-quarters or about 59 per quarter, with "
  f"{at['row_precision']*100:.1f}% precision. {round(at['bank_recall_actionable']*78)} of 78 failed banks (76%) received an alarm in time to act and 68 an alarm at some point; "
  f"the median lead was {at['median_lead_quarters']:.0f} quarters. The AP gain over B1 has a 95% interval of {boot['ap_diff_ci95']['B1'][0]:.3f} to "
  f"+{boot['ap_diff_ci95']['B1'][1]:.3f}, which includes zero; the gain over B0 does not. AP fell from {pr['2013-2019']['ap']:.3f} in 2013-19 "
  f"(67 failed banks) to {pr['2020-2025']['ap']:.3f} in 2020-25 (14 failed banks). Fewer alarms than in development (1.1% against 3.1%) reflect a calmer period. The model's scores are "
  f"reasonably calibrated: the mean predicted probability was 0.13% against an observed 0.09% (0.18% for B1).", align="j")
F("fig_p5_test_pr.png", 10.5, "Precision-recall curves on the test period.")
F("fig_p5_test_banks.png", 15.5, "Each failed bank in the test period by failure date and FDIC loss; orange marks banks without an alarm in time.")
P(f"The cost saving was small: USD {test['cost_thousand_usd']['saving']/1e6:.2f} billion, 2.1% of the cost without the system, because Silicon Valley Bank and "
  f"First Republic raised no alarm and account for {post['share_of_test_losses_in_two_banks']*100:.1f}% of test-period losses. Without those two banks the "
  f"saving would be 18.1%, a post hoc and descriptive comparison that informs no decision. SVB's last report before failure showed 7.4% equity to assets "
  f"and 0.19% noncurrent loans: the ratios showed no distress.", align="j")

# ---------------------------------------------------------------- Part 6
H("7. Part 6: Conclusions")
H("7.1 Answers to the three questions", 2)
P("**Q1.** Store the reports as typed, ZSTD-compressed Parquet, partitioned by year and sorted by bank and date (206 MB against 490 MB of CSV, selective "
  "queries in milliseconds), and process them with lazy Polars (1.8 s for the core pipeline, 8 times faster than pandas and 40 to 60 times faster than local "
  "Spark, with matching results). These data do not need a cluster. Spark's measured advantage was bounded memory, which matters once the working set outgrows "
  "the largest affordable machine and cannot be split, as with loan-level data, or when spilling breaks a time window or jobs need fault tolerance.", align="j")
P("**Q2.** The raw data contain reports filed after failure, zeros that mean \"not reported\", year-to-date flows, ratios that are undefined or "
  "inconsistent with their own levels, definition breaks, current-snapshot fields, and exits that look like failures but are not recorded as such. Each has a "
  "rule, applied before any feature is built. Look-ahead is prevented by construction (backward-only windows, label joined last, preprocessing fitted on "
  "training rows) and verified by truncation tests with positive controls.", align="j")
P("**Q3.** Yes, for ranking banks by the kind of failure that builds up in reported financials: with features, model and threshold fixed on development data "
  "and the test used once, the model reached an average precision about 370 times the base rate and alerted 76% of failed banks in time. No, for failures that arrive through a "
  "sudden funding run: the two largest test failures raised no alarm, and the complex model did not clearly beat a three-ratio logit. The output suits "
  "supervisory prioritisation alongside liquidity monitoring, not a stand-alone alarm.", align="j")
H("7.2 Limitations", 2)
B([
    "**Data vintage:** FDIC serves current values, which may include revisions, so truncation tests cannot exclude look-ahead from restated reports. "
    "Rule R1 also uses today's Institutions table; no bank it removes ever failed.",
    "**Few test events:** 78 failed banks, about half of them in 2013-14, give wide intervals and little evidence for recent years.",
    "**Assumed costs:** the exam size, the 45-day action lag and the 23% share of loss avoided (a correlation, not a causal effect) are assumptions; the threshold moves between 0.0046 and 0.0407 across the range tested.",
    "**Labels and selection:** 865 distressed exits count as non-failures; screening saw labels up to 2006, so folds F1 and F2 are slightly optimistic, and one screening tie-break was added after a first run.",
    "**Reconciliation and benchmarks:** the 1989-93 differences with FDIC's summary totals are unexplained; benchmarks ran on a laptop in daily use, and scaled data repeat real banks, so they test volume, not variety.",
])
H("7.3 What would change for Vietnam", 2)
P("The storage layout, point-in-time rules, tests and validation discipline carry over unchanged. The label, features, costs and model complexity do not.", align="j")
T(["Layer", "Change for the State Bank of Vietnam"], [
    ["Label", "Formal failures are rare: weak banks were placed under special control, and CB, OceanBank, GPBank and DongA Bank were compulsorily transferred in 2024-25 [7, 8]. The label must be a supervisory distress event (early intervention or special control [9]), dated when decided."],
    ["Validation", "Fifty test events are unlikely to exist. Results should be reported per event with expert review; risk rankings may borrow from models estimated abroad."],
    ["Model", "A few dozen commercial banks favour simple, explainable models; even with thousands of US banks the GBM's gain over a three-ratio logit was not established."],
    ["Features", "Add what US data lack. The SCB case centred on lending to parties linked to a major shareholder and a deposit run in October 2022 [10, 11]: related-party and large-exposure measures under the 2024 law [9], ownership links and frequent deposit flows."],
    ["Data and costs", "Supervisory data need restricted access and vintages kept as filed. There is no FDIC-style loss table: deposit insurance covers VND 125 million per depositor [12] while support for SCB was reported at about USD 24 billion [10], so contagion and public support enter the cost of a missed case and the threshold must be derived again."],
    ["Tools", "Bank-level panels would be far smaller than this one, so a single machine suffices; Spark becomes relevant only with loan-level credit-registry data, decided by measurement as in Part 3."],
], [2.6, 14.4], cap="What changes when the pipeline is moved to Vietnam.", size=8.5)

# ---------------------------------------------------------------- Reproducibility
H("8. Reproducibility")
P("The repository contains `README.md`, a pinned `requirements.txt` (Python 3.15; Polars 2.1.0, pandas 3.1.1, PyArrow 26.0.1, scikit-learn 1.9.1, PySpark 4.3.0), "
  "all scripts and their outputs in `reports/`; data files are not committed. Seeds are fixed in `src/config.py`. Spark needs a JDK (Temurin 21 was used; the README "
  "explains a Windows file-system shim). On 2026-11-08 the author reran the README in VS Code in a clean clone of the repository (`git clone`, "
  "a new virtual environment from `requirements.txt`, and the same raw download, checked against `reports/p1_raw_sha256.csv`). Every step completed, and "
  "`p5_test.py --reproduce` reproduced the frozen test evaluation exactly; it cannot overwrite the recorded results. `repro_compare.py` found all 61 "
  "deterministic results identical, including every bronze, silver and gold Parquet file byte for byte (`reports/repro_check.csv`). Timings and execution "
  "plans differed, as expected on a laptop in daily use, and one measurement did not repeat: Spark with a 2 GB heap finished sixteen copies of the data "
  "in the recorded run but ran out of execution memory at sixteen copies in two reruns, while eight copies always finished. `p6_conclusions.py` therefore "
  "reports the largest scale Spark completed instead of assuming sixteen. The rerun took about 3 hours on an idle laptop; the first run, on a busy one, "
  "took about 10 hours, mostly model development.", align="j")

# ---------------------------------------------------------------- References
H("References and use of AI tools")
refs = [
    "FDIC (2026). BankFind Suite API and field dictionary. https://api.fdic.gov/banks/docs/ (accessed 2026-11-07).",
    "BFT65B course team (2026). Starter script `fdic_download_starter.py`, reused by `fdic_download_full.py`.",
    "Cole, R. A., and White, L. J. (2012). Déjà vu all over again: The causes of U.S. commercial bank failures this time around. _Journal of Financial Services Research_, 42, 5-29.",
    "Software documentation: Polars (https://docs.pola.rs), Apache Arrow and Parquet, pandas, Apache Spark (https://spark.apache.org/docs/latest/) and delta-rs.",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research_, 12, 2825-2830.",
    "Saito, T., and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. _PLoS ONE_, 10(3), e0118432.",
    "The Investor. Compulsory transfer of 4 banks under special control approved. https://theinvestor.vn/compulsory-transfer-of-4-banks-under-special-control-approved-d4787.html",
    "Vietnamnet. Four underperforming banks in Vietnam restructured through mandatory transfer. https://vietnamnet.vn/en/four-underperforming-banks-in-vietnam-restructured-through-mandatory-transfer-2364609.html",
    "Law on Credit Institutions No. 32/2024/QH15; summaries by Mayer Brown (2024) and Deposit Insurance of Vietnam. https://div.gov.vn/restructuring-weak-credit-institutions-and-the-role-of-div-according-to-the-law-on-credit-institutions-2024a",
    "CNBC (2024, April 17). Vietnam mounts unprecedented $24 billion rescue for bank engulfed in giant fraud, documents show. https://www.cnbc.com/2024/04/17/vietnam-mounts-unprecedented-24-billion-rescue-for-bank-engulfed-in-giant-fraud-documents-show.html",
    "Tuoi Tre News (2022, October 16). Vietnam's central bank places lender under special scrutiny. https://tuoitrenews.vn/news/business/20221016/vietnam-s-central-bank-places-lender-under-special-scrutiny/69589.html",
    "Decision No. 32/2021/QD-TTg on the deposit insurance limit. https://english.luatvietnam.vn/decision-no-32-2021-qd-ttg-on-insurance-money-limit-211331-doc1.html",
]
for i, ref in enumerate(refs, 1):
    q = P(f"[{i}] {ref}", size=8.5, after=1.5)
    q.paragraph_format.left_indent = Cm(0.75)
    q.paragraph_format.first_line_indent = Cm(-0.75)
P("**AI tools.** Claude (Anthropic), through Claude Code and the Claude app, was used to discuss the design, write and debug code, and draft text. "
  "The author reviewed the code, reran the whole pipeline from the README in VS Code (Section 8), checked every number against the result files, "
  "and is responsible for the content.", size=9)

# ---------------------------------------------------------------- Appendix
page_break()
H("Appendix A. Tables")
TCOUNT[0] = 0
caption_prefix = "A"


def capA(text):
    TCOUNT[0] += 1
    p = doc.add_paragraph()
    r = p.add_run(f"Table A{TCOUNT[0]}. "); r.bold = True; r.font.size = Pt(SMALL)
    runs(p, text, SMALL)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)


capA("Issue register written by the automated data-quality report (`p2_issue_register.csv`).")
ir = C("p2_issue_register.csv")
T(["Dimension", "Issue", "Rows", "Banks", "Period", "Handling"],
  [[r["dimension"], r["issue"], n0(r["rows"]) if r["rows"] else "", n0(r["banks"]) if r["banks"] else "", r["period"], r["handling"]] for r in ir],
  [1.8, 5.4, 1.4, 1.1, 1.6, 5.7], size=7, align="llrrll")
capA("Candidate features (42). Selected features are in bold; the model excludes the three state features (st_*).")
fd = C("p4_feature_dictionary.csv")
chosen = set(sel["numeric"]) | set(sel["flags"])
T(["Feature", "Pillar", "Sign", "Definition"],
  [[f"**{r['feature']}**" if r["feature"] in chosen else r["feature"], r["pillar"], r["expected_sign"], r["formula"]] for r in fd],
  [2.6, 2.4, 1.0, 11.0], size=7, align="llcl")
capA("Fields left out of the features, and why (`p4_excluded.csv`).")
T(["Left out", "Reason"], [[r["left_out"], r["why"]] for r in C("p4_excluded.csv")], [4.6, 12.4], size=7)
capA("Cleaning rules applied to the panel, in order (`p2_waterfall.csv`).")
T(["Rule", "What", "Rows changed", "Banks", "Rows left"],
  [[r["rule"], r["what"], n0(r["rows_changed"]), n0(r["banks_affected"]), n0(r["rows_left"])] for r in C("p2_waterfall.csv")],
  [1.3, 7.2, 2.6, 2.2, 2.6], size=7.5, align="llrrr")
capA("Engine benchmark detail (`p3_engines.csv`, `p3_threads.csv`). Time in seconds, memory in GB.")
eng = C("p3_engines.csv")
T(["Engine", "Partitions", "Runs", "Median s", "CPU s", "Peak GB", "Free GB at start", "Shuffle GB"],
  [[r["engine"], r["shuffle_partitions"] if r["engine"] == "spark" else "", r["runs"], r["wall_s"], r["cpu_s"], r["peak_rss_gb"], r["avail_start_gb"], r["shuffle_gb"]] for r in eng],
  [3.0, 1.7, 1.2, 1.9, 1.8, 1.8, 2.6, 2.0], size=7.5, align="lrrrrrrr")
capA("Test results by sub-period (frozen model and threshold). A bank labelled in both sub-periods is counted in each, so 67 + 14 exceeds 78.")
T(["Period", "Rows", "y = 1", "Failed banks", "AP", "ROC AUC", "Alarm share", "Precision at threshold"],
  [[k, n0(v["rows"]), v["y1"], v["failed_banks"], f"{v['ap']:.3f}", f"{v['roc_auc']:.3f}", f"{v['alarm_share']*100:.2f}%", f"{v['precision_at_threshold']*100:.1f}%"] for k, v in pr.items()],
  [2.2, 1.8, 1.4, 2.0, 1.5, 1.8, 2.2, 3.0], size=7.5, align="lrrrrrrr")
capA("Threshold sensitivity on development data (`p5_threshold_sensitivity.csv`).")
T(["Share of loss saved", "Exam cost multiplier", "Threshold", "Alarm share", "Banks caught in time"],
  [[f"{float(r['share_saved'])*100:.0f}%", f"{float(r['exam_cost_multiplier']):g}x", f"{float(r['threshold']):.4f}", f"{float(r['alarm_share'])*100:.1f}%", f"{float(r['bank_recall'])*100:.0f}%"] for r in C("p5_threshold_sensitivity.csv")],
  [3.2, 3.2, 2.4, 2.4, 3.2], size=7.5, align="rrrrr")

page_break()
H("Appendix B. Additional figures")
FCOUNT[0] = 0


def FA(name, width, text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(0); p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(R / name), width=Cm(width))
    FCOUNT[0] += 1
    q = doc.add_paragraph(); r = q.add_run(f"Figure B{FCOUNT[0]}. "); r.bold = True; r.font.size = Pt(SMALL); runs(q, text, SMALL)
    q.paragraph_format.space_after = Pt(6)


FA("fig_p1_coverage.png", 13.0, "Institutions reporting per quarter, 1984Q1 to 2026Q2: consolidation from about 18,000 to about 4,300 banks.")
FA("fig_p2_failures.png", 13.0, "Labelled failures per year and the share of bank-quarters with y = 1.")
FA("fig_p3_threads.png", 10.0, "Speed-up over one thread for the core pipeline.")
FA("fig_p2_missing.png", 14.0, "Null values by field and year: concentrated in specific fields and eras.")
FA("fig_p2_ytd.png", 13.0, "Year-to-date income as reported and the derived quarterly flow (all banks, 2015-2019).")

doc.save(OUT)
print("saved", OUT)
