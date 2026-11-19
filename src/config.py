"""
Central configuration for the BFT65B bank-failure early-warning pipeline.

Every script imports paths, the field lists and the schema contract from here, so a
design change (e.g. adding a field) is made in exactly one place.

Environment overrides (used only to dry-run the code on another download):
    BFT_RAW  folder of the raw download   (default: data_raw/fdic_full)
    BFT_OUT  folder for data/ and reports/ (default: repository root)
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

SEED = 42

# ----------------------------------------------------------------------------
# Paths (data folders are git-ignored: the exam forbids committing data)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
_out = Path(os.environ.get("BFT_OUT", ROOT))
DATA_RAW = Path(os.environ.get("BFT_RAW", ROOT / "data_raw" / "fdic_full"))  # Layer 0, as downloaded
DATA = _out / "data"
DATA_BRONZE = DATA / "bronze"     # Layer 1: typed, schema-enforced, partitioned (Part 1)
DATA_SILVER = DATA / "silver"     # Layer 2: clean bank-quarter panel (Part 2)
DATA_GOLD = DATA / "gold"         # Layer 3: feature table + label (Part 4)
BENCH_DIR = DATA / "_bench"       # scratch copies for the storage benchmark
REPORTS = _out / "reports"        # tables, figures and JSON measurements for the report

# ----------------------------------------------------------------------------
# Scope fixed by the exam
# ----------------------------------------------------------------------------
START_Q = "1984-03-31"
END_Q = "2026-06-30"
LAST_MODEL_Q = "2025-06-30"       # labels after this quarter are incomplete
API_BASE = "https://api.fdic.gov/banks"

# ----------------------------------------------------------------------------
# Field lists, grouped by the CAMELS dimension they proxy.
# Must equal the fields requested by fdic_download_full.py (p1_verify.py checks it).
# ----------------------------------------------------------------------------
KEYS = ["CERT", "REPDTE", "STNAME"]

# Whole-number amounts in thousands of USD. Int64, not Int32: ASSET alone reaches about
# 4.1e9 (USD 4.1 trillion), above the Int32 maximum of 2.1e9. NUMEMP is a headcount.
INT_FIELDS = [
    # capital and size (C)
    "ASSET", "EQ", "LIAB", "NUMEMP",
    # asset quality and loan book (A)
    "LNLSGR", "LNLSNET", "LNRE", "LNRECONS", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    # earnings, reported year-to-date (E)
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    # liquidity and funding (L)
    "DEP", "DEPDOM", "COREDEP", "VOLIAB", "BRO", "CHBAL", "SC", "FREPP", "OTHBFHLB",
]
# Ratios in percent.
RATIO_FIELDS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ", "NCLNLSR", "NTLNLSR", "LNLSDEPR"]
FIN_FIELDS = KEYS + INT_FIELDS + RATIO_FIELDS

# Schema contract of the bronze financials table (YEAR is the hive partition key).
FIN_SCHEMA: dict[str, pl.DataType] = {
    "CERT": pl.Int32, "REPDTE": pl.Date, "STNAME": pl.String,
    **{c: pl.Int64 for c in INT_FIELDS},
    **{c: pl.Float64 for c in RATIO_FIELDS},
}
PARTITION_KEY = "YEAR"
SORT_KEYS = ["CERT", "REPDTE"]

# Chosen physical layout. The benchmark measures these choices against alternatives.
COMPRESSION, COMPRESSION_LEVEL = "zstd", 3
ROW_GROUP_SIZE = 16_384

# ----------------------------------------------------------------------------
# Part 2: panel rules
# ----------------------------------------------------------------------------
SCOPE_START = "1984-01-01"        # failures before the first report quarter cannot be labelled
LABEL_HORIZON_Q = 4               # y = 1 if the bank fails in (q, q + 4 quarters]

# Income-statement items are cumulative within the calendar year (year-to-date).
YTD_FLOWS = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]

# Balances that cannot be negative. A negative value is an error, not a signal. EQ, flows and
# LNLSNET (net of reserves, can fall below zero when loans are tiny) are deliberately left out.
NONNEG_STOCKS = ["LIAB", "DEP", "DEPDOM", "COREDEP", "VOLIAB", "BRO", "CHBAL", "SC", "FREPP", "OTHBFHLB",
                 "LNLSGR", "LNRE", "LNRECONS", "LNCI", "LNCON", "LNATRES", "NCLNLS", "P3ASSET", "P9ASSET",
                 "NAASSET", "ORE", "NUMEMP"]

# Meaning of Institutions.CHANGEC1, from the FDIC /history endpoint (CHANGECODE_DESC), 2026-11-07.
# Used only to describe NON-failure exits. Failure itself is defined by the Failures table.
EXIT_CODE_CLASS = {
    221: "merger", 222: "merger", 223: "merger", 224: "merger", 225: "merger",
    213: "assisted_merger",
    240: "voluntary_closing",
    211: "distress_no_failure_record",   # Failure - Whole Institution
    215: "distress_no_failure_record",   # Failure Multiple Acquirer
    216: "distress_no_failure_record",   # Bridge Bank Resolution
    217: "distress_no_failure_record",   # Passthrough Receivership/Conservatorship Resolution
    230: "distress_no_failure_record",   # Closing - Failure Payoff
    235: "distress_no_failure_record",   # RTC Supervised Payoffs, Liquidations, and Closings
}

# ----------------------------------------------------------------------------
# Part 4: time split. Quarters are report dates, so q = 2011-12-31 is 2011Q4.
#
#   development  1984Q1-2011Q4     labels of the last quarter look ahead to 2012Q4
#   gap          2012Q1-2012Q4     four quarters, used by neither side
#   test         2013Q1-2025Q2     used once, in Part 5
#
# Inside development, Part 4 fits and checks preprocessing on one split with its own gap. Part 5
# adds rolling folds. y-driven feature screening uses only PIPE_TRAIN_END and earlier.
# ----------------------------------------------------------------------------
DEV_END = "2011-12-31"
TEST_START = "2013-03-31"
TEST_END = LAST_MODEL_Q
PIPE_TRAIN_START = "1986-03-31"   # the 8-quarter windows need 1984-85 as history
PIPE_TRAIN_END = "2006-12-31"
PIPE_VAL_START = "2008-03-31"
PIPE_VAL_END = DEV_END
WINSOR_LOWER, WINSOR_UPPER = 0.005, 0.995

# ----------------------------------------------------------------------------
# Part 5: rolling folds inside development, each with a 4-quarter gap.
#   F1 a crisis the training data has barely seen         (696 failed banks)
#   F2 a quiet period, the closest match to the test      (71 failed banks)
#   F3 a later crisis; the only fold not used to screen features (463 failed banks)
# ----------------------------------------------------------------------------
FOLDS = {
    "F1 crisis 1990-91": {"train": ("1986-03-31", "1988-12-31"), "val": ("1990-03-31", "1991-12-31")},
    "F2 quiet 1994-2006": {"train": ("1986-03-31", "1992-12-31"), "val": ("1994-03-31", "2006-12-31")},
    "F3 crisis 2008-11": {"train": ("1986-03-31", "2006-12-31"), "val": ("2008-03-31", "2011-12-31")},
}
FINAL_TRAIN = ("1986-03-31", DEV_END)
# A report filed about 45 days after quarter end cannot trigger action before a failure inside that
# window. An assumption from the usual Call Report deadlines; the sensitivity is reported.
ACTION_LAG_DAYS = 45
CAPACITY_SHARE = 0.01             # alternative policy: review the riskiest 1% of banks each quarter
# Assumptions the data cannot settle. Everything else in the cost model comes from the data.
EXAM_PERSON_WEEKS = 16            # 4 examiners for 4 weeks
