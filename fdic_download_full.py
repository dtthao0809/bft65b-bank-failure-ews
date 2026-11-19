"""
BFT65B - full raw download: course starter logic with an extended field list.

Reuses `fdic_download_starter.py` (paging, retry, per-quarter resume, completeness check)
and only changes the field lists and the output folder. It has its own folder because the
starter's resume logic checks the quarter, not the fields: rerunning it on `data_raw/fdic`
with more fields would skip every quarter and add nothing.

Usage (from this folder; resumable, just rerun if it stops):
    python fdic_download_full.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import fdic_download_starter as starter

OUT = Path(__file__).resolve().parent / "data_raw" / "fdic_full"
START, END = date(1984, 3, 31), date(2026, 6, 30)  # scope fixed by the exam

EXTRA_FIN = [
    "LIAB", "COREDEP", "VOLIAB", "FREPP", "OTHBFHLB", "NUMEMP",     # funding, liquidity and staff (src/config.py)
    "LNRECONS", "NCLNLSR", "NTLNLSR", "LNLSDEPR",                   # construction loans; FDIC ratios checked in Part 2
]
EXTRA_FAIL = ["RESTYPE1", "SAVR"]
EXTRA_INST = ["REGAGNT", "SPECGRP"]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    starter.FIN_FIELDS = [*starter.FIN_FIELDS, *EXTRA_FIN]
    starter.FAIL_FIELDS = [*starter.FAIL_FIELDS, *EXTRA_FAIL]
    starter.INST_FIELDS = [*starter.INST_FIELDS, *EXTRA_INST]
    fields = {"financials": starter.FIN_FIELDS, "failures": starter.FAIL_FIELDS,
              "institutions": starter.INST_FIELDS}

    OUT.mkdir(parents=True, exist_ok=True)
    # Refuse to resume into a folder written with a different field list.
    fields_path = OUT / "fields.json"
    if fields_path.exists() and json.loads(fields_path.read_text()) != fields:
        sys.exit(f"{fields_path} has a different field list; use a new folder.")
    fields_path.write_text(json.dumps(fields, indent=1))

    started, t0 = now_utc(), time.time()
    quarters = starter.quarter_ends(START, END)
    print(f"Downloading {len(quarters)} quarters into {OUT}\n", flush=True)
    starter.download_financials(OUT, quarters)
    print()
    starter.download_small_tables(OUT)
    ok = starter.check(OUT)

    (OUT / "download_log.json").write_text(json.dumps({
        "source": starter.BASE, "started_utc": started, "finished_utc": now_utc(),
        "minutes": round((time.time() - t0) / 60, 1), "ok": ok, "python": sys.version.split()[0],
        "pandas": pd.__version__, "requests": requests.__version__,
    }, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
