"""
Part 1b - Storage: Layer 0 (raw, one file per quarter, as received) -> Layer 1 "bronze".

Design (each choice is measured in p1_storage_benchmark.py):
  Format        Parquet: columnar, typed, carries min/max statistics
  Compression   ZSTD level 3
  Partitioning  hive style, YEAR=YYYY/part-0.parquet. The key lives in the path only, so no
                engine sees two conflicting copies of YEAR
  Sort order    (CERT, REPDTE) inside each year: row-group statistics on CERT let a single-bank
                lookup skip most of a file, and lags later read rows already in order
  Schema        an explicit contract in config.py, not inferred from the data

Bronze changes types and physical layout only. No value is cleaned here: nulls, zeros and the
9999-12-31 sentinels in Institutions stay as received and are handled in Part 2, so every
cleaning decision can be seen and audited in one place.

Every run proves that nothing was lost: row counts, null counts and column totals are compared
with the raw files, and the build fails if any of them differ.

Usage:  python src/p1_build_bronze.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


# ----------------------------------------------------------------------------
# Financials
# ----------------------------------------------------------------------------
def cast_quarter(df: pl.DataFrame) -> pl.DataFrame:
    """Force one raw quarter onto the contract. Raw types drift between quarters (a column is
    int64 when it has no nulls, double when it has some, null when it is empty) and a field
    can be missing entirely, so each file must be cast on its own before concatenating."""
    exprs = []
    for col, dtype in C.FIN_SCHEMA.items():
        if col not in df.columns:
            exprs.append(pl.lit(None, dtype=dtype).alias(col))
        elif col == "REPDTE":
            exprs.append(pl.col(col).cast(pl.String).str.to_date("%Y%m%d", strict=True))
        elif dtype == pl.Int64:
            # Safe because assert_no_fractions() has already rejected non-whole values.
            exprs.append(pl.col(col).cast(pl.Float64).cast(pl.Int64, strict=True).alias(col))
        else:
            exprs.append(pl.col(col).cast(dtype, strict=True).alias(col))
    return df.select(exprs)


def assert_no_fractions(df: pl.DataFrame, name: str) -> None:
    """Polars truncates float -> int even in strict mode, so fractions are checked explicitly."""
    cols = [c for c in C.INT_FIELDS if c in df.columns]
    bad = df.select([(pl.col(c).cast(pl.Float64) % 1 != 0).sum().alias(c) for c in cols]).row(0, named=True)
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise ValueError(f"{name}: fractional values in integer columns {bad}; revise the contract")


def write_hive(df: pl.DataFrame, target: Path) -> None:
    """Write YEAR=YYYY/part-0.parquet with the key in the path only."""
    tmp = target.with_name(target.name + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for (year,), part in df.partition_by(C.PARTITION_KEY, as_dict=True, include_key=False).items():
        folder = tmp / f"{C.PARTITION_KEY}={year}"
        folder.mkdir(parents=True)
        part.write_parquet(folder / "part-0.parquet", compression=C.COMPRESSION,
                           compression_level=C.COMPRESSION_LEVEL, statistics=True,
                           row_group_size=C.ROW_GROUP_SIZE)
    shutil.rmtree(target, ignore_errors=True)   # swap only after a complete write
    tmp.replace(target)


def build_financials() -> dict:
    t0 = time.perf_counter()
    files = sorted((C.DATA_RAW / "financials").glob("REPDTE=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw files in {C.DATA_RAW / 'financials'}")

    frames, raw_rows, raw_nulls, raw_sums = [], 0, {}, {}
    absent: dict[str, int] = {}
    for f in files:
        raw = pl.read_parquet(f)
        assert_no_fractions(raw, f.name)
        frames.append(cast_quarter(raw))
        raw_rows += raw.height
        for c in C.FIN_SCHEMA:
            if c not in raw.columns:
                absent[c] = absent.get(c, 0) + 1
                raw_nulls[c] = raw_nulls.get(c, 0) + raw.height
                continue
            raw_nulls[c] = raw_nulls.get(c, 0) + raw[c].null_count()
            if c in C.INT_FIELDS:
                raw_sums[c] = raw_sums.get(c, 0) + int(raw[c].cast(pl.Float64).sum())

    df = (pl.concat(frames, how="vertical")
          .with_columns(pl.col("REPDTE").dt.year().cast(pl.Int16).alias(C.PARTITION_KEY))
          .sort(C.PARTITION_KEY, *C.SORT_KEYS))

    # Reconciliation: raw vs typed. Any difference stops the build.
    assert df.height == raw_rows, "row count changed"
    typed_nulls = {c: df[c].null_count() for c in C.FIN_SCHEMA}
    assert typed_nulls == raw_nulls, {c: (raw_nulls[c], typed_nulls[c])
                                      for c in raw_nulls if raw_nulls[c] != typed_nulls[c]}
    typed_sums = {c: int(df[c].sum() or 0) for c in raw_sums}
    assert typed_sums == raw_sums, "a column total changed"
    assert df.select(pl.struct(C.SORT_KEYS).is_duplicated().sum()).item() == 0, "duplicate key"

    out = C.DATA_BRONZE / "financials"
    write_hive(df, out)

    back = pl.scan_parquet(out / "**" / "*.parquet", hive_partitioning=True)
    assert back.select(pl.len()).collect().item() == df.height, "bronze read-back differs"

    (C.DATA_BRONZE / "_schema_financials.json").write_text(json.dumps(
        {**{c: str(t) for c, t in C.FIN_SCHEMA.items()}, f"{C.PARTITION_KEY} (hive key)": "Int16"}, indent=1))
    size = sum(p.stat().st_size for p in out.rglob("*.parquet")) / 2**20
    res = {"rows": df.height, "contract_columns": len(C.FIN_SCHEMA),
           "year_partitions": df[C.PARTITION_KEY].n_unique(), "size_mb": round(size, 2),
           "seconds": round(time.perf_counter() - t0, 1),
           "quarters_where_field_absent": absent,
           "reconciliation": "rows, null counts and column totals equal raw"}
    print(f"financials: {res['rows']:,} rows, {res['year_partitions']} partitions, "
          f"{res['size_mb']} MB, {res['seconds']} s")
    return res


# ----------------------------------------------------------------------------
# Failures and Institutions. Small tables: one file each, no partitioning.
# ----------------------------------------------------------------------------
def to_date(col: str) -> pl.Expr:
    return pl.col(col).str.to_date("%m/%d/%Y", strict=True)


def build_small_tables() -> dict:
    fail = pl.read_parquet(C.DATA_RAW / "failures.parquet")
    fail = fail.with_columns(
        pl.col("CERT").cast(pl.Int32), to_date("FAILDATE"), pl.col("FAILYR").cast(pl.Int16),
        *[pl.col(c).cast(pl.Float64) for c in ("COST", "QBFASSET", "QBFDEP")],
    ).sort("FAILDATE", "CERT", "NAME")
    inst = pl.read_parquet(C.DATA_RAW / "institutions.parquet")
    inst = inst.with_columns(
        pl.col("CERT").cast(pl.Int32), pl.col("ACTIVE").cast(pl.Int8), pl.col("CHANGEC1").cast(pl.Int32),
        *[to_date(c) for c in ("ESTYMD", "ENDEFYMD", "INSDATE")],
    ).sort("CERT")
    res = {}
    for name, df in (("failures", fail), ("institutions", inst)):
        df.write_parquet(C.DATA_BRONZE / f"{name}.parquet", compression=C.COMPRESSION,
                         compression_level=C.COMPRESSION_LEVEL)
        res[name] = {"rows": df.height, "columns": df.width,
                     "schema": {c: str(t) for c, t in df.schema.items()}}
        print(f"{name}: {df.height:,} rows, {df.width} columns")
    return res


def main() -> None:
    C.DATA_BRONZE.mkdir(parents=True, exist_ok=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    res = {"financials": build_financials(), **build_small_tables(), "polars": pl.__version__,
           "compression": f"{C.COMPRESSION}-{C.COMPRESSION_LEVEL}", "row_group_size": C.ROW_GROUP_SIZE,
           "sort": C.SORT_KEYS}
    (C.REPORTS / "p1_bronze_build.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
