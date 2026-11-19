from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# Measured times, ids, paths or machine state.
VARIABLE = ("p1_bench_env", "p1_polars_plans", "p1_storage", "p1_bronze_build", "p1_completeness", "p1_control_totals", "p3_bench_env", "p3_benchmark_runs", "p3_engines",
            "p3_threads", "p3_scale", "p3_spark_settings", "p3_cluster_case", "p3_plan", "p3_polars_", "p3_optimizer", "p3_spark_stages", "p3_spark_exchanges", "p3_spark_x8_crash", "p5_plan_mllib", "p5_spark_mllib",
            "p6_")  # p6 quotes the benchmark timings

TOL = 1e-9


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same_csv(a: Path, b: Path, skip: set[str]) -> tuple[bool, str]:
    x, y = pl.read_csv(a, infer_schema_length=None), pl.read_csv(b, infer_schema_length=None)
    if x.columns != y.columns or x.height != y.height:
        return False, f"shape {x.shape} vs {y.shape}"
    note = ""
    if not x.equals(y):
        # Rows tied on the sort key may come out in any order; compare the rows, not their order.
        keys = [c for c in x.columns if c not in skip]
        xs, ys = x.sort(keys, nulls_last=True), y.sort(keys, nulls_last=True)
        if not xs.equals(x) or not ys.equals(y):
            x, y, note = xs, ys, "same rows after sorting"
    bad = []
    for c in x.columns:
        if c in skip:
            continue
        u, v = x[c], y[c]
        if u.dtype.is_numeric() and v.dtype.is_numeric():
            d = (u.cast(pl.Float64) - v.cast(pl.Float64)).abs()
            au, av = u.cast(pl.Float64).abs().fill_null(0), v.cast(pl.Float64).abs().fill_null(0)
            scale = au.zip_with(au > av, av).clip(lower_bound=1.0)
            if ((d > TOL * scale).fill_null(False).any()) or (u.is_null() != v.is_null()).any():
                bad.append(c)
        elif not (u.cast(pl.String).eq_missing(v.cast(pl.String))).all():
            bad.append(c)
    return not bad, ",".join(bad) or note


def same_json(a: Path, b: Path) -> tuple[bool, str]:
    def walk(p, q, path=""):
        if isinstance(p, dict) and isinstance(q, dict):
            return [d for k in sorted(set(p) | set(q)) for d in walk(p.get(k), q.get(k), f"{path}/{k}")]
        if isinstance(p, list) and isinstance(q, list) and len(p) == len(q):
            return [d for i, (m, n) in enumerate(zip(p, q)) for d in walk(m, n, f"{path}[{i}]")]
        if isinstance(p, (int, float)) and isinstance(q, (int, float)) and not isinstance(p, bool):
            return [] if abs(p - q) <= TOL * max(abs(p), abs(q), 1) else [path]
        return [] if p == q else [path]

    bad = walk(json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8")))
    return not bad, ",".join(bad[:6])


def main() -> None:
    other = Path(sys.argv[1])
    rows = []
    for f in sorted(C.REPORTS.iterdir()):
        if f.suffix in (".png", ".dot") or f.name in ("repro_check.csv", "repro_status.csv"):
            continue
        g = other / "reports" / f.name
        kind = "variable" if f.name.startswith(VARIABLE) else "deterministic"
        if not g.exists():
            rows.append({"file": f.name, "kind": kind, "same": False, "detail": "missing in rerun"})
            continue
        if sha(f) == sha(g):
            ok, why = True, "bytes"
        elif f.suffix == ".csv":
            ok, why = same_csv(f, g, {"fit_s", "s_fit"})
        elif f.suffix == ".json":
            ok, why = same_json(f, g)
        else:
            ok, why = False, "text differs"
        rows.append({"file": f.name, "kind": kind, "same": ok, "detail": why})
    for layer in ("bronze", "silver", "gold/features"):
        a, b = C.DATA / layer, other / "data" / layer
        files = sorted(p.relative_to(a) for p in a.rglob("*.parquet"))
        differ = [str(p) for p in files if not (b / p).exists() or sha(a / p) != sha(b / p)]
        rows.append({"file": f"data/{layer} ({len(files)} files)", "kind": "deterministic", "same": not differ, "detail": ",".join(differ[:5])})
    t = pl.DataFrame(rows)
    t.write_csv(C.REPORTS / "repro_check.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=130, fmt_str_lengths=70):
        print(t.group_by("kind", "same").len().sort("kind", "same"))
        print(t.filter(~pl.col("same")))
    sys.exit(1 if t.filter(pl.col("kind") == "deterministic", ~pl.col("same")).height else 0)


if __name__ == "__main__":
    main()
