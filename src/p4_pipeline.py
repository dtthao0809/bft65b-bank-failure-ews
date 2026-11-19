from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Caps each column at quantiles learned in fit(). Nulls stay null for the imputer."""

    def __init__(self, lower: float = C.WINSOR_LOWER, upper: float = C.WINSOR_UPPER):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(a, self.lower, axis=0)
        self.upper_ = np.nanquantile(a, self.upper, axis=0)
        self.n_features_in_ = a.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def schema() -> dict:
    return json.loads((C.DATA_GOLD / "_schema_features.json").read_text())


def build_preprocessor(numeric: list[str], flags: list[str], scale: bool = True) -> ColumnTransformer:
    steps = [("winsor", Winsorizer()), ("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), numeric), ("flag", "passthrough", flags)], verbose_feature_names_out=False)


def load_xy(start: str, end: str, allow_test: bool = False, meta: tuple[str, ...] = ()) -> tuple:
    """Returns (X as pandas, y as numpy, meta as Polars) for labelled rows with start <= REPDTE <= end."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if e > date.fromisoformat(C.DEV_END) and not allow_test:
        raise PermissionError(f"Window ends {end}, after development. The test period is used once, in Part 5.")
    sc = schema()
    cols = sc["numeric"] + sc["flags"]
    d = (pl.scan_parquet(C.DATA_GOLD / "features" / "**" / "*.parquet", hive_partitioning=True, hive_schema={C.PARTITION_KEY: pl.Int16})
         .filter(pl.col("REPDTE").is_between(s, e), pl.col("y").is_not_null())
         .select("CERT", "REPDTE", *cols, "y", *meta).sort("CERT", "REPDTE").collect(engine="streaming"))
    return (d.select(cols).with_columns(pl.col(sc["flags"]).cast(pl.Float64)).to_pandas(), d["y"].to_numpy().astype(int),
            d.select("CERT", "REPDTE", *meta))


def psi(train: np.ndarray, other: np.ndarray, bins: int = 10) -> float:
    """Population stability index on training deciles. Rule of thumb: above 0.25 is a large shift."""
    edges = np.unique(np.nanquantile(train, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(train[~np.isnan(train)], edges)[0] / np.sum(~np.isnan(train))
    q = np.histogram(other[~np.isnan(other)], edges)[0] / np.sum(~np.isnan(other))
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def main() -> None:
    sc = schema()
    Xtr, ytr, _ = load_xy(C.PIPE_TRAIN_START, C.PIPE_TRAIN_END)
    Xva, yva, _ = load_xy(C.PIPE_VAL_START, C.PIPE_VAL_END)
    pre = build_preprocessor(sc["numeric"], sc["flags"]).fit(Xtr)
    joblib.dump(pre, C.DATA_GOLD / "preprocessor_part4.joblib")
    Ztr, Zva = pre.transform(Xtr), pre.transform(Xva)
    if np.isnan(Ztr).any() or np.isnan(Zva).any():
        raise SystemExit("nulls left after preprocessing")

    num = pre.named_transformers_["num"]
    win, imp, sca = num.named_steps["winsor"], num.named_steps["impute"], num.named_steps["scale"]
    # What fitting on training + validation would have learned instead: the leak this design prevents.
    both = np.vstack([Xtr[sc["numeric"]].to_numpy(dtype=float), Xva[sc["numeric"]].to_numpy(dtype=float)])
    leaked_up, leaked_med = np.nanquantile(both, C.WINSOR_UPPER, axis=0), np.nanmedian(both, axis=0)
    rows = []
    for i, f in enumerate(sc["numeric"]):
        tr, va = Xtr[f].to_numpy(dtype=float), Xva[f].to_numpy(dtype=float)
        rows.append({"feature": f, "lower": win.lower_[i], "upper": win.upper_[i], "median": imp.statistics_[i], "mean": sca.mean_[i], "std": sca.scale_[i],
                     "null_train_pct": np.isnan(tr).mean() * 100, "null_val_pct": np.isnan(va).mean() * 100,
                     "capped_val_pct": ((va < win.lower_[i]) | (va > win.upper_[i])).mean() * 100,
                     "psi_train_to_val": psi(tr, va), "upper_if_val_leaked": leaked_up[i], "median_if_val_leaked": leaked_med[i]})
    t = pl.DataFrame(rows).sort("psi_train_to_val", descending=True)
    t.write_csv(C.REPORTS / "p4_pipeline_params.csv", float_precision=5)
    facts = {"train": {"window": [C.PIPE_TRAIN_START, C.PIPE_TRAIN_END], "rows": len(ytr), "y1": int(ytr.sum())},
             "validation": {"window": [C.PIPE_VAL_START, C.PIPE_VAL_END], "rows": len(yva), "y1": int(yva.sum())},
             "features_out": Ztr.shape[1], "psi_over_0.25": t.filter(pl.col("psi_train_to_val") > 0.25)["feature"].to_list(),
             "max_capped_val_pct": float(t["capped_val_pct"].max())}
    (C.REPORTS / "p4_pipeline_facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))
    with pl.Config(tbl_rows=30, tbl_cols=12, tbl_width_chars=170, float_precision=4):
        print(t)


if __name__ == "__main__":
    main()
