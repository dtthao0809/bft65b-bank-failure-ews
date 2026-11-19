"""
Part 5c - A short Spark MLlib demo on fold F3: the same features, window and preprocessing idea as scikit-learn.

Spark runs locally here, so this shows the API and that the numbers agree. It is not a claim that Spark is
needed: Part 3 shows that this data does not need a cluster.

Preprocessing is fitted on the training window only, as in Part 4: quantile caps and medians are computed
there and applied to validation. Uses a logistic model with the same penalty as scikit-learn, so the two
should nearly agree, and a small gradient-boosted model.

scikit-learn minimises  C x sum(loss) + 0.5 x |w|^2
Spark minimises         mean(loss) + lambda x 0.5 x |w|^2,  so lambda = 1 / (C x n)

Usage:  python src/p5_spark.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p4_pipeline as PP  # noqa: E402
import p5_common as M  # noqa: E402
from p3_spark_session import get_spark  # noqa: E402

FOLD = "F3 crisis 2008-11"


def main() -> None:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    sc = PP.schema()
    w = C.FOLDS[FOLD]
    num, flags = sc["numeric"], sc["flags"]
    t0 = time.perf_counter()
    spark = get_spark(threads=6, driver_memory="3g", shuffle_partitions=12, app="bft65b-mllib", extra={"spark.local.dir": str(C.DATA / "_p3" / "_spark_tmp")})
    gold = (spark.read.parquet(str(C.DATA_GOLD / "features")).filter(F.col("y").isNotNull())
            .select("CERT", "REPDTE", *num, *[F.col(f).cast("double").alias(f) for f in flags], F.col("y").cast("double").alias("label")))
    win = lambda a, b: gold.filter(F.col("REPDTE").between(F.lit(date.fromisoformat(a)), F.lit(date.fromisoformat(b))))  # noqa: E731
    train, val = win(*w["train"]).persist(), win(*w["val"]).persist()
    n = train.count()
    # Caps from the training window. approxQuantile with a small error: exact quantiles need a full sort.
    bounds = train.approxQuantile(num, [C.WINSOR_LOWER, C.WINSOR_UPPER], 0.0005)
    def cap(df):
        return df.withColumns({c: F.least(F.greatest(F.col(c), F.lit(lo)), F.lit(hi)) for c, (lo, hi) in zip(num, bounds)})

    prep = [Imputer(strategy="median", inputCols=num, outputCols=num),
            VectorAssembler(inputCols=num + flags, outputCol="raw"),
            StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)]
    logit_c = json.loads((C.REPORTS / "p5_frozen.json").read_text()).get("chosen", {}).get("C", 1.0)
    models = {"Spark logistic": LogisticRegression(featuresCol="features", labelCol="label", regParam=1 / (logit_c * n), elasticNetParam=0.0, standardization=False, maxIter=200, tol=1e-7),
              "Spark GBT, 30 trees, depth 4": GBTClassifier(featuresCol="raw", labelCol="label", maxIter=30, maxDepth=4, stepSize=0.1, maxBins=64, seed=C.SEED, subsamplingRate=1.0)}
    rows = []
    for name, est in models.items():
        t = time.perf_counter()
        fitted = Pipeline(stages=[*prep, est]).fit(cap(train))
        fit_s = time.perf_counter() - t
        pred = fitted.transform(cap(val)).select("CERT", "REPDTE", "label", vector_to_array("probability")[1].alias("p")).toArrow()
        d = pl.from_arrow(pred).sort("CERT", "REPDTE")
        y, p = d["label"].to_numpy().astype(int), d["p"].to_numpy()
        rows.append({"engine": name, "fit_s": round(fit_s, 1), "ap": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)), "rows_train": n, "rows_val": y.size})
        print(rows[-1], flush=True)
        if name == "Spark logistic":
            spark_p = d
            plan = spark._jvm.PythonSQLUtils.explainString(fitted.transform(cap(val))._jdf.queryExecution(), "simple")
            (C.REPORTS / "p5_plan_mllib.txt").write_text(plan, encoding="utf-8")
    spark.stop()

    # The same logistic in scikit-learn.
    Xtr, ytr, _ = PP.load_xy(*w["train"])
    Xva, yva, mva = PP.load_xy(*w["val"])
    t = time.perf_counter()
    sk = M.make_model({"kind": "logit", "C": logit_c}).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t
    p = sk.predict_proba(Xva)[:, 1]
    rows.append({"engine": "scikit-learn logistic", "fit_s": round(fit_s, 1), "ap": float(average_precision_score(yva, p)), "roc_auc": float(roc_auc_score(yva, p)), "rows_train": len(ytr), "rows_val": len(yva)})
    j = mva.with_columns(pl.Series("sk", p)).join(spark_p, on=["CERT", "REPDTE"])
    rank = j.select(pl.corr(pl.col("sk").rank(), pl.col("p").rank())).item()
    t = pl.DataFrame(rows)
    t.write_csv(C.REPORTS / "p5_spark_mllib.csv", float_precision=5)
    facts = {"fold": FOLD, "rows_matched": j.height, "rank_corr_spark_vs_sklearn": rank, "max_abs_prob_diff": j.select((pl.col("sk") - pl.col("p")).abs().max()).item(),
             "total_s": round(time.perf_counter() - t0, 1), "logit_C": logit_c, "lambda": 1 / (logit_c * n)}
    (C.REPORTS / "p5_spark_mllib.json").write_text(json.dumps(facts, indent=1))
    print(t)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
