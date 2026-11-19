from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import p4_pipeline as PP  # noqa: E402
import p5_common as M  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

RES: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RES.append({"test": name, "result": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def qi(s: str) -> int:
    d = date.fromisoformat(s)
    return d.year * 4 + (d.month - 1) // 3


def main() -> None:
    dev_end = date.fromisoformat(C.DEV_END)
    for fold, w in C.FOLDS.items():
        gap = qi(w["val"][0]) - qi(w["train"][1]) - 1
        check(f"{fold}: 4-quarter gap, validation after training, inside development", gap >= 4 and date.fromisoformat(w["val"][1]) <= dev_end, f"gap {gap}")

    # Only p5_test.py may ask for the test period.
    src = Path(__file__).resolve().parent
    users = [p.name for p in src.glob("*.py") if p.name != "p5_checks.py" and re.search(r"allow_test\s*=\s*True", p.read_text(encoding="utf-8"))]
    check("only p5_test.py asks for the test period", users == ["p5_test.py"], str(users))

    frozen = json.loads((C.REPORTS / "p5_frozen.json").read_text())
    h = hashlib.sha256((C.DATA_GOLD / "p5_final_models.joblib").read_bytes()).hexdigest()
    check("saved models match the frozen hash", h == frozen["models_sha256"], h[:12])
    check("the frozen training window ends with development", frozen["train_window"][1] == C.DEV_END)
    res = C.REPORTS / "p5_test_results.json"
    if res.exists():
        lock = json.loads((C.REPORTS / "p5_test_lock.json").read_text())
        check("the recorded results match their lock", hashlib.sha256(res.read_bytes()).hexdigest() == lock["results_sha256"], lock["evaluated_at_utc"])
        check("the frozen model and threshold are the ones the test used", lock["model"] == frozen["chosen"]["name"] and abs(lock["threshold"] - frozen["threshold"]) < 1e-12, f"{lock['threshold']:.6f}")
        r = subprocess.run([sys.executable, str(src / "p5_test.py")], capture_output=True, text=True, env={**os.environ})
        check("a second test run is refused", r.returncode != 0 and "has been used" in (r.stdout + r.stderr))

    # Controls on the fold that did not help choose features.
    w = C.FOLDS["F3 crisis 2008-11"]
    Xtr, ytr, _ = PP.load_xy(*w["train"], meta=M.META)
    Xva, yva, _ = PP.load_xy(*w["val"], meta=M.META)
    rng = np.random.default_rng(C.SEED)
    spec = {"kind": "logit", "C": 1.0}
    m = M.make_model(spec).fit(Xtr, rng.permutation(ytr))
    ap_shuf = float(average_precision_score(yva, m.predict_proba(Xva)[:, 1]))
    prev = float(yva.mean())
    real = float(average_precision_score(yva, M.make_model(spec).fit(Xtr, ytr).predict_proba(Xva)[:, 1]))
    check("control: with shuffled training labels, AP collapses far below the real model", ap_shuf < real / 3, f"prevalence {prev:.4f}, shuffled {ap_shuf:.4f}, real {real:.4f}")

    # The cost model on a case small enough to work out by hand.
    toy = pl.DataFrame({"CERT": [1, 1, 2, 3, 4], "REPDTE": [date(2009, 3, 31), date(2009, 6, 30), date(2009, 6, 30), date(2009, 6, 30), date(2009, 6, 30)],
                        "y": [1, 1, 1, 0, 0], "score": [0.9, 0.2, 0.5, 0.8, 0.1], "_days_to_fail": [200, 30, 40, None, None]})
    losses = pl.DataFrame({"CERT": [1, 2], "loss": [1000.0, 500.0]})
    # t = 0.8: alarms on rows 0.9 and 0.8 = 2 exams x 10 = 20. Bank 1 caught, saves 25% of 1000. Bank 2 missed.
    c = M.cost_curve(toy, losses, {2009: 10.0}, 0.25, thresholds=np.array([0.8])).row(0, named=True)
    check("cost model: worked example at t = 0.8", c["alarms"] == 2 and c["exam_cost"] == 20 and c["failure_cost"] == 750 + 500 and c["banks_caught"] == 1, str(c["total_cost"]))
    # t = 0.5: bank 2's only alarm comes 40 days before failure, too late to act.
    c = M.cost_curve(toy, losses, {2009: 10.0}, 0.25, thresholds=np.array([0.5])).row(0, named=True)
    check("cost model: an alarm inside the action lag does not count as caught", c["alarms"] == 3 and c["banks_caught"] == 1, f"alarms {c['alarms']}")
    c = M.cost_curve(toy, losses, {2009: 10.0}, 0.25, thresholds=np.array([0.5]), lag_days=0).row(0, named=True)
    check("cost model: with no lag the same alarm counts", c["banks_caught"] == 2)
    # A failed bank counts once: lower the threshold so both of bank 1's rows alarm.
    c = M.cost_curve(toy, losses, {2009: 10.0}, 0.25, thresholds=np.array([0.2]), lag_days=0).row(0, named=True)
    check("cost model: a failed bank's loss counts once", c["failure_cost"] == 750 + 375)

    # The fold table covers what the selection rule says it does.
    t = pl.read_csv(C.REPORTS / "p5_cv_results.csv", infer_schema_length=None)
    chosen = frozen["chosen"]["name"]
    check("the chosen model was evaluated on all three folds", t.filter(pl.col("model") == chosen).height == 3)
    best = t.filter(~pl.col("model").str.starts_with("sens:"), ~pl.col("model").str.starts_with("B")).group_by("model").agg(pl.col("ap").mean())["ap"].max()
    check("the chosen model is within 1% of the best candidate", frozen["selection_mean_ap"] >= 0.99 * best - 1e-12, f"{frozen['selection_mean_ap']:.4f} vs {best:.4f}")

    out = pl.DataFrame(RES)
    out.write_csv(C.REPORTS / "p5_checks.csv")
    fails = out.filter(pl.col("result") == "FAIL").height
    print(f"{out.height - fails}/{out.height} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
