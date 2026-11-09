"""Operating point on a validation run: review share v target accuracy, per-item rules (val only).

    python -m hs_rag_classifier.eval.operating_point [--run RUN_ID] [--out DIR] [--no-mlflow]

Reads one finished validation run's saved predictions (MLflow predictions.parquet). Refuses any
run that is not a rulings validation run or that appears in test_runs.jsonl (check_val_only),
so no test-set prediction is ever opened. Never calls a model, a server or an arm.

Confidence and orders are the frozen review metrics' (arms.a3_pipeline, analysis.a3_agreement):
the prediction's confidence (the A4 joint probability at the refitted temperatures); policy (b),
the headline, reviews every A4/B1-ex cn8 disagreement first, then the agreements, each lowest
confidence first, ties by item id; policy (a) is confidence only.

Framing 1 (fixed share): for review shares 0-70% in 5-point steps, the accuracy of the
auto-accepted items and their number, with 95% intervals from eval.metrics.bootstrap_samples
(the split's near-duplicate clusters, 1,000 resamples, seed 20261021), for both orders and b - a.

Framing 2 (target accuracy): for each target, the smallest review share whose auto-accepted items
reach it, and the per-item rule that produces it:
    (a)  auto-accept when confidence >= t.
    (b)  review a disagreement when confidence < t_d and an agreement when confidence < t_a;
         a (b) cut inside the disagreements has t_a = 0 (no agreement reviewed), a cut inside the
         agreements has t_d = inf (every disagreement reviewed). The rule is exact (its review set
         is the order's first k items) whenever no confidence ties straddle the cut; "exact"
         records it. A rule is applied to each item alone, without ranking the batch.
Stability: per target, over the same resamples, the review share needed (median, 95% range) and
the accuracy of the point rule (share of resamples at or above the target).

HS6 carry-over (still val only): the same framings at HS6 with stage-1 confidence
(eval.metrics_hs6, A4's frozen T1), as HSCodeComp is scored, so its thresholds are fixed on val.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.analysis import a3_agreement as AG
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.items import DATA_ROOT, sha256_file

VAL_RUN = "04fee43dcca547e59f7440fe989f0362"  # A4 val reproduction, config 5da7ac1452b3
SHARES = tuple(round(0.05 * i, 2) for i in range(15))  # 0% to 70%
TARGETS = (0.75, 0.80, 0.85, 0.90)
POLICIES = ("b", "a")
OUT = DATA_ROOT / "eval" / "operating_point" / "val"
EXPERIMENT = "operating-point"


class NotValError(RuntimeError):
    """The input is not a rulings validation run (test predictions are never opened here)."""


def check_val_only(run_id: str, tags: Mapping[str, str], test_log: Path) -> None:
    if tags.get("split") != "val" or tags.get("dataset") != "rulings":
        raise NotValError(
            f"run {run_id} is {tags.get('dataset')} {tags.get('split')}, not rulings val"
        )
    if test_log.exists() and run_id in test_log.read_text(encoding="utf-8"):
        raise NotValError(f"run {run_id} is a test run (listed in {test_log.name})")


# --- orders and rules --------------------------------------------------------------------


def arrays(confidence, correct, agree, item_ids) -> dict[str, np.ndarray]:
    return {"id_rank": AG.id_rank(item_ids), "confidence": np.asarray(confidence, float),
            "correct": np.asarray(correct, float), "agree_cn8": np.asarray(agree, bool),
            "oof": np.zeros(len(confidence))}  # fmt: skip


def accepted(order: np.ndarray, correct: np.ndarray, k: int) -> tuple[float, int]:
    rest = np.asarray(correct, float)[order[k:]]
    return (float(rest.mean()) if len(rest) else math.nan), len(rest)


def rule_for_cut(policy: str, a: Mapping[str, np.ndarray], k: int) -> dict[str, float]:
    """The per-item thresholds whose review set is the policy's first k items (if exact)."""
    order = AG.review_order(policy, a)
    conf = a["confidence"]
    nxt = float(conf[order[k]]) if k < len(order) else math.inf
    if policy == "a":
        return {"t": nxt}
    dis = ~a["agree_cn8"]
    nd = int(dis.sum())
    if k <= nd:
        t_d = float(conf[order[k]]) if k < nd else math.inf
        return {"t_d": t_d, "t_a": 0.0}
    return {"t_d": math.inf, "t_a": nxt}


def apply_rule(policy: str, rule: Mapping[str, float], a: Mapping[str, np.ndarray]) -> np.ndarray:
    """Boolean mask of reviewed items, each item judged alone."""
    conf = a["confidence"]
    if policy == "a":
        return conf < rule["t"]
    dis = ~a["agree_cn8"]
    return (dis & (conf < rule["t_d"])) | (~dis & (conf < rule["t_a"]))


def target_rule(policy: str, a: Mapping[str, np.ndarray], target: float) -> dict[str, Any]:
    """Smallest review share reaching target, its per-item rule and whether the rule is exact."""
    order = AG.review_order(policy, a)
    n = len(order)
    acc = AG.accepted_accuracy(order, a["correct"])
    hit = np.nonzero(acc >= target - 1e-12)[0]
    if not len(hit):
        return {"policy": policy, "target": target, "reachable": False}
    k = int(hit[0])
    rule = rule_for_cut(policy, a, k)
    rev = apply_rule(policy, rule, a)
    first = np.zeros(n, bool)
    first[order[:k]] = True
    acc_rule = float(np.asarray(a["correct"], float)[~rev].mean()) if (~rev).any() else math.nan
    return {"policy": policy, "target": target, "reachable": True, "k_reviewed": k,
            "review_share": k / n, "accuracy": float(acc[k]), "n_accepted": n - k, **rule,
            "exact": bool(np.array_equal(rev, first)), "rule_review_share": float(rev.mean()),
            "rule_accuracy": acc_rule}  # fmt: skip


# --- tables ------------------------------------------------------------------------------


def _boot(a: Mapping[str, np.ndarray], cluster: Sequence) -> pd.DataFrame:
    return pd.DataFrame(
        {"cluster_id": np.asarray(cluster), **{k: np.asarray(v) for k, v in a.items()}}
    )


def share_table(a, cluster, shares=SHARES, n_resamples=M.BOOTSTRAP_RESAMPLES,
                seed=M.BOOTSTRAP_SEED) -> pd.DataFrame:  # fmt: skip
    n = len(a["confidence"])
    stats = {}
    for s in shares:
        k = AG.n_reviewed(s, n)
        for p in POLICIES:
            stats[f"{p}@{s:.2f}"] = lambda x, p=p, s=s: AG.accuracy_at(
                AG.review_order(p, x), x["correct"], s
            )
        stats[f"b-a@{s:.2f}"] = lambda x, s=s: (AG.accuracy_at(AG.review_order("b", x), x["correct"], s)
                                               - AG.accuracy_at(AG.review_order("a", x), x["correct"], s))  # fmt: skip
    samples = M.bootstrap_samples(_boot(a, cluster), stats, n_resamples, seed)
    rows = []
    for s in shares:
        k = AG.n_reviewed(s, n)
        row = {"review_share": s, "n_reviewed": k, "n_accepted": n - k}
        for p in POLICIES:
            acc, _ = accepted(AG.review_order(p, a), a["correct"], k)
            v = samples[f"{p}@{s:.2f}"]
            row.update({f"{p}_accuracy": acc, f"{p}_low": float(np.nanpercentile(v, 2.5)),
                        f"{p}_high": float(np.nanpercentile(v, 97.5))})  # fmt: skip
        d = samples[f"b-a@{s:.2f}"]
        row.update({"b_minus_a": row["b_accuracy"] - row["a_accuracy"], "b_minus_a_low": float(np.nanpercentile(d, 2.5)),
                    "b_minus_a_high": float(np.nanpercentile(d, 97.5))})  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


def target_table(a, targets=TARGETS) -> pd.DataFrame:
    return pd.DataFrame([target_rule(p, a, t) for p in POLICIES for t in targets])


def stability(a, cluster, rules: pd.DataFrame, n_resamples=M.BOOTSTRAP_RESAMPLES,
              seed=M.BOOTSTRAP_SEED) -> pd.DataFrame:  # fmt: skip
    stats = {}
    for r in rules.itertuples(index=False):
        if not r.reachable:
            continue
        key = f"{r.policy}@{r.target:.2f}"

        def need(x, p=r.policy, t=r.target):
            v = AG.rate_for(AG.review_order(p, x), x["correct"], t)
            return math.nan if v is None else v

        rule = {
            k: getattr(r, k)
            for k in ("t", "t_d", "t_a")
            if hasattr(r, k) and not pd.isna(getattr(r, k))
        }

        def acc(x, p=r.policy, rule=rule):
            rev = apply_rule(p, rule, x)
            return float(x["correct"][~rev].mean()) if (~rev).any() else math.nan

        def share(x, p=r.policy, rule=rule):
            return float(apply_rule(p, rule, x).mean())

        stats[f"need|{key}"], stats[f"acc|{key}"], stats[f"share|{key}"] = need, acc, share
    samples = M.bootstrap_samples(_boot(a, cluster), stats, n_resamples, seed)
    rows = []
    for r in rules.itertuples(index=False):
        if not r.reachable:
            rows.append({"policy": r.policy, "target": r.target, "reachable": False})
            continue
        key = f"{r.policy}@{r.target:.2f}"
        need, acc, share = samples[f"need|{key}"], samples[f"acc|{key}"], samples[f"share|{key}"]
        rows.append({"policy": r.policy, "target": r.target, "reachable": True,
                     "need_share_median": float(np.nanmedian(need)),
                     "need_share_low": float(np.nanpercentile(need, 2.5)),
                     "need_share_high": float(np.nanpercentile(need, 97.5)),
                     "need_unreachable_resamples": int(np.isnan(need).sum()),
                     "rule_accuracy_median": float(np.nanmedian(acc)),
                     "rule_accuracy_low": float(np.nanpercentile(acc, 2.5)),
                     "rule_accuracy_high": float(np.nanpercentile(acc, 97.5)),
                     "rule_share_at_or_above_target": float(np.mean(acc >= r.target - 1e-12)),
                     "rule_review_share_low": float(np.nanpercentile(share, 2.5)),
                     "rule_review_share_high": float(np.nanpercentile(share, 97.5))})  # fmt: skip
    return pd.DataFrame(rows)


def analyse(a, cluster) -> dict[str, pd.DataFrame]:
    rules = target_table(a)
    return {
        "fixed_share": share_table(a, cluster),
        "targets": rules,
        "stability": stability(a, cluster, rules),
    }


# --- run ---------------------------------------------------------------------------------


def run(run_id: str = VAL_RUN, out: Path = OUT, log_mlflow: bool = True) -> dict[str, Any]:
    import mlflow

    from hs_rag_classifier.analysis.a3_compare import load_predictions
    from hs_rag_classifier.arms import a4 as A4
    from hs_rag_classifier.eval import metrics_hs6 as MH
    from hs_rag_classifier.eval import runner as R
    from hs_rag_classifier.eval.items import EvalItem, load

    mlflow.set_tracking_uri(R.tracking_uri())
    r = mlflow.get_run(run_id)
    check_val_only(run_id, r.data.tags, R.TEST_LOG)
    path = Path(r.info.artifact_uri.removeprefix("file://")) / "predictions.parquet"
    f = pd.read_parquet(path)
    raw = A4.frozen()
    exp = raw["expected_val"]
    a = arrays(f["confidence"], f["correct_8"], f["agree_cn8"], f["item_id"])
    pb = AG.policy_metrics(AG.review_order("b", a), a["correct"])
    repro = {"cn8_top1": (float(f["correct_8"].mean()), exp["cn8_top1"]),
             **{f"b_acc_{k}": (pb["accuracy_at_review"][k], v) for k, v in exp["policy_b"]["accuracy_at_review"].items()},
             "b_aurc": (pb["aurc"], exp["policy_b"]["aurc"])}  # fmt: skip
    bad = {k: v for k, v in repro.items() if v[0] != v[1]}
    if bad:
        raise RuntimeError(f"val reproduction failed: {bad}")
    cn8 = analyse(a, f["cluster_id"])
    # HS6 carry-over on val: stage-1 confidence at A4's frozen T1, HS6 correctness and agreement
    valid8, valid10 = R.valid_codes()
    items = [EvalItem(**{**i.model_dump(), "label_code": i.label_code[:6], "label_depth": 6})
             for i in load("rulings", "val").items]  # fmt: skip
    t1 = float(A4.load_config()["a2"]["temperature"]["stage1"])
    f6 = MH.score(items, load_predictions(path), valid8, valid10, t1)
    if float(f6["correct_6"].mean()) != float(
        f.set_index("item_id").loc[f6["item_id"], "correct_6"].mean()
    ):
        raise RuntimeError("HS6 top-1 from metrics_hs6 differs from the run's correct_6")
    a6 = arrays(f6["confidence_hs6"], f6["correct_6"], f6["agree_hs6"], f6["item_id"])
    hs6 = analyse(a6, f6["cluster_id"])
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for level, res in (("cn8", cn8), ("hs6_stage1", hs6)):
        for name, df in res.items():
            p = out / f"{level}_{name}.csv"
            df.to_csv(p, index=False)
            files.append(p)
    summary = {"run_id": run_id, "split": r.data.tags.get("split"), "config_hash": r.data.params.get("config_hash"),
               "input": {"path": str(path), "sha256": sha256_file(path)},
               "test_runs_jsonl_sha256": sha256_file(R.TEST_LOG), "n_items": len(f),
               "n_clusters": int(f["cluster_id"].nunique()), "distinct_confidences": int(np.unique(a["confidence"]).size),
               "reproduction": {k: {"got": v[0], "expected": v[1]} for k, v in repro.items()},
               "hs6_carry_over": {"t1": t1, "hs6_top1": float(f6["correct_6"].mean()),
                                  "agree_hs6": float(f6["agree_hs6"].mean()),
                                  "ece_stage1": M.ece(a6["confidence"], a6["correct"])},
               "cn8": {k: v.to_dict(orient="records") for k, v in cn8.items()},
               "hs6_stage1": {k: v.to_dict(orient="records") for k, v in hs6.items()},
               "bootstrap": {"resamples": M.BOOTSTRAP_RESAMPLES, "seed": M.BOOTSTRAP_SEED, "unit": "cluster_id"}}  # fmt: skip
    sp = out / "summary.json"
    sp.write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    files.append(sp)
    if log_mlflow:
        exp_id = R._experiment_id(EXPERIMENT, R.MLFLOW_DIR / "artifacts")
        with mlflow.start_run(experiment_id=exp_id, run_name="a4_val_operating_point") as mr:
            mlflow.set_tags({"analysis": "true", "test_run": "false", "split": "val",
                             "git_commit": R.git_info()["commit"]})  # fmt: skip
            mlflow.log_params(
                {"input_run": run_id, "input_sha256": summary["input"]["sha256"][:16]}
            )
            for f_ in files:
                mlflow.log_artifact(str(f_))
            summary["mlflow_run_id"] = mr.info.run_id
        sp.write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="operating_point", description=__doc__.splitlines()[0])
    p.add_argument("--run", default=VAL_RUN)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    try:
        s = run(args.run, args.out, not args.no_mlflow)
    except NotValError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"reproduction": s["reproduction"], "mlflow_run_id": s.get("mlflow_run_id")}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
