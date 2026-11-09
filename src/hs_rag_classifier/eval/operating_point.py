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

Step 2c (python -m hs_rag_classifier.eval.operating_point fix-rule | apply --rule-commit C):
    fix-rule  Part A, val only (same guard): the lower-bound cuts for an 80% target (the
              smallest review share whose 5th-percentile resample accuracy is >= 0.80 and stays
              so at every larger share up to 80%), written to configs/operating_point_rule.json.
    apply     Part B, report only: reads thresholds only from that file, refuses unless the file
              is byte-identical to the one committed in C, and applies each rule once to A4's
              saved test predictions (rulings, A to Z, HSCodeComp c and a).

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


# --- step 2c, Part A: fix the rule on val (lower-bound margin) ------------------------------

RULE_PATH = Path(__file__).resolve().parents[3] / "configs" / "operating_point_rule.json"
RULE_OUT = DATA_ROOT / "eval" / "operating_point" / "rule"
APPLY_OUT = DATA_ROOT / "eval" / "operating_point" / "test_apply"
TARGET = 0.80
MAX_SHARE = 0.80
LOWER_PCT = 5.0
POINT_2A = {("cn8", "b"): "cn8_targets.csv", ("cn8", "a"): "cn8_targets.csv",
            ("hs6", "a"): "hs6_stage1_targets.csv"}  # fmt: skip
HEADLINE = {"cn8": "b", "hs6": "a"}


def resample_indices(cluster: Sequence, n_resamples: int = M.BOOTSTRAP_RESAMPLES,
                     seed: int = M.BOOTSTRAP_SEED) -> list[np.ndarray]:  # fmt: skip
    """The row indices of each resample, drawn exactly as eval.metrics.bootstrap_samples."""
    groups = list(pd.DataFrame({"c": np.asarray(cluster)}).groupby("c", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_resamples):
        drawn = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[g] for g in drawn]))
    return out


def cut_accuracies(policy: str, a: Mapping[str, np.ndarray], cluster: Sequence, kmax: int,
                   n_resamples: int = M.BOOTSTRAP_RESAMPLES,
                   seed: int = M.BOOTSTRAP_SEED) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(point accuracy per cut k = 0..kmax, resample accuracies [resample, k]) of the per-item
    rule of each cut. Each cut's rule is applied to the items and must accept exactly the
    order's items from position k on (checked), so a resample's accepted rows are its rows of
    those items."""
    order = AG.review_order(policy, a)
    n = len(order)
    pos = np.empty(n, int)
    pos[order] = np.arange(n)
    for k in range(kmax + 1):
        accept = ~apply_rule(policy, rule_for_cut(policy, a, k), a)
        if not np.array_equal(accept, pos >= k):
            raise ValueError(f"the per-item rule of cut {k} is not exact (confidence ties)")
    c = np.asarray(a["correct"], float)[order]
    point = np.array([c[k:].mean() if k < n else math.nan for k in range(kmax + 1)])
    acc = np.empty((n_resamples, kmax + 1))
    for b, idx in enumerate(resample_indices(cluster, n_resamples, seed)):
        w = np.bincount(pos[idx], minlength=n).astype(float)  # weight of each order position
        num = np.cumsum((w * c)[::-1])[::-1]
        den = np.cumsum(w[::-1])[::-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            acc[b] = (num / den)[: kmax + 1]
    return point, acc


def select_cut(ok: Sequence[bool]) -> tuple[int | None, int | None]:
    """(first k where the condition holds, smallest k from which it holds at every larger k)."""
    ok = np.asarray(ok, bool)
    first = int(np.argmax(ok)) if ok.any() else None
    head = next((k for k in range(len(ok)) if ok[k:].all()), None)
    return first, head


def lower_bound_cut(policy: str, a: Mapping[str, np.ndarray], cluster: Sequence,
                    target: float = TARGET, max_share: float = MAX_SHARE, pct: float = LOWER_PCT,
                    n_resamples: int = M.BOOTSTRAP_RESAMPLES,
                    seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:  # fmt: skip
    """The headline cut: the smallest review share whose resample accuracy's pct-th percentile is
    at least target and stays so at every larger candidate share up to max_share."""
    n = len(a["confidence"])
    kmax = math.floor(max_share * n + 1e-9)
    point, acc = cut_accuracies(policy, a, cluster, kmax, n_resamples, seed)
    lo = np.nanpercentile(acc, pct, axis=0)
    ok = lo >= target - 1e-12
    first, head = select_cut(ok)
    curve = pd.DataFrame({"k": np.arange(kmax + 1), "review_share": np.arange(kmax + 1) / n,
                          "accuracy": point, f"p{pct:g}": lo, "ok": ok})  # fmt: skip

    def describe(k):
        if k is None:
            return None
        return {"k_reviewed": k, "review_share": k / n, "n_accepted": n - k,
                "accuracy": float(point[k]), "ci_low": float(np.nanpercentile(acc[:, k], 2.5)),
                "ci_high": float(np.nanpercentile(acc[:, k], 97.5)), f"p{pct:g}": float(lo[k]),
                **rule_for_cut(policy, a, k)}  # fmt: skip

    return {
        "policy": policy,
        "headline": describe(head),
        "first": describe(first),
        "first_differs": head != first,
        "curve": curve,
    }


def _json_num(x):
    return "inf" if isinstance(x, float) and math.isinf(x) else x


def _num(x):
    return math.inf if x == "inf" else float(x)


def fix_rule(run_id: str = VAL_RUN, out: Path = RULE_OUT, rule_path: Path = RULE_PATH,
             point_dir: Path = OUT, log_mlflow: bool = True) -> dict[str, Any]:  # fmt: skip
    """Part A: the lower-bound cuts on val, written to configs/operating_point_rule.json."""
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
    valid8, valid10 = R.valid_codes()
    items = [EvalItem(**{**i.model_dump(), "label_code": i.label_code[:6], "label_depth": 6})
             for i in load("rulings", "val").items]  # fmt: skip
    t1 = float(A4.load_config()["a2"]["temperature"]["stage1"])
    f6 = MH.score(items, load_predictions(path), valid8, valid10, t1)
    levels = {"cn8": (arrays(f["confidence"], f["correct_8"], f["agree_cn8"], f["item_id"]), f["cluster_id"]),
              "hs6": (arrays(f6["confidence_hs6"], f6["correct_6"], f6["agree_hs6"], f6["item_id"]),
                      f6["cluster_id"])}  # fmt: skip
    # 1. the step 2a point thresholds for 80%, at full precision from the 2a CSVs
    point, repro = {}, {}
    for (lvl, pol), csv in POINT_2A.items():
        saved = pd.read_csv(point_dir / csv, float_precision="round_trip")
        row = saved[(saved["policy"] == pol) & (saved["target"] == TARGET)].iloc[0]
        now = target_rule(pol, levels[lvl][0], TARGET)
        keys = ("t",) if pol == "a" else ("t_d", "t_a")
        same = (
            all(float(row[k]) == float(now[k]) for k in keys)
            and float(row["review_share"]) == now["review_share"]
        )
        repro[f"{lvl}_{pol}"] = {"saved": {k: float(row[k]) for k in (*keys, "review_share")},
                                 "now": {k: now[k] for k in (*keys, "review_share")}, "same": same}  # fmt: skip
        if not same:
            raise RuntimeError(
                f"step 2a point threshold {lvl} ({pol}) does not reproduce: {repro[f'{lvl}_{pol}']}"
            )
        point[(lvl, pol)] = {k: _json_num(now[k]) for k in keys} | {"val_review_share": now["review_share"],
                                                                    "val_accuracy": now["accuracy"]}  # fmt: skip
    # 2. the lower-bound cuts
    cuts, files = {}, []
    out.mkdir(parents=True, exist_ok=True)
    for lvl, (a, cluster) in levels.items():
        for pol in ("b", "a"):
            res = lower_bound_cut(pol, a, cluster)
            cuts[(lvl, pol)] = res
            p = out / f"{lvl}_{pol}_lower_bound_curve.csv"
            res["curve"].to_csv(p, index=False)
            files.append(p)
    rule = {
        "version": 1,
        "status": "fixed 9 Nov 2026 on val (step 2c Part A), before any test prediction was opened",
        "target_accuracy_auto_accepted": TARGET,
        "framing": "per-item threshold is the headline; the fixed review share (the headline cut's val share) is reported alongside",
        "margin": {
            "rule": "lower bound",
            "percentile": LOWER_PCT,
            "max_review_share": MAX_SHARE,
            "definition": (
                "for each candidate cut k (review the first k items of the order, k/n <= 0.80), apply the "
                "cut's per-item rule to each of the 1,000 cluster-bootstrap resamples (seed 20261021, as in "
                "step 2a) and take the accuracy of the auto-accepted rows; the headline cut is the smallest "
                "k whose 5th percentile (numpy linear) is >= 0.80 and stays >= 0.80 at every larger k up "
                "to 80% review"
            ),
        },
        "rule_forms": {
            "a": "auto-accept when confidence >= t",
            "b": (
                "review when (cn8/HS6 differs from B1-ex's and confidence < t_d) or (agrees and confidence "
                "< t_a); t_a = 0 when the cut falls inside the disagreements, t_d = inf when it falls past "
                "them; 'inf' in this file means infinity"
            ),
        },
        "input": {
            "run_id": run_id,
            "path": str(path),
            "sha256": sha256_file(path),
            "split": "val",
            "config_hash": r.data.params.get("config_hash"),
        },
        "bootstrap": {
            "resamples": M.BOOTSTRAP_RESAMPLES,
            "seed": M.BOOTSTRAP_SEED,
            "unit": "cluster_id",
        },
        "levels": {},
        "sets": {
            "rulings": "cn8",
            "atoz": "cn8",
            "hscodecomp_c": "hs6 (headline input)",
            "hscodecomp_a": "hs6 (alongside)",
        },
    }
    conf_text = {
        "cn8": "A4 joint confidence (the prediction's confidence at the refitted temperatures)",
        "hs6": f"A4 stage-1 probability of the top-1 HS6 at the frozen T1 = {t1!r} (eval.metrics_hs6)",
    }
    for lvl in levels:
        orders = {}
        for pol in ("b", "a"):
            res = cuts[(lvl, pol)]
            h, fst = res["headline"], res["first"]
            keys = ("t",) if pol == "a" else ("t_d", "t_a")
            if h is None:
                raise RuntimeError(
                    f"no lower-bound cut for {lvl} ({pol}) up to {MAX_SHARE:.0%} review"
                )
            orders[pol] = {
                "lower_bound": {**{k: _json_num(h[k]) for k in keys},
                                **{k: h[k] for k in ("k_reviewed", "review_share", "n_accepted", "accuracy",
                                                     "ci_low", "ci_high", "p5")}},
                "first_share_condition_holds": None if not res["first_differs"] else
                {**{k: _json_num(fst[k]) for k in keys}, "review_share": fst["review_share"], "p5": fst["p5"]},
                "fixed_share": h["review_share"],
                "point_2a": point.get((lvl, pol)),
            }  # fmt: skip
        rule["levels"][lvl] = {"confidence": conf_text[lvl], "headline_order": HEADLINE[lvl], "orders": orders,
                               **({"t1": t1} if lvl == "hs6" else {})}  # fmt: skip
    rule_path.write_text(json.dumps(rule, indent=2) + "\n", encoding="utf-8")
    summary = {"rule_file": str(rule_path), "rule_sha256": sha256_file(rule_path), "reproduction_2a": repro,
               "rule": rule, "test_runs_jsonl_sha256": sha256_file(R.TEST_LOG)}  # fmt: skip
    sp = out / "summary.json"
    sp.write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    files.append(sp)
    if log_mlflow:
        exp_id = R._experiment_id(EXPERIMENT, R.MLFLOW_DIR / "artifacts")
        with mlflow.start_run(experiment_id=exp_id, run_name="a4_val_fix_rule") as mr:
            mlflow.set_tags({"analysis": "true", "test_run": "false", "split": "val", "step": "2c-A",
                             "git_commit": R.git_info()["commit"]})  # fmt: skip
            mlflow.log_params({"input_run": run_id, "rule_sha256": summary["rule_sha256"][:16]})
            for f_ in [*files, rule_path]:
                mlflow.log_artifact(str(f_))
            summary["mlflow_run_id"] = mr.info.run_id
        sp.write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    return summary


# --- step 2c, Part B: apply the fixed rule to saved test predictions (report only) ----------


class RuleFileError(RuntimeError):
    """The rule file is not the one committed in Part A."""


def check_rule_file(path: Path, commit: str, repo: Path) -> str:
    """The rule file's SHA-256, refusing unless it is byte-identical to the file at commit and
    commit is an ancestor of HEAD."""
    import hashlib
    import subprocess

    rel = Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    try:
        blob = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"], check=True,
                              capture_output=True).stdout  # fmt: skip
        subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"], check=True,
                       capture_output=True)  # fmt: skip
    except subprocess.CalledProcessError as exc:
        raise RuleFileError(
            f"{rel} is not in commit {commit}, or {commit} is not an ancestor of HEAD"
        ) from exc
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if got != hashlib.sha256(blob).hexdigest():
        raise RuleFileError(f"{rel} (sha256 {got}) differs from the file committed in {commit}")
    return got


def rules_for(rule: Mapping[str, Any], level: str, policy: str) -> dict[str, dict[str, Any]]:
    """{"lower_bound": {thresholds, val_share}, "point_2a": ..., "fixed_share": {...}} from the file."""
    o = rule["levels"][level]["orders"][policy]
    keys = ("t",) if policy == "a" else ("t_d", "t_a")
    out = {"lower_bound": {"thresholds": {k: _num(o["lower_bound"][k]) for k in keys},
                           "val_review_share": o["lower_bound"]["review_share"]}}  # fmt: skip
    if o.get("point_2a"):
        out["point_2a"] = {"thresholds": {k: _num(o["point_2a"][k]) for k in keys},
                           "val_review_share": o["point_2a"]["val_review_share"]}  # fmt: skip
    out["fixed_share"] = {"share": o["fixed_share"], "val_review_share": o["fixed_share"]}
    return out


def apply_one(policy: str, kind: str, spec: Mapping[str, Any], a: Mapping[str, np.ndarray], cluster: Sequence,
              target: float = TARGET, n_resamples: int = M.BOOTSTRAP_RESAMPLES,
              seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:  # fmt: skip
    """One rule on one set: review share triggered, accepted items, accuracy with interval, drift."""
    n = len(a["confidence"])
    correct = np.asarray(a["correct"], float)
    if kind == "fixed_share":
        k = AG.n_reviewed(spec["share"], n)
        order = AG.review_order(policy, a)
        rev = np.zeros(n, bool)
        rev[order[:k]] = True

        def stat(x, s=spec["share"]):
            return AG.accuracy_at(AG.review_order(policy, x), x["correct"], s)
    else:
        rev = apply_rule(policy, spec["thresholds"], a)

        def stat(x):
            m = ~x["review"]
            return float(x["correct"][m].mean()) if m.any() else math.nan

    boot = _boot({**a, "review": rev}, cluster)
    s = M.bootstrap_samples(boot, {"acc": stat}, n_resamples, seed)["acc"]
    acc = float(correct[~rev].mean()) if (~rev).any() else math.nan
    low, high = float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))
    share = float(rev.mean())
    return {"policy": policy, "rule": kind, **{k: _json_num(v) for k, v in spec.get("thresholds", {}).items()},
            **({"share": spec["share"]} if kind == "fixed_share" else {}),
            "review_share": share, "n_reviewed": int(rev.sum()), "n_accepted": int((~rev).sum()),
            "accuracy": acc, "ci_low": low, "ci_high": high,
            "target_point_holds": bool(acc >= target - 1e-12), "target_lower_end_holds": bool(low >= target - 1e-12),
            "val_review_share": spec["val_review_share"], "drift_review_share": share - spec["val_review_share"]}  # fmt: skip


def hindsight(policy: str, a: Mapping[str, np.ndarray], target: float = TARGET) -> float | None:
    """Not a rule: the smallest review share that would have reached target on this set."""
    return AG.rate_for(AG.review_order(policy, a), a["correct"], target)


def apply_rules(rule: Mapping[str, Any], level: str, a, cluster) -> pd.DataFrame:
    head = rule["levels"][level]["headline_order"]
    rows = []
    for pol in (head, *[p for p in ("b", "a") if p != head]):
        for kind, spec in rules_for(rule, level, pol).items():
            rows.append({"ordering": "headline" if pol == head else "alongside", **apply_one(pol, kind, spec, a, cluster),
                         "hindsight_share_for_target_not_a_rule": hindsight(pol, a)})  # fmt: skip
    return pd.DataFrame(rows)


EXPECTED_TOP1 = {"rulings": ("correct_8", 0.653), "atoz": ("correct_8", 0.643),
                 "hscodecomp_c": ("hs6_correct_6", 0.527), "hscodecomp_a": ("hs6_correct_6", None)}  # fmt: skip


def apply_to_test(rule_commit: str, rule_path: Path = RULE_PATH, out: Path = APPLY_OUT,
                  log_mlflow: bool = True) -> dict[str, Any]:  # fmt: skip
    """Part B: the committed rule file applied once to A4's saved test predictions."""
    import mlflow

    from hs_rag_classifier.eval import runner as R
    from hs_rag_classifier.eval.items import REPO_ROOT

    rule_sha = check_rule_file(rule_path, rule_commit, REPO_ROOT)
    rule = json.loads(Path(rule_path).read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(R.tracking_uri())
    lines = [json.loads(x) for x in R.TEST_LOG.read_text(encoding="utf-8").splitlines()]
    runs = {}
    for n, x in enumerate(lines, 1):
        if x["arm"] == "a4" and x["dataset"] in EXPECTED_TOP1:
            if x["dataset"] in runs:
                raise RuntimeError(f"more than one A4 {x['dataset']} test run")
            runs[x["dataset"]] = (n, x["mlflow_run_id"])
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"rule_file": str(rule_path), "rule_sha256": rule_sha, "rule_commit": rule_commit,
                               "test_runs_jsonl_sha256": sha256_file(R.TEST_LOG), "sets": {}}  # fmt: skip
    files = []
    for ds, (col, expected) in EXPECTED_TOP1.items():
        n, rid = runs[ds]
        path = (
            Path(mlflow.get_run(rid).info.artifact_uri.removeprefix("file://"))
            / "predictions.parquet"
        )
        f = pd.read_parquet(path)
        top1 = float(f[col].mean())
        if expected is not None and round(top1, 3) != expected:
            raise RuntimeError(f"{ds}: top-1 {top1} does not reproduce {expected}")
        level = rule["sets"][ds].split()[0]
        if level == "cn8":
            a = arrays(f["confidence"], f["correct_8"], f["agree_cn8"], f["item_id"])
        else:
            a = arrays(
                f["hs6_confidence_hs6"], f["hs6_correct_6"], f["hs6_agree_hs6"], f["item_id"]
            )
        table = apply_rules(rule, level, a, f["cluster_id"])
        p = out / f"{ds}.csv"
        table.to_csv(p, index=False)
        files.append(p)
        summary["sets"][ds] = {"test_run": n, "mlflow_run_id": rid, "path": str(path), "sha256": sha256_file(path),
                               "level": level, "n_items": len(f), "n_clusters": int(f["cluster_id"].nunique()),
                               "top1": top1, "rows": table.to_dict(orient="records")}  # fmt: skip
    sp = out / "summary.json"
    sp.write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    files.append(sp)
    if log_mlflow:
        exp_id = R._experiment_id(EXPERIMENT, R.MLFLOW_DIR / "artifacts")
        with mlflow.start_run(experiment_id=exp_id, run_name="a4_test_apply_rule") as mr:
            mlflow.set_tags({"analysis": "true", "test_run": "false", "step": "2c-B", "report_only": "true",
                             "git_commit": R.git_info()["commit"]})  # fmt: skip
            mlflow.log_params({"rule_sha256": rule_sha[:16], "rule_commit": rule_commit})
            for f_ in files:
                mlflow.log_artifact(str(f_))
            summary["mlflow_run_id"] = mr.info.run_id
        sp.write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    return summary


def main_fix_rule(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="operating_point fix-rule")
    p.add_argument("--run", default=VAL_RUN)
    p.add_argument("--out", type=Path, default=RULE_OUT)
    p.add_argument("--rule", type=Path, default=RULE_PATH)
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    try:
        s = fix_rule(args.run, args.out, args.rule, log_mlflow=not args.no_mlflow)
    except NotValError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"rule_sha256": s["rule_sha256"], "mlflow_run_id": s.get("mlflow_run_id")}, indent=1
        )
    )
    return 0


def main_apply(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="operating_point apply")
    p.add_argument("--rule-commit", required=True)
    p.add_argument("--rule", type=Path, default=RULE_PATH)
    p.add_argument("--out", type=Path, default=APPLY_OUT)
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)
    try:
        s = apply_to_test(args.rule_commit, args.rule, args.out, not args.no_mlflow)
    except RuleFileError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"rule_sha256": s["rule_sha256"], "mlflow_run_id": s.get("mlflow_run_id")}, indent=1
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "fix-rule":
        return main_fix_rule(argv[1:])
    if argv and argv[0] == "apply":
        return main_apply(argv[1:])
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
