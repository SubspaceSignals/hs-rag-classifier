"""A3 against A2 on saved predictions, and review-table reference rows; never calls a model.

    python -m hs_rag_classifier.analysis.a3_compare compare --a3 PATH --a2 PATH
        --dataset rulings --split val [--cross-server] [--out DIR]
    python -m hs_rag_classifier.analysis.a3_compare reference --a2 PATH --b1ex PATH
        --dataset rulings --split val [--out DIR]

Inputs are saved prediction files, any of: an A2 pass JSON (eval/repro_val750/passes format:
{"rows": [{"item_id", "candidates", "scores", "confidence", "details"}]}), a harness
predictions.parquet, or a harness partial predictions JSONL (eval/partial/RUN_ID.jsonl). Each is
scored against the dataset split's labels (eval.metrics.score, 8 digits) and must cover exactly
its items. The A3 file's details hold B1-ex's cn8 agreement (arms.a3_pipeline). A parquet without
a details column (B1-ex runs saved before predictions carried one) loads with details None for
every row; reference does not need details, and compare refuses an A3 file without them.

compare: paired cluster bootstrap (eval.metrics.bootstrap_samples: the split's clusters, 1,000
resamples, the harness seed), A3 minus A2, for cn8 top-1, HS6 top-1, top-3 recall at 8; accuracy
of auto-accepted items at 10 to 50% review with A3 ordered by policy (b) and A2 by its own
confidence (policy a); AUROC and AURC (analysis.a3_agreement functions). --cross-server marks
that the two files came from different llama-server sessions or builds; it is written into the
output and the MLflow tags.

reference: the review-table rows for A2 (by its own confidence) and B1-ex (by its top
neighbour's similarity, the first candidate's score), each lowest first, ties by item id.

Each command logs one MLflow run in experiment compare-<dataset>, tagged comparison=true, with
every input path and SHA-256. It is never a test run: it does not go through the test-run
guard and never writes to test_runs.jsonl, whatever split its inputs come from.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.analysis import a3_agreement as AG
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.arms import Candidate, Prediction
from hs_rag_classifier.eval.items import DATA_ROOT, sha256_file

OUT = DATA_ROOT / "eval" / "a3"
REVIEW_RATES = AG.REVIEW_RATES


def load_predictions(path: Path) -> list[Prediction]:
    """Predictions from an A2 pass JSON, a predictions.parquet or a partial JSONL."""
    path = Path(path)
    if path.suffix == ".jsonl":
        from hs_rag_classifier.eval.runner import read_partial

        return read_partial(path)[0]
    if path.suffix == ".parquet":
        f = pd.read_parquet(path)
        if "details" not in f.columns:  # e.g. B1-ex runs saved before predictions had details
            f = f.assign(details=None)
        return [
            Prediction(
                item_id=r.item_id,
                candidates=[
                    Candidate(code=c, score=float(s))
                    for c, s in zip(r.candidates, r.scores, strict=True)
                ],
                confidence=None if pd.isna(r.confidence) else float(r.confidence),
                abstain=bool(r.abstain),
                details=json.loads(r.details) if isinstance(r.details, str) else None,
            )
            for r in f.itertuples(index=False)
        ]
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        return [
            Prediction(
                item_id=r["item_id"],
                candidates=[
                    Candidate(code=c, score=float(s))
                    for c, s in zip(r["candidates"], r["scores"], strict=True)
                ],
                confidence=r["confidence"],
                details=r.get("details"),
            )
            for r in rows
        ]
    raise ValueError(f"{path}: expected .json (A2 pass), .parquet or .jsonl")


def scored(preds: list[Prediction], items, valid8: set[str], valid10: set[str]) -> pd.DataFrame:
    """Scored at 8 digits, rows in the items' order (score refuses missing or extra items)."""
    f = M.score(items, preds, valid8, valid10, max_depth=8)
    f["id_rank"] = AG.id_rank(f["item_id"])
    return f


def a3_agreement(
    preds: list[Prediction], ids: list[str], source: str = "the A3 file"
) -> np.ndarray:
    by_id = {p.item_id: p for p in preds}
    missing = [i for i in ids if not isinstance((by_id[i].details or {}).get("a3"), dict)
               or "agree_cn8" not in by_id[i].details["a3"]]  # fmt: skip
    if missing:
        raise ValueError(
            f"{source} has no A3 details (details['a3']['agree_cn8']) for {len(missing)} of "
            f"{len(ids)} items, e.g. item {missing[0]}; compare needs predictions saved by arm a3"
        )
    return np.array([bool(by_id[i].details["a3"]["agree_cn8"]) for i in ids])


def _arr(conf, correct, id_rank, agree=None) -> dict[str, np.ndarray]:
    return {"id_rank": np.asarray(id_rank), "confidence": np.asarray(conf, float),
            "correct": np.asarray(correct, float),
            "agree_cn8": np.zeros(len(conf), bool) if agree is None else np.asarray(agree, bool),
            "oof": np.zeros(len(conf))}  # fmt: skip


def compare(fa3: pd.DataFrame, fa2: pd.DataFrame, agree3: np.ndarray,
            n_resamples: int = M.BOOTSTRAP_RESAMPLES, seed: int = M.BOOTSTRAP_SEED) -> dict:  # fmt: skip
    """A3 minus A2 (rules in the module docstring); both frames in the same item order."""
    if list(fa3["item_id"]) != list(fa2["item_id"]):
        raise ValueError("the two prediction files must cover the same items")
    paired = {c: M.paired_difference(fa3, fa2, c, n_resamples, seed)
              for c in ("correct_8", "correct_6", "recall_3")}  # fmt: skip
    boot = pd.DataFrame({
        "cluster_id": fa3["cluster_id"].to_numpy(), "id_rank": fa3["id_rank"].to_numpy(),
        "c3": fa3["correct_8"].to_numpy(float), "p3": fa3["confidence"].to_numpy(float),
        "g3": np.asarray(agree3, bool), "c2": fa2["correct_8"].to_numpy(float),
        "p2": fa2["confidence"].to_numpy(float)})  # fmt: skip

    def pair(x):
        a3 = _arr(x["p3"], x["c3"], x["id_rank"], x["g3"])
        a2 = _arr(x["p2"], x["c2"], x["id_rank"])
        return (AG.review_order("b", a3), a3["correct"]), (AG.review_order("a", a2), a2["correct"])

    def acc(r):
        def fn(x):
            (o3, c3), (o2, c2) = pair(x)
            return AG.accuracy_at(o3, c3, r) - AG.accuracy_at(o2, c2, r)

        return fn

    def whole(f):
        def fn(x):
            (o3, c3), (o2, c2) = pair(x)
            return f(o3, c3) - f(o2, c2)

        return fn

    stats = {f"acc_review_{round(r * 100):02d}": acc(r) for r in REVIEW_RATES}
    stats["auroc"], stats["aurc"] = whole(AG.auroc), whole(AG.aurc)
    samples = M.bootstrap_samples(boot, stats, n_resamples, seed)
    full = {c: boot[c].to_numpy() for c in boot.columns}
    (o3, c3), (o2, c2) = pair(full)
    review = {}
    for k, fn in stats.items():
        review[k] = {"diff": fn(full), "low": float(np.nanpercentile(samples[k], 2.5)),
                     "high": float(np.nanpercentile(samples[k], 97.5)),
                     "share_le_0": float((samples[k] <= 0).mean())}  # fmt: skip
    return {
        "n_items": len(fa3),
        "n_clusters": int(fa3["cluster_id"].nunique()),
        "paired": {
            "cn8_top1": paired["correct_8"],
            "hs6_top1": paired["correct_6"],
            "top3_recall_8": paired["recall_3"],
        },
        "review": review,
        "a3_policy_b": {k: v for k, v in AG.policy_metrics(o3, c3).items() if k != "curve"},
        "a2_policy_a": {k: v for k, v in AG.policy_metrics(o2, c2).items() if k != "curve"},
        "a3_agree_rate": float(np.mean(agree3)),
        "resamples": n_resamples,
        "seed": seed,
        "unit": "cluster_id",
    }


def reference(fa2: pd.DataFrame, fb1: pd.DataFrame) -> dict:
    """Review-table rows: A2 by its confidence, B1-ex by its top similarity (lowest first)."""
    sim = [s[0] if len(s) else -math.inf for s in fb1["scores"]]
    out = {}
    for name, f, conf in (("a2_confidence", fa2, fa2["confidence"]), ("b1ex_similarity", fb1, sim)):
        a = _arr(conf, f["correct_8"], f["id_rank"])
        m = AG.policy_metrics(AG.review_order("a", a), a["correct"])
        out[name] = {k: v for k, v in m.items() if k != "curve"}
        out[name]["top1_8"] = float(f["correct_8"].mean())
    return out


def _flat(prefix: str, d: Any, out: dict[str, float]) -> dict[str, float]:
    if isinstance(d, dict):
        for k, v in d.items():
            _flat(f"{prefix}_{k}" if prefix else str(k).replace(".", "_"), v, out)
    elif isinstance(d, int | float) and not isinstance(d, bool) and not math.isnan(float(d)):
        out[prefix.replace(".", "_")] = float(d)
    return out


def log_run(kind: str, result: dict, inputs: dict[str, Path], dataset: str, split: str,
            tags: dict[str, str], mlflow_dir: Path | None = None) -> str:  # fmt: skip
    """One MLflow comparison run (never a test run; no guard, no test_runs.jsonl)."""
    import mlflow

    from hs_rag_classifier.eval import runner as R

    mlflow_dir = mlflow_dir or R.MLFLOW_DIR
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(R.tracking_uri(mlflow_dir))
    exp = R._experiment_id(f"compare-{dataset}", mlflow_dir / "artifacts")
    with mlflow.start_run(experiment_id=exp, run_name=f"a3-{kind}") as run:
        mlflow.set_tags({"comparison": "true", "kind": kind, "dataset": dataset, "split": split,
                         **{f"input_{k}": str(p) for k, p in inputs.items()},
                         **{f"input_{k}_sha256": sha256_file(p) for k, p in inputs.items()},
                         **tags})  # fmt: skip
        metrics = {k: v for k, v in _flat("", result, {}).items() if not k.startswith("inputs")}
        mlflow.log_metrics({k[:250]: v for k, v in metrics.items()})
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f"{kind}.json"
            p.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            mlflow.log_artifact(str(p))
        return run.info.run_id


def run_compare(a3: Path, a2: Path, items, dataset: str, split: str, cross_server: bool,
                out: Path | None = None, mlflow_dir: Path | None = None,
                n_resamples: int = M.BOOTSTRAP_RESAMPLES, valid=None) -> dict:  # fmt: skip
    from hs_rag_classifier.eval import runner as R

    valid8, valid10 = valid or R.valid_codes()
    p3, p2 = load_predictions(a3), load_predictions(a2)
    f3, f2 = scored(p3, items, valid8, valid10), scored(p2, items, valid8, valid10)
    result = compare(f3, f2, a3_agreement(p3, list(f3["item_id"]), str(a3)), n_resamples)
    result.update(cross_server=bool(cross_server), dataset=dataset, split=split,
                  inputs={"a3": {"path": str(a3), "sha256": sha256_file(a3)},
                          "a2": {"path": str(a2), "sha256": sha256_file(a2)}})  # fmt: skip
    result["mlflow_run_id"] = log_run("compare", result, {"a3": a3, "a2": a2}, dataset, split,
                                      {"cross_server": str(bool(cross_server)).lower()},
                                      mlflow_dir)  # fmt: skip
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"compare_{dataset}_{split}.json").write_text(json.dumps(result, indent=2))
    return result


def run_reference(a2: Path, b1ex: Path, items, dataset: str, split: str,
                  out: Path | None = None, mlflow_dir: Path | None = None, valid=None) -> dict:  # fmt: skip
    from hs_rag_classifier.eval import runner as R

    valid8, valid10 = valid or R.valid_codes()
    f2 = scored(load_predictions(a2), items, valid8, valid10)
    fb = scored(load_predictions(b1ex), items, valid8, valid10)
    result = reference(f2, fb)
    result.update(dataset=dataset, split=split,
                  inputs={"a2": {"path": str(a2), "sha256": sha256_file(a2)},
                          "b1ex": {"path": str(b1ex), "sha256": sha256_file(b1ex)}})  # fmt: skip
    result["mlflow_run_id"] = log_run("reference", result, {"a2": a2, "b1ex": b1ex}, dataset,
                                      split, {}, mlflow_dir)  # fmt: skip
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"reference_{dataset}_{split}.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    from hs_rag_classifier.eval.items import DATASETS, load

    ap = argparse.ArgumentParser(prog="a3_compare", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "reference"):
        p = sub.add_parser(name)
        p.add_argument("--a2", type=Path, required=True)
        p.add_argument("--dataset", choices=DATASETS, default="rulings")
        p.add_argument("--split", choices=("val", "test"), default="val")
        p.add_argument("--out", type=Path, default=OUT)
        if name == "compare":
            p.add_argument("--a3", type=Path, required=True)
            p.add_argument("--cross-server", action="store_true")
        else:
            p.add_argument("--b1ex", type=Path, required=True)
    a = ap.parse_args(argv)
    items = load(a.dataset, a.split).items
    if a.cmd == "compare":
        r = run_compare(a.a3, a.a2, items, a.dataset, a.split, a.cross_server, a.out)
        keep = ("paired", "review", "cross_server", "mlflow_run_id")
    else:
        r = run_reference(a.a2, a.b1ex, items, a.dataset, a.split, a.out)
        keep = ("a2_confidence", "b1ex_similarity", "mlflow_run_id")
    print(json.dumps({k: r[k] for k in keep}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
