"""Run an arm on a frozen set, score it, and log everything to the local MLflow store.

MLflow: backend sqlite at ~/data/hs-rag-classifier/mlflow/mlflow.db, artifacts under
~/data/hs-rag-classifier/mlflow/artifacts/<experiment>, one experiment per dataset
(eval-rulings, eval-atoz, eval-hand). Each run logs params (arm, config, config_hash), tags
(git commit and dirty flag, dataset, split, split SHA-256, nomenclature version, ablation),
flat metrics, and artifacts: predictions.parquet, metrics.json, per_chapter.csv,
reliability.png, coverage.png.

Test guard: a test-split run refuses a dirty git tree (untracked files included), --limit,
and sanity arms. Every test run is appended to ~/data/hs-rag-classifier/eval/test_runs.jsonl
as soon as its MLflow run exists, so a run that fails part-way is still counted.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hs_rag_classifier.data.rulings import NOMENCLATURE_VERSION
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.arms import Arm, OracleArm, RandomArm, config_hash
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalSet
from hs_rag_classifier.eval.plots import coverage_plot, reliability_plot

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

MLFLOW_DIR = DATA_ROOT / "mlflow"
TEST_LOG = DATA_ROOT / "eval" / "test_runs.jsonl"


class GuardError(RuntimeError):
    """A test-split run that the guard refuses."""


def tracking_uri(mlflow_dir: Path = MLFLOW_DIR) -> str:
    return f"sqlite:///{(mlflow_dir / 'mlflow.db').resolve()}"


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def valid_codes(root: Path = DATA_ROOT) -> tuple[set[str], set[str]]:
    """(current cn8s, current declarable code10s) in the pinned nomenclature."""
    tariff = root / "processed" / "tariff" / NOMENCLATURE_VERSION
    codes = pd.read_parquet(
        tariff / "codes.parquet", columns=["code10", "declarable", "superseded"]
    )
    valid10 = set(codes.loc[codes["declarable"] & ~codes["superseded"], "code10"])
    valid8 = set(pd.read_parquet(tariff / "codes8.parquet", columns=["cn8"])["cn8"])
    return valid8, valid10


def _b0(valid8: set[str]) -> Arm:
    from hs_rag_classifier.arms.b0_bm25 import B0BM25Arm

    return B0BM25Arm()


def _b1(valid8: set[str]) -> Arm:
    from hs_rag_classifier.arms.b1_embed import B1EmbedArm

    return B1EmbedArm()


def _b1ex(valid8: set[str]) -> Arm:
    from hs_rag_classifier.arms.b1ex_embed import B1exEmbedArm

    return B1exEmbedArm()


ARMS = {
    "oracle": lambda valid8: OracleArm(),
    "random": lambda valid8: RandomArm(sorted(valid8)),
    "b0": _b0,
    "b1": _b1,
    "b1ex": _b1ex,
}


def make_arm(name: str, valid8: set[str]) -> Arm:
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; choose from {', '.join(sorted(ARMS))}")
    return ARMS[name](valid8)


def check_test_guard(arm: Arm, evalset: EvalSet, git: dict, limit: int | None) -> None:
    if evalset.split != "test":
        return
    if getattr(arm, "sanity", False):
        raise GuardError(f"{arm.name} is a sanity arm; sanity arms never run on a test split")
    if limit is not None:
        raise GuardError("--limit is not allowed on a test split")
    if git["dirty"]:
        raise GuardError("the git tree is dirty; commit or stash before a test-split run")


def predict_all(arm: Arm, items: list) -> tuple[list, list[float], float]:
    """Predictions, per-item latency (each call's time split over its batch), total seconds."""
    size = max(1, int(getattr(arm, "batch_size", 1)))
    preds, latencies, total = [], [], 0.0
    for start in range(0, len(items), size):
        batch = items[start : start + size]
        t0 = time.perf_counter()
        out = arm.predict(batch)
        elapsed = time.perf_counter() - t0
        if len(out) != len(batch):
            raise ValueError(f"{arm.name} returned {len(out)} predictions for {len(batch)} items")
        total += elapsed
        preds.extend(out)
        latencies.extend([elapsed / len(batch)] * len(batch))
    return preds, latencies, total


def flat_metrics(m: dict) -> dict[str, float]:
    out: dict[str, float] = {"n_items": m["n_items"], "n_clusters": m["n_clusters"]}
    for d, v in m["top1"].items():
        out[f"top1_{d}"], out[f"n_{d}"] = v["accuracy"], v["n"]
    for k, v in m["recall"].items():
        out[f"recall_{k}_8digit"] = v
    for key in ("invalid_rate", "no_top1_rate", "abstain_rate", "ece", "headline"):
        out[key] = m[key]
    for c in m["coverage"]:
        out[f"coverage_acc_review_{round(c['review_rate'] * 100):02d}"] = c["accuracy"]
    out.update(m["latency"])
    for name, ci in m["bootstrap"]["intervals"].items():
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = ci["low"], ci["high"]
    return {k: float(v) for k, v in out.items() if v is not None and not math.isnan(float(v))}


def _experiment_id(name: str, artifact_root: Path) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(name, artifact_location=(artifact_root / name).as_uri())


def run_eval(
    arm: Arm,
    evalset: EvalSet,
    valid8: set[str],
    valid10: set[str],
    *,
    limit: int | None = None,
    review_rate: float = M.HEADLINE_REVIEW_RATE,
    mlflow_dir: Path = MLFLOW_DIR,
    test_log: Path = TEST_LOG,
    git: dict | None = None,
    n_resamples: int = M.BOOTSTRAP_RESAMPLES,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Guard, predict, score, log. Returns run_id, metrics and the scored frame."""
    git = git if git is not None else git_info()
    check_test_guard(arm, evalset, git, limit)
    items = evalset.items[:limit] if limit is not None else evalset.items
    if not items:
        raise ValueError(f"{evalset.dataset} {evalset.split} has no items to score")

    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    experiment = f"eval-{evalset.dataset}"
    exp_id = _experiment_id(experiment, mlflow_dir / "artifacts")
    chash = config_hash(arm.config)
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{arm.name}-{evalset.split}") as run:
        run_id = run.info.run_id
        if evalset.split == "test":
            test_log.parent.mkdir(parents=True, exist_ok=True)
            line = {
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
                "arm": arm.name,
                "config_hash": chash,
                "commit": git["commit"],
                "dataset": evalset.dataset,
                "split_sha256": evalset.split_sha256,
                "mlflow_run_id": run_id,
            }
            with test_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        mlflow.log_params(
            {
                "arm": arm.name,
                "config": json.dumps(arm.config, sort_keys=True, default=str),
                "config_hash": chash,
                "limit": limit,
                "headline_review_rate": review_rate,
                "bootstrap_resamples": n_resamples,
            }
        )
        mlflow.set_tags(
            {
                "git_commit": git["commit"],
                "git_dirty": str(git["dirty"]).lower(),
                "dataset": evalset.dataset,
                "split": evalset.split,
                "split_file": evalset.split_file,
                "split_sha256": evalset.split_sha256,
                "nomenclature_version": NOMENCLATURE_VERSION,
                "ablation": evalset.ablation,
                "sanity_arm": str(bool(getattr(arm, "sanity", False))).lower(),
                **(tags or {}),
            }
        )
        preds, latencies, total = predict_all(arm, items)
        frame = M.score(items, preds, valid8, valid10, latencies)
        metrics = M.compute_metrics(frame, total, review_rate, n_resamples)
        mlflow.log_metrics(flat_metrics(metrics))
        label = f"{arm.name}, {evalset.dataset} {evalset.split}"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            frame.to_parquet(out / "predictions.parquet", index=False)
            (out / "metrics.json").write_text(
                json.dumps(metrics, indent=2, default=str), encoding="utf-8"
            )
            M.per_chapter(frame).to_csv(out / "per_chapter.csv", index=False)
            reliability_plot(metrics["reliability"], metrics["ece"], label, out / "reliability.png")
            coverage_plot(metrics["coverage"], review_rate, label, out / "coverage.png")
            mlflow.log_artifacts(str(out))
    return {"run_id": run_id, "experiment": experiment, "metrics": metrics, "frame": frame}


def _fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def summary_lines(result: dict, arm: Arm, evalset: EvalSet) -> list[str]:
    m = result["metrics"]
    ci = m["bootstrap"]["intervals"]
    top1 = "  ".join(f"{d}: {_fmt(v['accuracy'])} (n {v['n']})" for d, v in m["top1"].items())
    lat = m["latency"]
    return [
        (
            f"Run {result['run_id']} ({result['experiment']}): arm {arm.name} "
            f"[{config_hash(arm.config)}], {evalset.dataset} {evalset.split}, ablation "
            f"{evalset.ablation}, {m['n_items']:,} items in {m['n_clusters']:,} clusters"
        ),
        f"  top-1  {top1}",
        (
            f"  top-1 at 8: {_fmt(m['top1'][8]['accuracy'])} "
            f"[{_fmt(ci['top1_8']['low'])}, {_fmt(ci['top1_8']['high'])}]; "
            f"recall at " + ", ".join(f"{k}: {_fmt(v)}" for k, v in m["recall"].items())
        ),
        (
            f"  invalid rate {_fmt(m['invalid_rate'])}; abstain rate {_fmt(m['abstain_rate'])}; "
            f"ECE {_fmt(m['ece'])} [{_fmt(ci['ece']['low'])}, {_fmt(ci['ece']['high'])}]"
        ),
        (
            f"  headline (8-digit accuracy at {m['headline_review_rate']:.0%} review): "
            f"{_fmt(m['headline'])} "
            f"[{_fmt(ci['headline']['low'])}, {_fmt(ci['headline']['high'])}]; "
            f"at 0% review: {_fmt(m['coverage'][0]['accuracy'])}"
        ),
        (
            f"  throughput {lat['items_per_minute']:,.0f} items/min; latency p50 "
            f"{lat['latency_p50_s'] * 1000:.3f} ms, p95 {lat['latency_p95_s'] * 1000:.3f} ms"
        ),
    ]


def read_test_runs(path: Path = TEST_LOG) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def count_test_runs(runs: list[dict]) -> Counter:
    return Counter(r["arm"] for r in runs)
