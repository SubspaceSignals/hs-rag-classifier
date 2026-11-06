"""A3: order-averaged A2 plus a review queue ordered by disagreement with B1-ex, as an arm.

Config: configs/a3_pipeline.json (frozen). It pins configs/a2_pipeline.json and the B1-ex locked
config configs/b1ex_embed.json by SHA-256 ({"path", "sha256"}, as the other configs pin their
parents); load_config refuses either when it has changed. The A2 part is A2's config with
order_averaging on and the A3 stage temperatures, run through A2's own graph (the 024bef5 code
path, pipeline/graph_a2.py); the rationale call is on, as in A2, and never changes the choice.
B1-ex runs with its locked config (k = 1, index from train rulings only).

Per item (predict): A2's prediction (candidates with joint confidences, details) plus
details["a3"]: B1-ex top-1 code, its cn8 and similarity, the cn8 agreement flag and B1-ex time.

Review queue, policy (b), set-level (after_scoring, called by the harness once all items are
scored): every item whose A2 top-1 cn8 differs from B1-ex's first, then the agreements; lowest A2
joint confidence first within each group; ties by item id in numeric order
(analysis.a3_agreement.review_order, unchanged). No fitted parameters. after_scoring adds the
per-item columns (review group and rank, top 3 with confidences, HS6 choice, B1-ex, citation
pass, score-only latency) and the A3 metrics: policy (b) and confidence-only (a) review tables,
agreement with B1-ex, and speed with and without the rationale call, with 95% cluster-bootstrap
intervals (the harness's clusters, resamples and seed). The harness headline (accuracy at 20%
review by confidence) stays under its own name; A3's main-table figure is a3_headline_b.

Allowed on test splits under every existing guard (test-run log, one-slot guard with the
multi-slot override refused on test, tracing refused on test).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.analysis import a3_agreement as AG
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.arms import Prediction
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem

CONFIG_PATH = REPO_ROOT / "configs" / "a3_pipeline.json"
POLICIES = ("b", "a")
TARGET = 0.80


def load_config(path: Path = CONFIG_PATH, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """A2's config with A3's averaging and temperatures, B1-ex's locked config, and A3's own."""
    from hs_rag_classifier.arms import b1ex_embed as BX

    a3 = json.loads(path.read_text(encoding="utf-8"))
    for key in ("base_config", "b1ex_config"):
        pin = a3[key]
        if sha256(repo / pin["path"]) != pin["sha256"]:
            raise RuntimeError(f"{pin['path']} changed since {path.name} was written")
    base = A2.load_config(repo / a3["base_config"]["path"], repo)
    a2 = {
        **base["a2"],
        "order_averaging": bool(a3["order_averaging"]),
        "temperature": a3["temperature"],
    }
    b1ex = BX.load_config(repo / a3["b1ex_config"]["path"])
    if int(b1ex["k"]) != 1:
        raise RuntimeError(f"B1-ex locked config has k = {b1ex['k']}, expected 1")
    return {**base, "a2": a2, "b1ex": b1ex, "a3": a3}


class A3Arm:
    name = "a3"
    max_depth = 8
    batch_size = 1

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 pipeline=None, b1ex=None):  # fmt: skip
        from hs_rag_classifier.arms import b1ex_embed as BX

        self.config = dict(config or load_config())
        self.a2 = A2.A2PipelineArm(self.config, root, pipeline)  # runs the one-slot guard
        self.b1ex = b1ex or BX.B1exEmbedArm(self.config["b1ex"], root)
        self.run_tags = dict(self.a2.run_tags)

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        out = []
        for item in items:
            p = self.a2.predict([item])[0]
            t0 = time.perf_counter()
            b = self.b1ex.predict([item])[0]
            b_s = time.perf_counter() - t0
            code = b.candidates[0].code if b.candidates else None
            b_cn8 = code[:8] if code else None
            a2_cn8 = p.candidates[0].code[:8] if p.candidates else None
            a3 = {"b1ex_top1": code, "b1ex_cn8": b_cn8,
                  "b1ex_sim": float(b.candidates[0].score) if b.candidates else None,
                  "agree_cn8": a2_cn8 is not None and a2_cn8 == b_cn8, "b1ex_s": b_s}  # fmt: skip
            out.append(p.model_copy(update={"details": {**(p.details or {}), "a3": a3}}))
        return out

    def after_scoring(self, frame: pd.DataFrame, n_resamples: int = M.BOOTSTRAP_RESAMPLES,
                      seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:  # fmt: skip
        return after_scoring(frame, n_resamples, seed)


# --- set-level outputs -------------------------------------------------------------------


def review_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The per-item A3 columns, including policy (b)'s review group and rank (1 = first)."""
    det = [json.loads(d) for d in frame["details"]]
    f = frame.copy()
    f["hs6_choice"] = [d["stage1"]["choice_code"] for d in det]
    f["top3_cn8"] = [[c[:8] for c in cs[:3]] for cs in f["candidates"]]
    f["top3_confidence"] = [list(s[:3]) for s in f["scores"]]
    f["b1ex_top1"] = [d["a3"]["b1ex_top1"] for d in det]
    f["b1ex_cn8"] = [d["a3"]["b1ex_cn8"] for d in det]
    f["b1ex_sim"] = [d["a3"]["b1ex_sim"] for d in det]
    f["agree_cn8"] = [bool(d["a3"]["agree_cn8"]) for d in det]
    f["citation_pass"] = [bool(((d.get("rationale") or {}).get("citation") or {}).get("pass"))
                          for d in det]  # fmt: skip
    rat = np.array([(d.get("timings") or {}).get("rationale", 0.0) for d in det], float)
    f["rationale_s"] = rat
    f["latency_score_only_s"] = f["latency_s"].to_numpy(float) - rat
    f["review_group"] = np.where(f["agree_cn8"], "agree", "disagree")
    order = AG.review_order("b", _arrays(f))
    rank = np.empty(len(f), int)
    rank[order] = np.arange(1, len(f) + 1)
    f["review_rank"] = rank
    return f


def _arrays(f: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "id_rank": AG.id_rank(f["item_id"]),
        "confidence": f["confidence"].to_numpy(float),
        "agree_cn8": f["agree_cn8"].to_numpy(bool),
        "correct": f["correct_8"].to_numpy(float),
        "oof": np.zeros(len(f)),
    }


def _policy_stats() -> dict[str, Any]:
    def acc(p, r):
        return lambda a: AG.accuracy_at(AG.review_order(p, a), a["correct"], r)

    def rate(p):
        def fn(a):
            v = AG.rate_for(AG.review_order(p, a), a["correct"], TARGET)
            return math.nan if v is None else v

        return fn

    stats: dict[str, Any] = {}
    for p in POLICIES:
        for r in AG.REVIEW_RATES:
            stats[f"{p}_acc_review_{round(r * 100):02d}"] = acc(p, r)
        stats[f"{p}_auroc"] = lambda a, p=p: AG.auroc(AG.review_order(p, a), a["correct"])
        stats[f"{p}_aurc"] = lambda a, p=p: AG.aurc(AG.review_order(p, a), a["correct"])
        stats[f"{p}_review_for_80"] = rate(p)
    stats["agree_rate"] = lambda a: float(np.mean(a["agree_cn8"]))
    return stats


def _speed(lat: np.ndarray) -> dict[str, float]:
    lat = lat[~np.isnan(lat)]
    if not len(lat):
        return {"items_per_minute": math.nan, "p50_s": math.nan, "p95_s": math.nan}
    return {"items_per_minute": 60 / float(lat.mean()), "p50_s": float(np.median(lat)),
            "p95_s": float(np.percentile(lat, 95))}  # fmt: skip


def after_scoring(frame: pd.DataFrame, n_resamples: int = M.BOOTSTRAP_RESAMPLES,
                  seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:  # fmt: skip
    """{"frame": with the A3 columns, "metrics": nested, "flat": MLflow metrics}."""
    f = review_columns(frame)
    a = _arrays(f)
    pols = {p: AG.policy_metrics(AG.review_order(p, a), a["correct"]) for p in POLICIES}
    stats = _policy_stats()
    boot = pd.DataFrame({"cluster_id": f["cluster_id"].to_numpy(), **a})
    samples = M.bootstrap_samples(boot, stats, n_resamples, seed)
    point = {k: fn(a) for k, fn in stats.items()}
    intervals = {k: {"low": float(np.nanpercentile(samples[k], 2.5)),
                     "high": float(np.nanpercentile(samples[k], 97.5))} for k in stats}  # fmt: skip
    speed = {"with_rationale": _speed(f["latency_s"].to_numpy(float)),
             "score_only": _speed(f["latency_score_only_s"].to_numpy(float))}  # fmt: skip
    metrics = {
        "review_policy": "b",
        "headline_policy_b": point["b_acc_review_20"],
        "headline_policy_b_ci": intervals["b_acc_review_20"],
        "hs6_top1": float(f["correct_6"].mean()),
        "policies": {p: {k: v for k, v in pols[p].items() if k != "curve"} for p in POLICIES},
        "curves": {p: pols[p]["curve"] for p in POLICIES},
        "agree_rate": point["agree_rate"],
        "point": point,
        "intervals": intervals,
        "citation_pass_rate": float(f["citation_pass"].mean()),
        "speed": speed,
        "bootstrap": {"resamples": n_resamples, "seed": seed, "unit": "cluster_id"},
    }
    flat = {"a3_headline_b": point["b_acc_review_20"]}
    for k, v in point.items():
        flat[f"a3_{k}"] = v
        flat[f"a3_{k}_ci_low"], flat[f"a3_{k}_ci_high"] = intervals[k]["low"], intervals[k]["high"]
    for mode, s in speed.items():
        for k, v in s.items():
            flat[f"a3_{mode}_{k}"] = v
    flat["a3_citation_pass_rate"] = metrics["citation_pass_rate"]
    flat = {k: float(v) for k, v in flat.items() if v is not None and not math.isnan(float(v))}
    return {"frame": f, "metrics": metrics, "flat": flat}
