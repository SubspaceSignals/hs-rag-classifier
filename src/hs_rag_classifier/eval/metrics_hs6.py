"""Metrics at HS6 for sets whose labels are 6 digits (HSCodeComp): pure functions, no model.

Rules (fixed 9 Nov 2026):
    top-1         the arm's top-1 code, scored at 2, 4 and 6 digits exactly as eval.metrics
                  (an invalid code is wrong at every depth); never at 8 or 10.
    confidence    A3 and A4 (STAGE1_ARMS): the stage-1 probability of the top-1 code's HS6,
                  from the saved order-averaged stage-1 log-probabilities renormalised at the
                  arm's frozen stage-1 temperature (pipeline.scoring.renormalise); 0 when that
                  HS6 is not among the stage-1 options. Every other arm: its own confidence.
    top-k at HS6  A3/A4: the top-1 code's HS6, then the other stage-1 options by stage-1
                  probability (earlier option first on ties). Other arms: the first k distinct
                  HS6s of the ranked candidates. Recall at k counts a valid HS6 equal to the
                  label among the first k (k = 3).
    ECE           eval.metrics.ece (15 equal-mass bins) of the HS6 confidence against
                  correctness at 6 digits.
    review        three orders, first reviewed first, each with accuracy of the auto-accepted
                  items at 10-50% review, AUROC, AURC and the review rate needed for 80%
                  (analysis.a3_agreement functions):
                    a       lowest confidence first; ties by item id (numeric). The headline.
                    b       items whose HS6 differs from B1-ex's top-1 HS6 first, then lowest
                            confidence, ties by id. Only when every item has a B1-ex HS6 (A3/A4
                            details["a3"], or a map passed in); otherwise not reported.
                    length  shortest input sent first (characters after pipeline input_check
                            with the arm's max_input_chars, 1,000), then lowest confidence,
                            ties by id.
    bootstrap     eval.metrics.bootstrap_samples (clusters, 1,000 resamples, harness seed).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.analysis import a3_agreement as AG
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.arms import Prediction
from hs_rag_classifier.eval.items import EvalItem

DEPTHS = (2, 4, 6)
TOP_K = 3
STAGE1_ARMS = ("a3", "a4")
ORDERS = ("a", "b", "length")
MAX_INPUT_CHARS = 1000  # A1/A2's max_input_chars (input_check)
REVIEW_RATES = AG.REVIEW_RATES
TARGET = 0.80


def stage1_temperature(arm) -> float | None:
    """The frozen stage-1 temperature of an A3/A4 arm; None for every other arm."""
    if getattr(arm, "name", None) not in STAGE1_ARMS:
        return None
    return float(arm.config["a2"]["temperature"]["stage1"])


def stage1_probs(details: Mapping[str, Any], t1: float) -> list[tuple[str, float]]:
    """(HS6, probability) of the saved stage-1 options at t1, best first (ties: option order)."""
    from hs_rag_classifier.pipeline import scoring

    s1 = details["stage1"]
    opts, lps = list(s1["options"]), list(s1["logprobs"])
    if not opts or len(opts) != len(lps):
        raise ValueError("stage-1 options and log-probabilities are missing or misaligned")
    p = scoring.renormalise(lps, t1)
    order = scoring.top_k_indices(lps, len(lps))
    return [(str(opts[i])[:6], float(p[i])) for i in order]


def details_of(p: Prediction) -> dict[str, Any]:
    d = p.details
    return json.loads(d) if isinstance(d, str) else (d or {})


def input_chars(text: str, max_chars: int = MAX_INPUT_CHARS) -> int:
    from hs_rag_classifier.pipeline.checks import input_check

    return len(input_check(text, max_chars)["description"])


def score(items: Sequence[EvalItem], predictions: Sequence[Prediction], valid8: set[str],
          valid10: set[str], t1: float | None = None,
          b1ex_hs6: Mapping[str, str] | None = None) -> pd.DataFrame:  # fmt: skip
    """One row per item at HS6 (rules in the module docstring); t1 set means A3/A4 mode."""
    by_id = {p.item_id: p for p in predictions}
    if set(by_id) != {i.item_id for i in items} or len(by_id) != len(predictions):
        raise ValueError("predictions must cover every item exactly once")
    valid6 = {c[:6] for c in valid8}
    rows = []
    for item in items:
        p = by_id[item.item_id]
        label = item.label_code[:6]
        codes = [c.code for c in p.candidates]
        top1 = codes[0] if codes else None
        valid = M.is_valid(top1, valid8, valid10, valid6)
        top_hs6 = top1[:6] if top1 and len(top1) >= 6 else None
        det = details_of(p)
        if t1 is not None:
            probs = stage1_probs(det, t1)
            conf = next((q for h, q in probs if h == top_hs6), 0.0)
            ranked = [top_hs6] if top_hs6 else []
            ranked += [h for h, _ in probs if h not in ranked]
        else:
            conf = float(p.confidence)
            ranked = []
            for c in codes:
                if len(c) >= 6 and c[:6] not in ranked:
                    ranked.append(c[:6])
        rank_valid = [valid] + [h in valid6 for h in ranked[1:]] if ranked else []
        b1 = (b1ex_hs6 or {}).get(item.item_id)
        if b1 is None and isinstance(det.get("a3"), dict) and det["a3"].get("b1ex_top1"):
            b1 = str(det["a3"]["b1ex_top1"])[:6]
        row = {"item_id": item.item_id, "cluster_id": item.cluster_id, "chapter": item.chapter,
               "label_hs6": label, "top1": top1, "top1_valid": valid, "top1_hs6": top_hs6,
               "ranked_hs6": ranked[:10], "confidence_hs6": float(conf), "b1ex_hs6": b1,
               "agree_hs6": b1 is not None and top_hs6 == b1,
               "input_chars": input_chars(item.text)}  # fmt: skip
        for d in DEPTHS:
            row[f"correct_{d}"] = M.correct_at(top1, valid, label, 6, d)
        row[f"recall_{TOP_K}"] = float(any(ok and h == label for h, ok in
                                           zip(ranked[:TOP_K], rank_valid[:TOP_K], strict=False)))  # fmt: skip
        rows.append(row)
    f = pd.DataFrame(rows)
    f["id_rank"] = AG.id_rank(f["item_id"])
    return f


def arrays(f: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"id_rank": f["id_rank"].to_numpy(), "confidence": f["confidence_hs6"].to_numpy(float),
            "agree_cn8": f["agree_hs6"].to_numpy(bool), "correct": f["correct_6"].to_numpy(float),
            "chars": f["input_chars"].to_numpy(int), "oof": np.zeros(len(f))}  # fmt: skip


def review_order(order: str, a: Mapping[str, np.ndarray]) -> np.ndarray:
    """Row indices, first reviewed first (rules in the module docstring)."""
    if order in ("a", "b"):
        return AG.review_order(order, a)  # b: agree_cn8 holds the HS6 agreement here
    if order == "length":
        return np.lexsort((a["id_rank"], a["confidence"], a["chars"]))
    raise ValueError(f"unknown order {order!r}")


def _policy(order: str, a) -> dict[str, Any]:
    m = AG.policy_metrics(review_order(order, a), a["correct"])
    return {k: v for k, v in m.items() if k != "curve"}


def available_orders(f: pd.DataFrame) -> tuple[str, ...]:
    return ORDERS if f["b1ex_hs6"].notna().all() else tuple(o for o in ORDERS if o != "b")


def _stats(orders: Sequence[str]) -> dict[str, M.Statistic]:
    stats: dict[str, M.Statistic] = {
        f"top1_{d}": (lambda x, d=d: float(np.nanmean(x[f"correct_{d}"]))) for d in DEPTHS
    }
    stats[f"recall_{TOP_K}"] = lambda x: float(np.mean(x[f"recall_{TOP_K}"]))
    stats["ece"] = lambda x: M.ece(x["confidence"], x["correct"])
    for o in orders:
        for r in REVIEW_RATES:
            stats[f"{o}_acc_review_{round(r * 100):02d}"] = lambda x, o=o, r=r: AG.accuracy_at(
                review_order(o, x), x["correct"], r
            )
        stats[f"{o}_auroc"] = lambda x, o=o: AG.auroc(review_order(o, x), x["correct"])
        stats[f"{o}_aurc"] = lambda x, o=o: AG.aurc(review_order(o, x), x["correct"])

        def need(x, o=o):
            v = AG.rate_for(review_order(o, x), x["correct"], TARGET)
            return math.nan if v is None else v

        stats[f"{o}_review_for_80"] = need
    return stats


def speed(details: Sequence[Mapping[str, Any]], latency: np.ndarray) -> dict[str, Any]:
    lat = np.asarray(latency, float)
    rat = np.array([float((d.get("timings") or {}).get("rationale", 0.0)) for d in details])
    out = {}
    for name, x in (("with_rationale", lat), ("score_only", lat - rat)):
        x = x[~np.isnan(x)]
        out[name] = {"items_per_minute": 60 / float(x.mean()) if len(x) else math.nan,
                     "p50_s": float(np.median(x)) if len(x) else math.nan,
                     "p95_s": float(np.percentile(x, 95)) if len(x) else math.nan}  # fmt: skip
    return out


def compute(f: pd.DataFrame, n_resamples: int = M.BOOTSTRAP_RESAMPLES,
            seed: int = M.BOOTSTRAP_SEED, mode: str = "native") -> dict[str, Any]:  # fmt: skip
    """Every HS6 metric for one scored HS6 frame, JSON-friendly, with bootstrap intervals."""
    a = arrays(f)
    orders = available_orders(f)
    boot = pd.DataFrame({"cluster_id": f["cluster_id"].to_numpy(), **a,
                         **{f"correct_{d}": f[f"correct_{d}"].to_numpy(float) for d in DEPTHS},
                         f"recall_{TOP_K}": f[f"recall_{TOP_K}"].to_numpy(float)})  # fmt: skip
    stats = _stats(orders)
    full = {c: boot[c].to_numpy() for c in boot.columns}
    samples = M.bootstrap_samples(boot, stats, n_resamples, seed)
    point = {k: fn(full) for k, fn in stats.items()}
    intervals = {k: {"low": float(np.nanpercentile(v, 2.5)), "high": float(np.nanpercentile(v, 97.5))}
                 for k, v in samples.items()}  # fmt: skip
    return {
        "depth": 6,
        "confidence_mode": mode,
        "n_items": len(f),
        "n_clusters": int(f["cluster_id"].nunique()),
        "top1": {d: float(np.nanmean(f[f"correct_{d}"])) for d in DEPTHS},
        f"recall_{TOP_K}": float(f[f"recall_{TOP_K}"].mean()),
        "ece": M.ece(a["confidence"], a["correct"]),
        "reliability": M.reliability(a["confidence"], a["correct"]),
        "invalid_rate": float((~f.loc[f["top1"].notna(), "top1_valid"]).mean())
        if f["top1"].notna().any()
        else math.nan,
        "agree_rate_b1ex_hs6": float(f["agree_hs6"].mean()) if "b" in orders else None,
        "headline_order": "a",
        "orders": {o: _policy(o, a) for o in orders},
        "point": point,
        "intervals": intervals,
        "bootstrap": {"resamples": n_resamples, "seed": seed, "unit": "cluster_id"},
    }


def flat(m: Mapping[str, Any]) -> dict[str, float]:
    """MLflow metrics, prefixed hs6_."""
    out: dict[str, float] = {}
    for k, v in m["point"].items():
        if v is not None and not math.isnan(float(v)):
            out[f"hs6_{k}"] = float(v)
            out[f"hs6_{k}_ci_low"] = m["intervals"][k]["low"]
            out[f"hs6_{k}_ci_high"] = m["intervals"][k]["high"]
    return {k: v for k, v in out.items() if not math.isnan(v)}
