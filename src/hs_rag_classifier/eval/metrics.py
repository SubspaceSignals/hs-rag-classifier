"""Evaluation metrics: pure functions over a scored frame (one row per item) or plain arrays.

Rules:
    valid code    an 8-digit code is valid when it is a current cn8; a 10-digit code when it is
                  a current declarable line (nomenclature v4.0.1633). Any other code is invalid,
                  and an invalid code always counts as wrong, at every depth.
    top-1         correct at d digits (2, 4, 6, 8, 10) when the top-1 code is valid, has at
                  least d digits, and its first d match the label's. Rows with label_depth 8
                  are scored up to 8 only (NaN at 10); n is reported for each depth.
    recall at k   at 8 digits: any of the first k candidates is a valid code whose first 8
                  digits match the label's. k = 3 and 10 always; k = 20 and 50 are diagnostics,
                  reported only when the arm returns that many candidates for some item.
    invalid rate  share of items with a top-1 code whose top-1 is invalid.
    ECE           15 equal-mass bins of top-1 confidence (sorted, split as evenly as possible);
                  sum over bins of |bin| / n * |accuracy at 8 digits - mean confidence|.
    coverage      items the arm abstains on go to review first, then the least confident. For
                  each target review rate r, round(r * n) items (at least every abstain) are
                  reviewed; the rest are auto-accepted and scored at 8 digits.
    headline      8-digit accuracy of the auto-accepted items at HEADLINE_REVIEW_RATE.
    bootstrap     resamples clusters, not rows: each resample draws as many clusters as there
                  are, with replacement, and keeps every row of each drawn cluster. 95%
                  percentile intervals; the seed is fixed.
    max depth     an arm may predict to fewer digits (max_depth, default 10). Top-1 deeper than
                  max_depth is n/a (NaN, n 0), never 0. The eval depth is min(8, max_depth):
                  recall at k, ECE, the coverage curve and the headline use it instead of 8.
                  A 6-digit code is valid when it is the first 6 digits of a current cn8.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from hs_rag_classifier.eval.arms import Prediction
from hs_rag_classifier.eval.items import EvalItem

DEPTHS = (2, 4, 6, 8, 10)
FULL_DEPTH = 10
RECALL_KS = (3, 10)
DIAGNOSTIC_RECALL_KS = (20, 50)
ECE_BINS = 15
REVIEW_RATES = tuple(round(0.05 * i, 2) for i in range(11))  # 0% to 50% in 5-point steps
HEADLINE_REVIEW_RATE = 0.20  # placeholder, configurable per run
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261021


def eval_depth(max_depth: int = FULL_DEPTH) -> int:
    """Depth for recall, ECE, coverage and the headline: 8, or the arm's max depth if lower."""
    return min(8, max_depth)


def is_valid(
    code: str | None, valid8: set[str], valid10: set[str], valid6: set[str] | None = None
) -> bool:
    if code is None:
        return False
    if len(code) == 6 and valid6 is not None:
        return code in valid6
    if len(code) == 8:
        return code in valid8
    if len(code) == 10:
        return code in valid10
    return False


def correct_at(
    code: str | None,
    valid: bool,
    label: str,
    label_depth: int,
    d: int,
    max_depth: int = FULL_DEPTH,
) -> float:
    """1.0 or 0.0 at d digits; NaN when d is deeper than the label or the arm's max depth."""
    if d > label_depth or d > max_depth:
        return math.nan
    if code is None or not valid or len(code) < d:
        return 0.0
    return float(code[:d] == label[:d])


def recall_at(
    codes: Sequence[str],
    label: str,
    k: int,
    valid8: set[str],
    valid10: set[str],
    depth: int = 8,
    valid6: set[str] | None = None,
) -> float:
    return float(
        any(
            is_valid(c, valid8, valid10, valid6) and c[:depth] == label[:depth]
            for c in codes[:k]
            if len(c) >= depth
        )
    )


def score(
    items: Sequence[EvalItem],
    predictions: Sequence[Prediction],
    valid8: set[str],
    valid10: set[str],
    latencies: Sequence[float] | None = None,
    max_depth: int = FULL_DEPTH,
) -> pd.DataFrame:
    """One row per item: label, top-1, validity, correctness by depth, recall, confidence.

    Recall columns are at the eval depth (8, or max_depth if lower).
    """
    by_id = {p.item_id: p for p in predictions}
    if set(by_id) != {i.item_id for i in items} or len(by_id) != len(predictions):
        raise ValueError("predictions must cover every item exactly once")
    depth = eval_depth(max_depth)
    valid6 = {c[:6] for c in valid8} if max_depth < 8 else None
    rows = []
    for n, item in enumerate(items):
        p = by_id[item.item_id]
        codes = [c.code for c in p.candidates]
        top1 = codes[0] if codes else None
        valid = is_valid(top1, valid8, valid10, valid6)
        row = {
            "item_id": item.item_id,
            "label_code": item.label_code,
            "label_depth": item.label_depth,
            "cluster_id": item.cluster_id,
            "chapter": item.chapter,
            "top1": top1,
            "candidates": codes,
            "scores": [c.score for c in p.candidates],
            "confidence": p.confidence,
            "abstain": p.abstain,
            "top1_valid": valid,
            "rationale": p.rationale,
            "tokens_in": p.tokens_in,
            "tokens_out": p.tokens_out,
            "latency_s": latencies[n] if latencies is not None else math.nan,
        }
        for d in DEPTHS:
            row[f"correct_{d}"] = correct_at(
                top1, valid, item.label_code, item.label_depth, d, max_depth
            )
        for k in RECALL_KS + DIAGNOSTIC_RECALL_KS:
            row[f"recall_{k}"] = recall_at(
                codes, item.label_code, k, valid8, valid10, depth, valid6
            )
        rows.append(row)
    return pd.DataFrame(rows)


def top1_accuracy(frame: pd.DataFrame) -> dict[int, dict[str, float]]:
    out = {}
    for d in DEPTHS:
        col = frame[f"correct_{d}"].dropna()
        out[d] = {"accuracy": float(col.mean()) if len(col) else math.nan, "n": len(col)}
    return out


def recall_ks(frame: pd.DataFrame) -> tuple[int, ...]:
    """3 and 10, plus each diagnostic k the arm returned at least k candidates for."""
    most = int(frame["candidates"].map(len).max()) if len(frame) else 0
    return RECALL_KS + tuple(k for k in DIAGNOSTIC_RECALL_KS if most >= k)


def recall(frame: pd.DataFrame, k: int) -> float:
    return float(frame[f"recall_{k}"].mean()) if len(frame) else math.nan


def invalid_rate(frame: pd.DataFrame) -> float:
    has = frame["top1"].notna()
    return float((~frame.loc[has, "top1_valid"]).mean()) if has.any() else math.nan


def _equal_mass_bins(confidence: np.ndarray, n_bins: int) -> list[np.ndarray]:
    order = np.argsort(confidence, kind="stable")
    return [b for b in np.array_split(order, min(n_bins, len(order))) if len(b)]


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> float:
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    n = len(confidence)
    if n == 0:
        return math.nan
    return float(
        sum(
            len(b) / n * abs(correct[b].mean() - confidence[b].mean())
            for b in _equal_mass_bins(confidence, n_bins)
        )
    )


def reliability(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS
) -> list[dict[str, float]]:
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    return [
        {
            "bin": i,
            "n": len(b),
            "conf_low": float(confidence[b].min()),
            "conf_high": float(confidence[b].max()),
            "mean_confidence": float(confidence[b].mean()),
            "accuracy": float(correct[b].mean()),
        }
        for i, b in enumerate(_equal_mass_bins(confidence, n_bins))
    ]


def coverage_curve(
    confidence: np.ndarray,
    correct: np.ndarray,
    abstain: np.ndarray,
    rates: Sequence[float] = REVIEW_RATES,
) -> list[dict[str, float]]:
    confidence = np.asarray(confidence, float)
    correct, abstain = np.asarray(correct, float), np.asarray(abstain, bool)
    n = len(confidence)
    # review queue: abstains first, then ascending confidence (stable for ties)
    order = np.lexsort((confidence, ~abstain))
    n_abstain = int(abstain.sum())
    out = []
    for r in rates:
        n_review = min(n, max(math.floor(r * n + 0.5), n_abstain))
        accepted = order[n_review:]
        out.append(
            {
                "review_rate": float(r),
                "actual_review_rate": n_review / n if n else math.nan,
                "n_accepted": len(accepted),
                "accuracy": float(correct[accepted].mean()) if len(accepted) else math.nan,
            }
        )
    return out


def accuracy_at_review(
    confidence: np.ndarray, correct: np.ndarray, abstain: np.ndarray, rate: float
) -> float:
    return coverage_curve(confidence, correct, abstain, [rate])[0]["accuracy"]


def latency_stats(latencies: Sequence[float], total_seconds: float) -> dict[str, float]:
    lat = np.asarray(latencies, float)
    return {
        "items_per_minute": len(lat) / total_seconds * 60 if total_seconds > 0 else math.nan,
        "latency_p50_s": float(np.percentile(lat, 50)) if len(lat) else math.nan,
        "latency_p95_s": float(np.percentile(lat, 95)) if len(lat) else math.nan,
    }


def per_chapter(frame: pd.DataFrame, max_depth: int = FULL_DEPTH) -> pd.DataFrame:
    """Per chapter: n, top-1 at 8 (and at 6 for a depth-limited arm), recall at 10."""
    cols = {"n": ("item_id", "size")}
    if max_depth < 8:
        cols["top1_6"] = ("correct_6", "mean")
    cols.update(top1_8=("correct_8", "mean"), recall_10=("recall_10", "mean"))
    return frame.groupby("chapter", sort=True).agg(**cols).reset_index()


Arrays = dict[str, np.ndarray]
Statistic = Callable[[Arrays], float]


def headline_stats(
    review_rate: float = HEADLINE_REVIEW_RATE, max_depth: int = FULL_DEPTH
) -> dict[str, Statistic]:
    """The bootstrapped statistics, as functions of the per-item arrays."""
    c = f"correct_{eval_depth(max_depth)}"
    stats = {
        "top1_6": lambda a: float(np.nanmean(a["correct_6"])),
        "top1_8": lambda a: float(np.nanmean(a["correct_8"])),
        "recall_3": lambda a: float(a["recall_3"].mean()),
        "ece": lambda a: ece(a["confidence"], a[c]),
        "headline": lambda a: accuracy_at_review(a["confidence"], a[c], a["abstain"], review_rate),
    }
    if max_depth < 8:
        del stats["top1_8"]
    return stats


def bootstrap_samples(
    frame: pd.DataFrame,
    stats: dict[str, Statistic],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    cluster_col: str = "cluster_id",
) -> dict[str, np.ndarray]:
    """Each statistic over n_resamples cluster resamples (see the module docstring)."""
    arrays = {c: frame[c].to_numpy() for c in frame.columns if c not in ("candidates", "scores")}
    groups = list(frame.groupby(cluster_col, sort=True).indices.values())
    rng = np.random.default_rng(seed)
    out = {name: np.empty(n_resamples) for name in stats}
    for b in range(n_resamples):
        drawn = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in drawn])
        sub = {c: v[idx] for c, v in arrays.items()}
        for name, fn in stats.items():
            out[name][b] = fn(sub)
    return out


def bootstrap_intervals(
    frame: pd.DataFrame,
    stats: dict[str, Statistic],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    samples = bootstrap_samples(frame, stats, n_resamples, seed)
    return {
        name: {
            "low": float(np.nanpercentile(v, 2.5)),
            "high": float(np.nanpercentile(v, 97.5)),
        }
        for name, v in samples.items()
    }


def compute_metrics(
    frame: pd.DataFrame,
    total_seconds: float,
    review_rate: float = HEADLINE_REVIEW_RATE,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    max_depth: int = FULL_DEPTH,
) -> dict:
    """Every metric for one run, JSON-friendly (correctness at the eval depth; see max depth)."""
    depth = eval_depth(max_depth)
    conf, c8 = frame["confidence"].to_numpy(float), frame[f"correct_{depth}"].to_numpy(float)
    abstain = frame["abstain"].to_numpy(bool)
    return {
        "n_items": len(frame),
        "n_clusters": int(frame["cluster_id"].nunique()),
        "max_depth": max_depth,
        "eval_depth": depth,
        "top1": top1_accuracy(frame),
        "recall": {k: recall(frame, k) for k in recall_ks(frame)},
        "invalid_rate": invalid_rate(frame),
        "no_top1_rate": float(frame["top1"].isna().mean()),
        "abstain_rate": float(abstain.mean()),
        "ece": ece(conf, c8),
        "ece_bins": ECE_BINS,
        "reliability": reliability(conf, c8),
        "coverage": coverage_curve(conf, c8, abstain),
        "headline_review_rate": review_rate,
        "headline": accuracy_at_review(conf, c8, abstain, review_rate),
        "latency": latency_stats(frame["latency_s"].to_numpy(float), total_seconds),
        "bootstrap": {
            "resamples": n_resamples,
            "seed": seed,
            "unit": "cluster_id",
            "intervals": bootstrap_intervals(
                frame, headline_stats(review_rate, max_depth), n_resamples, seed
            ),
        },
    }
