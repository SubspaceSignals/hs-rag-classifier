"""Reciprocal rank fusion, the HS6 view, and the shortlist-length rule. Pure functions.

RRF: fused(c) = sum over sources s of w_s / (k_rrf + rank_s(c)), ranks from 1; a cn8 missing
from a source gets nothing from it. Weights are normalised to sum to 1, so (1, 1, 1) and
(2, 2, 2) are the same fusion. Ties: the better best rank across sources, then the lower cn8.

Shortlist length: the smallest K whose recall gain to the next K is under min_gain (the knee),
provided the K-candidate list fits the token budget at p95. If the budget binds first, the
largest K that fits is chosen and budget_binds is set; both Ks are reported.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence


def normalise_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0 or any(w < 0 for w in weights.values()):
        raise ValueError(f"weights must be non-negative with a positive sum: {weights}")
    return {s: w / total for s, w in weights.items()}


def weight_grid(sources: Sequence[str], values: Sequence[float]) -> list[dict[str, float]]:
    """Every weight combination, normalised, with duplicate ratios dropped (first kept)."""
    seen, out = set(), []
    for combo in itertools.product(values, repeat=len(sources)):
        w = normalise_weights(dict(zip(sources, combo, strict=True)))
        key = tuple(round(w[s], 9) for s in sources)
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def rrf(
    rankings: Mapping[str, Sequence[str]], weights: Mapping[str, float], k_rrf: float
) -> list[tuple[str, float]]:
    """Fused (cn8, score), best first. rankings: per source, cn8s in rank order (rank 1 first)."""
    w = normalise_weights({s: weights[s] for s in rankings})
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for s, ranked in rankings.items():
        for rank, c in enumerate(ranked, start=1):
            score[c] = score.get(c, 0.0) + w[s] / (k_rrf + rank)
            best[c] = min(best.get(c, rank), rank)
    return sorted(score.items(), key=lambda e: (-e[1], best[e[0]], e[0]))


def hs6_view(cn8s: Sequence[str]) -> list[str]:
    """Unique HS6 codes in shortlist order."""
    return list(dict.fromkeys(c[:6] for c in cn8s))


def select_k(
    ks: Sequence[int],
    recall: Mapping[int, float],
    p95_tokens: Mapping[int, float],
    budget: float,
    min_gain: float = 0.01,
) -> dict:
    """Chosen K, the knee K, and whether the token budget bound (rule in the module doc)."""
    ks = sorted(ks)
    pairs = itertools.pairwise(ks)
    knee = next((k for k, nxt in pairs if recall[nxt] - recall[k] < min_gain), ks[-1])
    fits = [k for k in ks if p95_tokens[k] <= budget]
    if knee in fits:
        return {"k": knee, "knee_k": knee, "budget_binds": False}
    below = [k for k in fits if k < knee]
    return {"k": below[-1] if below else None, "knee_k": knee, "budget_binds": True}
