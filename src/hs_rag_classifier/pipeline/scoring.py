"""A2 option scoring: label distributions from one next-token step, pair ranking, temperature.

Label distribution: the decider is asked for the option label only and generates one token; the
server returns the log-probabilities of its top tokens before sampling (no grammar, no JSON
constraint). Each shown label takes its token's log-probability; a label absent from the
returned list takes the lowest returned log-probability as a floor. The outside-label mass is
1 minus the summed probability of the labels that were returned (the floor adds nothing), so it
measures how much of the raw distribution falls on tokens that are not option labels.

Renormalising: p(label) = softmax(logprob / T) over the shown labels, T the stage temperature.

Pairs: stage 1 gives p1 over HS6 options; stage 2 is run for the top two HS6 (a single-child
HS6 needs no call: p2 = 1). Every (hs6, cn8) pair scores p1(hs6) x p2(cn8 | hs6); candidates are
all pairs by that joint score, ties broken by stage-1 rank, then stage-2 rank.

Temperature: fitted on rulings validation by minimising the negative log-likelihood of the true
option, per stage (stage 1: items whose true HS6 was shown; stage 2: scorings of the true HS6
with more than one child and the true cn8 shown) or shared across both stages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

T_BOUNDS = (0.05, 20.0)


def label_logprobs(top: Sequence[Mapping], labels: Sequence[str]) -> dict:
    """Raw log-probabilities of each label from the returned top tokens (rule in the docstring).

    top: [{"token": str, "logprob": float}, ...] as the server returns them.
    Returns {"logprobs" (aligned with labels), "floored" (labels that took the floor),
    "outside_mass", "floor"}.
    """
    if not top:
        raise ValueError("the server returned no top log-probabilities")
    seen: dict[str, float] = {}
    for t in top:
        seen.setdefault(t["token"], float(t["logprob"]))
    floor = min(float(t["logprob"]) for t in top)
    lps = [seen.get(lab, floor) for lab in labels]
    floored = [lab for lab in labels if lab not in seen]
    inside = sum(math.exp(seen[lab]) for lab in set(labels) if lab in seen)
    return {
        "logprobs": lps,
        "floored": floored,
        "outside_mass": max(0.0, 1.0 - inside),
        "floor": floor,
    }


def renormalise(logprobs: Sequence[float], t: float = 1.0) -> np.ndarray:
    z = np.asarray(logprobs, float) / t
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def top_k_indices(logprobs: Sequence[float], k: int) -> list[int]:
    """The k best options, highest log-probability first, earlier option first on ties."""
    lps = np.asarray(logprobs, float)
    return list(np.lexsort((np.arange(len(lps)), -lps))[:k])


def rank_pairs(
    stage1: Mapping, stage2: Sequence[Mapping], t1: float = 1.0, t2: float = 1.0
) -> list[dict]:
    """All (hs6, cn8) pairs by joint probability (rule in the module docstring).

    stage1: {"options": [hs6...], "logprobs": [...]}; stage2: one entry per scored HS6,
    {"hs6", "cn8s", "logprobs"} (a single-child HS6 has one cn8 and logprobs [0.0]).
    """
    p1 = renormalise(stage1["logprobs"], t1)
    rank1 = {h: i for i, h in enumerate(stage1["options"])}
    pairs = []
    for s in stage2:
        i = rank1[s["hs6"]]
        p2 = renormalise(s["logprobs"], t2) if len(s["cn8s"]) > 1 else np.ones(1)
        for j, cn8 in enumerate(s["cn8s"]):
            pairs.append(
                {
                    "hs6": s["hs6"],
                    "cn8": cn8,
                    "p1": float(p1[i]),
                    "p2": float(p2[j]),
                    "joint": float(p1[i] * p2[j]),
                    "rank1": i,
                    "rank2": j,
                }
            )
    return sorted(pairs, key=lambda p: (-p["joint"], p["rank1"], p["rank2"]))


def nll(groups: Sequence[tuple[Sequence[float], int]], t: float) -> float:
    """Summed negative log-likelihood of the true option over (logprobs, true index) groups."""
    total = 0.0
    for lps, k in groups:
        z = np.asarray(lps, float) / t
        m = z.max()
        total -= z[k] - (m + math.log(np.exp(z - m).sum()))
    return float(total)


def fit_temperature(
    groups: Sequence[tuple[Sequence[float], int]], bounds: tuple[float, float] = T_BOUNDS
) -> dict:
    """T minimising nll over the groups (bounded search on log T). {"t", "nll", "nll_t1", "n"}."""
    from scipy.optimize import minimize_scalar

    if not groups:
        raise ValueError("no groups to fit a temperature on")
    lo, hi = (math.log(b) for b in bounds)
    res = minimize_scalar(
        lambda x: nll(groups, math.exp(x)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-4},
    )
    t = float(math.exp(res.x))
    return {"t": t, "nll": nll(groups, t), "nll_t1": nll(groups, 1.0), "n": len(groups)}
