"""A2: option scoring from token probabilities, plus a short post-hoc rationale, as an arm.

The pipeline is A1's (configs/a1_pipeline.json, used unchanged: same options, labels, evidence
and retriever) with the decisions read from the next-token distribution (pipeline/graph_a2.py).
Candidates: every scored (hs6, cn8) pair's cn8 in joint order, the validated choice first,
capped at the harness's MAX_CANDIDATES (50).
Confidence: the final pair's joint probability after the stage temperatures. Never abstains.
One item per predict call, so the harness latency is single-item wall time including the
rationale; details hold per-node timings, so score-only latency is that minus the rationale node.

Config: configs/a2_pipeline.json. load_config refuses it when configs/a1_pipeline.json, the
retriever config or any A1 or A2 prompt template has changed since (SHA-256).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval.arms import MAX_CANDIDATES, Candidate, Prediction
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem
from hs_rag_classifier.pipeline import scoring
from hs_rag_classifier.pipeline.graph_a2 import A2_PROMPTS

CONFIG_PATH = REPO_ROOT / "configs" / "a2_pipeline.json"
BASE_CONFIG = "configs/a1_pipeline.json"
RANKED_KEPT = 10


def prompt_entries(repo: Path = REPO_ROOT) -> dict[str, Any]:
    d = A1.PROMPT_DIR
    return {
        p: {"path": f"{d}/{p}.txt", "sha256": sha256(repo / d / f"{p}.txt")} for p in A2_PROMPTS
    }


def check_files(a2: dict[str, Any], repo: Path = REPO_ROOT) -> None:
    for e in [a2["base_config"], *a2["prompts"].values()]:
        if sha256(repo / e["path"]) != e["sha256"]:
            raise RuntimeError(f"{e['path']} changed since configs/a2_pipeline.json was written")


def load_config(path: Path = CONFIG_PATH, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """A1's locked config with the A2 section under "a2"."""
    a2 = json.loads(path.read_text(encoding="utf-8"))
    check_files(a2, repo)
    base = A1.load_config(repo / a2["base_config"]["path"], repo)
    return {**base, "a2": a2}


def stage2_view(d: dict) -> dict:
    keys = (
        "hs6",
        "cn8s",
        "logprobs",
        "floored",
        "outside_mass",
        "token",
        "top",
        "skipped",
        "overflow",
        "n_children",
        "llm_s",
    )
    return {k: d.get(k) for k in keys}


def to_prediction(item_id: str, state: dict) -> Prediction:
    final = state["final"]
    tokens = state.get("tokens", {})
    joint = {p["cn8"]: p["joint"] for p in state["ranked"]}
    s1 = state["stage1"]
    details = {
        "flags": final["flags"],
        "disclaimer": final["disclaimer"],
        "timings": state.get("timings", {}),
        "tokens": tokens,
        "evidence": state.get("evidence"),
        "stage1": {
            k: s1.get(k)
            for k in (
                "options",
                "logprobs",
                "floored",
                "outside_mass",
                "token",
                "top",
                "choice_code",
                "llm_s",
            )
        },
        "stage2": [stage2_view(d) for d in state["stage2_scored"]],
        "ranked": state["ranked"][:RANKED_KEPT],
        "validation": state["validation"],
        "rationale": state.get("rationale"),
    }
    return Prediction(
        item_id=item_id,
        candidates=[
            Candidate(code=c, score=joint[c]) for c in final["candidates"][:MAX_CANDIDATES]
        ],
        confidence=final["confidence"],
        abstain=False,
        rationale=final["rationale"],
        tokens_in=sum(v[0] or 0 for v in tokens.values()),
        tokens_out=sum(v[1] or 0 for v in tokens.values()),
        details=details,
    )


class A2PipelineArm:
    name = "a2"
    max_depth = 8
    batch_size = 1

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT, pipeline=None):
        self.config = dict(config or load_config())
        self.pipeline = pipeline or build_pipeline(self.config, root)
        self.graph = self.pipeline.build()

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        out = []
        for item in items:
            state = self.graph.invoke({"item_id": item.item_id, "raw": item.text})
            out.append(to_prediction(item.item_id, state))
        return out


def build_pipeline(
    config: dict[str, Any], root: Path = DATA_ROOT, retriever=None, tariff=None, api: bool = True
):
    """The real parts: A1's (chat client, retriever, tariff, validator, evidence) plus a raw
    HTTP client for scoring and the cn8 path texts."""
    import httpx

    from hs_rag_classifier.pipeline.graph_a2 import A2Pipeline
    from hs_rag_classifier.pipeline.options import load_cn8_paths

    a1 = A1.build_pipeline(config, root, retriever, tariff, api)
    http = httpx.Client(
        base_url=config["client"]["base_url"], timeout=config["client"].get("timeout_s", 120)
    )
    return A2Pipeline(
        config,
        a1.chat,
        a1.retriever,
        a1.tariff,
        a1.validator,
        http,
        load_cn8_paths(root),
        a1.evidence_index,
    )


def rescore(details: dict, t1: float, t2: float) -> tuple[list[str], float]:
    """Candidates and confidence of a saved prediction under other temperatures (no model run).

    The scored HS6 do not depend on T (it keeps the stage-1 order), so only the ranking and the
    joint probabilities change. Codes the validator rejected stay rejected; others are valid.
    """
    stage1 = {"options": details["stage1"]["options"], "logprobs": details["stage1"]["logprobs"]}
    ranked = scoring.rank_pairs(stage1, details["stage2"], t1, t2)
    checks = details["validation"]["checks"]
    final = next((p for p in ranked if checks.get(p["cn8"], {"valid": True})["valid"]), ranked[0])
    order = list(dict.fromkeys([final["cn8"], *(p["cn8"] for p in ranked)]))
    return order[:MAX_CANDIDATES], final["joint"]


def temperature_groups(rows: Sequence[dict], labels: Sequence[str]) -> dict[str, list]:
    """(logprobs, true index) groups per stage for fitting T (rule in pipeline/scoring.py)."""
    g1, g2 = [], []
    for r, lab in zip(rows, labels, strict=True):
        opts = r["stage1"]["options"]
        if lab[:6] in opts:
            g1.append((r["stage1"]["logprobs"], opts.index(lab[:6])))
        for s in r["stage2"]:
            if s["hs6"] == lab[:6] and not s["skipped"] and lab[:8] in s["cn8s"]:
                g2.append((s["logprobs"], s["cn8s"].index(lab[:8])))
    return {"stage1": g1, "stage2": g2}


def diagnostics(frame) -> dict[str, Any]:
    """A2-specific rates from the scored frame's details column."""
    rows = [json.loads(d) for d in frame["details"]]
    n = len(rows)
    labels = list(frame["label_code"])
    s2 = [s for r in rows for s in r["stage2"] if not s["skipped"]]
    out1 = np.array([r["stage1"]["outside_mass"] for r in rows])
    out2 = np.array([s["outside_mass"] for s in s2]) if s2 else np.array([np.nan])
    rat = [r["rationale"] or {} for r in rows]
    t_all = np.asarray(frame["latency_s"], float) if "latency_s" in frame else None
    t_rat = np.array([r["timings"].get("rationale", 0.0) for r in rows])
    flags: dict[str, int] = {}
    for r in rows:
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    labels_all = json.loads((REPO_ROOT / BASE_CONFIG).read_text(encoding="utf-8"))["labels"]

    def tsum(r: dict, prefixes: tuple[str, ...]) -> tuple[int, int]:
        v = [x for k, x in r["tokens"].items() if k.startswith(prefixes)]
        return sum(x[0] or 0 for x in v), sum(x[1] or 0 for x in v)

    score_tok = [tsum(r, ("stage1", "stage2")) for r in rows]
    rat_tok = [tsum(r, ("rationale",)) for r in rows]
    out = {
        "n_items": n,
        "hs6_recall_at_m": float(
            np.mean(
                [lab[:6] in r["stage1"]["options"] for lab, r in zip(labels, rows, strict=True)]
            )
        ),
        "stage1_hs6_accuracy": float(
            np.mean(
                [lab[:6] == r["stage1"]["choice_code"] for lab, r in zip(labels, rows, strict=True)]
            )
        ),
        "outside_mass_stage1": {"mean": float(out1.mean()), "p95": float(np.percentile(out1, 95))},
        "outside_mass_stage2": {
            "mean": float(np.nanmean(out2)),
            "p95": float(np.nanpercentile(out2, 95)),
            "n": len(s2),
        },
        "floor_needed_stage1": float(np.mean([bool(r["stage1"]["floored"]) for r in rows])),
        "floor_needed_stage2": float(np.mean([bool(s["floored"]) for s in s2])) if s2 else None,
        "mean_floored_labels_stage1": float(np.mean([len(r["stage1"]["floored"]) for r in rows])),
        "first_token_not_a_label_stage1": float(
            np.mean(
                [
                    r["stage1"]["token"] not in labels_all[: len(r["stage1"]["options"])]
                    for r in rows
                ]
            )
        ),
        "stage2_calls_per_item": len(s2) / n,
        "citation_pass_rate": float(
            np.mean([bool((x.get("citation") or {}).get("pass")) for x in rat])
        ),
        "rationale_parse_errors": int(sum(not x.get("parsed") for x in rat)),
        "explanation_over_60_words": int(sum((x.get("n_words") or 0) > 60 for x in rat)),
        "fallbacks": int(sum(r["validation"]["fallback"] for r in rows)),
        "stage2_overflow": int(sum(any(s["overflow"] for s in r["stage2"]) for r in rows)),
        "flags": flags,
        "mean_tokens_scoring": [
            float(np.mean([t[0] for t in score_tok])),
            float(np.mean([t[1] for t in score_tok])),
        ],
        "mean_tokens_rationale": [
            float(np.mean([t[0] for t in rat_tok])),
            float(np.mean([t[1] for t in rat_tok])),
        ],
        "rationale_latency_p50_s": float(np.median(t_rat)),
        "rationale_latency_p95_s": float(np.percentile(t_rat, 95)),
    }
    if t_all is not None:
        score_only = t_all - t_rat
        out["speed"] = {
            "with_rationale": {
                "items_per_minute": 60 / t_all.mean(),
                "p50_s": float(np.median(t_all)),
                "p95_s": float(np.percentile(t_all, 95)),
            },
            "score_only": {
                "items_per_minute": 60 / score_only.mean(),
                "p50_s": float(np.median(score_only)),
                "p95_s": float(np.percentile(score_only, 95)),
            },
        }
    return out
