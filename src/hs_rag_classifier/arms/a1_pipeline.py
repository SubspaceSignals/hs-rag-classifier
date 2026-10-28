"""A1: the LangGraph pipeline (profile, retrieve, two-stage decide, validate) as an arm.

Candidates: the pipeline's top 3 cn8 codes (8 digits: max_depth 8, so 10 digits is n/a).
Confidence: stage-1 x stage-2 stated confidence / 100^2. Never abstains (the threshold is tuned
in Phase 3). One item per predict call, so the harness latency is single-item wall time; the
prediction's details hold per-node latency and tokens, flags, both stages' choices and options,
the citation checks and the validation.

Config: configs/a1_pipeline.json. load_config refuses it when the hybrid retriever config or a
prompt template has changed since (SHA-256).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval.arms import Candidate, Prediction
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem
from hs_rag_classifier.pipeline.graph import PROMPTS

CONFIG_PATH = REPO_ROOT / "configs" / "a1_pipeline.json"
PROMPT_DIR = "src/hs_rag_classifier/pipeline/prompts"
RETRIEVER_CONFIG = "configs/hybrid_retrieval.json"
SEED = 20261020

DEFAULTS: dict[str, Any] = {
    "profile": False,
    "evidence": False,
    "evidence_k": 3,
    "evidence_chars": 300,
    "max_input_chars": 1000,
    "token_budget": 4000,
    "rationale_chars": 400,
    "profile_field_chars": 80,
    "max_tokens": {"profile": 200, "decide": 260},
    "client": {
        "temperature": 0,
        "seed": SEED,
        "enable_thinking": False,
        "base_url": "http://localhost:8080",
    },
}


def file_entries(repo: Path = REPO_ROOT) -> dict[str, Any]:
    return {
        "retriever_config": {"path": RETRIEVER_CONFIG, "sha256": sha256(repo / RETRIEVER_CONFIG)},
        "prompts": {
            p: {"path": f"{PROMPT_DIR}/{p}.txt", "sha256": sha256(repo / PROMPT_DIR / f"{p}.txt")}
            for p in PROMPTS
        },
    }


def check_files(config: dict[str, Any], repo: Path = REPO_ROOT) -> None:
    entries = [config["retriever_config"], *config["prompts"].values()]
    for e in entries:
        if sha256(repo / e["path"]) != e["sha256"]:
            raise RuntimeError(f"{e['path']} changed since configs/a1_pipeline.json was written")


def load_config(path: Path = CONFIG_PATH, repo: Path = REPO_ROOT) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    check_files(config, repo)
    return config


def to_prediction(item_id: str, state: dict) -> Prediction:
    final = state["final"]
    top3 = final["top3"]
    tokens = state.get("tokens", {})
    t_in = sum(v[0] or 0 for v in tokens.values())
    t_out = sum(v[1] or 0 for v in tokens.values())
    s1, s2 = state["stage1"], state["stage2"]
    details = {
        "flags": final["flags"],
        "disclaimer": final["disclaimer"],
        "timings": state.get("timings", {}),
        "tokens": tokens,
        "profile": state.get("profile"),
        "evidence": state.get("evidence"),
        "stage1": {
            k: s1[k]
            for k in (
                "options",
                "choice_code",
                "alternative_codes",
                "confidence",
                "rationale",
                "citation",
                "parsed",
            )
        },
        "stage2": {
            k: s2.get(k)
            for k in (
                "choice_code",
                "alternative_codes",
                "confidence",
                "rationale",
                "citation",
                "skipped",
                "overflow",
                "n_children",
                "n_options",
                "parsed",
            )
        },
        "validation": state["validation"],
    }
    return Prediction(
        item_id=item_id,
        candidates=[Candidate(code=c, score=1.0 / (r + 1)) for r, c in enumerate(top3)],
        confidence=final["confidence"],
        abstain=False,
        rationale=final["rationale"],
        tokens_in=t_in,
        tokens_out=t_out,
        details=details,
    )


class A1PipelineArm:
    name = "a1"
    max_depth = 8
    batch_size = 1

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 pipeline=None):  # fmt: skip
        self.config = dict(config or load_config())
        self.pipeline = pipeline or build_pipeline(self.config, root)
        self.graph = self.pipeline.build()

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        out = []
        for item in items:
            state = self.graph.invoke({"item_id": item.item_id, "raw": item.text})
            out.append(to_prediction(item.item_id, state))
        return out


def build_pipeline(config: dict[str, Any], root: Path = DATA_ROOT, retriever=None, tariff=None,
                   api: bool = True):  # fmt: skip
    """The real parts: llama-server chat client, hybrid retriever, tariff texts, validator."""
    from hs_rag_classifier.pipeline.graph import Pipeline
    from hs_rag_classifier.pipeline.llm import make_chat
    from hs_rag_classifier.pipeline.options import load_tariff
    from hs_rag_classifier.pipeline.validate import EvidenceIndex, Validator, make_api
    from hs_rag_classifier.retrieval.hybrid import HybridRetriever
    from hs_rag_classifier.retrieval.hybrid import load_config as load_hybrid

    tariff = tariff or load_tariff(root)
    retriever = retriever or HybridRetriever(load_hybrid(), root)
    chat = make_chat(config["client"], config["client"]["base_url"])
    validator = Validator(tariff, make_api() if api else None)
    evidence = EvidenceIndex(root) if config["evidence"] else None
    return Pipeline(config, chat, retriever, tariff, validator, evidence)


def diagnostics(frame) -> dict[str, Any]:
    """A1-specific rates from the scored frame's details column."""
    import numpy as np

    rows = [json.loads(d) for d in frame["details"]]
    n = len(rows)
    labels = list(frame["label_code"])

    def cite(r: dict) -> bool:
        stages = [r["stage1"]["citation"]] + (
            [] if r["stage2"]["skipped"] else [r["stage2"]["citation"]]
        )
        return all(c and c["pass"] for c in stages)

    flags: dict[str, int] = {}
    for r in rows:
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    tokens = [sum((v[0] or 0) + (v[1] or 0) for v in r["tokens"].values()) for r in rows]
    nodes = sorted({k for r in rows for k in r["timings"]})
    s2 = [r for r in rows if not r["stage2"]["skipped"]]
    return {
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
        "citation_pass_rate": float(np.mean([cite(r) for r in rows])),
        "citation_pass_stage1": float(
            np.mean([bool(r["stage1"]["citation"]["pass"]) for r in rows])
        ),
        "citation_pass_stage2": float(np.mean([bool(r["stage2"]["citation"]["pass"]) for r in s2]))
        if s2
        else None,
        "fallbacks": int(sum(r["validation"]["fallback"] for r in rows)),
        "stage2_skipped": n - len(s2),
        "stage2_overflow": int(sum(bool(r["stage2"]["overflow"]) for r in rows)),
        "flags": flags,
        "mean_tokens_per_item": float(np.mean(tokens)),
        "mean_tokens_in": float(frame["tokens_in"].mean()),
        "mean_tokens_out": float(frame["tokens_out"].mean()),
        "node_latency_p50_s": {
            k: float(np.median([r["timings"][k] for r in rows if k in r["timings"]])) for k in nodes
        },
        "node_latency_p95_s": {
            k: float(np.percentile([r["timings"][k] for r in rows if k in r["timings"]], 95))
            for k in nodes
        },
    }
