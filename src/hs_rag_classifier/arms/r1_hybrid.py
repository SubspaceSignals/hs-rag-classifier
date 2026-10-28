"""R1: the hybrid retriever as a retrieval-only arm (validation diagnostics; no decider).

Candidates: the top 50 fused cn8s (8-digit codes, so the harness counts 10 digits as wrong);
top-1 = the top fused candidate. Confidence: softmax over the top-10 fused scores at the
temperature r1_temperature in configs/hybrid_retrieval.json (fitted on rulings validation).
Never abstains.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hs_rag_classifier.arms.b0_bm25 import SOFTMAX_TOP, softmax_top1
from hs_rag_classifier.eval.arms import Candidate, Prediction
from hs_rag_classifier.eval.items import DATA_ROOT, EvalItem
from hs_rag_classifier.retrieval.hybrid import HybridRetriever, load_config

CANDIDATE_CAP = 50


class R1HybridArm:
    name = "r1"
    batch_size = 64

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 retriever: HybridRetriever | None = None):  # fmt: skip
        self.config = dict(config or load_config())
        self.retriever = retriever or HybridRetriever(self.config, root)

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        temp = float(self.config["r1_temperature"])
        out = []
        for item, lists in zip(items, self.retriever.source_lists([i.text for i in items]),
                               strict=True):  # fmt: skip
            short = self.retriever.fuse(lists, k=CANDIDATE_CAP)
            scores = [c.fused for c in short.candidates]
            out.append(
                Prediction(
                    item_id=item.item_id,
                    candidates=[Candidate(code=c.cn8, score=c.fused) for c in short.candidates],
                    confidence=softmax_top1(scores, temp, SOFTMAX_TOP),
                    abstain=False,
                )
            )
        return out
