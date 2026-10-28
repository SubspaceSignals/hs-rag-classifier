"""B1: dense-embedding baseline over the tariff path text.

Documents: the same as B0, one per current declarable 10-digit line in the pinned nomenclature,
text = its path_text (cleaned descriptions, chapter down to the line). No chapter notes, no
search references, no rulings text. The builder passes its input file list through
guards.check_retrieval_inputs.

Encoder: arms.embed (pinned model revision, fp16, max length 512, normalised vectors; the Qwen3
instruction prefix on queries only). Similarity is the dot product.

Ranking: exactly as B0. Every line is scored; each cn8 takes the max score of its lines; the top
50 cn8s are returned, each as its best-scoring 10-digit line. Ties go to the lower code.

Confidence: softmax over the top-10 cn8 scores at temperature T (fitted on rulings validation by
minimising the NLL of the top-1-correct indicator, as for B0). B1 never abstains.

Index cache: ~/data/hs-rag-classifier/indexes/b1/<model>/<index hash>/, keyed on the encoder
settings, the documents, the nomenclature version and the SHA-256 of codes.parquet.

Settings: configs/b1_embed.json, loaded by default; written by
    uv run python -m hs_rag_classifier.arms.embed_tune
which compares the models on rulings validation (see that module).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.arms import embed as E
from hs_rag_classifier.arms.b0_bm25 import (
    CANDIDATE_CAP,
    SOFTMAX_TOP,
    documents,
    rank_cn8,
    softmax_top1,
)
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.data.rulings import NOMENCLATURE_VERSION
from hs_rag_classifier.eval.arms import Candidate, Prediction, config_hash
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem
from hs_rag_classifier.guards import check_retrieval_inputs

CONFIG_PATH = REPO_ROOT / "configs" / "b1_embed.json"
INDEX_DIR = Path("indexes") / "b1"
DOCUMENTS = "current declarable 10-digit lines; text = path_text (chapter down to the line)"
INDEX_KEYS = ("model", "hf_id", "revision", "dtype", "max_length", "documents",
              "nomenclature_version")  # fmt: skip


def default_config(model: str = "qwen3-embedding-0.6b", temperature: float = 1.0) -> dict[str, Any]:
    return {
        **E.model_config(model),
        "temperature": temperature,
        "documents": DOCUMENTS,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "candidate_cap": CANDIDATE_CAP,
        "softmax_top": SOFTMAX_TOP,
    }


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_key(config: dict[str, Any], codes_sha256: str) -> str:
    return config_hash({**{k: config[k] for k in INDEX_KEYS}, "codes_sha256": codes_sha256})


class B1Index:
    """Normalised line embeddings sorted by code10, with the cn8 grouping B0 uses."""

    def __init__(self, embeddings: np.ndarray, code10: Sequence[str], meta: dict | None = None):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.code10 = np.asarray(code10)
        cn8 = np.array([c[:8] for c in self.code10])
        self.starts = np.flatnonzero(np.r_[True, cn8[1:] != cn8[:-1]])
        self.cn8 = cn8[self.starts]
        self.meta = meta or {}

    def line_scores(self, queries: np.ndarray) -> np.ndarray:
        """(n_queries, n_lines) dot products."""
        return queries @ self.embeddings.T

    def rank(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        return rank_cn8(scores, self.starts, self.code10, k)


def build_index(
    config: dict[str, Any],
    encoder: E.Encoder,
    root: Path = DATA_ROOT,
    codes_path: Path | None = None,
    cache: bool = True,
) -> B1Index:
    """Build (or load the cached) index; the input file list goes through the leakage guard."""
    codes_path = (
        codes_path or root / "processed" / "tariff" / NOMENCLATURE_VERSION / "codes.parquet"
    )
    (codes_path,) = check_retrieval_inputs([codes_path], root)
    key = index_key(config, sha256(codes_path))
    cols = ["code10", "cn8", "path_text", "declarable", "superseded"]
    docs = documents(pd.read_parquet(codes_path, columns=cols))
    meta = {"config": {k: config[k] for k in INDEX_KEYS}, "codes_path": str(codes_path),
            "code10": list(docs["code10"])}  # fmt: skip
    cache_dir = root / INDEX_DIR / config["model"] / key
    emb, meta = E.cached_embeddings(cache_dir, list(docs["path_text"]), encoder, meta, cache)
    return B1Index(emb, list(docs["code10"]), meta)


class B1EmbedArm:
    """Dense retrieval over tariff path text, grouped to cn8 (rules in the module docstring)."""

    name = "b1"
    batch_size = 64

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 encoder: E.Encoder | None = None, index: B1Index | None = None):  # fmt: skip
        self.config = dict(config or load_config())
        self.encoder = encoder or E.Encoder(self.config)
        self.index = index or build_index(self.config, self.encoder, root)

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        cap, temp = int(self.config["candidate_cap"]), float(self.config["temperature"])
        top = int(self.config["softmax_top"])
        scores = self.index.line_scores(self.encoder.queries([i.text for i in items]))
        out = []
        for item, row in zip(items, scores, strict=True):
            ranked = self.index.rank(row, cap)
            out.append(
                Prediction(
                    item_id=item.item_id,
                    candidates=[Candidate(code=c, score=s) for c, s in ranked],
                    confidence=softmax_top1([s for _, s in ranked], temp, top),
                    abstain=False,
                )
            )
        return out
