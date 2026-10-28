"""B1-ex: nearest training ruling (embedding kNN over train-split ruling descriptions).

Index: TRAIN rulings only (processed/rulings/v1/train.parquet), one vector per de-duplicated
ruling, text = its description only. Never keywords, never justifications, never validation or
test. Enforced in build_index: the input must be exactly the train split file
(guards.check_train_split_only) and goes through guards.check_retrieval_inputs; only the
columns in TRAIN_COLUMNS are read.

Encoder: arms.embed, the same model and settings as B1; rulings are documents (no prefix),
the query gets the model's instruction prefix.

Prediction, for a query with neighbours in descending similarity (ties: lower row first):
    pool       the top max(k, 50) training rulings.
    vote       each cn8 scores the sum of similarities of its neighbours among the top k
               (label_depth 8 and 10 rows both count at cn8); the pool score is the same sum
               over the whole pool.
    ranking    cn8s found in the pool, by vote, then pool score, then first appearance; at
               most 50. The top-1 cn8 is the vote winner among the k neighbours; the rest of
               the pool fills the list so recall at 20 and 50 is measured on the same pool.
    10 digits  each cn8 is returned as the code10 of its most similar label_depth-10
               neighbour in the pool; a cn8 with none is returned at 8 digits (counted wrong at
               10 by the harness).
    score      the vote (0 for cn8s found only in the pool).
Confidence: softmax over the top-10 votes at temperature T. B1-ex never abstains.

k is tuned on rulings validation from K_GRID and T fitted there (arms.embed_tune).

Index cache: ~/data/hs-rag-classifier/indexes/b1ex/<model>/<index hash>/, keyed on the encoder
settings, the index text and the SHA-256 of train.parquet.

Settings: configs/b1ex_embed.json, loaded by default.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.arms import embed as E
from hs_rag_classifier.arms.b0_bm25 import CANDIDATE_CAP, SOFTMAX_TOP, softmax_top1
from hs_rag_classifier.data import rulings as rulings_data
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval.arms import Candidate, Prediction, config_hash
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem
from hs_rag_classifier.guards import check_retrieval_inputs, check_train_split_only

CONFIG_PATH = REPO_ROOT / "configs" / "b1ex_embed.json"
INDEX_DIR = Path("indexes") / "b1ex"
TRAIN_REL = Path("processed") / "rulings" / rulings_data.SPLIT_NAME / "train.parquet"
TRAIN_REFS = REPO_ROOT / "splits" / f"rulings_{rulings_data.SPLIT_NAME}" / "train_refs.txt"
TRAIN_COLUMNS = ["ref", "description", "code10", "cn8", "label_depth"]
INDEX_TEXT = "train-split ruling description only (no keywords, no justification)"
POOL_MIN = 50
K_GRID = (1, 5, 10, 20, 50)
INDEX_KEYS = ("model", "hf_id", "revision", "dtype", "max_length", "index_text")


def default_config(model: str = "qwen3-embedding-0.6b", k: int = 10,
                   temperature: float = 1.0) -> dict[str, Any]:  # fmt: skip
    return {
        **E.model_config(model),
        "k": k,
        "pool": f"max(k, {POOL_MIN})",
        "temperature": temperature,
        "index_text": INDEX_TEXT,
        "train_split_file": str(TRAIN_REL),
        "train_refs_sha256": _sha_or_none(TRAIN_REFS),
        "nomenclature_version": rulings_data.NOMENCLATURE_VERSION,
        "candidate_cap": CANDIDATE_CAP,
        "softmax_top": SOFTMAX_TOP,
    }


def _sha_or_none(path: Path) -> str | None:
    return sha256(path) if path.exists() else None


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_key(config: dict[str, Any], train_sha256: str) -> str:
    return config_hash({**{k: config[k] for k in INDEX_KEYS}, "train_sha256": train_sha256})


def vote(
    sims: Sequence[float],
    cn8: Sequence[str],
    code10: Sequence[str],
    depth: Sequence[int],
    k: int,
    cap: int = CANDIDATE_CAP,
) -> list[tuple[str, float]]:
    """Ranked (code, vote) from neighbours already in descending similarity (the pool)."""
    votes: dict[str, float] = {}
    pool: dict[str, float] = {}
    first: dict[str, int] = {}
    best10: dict[str, str] = {}
    for i, (s, c8, c10, d) in enumerate(zip(sims, cn8, code10, depth, strict=True)):
        s = float(s)
        first.setdefault(c8, i)
        pool[c8] = pool.get(c8, 0.0) + s
        if i < k:
            votes[c8] = votes.get(c8, 0.0) + s
        if int(d) == 10 and c8 not in best10:
            best10[c8] = c10
    order = sorted(first, key=lambda c: (-votes.get(c, 0.0), -pool[c], first[c]))[:cap]
    return [(best10.get(c, c), votes.get(c, 0.0)) for c in order]


class B1exIndex:
    """Normalised train-ruling embeddings with their labels."""

    def __init__(self, embeddings: np.ndarray, rows: pd.DataFrame, meta: dict | None = None):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.cn8 = rows["cn8"].to_numpy(str)
        self.code10 = rows["code10"].to_numpy(str)
        self.depth = rows["label_depth"].to_numpy(int)
        self.meta = meta or {}

    def neighbours(self, queries: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(indices, similarities) of the top n rulings per query, most similar first."""
        sims = queries @ self.embeddings.T
        idx = np.argsort(-sims, axis=1, kind="stable")[:, :n]
        return idx, np.take_along_axis(sims, idx, axis=1)

    def predict_codes(self, query: np.ndarray, k: int, cap: int) -> list[tuple[str, float]]:
        (idx,), (sims,) = self.neighbours(query[None, :], max(k, POOL_MIN))
        return vote(sims, self.cn8[idx], self.code10[idx], self.depth[idx], k, cap)


def build_index(
    config: dict[str, Any],
    encoder: E.Encoder,
    root: Path = DATA_ROOT,
    train_path: Path | None = None,
    cache: bool = True,
) -> B1exIndex:
    """Build (or load the cached) index from the train split file only."""
    train_path = train_path or root / TRAIN_REL
    train_path = check_train_split_only(train_path, root / TRAIN_REL)
    (train_path,) = check_retrieval_inputs([train_path], root)
    rows = pd.read_parquet(train_path, columns=TRAIN_COLUMNS).reset_index(drop=True)
    key = index_key(config, sha256(train_path))
    meta = {"config": {k: config[k] for k in INDEX_KEYS}, "train_path": str(train_path),
            "refs": list(rows["ref"])}  # fmt: skip
    cache_dir = root / INDEX_DIR / config["model"] / key
    emb, meta = E.cached_embeddings(cache_dir, list(rows["description"]), encoder, meta, cache)
    return B1exIndex(emb, rows, meta)


class B1exEmbedArm:
    """kNN over train ruling descriptions, cn8 by summed similarity (rules in the docstring)."""

    name = "b1ex"
    batch_size = 64

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 encoder: E.Encoder | None = None, index: B1exIndex | None = None):  # fmt: skip
        self.config = dict(config or load_config())
        self.encoder = encoder or E.Encoder(self.config)
        self.index = index or build_index(self.config, self.encoder, root)

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        k, cap = int(self.config["k"]), int(self.config["candidate_cap"])
        temp, top = float(self.config["temperature"]), int(self.config["softmax_top"])
        queries = self.encoder.queries([i.text for i in items])
        out = []
        for item, q in zip(items, queries, strict=True):
            ranked = self.index.predict_codes(q, k, cap)
            out.append(
                Prediction(
                    item_id=item.item_id,
                    candidates=[Candidate(code=c, score=s) for c, s in ranked],
                    confidence=softmax_top1([s for _, s in ranked], temp, top),
                    abstain=False,
                )
            )
        return out
