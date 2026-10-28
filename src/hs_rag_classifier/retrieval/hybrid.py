"""Hybrid retriever: three sources fused by reciprocal rank fusion into a cn8 shortlist.

Sources (each the LOCKED arm config, unchanged; each returns up to SOURCE_DEPTH ranked cn8s):
    keyword       B0 BM25 over tariff path text (configs/b0_bm25.json); a cn8 scores its best
                  line, as in B0. cn8s whose lines match no query term (score 0) are left out.
    tariff_embed  B1 bge-m3 line embeddings (configs/b1_embed.json) in a VectorStore, ids =
                  code10; a cn8 scores its best line, as in B1 (exact search over every line).
    rulings       B1-ex bge-m3 train-ruling embeddings (configs/b1ex_embed.json) in a
                  VectorStore, ids = ruling ref; cn8s ranked by B1-ex's neighbour vote (k and
                  the max(k, 50) pool as locked), so this source returns at most as many cn8s
                  as the pool holds. Each hit keeps its supporting train refs and similarities.
The embedding sources share one query encoding (both configs use the same encoder settings).
Index builders are the arms' own: every input goes through guards.check_retrieval_inputs, and
the rulings index also through guards.check_train_split_only. The cn8 metadata
(codes8.parquet) goes through check_retrieval_inputs too.

Fusion (retrieval.fusion.rrf): weights and k_rrf from configs/hybrid_retrieval.json, which the
retriever loads by default and checks against the SHA-256 of each source config.

Shortlist: the top K fused cn8s as candidate records (cn8, hs6, heading, chapter, cleaned
path_text, fused score, and per source its rank and score, or None when absent), plus the HS6
view (unique HS6 codes in shortlist order) for the decider's first stage.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.arms import b0_bm25 as B0
from hs_rag_classifier.arms import b1_embed as B1
from hs_rag_classifier.arms import b1ex_embed as BX
from hs_rag_classifier.arms import embed as E
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.data.rulings import NOMENCLATURE_VERSION
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT
from hs_rag_classifier.guards import check_retrieval_inputs
from hs_rag_classifier.retrieval.fusion import hs6_view, rrf
from hs_rag_classifier.retrieval.store import InMemoryStore, VectorStore

CONFIG_PATH = REPO_ROOT / "configs" / "hybrid_retrieval.json"
SOURCES = ("keyword", "tariff_embed", "rulings")
SOURCE_CONFIGS = {"keyword": "configs/b0_bm25.json", "tariff_embed": "configs/b1_embed.json",
                  "rulings": "configs/b1ex_embed.json"}  # fmt: skip
SOURCE_DEPTH = 100
ENCODER_KEYS = ("model", "hf_id", "revision", "dtype", "max_length", "query_instruction")


@dataclass(frozen=True)
class Hit:
    cn8: str
    rank: int
    score: float
    support: tuple[tuple[str, float], ...] = ()


@dataclass
class CandidateRecord:
    cn8: str
    hs6: str
    heading: str
    chapter: str
    path_text: str
    fused: float
    sources: dict[str, dict[str, float] | None]
    rulings_support: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Shortlist:
    candidates: list[CandidateRecord]
    hs6: list[str]

    @property
    def cn8(self) -> list[str]:
        return [c.cn8 for c in self.candidates]


def source_config_entries(repo: Path = REPO_ROOT) -> dict[str, dict[str, str]]:
    return {s: {"path": p, "sha256": sha256(repo / p)} for s, p in SOURCE_CONFIGS.items()}


def default_config(k_rrf: float = 60, weights: dict[str, float] | None = None,
                   k: int = 20) -> dict[str, Any]:  # fmt: skip
    return {
        "sources": source_config_entries(),
        "source_depth": SOURCE_DEPTH,
        "k_rrf": k_rrf,
        "weights": weights or {s: 1 / 3 for s in SOURCES},
        "k": k,
        "nomenclature_version": NOMENCLATURE_VERSION,
    }


def load_config(path: Path = CONFIG_PATH, repo: Path = REPO_ROOT) -> dict[str, Any]:
    """The hybrid config; refuses it if a locked source config has changed since."""
    config = json.loads(path.read_text(encoding="utf-8"))
    for s, entry in config["sources"].items():
        if sha256(repo / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"{entry['path']} changed since the hybrid config was tuned ({s})")
    return config


# --- Sources ---------------------------------------------------------------------------------


class KeywordSource:
    name = "keyword"

    def __init__(self, index: B0.B0Index):
        self.index = index

    def search_tokens(self, tokens: list[str], n: int) -> list[Hit]:
        """B0's ranking, without lines that match no query term (score 0): in a fusion those
        would only add votes for the lowest codes."""
        ranked = [(c, s) for c, s in self.index.rank(self.index.line_scores(tokens), n) if s > 0]
        return [Hit(code[:8], r, s) for r, (code, s) in enumerate(ranked, start=1)]

    def search(self, text: str, n: int = SOURCE_DEPTH) -> list[Hit]:
        return self.search_tokens(B0.tokenise([text])[0], n)


class TariffEmbedSource:
    name = "tariff_embed"

    def __init__(self, store: VectorStore):
        self.store = store

    def search(self, query: np.ndarray, n: int = SOURCE_DEPTH) -> list[Hit]:
        (ids,), (scores,) = self.store.search(query[None, :], len(self.store))
        cn8 = ids.astype("<U8")  # ids are code10: the first 8 digits
        _, first = np.unique(cn8, return_index=True)  # best line per cn8 (results sorted)
        top = np.sort(first)[:n]
        return [Hit(str(cn8[i]), r, float(scores[i])) for r, i in enumerate(top, start=1)]


class RulingsSource:
    name = "rulings"

    def __init__(self, store: VectorStore, labels: pd.DataFrame, k: int, pool: int):
        self.store, self.k, self.pool = store, k, pool
        self.row = {ref: i for i, ref in enumerate(labels["ref"])}
        self.cn8 = labels["cn8"].to_numpy(str)
        self.code10 = labels["code10"].to_numpy(str)
        self.depth = labels["label_depth"].to_numpy(int)

    def search(self, query: np.ndarray, n: int = SOURCE_DEPTH) -> list[Hit]:
        (ids,), (sims,) = self.store.search(query[None, :], self.pool)
        pos = np.array([self.row[r] for r in ids])
        ranked = BX.vote(sims, self.cn8[pos], self.code10[pos], self.depth[pos], self.k, n)
        support: dict[str, list[tuple[str, float]]] = {}
        for ref, c8, s in zip(ids, self.cn8[pos], sims, strict=True):
            support.setdefault(str(c8), []).append((str(ref), round(float(s), 6)))
        return [Hit(code[:8], r, s, tuple(support[code[:8]]))
                for r, (code, s) in enumerate(ranked, start=1)]  # fmt: skip


def build_sources(root: Path = DATA_ROOT, repo: Path = REPO_ROOT, encoder: E.Encoder | None = None,
                  ) -> tuple[dict[str, Any], E.Encoder]:  # fmt: skip
    """The three sources from their locked configs (each builder runs the guards)."""
    b0_cfg = B0.load_config(repo / SOURCE_CONFIGS["keyword"])
    b1_cfg = B1.load_config(repo / SOURCE_CONFIGS["tariff_embed"])
    bx_cfg = BX.load_config(repo / SOURCE_CONFIGS["rulings"])
    if any(b1_cfg[k] != bx_cfg[k] for k in ENCODER_KEYS):
        raise ValueError("the tariff and rulings sources must share one encoder")
    encoder = encoder or E.Encoder(b1_cfg)
    b1 = B1.build_index(b1_cfg, encoder, root)
    tariff = InMemoryStore()
    tariff.add(list(b1.code10), b1.embeddings)
    bx = BX.build_index(bx_cfg, encoder, root)
    refs = bx.meta["refs"]
    rulings = InMemoryStore()
    rulings.add(refs, bx.embeddings)
    labels = pd.DataFrame({"ref": refs, "cn8": bx.cn8, "code10": bx.code10,
                           "label_depth": bx.depth})  # fmt: skip
    k = int(bx_cfg["k"])
    sources = {
        "keyword": KeywordSource(B0.build_index(b0_cfg, root)),
        "tariff_embed": TariffEmbedSource(tariff),
        "rulings": RulingsSource(rulings, labels, k, max(k, BX.POOL_MIN)),
    }
    return sources, encoder


def load_cn8_meta(root: Path = DATA_ROOT) -> dict[str, dict[str, str]]:
    path = root / "processed" / "tariff" / NOMENCLATURE_VERSION / "codes8.parquet"
    (path,) = check_retrieval_inputs([path], root)
    df = pd.read_parquet(path, columns=["cn8", "subheading", "heading", "chapter", "path_text"])
    return {r.cn8: {"hs6": r.subheading, "heading": r.heading, "chapter": r.chapter,
                    "path_text": r.path_text} for r in df.itertuples(index=False)}  # fmt: skip


# --- Retriever -------------------------------------------------------------------------------


class HybridRetriever:
    """Source lists, RRF fusion and candidate records (rules in the module docstring)."""

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 sources: dict[str, Any] | None = None, encoder: E.Encoder | None = None,
                 meta: dict[str, dict[str, str]] | None = None):  # fmt: skip
        self.config = dict(config or load_config())
        if sources is None:
            sources, encoder = build_sources(root, encoder=encoder)
        self.sources, self.encoder = sources, encoder
        self.meta = meta if meta is not None else load_cn8_meta(root)
        self.depth = int(self.config.get("source_depth", SOURCE_DEPTH))

    def source_lists(self, texts: Sequence[str]) -> list[dict[str, list[Hit]]]:
        """Per text, each source's ranked hits (one batched query encoding)."""
        queries = self.encoder.queries(list(texts))
        tokens = B0.tokenise(list(texts))
        return [
            {
                "keyword": self.sources["keyword"].search_tokens(tok, self.depth),
                "tariff_embed": self.sources["tariff_embed"].search(q, self.depth),
                "rulings": self.sources["rulings"].search(q, self.depth),
            }
            for tok, q in zip(tokens, queries, strict=True)
        ]

    def fuse(self, lists: dict[str, list[Hit]], k: int | None = None,
             weights: dict[str, float] | None = None, k_rrf: float | None = None) -> Shortlist:  # fmt: skip
        weights = weights or self.config["weights"]
        k_rrf = self.config["k_rrf"] if k_rrf is None else k_rrf
        k = int(k or self.config["k"])
        fused = rrf({s: [h.cn8 for h in hits] for s, hits in lists.items()}, weights, k_rrf)[:k]
        by_source = {s: {h.cn8: h for h in hits} for s, hits in lists.items()}
        records = []
        for cn8, score in fused:
            m = self.meta.get(cn8, {"hs6": cn8[:6], "heading": cn8[:4], "chapter": cn8[:2],
                                    "path_text": ""})  # fmt: skip
            per = {s: ({"rank": h.rank, "score": h.score} if (h := by_source[s].get(cn8)) else None)
                   for s in lists}  # fmt: skip
            hit = by_source.get("rulings", {}).get(cn8)
            records.append(CandidateRecord(cn8=cn8, hs6=m["hs6"], heading=m["heading"],
                                           chapter=m["chapter"], path_text=m["path_text"],
                                           fused=score, sources=per,
                                           rulings_support=list(hit.support) if hit else []))  # fmt: skip
        return Shortlist(records, hs6_view([r.cn8 for r in records]))

    def retrieve(self, text: str, k: int | None = None) -> Shortlist:
        return self.fuse(self.source_lists([text])[0], k)

    def retrieve_many(self, texts: Sequence[str], k: int | None = None) -> list[Shortlist]:
        return [self.fuse(lists, k) for lists in self.source_lists(texts)]
