"""Vector store interface and the in-memory implementation.

VectorStore is what the retrieval sources depend on: add vectors under string ids, search the
top k by dot product (vectors are normalised, so this is cosine), and persist or load. A
QdrantStore (Phase 4) implements the same methods; nothing else changes.

InMemoryStore: exact search in numpy. Results are in descending score; ties keep insertion
order (stable sort), so a store filled in code order breaks ties by the lower code.
Persisted as <dir>/vectors.npy (float32) and <dir>/ids.json.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorStore(Protocol):
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(ids, scores), each (n_queries, min(k, len)), most similar first."""
        ...

    def save(self, path: Path) -> None: ...

    def __len__(self) -> int: ...


class InMemoryStore:
    """Exact dot-product search over vectors held in memory."""

    def __init__(self, dim: int | None = None):
        self.dim = dim
        self.ids = np.array([], dtype=str)
        self.vectors = np.zeros((0, dim or 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(ids) != len(vectors):
            raise ValueError("ids and a 2-d array of vectors must have the same length")
        if self.dim is None:
            self.dim = vectors.shape[1]
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vectors have dimension {vectors.shape[1]}, store has {self.dim}")
        new = np.asarray(list(ids), dtype=str)
        if len(set(new)) != len(new) or np.isin(new, self.ids).any():
            raise ValueError("ids must be unique")
        self.ids = np.concatenate([self.ids, new])
        self.vectors = np.concatenate([self.vectors, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        scores = queries @ self.vectors.T
        top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return self.ids[top], np.take_along_axis(scores, top, axis=1)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "ids.json").write_text(json.dumps(self.ids.tolist()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> InMemoryStore:
        path = Path(path)
        vectors = np.load(path / "vectors.npy")
        store = cls(vectors.shape[1])
        store.add(json.loads((path / "ids.json").read_text(encoding="utf-8")), vectors)
        return store
