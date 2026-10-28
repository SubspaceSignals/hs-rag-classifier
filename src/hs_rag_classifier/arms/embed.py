"""Shared dense-embedding encoder for B1 and B1-ex.

Models (sentence-transformers, dense vectors only, pinned to a Hugging Face commit):
    qwen3-embedding-0.6b  Qwen/Qwen3-Embedding-0.6B; queries carry QWEN3_INSTRUCTION, documents
                          carry no prefix
    bge-m3                BAAI/bge-m3; no prefix on either side
Both run in fp16 on the GPU with a max sequence length of 512 tokens. Embeddings are
L2-normalised, so similarity is the dot product (cosine).

torch and sentence-transformers sit in the "ml" dependency group and are imported only when a
real model is loaded; the rest of this module (and every arm's ranking code) imports without
them. Tests pass a fake backend: any callable from a list of texts to an (n, dim) array.

Embedding caches: <data root>/indexes/<arm>/<model>/<config hash>/ holds embeddings.npy (float32,
row order as meta.json lists it) and meta.json (settings, input files, encode seconds, VRAM).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

QWEN3_INSTRUCTION = (
    "Instruct: Given a product description, retrieve the UK tariff commodity line that "
    "classifies it\nQuery: "
)
DTYPE = "float16"
MAX_LENGTH = 512
ENCODE_BATCH = 64

Backend = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    revision: str
    query_prefix: str = ""


MODELS = {
    m.name: m
    for m in (
        ModelSpec(
            "qwen3-embedding-0.6b",
            "Qwen/Qwen3-Embedding-0.6B",
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            QWEN3_INSTRUCTION,
        ),
        ModelSpec("bge-m3", "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    )
}


def model_config(name: str) -> dict[str, Any]:
    """The encoder settings an arm config records (and its index key hashes)."""
    spec = MODELS[name]
    return {
        "model": spec.name,
        "hf_id": spec.hf_id,
        "revision": spec.revision,
        "dtype": DTYPE,
        "max_length": MAX_LENGTH,
        "query_instruction": spec.query_prefix,
        "normalise": True,
        "similarity": "dot (cosine)",
    }


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def sentence_transformer_backend(config: dict[str, Any], batch_size: int = ENCODE_BATCH) -> Backend:
    """Load the pinned model on the GPU (fp16, max length 512) as an encode callable."""
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; the embedding arms run on the GPU")
    model = SentenceTransformer(
        config["hf_id"],
        revision=config["revision"],
        device="cuda",
        model_kwargs={"dtype": getattr(torch, config["dtype"])},
    )
    model.max_seq_length = int(config["max_length"])

    def encode(texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)  # fmt: skip

    encode.model = model  # kept so the caller can release it
    return encode


class Encoder:
    """Queries get the model's instruction prefix; documents never do. Output is normalised."""

    def __init__(self, config: dict[str, Any], backend: Backend | None = None):
        self.config = config
        self.prefix = config["query_instruction"]
        self.backend = backend or sentence_transformer_backend(config)

    def queries(self, texts: Sequence[str]) -> np.ndarray:
        return normalise(self.backend([self.prefix + t for t in texts]))

    def documents(self, texts: Sequence[str]) -> np.ndarray:
        return normalise(self.backend(list(texts)))


def release(encoder: Encoder | None) -> None:
    """Drop a real model from the GPU (no-op for a fake backend)."""
    model = getattr(getattr(encoder, "backend", None), "model", None)
    if model is None:
        return
    del encoder.backend.model
    encoder.backend = None
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def gpu_memory() -> dict[str, float] | None:
    """Peak torch allocation in GB since the last reset, or None without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}


def cached_embeddings(
    cache_dir: Path, texts: Sequence[str], encoder: Encoder, meta: dict[str, Any], cache: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """Document embeddings from cache_dir, or encode them (and save when cache is set)."""
    emb_path, meta_path = cache_dir / "embeddings.npy", cache_dir / "meta.json"
    if cache and emb_path.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        emb = np.load(emb_path)
        if emb.shape[0] == len(texts) == saved["n_documents"]:
            return emb, saved
    t0 = time.perf_counter()
    emb = encoder.documents(texts)
    meta = {**meta, "n_documents": len(texts), "dim": int(emb.shape[1]),
            "encode_seconds": round(time.perf_counter() - t0, 3), "gpu": gpu_memory()}  # fmt: skip
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, emb.astype(np.float32))
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return emb, meta
