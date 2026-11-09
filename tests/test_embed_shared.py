"""One bge-m3 per process: components that ask for it share one backend, and the shared path
gives the same embeddings as a private copy (the real-model test skips without the model)."""

import numpy as np
import pytest

from hs_rag_classifier.arms import embed as E


@pytest.fixture
def fake_loader(monkeypatch):
    monkeypatch.setattr(E, "_SHARED", {})
    calls = []

    def loader(config, batch_size=E.ENCODE_BATCH):
        calls.append(config["model"])

        def encode(texts):
            return np.ones((len(texts), 4), np.float32)

        encode.model = object()
        return encode

    monkeypatch.setattr(E, "sentence_transformer_backend", loader)
    return calls


def test_two_components_get_the_same_instance(fake_loader):
    cfg = E.model_config("bge-m3")
    retriever_side, b1ex_side = E.Encoder(dict(cfg)), E.Encoder(dict(cfg))
    assert retriever_side.backend is b1ex_side.backend and fake_loader == ["bge-m3"]
    other = E.Encoder(E.model_config("qwen3-embedding-0.6b"))
    assert other.backend is not retriever_side.backend and len(fake_loader) == 2
    assert E.backend_key(cfg) != E.backend_key({**cfg, "dtype": "float32"})


def test_hybrid_and_b1ex_configs_share_one_key():
    from hs_rag_classifier.arms import a3_pipeline as A3
    from hs_rag_classifier.arms import b1_embed as B1
    from hs_rag_classifier.arms import b1ex_embed as BX

    b1, bx = B1.load_config(), BX.load_config()
    assert E.backend_key(b1) == E.backend_key(bx) == E.backend_key(A3.load_config()["b1ex"])


def test_release_drops_the_shared_copy(fake_loader):
    cfg = E.model_config("bge-m3")
    enc = E.Encoder(cfg)
    E.release(enc)
    assert E._SHARED == {} and E.Encoder(cfg).backend is not None and len(fake_loader) == 2


def _model_ready(cfg):
    try:
        import torch
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache(cfg["hf_id"], "config.json", revision=cfg["revision"])
    return torch.cuda.is_available() and isinstance(hit, str)


@pytest.mark.skipif(
    not _model_ready(E.model_config("bge-m3")), reason="bge-m3 files or CUDA absent"
)
def test_shared_embeddings_equal_the_old_path(monkeypatch):
    monkeypatch.setattr(E, "_SHARED", {})
    cfg = E.model_config("bge-m3")
    texts = ["Stainless steel table knives with fixed blades", "Live pure-bred breeding horses",
             "Waste, parings and scrap of polyethylene", "x"]  # fmt: skip
    old = E.Encoder(cfg, backend=E.sentence_transformer_backend(cfg))  # a private copy, as before
    shared = E.Encoder(cfg)
    try:
        assert shared.backend is E.Encoder(cfg).backend and shared.backend is not old.backend
        for fn in ("queries", "documents"):
            a, b = getattr(old, fn)(texts), getattr(shared, fn)(texts)
            assert a.dtype == b.dtype and np.array_equal(a, b)
    finally:
        E.release(old)
        E.release(shared)
