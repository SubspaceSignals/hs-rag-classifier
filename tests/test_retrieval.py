"""Hybrid retrieval: RRF, weights, the HS6 view, the in-memory store, the K rule, guards, and a
retriever on fixture data with the fake encoder from test_b1_embed (no model downloads)."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hs_rag_classifier.arms import b0_bm25 as B0
from hs_rag_classifier.arms import b1_embed as B1
from hs_rag_classifier.arms import b1ex_embed as BX
from hs_rag_classifier.arms import embed as E
from hs_rag_classifier.data.nomenclature import build_codes, build_codes8, read_commodities_csv
from hs_rag_classifier.guards import ForbiddenInputError
from hs_rag_classifier.retrieval import hybrid as H
from hs_rag_classifier.retrieval.fusion import (
    hs6_view,
    normalise_weights,
    rrf,
    select_k,
    weight_grid,
)
from hs_rag_classifier.retrieval.store import InMemoryStore, VectorStore

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]


def fake_backend(texts):
    """Hashed bag of words (as in test_b1_embed): shared words give similar vectors."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for n, t in enumerate(texts):
        for w in re.findall(r"[a-z]{3,}", t.lower()):
            out[n, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
    return out


# --- Fusion ----------------------------------------------------------------------------------


def test_rrf_arithmetic_with_a_cn8_absent_from_one_source():
    rankings = {"a": ["X", "Y", "Z"], "b": ["Y", "W"]}  # Z and X absent from b, W from a
    fused = dict(rrf(rankings, {"a": 1, "b": 1}, k_rrf=10))
    assert fused["X"] == pytest.approx(0.5 / 11)
    assert fused["Y"] == pytest.approx(0.5 / 12 + 0.5 / 11)
    assert fused["Z"] == pytest.approx(0.5 / 13)
    assert fused["W"] == pytest.approx(0.5 / 12)
    order = [c for c, _ in rrf(rankings, {"a": 1, "b": 1}, k_rrf=10)]
    assert order == ["Y", "X", "W", "Z"]
    weighted = dict(rrf(rankings, {"a": 3, "b": 1}, k_rrf=0))
    assert weighted["X"] == pytest.approx(0.75 / 1) and weighted["W"] == pytest.approx(0.25 / 2)
    tie = [c for c, _ in rrf({"a": ["P"], "b": ["O"]}, {"a": 1, "b": 1}, 60)]
    assert tie == ["O", "P"]  # equal score and best rank: lower cn8 first


def test_weight_normalisation_and_grid_dedup():
    assert normalise_weights({"a": 2, "b": 2, "c": 4}) == {"a": 0.25, "b": 0.25, "c": 0.5}
    with pytest.raises(ValueError):
        normalise_weights({"a": 0, "b": 0})
    grid = weight_grid(("a", "b", "c"), (0.5, 1, 2))
    assert len(grid) == 19  # 27 combinations, 8 duplicate ratios (e.g. 1:1:1 = 2:2:2)
    keys = {tuple(round(w[s], 9) for s in "abc") for w in grid}
    assert len(keys) == len(grid)
    assert all(sum(w.values()) == pytest.approx(1) for w in grid)
    assert grid[0] == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_hs6_view_keeps_order_and_dedups():
    assert hs6_view(["82119100", "01012100", "82119200", "01012900", "82119100"]) == [
        "821191", "010121", "821192", "010129"]  # fmt: skip
    assert hs6_view([]) == []


def test_select_k_knee_and_budget_cases():
    ks = (10, 20, 30, 40, 50)
    recall = {10: 0.70, 20: 0.80, 30: 0.85, 40: 0.855, 50: 0.86}
    cheap = {k: k * 10 for k in ks}
    assert select_k(ks, recall, cheap, 2000) == {"k": 30, "knee_k": 30, "budget_binds": False}
    tight = {k: k * 90 for k in ks}  # only K = 10 and 20 fit 2,000 tokens
    assert select_k(ks, recall, tight, 2000) == {"k": 20, "knee_k": 30, "budget_binds": True}
    steady = {k: 0.5 + k / 50 for k in ks}  # never flattens: the knee is the last K
    assert select_k(ks, steady, cheap, 2000)["knee_k"] == 50
    assert select_k(ks, recall, {k: 5000 for k in ks}, 2000)["k"] is None  # nothing fits


# --- Store -----------------------------------------------------------------------------------


def test_in_memory_store_search_and_round_trip(tmp_path):
    store = InMemoryStore()
    assert isinstance(store, VectorStore)
    vecs = np.eye(3, dtype=np.float32)
    store.add(["a", "b", "c"], vecs)
    store.add(["d"], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))  # ties "a"
    ids, scores = store.search(np.array([[1.0, 0.5, 0.0]]), 3)
    assert ids.tolist() == [["a", "d", "b"]]  # tie keeps insertion order
    assert scores[0].tolist() == pytest.approx([1.0, 1.0, 0.5])
    ids, _ = store.search(np.array([[0, 0, 1.0], [0, 1.0, 0]]), 1)
    assert ids.tolist() == [["c"], ["b"]]
    assert store.search(np.array([[1.0, 0, 0]]), 10)[0].shape == (1, 4)  # k capped at size
    with pytest.raises(ValueError):
        store.add(["a"], np.ones((1, 3)))  # duplicate id
    with pytest.raises(ValueError):
        store.add(["e"], np.ones((1, 2)))  # wrong dimension
    store.save(tmp_path / "s")
    loaded = InMemoryStore.load(tmp_path / "s")
    assert len(loaded) == 4 and loaded.ids.tolist() == store.ids.tolist()
    q = np.array([[0.2, 0.3, 0.9]])
    assert loaded.search(q, 4)[0].tolist() == store.search(q, 4)[0].tolist()


# --- Retriever on fixture data ---------------------------------------------------------------


@pytest.fixture()
def env(tmp_path):
    """A data root (tariff fixture, codes8, tiny train split) and a repo with the locked configs."""
    root, repo = tmp_path / "data", tmp_path / "repo"
    codes = build_codes(read_commodities_csv(FIXTURES / "tariff" / "commodities.csv"))
    tariff = root / "processed" / "tariff" / B0.NOMENCLATURE_VERSION
    tariff.mkdir(parents=True)
    codes.to_parquet(tariff / "codes.parquet", index=False)
    build_codes8(codes).to_parquet(tariff / "codes8.parquet", index=False)
    rulings = root / BX.TRAIN_REL.parent
    rulings.mkdir(parents=True)
    pd.DataFrame({
        "ref": ["1", "2", "3"],
        "description": ["stainless steel table knife", "steel knife fixed blade", "live horse"],
        "code10": ["8211910000", "8211910000", "0101210000"],
        "cn8": ["82119100", "82119100", "01012100"],
        "label_depth": [10, 10, 10],
    }).to_parquet(rulings / "train.parquet", index=False)  # fmt: skip
    (repo / "configs").mkdir(parents=True)
    for p in H.SOURCE_CONFIGS.values():
        shutil.copy(REPO / p, repo / p)
    return root, repo


def encoder():
    return E.Encoder(B1.load_config(REPO / H.SOURCE_CONFIGS["tariff_embed"]), backend=fake_backend)


def test_every_builder_runs_the_guards(env, monkeypatch):
    root, repo = env
    calls = []
    for mod in (B0, B1, BX, H):
        real = mod.check_retrieval_inputs
        monkeypatch.setattr(mod, "check_retrieval_inputs",
                            lambda p, r, m=mod, f=real: calls.append(m.__name__) or f(p, r))  # fmt: skip
    real_train = BX.check_train_split_only
    train_calls = []
    monkeypatch.setattr(BX, "check_train_split_only",
                        lambda p, t: train_calls.append(Path(p)) or real_train(p, t))  # fmt: skip
    H.build_sources(root, repo, encoder())
    H.load_cn8_meta(root)
    assert sorted(set(calls)) == sorted(m.__name__ for m in (B0, B1, BX, H))
    assert train_calls == [root / BX.TRAIN_REL]
    with pytest.raises(ForbiddenInputError):  # the rulings index refuses anything but train
        BX.build_index(BX.load_config(repo / H.SOURCE_CONFIGS["rulings"]), encoder(), root,
                       train_path=root / BX.TRAIN_REL.parent / "val.parquet", cache=False)  # fmt: skip
    leaked = root / "processed" / "atoz" / "codes8.parquet"
    leaked.parent.mkdir(parents=True)
    with pytest.raises(ForbiddenInputError):
        H.check_retrieval_inputs([leaked], root)


def test_retriever_records_hs6_view_and_rulings_support(env):
    root, repo = env
    sources, enc = H.build_sources(root, repo, encoder())
    config = {**H.default_config(k=5), "weights": {"keyword": 1, "tariff_embed": 1, "rulings": 2}}
    r = H.HybridRetriever(config, root, sources=sources, encoder=enc, meta=H.load_cn8_meta(root))
    lists = r.source_lists(["steel table knife"])[0]
    assert set(lists) == set(H.SOURCES)
    assert [h.cn8 for h in lists["rulings"]] == ["82119100", "01012100"]  # vote, then pool
    assert all(h.score > 0 for h in lists["keyword"])  # no-match lines are left out
    short = r.fuse(lists)
    fused = rrf({s: [h.cn8 for h in v] for s, v in lists.items()}, config["weights"], 60)
    assert short.cn8 == [c for c, _ in fused][:5]
    top = next(c for c in short.candidates if c.cn8 == "82119100")
    assert top.hs6 == "821191" and top.heading == "8211" and top.chapter == "82"
    assert "knives" in top.path_text.lower()
    assert top.sources["rulings"]["rank"] == 1
    assert [ref for ref, _ in top.rulings_support] == ["1", "2"]
    assert all(0 < s <= 1 for _, s in top.rulings_support)
    absent = [c for c in short.candidates if c.sources["rulings"] is None]
    assert absent and all(c.rulings_support == [] for c in absent)
    assert short.hs6 == hs6_view(short.cn8)
    assert json.dumps(top.to_dict())  # serialisable for the pipeline
    assert top.fused == pytest.approx(dict(fused)[top.cn8])


def test_tariff_source_matches_b1_ranking(env):
    root, repo = env
    sources, enc = H.build_sources(root, repo, encoder())
    cfg = B1.load_config(repo / H.SOURCE_CONFIGS["tariff_embed"])
    arm = B1.B1EmbedArm(cfg, root=root, encoder=enc)
    q = enc.queries(["live horses for breeding"])[0]
    expected = [c[:8] for c, _ in arm.index.rank(arm.index.line_scores(q[None, :])[0], 100)]
    assert [h.cn8 for h in sources["tariff_embed"].search(q)] == expected


def test_load_config_refuses_changed_source_config(env, tmp_path):
    _, repo = env
    cfg = {**H.default_config(), "sources": H.source_config_entries(repo)}
    path = tmp_path / "hybrid.json"
    path.write_text(json.dumps(cfg))
    assert H.load_config(path, repo)["k_rrf"] == 60
    (repo / H.SOURCE_CONFIGS["keyword"]).write_text("{}")
    with pytest.raises(RuntimeError, match="changed"):
        H.load_config(path, repo)
