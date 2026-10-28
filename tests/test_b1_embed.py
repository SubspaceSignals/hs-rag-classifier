import hashlib
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hs_rag_classifier.arms import b1_embed as B1
from hs_rag_classifier.arms import b1ex_embed as BX
from hs_rag_classifier.arms import embed as E
from hs_rag_classifier.data.nomenclature import build_codes, read_commodities_csv
from hs_rag_classifier.eval.items import EvalItem
from hs_rag_classifier.guards import ForbiddenInputError

FIXTURES = Path(__file__).parent / "fixtures"
DIM = 64


def fake_backend(texts):
    """Hashed bag of words: texts sharing words get similar vectors. No model needed."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for n, t in enumerate(texts):
        for w in re.findall(r"[a-z]{3,}", t.lower()):
            out[n, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return out


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return fake_backend(texts)


def encoder(model="qwen3-embedding-0.6b", backend=fake_backend):
    return E.Encoder(E.model_config(model), backend=backend)


def item(item_id, text):
    return EvalItem(item_id=item_id, text=text, label_code="82119100", label_depth=8,
                    cluster_id=item_id, chapter="82", dataset="d", split="val")  # fmt: skip


@pytest.fixture()
def root(tmp_path):
    """A data root with the tariff fixture as codes.parquet and a tiny rulings split."""
    codes = build_codes(read_commodities_csv(FIXTURES / "tariff" / "commodities.csv"))
    tariff = tmp_path / "processed" / "tariff" / B1.NOMENCLATURE_VERSION
    tariff.mkdir(parents=True)
    codes.to_parquet(tariff / "codes.parquet", index=False)
    rulings = tmp_path / BX.TRAIN_REL.parent
    rulings.mkdir(parents=True)
    train = pd.DataFrame({
        "ref": ["1", "2", "3", "4"],
        "description": ["stainless steel table knife", "steel table knife with fixed blade",
                        "live horse for breeding", "pure bred breeding horse"],
        "keywords": [["KNIFE"], ["KNIFE"], ["HORSE"], ["HORSE"]],
        "code10": ["8211910000", "8211910000", "0101210000", "0101210000"],
        "cn8": ["82119100", "82119100", "01012100", "01012100"],
        "label_depth": [10, 8, 10, 10],
    })  # fmt: skip
    for name in ("train", "val", "test"):
        train.to_parquet(rulings / f"{name}.parquet", index=False)
    pd.DataFrame({"ref": ["1"], "justification": ["x"]}).to_parquet(
        rulings / "justifications.parquet", index=False
    )
    return tmp_path


def test_query_prefix_applied_to_queries_only_for_qwen3():
    rec = Recorder()
    enc = encoder("qwen3-embedding-0.6b", rec)
    enc.queries(["steel knife"])
    enc.documents(["Chapter 82 > knives"])
    assert rec.calls == [[E.QWEN3_INSTRUCTION + "steel knife"], ["Chapter 82 > knives"]]
    assert E.QWEN3_INSTRUCTION.endswith("\nQuery: ")
    rec = Recorder()
    bge = encoder("bge-m3", rec)
    bge.queries(["steel knife"])
    assert rec.calls == [["steel knife"]]
    v = enc.queries(["steel knife", "horse"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)  # normalised


def test_models_are_pinned_fp16_512():
    for name in E.MODELS:
        cfg = E.model_config(name)
        assert re.fullmatch(r"[0-9a-f]{40}", cfg["revision"])
        assert cfg["dtype"] == "float16" and cfg["max_length"] == 512 and cfg["normalise"]


def test_b1_groups_to_cn8_by_max_line_score():
    code10 = ["0101210000", "0101290000", "0102291010", "0102291090", "8211910000"]
    index = B1.B1Index(np.eye(5, dtype=np.float32), code10)
    assert list(index.cn8) == ["01012100", "01012900", "01022910", "82119100"]
    scores = np.array([0.1, 0.3, 0.2, 0.25, 0.0], dtype=np.float32)
    ranked = index.rank(scores, 3)
    assert [c for c, _ in ranked] == ["0101290000", "0102291090", "0101210000"]
    assert ranked[1][1] == pytest.approx(0.25)  # max of the cn8's lines, not the sum


def test_b1_builder_calls_guard_and_refuses_raw_and_atoz_paths(root, monkeypatch):
    calls = []
    real = B1.check_retrieval_inputs

    def spy(paths, r):
        calls.append([Path(p) for p in paths])
        return real(paths, r)

    monkeypatch.setattr(B1, "check_retrieval_inputs", spy)
    codes = root / "processed" / "tariff" / B1.NOMENCLATURE_VERSION / "codes.parquet"
    B1.build_index(B1.default_config(), encoder(), root=root, cache=False)
    assert calls == [[codes]]
    for leaked in (root / "raw" / "atoz" / "codes.parquet",
                   root / "processed" / "atoz" / "v1" / "codes.parquet"):  # fmt: skip
        leaked.parent.mkdir(parents=True, exist_ok=True)
        leaked.write_bytes(codes.read_bytes())
        with pytest.raises(ForbiddenInputError):
            B1.build_index(B1.default_config(), encoder(), root=root, codes_path=leaked,
                           cache=False)  # fmt: skip


def test_b1_arm_on_fixture_and_cache(root):
    config = B1.default_config(temperature=0.05)
    arm = B1.B1EmbedArm(config, root=root, encoder=encoder())
    (pred,) = arm.predict([item("1", "table knives with fixed blades")])
    q = arm.encoder.queries(["table knives with fixed blades"])[0]
    line_scores = arm.index.embeddings @ q
    best = int(np.argmax(line_scores))
    assert pred.candidates[0].code == arm.index.code10[best] and not pred.abstain
    assert pred.candidates[0].code.startswith("8211")
    assert pred.candidates[0].score == pytest.approx(float(line_scores[best]))
    assert len(pred.candidates) == len(arm.index.cn8)
    assert pred.confidence == pytest.approx(
        B1.softmax_top1([c.score for c in pred.candidates], 0.05)
    )
    key = B1.index_key(config, B1.sha256(root / "processed" / "tariff" /
                                         B1.NOMENCLATURE_VERSION / "codes.parquet"))  # fmt: skip
    cache = root / "indexes" / "b1" / "qwen3-embedding-0.6b" / key
    assert (cache / "embeddings.npy").exists()
    rec = Recorder()
    cached = B1.B1EmbedArm(config, root=root, encoder=encoder(backend=rec))
    assert cached.predict([item("1", "table knives with fixed blades")])[0] == pred
    assert len(rec.calls) == 1  # the query only: documents came from the cache
    assert B1.index_key(B1.default_config(temperature=9.0), "x") == B1.index_key(config, "x")
    assert B1.index_key(B1.default_config("bge-m3"), "x") != B1.index_key(config, "x")


def test_b1ex_vote_scoring_and_ten_digit_choice():
    sims = [0.9, 0.8, 0.7, 0.6, 0.5]
    cn8 = ["11111111", "22222222", "22222222", "33333333", "11111111"]
    code10 = ["1111111110", "2222222220", "2222222299", "3333333330", "1111111190"]
    depth = [8, 10, 10, 10, 10]
    # k = 1: nearest neighbour wins; the rest of the pool fills the list by pool score
    assert vote_codes(sims, cn8, code10, depth, 1) == ["1111111190", "2222222220", "3333333330"]
    # k = 3: cn8 2 sums 0.8 + 0.7 = 1.5 > 0.9; its code10 is its most similar depth-10 row
    ranked = BX.vote(sims, cn8, code10, depth, 3)
    assert ranked[0] == ("2222222220", pytest.approx(1.5))
    assert ranked[1] == ("1111111190", pytest.approx(0.9))
    assert ranked[2] == ("3333333330", 0.0)  # only in the pool: vote 0
    # a cn8 whose neighbours are all depth 8 stays at 8 digits
    assert BX.vote([0.9], ["44444444"], ["4444444410"], [8], 1) == [("44444444", 0.9)]
    # cap on distinct cn8s
    many = [f"{i:08d}" for i in range(60)]
    assert len(BX.vote([1.0] * 60, many, [c + "00" for c in many], [10] * 60, 60)) == 50


def vote_codes(*args):
    return [c for c, _ in BX.vote(*args)]


def test_b1ex_builder_accepts_train_only(root, monkeypatch):
    calls = []
    real = BX.check_retrieval_inputs

    def spy(paths, r):
        calls.append([Path(p) for p in paths])
        return real(paths, r)

    monkeypatch.setattr(BX, "check_retrieval_inputs", spy)
    BX.build_index(BX.default_config(), encoder(), root=root, cache=False)
    assert calls == [[root / BX.TRAIN_REL]]
    for name in ("val.parquet", "test.parquet", "justifications.parquet"):
        with pytest.raises(ForbiddenInputError):
            BX.build_index(BX.default_config(), encoder(), root=root, cache=False,
                           train_path=root / BX.TRAIN_REL.parent / name)  # fmt: skip
    leaked = root / "processed" / "atoz" / "train.parquet"
    leaked.parent.mkdir(parents=True)
    leaked.write_bytes((root / BX.TRAIN_REL).read_bytes())
    with pytest.raises(ForbiddenInputError):
        BX.build_index(BX.default_config(), encoder(), root=root, cache=False, train_path=leaked)


def test_b1ex_reads_description_only(root):
    rec = Recorder()
    BX.build_index(BX.default_config(), encoder(backend=rec), root=root, cache=False)
    assert rec.calls == [["stainless steel table knife", "steel table knife with fixed blade",
                          "live horse for breeding", "pure bred breeding horse"]]  # fmt: skip
    assert "keywords" not in BX.TRAIN_COLUMNS and "justification" not in BX.TRAIN_COLUMNS


def test_b1ex_arm_predicts_from_neighbours(root):
    config = BX.default_config(k=2, temperature=0.1)
    arm = BX.B1exEmbedArm(config, root=root, encoder=encoder())
    knife, horse = arm.predict([item("1", "steel table knife"), item("2", "breeding horse")])
    assert knife.candidates[0].code == "8211910000"
    assert horse.candidates[0].code == "0101210000"
    assert len(knife.candidates) == 2 and 0 < knife.confidence <= 1
    cache = root / "indexes" / "b1ex" / "qwen3-embedding-0.6b"
    assert len(list(cache.glob("*/embeddings.npy"))) == 1


def _gpu_model_available(hf_id: str) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_id.replace('/', '--')}"
    return torch.cuda.is_available() and hub.exists()


@pytest.mark.skipif(not _gpu_model_available("Qwen/Qwen3-Embedding-0.6B"),
                    reason="needs the GPU and the cached Qwen3 embedding model")  # fmt: skip
def test_real_qwen3_encoder_on_gpu():
    enc = E.Encoder(E.model_config("qwen3-embedding-0.6b"))
    q = enc.queries(["stainless steel kitchen knife"])
    d = enc.documents(["Table knives having fixed blades", "Live horses for breeding"])
    assert q.shape[1] == d.shape[1] and np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-3)
    sims = q @ d.T
    assert sims[0, 0] > sims[0, 1]
    E.release(enc)


def test_package_imports_without_the_ml_group():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'sentence_transformers'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import hs_rag_classifier.arms.b1_embed, hs_rag_classifier.arms.b1ex_embed\n"
        "import hs_rag_classifier.arms.embed_tune, hs_rag_classifier.eval.runner\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
