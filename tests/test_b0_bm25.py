import math
from pathlib import Path

import numpy as np
import pytest

from hs_rag_classifier.arms import b0_bm25 as B0
from hs_rag_classifier.data.nomenclature import build_codes, read_commodities_csv
from hs_rag_classifier.eval.items import EvalItem
from hs_rag_classifier.guards import ForbiddenInputError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def root(tmp_path):
    """A data root holding the tariff fixture as codes.parquet."""
    codes = build_codes(read_commodities_csv(FIXTURES / "tariff" / "commodities.csv"))
    out = tmp_path / "processed" / "tariff" / B0.NOMENCLATURE_VERSION
    out.mkdir(parents=True)
    codes.to_parquet(out / "codes.parquet", index=False)
    return tmp_path


def item(item_id, text):
    return EvalItem(item_id=item_id, text=text, label_code="82119100", label_depth=8,
                    cluster_id=item_id, chapter="82", dataset="d", split="val")  # fmt: skip


def test_rank_groups_to_cn8_by_max_line_score_and_picks_best_line():
    code10 = ["0101210000", "0101290000", "0102291010", "0102291090", "8211910000"]
    index = B0.B0Index(retriever=None, code10=code10)
    assert list(index.cn8) == ["01012100", "01012900", "01022910", "82119100"]
    scores = np.array([1.0, 3.0, 2.0, 2.5, 0.0])
    assert index.rank(scores, 3) == [("0101290000", 3.0), ("0102291090", 2.5),
                                     ("0101210000", 1.0)]  # fmt: skip
    # a cn8 scores its best line, not the sum: 01022910 (2.0 + 2.5) stays below 01012900 (3.0)
    ties = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # ties: lower cn8, then lower line, first
    assert index.rank(ties, 2) == [("0101210000", 1.0), ("0101290000", 1.0)]
    assert index.rank(ties, 10)[2] == ("0102291010", 1.0)


def test_softmax_confidence_and_temperature():
    assert B0.softmax_top1([2.0, 1.0, 0.0], 1.0) == pytest.approx(
        math.exp(2) / (math.exp(2) + math.exp(1) + 1)
    )
    assert B0.softmax_top1([2.0, 1.0, 0.0], 1e6) == pytest.approx(1 / 3, abs=1e-5)  # hot: uniform
    assert B0.softmax_top1([2.0, 1.0, 0.0], 1e-3) == pytest.approx(1.0)  # cold: argmax
    assert B0.softmax_top1([5.0] * 11, 1.0) == pytest.approx(0.1)  # only the top 10 count
    assert B0.softmax_top1([], 1.0) == 0.0


def test_builder_calls_guard_and_refuses_test_only_paths(root, monkeypatch):
    calls = []
    real = B0.check_retrieval_inputs

    def spy(paths, r):
        calls.append(([Path(p) for p in paths], Path(r)))
        return real(paths, r)

    monkeypatch.setattr(B0, "check_retrieval_inputs", spy)
    codes = root / "processed" / "tariff" / B0.NOMENCLATURE_VERSION / "codes.parquet"
    B0.build_index(B0.default_config(), root=root, cache=False)
    assert calls == [([codes], root)]
    leaked = root / "raw" / "atoz" / "codes.parquet"
    leaked.parent.mkdir(parents=True)
    leaked.write_bytes(codes.read_bytes())
    with pytest.raises(ForbiddenInputError):
        B0.build_index(B0.default_config(), root=root, codes_path=leaked, cache=False)


def test_arm_on_fixture_index_and_cache(root):
    config = B0.default_config(temperature=0.5)
    arm = B0.B0BM25Arm(config, root=root)
    preds = arm.predict([item("1", "table knives with fixed blades"), item("2", "of the")])
    top = preds[0]
    assert top.candidates[0].code == "8211910000" and not top.abstain
    assert len(top.candidates) == len(arm.index.cn8)  # fewer cn8s than the cap of 50
    assert all(len(c.code) == 10 for c in top.candidates)
    assert 0 < top.confidence <= 1
    scores = [c.score for c in top.candidates]
    assert top.confidence == pytest.approx(B0.softmax_top1(scores, 0.5))
    assert preds[1].candidates[0].score == 0.0  # stopwords only: nothing matches
    key = B0.index_key(config, B0.sha256(root / "processed" / "tariff" /
                                         B0.NOMENCLATURE_VERSION / "codes.parquet"))  # fmt: skip
    assert (root / "indexes" / "b0" / key / "code10.json").exists()
    cached = B0.B0BM25Arm(config, root=root)  # loads the saved index
    assert cached.predict([item("1", "table knives with fixed blades")])[0] == top
    assert B0.index_key(B0.default_config(temperature=9.0), "x") == B0.index_key(config, "x")
    assert B0.index_key(B0.default_config(k1=0.9), "x") != B0.index_key(config, "x")


def test_temperature_fit_and_grid_selection():
    rng = np.random.default_rng(0)
    top_scores = [sorted(rng.uniform(0, 10, 10), reverse=True) for _ in range(200)]
    gap = np.array([s[0] - s[1] for s in top_scores])
    correct = (gap > np.median(gap)).astype(float)  # confident items tend to be right
    t = B0.fit_temperature(top_scores, correct)
    nll = B0.nll_top1(t, top_scores, correct)
    assert nll <= B0.nll_top1(1.0, top_scores, correct) + 1e-9
    assert nll <= B0.nll_top1(100.0, top_scores, correct) + 1e-9
    grid = [{"k1": 1, "top1_8": 0.3, "recall_10": 0.5}, {"k1": 2, "top1_8": 0.3, "recall_10": 0.6},
            {"k1": 3, "top1_8": 0.2, "recall_10": 0.9}]  # fmt: skip
    assert B0.select(grid)["k1"] == 2  # tie on top-1 at 8 broken by recall at 10
