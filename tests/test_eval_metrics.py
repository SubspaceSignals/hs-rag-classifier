import math

import numpy as np
import pandas as pd
import pytest

from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.arms import Candidate, OracleArm, Prediction, RandomArm, config_hash
from hs_rag_classifier.eval.items import EvalItem

VALID8 = {"01012100", "01022910", "01013000"}
VALID10 = {"0101210000", "0101290000", "0102291010", "0101300000"}


def item(item_id, label, depth, cluster=None, chapter="01"):
    return EvalItem(item_id=item_id, text="t", label_code=label, label_depth=depth,
                    cluster_id=cluster or item_id, chapter=chapter, dataset="d", split="val")  # fmt: skip


def pred(item_id, *codes, confidence=0.5, abstain=False):
    cands = [Candidate(code=c, score=1.0 - n / 10) for n, c in enumerate(codes)]
    return Prediction(item_id=item_id, candidates=cands, confidence=confidence, abstain=abstain)


@pytest.fixture()
def mixed():
    items = [
        item("A", "0101210000", 10),  # exact
        item("B", "01012100", 8),  # 8-digit label; prediction right to 4 digits only
        item("C", "0102291010", 10),  # valid cn8 predicted for a 10-digit label
        item("D", "0101300000", 10),  # invalid 10-digit code with the right prefix
    ]
    preds = [pred("A", "0101210000"), pred("B", "0101290000"), pred("C", "01022910"),
             pred("D", "0101300099")]  # fmt: skip
    return M.score(items, preds, VALID8, VALID10)


def test_hierarchical_accuracy_with_mixed_label_depth(mixed):
    acc = M.top1_accuracy(mixed)
    assert acc[2] == {"accuracy": 0.75, "n": 4}  # D is invalid: wrong even at 2 digits
    assert acc[4] == {"accuracy": 0.75, "n": 4}
    assert acc[6] == {"accuracy": 0.5, "n": 4}
    assert acc[8] == {"accuracy": 0.5, "n": 4}
    assert acc[10]["n"] == 3  # B (label_depth 8) is scored to 8 only
    assert acc[10]["accuracy"] == pytest.approx(1 / 3)  # C's 8-digit code is wrong at 10
    assert math.isnan(mixed.set_index("item_id").at["B", "correct_10"])
    assert M.invalid_rate(mixed) == 0.25


def test_recall_at_k_at_8_digits():
    label = "0101210000"
    codes = ["01013000", "0101290000", "0101210000"]
    assert M.recall_at(codes, label, 2, VALID8, VALID10) == 0.0
    assert M.recall_at(codes, label, 3, VALID8, VALID10) == 1.0
    assert M.recall_at(["01012100"], label, 1, VALID8, VALID10) == 1.0  # cn8 matches at 8
    assert M.recall_at(["0101210099"], label, 1, VALID8, VALID10) == 0.0  # invalid: wrong
    assert M.recall_at(["010121"], label, 1, VALID8, VALID10) == 0.0  # too short


def test_ece_equal_mass_bins():
    conf = np.array([0.4, 0.1, 0.3, 0.2])
    correct = np.array([1, 0, 1, 0])
    # bins by sorted confidence: {0.1, 0.2} acc 0 conf 0.15; {0.3, 0.4} acc 1 conf 0.35
    assert M.ece(conf, correct, n_bins=2) == pytest.approx(0.5 * 0.15 + 0.5 * 0.65)
    assert M.ece(np.full(4, 1.0), np.ones(4)) == 0.0
    bins = M.reliability(np.linspace(0, 1, 30), np.zeros(30))
    assert len(bins) == 15 and {b["n"] for b in bins} == {2}  # equal mass, not equal width
    skewed = M.reliability(np.r_[np.full(28, 0.99), [0.1, 0.2]], np.ones(30))
    assert [b["n"] for b in skewed] == [2] * 15


def test_coverage_with_abstains():
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.95])
    correct = np.array([1, 1, 0, 0, 0])
    abstain = np.array([False, False, False, False, True])  # confident, but abstains
    curve = {c["review_rate"]: c for c in M.coverage_curve(conf, correct, abstain,
                                                          [0.0, 0.4, 0.6, 1.0])}  # fmt: skip
    assert curve[0.0]["actual_review_rate"] == 0.2  # the abstain is always reviewed
    assert curve[0.0]["accuracy"] == 0.5
    assert curve[0.4]["accuracy"] == pytest.approx(2 / 3)  # abstain + least confident (0.6)
    assert curve[0.6]["accuracy"] == 1.0
    assert curve[1.0]["n_accepted"] == 0 and math.isnan(curve[1.0]["accuracy"])
    assert M.accuracy_at_review(conf, correct, abstain, 0.6) == 1.0


def test_cluster_bootstrap_is_deterministic_and_resamples_clusters():
    frame = pd.DataFrame({"cluster_id": ["a", "a", "a", "b"], "correct_8": [1.0, 1.0, 1.0, 0.0]})
    stats = {"m": lambda a: float(a["correct_8"].mean())}
    s1 = M.bootstrap_samples(frame, stats, n_resamples=200, seed=7)["m"]
    s2 = M.bootstrap_samples(frame, stats, n_resamples=200, seed=7)["m"]
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, M.bootstrap_samples(frame, stats, 200, seed=8)["m"])
    # whole clusters only: {a,a} 1.0, {a,b} 0.75, {b,b} 0.0; resampling rows would give 0.5 etc.
    assert set(np.round(s1, 6)) <= {1.0, 0.75, 0.0} and len(set(s1)) == 3
    ci = M.bootstrap_intervals(frame, stats, n_resamples=200, seed=7)["m"]
    assert ci["low"] <= 0.75 <= ci["high"]


def test_latency_and_per_chapter(mixed):
    lat = M.latency_stats([1.0, 2.0, 3.0, 4.0], total_seconds=10.0)
    assert lat == {"items_per_minute": 24.0, "latency_p50_s": 2.5,
                   "latency_p95_s": pytest.approx(3.85)}  # fmt: skip
    table = M.per_chapter(mixed)
    assert table.to_dict("records") == [{"chapter": "01", "n": 4, "top1_8": 0.5,
                                         "recall_10": 0.5}]  # fmt: skip


def test_predictions_must_cover_items_once(mixed):
    items = [item("A", "0101210000", 10)]
    with pytest.raises(ValueError, match="exactly once"):
        M.score(items, [pred("A", "0101210000"), pred("A", "0101210000")], VALID8, VALID10)
    with pytest.raises(ValueError):
        Prediction(item_id="x", candidates=[Candidate(code="1", score=0)] * 51, confidence=0.5)
    with pytest.raises(ValueError):
        Prediction(item_id="x", confidence=1.5)


def test_sanity_arms():
    items = [item("A", "0101210000", 10), item("B", "01012100", 8)]
    frame = M.score(items, OracleArm().predict(items), VALID8, VALID10)
    m = M.compute_metrics(frame, total_seconds=1.0, n_resamples=50)
    assert all(v["accuracy"] == 1.0 for v in m["top1"].values())
    assert m["recall"] == {3: 1.0, 10: 1.0} and m["invalid_rate"] == 0.0
    assert m["coverage"][0]["accuracy"] == 1.0 and m["ece"] == 0.0
    a, b = RandomArm(sorted(VALID8), seed=1, k=2), RandomArm(sorted(VALID8), seed=1, k=2)
    assert a.predict(items) == b.predict(items)  # seeded
    assert all(c.code in VALID8 for p in a.predict(items) for c in p.candidates)
    assert config_hash({"seed": 1, "k": 2}) == config_hash({"k": 2, "seed": 1})
    assert config_hash({"seed": 1}) != config_hash({"seed": 2})


def test_candidate_cap_and_diagnostic_recall(monkeypatch):
    items = [item("A", "0101210000", 10), item("B", "01022910", 8)]
    filler = [f"9{n:09d}" for n in range(48)]  # not valid codes: never a match
    preds = [pred("A", *filler, "0101300000", "0101210000"),  # right code at rank 50
             pred("B", "0102291010")]  # fmt: skip
    frame = M.score(items, preds, VALID8, VALID10)
    m = M.compute_metrics(frame, total_seconds=1.0, n_resamples=20)
    assert m["recall"] == {3: 0.5, 10: 0.5, 20: 0.5, 50: 1.0}
    short = M.compute_metrics(frame.iloc[[1]], total_seconds=1.0, n_resamples=20)
    assert set(short["recall"]) == {3, 10}  # diagnostics only when that many were returned
    monkeypatch.setattr("hs_rag_classifier.eval.arms.MAX_CANDIDATES", 5)
    with pytest.raises(ValueError, match="at most 5"):
        pred("A", *filler[:6])
