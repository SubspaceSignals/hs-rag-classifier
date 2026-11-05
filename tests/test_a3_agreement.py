"""A3 agreement analysis on synthetic data: each policy's review order (ties by item id), the
review-rate arithmetic, the metrics, GroupKFold keeping clusters whole, and the refusal of a
test split before anything is loaded. Needs no data, model or server."""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from hs_rag_classifier.analysis import a3_agreement as A3
from hs_rag_classifier.eval import runner as R


def arrays(conf, agree, oof, ids, correct=None):
    return {"id_rank": A3.id_rank(ids), "confidence": np.array(conf, float),
            "agree_cn8": np.array(agree, bool), "oof": np.array(oof, float),
            "correct": np.array(correct if correct is not None else [1] * len(ids), float)}  # fmt: skip


IDS = ["30", "4", "100", "7", "12"]


def ids_in(order, ids=IDS):
    return [ids[i] for i in order]


def test_id_rank_is_numeric_order():
    assert ids_in(np.argsort(A3.id_rank(IDS))) == ["4", "7", "12", "30", "100"]


def test_policy_a_lowest_confidence_first_ties_by_id():
    a = arrays([0.5, 0.2, 0.5, 0.9, 0.2], [1, 1, 1, 1, 1], [0] * 5, IDS)
    assert ids_in(A3.review_order("a", a)) == ["4", "12", "30", "100", "7"]


def test_policy_b_disagreements_first_then_confidence_then_id():
    a = arrays([0.9, 0.1, 0.3, 0.3, 0.05], [0, 1, 0, 0, 1], [0] * 5, IDS)
    # disagree: 30 (0.9), 100 (0.3), 7 (0.3) -> 7 and 100 tie, 7 first; then agree: 12, 4
    assert ids_in(A3.review_order("b", a)) == ["7", "100", "30", "12", "4"]


def test_policy_c_lowest_oof_first_ties_by_id():
    a = arrays([0.1] * 5, [0] * 5, [0.6, 0.6, 0.2, 0.9, 0.2], IDS)
    assert ids_in(A3.review_order("c", a)) == ["12", "100", "4", "30", "7"]


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        A3.review_order("d", arrays([0.1], [0], [0], ["1"]))


@pytest.mark.parametrize("rate,n,k", [(0.10, 750, 75), (0.20, 750, 150), (0.50, 750, 375),
                                      (0.25, 10, 3), (0.15, 10, 2), (0.0, 7, 0), (1.0, 7, 7)])  # fmt: skip
def test_review_count_is_round_half_up(rate, n, k):
    assert A3.n_reviewed(rate, n) == k


def test_accuracy_at_review_and_curve():
    order = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    correct = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1], float)
    assert A3.accuracy_at(order, correct, 0.0) == pytest.approx(0.7)
    assert A3.accuracy_at(order, correct, 0.20) == pytest.approx(7 / 8)
    assert A3.accuracy_at(order, correct, 0.40) == pytest.approx(1.0)
    curve = A3.accepted_accuracy(order, correct)
    assert len(curve) == 10 and curve[0] == pytest.approx(0.7) and curve[-1] == 1.0
    assert A3.rate_for(order, correct, 0.80) == pytest.approx(0.2)
    assert A3.rate_for(order, correct, 0.90) == pytest.approx(0.4)
    assert A3.rate_for(order, np.zeros(10), 0.9) is None
    assert A3.aurc(order, correct) == pytest.approx(np.mean(1 - curve))


def test_auroc_of_the_review_position():
    correct = np.array([0, 0, 1, 1], float)
    assert A3.auroc(np.array([0, 1, 2, 3]), correct) == 1.0  # wrong items reviewed first
    assert A3.auroc(np.array([3, 2, 1, 0]), correct) == 0.0
    assert math.isnan(A3.auroc(np.array([0, 1]), np.ones(2)))


def test_group_kfold_never_splits_a_cluster():
    rng = np.random.default_rng(0)
    groups = np.repeat([f"c{i}" for i in range(40)], rng.integers(1, 5, 40))
    fs = A3.folds(groups)
    assert len(fs) == 5
    seen = np.concatenate([test for _, test in fs])
    assert sorted(seen) == list(range(len(groups)))  # every row tested exactly once
    for train, test in fs:
        assert not set(groups[train]) & set(groups[test])


def test_oof_scores_are_out_of_fold_and_coefficients_reported():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=(n, len(A3.FEATURES)))
    y = (x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(float)
    groups = np.repeat(np.arange(100), 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no convergence or deprecation warnings
        oof, per_fold = A3.oof_scores(x, y, groups)
    assert not np.isnan(oof).any() and ((oof > 0) & (oof < 1)).all()
    assert len(per_fold) == 5 and all(set(A3.FEATURES) <= set(c) for c in per_fold)
    assert all(c["logit_conf"] > 0 for c in per_fold)  # the informative feature, every fold
    full = A3.full_fit(x, y)
    assert full["status"] == "not frozen, not used anywhere"


def test_paired_differences_shape_and_zero_for_identical_policies():
    n = 60
    rng = np.random.default_rng(2)
    conf = rng.random(n)
    frame = pd.DataFrame({"cluster_id": [str(i // 2) for i in range(n)],
                          "correct": (rng.random(n) < conf).astype(float), "confidence": conf,
                          "agree_cn8": np.ones(n, bool), "oof": conf,
                          "id_rank": np.arange(n)})  # fmt: skip
    out = A3.paired_differences(frame, n_resamples=50)
    assert out["n_clusters"] == 30
    for key in ("b-a@0.20", "c-a@0.50", "c-b@auroc"):  # all agree, oof = conf: same order
        d = out["differences"][key]
        assert d["diff"] == pytest.approx(0) and d["low"] == pytest.approx(0)


def test_combined_features_and_agreement():
    a2 = pd.DataFrame({"item_id": ["1", "2", "3"], "chapter": ["82"] * 3, "cluster_id": ["1", "2", "3"],
                       "label_code": ["8211910000"] * 3, "label_depth": [10] * 3,
                       "top1": ["82119100", "82119100", "82119290"], "confidence": [0.9, 0.5, 0.2],
                       "a2_top3": [["82119100"], ["82119100", "82119290"], ["82119290", "82119100"]],
                       "a2_p1": [1, 1, 1], "a2_p2": [1, 1, 1], "correct_8": [1.0, 1.0, 0.0]})  # fmt: skip
    b1 = pd.DataFrame({"item_id": ["3", "2", "1"], "top1": ["8211910000", "8211919000", "8211910000"],
                       "b1ex_sim": [0.7, 0.8, 0.9], "correct_8": [1.0, 0.0, 1.0]})  # fmt: skip
    f = A3.combine(a2, b1, np.array([0.1, 0.2, 0.3]))
    assert list(f["agree_cn8"]) == [True, False, False]
    assert list(f["agree_hs6"]) == [True, True, False]  # 82119190 shares A2's HS6 821191
    assert list(f["b1ex_in_top3_not_top1"]) == [False, False, True]
    assert list(f["b1ex_sim"]) == [0.9, 0.8, 0.7]
    x = A3.feature_matrix(f)
    assert x.shape == (3, 5) and x[0, 0] == pytest.approx(math.log(9))


def test_near_duplicate_verdict():
    def bins(rates, gap):
        return [{"agree_rate": r, "n_agree": int(100 * r), "a2_acc_agree": 0.5 + gap,
                 "a2_acc_disagree": 0.5} for r in rates]  # fmt: skip

    assert A3.near_duplicate_finding(bins([0.1, 0.5], 0.3), bins([0.2, 0.6], 0.3)).startswith(
        "Yes: the agreement signal is concentrated"
    )
    text = A3.near_duplicate_finding(bins([0.3, 0.4], 0.1), bins([0.3, 0.4], 0.3))
    assert text.startswith("No:") and "weakens" in text


def test_test_split_is_refused_by_the_harness_guard():
    with pytest.raises(R.GuardError, match="never run on a test split"):
        A3.check_split("test")
    A3.check_split("val")


def test_cli_refuses_test_before_loading_anything(monkeypatch, capsys):
    monkeypatch.setattr(R, "git_info", lambda: {"commit": "x", "dirty": False})
    monkeypatch.setattr("hs_rag_classifier.eval.items.load",
                        lambda *a, **k: pytest.fail("loaded a split"))  # fmt: skip
    assert A3.main(["--split", "test"]) == 2
    assert "test split" in capsys.readouterr().err
