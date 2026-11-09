"""Operating point on small synthetic data: the val-only guard, the threshold search, the (b)
per-item form and its exactness, the fixed-share table and stability. No model, no MLflow."""

import math

import numpy as np
import pytest

from hs_rag_classifier.analysis import a3_agreement as AG
from hs_rag_classifier.eval import operating_point as O

# 10 items: confidence, correct, agrees with B1-ex
CONF = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
CORR = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
AGREE = [False, True, False, True, False, True, True, True, False, True]
IDS = [str(i) for i in range(1, 11)]


def a():
    return O.arrays(CONF, CORR, AGREE, IDS)


def test_val_only_guard(tmp_path):
    log = tmp_path / "test_runs.jsonl"
    log.write_text('{"mlflow_run_id": "abc"}\n')
    O.check_val_only("v1", {"split": "val", "dataset": "rulings"}, log)
    for tags in ({"split": "test", "dataset": "rulings"}, {"split": "test", "dataset": "atoz"},
                 {"split": "test", "dataset": "hscodecomp_c"}, {"split": "val", "dataset": "atoz"}):  # fmt: skip
        with pytest.raises(O.NotValError):
            O.check_val_only("v1", tags, log)
    with pytest.raises(O.NotValError, match="test run"):
        O.check_val_only("abc", {"split": "val", "dataset": "rulings"}, log)


def test_threshold_search_under_a():
    r = O.target_rule("a", a(), 0.80)
    # by confidence: accepting the top 6 (0.5..0.95) gives 5/6 < 0.8; top 3 gives 3/3
    order = AG.review_order("a", a())
    acc = AG.accepted_accuracy(order, np.array(CORR, float))
    k = int(np.nonzero(acc >= 0.80)[0][0])
    assert r["k_reviewed"] == k and r["review_share"] == k / 10 and r["exact"]
    assert r["t"] == CONF[k] and r["accuracy"] == pytest.approx(acc[k])
    rev = O.apply_rule("a", {"t": r["t"]}, a())
    assert rev.sum() == k and r["rule_accuracy"] == pytest.approx(r["accuracy"])
    assert not O.target_rule("a", O.arrays([0.5, 0.6], [0, 0], [True, True], ["1", "2"]), 0.9)[
        "reachable"
    ]


def test_b_per_item_form_is_exact_inside_and_beyond_the_disagreements():
    x = a()
    nd = int((~x["agree_cn8"]).sum())  # 4 disagreements
    order = AG.review_order("b", x)
    for k in range(11):
        rule = O.rule_for_cut("b", x, k)
        rev = O.apply_rule("b", rule, x)
        first = np.zeros(10, bool)
        first[order[:k]] = True
        assert np.array_equal(rev, first), k
        if k <= nd:
            assert rule["t_a"] == 0.0
        else:
            assert math.isinf(rule["t_d"])
    r = O.target_rule("b", x, 0.80)
    assert r["exact"] and r["rule_review_share"] == r["review_share"]


def test_ties_make_a_cut_inexact():
    x = O.arrays([0.5, 0.5, 0.9], [0, 1, 1], [True, True, True], ["1", "2", "3"])
    rule = O.rule_for_cut("a", x, 1)  # reviews only item 1, but item 2 has the same confidence
    rev = O.apply_rule("a", rule, x)
    assert rev.sum() == 0  # conf >= t accepts both tied items: not the order's cut
    assert not O.target_rule("a", x, 0.99)["exact"]


def test_share_table_and_stability_shapes():
    x = a()
    t = O.share_table(x, IDS, shares=(0.0, 0.2, 0.5), n_resamples=30)
    assert list(t["n_reviewed"]) == [0, 2, 5] and list(t["n_accepted"]) == [10, 8, 5]
    assert t.loc[0, "a_accuracy"] == t.loc[0, "b_accuracy"] == 0.6
    assert (t["a_low"] <= t["a_accuracy"] + 1e-9).all() and (
        t["b_minus_a"] == t["b_accuracy"] - t["a_accuracy"]
    ).all()
    rules = O.target_table(x, targets=(0.8, 0.99))
    s = O.stability(x, IDS, rules, n_resamples=30)
    assert len(s) == 4 and set(s["policy"]) == {"a", "b"}
    row = s[(s["policy"] == "a") & (s["target"] == 0.8)].iloc[0]
    assert (
        0 <= row["rule_share_at_or_above_target"] <= 1
        and row["need_share_low"] <= row["need_share_high"]
    )
