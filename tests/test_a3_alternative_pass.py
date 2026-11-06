"""a3_agreement's optional alternative A2 pass (--a2-partial, --t1, --t2, --b1ex-csv): with none
of these options the CLI calls run as before; the pass is aligned by item id and refused when
the ids differ; the temperatures are applied through A2's rescore; B1-ex comes from a CSV,
aligned by item id; test splits and the default output folder are refused; the b - a bootstrap
and the decision rule. Predictions come from the A2 fakes in test_a2_pipeline. No live server,
no data."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_a2_pipeline import a2, run, top

from hs_rag_classifier.analysis import a3_agreement as A3
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.eval import runner as R

STAGE1S = {
    "5": top(("A", 0.6), ("B", 0.3), ("C", 0.1)),  # 821192 -> stage 2
    "12": top(("B", 0.5), ("C", 0.4), ("A", 0.1)),  # 821191, single child
    "30": top(("C", 0.7), ("A", 0.2), ("B", 0.1)),  # 010121, single child
}
STAGE2 = {"821192": top(("B", 0.75), ("A", 0.25))}
GOLD = {"5": "8211929000", "12": "8211910000", "30": "0102290000"}
VALID8 = {"82119210", "82119290", "82119100", "01012100"}


def items():
    return [SimpleNamespace(item_id=i, label_code=GOLD[i], label_depth=10, cluster_id=n,
                            chapter=GOLD[i][:2]) for n, i in enumerate(GOLD)]  # fmt: skip


def partial_rows(order=("30", "5", "12")):
    """Harness partial JSONL rows (as write_partial makes them), in the given order."""
    out = []
    for i in order:
        pipe, _, _ = a2({"stage1": STAGE1S[i], **STAGE2})
        pred = A2.to_prediction(i, run(pipe))
        out.append(json.loads(json.dumps({"prediction": pred.model_dump(), "latency_s": 1.0})))
    return out


def test_no_new_options_calls_run_as_before(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(A3, "run", lambda *a: seen.setdefault("run", a) and {"consistency": {}})
    monkeypatch.setattr(A3, "run_alternative", lambda *a: pytest.fail("alternative called"))
    assert A3.main(["--out", str(tmp_path)]) == 0
    assert seen["run"] == (A3.A2_PASS, tmp_path, A3.M.BOOTSTRAP_RESAMPLES, "val")
    seen.clear()
    assert A3.main([]) == 0
    assert seen["run"] == (A3.A2_PASS, A3.OUT, A3.M.BOOTSTRAP_RESAMPLES, "val")
    with pytest.raises(SystemExit):  # the new options only go with --a2-partial
        A3.main(["--t1", "1.2"])
    with pytest.raises(SystemExit):
        A3.main(["--a2-partial", "x.jsonl", "--t1", "1.2"])  # needs --t2 and --b1ex-csv


def test_cli_passes_the_alternative_inputs(monkeypatch, tmp_path):
    seen = {}

    def spy(*a):
        seen["args"] = a
        return {"decision": {"case": "KEEP (b)"}}

    monkeypatch.setattr(A3, "run_alternative", spy)
    monkeypatch.setattr(A3, "run", lambda *a: pytest.fail("run called"))
    argv = ["--a2-partial", "p.jsonl", "--t1", "1.1645", "--t2", "1.373", "--b1ex-csv", "b.csv",
            "--out", str(tmp_path)]  # fmt: skip
    assert A3.main(argv) == 0
    a = seen["args"]
    assert a[:5] == (A3.Path("p.jsonl"), 1.1645, 1.373, A3.Path("b.csv"), tmp_path)
    assert a[6] == "val" and a[7] == A3.A2_PASS


def test_alternative_pass_aligns_by_item_id():
    its = items()
    f = A3.alt_a2_rows(partial_rows(), 1.0, 1.0, VALID8, set(), its)
    assert list(f["item_id"]) == ["5", "12", "30"]  # the items' order, not the file's
    assert list(f["top1"]) == ["82119290", "82119100", "01012100"]
    assert list(f["correct_8"]) == [1.0, 1.0, 0.0]
    assert f.loc[0, "a2_top3"] == ["82119290", "82119100", "82119210"]
    with pytest.raises(R.GuardError, match="item ids"):
        A3.alt_a2_rows(partial_rows(("30", "5")), 1.0, 1.0, VALID8, set(), its)
    with pytest.raises(R.GuardError, match="item ids"):
        A3.alt_a2_rows(partial_rows(("30", "5", "5", "12")), 1.0, 1.0, VALID8, set(), its)


def test_temperatures_are_applied():
    rows = partial_rows()
    by_id = {r["prediction"]["item_id"]: r["prediction"]["details"] for r in rows}
    f1 = A3.alt_a2_rows(rows, 1.0, 1.0, VALID8, set(), items())
    f2 = A3.alt_a2_rows(rows, 2.0, 1.5, VALID8, set(), items())
    for f, (t1, t2) in ((f1, (1.0, 1.0)), (f2, (2.0, 1.5))):
        for i, conf in zip(f["item_id"], f["confidence"], strict=True):
            assert conf == pytest.approx(A2.rescore(by_id[i], t1, t2)[1])
    assert f1.loc[0, "confidence"] == pytest.approx(0.6 * 0.75)  # stage 1 renormalised over A-C
    assert f2.loc[0, "confidence"] < f1.loc[0, "confidence"]
    # at T1 = 2 the flatter stage 1 lets the single-child 821191 (p1 0.334, p2 1) overtake
    # 821192's best pair (0.473 x 0.675 = 0.319): the temperature can change the top-1
    lp = np.log([0.6, 0.3, 0.1])
    p1 = np.exp(lp / 2.0) / np.exp(lp / 2.0).sum()
    p2 = np.exp(np.log([0.25, 0.75]) / 1.5) / np.exp(np.log([0.25, 0.75]) / 1.5).sum()
    assert p1[1] > p1[0] * p2[1]
    assert f2.loc[0, "top1"] == "82119100" and f1.loc[0, "top1"] == "82119290"
    assert f2.loc[0, "a2_p1"] == pytest.approx(p1[1]) and f2.loc[0, "a2_p2"] == 1.0
    assert f2.loc[0, "confidence"] == pytest.approx(p1[1])
    assert f1.loc[0, "a2_p1"] == pytest.approx(0.6) and f1.loc[0, "a2_p2"] == pytest.approx(0.75)


def test_b1ex_csv_aligns_by_id_and_refuses_other_ids(tmp_path):
    p = tmp_path / "per_item.csv"
    pd.DataFrame({"item_id": ["30", "5", "12"], "b1ex_cn8": ["01012100", "82119210", "82119100"],
                  "b1ex_sim": [0.9, 0.8, 0.7], "b1ex_correct": [0.0, 0.0, 1.0],
                  "tfidf_nearest_train": [0.3, 0.2, 0.1]}).to_csv(p, index=False)  # fmt: skip
    b1, tfidf = A3.b1ex_from_csv(p, ["5", "12", "30"])
    assert list(b1["item_id"]) == ["5", "12", "30"]
    assert list(b1["top1"]) == ["82119210", "82119100", "01012100"]
    assert list(tfidf) == [0.2, 0.1, 0.3]
    with pytest.raises(R.GuardError, match="item ids"):
        A3.b1ex_from_csv(p, ["5", "12", "31"])
    # combine sees the agreement the policies use
    a = A3.alt_a2_rows(partial_rows(), 1.0, 1.0, VALID8, set(), items())
    f = A3.combine(a, b1, tfidf)
    assert list(f["agree_cn8"]) == [False, True, True]


def test_test_split_and_default_out_are_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(R, "git_info", lambda: {"commit": "x", "dirty": False})
    monkeypatch.setattr("hs_rag_classifier.eval.items.load",
                        lambda *a, **k: pytest.fail("loaded data"))  # fmt: skip
    with pytest.raises(R.GuardError):
        A3.run_alternative(tmp_path / "p.jsonl", 1.0, 1.0, tmp_path / "b.csv", tmp_path,
                           split="test")  # fmt: skip
    with pytest.raises(R.GuardError, match="give --out"):
        A3.run_alternative(tmp_path / "p.jsonl", 1.0, 1.0, tmp_path / "b.csv", A3.OUT)
    argv = ["--split", "test", "--a2-partial", "p.jsonl", "--t1", "1", "--t2", "1",
            "--b1ex-csv", "b.csv", "--out", str(tmp_path)]  # fmt: skip
    assert A3.main(argv) == 2
    assert "Refused" in capsys.readouterr().err


def test_paired_b_minus_a_and_decision():
    n = 60
    rng = np.random.default_rng(3)
    ids = [str(i) for i in range(n)]
    frame = pd.DataFrame({"cluster_id": np.arange(n) // 2, "id_rank": A3.id_rank(ids),
                          "correct": rng.integers(0, 2, n).astype(float),
                          "confidence": rng.random(n), "agree_cn8": True})  # fmt: skip
    same = A3.paired_b_minus_a(frame, n_resamples=50)["differences"]
    assert set(same) == {*(f"b-a@{r:.2f}" for r in A3.REVIEW_RATES), "b-a@auroc", "b-a@aurc"}
    assert all(v["diff"] == 0 and v["low"] == 0 and v["high"] == 0 for v in same.values())
    frame["agree_cn8"] = frame["correct"] == 1  # a perfect disagreement signal: b better
    d = A3.paired_b_minus_a(frame, n_resamples=50)["differences"]
    assert d["b-a@aurc"]["diff"] < 0 and d["b-a@auroc"]["diff"] > 0
    pols = {"a": {"aurc": 0.20}, "b": {"aurc": 0.25}}
    above = {"differences": {"b-a@aurc": {"diff": 0.05, "low": 0.01, "high": 0.09}}}
    spans = {"differences": {"b-a@aurc": {"diff": 0.05, "low": -0.01, "high": 0.09}}}
    assert A3.ordering_decision(pols, above)["case"] == "SWITCH to (a)"
    assert A3.ordering_decision(pols, spans)["case"] == "KEEP (b)"
    assert A3.ordering_decision({"a": {"aurc": 0.3}, "b": {"aurc": 0.25}}, above)["case"] == (
        "KEEP (b)"
    )
