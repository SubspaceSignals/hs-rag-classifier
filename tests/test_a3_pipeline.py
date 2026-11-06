"""A3: the frozen config and its SHA-256 pins, the temperatures, policy (b) ordering with its
tie-breaks, the arm on fakes, the guards (a3 allowed on test, a2_order_avg still refused), the
optional runner hook (non-hook arms unchanged) and the comparison script on synthetic
predictions (never a test run). No live server, no real data."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from test_a2_order_averaging import averaging
from test_a2_pipeline import run as run_graph
from test_eval_harness import VALID8, VALID10, FixedArm, data, make_set  # noqa: F401
from test_eval_harness import run as run_harness

from hs_rag_classifier.analysis import a3_compare as C
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.arms import a3_pipeline as A3
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.arms import Candidate, Prediction
from hs_rag_classifier.eval.items import REPO_ROOT

T1, T2 = 1.1645301855885621, 1.3730046264063607

# --- config ------------------------------------------------------------------------------


def test_config_pins_and_loads():
    raw = json.loads(A3.CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["arm"] == "a3" and raw["display_name"] == "A3 Order-averaged A2 + review queue"
    assert "config_hash" not in raw
    for key, path in (("base_config", "configs/a2_pipeline.json"),
                      ("b1ex_config", "configs/b1ex_embed.json")):  # fmt: skip
        assert raw[key] == {"path": path, "sha256": sha256(REPO_ROOT / path)}
    cfg = A3.load_config()
    base = A2.load_config()
    assert cfg["a2"]["order_averaging"] is True and cfg["a2"]["temperature"] == raw["temperature"]
    rest = {k: v for k, v in cfg["a2"].items() if k not in ("order_averaging", "temperature")}
    assert rest == {k: v for k, v in base["a2"].items() if k != "temperature"}
    assert cfg["b1ex"]["k"] == 1 and cfg["b1ex"]["model"] == "bge-m3"
    assert cfg["b1ex"]["revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert (
        raw["review_policy"]["name"] == "b" and raw["review_policy"]["fitted_parameters"] == "none"
    )


def test_temperatures_are_the_refitted_values_at_full_precision():
    t = A3.load_config()["a2"]["temperature"]
    assert t["stage1"] == T1 and t["stage2"] == T2 and t["mode"] == "per_stage"


@pytest.mark.parametrize("key,name", [("base_config", "a2_pipeline.json"),
                                      ("b1ex_config", "b1ex_embed.json")])  # fmt: skip
def test_a_wrong_pin_is_refused(tmp_path, key, name):
    raw = json.loads(A3.CONFIG_PATH.read_text(encoding="utf-8"))
    raw[key] = {**raw[key], "sha256": "0" * 64}
    p = tmp_path / "a3.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match=f"{name} changed"):
        A3.load_config(p)


# --- policy (b) --------------------------------------------------------------------------


def frame(ids, conf, agree, correct, latency=None, rationale_s=None):
    n = len(ids)
    details = [json.dumps({"stage1": {"choice_code": "821192"},
                           "a3": {"b1ex_top1": "8211920000", "b1ex_cn8": "82119200",
                                  "b1ex_sim": 0.8, "agree_cn8": bool(g), "b1ex_s": 0.01},
                           "rationale": {"citation": {"pass": True}},
                           "timings": {"rationale": (rationale_s or [0.5] * n)[i]}})
               for i, g in enumerate(agree)]  # fmt: skip
    return pd.DataFrame({
        "item_id": ids, "cluster_id": [f"c{i // 2}" for i in range(n)],
        "confidence": conf, "correct_8": correct, "correct_6": correct,
        "candidates": [["82119290", "82119210", "82119100"]] * n,
        "scores": [[0.5, 0.2, 0.1]] * n, "details": details,
        "latency_s": latency or [2.0] * n})  # fmt: skip


def test_policy_b_orders_disagreements_first_then_confidence_then_id():
    ids = ["30", "4", "100", "7", "12", "9"]
    conf = [0.9, 0.1, 0.3, 0.3, 0.05, 0.3]
    agree = [False, True, False, False, True, True]
    f = A3.review_columns(frame(ids, conf, agree, [1.0] * 6))
    order = list(f.sort_values("review_rank")["item_id"])
    # disagree: 7 and 100 tie at 0.3 (7 first by numeric id), then 30; agree: 12, 4, 9
    assert order == ["7", "100", "30", "12", "4", "9"]
    assert (
        list(f.set_index("item_id").loc[order, "review_group"]) == ["disagree"] * 3 + ["agree"] * 3
    )
    assert sorted(f["review_rank"]) == list(range(1, 7))
    assert f.loc[0, "top3_cn8"] == ["82119290", "82119210", "82119100"]
    assert list(f["latency_score_only_s"]) == [1.5] * 6


def test_after_scoring_metrics_and_intervals():
    rng = np.random.default_rng(0)
    n = 40
    ids = [str(i) for i in range(n)]
    correct = rng.integers(0, 2, n).astype(float)
    out = A3.after_scoring(frame(ids, list(rng.random(n)), list(correct == 1), list(correct)),
                           n_resamples=50)  # fmt: skip
    m = out["metrics"]
    assert m["agree_rate"] == pytest.approx(correct.mean())
    assert m["headline_policy_b"] == m["policies"]["b"]["accuracy_at_review"]["0.20"]
    # a perfect disagreement signal: (b) reviews every wrong item first
    assert m["policies"]["b"]["auroc"] == 1.0
    assert m["bootstrap"] == {"resamples": 50, "seed": M.BOOTSTRAP_SEED, "unit": "cluster_id"}
    for k in ("b_acc_review_20", "a_auroc", "b_aurc", "agree_rate"):
        lo, hi = m["intervals"][k]["low"], m["intervals"][k]["high"]
        assert lo <= m["point"][k] + 1e-12 and m["point"][k] <= hi + 1e-12
    assert m["speed"]["with_rationale"]["items_per_minute"] == pytest.approx(30.0)
    assert m["speed"]["score_only"]["items_per_minute"] == pytest.approx(40.0)
    assert "a3_headline_b" in out["flat"] and "review_rank" in out["frame"]


# --- arm on fakes ------------------------------------------------------------------------


class FakeB1ex:
    def __init__(self, code):
        self.code = code

    def predict(self, items):
        return [Prediction(item_id=i.item_id, candidates=[Candidate(code=self.code, score=0.83)],
                           confidence=0.5) for i in items]  # fmt: skip


def test_arm_adds_b1ex_and_agreement_to_a2_details():
    pipe, _ = averaging(temperature={"stage1": T1, "stage2": T2})
    cfg = {**pipe.cfg, "b1ex": {"k": 1}, "a3": {}}
    item = SimpleNamespace(item_id="1", text="a kitchen knife")
    top = A2.to_prediction("1", run_graph(pipe)).candidates[0].code[:8]
    other = next(c for c in ("82119290", "82119100") if c != top)
    for code, agree in ((top + "00", True), (other + "00", False)):
        arm = A3.A3Arm(cfg, pipeline=pipe, b1ex=FakeB1ex(code))
        p = arm.predict([item])[0]
        expected = A2.to_prediction("1", run_graph(pipe))
        assert [c.code for c in p.candidates] == [c.code for c in expected.candidates]
        assert p.confidence == pytest.approx(expected.confidence)
        a3 = p.details["a3"]
        assert a3["b1ex_cn8"] == code[:8] and a3["agree_cn8"] is agree and a3["b1ex_sim"] == 0.83
        assert "orders" in p.details["stage1"]  # the averaging path
    assert arm.name == "a3" and not getattr(arm, "val_only", False)


# --- guards ------------------------------------------------------------------------------


def test_a3_allowed_on_test_while_a2_order_avg_is_refused(monkeypatch):
    git = {"commit": "x", "dirty": False}
    test = SimpleNamespace(split="test")
    R.check_test_guard(A3.A3Arm.__new__(A3.A3Arm), test, git, None)  # no error
    with pytest.raises(R.GuardError, match="validation-only"):
        R.check_test_guard(A2.A2OrderAvgArm.__new__(A2.A2OrderAvgArm), test, git, None)
    with pytest.raises(R.GuardError, match="dirty"):
        R.check_test_guard(A3.A3Arm.__new__(A3.A3Arm), test, {"commit": "x", "dirty": True}, None)
    monkeypatch.setenv("HS_ALLOW_MULTISLOT", "1")
    with pytest.raises(R.GuardError, match="multi-slot"):
        R.check_test_guard(A3.A3Arm.__new__(A3.A3Arm), test, git, None)
    assert "a3" in R.ARMS and "a2_order_avg" in R.ARMS


# --- runner hook -------------------------------------------------------------------------


def test_non_hook_arm_metrics_unchanged_by_the_hook_machinery(data, tmp_path):  # noqa: F811
    result = run_harness(FixedArm(), make_set(data, split="val"), tmp_path)
    f = result["frame"]
    expected = M.compute_metrics(f, float(f["latency_s"].sum()), n_resamples=20)
    assert json.dumps(result["metrics"], sort_keys=True, default=str) == json.dumps(
        expected, sort_keys=True, default=str
    )
    assert "arm_metrics" not in result["metrics"]
    assert R.arm_summary_lines(result) == []
    assert not {"review_rank", "agree_cn8"} & set(f.columns)


class HookArm(FixedArm):
    name = "hooked"

    def after_scoring(self, frame, n_resamples, seed):
        self.seen = (n_resamples, seed, list(frame["cluster_id"]))
        return {"frame": frame.assign(extra=1), "metrics": {"x": 1.5}, "flat": {"hook_x": 1.5}}


def test_hook_adds_columns_metrics_and_flat_metrics(data, tmp_path):  # noqa: F811
    arm = HookArm()
    result = run_harness(arm, make_set(data, split="val"), tmp_path)
    assert result["metrics"]["arm_metrics"] == {"x": 1.5} and "extra" in result["frame"]
    assert arm.seen[0] == 20 and arm.seen[1] == M.BOOTSTRAP_SEED
    client = MlflowClient(tracking_uri=R.tracking_uri(tmp_path / "mlflow"))
    assert client.get_run(result["run_id"]).data.metrics["hook_x"] == 1.5
    plain = run_harness(FixedArm(), make_set(data, split="val"), tmp_path)["metrics"]
    hooked = {k: v for k, v in result["metrics"].items() if k not in ("arm_metrics", "latency")}
    assert json.dumps(hooked, sort_keys=True, default=str) == json.dumps(
        {k: v for k, v in plain.items() if k != "latency"}, sort_keys=True, default=str
    )


# --- comparison script -------------------------------------------------------------------

GOLD = ["8211910000", "8211920000", "0101210000", "8211910000", "0101210000", "8211920000"]
CV8 = {"82119100", "82119200", "01012100"}


def cmp_items():
    return [SimpleNamespace(item_id=str(i + 1), label_code=g, label_depth=10, cluster_id=f"c{i // 2}",
                            chapter=g[:2]) for i, g in enumerate(GOLD)]  # fmt: skip


def write_inputs(tmp_path):
    a3_top = ["82119100", "82119200", "01012100", "82119200", "01012100", "82119100"]  # 4 right
    a2_top = ["82119100", "82119100", "01012100", "82119200", "82119100", "82119100"]  # 2 right
    conf3, conf2 = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1], [0.9, 0.3, 0.8, 0.2, 0.1, 0.5]
    agree = [True, True, True, False, True, False]
    with (tmp_path / "a3.jsonl").open("w") as fh:
        for i, (c, p, g) in enumerate(zip(a3_top, conf3, agree, strict=True)):
            pred = Prediction(item_id=str(i + 1), candidates=[Candidate(code=c, score=p)],
                              confidence=p, details={"a3": {"agree_cn8": g}})  # fmt: skip
            fh.write(json.dumps({"prediction": pred.model_dump(), "latency_s": 1.0}) + "\n")
    rows = [{"item_id": str(i + 1), "candidates": [c], "scores": [p], "confidence": p}
            for i, (c, p) in enumerate(zip(a2_top, conf2, strict=True))]  # fmt: skip
    (tmp_path / "a2.json").write_text(json.dumps({"rows": rows[::-1]}))  # any order
    return tmp_path / "a3.jsonl", tmp_path / "a2.json"


def test_comparison_on_synthetic_predictions_is_never_a_test_run(tmp_path, monkeypatch):
    a3, a2 = write_inputs(tmp_path)
    monkeypatch.setattr(R, "TEST_LOG", tmp_path / "eval" / "test_runs.jsonl")
    monkeypatch.setattr(R, "check_test_guard", lambda *a, **k: pytest.fail("guard called"))
    r = C.run_compare(a3, a2, cmp_items(), "rulings", "test", True, out=tmp_path / "out",
                      mlflow_dir=tmp_path / "mlflow", n_resamples=50, valid=(CV8, set()))  # fmt: skip
    assert r["paired"]["cn8_top1"]["diff"] == pytest.approx(4 / 6 - 2 / 6)
    assert r["paired"]["cn8_top1"]["a"] == pytest.approx(4 / 6)
    assert r["cross_server"] is True and r["a3_agree_rate"] == pytest.approx(4 / 6)
    assert set(r["review"]) == {"acc_review_10", "acc_review_20", "acc_review_30",
                                "acc_review_40", "acc_review_50", "auroc", "aurc"}  # fmt: skip
    assert not (tmp_path / "eval").exists()
    run = MlflowClient(tracking_uri=R.tracking_uri(tmp_path / "mlflow")).get_run(r["mlflow_run_id"])
    tags = run.data.tags
    assert (
        tags["comparison"] == "true" and tags["cross_server"] == "true" and tags["split"] == "test"
    )
    assert tags["input_a3"] == str(a3) and tags["input_a3_sha256"] == sha256(a3)
    assert tags["input_a2_sha256"] == sha256(a2)
    assert json.loads((tmp_path / "out" / "compare_rulings_test.json").read_text())["cross_server"]
    r2 = C.run_compare(a3, a2, cmp_items(), "rulings", "val", False, mlflow_dir=tmp_path / "mlflow",
                       n_resamples=50, valid=(CV8, set()))  # fmt: skip
    assert r2["cross_server"] is False
    tags2 = (
        MlflowClient(tracking_uri=R.tracking_uri(tmp_path / "mlflow"))
        .get_run(r2["mlflow_run_id"])
        .data.tags
    )
    assert tags2["cross_server"] == "false"


def test_comparison_reads_parquet_and_writes_reference_rows(tmp_path):
    _, a2 = write_inputs(tmp_path)
    preds = C.load_predictions(a2)
    f = M.score(cmp_items(), preds, CV8, set(), max_depth=8)
    f.to_parquet(tmp_path / "predictions.parquet", index=False)
    back = C.load_predictions(tmp_path / "predictions.parquet")
    assert {p.item_id: p.candidates[0].code for p in back} == {
        p.item_id: p.candidates[0].code for p in preds
    }
    r = C.run_reference(tmp_path / "predictions.parquet", a2, cmp_items(), "rulings", "val",
                        mlflow_dir=tmp_path / "mlflow", valid=(CV8, set()))  # fmt: skip
    assert r["a2_confidence"]["top1_8"] == pytest.approx(2 / 6)
    assert set(r) >= {"a2_confidence", "b1ex_similarity", "mlflow_run_id"}
    with pytest.raises(ValueError):
        C.load_predictions(tmp_path / "x.csv")


def test_comparison_refuses_mismatched_items(tmp_path):
    a3, a2 = write_inputs(tmp_path)
    with pytest.raises(ValueError):
        C.run_compare(a3, a2, cmp_items()[:5], "rulings", "val", False,
                      mlflow_dir=tmp_path / "mlflow", n_resamples=10, valid=(CV8, set()))  # fmt: skip


def _old_parquet_loader(path):
    """load_predictions' parquet branch as it was before details became optional (68c40a4)."""
    f = pd.read_parquet(path)
    return [
        Prediction(item_id=r.item_id,
                   candidates=[Candidate(code=c, score=float(s))
                               for c, s in zip(r.candidates, r.scores, strict=True)],
                   confidence=None if pd.isna(r.confidence) else float(r.confidence),
                   abstain=bool(r.abstain),
                   details=json.loads(r.details) if isinstance(r.details, str) else None)
        for r in f.itertuples(index=False)
    ]  # fmt: skip


def _scored_parquet(tmp_path, details: bool):
    a3, _ = write_inputs(tmp_path)
    f = M.score(cmp_items(), C.load_predictions(a3), CV8, set(), max_depth=8)
    if details:
        f["details"] = [json.dumps({"a3": {"agree_cn8": i % 2 == 0}}) for i in range(len(f))]
        f.loc[1, "details"] = None  # a row without details stays None
    else:
        f = f.drop(columns=[c for c in ("details",) if c in f.columns])
    p = tmp_path / ("with.parquet" if details else "without.parquet")
    f.to_parquet(p, index=False)
    return p


def test_parquet_without_details_loads_with_details_none(tmp_path):
    p = _scored_parquet(tmp_path, details=False)
    assert "details" not in pd.read_parquet(p).columns
    preds = C.load_predictions(p)
    assert len(preds) == 6 and all(x.details is None for x in preds)
    assert [x.candidates[0].code for x in preds] == list(pd.read_parquet(p)["top1"])
    # reference needs no details: a B1-ex file without the column gives its rows
    _, a2 = write_inputs(tmp_path)
    r = C.run_reference(a2, p, cmp_items(), "rulings", "val", mlflow_dir=tmp_path / "mlflow",
                        valid=(CV8, set()))  # fmt: skip
    assert r["b1ex_similarity"]["top1_8"] == pytest.approx(4 / 6)


def test_parquet_with_details_loads_as_before(tmp_path):
    p = _scored_parquet(tmp_path, details=True)
    new, old = C.load_predictions(p), _old_parquet_loader(p)
    assert [x.model_dump() for x in new] == [x.model_dump() for x in old]
    assert new[0].details == {"a3": {"agree_cn8": True}} and new[1].details is None


def test_compare_without_a3_details_fails_with_a_clear_message(tmp_path):
    p = _scored_parquet(tmp_path, details=False)
    _, a2 = write_inputs(tmp_path)
    with pytest.raises(ValueError, match=r"has no A3 details .* 6 of 6 items, e\.g\. item 1"):
        C.run_compare(p, a2, cmp_items(), "rulings", "val", False, mlflow_dir=tmp_path / "mlflow",
                      n_resamples=10, valid=(CV8, set()))  # fmt: skip
    with pytest.raises(ValueError, match="1 of 6 items, e.g. item 2"):  # one row without
        C.a3_agreement(C.load_predictions(_scored_parquet(tmp_path, details=True)),
                       [str(i) for i in range(1, 7)])  # fmt: skip
