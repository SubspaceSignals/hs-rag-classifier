"""A2 order-averaging (config a2.order_averaging), on the fakes in test_a2_pipeline: off (or
absent) the graph, its requests and its prediction are unchanged; on, each stage's two orders
are mapped back to codes before averaging, the averaged top two HS6 drive stage 2, a
single-child HS6 still needs no call, and the stage temperature divides log(average). The
candidate config pins configs/a2_pipeline.json. A fake server answers by the order it was
shown. No live server."""

import json
import math
import re

import httpx
import pytest
from test_a2_pipeline import A2CFG, PATHS, RATIONALE, FakeChat, FakeRetriever, a2, make_tariff, run

from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.items import REPO_ROOT
from hs_rag_classifier.pipeline import scoring
from hs_rag_classifier.pipeline.graph_a2 import A2Pipeline
from hs_rag_classifier.pipeline.validate import Validator

FWD = ("821192", "821191", "010121")  # the usual stage-1 order on these fakes (FUSED)
REV = tuple(reversed(FWD))
CN8_FWD = ("82119210", "82119290")
CN8_REV = tuple(reversed(CN8_FWD))
# probability by code for each order shown; deliberately not the same in both orders
STAGE1 = {FWD: {"821192": 0.5, "821191": 0.3, "010121": 0.2},
          REV: {"010121": 0.6, "821191": 0.05, "821192": 0.35}}  # fmt: skip
STAGE2 = {CN8_FWD: {"82119210": 0.2, "82119290": 0.8},
          CN8_REV: {"82119290": 0.3, "82119210": 0.7}}  # fmt: skip


class OrderServer:
    """Replies with each shown option's probability for the order shown; records the orders."""

    def __init__(self, stage1=STAGE1, stage2=STAGE2):
        self.stage1, self.stage2, self.shown = stage1, stage2, []

    def handler(self, request: httpx.Request) -> httpx.Response:
        user = json.loads(request.content)["messages"][1]["content"]
        shown = re.findall(r"^([A-Z])\. (\d{6}|\d{8}):", user, re.MULTILINE)
        codes = tuple(c for _, c in shown)
        self.shown.append(codes)
        table = (self.stage2 if len(codes[0]) == 8 else self.stage1)[codes]
        toks = sorted(({"token": lab, "logprob": math.log(table[c])} for lab, c in shown),
                      key=lambda t: -t["logprob"])  # fmt: skip
        step = {"token": toks[0]["token"], "logprob": toks[0]["logprob"], "top_logprobs": toks}
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [step]}}],
                                         "usage": {"prompt_tokens": 300, "completion_tokens": 1}})  # fmt: skip


def averaging(stage1=STAGE1, stage2=STAGE2, **a2kw):
    server = OrderServer(stage1, stage2)
    http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(server.handler))
    from hs_rag_classifier.arms import a1_pipeline as A1

    cfg = {**A1.DEFAULTS, "m": 3, "labels": ["A", "B", "C"],
           "a2": {**A2CFG, "order_averaging": True, **a2kw}}  # fmt: skip
    tariff = make_tariff()
    pipe = A2Pipeline(cfg, FakeChat(RATIONALE), FakeRetriever(), tariff, Validator(tariff), http,
                      PATHS)  # fmt: skip
    return pipe, server


def strip_timings(x):
    """A state or details without wall-clock fields (they differ between any two runs)."""
    if isinstance(x, dict):
        return {k: strip_timings(v) for k, v in x.items() if k not in ("timings", "llm_s")}
    if isinstance(x, list):
        return [strip_timings(v) for v in x]
    return x


# --- off ---------------------------------------------------------------------------------


def test_off_is_identical_to_a2():
    absent, s_absent, _ = a2()
    off, s_off, _ = a2(order_averaging=False)
    assert not absent.order_averaging and not off.order_averaging
    st_absent, st_off = run(absent), run(off)
    assert strip_timings(st_absent) == strip_timings(st_off)
    assert s_absent.requests == s_off.requests and len(s_off.requests) == 2  # one call per stage
    p_absent = A2.to_prediction("1", st_absent)
    p_off = A2.to_prediction("1", st_off)
    assert strip_timings(p_absent.model_dump()) == strip_timings(p_off.model_dump())
    d = p_off.details
    assert not set(A2.ORDER_KEYS) & set(d["stage1"])
    assert all(not set(A2.ORDER_KEYS) & set(s) for s in d["stage2"])
    # the values the existing A2 tests pin
    assert d["stage1"]["choice_code"] == "821192" and p_off.candidates[0].code == "82119290"
    assert p_off.confidence == pytest.approx(0.6 / 0.95 * 0.75)


# --- on ----------------------------------------------------------------------------------


def test_reversed_order_maps_back_by_code():
    pipe, server = averaging()
    state = run(pipe)
    assert server.shown[:2] == [FWD, REV]  # stage 1 as usual, then reversed (labels by position)
    s1 = state["stage1"]
    assert s1["options"] == list(FWD)
    assert s1["orders"]["forward"]["codes"] == list(FWD)
    assert s1["orders"]["reversed"]["codes"] == list(REV)
    for order, codes in (("forward", FWD), ("reversed", REV)):
        total = sum(STAGE1[codes].values())
        assert s1["orders"][order]["probs"] == pytest.approx(
            {c: p / total for c, p in STAGE1[codes].items()}
        )
    pf, pr = s1["orders"]["forward"]["probs"], s1["orders"]["reversed"]["probs"]
    assert s1["averaged"] == pytest.approx({c: (pf[c] + pr[c]) / 2 for c in FWD})
    assert s1["logprobs"] == pytest.approx([math.log(s1["averaged"][c]) for c in FWD])
    # reversed label A is 010121; its probability went to 010121, not to 821192
    assert s1["orders"]["reversed"]["token"] == "A" and pr["010121"] == pytest.approx(0.6)
    # stage 2 the same way
    s2 = state["stage2_scored"][0]
    assert s2["hs6"] == "821192" and s2["orders"]["reversed"]["codes"] == list(CN8_REV)
    assert s2["orders"]["reversed"]["probs"] == pytest.approx({"82119290": 0.3, "82119210": 0.7})
    assert s2["averaged"] == pytest.approx({"82119210": 0.45, "82119290": 0.55})
    assert state["tokens"]["stage1"] == [600, 2]  # both calls counted
    # saved in the prediction for offline variants
    d = A2.to_prediction("1", state).details
    assert d["stage1"]["orders"]["reversed"]["probs"] == pr
    assert d["stage2"][0]["orders"]["forward"]["probs"] == pytest.approx(
        {"82119210": 0.2, "82119290": 0.8}
    )


def test_averaged_top_two_hs6_drive_stage_2():
    # forward alone would pass 821192 and 821191 on; the average is 0.425, 0.175, 0.4
    pipe, server = averaging()
    state = run(pipe)
    assert [s["hs6"] for s in state["stage2_scored"]] == ["821192", "010121"]
    assert scoring.top_k_indices(state["stage1"]["orders"]["forward"]["logprobs"], 2) == [0, 1]
    assert server.shown == [FWD, REV, CN8_FWD, CN8_REV]
    assert state["stage1"]["choice_code"] == "821192"
    # the averaged choice can differ from the forward one
    flipped = {FWD: {"821192": 0.5, "821191": 0.3, "010121": 0.2},
               REV: {"010121": 0.9, "821191": 0.05, "821192": 0.05}}  # fmt: skip
    pipe, server = averaging(stage1=flipped)
    state = run(pipe)
    assert state["stage1"]["choice_code"] == "010121"
    assert [s["hs6"] for s in state["stage2_scored"]] == ["010121", "821192"]


def test_stage_2_skipped_when_one_cn8():
    one_child = {FWD: {"821192": 0.1, "821191": 0.6, "010121": 0.3},
                 REV: {"010121": 0.3, "821191": 0.6, "821192": 0.1}}  # fmt: skip
    pipe, server = averaging(stage1=one_child)
    state = run(pipe)
    assert server.shown == [FWD, REV]  # no stage-2 call in either order
    assert all(s["skipped"] for s in state["stage2_scored"])
    assert all("orders" not in s for s in state["stage2_scored"])
    assert state["stage2_scored"][0]["logprobs"] == [0.0]
    assert state["final"]["confidence"] == pytest.approx(0.6)


def test_temperature_divides_log_of_the_average():
    t1, t2 = 2.0, 1.5
    pipe, _ = averaging(temperature={"stage1": t1, "stage2": t2})
    state = run(pipe)
    avg1 = state["stage1"]["averaged"]
    avg2 = state["stage2_scored"][0]["averaged"]
    p1 = scoring.renormalise([math.log(avg1[c]) for c in FWD], t1)[0]
    p2 = scoring.renormalise([math.log(avg2[c]) for c in CN8_FWD], t2)
    joint = {"82119290": p1 * p2[1], "82119210": p1 * p2[0]}
    p1_010121 = scoring.renormalise([math.log(avg1[c]) for c in FWD], t1)[2]
    best = max([*joint.values(), p1_010121])
    assert state["final"]["confidence"] == pytest.approx(best)
    assert state["cn8"] == max({**joint, "01012100": p1_010121}.items(), key=lambda x: x[1])[0]
    # rescore on the saved details gives the same, and refits from them
    pred = A2.to_prediction("1", state)
    order, conf = A2.rescore(pred.details, t1, t2)
    assert order == [c.code for c in pred.candidates] and conf == pytest.approx(pred.confidence)
    g = A2.temperature_groups([pred.details], ["8211929000"])
    assert g["stage1"][0][0] == pytest.approx([math.log(avg1[c]) for c in FWD])
    assert g["stage2"][0][0] == pytest.approx([math.log(avg2[c]) for c in CN8_FWD])


def test_average_orders_rejects_different_options():
    with pytest.raises(ValueError):
        scoring.average_orders(["a", "b"], [0.0, 0.0], ["a", "c"], [0.0, 0.0])
    out = scoring.average_orders(["a", "b"], [0.0, -1e6], ["b", "a"], [0.0, -1e6])
    assert out["averaged"] == pytest.approx({"a": 0.5, "b": 0.5})
    assert all(math.isfinite(x) for x in out["logprobs"])


# --- candidate config --------------------------------------------------------------------


def test_candidate_config_pins_a2_pipeline():
    cand = json.loads(A2.CANDIDATE_PATH.read_text(encoding="utf-8"))
    assert cand["base_config"]["path"] == "configs/a2_pipeline.json"
    assert cand["base_config"]["sha256"] == sha256(REPO_ROOT / "configs" / "a2_pipeline.json")
    assert cand["order_averaging"] is True
    assert "to be refitted on val" in cand["temperature"]["status"]
    assert "not frozen" in cand["status"] and "validation only" in cand["status"]
    cfg, base = A2.load_candidate_config(), A2.load_config()
    assert cfg["a2"]["order_averaging"] is True and "order_averaging" not in base["a2"]
    assert cfg["a2"]["temperature"] == cand["temperature"]
    rest = {k: v for k, v in cfg["a2"].items() if k not in ("order_averaging", "temperature")}
    assert rest == {k: v for k, v in base["a2"].items() if k != "temperature"}
    assert {k: v for k, v in cfg.items() if k not in ("a2", "candidate_config")} == {
        k: v for k, v in base.items() if k != "a2"
    }


def test_candidate_config_refuses_a_changed_a2_config(tmp_path):
    cand = json.loads(A2.CANDIDATE_PATH.read_text(encoding="utf-8"))
    cand["base_config"] = {**cand["base_config"], "sha256": "0" * 64}
    p = tmp_path / "cand.json"
    p.write_text(json.dumps(cand), encoding="utf-8")
    with pytest.raises(RuntimeError, match="a2_pipeline.json changed"):
        A2.load_candidate_config(p)


def test_order_avg_arm_is_refused_on_test():
    arm = type("Arm", (), {"name": "a2_order_avg", "val_only": True})()
    git = {"dirty": False}
    with pytest.raises(R.GuardError, match="validation-only"):
        R.check_test_guard(arm, type("E", (), {"split": "test"})(), git, None)
    R.check_test_guard(arm, type("E", (), {"split": "val"})(), git, None)
    assert A2.A2OrderAvgArm.val_only and A2.A2OrderAvgArm.name == "a2_order_avg"
    assert "a2_order_avg" in R.ARMS
