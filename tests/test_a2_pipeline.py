"""A2 option scoring: label distributions with the floor, outside-label mass, joint scoring over
the top two HS6, temperature fitting, single-child stage 2, the rationale citation check, the
validator fallback and A1's behaviour unchanged. A fake llama-server (httpx MockTransport) answers
the scoring calls and a fake chat model the rationale; the one server test skips when
llama-server is not answering."""

import json
import math
import re
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.pipeline import checks, scoring
from hs_rag_classifier.pipeline.graph import DISCLAIMER, Pipeline
from hs_rag_classifier.pipeline.graph_a2 import A2Pipeline
from hs_rag_classifier.pipeline.llm import ServerInfo, score_call
from hs_rag_classifier.pipeline.options import Tariff
from hs_rag_classifier.pipeline.validate import Validator

# --- label distribution ------------------------------------------------------------------


def top(*pairs):
    return [{"token": t, "logprob": math.log(p)} for t, p in pairs]


def test_label_logprobs_floor_and_outside_mass():
    d = scoring.label_logprobs(
        top(("A", 0.6), ("x", 0.2), ("B", 0.1), (" A", 0.05)), ["A", "B", "C"]
    )
    assert d["floored"] == ["C"]
    assert d["logprobs"][2] == pytest.approx(math.log(0.05))  # lowest returned as the floor
    assert d["outside_mass"] == pytest.approx(0.3)  # 1 - (0.6 + 0.1); " A" is not the label
    p = scoring.renormalise(d["logprobs"])
    assert p.sum() == pytest.approx(1.0)
    assert p == pytest.approx(np.array([0.6, 0.1, 0.05]) / 0.75)
    with pytest.raises(ValueError):
        scoring.label_logprobs([], ["A"])


def test_temperature_flattens_or_sharpens():
    lps = [math.log(0.7), math.log(0.2), math.log(0.1)]
    assert scoring.renormalise(lps, 1.0) == pytest.approx([0.7, 0.2, 0.1])
    assert scoring.renormalise(lps, 2.0)[0] < 0.7 < scoring.renormalise(lps, 0.5)[0]
    assert scoring.top_k_indices([-1.0, -0.5, -0.5, -3.0], 2) == [1, 2]  # ties: earlier first


def test_rank_pairs_joint_over_two_hs6_and_single_child():
    stage1 = {"options": ["821192", "821191", "010121"],
              "logprobs": [math.log(0.6), math.log(0.3), math.log(0.1)]}  # fmt: skip
    stage2 = [
        {
            "hs6": "821192",
            "cn8s": ["82119210", "82119290"],
            "logprobs": [math.log(0.25), math.log(0.75)],
        },
        {"hs6": "821191", "cn8s": ["82119100"], "logprobs": [0.0]},
    ]
    ranked = scoring.rank_pairs(stage1, stage2)
    assert [p["cn8"] for p in ranked] == ["82119290", "82119100", "82119210"]
    assert [p["joint"] for p in ranked] == pytest.approx([0.45, 0.3, 0.15])
    assert ranked[1]["p2"] == 1.0  # single child: p2 = 1


def test_fit_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(3000):
        logits = rng.normal(0, 1.5, 6)
        true = rng.choice(6, p=scoring.renormalise(logits))
        groups.append((list(logits * 2.0), int(true)))  # overconfident by a factor of 2
    fit = scoring.fit_temperature(groups)
    assert fit["t"] == pytest.approx(2.0, rel=0.1)
    assert fit["nll"] < fit["nll_t1"]
    with pytest.raises(ValueError):
        scoring.fit_temperature([])


def test_quote_check():
    text = "Knives > Other knives having fixed blades\nA. 821192: Knives > Other"
    assert checks.quote_check("'other KNIVES having   fixed blades'", text)["pass"]
    assert checks.quote_check("Knives ... fixed blades", text)["pass"]
    assert not checks.quote_check("ceramic knives", text)["pass"]
    assert not checks.quote_check("", text)["pass"]


# --- fake server and parts ---------------------------------------------------------------


def make_tariff() -> Tariff:
    hs6 = {"821191": "Knives > Table knives having fixed blades",
           "010121": "Live horses > Pure-bred breeding animals",
           "821192": "Knives > Other knives having fixed blades"}  # fmt: skip
    cn8 = {"82119100": "Table knives having fixed blades",
           "01012100": "Pure-bred breeding animals",
           "82119210": "Hunting knives", "82119290": "Other"}  # fmt: skip
    children = {"821191": ["82119100"], "010121": ["01012100"],
                "821192": ["82119210", "82119290"]}  # fmt: skip
    return Tariff(hs6, cn8, children, {c: [c + "00"] for c in cn8})


PATHS = {"82119290": "KNIVES > Knives > Other knives having fixed blades > Other",
         "82119210": "KNIVES > Knives > Other knives having fixed blades > Hunting knives",
         "82119100": "KNIVES > Knives > Table knives having fixed blades",
         "01012100": "LIVE ANIMALS > Live horses > Pure-bred breeding animals"}  # fmt: skip
FUSED = ["82119290", "82119100", "01012100", "82119210"]


class FakeServer:
    """Scoring replies by stage ("stage1", or the HS6 in a stage-2 prompt); records requests."""

    def __init__(self, replies):
        self.replies, self.requests = replies, []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        user = body["messages"][1]["content"]
        m = re.search(r"Subheading (\d{6}):", user)
        key = m.group(1) if m else "stage1"
        toks = self.replies[key][: body["top_logprobs"]]
        step = {"token": toks[0]["token"], "logprob": toks[0]["logprob"], "top_logprobs": toks}
        return httpx.Response(200, json={
            "choices": [{"logprobs": {"content": [step]}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 1}})  # fmt: skip


class FakeChat:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def invoke(self, messages, response_format, max_tokens):
        self.calls.append({"messages": messages, "schema": response_format["json_schema"]})
        if isinstance(self.reply, Exception):
            raise self.reply
        content = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return SimpleNamespace(content=content,
                               usage_metadata={"input_tokens": 50, "output_tokens": 30})  # fmt: skip


class FakeRetriever:
    def __init__(self):
        self.config = {"weights": {"kw": 1.0}, "k_rrf": 10}

    def source_lists(self, queries):
        return [{"kw": [SimpleNamespace(cn8=c) for c in FUSED]}]


A2CFG = {"top_logprobs": 20, "stage2_top_hs6": 2,
         "temperature": {"stage1": 1.0, "stage2": 1.0},
         "rationale": {"max_tokens": 150, "quote_chars": 160, "explanation_chars": 360}}  # fmt: skip
STAGE1 = top(("A", 0.6), ("B", 0.3), ("C", 0.05), ("x", 0.05))  # A 821192, B 821191, C 010121
STAGE2 = {"821192": top(("B", 0.75), ("A", 0.25))}
RATIONALE = {"quote": "Other knives having fixed blades", "explanation": "A kitchen knife."}


def a2(replies=None, rationale=RATIONALE, validator=None, **a2kw):
    tariff = make_tariff()
    server = FakeServer(replies or {"stage1": STAGE1, **STAGE2})
    http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(server.handler))
    cfg = {**A1.DEFAULTS, "m": 3, "labels": ["A", "B", "C"], "a2": {**A2CFG, **a2kw}}
    chat = FakeChat(rationale)
    pipe = A2Pipeline(cfg, chat, FakeRetriever(), tariff, validator or Validator(tariff), http,
                      PATHS)  # fmt: skip
    return pipe, server, chat


def run(pipe):
    return pipe.build().invoke({"item_id": "1", "raw": "a kitchen knife"})


# --- graph -------------------------------------------------------------------------------


def test_a2_scores_both_stages_and_ranks_pairs():
    pipe, server, chat = a2()
    state = run(pipe)
    assert len(server.requests) == 2  # stage 1, then stage 2 for 821192 only (821191: one child)
    req = server.requests[0]
    assert req["max_tokens"] == 1 and req["logprobs"] and req["top_logprobs"] == 20
    assert "response_format" not in req and "grammar" not in req  # raw next-token distribution
    assert req["messages"][1]["content"].rstrip().endswith("label of the best option only.")
    s1 = state["stage1"]
    assert s1["floored"] == [] and s1["outside_mass"] == pytest.approx(0.05)
    assert [s["hs6"] for s in state["stage2_scored"]] == ["821192", "821191"]
    assert state["stage2_scored"][1]["skipped"]
    final = state["final"]
    assert final["candidates"] == ["82119290", "82119100", "82119210"]
    joint = 0.6 / 0.95 * 0.75
    assert final["confidence"] == pytest.approx(joint)
    assert final["disclaimer"] == DISCLAIMER and final["flags"] == []
    assert state["rationale"]["citation"]["pass"]
    assert "Other knives having fixed blades" in final["rationale"]
    user = chat.calls[0]["messages"][1][1]
    assert "Commodity code 82119290: KNIVES > Knives > Other knives" in user
    assert "Subheading line: A. 821192: Knives > Other knives having fixed blades" in user
    assert set(state["timings"]) >= {"stage1_score", "stage2_score", "combine", "rationale"}
    assert state["tokens"]["rationale"] == [50, 30]


def test_single_child_top_hs6_needs_no_stage2_call():
    stage1 = top(("B", 0.7), ("C", 0.2), ("A", 0.1))  # 821191 then 010121: both single-child
    pipe, server, _ = a2({"stage1": stage1})
    state = run(pipe)
    assert len(server.requests) == 1
    assert state["final"]["candidates"][:2] == ["82119100", "01012100"]
    assert state["final"]["confidence"] == pytest.approx(0.7)


def test_floor_is_used_when_a_label_is_missing():
    pipe, _, _ = a2({"stage1": top(("A", 0.6), ("B", 0.3)), **STAGE2})
    s1 = run(pipe)["stage1"]
    assert s1["floored"] == ["C"] and s1["logprobs"][2] == pytest.approx(math.log(0.3))


def test_rationale_citation_failure_and_errors_never_change_the_choice():
    pipe, _, _ = a2(rationale={"quote": "ceramic blades", "explanation": "x"})
    state = run(pipe)
    assert not state["rationale"]["citation"]["pass"] and state["cn8"] == "82119290"
    pipe, _, _ = a2(rationale=ValueError("cut off"))
    state = run(pipe)
    assert state["cn8"] == "82119290" and "rationale_parse_error" in state["final"]["flags"]
    assert state["final"]["rationale"] == ""


class Reject:
    def __init__(self, bad):
        self.bad = bad

    def check(self, cn8):
        return {"valid": cn8 not in self.bad, "api_error": None}


def test_validator_falls_back_to_the_next_pair():
    pipe, _, chat = a2(validator=Reject({"82119290"}))
    state = run(pipe)
    assert state["cn8"] == "82119100" and "validator_fallback" in state["final"]["flags"]
    assert state["final"]["candidates"][0] == "82119100"
    assert state["final"]["confidence"] == pytest.approx(0.3 / 0.95)
    assert "Commodity code 82119100" in chat.calls[0]["messages"][1][1]


def test_temperature_changes_confidence_not_the_scored_hs6():
    pipe, server, _ = a2(temperature={"stage1": 2.0, "stage2": 1.0})
    state = run(pipe)
    assert len(server.requests) == 2
    p1 = scoring.renormalise([math.log(x) for x in (0.6, 0.3, 0.05)], 2.0)[0]
    assert state["final"]["confidence"] == pytest.approx(p1 * 0.75)


def test_rescore_matches_the_graph_and_feeds_temperature_groups():
    pipe, _, _ = a2()
    pred = A2.to_prediction("1", run(pipe))
    order, conf = A2.rescore(pred.details, 1.0, 1.0)
    assert order == [c.code for c in pred.candidates] and conf == pytest.approx(pred.confidence)
    g = A2.temperature_groups([pred.details], ["8211929000"])
    assert g["stage1"] == [(pred.details["stage1"]["logprobs"], 0)]
    assert g["stage2"] == [(pred.details["stage2"][0]["logprobs"], 1)]


def test_a1_graph_unchanged_by_a2():
    """A1 on the same parts still makes its two JSON calls and its own confidence."""
    tariff = make_tariff()
    s1 = {"choice": "A", "alternatives": ["B"], "confidence": 80,
          "rationale": "Matches 'other knives having fixed blades'."}  # fmt: skip
    s2 = {"choice": "B", "alternatives": ["A"], "confidence": 50, "rationale": "'Other' knives."}

    class Chat:
        def __init__(self):
            self.names = []

        def invoke(self, messages, response_format, max_tokens):
            name = response_format["json_schema"]["name"]
            self.names.append(name)
            return SimpleNamespace(content=json.dumps({"stage1": s1, "stage2": s2}[name]),
                                   usage_metadata={})  # fmt: skip

    cfg = {**A1.DEFAULTS, "m": 3, "labels": ["A", "B", "C"]}
    chat = Chat()
    state = (
        Pipeline(cfg, chat, FakeRetriever(), tariff, Validator(tariff))
        .build()
        .invoke({"item_id": "1", "raw": "a kitchen knife"})
    )
    assert chat.names == ["stage1", "stage2"]
    assert state["cn8"] == "82119290" and state["final"]["confidence"] == pytest.approx(0.4)
    assert "ranked" not in state and "stage1_score" not in state["timings"]


# --- config, metrics, server -------------------------------------------------------------


def test_load_config_refuses_a_changed_a1_config(tmp_path):
    cfg = json.loads(A2.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["base_config"] = {**cfg["base_config"], "sha256": "0" * 64}
    p = tmp_path / "a2.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="a1_pipeline.json changed"):
        A2.load_config(p)


def test_paired_review_difference():
    n = 200
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"item_id": [str(i) for i in range(n)], "cluster_id": np.arange(n) // 2,
                         "correct_8": rng.integers(0, 2, n).astype(float),
                         "confidence": rng.random(n), "abstain": False})  # fmt: skip
    same = M.paired_review_difference(base, base.copy(), n_resamples=200)
    assert same["diff"] == 0 and same["low"] == 0 and same["high"] == 0
    better = base.copy()
    better["correct_8"] = 1.0
    d = M.paired_review_difference(better, base, n_resamples=200)
    assert d["a"] == 1.0 and d["diff"] > 0 and d["low"] > 0


def server_up() -> bool:
    return ServerInfo(A1.DEFAULTS["client"]["base_url"]).up()


@pytest.mark.skipif(not server_up(), reason="llama-server is not running")
def test_real_server_returns_every_label():
    http = httpx.Client(base_url=A1.DEFAULTS["client"]["base_url"], timeout=120)
    opts = "\n".join(f"{lab}. option {i}" for i, lab in enumerate("ABCDEFGHIJ"))
    r = score_call(http, A1.DEFAULTS["client"], "Reply with the label of the best option only.",
                   f"Goods: a steel knife\nOptions:\n{opts}\nAnswer with the label only.", 200)  # fmt: skip
    d = scoring.label_logprobs(r["top"], list("ABCDEFGHIJ"))
    assert r["token"] in "ABCDEFGHIJ" and d["floored"] == [] and d["outside_mass"] < 0.05
