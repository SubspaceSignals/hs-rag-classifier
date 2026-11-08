"""A4 candidate on val: optional per-request LoRA scales and the adapter guard.

With no adapter block every request is byte-identical to the requests captured before LoRA
support existed (fixtures/a2_requests_before_lora.json: A2 order-averaged, four scoring calls and
the rationale) and every existing config_hash is unchanged. With the block, the scoring calls
carry the scoring scale and the rationale call the rationale scale, and nothing else changes. The
adapter guard refuses every mismatch between config, file on disk and /lora-adapters. The
a4_candidate arm adds its candidate and adapter SHA-256 to details and MLflow params and is
refused on a test split. No GPU, no llama-server."""

import functools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import httpx
import langchain_openai
import pytest
from mlflow.tracking import MlflowClient
from test_a2_order_averaging import averaging
from test_a2_pipeline import RATIONALE
from test_a3_pipeline import FakeB1ex
from test_eval_harness import FixedArm, data, make_set  # noqa: F401
from test_eval_harness import run as run_harness

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.arms import a3_pipeline as A3
from hs_rag_classifier.arms import a4_candidate as A4
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.arms import config_hash
from hs_rag_classifier.eval.items import REPO_ROOT
from hs_rag_classifier.pipeline import graph_a2, llm
from hs_rag_classifier.pipeline.graph_a2 import A2Pipeline

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "a2_requests_before_lora.json")
                    .read_text(encoding="utf-8"))  # fmt: skip
HASHES = {"a1": "7415d4a5e70c", "a2": "a26e01c99d65", "a2_order_avg": "aae040fc9763",
          "a3": "57c54a509634"}  # fmt: skip
SHA = "ab" * 32


def recorded(pipe, server, chat_lora=None):
    """Run the graph, recording the raw bytes of every scoring and rationale request."""
    scoring, chat = [], []
    inner = server.handler

    def score(req):
        scoring.append(req.content.decode())
        return inner(req)

    def rationale(req):
        chat.append(req.content.decode())
        msg = {"role": "assistant", "content": json.dumps(RATIONALE)}
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})  # fmt: skip

    pipe.http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(score))
    real = langchain_openai.ChatOpenAI
    client = httpx.Client(transport=httpx.MockTransport(rationale))
    langchain_openai.ChatOpenAI = functools.partial(real, http_client=client)
    try:
        pipe.chat = llm.make_chat(A1.DEFAULTS["client"], "http://fake", lora=chat_lora)
    finally:
        langchain_openai.ChatOpenAI = real
    pipe.build().invoke({"item_id": "1", "raw": "a kitchen knife"})
    return scoring, chat


# --- requests ----------------------------------------------------------------------------


def test_without_an_adapter_requests_are_byte_identical_to_before():
    pipe, server = averaging()
    scoring, chat = recorded(pipe, server)
    assert scoring == GOLDEN["scoring"] and chat == GOLDEN["rationale"]
    assert len(scoring) == 4 and all('"lora"' not in b for b in scoring + chat)


def with_adapter(**a2kw):
    """averaging()'s pipeline, built with an adapter block and LoRA scales in its config."""
    pipe, server = averaging()
    cfg = {**pipe.cfg, "a2": {**pipe.cfg["a2"], **a2kw},
           "adapter": {"path": "x.gguf", "sha256": SHA}}  # fmt: skip
    again = A2Pipeline(cfg, pipe.chat, pipe.retriever, pipe.tariff, pipe.validator, pipe.http,
                       pipe.paths)  # fmt: skip
    return again, server


def test_with_scales_only_the_lora_field_is_added():
    pipe, server = with_adapter(lora_scales={"scoring": 1.0, "rationale": 0.0})
    assert pipe.scoring_lora == [{"id": 0, "scale": 1.0}]
    scoring, chat = recorded(pipe, server, chat_lora=llm.lora_request(0.0))
    assert len(scoring) == 4 and len(chat) == 1
    for got, before, scale in zip(scoring + chat, GOLDEN["scoring"] + GOLDEN["rationale"],
                                  [1.0] * 4 + [0.0], strict=True):  # fmt: skip
        g, b = json.loads(got), json.loads(before)
        assert g.pop("lora") == [{"id": 0, "scale": scale}]
        assert g == b  # nothing else in the request changes


def test_pipeline_reads_the_scoring_scale_and_refuses_half_a_setting():
    pipe, _ = averaging()
    assert pipe.scoring_lora is None
    with pytest.raises(ValueError, match="both or neither"):
        averaging(lora_scales={"scoring": 1.0, "rationale": 0.0})  # no adapter block
    cfg = {"a2": {"lora_scales": {"scoring": 0.5, "rationale": 0.0}},
           "adapter": {"path": "x", "sha256": SHA}}  # fmt: skip
    assert graph_a2.lora_scales(cfg) == {"scoring": 0.5, "rationale": 0.0}
    with pytest.raises(ValueError, match="both or neither"):
        graph_a2.lora_scales({"a2": {}, "adapter": {"path": "x", "sha256": SHA}})
    with pytest.raises(ValueError, match="exactly"):
        graph_a2.lora_scales({"a2": {"lora_scales": {"scoring": 1.0}}, "adapter": {}})


def test_score_call_and_chat_send_lora_only_when_given():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        step = {"token": "A", "logprob": -0.1, "top_logprobs": [{"token": "A", "logprob": -0.1}]}
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [step]}}]})

    http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    llm.score_call(http, A1.DEFAULTS["client"], "s", "u", 5)
    llm.score_call(http, A1.DEFAULTS["client"], "s", "u", 5, lora=llm.lora_request(1))
    assert "lora" not in bodies[0] and bodies[1]["lora"] == [{"id": 0, "scale": 1.0}]
    assert "lora" not in llm.make_chat(A1.DEFAULTS["client"]).extra_body
    chat = llm.make_chat(A1.DEFAULTS["client"], lora=llm.lora_request(0))
    assert chat.extra_body["lora"] == [{"id": 0, "scale": 0.0}]


def test_build_pipeline_gives_the_rationale_chat_its_own_scale(monkeypatch):
    parts = SimpleNamespace(chat="a1 chat", retriever=None, tariff=None, validator=None,
                            evidence_index=None, server_tags={})  # fmt: skip
    monkeypatch.setattr(A1, "build_pipeline", lambda *a, **k: parts)
    monkeypatch.setattr("hs_rag_classifier.pipeline.options.load_cn8_paths", lambda root: {})
    base = averaging()[0].cfg
    assert A2.build_pipeline(base).chat == "a1 chat"
    scales = {"scoring": 1.0, "rationale": 0.0}
    cfg = {**base, "a2": {**base["a2"], "lora_scales": scales},
           "adapter": {"path": "x.gguf", "sha256": SHA}}  # fmt: skip
    pipe = A2.build_pipeline(cfg)
    assert pipe.chat.extra_body["lora"] == [{"id": 0, "scale": 0.0}]
    assert pipe.scoring_lora == [{"id": 0, "scale": 1.0}]


# --- adapter guard -----------------------------------------------------------------------


def lora_server(listed):
    def handler(request):
        assert request.url.path == "/lora-adapters"
        return httpx.Response(200, json=listed)

    return httpx.MockTransport(handler)


def test_adapter_guard(tmp_path):
    gguf = tmp_path / "a.gguf"
    gguf.write_bytes(b"adapter")
    good = {"path": str(gguf), "sha256": sha256(gguf)}
    entry = {"id": 0, "path": str(gguf), "scale": 1.0}
    assert llm.check_server_adapters("http://fake", None, lora_server([])) == {}
    tags = llm.check_server_adapters("http://fake", good, lora_server([entry]))
    assert tags == {"lora_adapter": "a.gguf", "lora_adapter_sha256": good["sha256"]}
    other = tmp_path / "b.gguf"
    other.write_bytes(b"other")
    cases = [
        (None, [entry], "does not declare"),
        (good, [], "lists none"),
        ({**good, "sha256": SHA}, [entry], "SHA-256"),
        (good, [{**entry, "path": str(other)}], "declares exactly"),
        (good, [entry, {**entry, "id": 1, "path": str(other)}], "declares exactly"),
        ({"path": str(tmp_path / "missing.gguf"), "sha256": SHA}, [entry], "not a file"),
    ]
    for adapter, listed, msg in cases:
        with pytest.raises(llm.AdapterError, match=msg):
            llm.check_server_adapters("http://fake", adapter, lora_server(listed))
    broken = httpx.MockTransport(lambda r: httpx.Response(404))
    with pytest.raises(llm.AdapterError, match="cannot read"):
        llm.check_server_adapters("http://fake", None, broken)


# --- config ------------------------------------------------------------------------------


def test_existing_config_hashes_are_unchanged():
    got = {"a1": A1.load_config(), "a2": A2.load_config(),
           "a2_order_avg": A2.load_candidate_config(), "a3": A3.load_config()}  # fmt: skip
    assert {k: config_hash(v) for k, v in got.items()} == HASHES
    assert all("adapter" not in c and "lora_scales" not in c.get("a2", {}) for c in got.values())


def test_candidate_config_is_a3_with_t1_scales_and_the_adapter(tmp_path):
    raw = json.loads(A4.CONFIG_PATH.read_text(encoding="utf-8"))
    pin = raw["base_config"]
    assert pin == {"path": "configs/a3_pipeline.json",
                   "sha256": sha256(REPO_ROOT / "configs/a3_pipeline.json")}  # fmt: skip
    assert raw["candidates"] == ["step_00250", "step_00500", "final"]
    assert "adapter" not in raw  # supplied per candidate
    cfg = A4.load_config(tmp_path / "x.gguf", SHA, "step_00500")
    a3 = A3.load_config()
    t = cfg["a2"]["temperature"]
    assert t["stage1"] == 1.0 and t["stage2"] == 1.0 and t["mode"] == "per_stage"
    assert cfg["a2"]["lora_scales"] == {"scoring": 1.0, "rationale": 0.0}
    assert cfg["adapter"] == {"path": str((tmp_path / "x.gguf").resolve()), "sha256": SHA}
    assert cfg["a4"]["candidate"] == "step_00500"
    rest = {k: v for k, v in cfg["a2"].items() if k not in ("temperature", "lora_scales")}
    assert rest == {k: v for k, v in a3["a2"].items() if k != "temperature"}
    assert rest["order_averaging"] is True
    assert {k: v for k, v in cfg.items() if k not in ("a2", "adapter", "a4")} == {
        k: v for k, v in a3.items() if k != "a2"
    }
    other = A4.load_config(tmp_path / "y.gguf", SHA, "step_00500")
    assert config_hash(other) != config_hash(cfg)  # the adapter is in the config_hash


def test_candidate_config_refusals(tmp_path):
    with pytest.raises(ValueError, match="not one of"):
        A4.load_config(tmp_path / "x.gguf", SHA, "step_00100")
    with pytest.raises(ValueError, match="hex"):
        A4.load_config(tmp_path / "x.gguf", "ABC", "final")
    raw = json.loads(A4.CONFIG_PATH.read_text(encoding="utf-8"))
    raw["base_config"] = {**raw["base_config"], "sha256": "0" * 64}
    p = tmp_path / "a4.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="a3_pipeline.json changed"):
        A4.load_config(tmp_path / "x.gguf", SHA, "final", path=p)


# --- arm, guard, CLI ---------------------------------------------------------------------


def candidate_arm(tmp_path):
    pipe, _ = averaging(temperature={"stage1": 1.0, "stage2": 1.0})
    cfg = {**pipe.cfg, "b1ex": {"k": 1}, "a3": {},
           "adapter": {"path": str(tmp_path / "x.gguf"), "sha256": SHA},
           "a4": {"candidate": "final"}}  # fmt: skip
    return A4.A4CandidateArm(cfg, pipeline=pipe, b1ex=FakeB1ex("8211929000"))


def test_arm_records_candidate_and_adapter(tmp_path):
    arm = candidate_arm(tmp_path)
    p = arm.predict([SimpleNamespace(item_id="1", text="a kitchen knife")])[0]
    assert p.details["a4"] == {"candidate": "final", "adapter_sha256": SHA,
                               "adapter_path": str(tmp_path / "x.gguf")}  # fmt: skip
    assert "a3" in p.details and "orders" in p.details["stage1"]
    assert arm.run_params == {"candidate_step": "final", "adapter_sha256": SHA}
    with pytest.raises(ValueError, match="load_config"):
        A4.A4CandidateArm({"a2": {}})


def test_candidate_is_refused_on_test_and_logs_its_params(data, tmp_path):  # noqa: F811
    git = {"commit": "x", "dirty": False}
    arm = A4.A4CandidateArm.__new__(A4.A4CandidateArm)
    with pytest.raises(R.GuardError, match="validation-only"):
        R.check_test_guard(arm, SimpleNamespace(split="test"), git, None)

    class ParamArm(FixedArm):
        run_params: ClassVar[dict[str, str]] = {"candidate_step": "final", "adapter_sha256": SHA}

    plain = run_harness(FixedArm(), make_set(data, split="val"), tmp_path / "a")
    extra = run_harness(ParamArm(), make_set(data, split="val"), tmp_path / "b")
    pa = MlflowClient(R.tracking_uri(tmp_path / "a" / "mlflow")).get_run(plain["run_id"])
    pb = MlflowClient(R.tracking_uri(tmp_path / "b" / "mlflow")).get_run(extra["run_id"])
    assert "candidate_step" not in pa.data.params and "adapter_sha256" not in pa.data.params
    assert pb.data.params["candidate_step"] == "final" and pb.data.params["adapter_sha256"] == SHA
    rest = {k: v for k, v in pb.data.params.items() if k not in ParamArm.run_params}
    assert rest == pa.data.params  # the harness's own params are unchanged


def test_make_arm_options():
    with pytest.raises(ValueError, match="takes no adapter"):
        R.make_arm("a3", set(), adapter="x.gguf")
    with pytest.raises(ValueError, match="needs --adapter"):
        R.make_arm("a4_candidate", set(), adapter="x.gguf")
    assert R.ARM_OPTIONS == {"a4_candidate": ("adapter", "adapter_sha256", "candidate")}
