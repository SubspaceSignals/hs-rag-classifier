"""cache_prompt on every generation request: A2 scoring, the A2 rationale, A1's decisions and the
profile, checked in the JSON that leaves the client. A real ChatOpenAI talks to a mock transport;
needs neither a GPU nor llama-server."""

import functools
import json

import httpx
import langchain_openai
import pytest
from test_a2_pipeline import RATIONALE, a2
from test_pipeline import pipeline

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.pipeline import llm
from hs_rag_classifier.pipeline.llm import call_json, make_chat, score_call

CLIENT = A1.DEFAULTS["client"]
REPLIES = {
    "profile": {"what": "knife", "material": "steel", "function": "cut", "form": "whole",
                "use": "kitchen"},
    "stage1": {"choice": "A", "alternatives": [], "confidence": 80, "rationale": "'Knives'"},
    "stage2": {"choice": "A", "alternatives": [], "confidence": 80, "rationale": "'Other'"},
    "rationale": RATIONALE,
}  # fmt: skip


def wired_chat(monkeypatch, bodies: list[dict]):
    """make_chat's own ChatOpenAI, its HTTP going to a handler that records each JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        reply = REPLIES[body["response_format"]["json_schema"]["name"]]
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(reply)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})  # fmt: skip

    client = httpx.Client(transport=httpx.MockTransport(handler))
    real = langchain_openai.ChatOpenAI
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", functools.partial(real, http_client=client))
    return make_chat(CLIENT, "http://fake")


def test_the_default_is_off():
    assert llm.CACHE_PROMPT is False


def test_score_call_sends_it():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        step = {"token": "A", "logprob": -0.1, "top_logprobs": [{"token": "A", "logprob": -0.1}]}
        return httpx.Response(200, json={"choices": [{"logprobs": {"content": [step]}}]})

    http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(handler))
    score_call(http, CLIENT, "s", "u", 5)
    assert bodies[0]["cache_prompt"] is False


def test_chat_sends_it_at_the_top_level_of_the_body(monkeypatch):
    bodies = []
    r = call_json(wired_chat(monkeypatch, bodies), "s", "u", {"type": "object"}, "profile", 20)
    assert r["data"]["what"] == "knife"
    assert bodies[0]["cache_prompt"] is False
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}  # still sent
    assert "extra_body" not in bodies[0]


def test_every_a2_request_sends_it(monkeypatch):
    pipe, server, _ = a2()
    chat_bodies = []
    pipe.chat = wired_chat(monkeypatch, chat_bodies)
    state = pipe.build().invoke({"item_id": "1", "raw": "a kitchen knife"})
    assert state["rationale"]["parsed"]
    assert len(server.requests) == 2 and len(chat_bodies) == 1  # stage 1, stage 2, rationale
    assert all(b["cache_prompt"] is False for b in server.requests + chat_bodies)


def test_every_a1_request_sends_it(monkeypatch):
    bodies = []
    pipe, _ = pipeline(REPLIES, profile=True)
    pipe.chat = wired_chat(monkeypatch, bodies)
    state = pipe.build().invoke({"item_id": "1", "raw": "a kitchen knife"})
    assert state["profile"]["what"] == "knife"
    names = [b["response_format"]["json_schema"]["name"] for b in bodies]
    assert names == ["profile", "stage1", "stage2"]
    assert all(b["cache_prompt"] is False for b in bodies)


@pytest.mark.parametrize("value", [True, False])
def test_the_constant_switches_every_path(monkeypatch, value):
    monkeypatch.setattr(llm, "CACHE_PROMPT", value)
    pipe, server, _ = a2()
    chat_bodies = []
    pipe.chat = wired_chat(monkeypatch, chat_bodies)
    pipe.build().invoke({"item_id": "1", "raw": "a kitchen knife"})
    assert {b["cache_prompt"] for b in server.requests + chat_bodies} == {value}
