"""The decider client: llama-server's OpenAI-compatible endpoint through LangChain's ChatOpenAI.

Every call: temperature 0, a fixed seed, thinking disabled through the chat template
(chat_template_kwargs.enable_thinking = false), and a JSON schema as response_format, so
llama-server constrains decoding to the schema and every reply parses. ServerInfo reads the
server's own tokenizer (/tokenize), chat template (/apply-template) and build (/props).
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

BASE_URL = "http://localhost:8080"


def make_chat(client: dict[str, Any], base_url: str = BASE_URL):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",  # llama-server takes no key; nothing secret here
        model=client.get("model", "qwen3.5-9b"),
        temperature=client["temperature"],
        seed=client["seed"],
        timeout=client.get("timeout_s", 120),
        max_retries=client.get("max_retries", 2),
        extra_body={"chat_template_kwargs": {"enable_thinking": client["enable_thinking"]}},
    )


def decision_schema(labels: list[str], max_rationale: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": labels},
            "alternatives": {
                "type": "array",
                "items": {"type": "string", "enum": labels},
                "maxItems": 2,
            },
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "rationale": {"type": "string", "maxLength": max_rationale},
        },
        "required": ["choice", "alternatives", "confidence", "rationale"],
        "additionalProperties": False,
    }


PROFILE_FIELDS = ("what", "material", "function", "form", "use")


def profile_schema(max_field: int) -> dict:
    return {
        "type": "object",
        "properties": {f: {"type": "string", "maxLength": max_field} for f in PROFILE_FIELDS},
        "required": list(PROFILE_FIELDS),
        "additionalProperties": False,
    }


def call_json(chat, system: str, user: str, schema: dict, name: str, max_tokens: int) -> dict:
    """One schema-constrained call: {"data" (dict or None), "raw", "tokens_in", "tokens_out",
    "seconds", "error"}. A reply that still fails to parse is returned with data None."""
    fmt = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
    t0 = time.perf_counter()
    msg = chat.invoke([("system", system), ("user", user)], response_format=fmt,
                      max_tokens=max_tokens)  # fmt: skip
    seconds = time.perf_counter() - t0
    usage = getattr(msg, "usage_metadata", None) or {}
    out = {"raw": msg.content, "tokens_in": usage.get("input_tokens"),
           "tokens_out": usage.get("output_tokens"), "seconds": seconds, "error": None}  # fmt: skip
    try:
        data = json.loads(msg.content)
        if not isinstance(data, dict):
            raise TypeError("reply is not a JSON object")
        out["data"] = data
    except (ValueError, TypeError) as exc:
        out["data"], out["error"] = None, f"{type(exc).__name__}: {exc}"[:200]
    return out


class ServerInfo:
    """llama-server's tokenizer, chat template and build details."""

    def __init__(self, base_url: str = BASE_URL, transport: httpx.BaseTransport | None = None):
        self.client = httpx.Client(base_url=base_url, timeout=60.0, transport=transport)

    def up(self) -> bool:
        try:
            return self.client.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def props(self) -> dict:
        return self.client.get("/props").json()

    def tokenize(self, text: str) -> list[int]:
        r = self.client.post("/tokenize", json={"content": text, "add_special": False})
        r.raise_for_status()
        return [t if isinstance(t, int) else t["id"] for t in r.json()["tokens"]]

    def prompt_tokens(self, system: str, user: str, enable_thinking: bool = False) -> int:
        """Tokens of the full chat prompt as the server renders it (template included)."""
        r = self.client.post("/apply-template", json={
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": enable_thinking}})  # fmt: skip
        r.raise_for_status()
        return len(self.tokenize(r.json()["prompt"]))


def score_call(http: httpx.Client, client: dict[str, Any], system: str, user: str,
               top_logprobs: int) -> dict:  # fmt: skip
    """A2 scoring: one generated token, no grammar or schema, with the top_logprobs most likely
    next tokens (log-probabilities before sampling). {"token", "top", "tokens_in", "tokens_out",
    "seconds"}; top is [{"token", "logprob"}, ...]."""
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 1,
        "temperature": client["temperature"],
        "seed": client["seed"],
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": {"enable_thinking": client["enable_thinking"]},
    }
    t0 = time.perf_counter()
    r = http.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    seconds = time.perf_counter() - t0
    data = r.json()
    step = data["choices"][0]["logprobs"]["content"][0]
    usage = data.get("usage") or {}
    return {"token": step["token"],
            "top": [{"token": t["token"], "logprob": t["logprob"]} for t in step["top_logprobs"]],
            "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
            "seconds": seconds}  # fmt: skip
