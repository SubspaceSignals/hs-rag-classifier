"""The decider client: llama-server's OpenAI-compatible endpoint through LangChain's ChatOpenAI.

Every call: temperature 0, a fixed seed, prompt reuse off (CACHE_PROMPT), thinking disabled
through the chat template (chat_template_kwargs.enable_thinking = false), and a JSON schema as
response_format, so
llama-server constrains decoding to the schema and every reply parses. ServerInfo reads the
server's own tokenizer (/tokenize), chat template (/apply-template) and build (/props).

One-slot guard: check_server_slots() stops a run unless llama-server reports exactly one slot
(total_slots in /props), and returns the run's MLflow tags (server_slots, cache_prompt). The
dev override (HS_ALLOW_MULTISLOT=1 or the entry points' --allow-multislot) continues with a
warning and the tag multislot_override; a test split refuses it (eval/runner.py
check_test_guard), as it refuses tracing.

LoRA adapters (off unless a config declares one): score_call and make_chat take an optional
per-request lora list (llama-server's "lora" field, [{"id": 0, "scale": s}]); with none, the
request has no "lora" field and is byte-identical to a build without adapters.
check_server_adapters() runs next to the one-slot guard: a config's adapter block {"path",
"sha256"} must match the file on disk and be the one adapter llama-server lists in
/lora-adapters; with no block, the server must list none. Any mismatch stops the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import httpx

from hs_rag_classifier import tracing

BASE_URL = "http://localhost:8080"
# Sent as cache_prompt on every generation request. A2's output depended on llama-server's slot
# and prompt-cache state; False with a -np 1 server gives bit-identical runs (neither does alone).
CACHE_PROMPT = False
# The decider server's start command (why each flag: pipeline/tune.py SERVER)
SERVER_COMMAND = (
    "~/llama.cpp/build/bin/llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL "
    "-ngl 99 -c 16384 --port 8080 --jinja --cache-ram 0 -np 1"
)
ENV_ALLOW_MULTISLOT = "HS_ALLOW_MULTISLOT"
MULTISLOT_TEST_REFUSED = "the multi-slot override is for dev runs, never a test split"

_allow_multislot = False


class ServerSlotsError(RuntimeError):
    """llama-server does not have exactly one slot, or its slot count cannot be read."""


class AdapterError(RuntimeError):
    """llama-server's LoRA adapters, the config's adapter block and the file on disk disagree."""


def allow_multislot(flag: bool = False) -> None:
    """The entry points' --allow-multislot; HS_ALLOW_MULTISLOT=1 does the same."""
    global _allow_multislot
    _allow_multislot = _allow_multislot or flag


def multislot_requested(flag: bool = False) -> bool:
    return flag or _allow_multislot or os.environ.get(ENV_ALLOW_MULTISLOT) == "1"


def slot_count(props: dict) -> int | None:
    """llama-server's slot count (-np) from its /props reply; None when it is not there."""
    n = props.get("total_slots") if isinstance(props, dict) else None
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def check_server_slots(base_url: str = BASE_URL,
                       transport: httpx.BaseTransport | None = None) -> dict[str, str]:  # fmt: skip
    """Stop unless llama-server has one slot (see the module docstring); the run's MLflow tags."""
    where = f"llama-server at {base_url}"
    try:
        n = slot_count(ServerInfo(base_url, transport).props())
        problem = None if n == 1 else f"{where} reports {n} slots"
        if n is None:
            problem = f"{where} does not report its slot count (/props total_slots)"
    except (httpx.HTTPError, ValueError) as exc:
        n = None
        problem = f"cannot read the slot count from {where} ({type(exc).__name__}: {exc})"
    tags = {"server_slots": "unknown" if n is None else str(n),
            "cache_prompt": str(CACHE_PROMPT).lower()}  # fmt: skip
    if problem is None:
        return tags
    msg = (f"{problem}. Runs are bit-reproducible only with one slot (-np 1) and "
           f"cache_prompt=false. Start the server with:\n  {SERVER_COMMAND}")  # fmt: skip
    if not multislot_requested():
        raise ServerSlotsError(f"{msg}\n(Dev only: {ENV_ALLOW_MULTISLOT}=1 or --allow-multislot "
                               "continues anyway; never on a test split.)")  # fmt: skip
    warnings.warn(f"\n{'!' * 78}\nMULTI-SLOT OVERRIDE: continuing anyway. This run is NOT "
                  f"bit-reproducible and is tagged multislot_override=true.\n{msg}\n{'!' * 78}",
                  RuntimeWarning, stacklevel=2)  # fmt: skip
    return {**tags, "multislot_override": "true"}


def lora_request(scale: float) -> list[dict[str, Any]]:
    """The per-request "lora" field for the one declared adapter (llama-server id 0)."""
    return [{"id": 0, "scale": float(scale)}]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_server_adapters(base_url: str = BASE_URL, adapter: dict[str, str] | None = None,
                          transport: httpx.BaseTransport | None = None) -> dict[str, str]:  # fmt: skip
    """Stop unless the server's adapters match the config's block (module docstring); tags."""
    where = f"llama-server at {base_url}"
    try:
        loras = ServerInfo(base_url, transport).lora_adapters()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise AdapterError(f"cannot read /lora-adapters from {where} "
                           f"({type(exc).__name__}: {exc})") from exc  # fmt: skip
    listed = [str(x.get("path")) for x in loras]
    if adapter is None:
        if loras:
            raise AdapterError(f"{where} has LoRA adapters the config does not declare: {listed}")
        return {}
    path = Path(adapter["path"]).expanduser()
    if not path.is_file():
        raise AdapterError(f"the config's adapter {path} is not a file")
    got = _file_sha256(path)
    if got != adapter["sha256"]:
        raise AdapterError(f"{path} has SHA-256 {got}, the config declares {adapter['sha256']}")
    _check_listed(where, loras, path)
    return {"lora_adapter": path.name, "lora_adapter_sha256": got}


def _check_listed(where: str, loras: list[dict], path: Path) -> dict:
    """The one listed adapter, which must be path; AdapterError otherwise. No hashing."""
    listed = [str(x.get("path")) for x in loras]
    if not loras:
        raise AdapterError(f"the config declares the adapter {path}, but {where} lists none")
    if len(loras) != 1 or Path(listed[0]).expanduser().resolve() != path.resolve():
        raise AdapterError(f"{where} lists {listed}; the config declares exactly {path}")
    return loras[0]


def server_adapter(base_url: str, path: str | Path,
                   transport: httpx.BaseTransport | None = None) -> dict:  # fmt: skip
    """The server's one /lora-adapters entry ({"id", "path", ...}), which must be path. Reads
    the listing only and never hashes the file (the service hashes it once at start-up)."""
    where = f"llama-server at {base_url}"
    try:
        loras = ServerInfo(base_url, transport).lora_adapters()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise AdapterError(f"cannot read /lora-adapters from {where} "
                           f"({type(exc).__name__}: {exc})") from exc  # fmt: skip
    return _check_listed(where, loras, Path(path).expanduser())


def make_chat(client: dict[str, Any], base_url: str = BASE_URL,
              lora: list[dict[str, Any]] | None = None):  # fmt: skip
    """The chat client; lora (lora_request) is sent on every call when given, else not at all."""
    from langchain_openai import ChatOpenAI

    extra_body: dict[str, Any] = {
        "cache_prompt": CACHE_PROMPT,
        "chat_template_kwargs": {"enable_thinking": client["enable_thinking"]},
    }
    if lora is not None:
        extra_body["lora"] = lora
    return ChatOpenAI(
        base_url=f"{base_url}/v1",
        api_key="not-needed",  # llama-server takes no key; nothing secret here
        model=client.get("model", "qwen3.5-9b"),
        temperature=client["temperature"],
        seed=client["seed"],
        timeout=client.get("timeout_s", 120),
        max_retries=client.get("max_retries", 2),
        extra_body=extra_body,
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
        r = self.client.get("/props")
        r.raise_for_status()
        return r.json()

    def lora_adapters(self) -> list[dict]:
        r = self.client.get("/lora-adapters")
        r.raise_for_status()
        out = r.json()
        if not isinstance(out, list):
            raise TypeError("/lora-adapters did not return a list")
        return out

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
               top_logprobs: int, name: str = "score_call",
               lora: list[dict[str, Any]] | None = None) -> dict:  # fmt: skip
    """A2 scoring: one generated token, no grammar or schema, with the top_logprobs most likely
    next tokens (log-probabilities before sampling). {"token", "top", "tokens_in", "tokens_out",
    "seconds"}; top is [{"token", "logprob"}, ...]. lora (lora_request): sent only when given."""
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 1,
        "temperature": client["temperature"],
        "seed": client["seed"],
        "cache_prompt": CACHE_PROMPT,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": {"enable_thinking": client["enable_thinking"]},
    }
    if lora is not None:
        body["lora"] = lora
    with tracing.llm_span(name, client.get("model", "qwen3.5-9b"), system, user) as call:
        t0 = time.perf_counter()
        r = http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        seconds = time.perf_counter() - t0
        data = r.json()
        step = data["choices"][0]["logprobs"]["content"][0]
        usage = data.get("usage") or {}
        call.finish(step["token"], usage.get("prompt_tokens"), usage.get("completion_tokens"))
    return {"token": step["token"],
            "top": [{"token": t["token"], "logprob": t["logprob"]} for t in step["top_logprobs"]],
            "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
            "seconds": seconds}  # fmt: skip
