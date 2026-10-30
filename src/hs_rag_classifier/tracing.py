"""Opt-in OpenTelemetry tracing of the LangGraph pipeline to a local Phoenix viewer.

Off by default, for dev, smoke and demo runs only: evaluation numbers are measured with it off,
and a test split refuses it (eval/runner.py check_test_guard). The entry points call enable(),
which does nothing unless given their --trace flag or HS_TRACING=1 is set. The tracing packages
(dependency group "tracing") are imported only there, so a normal install runs without them.

LangChainInstrumentor gives one span per graph node, and an LLM span with token counts for each
ChatOpenAI call (A1's decisions, the A2 rationale, the profile). A2's scoring calls go through
raw httpx, which it cannot see, so score_call opens its own LLM span with llm_span().

Spans carry the graph state and the prompts. That state never holds a label or a ruling
justification: the arms pass only item_id and the description, and the loaders never read
justifications.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

ENV_FLAG = "HS_TRACING"
ENV_ENDPOINT = "PHOENIX_COLLECTOR_ENDPOINT"
DEFAULT_ENDPOINT = "http://localhost:6006"
PROJECT = "hs-rag-classifier"
TEST_REFUSED = "tracing is for dev, smoke and demo runs, never a test split"

_tracer: Any = None


def requested(flag: bool = False) -> bool:
    return flag or _tracer is not None or os.environ.get(ENV_FLAG) == "1"


def active() -> bool:
    return _tracer is not None


def _instrument(provider: Any) -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    global _tracer
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _tracer = provider.get_tracer("hs_rag_classifier")


def enable(flag: bool = False, endpoint: str | None = None) -> bool:
    """Send traces to Phoenix when asked to (flag or HS_TRACING=1); True if tracing is on."""
    if _tracer is not None:
        return True
    if not requested(flag):
        return False
    try:
        import openinference.instrumentation.langchain  # noqa: F401
        from phoenix.otel import register
    except ImportError as exc:
        raise SystemExit(
            f"Tracing was requested but its packages are missing ({exc.name}). Install them with "
            "uv sync --group llm --group ml --group tracing"
        ) from exc
    base = endpoint or os.environ.get(ENV_ENDPOINT, DEFAULT_ENDPOINT)
    # Its own provider, not the global one, which MLflow may use; batched, so export is off the
    # request path. The provider flushes at exit.
    provider = register(
        endpoint=f"{base.rstrip('/')}/v1/traces",
        project_name=PROJECT,
        protocol="http/protobuf",
        batch=True,
        set_global_tracer_provider=False,
        verbose=False,
    )
    _instrument(provider)
    return True


class LlmCall:
    """Handle on an llm_span; does nothing when tracing is off."""

    def __init__(self, span: Any = None):
        self.span = span

    def finish(self, output: str, tokens_in: int | None, tokens_out: int | None) -> None:
        if self.span is None:
            return
        self.span.set_attributes(
            {
                "output.value": output,
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": output,
                "llm.token_count.prompt": tokens_in or 0,
                "llm.token_count.completion": tokens_out or 0,
                "llm.token_count.total": (tokens_in or 0) + (tokens_out or 0),
            }
        )


@contextmanager
def llm_span(name: str, model: str, system: str, user: str) -> Iterator[LlmCall]:
    """An OpenInference LLM span under the running graph node, for a call LangChain cannot see."""
    if _tracer is None:
        yield LlmCall()
        return
    from openinference.instrumentation.langchain import get_current_span
    from opentelemetry import trace

    # The instrumentor keeps its spans out of the OpenTelemetry context, so ask it for the parent
    parent = get_current_span()
    context = trace.set_span_in_context(parent) if parent else None
    attributes = {
        "openinference.span.kind": "LLM",
        "llm.model_name": model,
        "input.value": user,
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": system,
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.content": user,
    }
    with _tracer.start_as_current_span(name, context=context, attributes=attributes) as span:
        yield LlmCall(span)
