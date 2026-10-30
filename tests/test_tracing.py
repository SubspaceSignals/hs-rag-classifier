"""Tracing: off by default, imports lazy, a clear error without the packages, refused on a test
split. Needs neither Phoenix nor a GPU. The instrumented-graph test runs the A2 graph on the fakes
in test_a2_pipeline with spans kept in memory, and skips without the "tracing" group."""

import json
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from hs_rag_classifier import tracing
from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.eval import __main__ as cli
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.pipeline.llm import score_call

LAZY_CHECK = """
import sys
import hs_rag_classifier.tracing, hs_rag_classifier.pipeline.llm
import hs_rag_classifier.pipeline.graph_a2, hs_rag_classifier.arms.a2_pipeline
loaded = [m for m in sys.modules if m.split(".")[0] in ("phoenix", "openinference")]
assert not loaded, loaded
"""


@pytest.fixture(autouse=True)
def off(monkeypatch):
    monkeypatch.delenv(tracing.ENV_FLAG, raising=False)
    monkeypatch.setattr(tracing, "_tracer", None)


def fake_server(request: httpx.Request) -> httpx.Response:
    step = {"token": "A", "logprob": -0.1, "top_logprobs": [{"token": "A", "logprob": -0.1}]}
    return httpx.Response(200, json={"choices": [{"logprobs": {"content": [step]}}],
                                     "usage": {"prompt_tokens": 300, "completion_tokens": 1}})  # fmt: skip


def http() -> httpx.Client:
    return httpx.Client(base_url="http://fake", transport=httpx.MockTransport(fake_server))


def test_disabled_is_a_no_op():
    assert not tracing.requested()
    assert tracing.enable() is False
    assert not tracing.active()
    with tracing.llm_span("x", "m", "s", "u") as call:
        call.finish("A", 1, 1)
    assert call.span is None
    assert score_call(http(), A1.DEFAULTS["client"], "s", "u", 5)["tokens_in"] == 300


@pytest.mark.parametrize("value,on", [("1", True), ("0", False), ("", False), ("true", False)])
def test_only_hs_tracing_1_requests_it(monkeypatch, value, on):
    monkeypatch.setenv(tracing.ENV_FLAG, value)
    assert tracing.requested() is on


def test_imports_are_lazy():
    subprocess.run([sys.executable, "-c", LAZY_CHECK], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("missing", ["phoenix.otel", "openinference.instrumentation.langchain"])
def test_enabling_without_the_packages_says_what_to_install(monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)  # import raises ImportError
    with pytest.raises(SystemExit, match="--group tracing"):
        tracing.enable(True)
    assert not tracing.active()


def test_env_flag_alone_enables(monkeypatch):
    monkeypatch.setenv(tracing.ENV_FLAG, "1")
    monkeypatch.setitem(sys.modules, "phoenix.otel", None)
    with pytest.raises(SystemExit):
        tracing.enable()


def test_test_split_refuses_tracing(monkeypatch):
    git = {"dirty": False}
    arm = SimpleNamespace(name="a2")
    test_set = SimpleNamespace(split="test")
    R.check_test_guard(arm, test_set, git, None)
    R.check_test_guard(arm, SimpleNamespace(split="val"), git, None)
    monkeypatch.setenv(tracing.ENV_FLAG, "1")
    R.check_test_guard(arm, SimpleNamespace(split="val"), git, None)
    with pytest.raises(R.GuardError, match="never a test split"):
        R.check_test_guard(arm, test_set, git, None)


@pytest.mark.parametrize("flag,env", [(True, None), (False, "1")])
def test_eval_cli_refuses_before_loading_anything(monkeypatch, capsys, flag, env):
    if env:
        monkeypatch.setenv(tracing.ENV_FLAG, env)
    monkeypatch.setattr(cli, "load", lambda *a, **k: pytest.fail("loaded the test set"))
    argv = ["run", "--arm", "a2", "--dataset", "rulings", "--split", "test"]
    assert cli.main(argv + (["--trace"] if flag else [])) == 2
    assert "never a test split" in capsys.readouterr().err


@pytest.fixture
def spans(monkeypatch):
    """The real instrumentor with spans kept in memory: list of finished spans by name."""
    pytest.importorskip("openinference.instrumentation.langchain")
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing._instrument(provider)
    yield exporter
    LangChainInstrumentor().uninstrument()
    tracing._tracer = None


def fake_chat():
    """A real LangChain model, so the instrumentor sees the rationale call."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from test_a2_pipeline import RATIONALE

    usage = {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}
    message = AIMessage(content=json.dumps(RATIONALE), usage_metadata=usage)
    return GenericFakeChatModel(messages=iter([message] * 4))


def traced_a2(profile: bool):
    from test_a2_pipeline import a2

    pipe, server, _ = a2()
    pipe.cfg = {**pipe.cfg, "profile": profile}
    pipe.chat = fake_chat()
    return pipe, server


def test_graph_spans_tokens_and_parents(spans):
    from test_a2_pipeline import run

    pipe, _ = traced_a2(profile=False)
    state = run(pipe)
    done = spans.get_finished_spans()
    names = [s.name for s in done]
    for n in ("input_check", "retrieve", "stage1_score", "stage2_score", "combine", "validate",
              "rationale", "finalise"):  # fmt: skip
        assert n in names
    by_id = {s.context.span_id: s for s in done}
    s1 = next(s for s in done if s.name == "stage1_score_call")
    assert by_id[s1.parent.span_id].name == "stage1_score"
    assert s1.attributes["openinference.span.kind"] == "LLM"
    assert s1.attributes["llm.token_count.prompt"] == 300
    assert s1.attributes["llm.token_count.completion"] == 1
    s2 = next(s for s in done if s.name == "stage2_score_call 821192")
    assert by_id[s2.parent.span_id].name == "stage2_score"
    llm = [s for s in done if s.attributes.get("openinference.span.kind") == "LLM"
           and "score_call" not in s.name]  # fmt: skip
    assert len(llm) == 1 and by_id[llm[0].parent.span_id].name == "rationale"
    assert llm[0].attributes["llm.token_count.prompt"] == 50
    assert llm[0].attributes["llm.token_count.completion"] == 30
    assert len({s.context.trace_id for s in done}) == 1
    assert state["tokens"]["rationale"] == [50, 30]


def test_profile_node_gets_a_span_when_the_toggle_is_on(spans):
    from test_a2_pipeline import run

    run(traced_a2(profile=True)[0])
    assert "profile" in [s.name for s in spans.get_finished_spans()]


def test_tracing_does_not_change_the_output(request):
    from test_a2_pipeline import run

    plain = run(traced_a2(profile=False)[0])
    exporter = request.getfixturevalue("spans")
    traced = run(traced_a2(profile=False)[0])
    assert exporter.get_finished_spans()
    assert traced["final"] == plain["final"]
    assert traced["ranked"] == plain["ranked"]
