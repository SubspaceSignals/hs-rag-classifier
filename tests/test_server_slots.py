"""The one-slot guard: llama-server's slot count from /props, the dev override, the test-split
refusal, the MLflow tags, and which runs call it. Mock HTTP only; needs no llama-server."""

import httpx
import mlflow
import pytest
from test_b1_embed import encoder, root  # noqa: F401 (root is a fixture)
from test_pipeline import CLEAN, S1, S2, VALID8, VALID10, atoz, pipeline  # noqa: F401

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.arms import b1ex_embed as BX
from hs_rag_classifier.eval import __main__ as cli
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.pipeline import llm


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    monkeypatch.delenv(llm.ENV_ALLOW_MULTISLOT, raising=False)
    monkeypatch.setattr(llm, "_allow_multislot", False)


def props_server(reply=None, status=200):
    """A transport answering /props with `reply` (default {"total_slots": 1})."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/props"
        return httpx.Response(status, json={"total_slots": 1} if reply is None else reply)

    return httpx.MockTransport(handler)


def unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def check(transport) -> dict:
    return llm.check_server_slots("http://fake", transport=transport)


def test_one_slot_passes_with_the_tags():
    assert check(props_server()) == {"server_slots": "1", "cache_prompt": "false"}


def test_cache_prompt_tag_is_the_value_sent(monkeypatch):
    monkeypatch.setattr(llm, "CACHE_PROMPT", True)
    assert check(props_server())["cache_prompt"] == "true"


@pytest.mark.parametrize("transport,match", [
    (props_server({"total_slots": 4}), "reports 4 slots"),
    (props_server({"build_info": "b11510"}), "does not report its slot count"),
    (props_server({"total_slots": True}), "does not report its slot count"),
    (props_server({"error": "not found"}, status=404), "cannot read the slot count"),
    (httpx.MockTransport(unreachable), "cannot read the slot count"),
])  # fmt: skip
def test_anything_but_one_slot_raises_and_says_how_to_start(transport, match):
    with pytest.raises(llm.ServerSlotsError, match=match) as exc:
        check(transport)
    assert "-np 1" in str(exc.value) and "cache_prompt=false" in str(exc.value)
    assert llm.SERVER_COMMAND in str(exc.value)


@pytest.mark.parametrize("how", ["env", "flag"])
def test_override_continues_with_a_warning_and_a_tag(monkeypatch, how):
    if how == "env":
        monkeypatch.setenv(llm.ENV_ALLOW_MULTISLOT, "1")
    else:
        llm.allow_multislot(True)
    with pytest.warns(RuntimeWarning, match="MULTI-SLOT OVERRIDE"):
        tags = check(props_server({"total_slots": 4}))
    assert tags == {"server_slots": "4", "cache_prompt": "false", "multislot_override": "true"}
    with pytest.warns(RuntimeWarning):
        assert check(httpx.MockTransport(unreachable))["server_slots"] == "unknown"


@pytest.mark.parametrize("value,on", [("1", True), ("0", False), ("", False), ("true", False)])
def test_only_hs_allow_multislot_1_requests_it(monkeypatch, value, on):
    monkeypatch.setenv(llm.ENV_ALLOW_MULTISLOT, value)
    assert llm.multislot_requested() is on


def test_test_split_refuses_the_override(monkeypatch):
    from types import SimpleNamespace

    arm, git = SimpleNamespace(name="a2"), {"dirty": False}
    test_set, val_set = SimpleNamespace(split="test"), SimpleNamespace(split="val")
    R.check_test_guard(arm, test_set, git, None)
    monkeypatch.setenv(llm.ENV_ALLOW_MULTISLOT, "1")
    R.check_test_guard(arm, val_set, git, None)
    with pytest.raises(R.GuardError, match="never a test split"):
        R.check_test_guard(arm, test_set, git, None)


@pytest.mark.parametrize("flag,env", [(True, None), (False, "1")])
def test_eval_cli_refuses_the_override_before_loading_anything(monkeypatch, capsys, flag, env):
    if env:
        monkeypatch.setenv(llm.ENV_ALLOW_MULTISLOT, env)
    monkeypatch.setattr(cli, "load", lambda *a, **k: pytest.fail("loaded the test set"))
    argv = ["run", "--arm", "a2", "--dataset", "rulings", "--split", "test"]
    assert cli.main(argv + (["--allow-multislot"] if flag else [])) == 2
    assert llm.MULTISLOT_TEST_REFUSED in capsys.readouterr().err


class Called(Exception):
    pass


def spy(monkeypatch) -> list[str]:
    """Replace the guard with one that records its base_url and stops the build."""
    seen = []

    def guard(base_url, transport=None):
        seen.append(base_url)
        raise Called

    monkeypatch.setattr(llm, "check_server_slots", guard)
    return seen


@pytest.mark.parametrize("build", [A1.build_pipeline, A2.build_pipeline])
def test_build_pipeline_runs_the_guard_before_loading_anything(monkeypatch, build):
    seen = spy(monkeypatch)
    monkeypatch.setattr("hs_rag_classifier.pipeline.options.load_tariff",
                        lambda *a: pytest.fail("loaded the tariff before the guard"))  # fmt: skip
    with pytest.raises(Called):
        build({**A1.DEFAULTS, "evidence": False}, tariff=None, api=False)
    assert seen == [A1.DEFAULTS["client"]["base_url"]]


def tags_of(result) -> dict:
    return mlflow.get_run(result["run_id"]).data.tags


def test_llm_run_is_tagged(atoz, tmp_path):  # noqa: F811
    pipe, _ = pipeline({"stage1": S1, "stage2": S2})
    pipe.server_tags = check(props_server())
    tags = tags_of(R.run_eval(A1.A1PipelineArm(pipe.cfg, pipeline=pipe), atoz, VALID8, VALID10,
                              mlflow_dir=tmp_path / "mlflow", git=CLEAN, n_resamples=20,
                              test_log=tmp_path / "eval" / "test_runs.jsonl"))  # fmt: skip
    assert tags["server_slots"] == "1" and tags["cache_prompt"] == "false"
    assert "multislot_override" not in tags


def test_override_run_is_tagged(monkeypatch, atoz, tmp_path):  # noqa: F811
    monkeypatch.setenv(llm.ENV_ALLOW_MULTISLOT, "1")
    pipe, _ = pipeline({"stage1": S1, "stage2": S2})
    with pytest.warns(RuntimeWarning):
        pipe.server_tags = check(props_server({"total_slots": 4}))
    dev_set = atoz.model_copy(update={"split": "val"})  # a test split refuses the override
    tags = tags_of(R.run_eval(A1.A1PipelineArm(pipe.cfg, pipeline=pipe), dev_set, VALID8,
                              VALID10, mlflow_dir=tmp_path / "mlflow", git=CLEAN, n_resamples=20,
                              test_log=tmp_path / "eval" / "test_runs.jsonl"))  # fmt: skip
    assert tags["server_slots"] == "4" and tags["multislot_override"] == "true"
    assert not (tmp_path / "eval" / "test_runs.jsonl").exists()


def test_b1ex_run_never_calls_the_guard(monkeypatch, root, atoz, tmp_path):  # noqa: F811
    monkeypatch.setattr(llm, "check_server_slots", lambda *a, **k: pytest.fail("guard called"))
    monkeypatch.setattr(llm.ServerInfo, "props", lambda self: pytest.fail("asked llama-server"))
    arm = BX.B1exEmbedArm(BX.default_config(k=2, temperature=0.1), root=root, encoder=encoder())
    tags = tags_of(R.run_eval(arm, atoz, VALID8, VALID10, mlflow_dir=tmp_path / "mlflow",
                              git=CLEAN, n_resamples=20,
                              test_log=tmp_path / "eval" / "test_runs.jsonl"))  # fmt: skip
    assert "server_slots" not in tags and "cache_prompt" not in tags
