"""A4 arm: frozen config, start-up pin checks, request bodies, MLflow params.

Every pin mismatch (a3_pipeline.json, adapter block or file, the adapter guard against
/lora-adapters, base GGUF on disk or on the server, temperatures, LoRA scales) is refused
before any scoring request. With the adapter block removed the requests are byte-identical to
the a4049ad fixture; with it, scoring carries scale 1 and the rationale scale 0. Temperatures are
read at full precision. The test-split entry point exists (arm "a4", not val_only) but no test
runs it. No GPU, no llama-server."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_a2_order_averaging import averaging
from test_a3_pipeline import FakeB1ex
from test_a4_candidate import GOLDEN, HASHES, lora_server, recorded

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.arms import a3_pipeline as A3
from hs_rag_classifier.arms import a4 as A4
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.arms import config_hash
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT
from hs_rag_classifier.pipeline import llm
from hs_rag_classifier.pipeline.graph_a2 import A2Pipeline

T1, T2 = 1.2555214195707303, 1.2685136415808747
A4_HASH = "5da7ac1452b3"


def env(tmp_path, **raw_changes):
    """A frozen config in tmp_path whose adapter, base GGUF and refit file are small tmp files."""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.gguf").write_bytes(b"adapter")
    base = tmp_path / "base.gguf"
    base.write_bytes(b"base")
    refit = root / "refit.json"
    refit.write_text(json.dumps({"temperature": {"stage1": T1, "stage2": T2}}), encoding="utf-8")
    raw = json.loads(A4.CONFIG_PATH.read_text(encoding="utf-8"))
    raw["adapter"] = {**raw["adapter"], "path": "a.gguf", "sha256": sha256(root / "a.gguf")}
    raw["base_model"] = {**raw["base_model"], "path": str(base), "sha256": sha256(base)}
    raw["temperature"]["source"] = {"path": "refit.json", "sha256": sha256(refit)}
    raw.update(raw_changes)
    path = tmp_path / "a4.json"
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    cfg = A4.load_config(path, root=root)
    cfg["server"] = {**cfg["server"], "model_sha256": raw["base_model"]["sha256"]}
    return SimpleNamespace(root=root, path=path, cfg=cfg, raw=raw, base=base, refit=refit)


def refused(e, cfg, match, path=None):
    with pytest.raises(A4.PinError, match=match):
        A4.check_pins(cfg, path or e.path, root=e.root)


# --- frozen config -----------------------------------------------------------------------


def test_frozen_config_pins():
    raw = A4.frozen()
    assert raw["base_config"] == {"path": "configs/a3_pipeline.json",
                                  "sha256": sha256(REPO_ROOT / "configs/a3_pipeline.json")}  # fmt: skip
    ad = raw["adapter"]
    assert ad["sha256"] == "dea88b27ab881e47f3a18cf9166cfb5a7fdb948f3dc015a952db29a4df94349f"
    assert (ad["size_bytes"], ad["candidate"], ad["step"]) == (160460288, "final", 745)
    server = A1.load_config()["server"]
    assert raw["base_model"]["sha256"] == server["model_sha256"]
    assert raw["server"]["command"] == (server["command"] + " -np 1 --offline --lora "
                                        "~/data/hs-rag-classifier/" + ad["path"])  # fmt: skip
    assert raw["lora_scales"] == {"scoring": 1.0, "rationale": 0.0}
    assert raw["review_policy"]["name"] == "b" and raw["review_policy"]["reported_only"] == ["a"]
    assert raw["training"]["builder_commit"].startswith("48f2898")
    assert raw["training"]["trainer_commit"].startswith("a4049ad")
    assert raw["training"]["step5_commit"].startswith("3d021e0")
    assert raw["training"]["mlflow_training_run"].startswith("50191630")


def test_temperatures_are_read_at_full_precision():
    text = A4.CONFIG_PATH.read_text(encoding="utf-8")
    assert '"stage1": 1.2555214195707303,' in text and '"stage2": 1.2685136415808747,' in text
    t = A4.load_config()["a2"]["temperature"]
    assert repr(t["stage1"]) == repr(T1) and repr(t["stage2"]) == repr(T2)
    assert t["stage1"] == T1 and t["stage2"] == T2 and t["mode"] == "per_stage"


def test_config_is_a3_plus_adapter_scales_and_temperatures():
    cfg, a3 = A4.load_config(), A3.load_config()
    assert cfg["adapter"] == {"path": str((DATA_ROOT / A4.frozen()["adapter"]["path"]).resolve()),
                              "sha256": A4.frozen()["adapter"]["sha256"]}  # fmt: skip
    rest = {k: v for k, v in cfg["a2"].items() if k not in ("temperature", "lora_scales")}
    assert rest == {k: v for k, v in a3["a2"].items() if k != "temperature"}
    assert rest["order_averaging"] is True
    assert {k: v for k, v in cfg.items() if k not in ("a2", "adapter", "a4")} == {
        k: v for k, v in a3.items() if k != "a2"
    }
    assert cfg["a4"]["candidate"] == "final"


# A4's hash includes the adapter's absolute path under HOME (adapter_block), so it holds only on
# the dev machine; CI deselects it (pytest -m "not frozen_host").
@pytest.mark.frozen_host
def test_config_hashes():
    got = {"a1": A1.load_config(), "a2": A2.load_config(),
           "a2_order_avg": A2.load_candidate_config(), "a3": A3.load_config()}  # fmt: skip
    assert {k: config_hash(v) for k, v in got.items()} == HASHES
    assert config_hash(A4.load_config()) == A4_HASH


# --- pins --------------------------------------------------------------------------------


def test_pins_pass_on_a_matching_config(tmp_path):
    e = env(tmp_path)
    A4.check_pins(e.cfg, e.path, root=e.root)


def test_a3_pin_mismatch_is_refused(tmp_path):
    e = env(tmp_path)
    bad = copy.deepcopy(e.raw)
    bad["base_config"]["sha256"] = "0" * 64
    other = tmp_path / "bad.json"
    other.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(A4.PinError, match="a3_pipeline.json changed"):
        A4.load_config(other, root=e.root)
    cfg = {**e.cfg, "a4": {**e.cfg["a4"], "frozen": bad,
                           "config": {"path": other.name, "sha256": sha256(other)}}}  # fmt: skip
    refused(e, cfg, "a3_pipeline.json changed", other)
    refused(e, e.cfg, "not built from", other)  # a config built from another file


def test_adapter_mismatches_are_refused(tmp_path):
    e = env(tmp_path)
    refused(e, {**e.cfg, "adapter": {**e.cfg["adapter"], "sha256": "ab" * 32}}, "adapter block")
    refused(e, {**e.cfg, "adapter": {**e.cfg["adapter"], "path": str(e.base)}}, "adapter block")
    (e.root / "a.gguf").write_bytes(b"changed")
    refused(e, e.cfg, "does not have SHA-256")


def test_adapter_guard_against_lora_adapters(tmp_path):
    e = env(tmp_path)
    block = e.cfg["adapter"]
    entry = {"id": 0, "path": block["path"], "scale": 1.0}
    assert llm.check_server_adapters("http://f", block, lora_server([entry]))["lora_adapter"]
    for listed in ([], [{**entry, "path": str(e.base)}], [entry, {**entry, "id": 1}]):
        with pytest.raises(llm.AdapterError):
            llm.check_server_adapters("http://f", block, lora_server(listed))


def test_base_gguf_mismatches_are_refused(tmp_path):
    e = env(tmp_path)
    refused(e, {**e.cfg, "server": {**e.cfg["server"], "model_sha256": "0" * 64}}, "A1 server")
    e.base.write_bytes(b"other base")
    refused(e, e.cfg, "base GGUF")

    def props(path):
        return httpx.MockTransport(lambda r: httpx.Response(200, json={"model_path": path}))

    A4.check_server_model(e.cfg, props(e.raw["base_model"]["path"]))
    with pytest.raises(A4.PinError, match="not the frozen base GGUF"):
        A4.check_server_model(e.cfg, props(str(tmp_path / "other.gguf")))
    with pytest.raises(A4.PinError, match="cannot read"):
        A4.check_server_model(e.cfg, httpx.MockTransport(lambda r: httpx.Response(500)))


def test_temperature_and_scale_mismatches_are_refused(tmp_path):
    e = env(tmp_path)

    def with_t(**t):
        return {**e.cfg, "a2": {**e.cfg["a2"], "temperature": {**e.cfg["a2"]["temperature"], **t}}}

    refused(e, with_t(stage1=1.2555), "stage1 temperature")  # rounded
    refused(e, with_t(stage2=1.2685136415808749), "stage2 temperature")  # one ulp off
    refused(e, with_t(stage1="1.2555214195707303"), "stage1 temperature")  # not a float
    refused(e, with_t(mode="shared"), "per_stage")
    refused(e, {**e.cfg, "a2": {**e.cfg["a2"], "lora_scales": {"scoring": 1.0, "rationale": 1.0}}},
            "LoRA scales")  # fmt: skip
    e.refit.write_text(json.dumps({"temperature": {"stage1": T1, "stage2": T2}}) + " ")
    refused(e, e.cfg, "refit file")  # the refit file changed
    # a refit file whose values differ from the frozen ones (with its SHA-256 re-pinned)
    e2 = env(tmp_path / "b")
    e2.refit.write_text(json.dumps({"temperature": {"stage1": 1.3, "stage2": T2}}))
    raw = {
        **e2.raw,
        "temperature": {
            **e2.raw["temperature"],
            "source": {"path": "refit.json", "sha256": sha256(e2.refit)},
        },
    }
    e2.path.write_text(json.dumps(raw), encoding="utf-8")
    cfg = {
        **e2.cfg,
        "a4": {
            **e2.cfg["a4"],
            "frozen": raw,
            "config": {"path": e2.path.name, "sha256": sha256(e2.path)},
        },
    }
    with pytest.raises(A4.PinError, match="refit file 1.3"):
        A4.check_pins(cfg, e2.path, root=e2.root)


def test_arm_refuses_before_any_request(tmp_path, monkeypatch):
    e = env(tmp_path)

    def no_server(*a, **k):
        raise AssertionError("a request was made before the pins were checked")

    monkeypatch.setattr(llm, "ServerInfo", no_server)
    monkeypatch.setattr(llm, "check_server_slots", no_server)
    bad = {**e.cfg, "adapter": {**e.cfg["adapter"], "sha256": "ab" * 32}}
    with pytest.raises(A4.PinError):
        A4.A4Arm(bad, root=e.root, path=e.path)


# --- requests ----------------------------------------------------------------------------


def a4_pipeline(adapter: bool):
    """averaging()'s pipeline with the frozen A4 temperatures, and the adapter if asked."""
    a2 = A4.load_config()["a2"]
    pipe, server = averaging(temperature=a2["temperature"])
    if not adapter:
        return pipe, server
    cfg = {**pipe.cfg, "a2": {**pipe.cfg["a2"], "lora_scales": a2["lora_scales"]},
           "adapter": A4.load_config()["adapter"]}  # fmt: skip
    return A2Pipeline(cfg, pipe.chat, pipe.retriever, pipe.tariff, pipe.validator, pipe.http,
                      pipe.paths), server  # fmt: skip


def test_without_the_adapter_block_requests_match_the_a4049ad_fixture():
    pipe, server = a4_pipeline(adapter=False)
    scoring, chat = recorded(pipe, server)
    assert scoring == GOLDEN["scoring"] and chat == GOLDEN["rationale"]


def test_scoring_carries_scale_1_and_the_rationale_scale_0(monkeypatch):
    pipe, server = a4_pipeline(adapter=True)
    scales = A4.load_config()["a2"]["lora_scales"]
    scoring, chat = recorded(pipe, server, chat_lora=llm.lora_request(scales["rationale"]))
    assert len(scoring) == 4 and len(chat) == 1
    for got, before, scale in zip(scoring + chat, GOLDEN["scoring"] + GOLDEN["rationale"],
                                  [1.0] * 4 + [0.0], strict=True):  # fmt: skip
        g = json.loads(got)
        assert g.pop("lora") == [{"id": 0, "scale": scale}]
        assert g == json.loads(before)
    # and build_pipeline gives the real A4 config's chat client scale 0, scoring scale 1
    parts = SimpleNamespace(chat=None, retriever=None, tariff=None, validator=None,
                            evidence_index=None, server_tags={})  # fmt: skip
    monkeypatch.setattr(A1, "build_pipeline", lambda *a, **k: parts)
    monkeypatch.setattr("hs_rag_classifier.pipeline.options.load_cn8_paths", lambda root: {})
    built = A2.build_pipeline({**A4.load_config(), "evidence": False})  # no index here
    assert built.chat.extra_body["lora"] == [{"id": 0, "scale": 0.0}]
    assert built.scoring_lora == [{"id": 0, "scale": 1.0}]


# --- arm, params, entry point -------------------------------------------------------------


def test_arm_logs_its_params_and_details(tmp_path):
    e = env(tmp_path)
    pipe, _ = a4_pipeline(adapter=True)
    arm = A4.A4Arm(e.cfg, root=e.root, pipeline=pipe, b1ex=FakeB1ex("8211929000"), path=e.path)
    assert arm.name == "a4" and arm.val_only is False
    assert arm.run_params == {"candidate_step": "final", "adapter_sha256": e.raw["adapter"]["sha256"],
                              "a4_config_sha256": sha256(e.path),
                              "temperature_stage1": "1.2555214195707303",
                              "temperature_stage2": "1.2685136415808747"}  # fmt: skip
    p = arm.predict([SimpleNamespace(item_id="1", text="a kitchen knife")])[0]
    assert p.details["a4"]["adapter_sha256"] == e.raw["adapter"]["sha256"]
    assert "a3" in p.details


def test_test_split_entry_point_exists():
    # never called here: only checked to be registered and not validation-only
    assert R.ARMS["a4"].__name__ == "_a4"
    assert getattr(A4.A4Arm, "val_only", False) is False
    with pytest.raises(ValueError, match="takes no adapter"):
        R.make_arm("a4", set(), adapter="x.gguf")
    assert Path(A4.__file__).name == "a4.py"
