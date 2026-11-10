"""The FastAPI service with fakes: no GPU, model or llama-server. The real A4Service class runs
on a fake arm and a fake OOD index, with the real operating-point rule file (policy (b))."""

import json
import logging
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hs_rag_classifier import ood as OOD
from hs_rag_classifier.arms import embed as E
from hs_rag_classifier.eval.arms import Candidate, Prediction
from hs_rag_classifier.service import core
from hs_rag_classifier.service.app import create_app
from hs_rag_classifier.service.schemas import MAX_DESCRIPTION_CHARS, ClassifyResponse

OP_RULE = json.loads(core.OPERATING_POINT_RULE.read_text())
OOD_RULE = {"threshold": 0.5, "flag": "similarity < threshold"}
# texts starting "near" sit on ruling r1 (similarity 1); "far" ones at 0.3 from every ruling
VECS = {"near": [1.0, 0.0, 0.0], "far": [0.3, 0.3, float(np.sqrt(1 - 0.18))]}
INDEX = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)  # ref r1 (cn8 11111111), r2
ADAPTER = {"path": "/models/a4.gguf", "sha256": "d" * 64, "id": 0}


def fake_vectors(texts):
    return np.array([VECS[t.split()[0]] for t in texts], np.float32)


class FakeArm:
    """A4's predict signature; confidence and agreement come from the description's words."""

    def __init__(self, raises=None, rationale_error=None):
        self.raises, self.rationale_error, self.calls = raises, rationale_error, 0

    def predict(self, items):
        self.calls += 1
        if self.raises:
            raise self.raises
        (item,) = items
        words = item.text.split()
        conf = 0.5 if "low" in words else 0.95
        det = {"a3": {"agree_cn8": "agree" in words},
               "rationale": {"quote": "Of plastics", "citation": {"quote": "of plastics", "pass": True},
                             "error": self.rationale_error},
               "disclaimer": core.ADVISORY}  # fmt: skip
        cands = [Candidate(code="95030095", score=conf), Candidate(code="9503009900", score=0.03),
                 Candidate(code="39269097", score=0.01), Candidate(code="39269099", score=0.005)]  # fmt: skip
        return [Prediction(item_id=item.item_id, candidates=cands, confidence=conf,
                           rationale="'Of plastics': toy.", details=det)]  # fmt: skip


def service(arm=None, encode=fake_vectors, ruling_cn8=("11111111", "22222222")):
    scorer = OOD.OODScorer(OOD.RulingsIndex(INDEX, ["r1", "r2"]), encode, max_input_chars=1000)
    prov = {"config_hash": core.A4_CONFIG_HASH, "operating_point_rule_sha": core.OPERATING_POINT_RULE_SHA,
            "ood_rule_sha": core.OOD_RULE_SHA, "git_commit": "abc"}  # fmt: skip
    return core.A4Service(arm or FakeArm(), scorer, {"adapter": {}}, OP_RULE, OOD_RULE, prov,
                          "http://127.0.0.1:8080", ruling_cn8=np.array(ruling_cn8), adapter=ADAPTER)  # fmt: skip


def client(pipe=None):
    return TestClient(create_app(pipeline=pipe or service()))


# --- validation and schema ----------------------------------------------------------------


@pytest.mark.parametrize("desc", ["", "   ", "\n\t ", "x" * (MAX_DESCRIPTION_CHARS + 1)])
def test_rejects_empty_whitespace_and_over_length(desc):
    with client() as c:
        assert c.post("/classify", json={"description": desc}).status_code == 422
        assert c.post("/classify", json={}).status_code == 422


def test_accepts_max_length_and_returns_the_schema():
    with client() as c:
        r = c.post(
            "/classify", json={"description": "near agree " + "x" * (MAX_DESCRIPTION_CHARS - 11)}
        )
        assert r.status_code == 200
        body = r.json()
    ClassifyResponse.model_validate(body)
    assert set(body) == {"request_id", "cn8", "confidence", "top3", "needs_review", "review_reasons",
                         "ood", "rationale", "citation", "precedent", "provenance", "advisory"}  # fmt: skip
    assert body["cn8"] == "95030095" and [t["cn8"] for t in body["top3"]] == [
        "95030095",
        "95030099",
        "39269097",
    ]
    assert body["citation"] == {"quote": "of plastics", "pass": True}
    assert body["precedent"] == {"ref": "r1", "cn8": "11111111", "similarity": pytest.approx(1.0),
                                 "agrees_with_prediction": False}  # fmt: skip
    assert set(body["ood"]) == {
        "similarity",
        "threshold",
        "flagged",
        "nearest_ruling_ref",
        "message",
    }
    assert body["provenance"]["config_hash"] == "5da7ac1452b3" and len(body["request_id"]) == 32
    assert "/classify" in client().get("/openapi.json").json()["paths"]


# --- needs_review -------------------------------------------------------------------------


@pytest.mark.parametrize(("desc", "op", "ood"), [
    ("near agree", False, False),        # agrees with B1-ex: t_a = 0, never reviewed
    ("near low", True, False),           # disagrees, confidence 0.5 < t_d
    ("far agree", False, True),          # OOD only
    ("far low", True, True),             # both
])  # fmt: skip
def test_needs_review_is_operating_point_or_ood(desc, op, ood):
    with client() as c:
        b = c.post("/classify", json={"description": desc}).json()
    want = [r for r, on in (("operating_point", op), ("out_of_distribution", ood)) if on]
    assert b["needs_review"] == (op or ood) and b["review_reasons"] == want
    assert b["ood"]["flagged"] == ood
    assert b["ood"]["message"] == (OOD.REASON if ood else None)


def test_operating_point_uses_the_rule_file_thresholds():
    t = OP_RULE["levels"]["cn8"]["orders"]["b"]["lower_bound"]
    th = {"t_d": float(t["t_d"]), "t_a": float(t["t_a"])}
    assert core.operating_point_review(t_d := th["t_d"], False, th) is False  # strict <
    assert core.operating_point_review(np.nextafter(t_d, 0), False, th) is True
    assert core.operating_point_review(0.01, True, th) is False
    assert core.decide(False, False) == (False, [])
    assert core.decide(True, True) == (True, ["operating_point", "out_of_distribution"])


# --- errors and health --------------------------------------------------------------------


def test_llama_server_failure_is_503_without_a_trace():
    for arm in (FakeArm(raises=httpx.ConnectError("refused")),
                FakeArm(rationale_error="APIConnectionError: Connection error.")):  # fmt: skip
        with client(service(arm)) as c:
            r = c.post("/classify", json={"description": "near agree"})
        assert r.status_code == 503 and r.json()["status"] == "unavailable"
        assert "Traceback" not in r.text and "refused" not in r.text
    with client(service(FakeArm(rationale_error="ValueError: reply is not JSON"))) as c:
        assert c.post("/classify", json={"description": "near agree"}).status_code == 200


class FakeHealth:
    def __init__(self, ok, reason=None):
        self.ok, self.reason = ok, reason

    def health(self):
        return self.ok, self.reason


def test_health_ok_and_failing(monkeypatch):
    with client(FakeHealth(True)) as c:
        assert c.get("/health").json() == {"status": "ok", "reason": None}
    with client(FakeHealth(False, "llama-server down")) as c:
        r = c.get("/health")
    assert r.status_code == 503 and r.json()["reason"] == "llama-server down"
    from hs_rag_classifier.pipeline import llm

    s = service()
    monkeypatch.setattr(llm.ServerInfo, "up", lambda self: False)
    assert s.health()[0] is False and "not reachable" in s.health()[1]
    monkeypatch.setattr(llm.ServerInfo, "up", lambda self: True)
    monkeypatch.setattr(llm.ServerInfo, "lora_adapters", lambda self: [])
    assert s.health()[0] is False and "lists none" in s.health()[1]
    monkeypatch.setattr(llm.ServerInfo, "lora_adapters", lambda self: [dict(ADAPTER)])
    assert s.health() == (True, None)


# --- precedent ----------------------------------------------------------------------------


@pytest.mark.parametrize(("ruling_cn8", "agrees"), [(("95030095", "22222222"), True),
                                                    (("11111111", "22222222"), False)])  # fmt: skip
def test_precedent_agrees_with_prediction(ruling_cn8, agrees):
    with client(service(ruling_cn8=ruling_cn8)) as c:
        b = c.post("/classify", json={"description": "near agree"}).json()
    assert b["cn8"] == "95030095" and b["precedent"]["ref"] == "r1"
    assert b["precedent"]["agrees_with_prediction"] is agrees
    schema = client().get("/openapi.json").json()["components"]["schemas"]["Precedent"]
    assert "agrees_with_prediction" in schema["required"]
    assert schema["properties"]["agrees_with_prediction"]["examples"] == [True]


# --- adapter: hashed once at start-up, never on /health -----------------------------------


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _frozen_setup(monkeypatch, tmp_path, declared_sha=None):
    """from_frozen on a fake arm; the adapter is a real small file. Returns the hash counters."""
    import hashlib

    from hs_rag_classifier.arms import a4 as A4
    from hs_rag_classifier.eval.arms import config_hash
    from hs_rag_classifier.pipeline import llm

    gguf = tmp_path / "a4.gguf"
    gguf.write_bytes(b"adapter bytes")
    sha = hashlib.sha256(b"adapter bytes").hexdigest()
    cfg = {"adapter": {"path": str(gguf), "sha256": declared_sha or sha}, "max_input_chars": 1000,
           "client": {"base_url": "x"}}  # fmt: skip
    monkeypatch.setattr(A4, "load_config", lambda root=None: cfg)
    monkeypatch.setattr(core, "A4_CONFIG_HASH", config_hash(cfg))
    loaded = []

    def fake_arm(config, root):
        loaded.append(1)
        idx = _NS(embeddings=INDEX, meta={"refs": ["r1", "r2"]}, cn8=np.array(["11111111", "2"]))
        arm = FakeArm()
        arm.b1ex = _NS(index=idx, encoder=_NS(queries=fake_vectors))
        return arm

    monkeypatch.setattr(A4, "A4Arm", fake_arm)
    counts = {"adapter": 0, "llm": 0}
    real = core.sha256_file

    def counting(path):
        counts["adapter"] += Path(path) == gguf
        return real(path)

    def no_llm_hash(path):
        counts["llm"] += 1
        raise AssertionError("llm._file_sha256 must not run in the service")

    monkeypatch.setattr(core, "sha256_file", counting)
    monkeypatch.setattr(llm, "_file_sha256", no_llm_hash)
    listing = {"loras": [{"id": 0, "path": str(gguf), "scale": 1.0}]}
    monkeypatch.setattr(llm.ServerInfo, "up", lambda self: True)
    monkeypatch.setattr(llm.ServerInfo, "lora_adapters", lambda self: listing["loras"])
    return gguf, sha, counts, listing, loaded


def test_health_never_rehashes_the_adapter(monkeypatch, tmp_path):
    gguf, sha, counts, _, _ = _frozen_setup(monkeypatch, tmp_path)
    with TestClient(create_app()) as c:
        assert counts["adapter"] == 1  # start-up hashed it once
        assert c.app.state.pipeline.adapter == {"path": str(gguf), "sha256": sha, "id": 0}
        for _ in range(5):
            assert c.get("/health").status_code == 200
    assert counts == {"adapter": 1, "llm": 0}


def test_health_503_when_the_server_reports_a_different_adapter(monkeypatch, tmp_path):
    gguf, _, counts, listing, _ = _frozen_setup(monkeypatch, tmp_path)
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 200
        for other in ([{"id": 0, "path": str(tmp_path / "other.gguf")}],
                      [{"id": 1, "path": str(gguf)}],
                      [{"id": 0, "path": str(gguf)}, {"id": 1, "path": str(gguf)}], []):  # fmt: skip
            listing["loras"] = other
            r = c.get("/health")
            assert r.status_code == 503 and "adapter check failed" in r.json()["reason"]
    assert counts == {"adapter": 1, "llm": 0}


def test_startup_refuses_a_wrong_adapter_hash(monkeypatch, tmp_path):
    _, _, counts, _, loaded = _frozen_setup(monkeypatch, tmp_path, declared_sha="0" * 64)
    with pytest.raises(core.StartupError, match="adapter .* has SHA-256"), TestClient(create_app()):
        pass
    assert loaded == [] and counts["adapter"] == 1  # refused before the arm loads


def test_startup_refuses_without_git_provenance(monkeypatch, tmp_path):
    from hs_rag_classifier.eval import runner as R

    _, _, _, _, loaded = _frozen_setup(monkeypatch, tmp_path)
    real = R.git_info
    monkeypatch.setattr(R, "git_info", lambda: real(repo=tmp_path, environ={}))
    with pytest.raises(core.StartupError, match="no git provenance"), TestClient(create_app()):
        pass
    assert loaded == []  # refused before the arm loads
    sha = "f" * 40
    env = {R.ENV_GIT_COMMIT: sha, R.ENV_GIT_DIRTY: "false"}
    monkeypatch.setattr(R, "git_info", lambda: real(repo=tmp_path, environ=env))
    with TestClient(create_app()) as c:
        assert c.app.state.pipeline.provenance["git_commit"] == sha


def test_startup_refuses_when_the_server_lists_another_adapter(monkeypatch, tmp_path):
    _, _, _, listing, _ = _frozen_setup(monkeypatch, tmp_path)
    listing["loras"] = [{"id": 0, "path": str(tmp_path / "other.gguf")}]
    with pytest.raises(core.StartupError, match="lists"), TestClient(create_app()):
        pass


# --- start-up refusal ---------------------------------------------------------------------


def test_startup_refuses_a_wrong_config_hash_or_rule_sha(monkeypatch, tmp_path):
    from hs_rag_classifier.arms import a4 as A4
    from hs_rag_classifier.eval.arms import config_hash

    def no_arm(*a, **k):
        raise AssertionError("the arm must not load before the pins are checked")

    monkeypatch.setattr(A4, "A4Arm", no_arm)
    monkeypatch.setattr(A4, "load_config", lambda root=None: {"x": 1})
    with pytest.raises(core.StartupError, match="config_hash"), TestClient(create_app()):
        pass
    monkeypatch.setattr(core, "A4_CONFIG_HASH", config_hash({"x": 1}))
    other = tmp_path / "ood_rule.json"
    other.write_text('{"threshold": 0.1}\n')
    settings = core.Settings(ood_rule=other)
    with (
        pytest.raises(core.StartupError, match="ood_rule_sha") as e,
        TestClient(create_app(settings=settings)),
    ):
        pass
    assert "config_hash" not in str(e.value)
    settings = core.Settings(operating_point_rule=other)
    with (
        pytest.raises(core.StartupError, match="operating_point_rule_sha"),
        TestClient(create_app(settings=settings)),
    ):
        pass


# --- shared encoder -----------------------------------------------------------------------


def test_two_requests_reuse_the_one_shared_encoder(monkeypatch):
    monkeypatch.setattr(E, "_SHARED", {})
    loads = []

    def loader(config, batch_size=E.ENCODE_BATCH):
        loads.append(1)

        def encode(texts):
            return fake_vectors(texts)

        encode.model = object()
        return encode

    monkeypatch.setattr(E, "sentence_transformer_backend", loader)

    def no_release(enc):
        raise AssertionError("release() on a request path")

    monkeypatch.setattr(E, "release", no_release)
    cfg = E.model_config("bge-m3")
    arm = FakeArm()
    arm.b1ex = type("B", (), {"encoder": E.Encoder(cfg)})()
    retriever = type("R", (), {"encoder": E.Encoder(cfg)})()
    arm.a2 = type("A2", (), {"pipeline": type("P", (), {"retriever": retriever})()})()
    s = service(arm, encode=arm.b1ex.encoder.queries)
    before = s.encoder_backends()
    assert len({id(b) for b in before.values()}) == 1 and loads == [1]
    with client(s) as c:
        for _ in range(2):
            assert c.post("/classify", json={"description": "near agree"}).status_code == 200
    after = s.encoder_backends()
    assert all(after[k] is before[k] for k in before) and loads == [1] and arm.calls == 2


# --- settings, entry point and logs -------------------------------------------------------


def test_default_host_port_and_llama_url(monkeypatch):
    s = core.Settings.from_env({})
    assert (s.host, s.port, s.llama_url) == ("127.0.0.1", 8000, "http://127.0.0.1:8080")
    assert core.Settings.from_env({"PORT": "8123", "LLAMA_URL": "http://127.0.0.1:9"}).port == 8123
    import uvicorn

    from hs_rag_classifier.service import __main__ as main

    got = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: got.update(kw))
    for k in ("HOST", "PORT"):
        monkeypatch.delenv(k, raising=False)
    main.main()
    assert got == {"host": "127.0.0.1", "port": 8000, "workers": 1}


def test_logs_never_contain_the_description(caplog):
    secret = "near agree zebra-striped-teapot-7f3a"
    with caplog.at_level(logging.DEBUG), client() as c:
        assert c.post("/classify", json={"description": secret}).status_code == 200
        c.post("/classify", json={"description": "near low"})
    with (
        caplog.at_level(logging.DEBUG),
        client(service(FakeArm(raises=httpx.ConnectError("x")))) as c,
    ):
        c.post("/classify", json={"description": secret})
    assert "zebra-striped-teapot" not in caplog.text
    assert "needs_review=False" in caplog.text and "review_reasons=operating_point" in caplog.text
    assert "status=503" in caplog.text
