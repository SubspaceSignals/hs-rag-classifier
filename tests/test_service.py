"""The FastAPI service with fakes: no GPU, model or llama-server. The real A4Service class runs
on a fake arm and a fake OOD index, with the real operating-point rule file (policy (b))."""

import json
import logging

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


def service(arm=None, encode=fake_vectors):
    scorer = OOD.OODScorer(OOD.RulingsIndex(INDEX, ["r1", "r2"]), encode, max_input_chars=1000)
    prov = {"config_hash": core.A4_CONFIG_HASH, "operating_point_rule_sha": core.OPERATING_POINT_RULE_SHA,
            "ood_rule_sha": core.OOD_RULE_SHA, "git_commit": "abc"}  # fmt: skip
    return core.A4Service(arm or FakeArm(), scorer, {"adapter": {}}, OP_RULE, OOD_RULE, prov,
                          "http://127.0.0.1:8080", ruling_cn8=np.array(["11111111", "22222222"]))  # fmt: skip


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
    assert body["precedent"] == {"ref": "r1", "cn8": "11111111", "similarity": pytest.approx(1.0)}
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

    def bad(url, adapter):
        raise llm.AdapterError("lists []")

    monkeypatch.setattr(llm, "check_server_adapters", bad)
    assert s.health() == (False, "llama-server adapter check failed: lists []")
    monkeypatch.setattr(llm, "check_server_adapters", lambda url, adapter: {})
    assert s.health() == (True, None)


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
