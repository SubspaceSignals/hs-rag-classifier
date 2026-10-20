import httpx
import pytest

from hs_rag_classifier.tariff_auth import fetch_token, load_env

ENV = {
    "TRADE_TARIFF_TOKEN_URL": "https://auth.example/oauth2/token",
    "TRADE_TARIFF_CLIENT_ID": "id123",
    "TRADE_TARIFF_CLIENT_SECRET": "s3cret",
}


def test_load_env_ignores_comments(tmp_path, monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    p = tmp_path / ".env"
    p.write_text("# comment\nTRADE_TARIFF_CLIENT_ID = 'abc'\n\nNOT A PAIR\n")
    assert load_env(p) == {"TRADE_TARIFF_CLIENT_ID": "abc"}


def test_basic_auth_success():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"].startswith("Basic ")
        return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})

    assert fetch_token(ENV, httpx.MockTransport(handler))["access_token"] == "t"


def test_falls_back_to_form_credentials():
    def handler(req: httpx.Request) -> httpx.Response:
        if "authorization" in req.headers:
            return httpx.Response(401)
        assert b"client_secret=s3cret" in req.content
        return httpx.Response(200, json={"access_token": "t"})

    assert fetch_token(ENV, httpx.MockTransport(handler))["access_token"] == "t"


def test_failure_hides_secret():
    transport = httpx.MockTransport(lambda r: httpx.Response(400, text="bad s3cret"))
    with pytest.raises(RuntimeError) as exc:
        fetch_token(ENV, transport)
    assert "s3cret" not in str(exc.value)


def test_missing_values():
    with pytest.raises(RuntimeError, match="Missing"):
        fetch_token({})
