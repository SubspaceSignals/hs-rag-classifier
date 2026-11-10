"""TARIFF_API_OFFLINE=1: a cache miss makes no request (no token either) and writes nothing, and
the validator takes its existing fallback to the local nomenclature. Unset: unchanged."""

import json
from types import SimpleNamespace

import httpx
import pytest

from hs_rag_classifier.data.nomenclature import ENV_API_OFFLINE, TariffApi, api_offline
from hs_rag_classifier.pipeline.validate import Validator

TARIFF = SimpleNamespace(code10={"82119100": ["8211910000"]})
# Fake credentials so a token fetch would be attempted if any request were made.
CREDS = {"TRADE_TARIFF_TOKEN_URL": "https://auth.invalid/token",
         "TRADE_TARIFF_CLIENT_ID": "id", "TRADE_TARIFF_CLIENT_SECRET": "s"}  # fmt: skip
DECLARABLE = {"data": {"attributes": {"declarable": True}}}


def fake_api(tmp_path, env=None):
    """TariffApi on a recording fake fetcher: the token endpoint and the commodity endpoint."""
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if "token" in request.url.path:
            return httpx.Response(200, json={"access_token": "t", "expires_in": 3600})
        return httpx.Response(200, json=DECLARABLE)

    api = TariffApi(tmp_path, env=CREDS if env is None else env,
                    transport=httpx.MockTransport(handler), sleep=lambda s: None)  # fmt: skip
    return api, requests


def files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_switch_reads_only_the_exact_value_one():
    assert api_offline({ENV_API_OFFLINE: "1"}) is True
    for env in ({}, {ENV_API_OFFLINE: "0"}, {ENV_API_OFFLINE: ""}, {ENV_API_OFFLINE: "true"}):
        assert api_offline(env) is False


def test_offline_miss_no_network_no_write_nomenclature_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_API_OFFLINE, "1")
    api, requests = fake_api(tmp_path)
    r = Validator(TARIFF, api).check("82119100")
    assert r == {"valid": True, "nomenclature": True, "api": None, "api_error": "TariffApiOffline"}
    assert requests == [] and api._token is None and api.fetched == 0
    assert files(tmp_path) == []


def test_online_cache_miss_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_API_OFFLINE, raising=False)
    api, requests = fake_api(tmp_path)
    r = Validator(TARIFF, api).check("82119100")
    assert r == {"valid": True, "nomenclature": True, "api": "8211910000", "api_error": None}
    assert [u.split("?")[0] for u in requests] == [
        "https://auth.invalid/token",
        "https://api.trade-tariff.service.gov.uk/uk/api/commodities/8211910000",
    ]
    assert files(tmp_path) == ["commodities/8211910000.json"] and api.fetched == 1


@pytest.mark.parametrize("cached", [DECLARABLE, {"data": {"attributes": {"declarable": False}}},
                                    None])  # fmt: skip
def test_cache_hits_are_identical_offline_and_online(tmp_path, monkeypatch, cached):
    (tmp_path / "commodities").mkdir()
    (tmp_path / "commodities" / "8211910000.json").write_text(json.dumps(cached))
    got = {}
    for mode in ("1", None):
        if mode:
            monkeypatch.setenv(ENV_API_OFFLINE, mode)
        else:
            monkeypatch.delenv(ENV_API_OFFLINE, raising=False)
        api, requests = fake_api(tmp_path)
        got[mode] = Validator(TARIFF, api).check("82119100")
        assert requests == []
    assert got["1"] == got[None] and got["1"]["api_error"] is None
    assert files(tmp_path) == ["commodities/8211910000.json"]
