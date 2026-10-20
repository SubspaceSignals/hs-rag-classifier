import httpx
import pytest

from hs_rag_classifier.data.atar import PoliteClient, RobotsDisallowed, extract_ruling_refs


def make_client(handler, robots="User-agent: *\nAllow: /\n", **kwargs):
    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots)
        return handler(request)

    sleeps: list[float] = []
    client = PoliteClient(transport=httpx.MockTransport(route), sleep=sleeps.append, **kwargs)
    return client, sleeps


def test_extract_ruling_refs_dedupes_in_order():
    html = (
        '<a href="/search-for-advance-tariff-rulings/ruling/600016899">View</a>'
        '<a href="/search-for-advance-tariff-rulings/ruling/600016720">View</a>'
        '<a href="/search-for-advance-tariff-rulings/ruling/600016899">Again</a>'
    )
    assert extract_ruling_refs(html) == ["600016899", "600016720"]


def test_robots_disallow_blocks_request():
    client, _ = make_client(lambda r: httpx.Response(200), robots="User-agent: *\nDisallow: /\n")
    with pytest.raises(RobotsDisallowed):
        client.get("/search-for-advance-tariff-rulings/search", params={"page": 1})


def test_crawl_delay_raises_minimum_delay():
    client, _ = make_client(
        lambda r: httpx.Response(200), robots="User-agent: *\nCrawl-delay: 5\n", delay_s=1.5
    )
    assert client.delay_s == 5


def test_retries_on_503_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, text="ok")

    client, _ = make_client(handler)
    assert client.get("/x") == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_retries():
    client, _ = make_client(lambda r: httpx.Response(503), max_retries=2)
    with pytest.raises(RuntimeError, match="Giving up"):
        client.get("/x")


def test_404_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    client, _ = make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.get("/x")
    assert calls["n"] == 1
