from pathlib import Path

import pytest

from hs_rag_classifier.data.atar import extract_ruling_refs
from hs_rag_classifier.data.atar_parse import (
    RulingParseError,
    listing_page_count,
    parse_listing_total,
    parse_ruling,
)

FIXTURES = Path(__file__).parent / "fixtures" / "atar"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_listing_total_and_pages():
    total = parse_listing_total(read("listing_page1.html"))
    assert total == 6131
    assert listing_page_count(total) == 246


def test_listing_has_25_refs():
    refs = extract_ruling_refs(read("listing_page1.html"))
    assert len(refs) == 25
    assert refs[0] == "600016899"


def test_parse_ruling_fields():
    r = parse_ruling(read("ruling_600016899.html"))
    assert r["reference"] == "600016899"
    assert r["commodity_code"] == "4202329090"
    assert r["start_date"] == "2026-10-20"
    assert r["expiry_date"] == "2029-10-19"
    assert r["description"].startswith("A spectacle pouch. With an outer surface")
    assert "\n" not in r["description"]
    assert r["keywords"][0] == "OF POLYESTER"
    assert len(r["keywords"]) == 8
    assert r["justification"].startswith("Classification has been determined")
    assert "GIR 6" in r["justification"]
    assert "extra_fields" not in r


def test_parse_second_ruling():
    r = parse_ruling(read("ruling_600016720.html"))
    assert r["reference"] == "600016720"
    assert r["commodity_code"] == "3917290099"


def test_non_ruling_page_raises():
    with pytest.raises(RulingParseError):
        parse_ruling(read("listing_page1.html"))
