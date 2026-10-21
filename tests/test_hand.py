"""Checker tests. The CSV fixture is a unit-test fixture only, not evaluation data."""

import csv
from pathlib import Path

import pytest

from hs_rag_classifier.data import hand as H
from hs_rag_classifier.data.nomenclature import build_codes, read_commodities_csv

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_CSV = FIXTURES / "hand" / "hand_unit_test_fixture.csv"


@pytest.fixture(scope="module")
def codes():
    return build_codes(read_commodities_csv(FIXTURES / "tariff" / "commodities.csv"))


def rows_of(path=FIXTURE_CSV):
    return H.read_rows(path)


def check(rows, codes, header=None):
    return H.check_rows(header or H.HEADER, rows, codes)


def test_fixture_passes_with_flags(codes):
    header, rows = rows_of()
    failures, flags = check(rows, codes, header)
    assert failures == []
    assert any(f.startswith("H003: near duplicate of H001") for f in flags)
    assert len(rows) == 4


def test_ids_unique_and_sequential(codes):
    _, rows = rows_of()
    rows[2]["id"] = "H002"
    failures, _ = check(rows, codes)
    assert "H002: id used more than once" in failures
    assert any("id out of sequence (expected H003)" in f for f in failures)


def test_description_rules(codes):
    _, rows = rows_of()
    rows[0]["description"] = ""
    rows[1]["description"] = "x" * 201
    failures, _ = check(rows, codes)
    assert "H001: description is empty" in failures
    assert any(f.startswith("H002: description is 201 characters") for f in failures)


def test_cn8_and_code10_rules(codes):
    _, rows = rows_of()
    rows[0]["cn8"] = "8211910"  # 7 digits
    rows[1]["cn8"] = "82119900"  # not current
    rows[2]["code10"] = "0101210000"  # current but under another cn8
    rows[3]["code10"] = "0101290000"  # not declarable
    failures, _ = check(rows, codes)
    assert "H001: cn8 '8211910' is not 8 digits" in failures
    assert any(f.startswith("H002: cn8 82119900 is not current") for f in failures)
    assert "H003: code10 0101210000 is not under cn8 82119100" in failures
    assert "H004: code10 0101290000 is not a current declarable line" in failures


def test_confidence_and_source_url(codes):
    _, rows = rows_of()
    rows[0]["label_confidence"] = "maybe"
    rows[1]["source_url"] = ""
    rows[2]["source_url"] = "example.com/x"
    failures, _ = check(rows, codes)
    assert any(f.startswith("H001: label_confidence") for f in failures)
    assert "H002: source_url is empty" in failures
    assert "H003: source_url is not an http(s) URL" in failures


def test_flags_exact_duplicate_and_own_code(codes):
    _, rows = rows_of()
    rows[2]["description"] = "Table knife; fixed blade, stainless, PK 12"
    rows[3]["description"] = "breeding mare, heading 0101"
    failures, flags = check(rows, codes)
    assert failures == []
    assert "H003: exact duplicate of H001" in flags
    assert "H004: description contains its own code prefix" in flags


def test_wrong_header_fails(codes):
    failures, _ = H.check_rows(["id", "description"], [], codes)
    assert failures and failures[0].startswith("header must be exactly")


def test_check_never_modifies_csv(tmp_path, codes, monkeypatch):
    path = tmp_path / "hand.csv"
    path.write_bytes(FIXTURE_CSV.read_bytes())
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    monkeypatch.setattr(H.pd, "read_parquet", lambda *a, **k: codes)
    assert H.main(["--csv", str(path), "--root", str(tmp_path), "check"]) == 0
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_check_exit_code_on_failure(tmp_path, codes, monkeypatch):
    path = tmp_path / "hand.csv"
    _, rows = rows_of()
    rows[0]["cn8"] = "bad"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=H.HEADER)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(H.pd, "read_parquet", lambda *a, **k: codes)
    assert H.main(["--csv", str(path), "--root", str(tmp_path), "check"]) == 1


def test_init_writes_header_only_and_never_overwrites(tmp_path):
    path = tmp_path / "hand" / "hand_v1.csv"
    assert H.main(["--csv", str(path), "init"]) == 0
    assert path.read_text() == ",".join(H.HEADER) + "\n"
    path.write_text(path.read_text() + "H001,x,82119100,,sure,https://a,b,\n")
    assert H.main(["--csv", str(path), "init"]) == 0
    assert "H001" in path.read_text()  # left unchanged
