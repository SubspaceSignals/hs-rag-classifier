import gzip
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from hs_rag_classifier.data.nomenclature import (
    API_BASE_AUTH,
    API_BASE_OPEN,
    TariffApi,
    build_codes,
    build_codes8,
    chapter_section_map,
    check_superseded,
    clean_note,
    clean_text,
    download_csv,
    parse_chapter,
    parse_section,
    parse_section_ranges,
    read_commodities_csv,
    resolve_latest_version,
    run_checks,
)

FIXTURES = Path(__file__).parent / "fixtures" / "tariff"
SECTIONS = {"01": 1, "82": 15}
ENV = {
    "TRADE_TARIFF_TOKEN_URL": "https://auth.example/oauth2/token",
    "TRADE_TARIFF_CLIENT_ID": "id123",
    "TRADE_TARIFF_CLIENT_SECRET": "s3cret",
}


def load_json(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def codes():
    return build_codes(read_commodities_csv(FIXTURES / "commodities.csv"), SECTIONS)


@pytest.fixture(scope="module")
def codes8(codes):
    return build_codes8(codes).set_index("cn8")


def line(codes, code10, suffix="80"):
    return codes[(codes["code10"] == code10) & (codes["suffix"] == suffix)].iloc[0]


def test_clean_text_removes_markup():
    assert clean_text("Knives, serrated or not<br>") == "Knives, serrated or not"
    assert clean_text("15|cm<sup>2</sup> &amp; over") == "15 cm2 & over"
    assert clean_text("  a\xa0 b ") == "a b"
    assert clean_text(None) == ""


def test_clean_note_strips_markdown():
    raw = "### Note\r\n\r\n\r\n\r\n1. See heading [0301](/headings/0301).  \r\n"
    assert clean_note(raw) == "Note\n\n1. See heading 0301."


def test_csv_reads_na_and_keeps_leading_zeros(codes):
    chapter = line(codes, "0100000000")
    assert pd.isna(chapter["parent_sid"])
    assert chapter["depth"] == 0
    assert line(codes, "0101299000")["validity_end"].isoformat() == "2027-01-01"


def test_path_built_from_chapter_down(codes):
    leaf = line(codes, "0102291010")
    assert list(leaf["path_sids"]) == [1000, 1010, 1011, 1012, 1013, 1014]
    assert leaf["path_text"] == (
        "LIVE ANIMALS > Live bovine animals > Other > Other > "
        "Of a weight not exceeding 80 kg > Young male bovine animals, intended for fattening"
    )
    assert leaf["depth"] == 5
    assert leaf["path_complete"]


def test_derived_codes_and_section(codes):
    knife = line(codes, "8211910000")
    assert (knife["chapter"], knife["heading"], knife["subheading"], knife["cn8"]) == (
        "82", "8211", "821191", "82119100",
    )  # fmt: skip
    assert knife["section_id"] == 15
    assert knife["description_raw"] == "Table knives having fixed blades"


def test_declarable_flag(codes):
    assert line(codes, "0101210000")["declarable"]  # suffix 80 leaf
    assert not line(codes, "0101210000", "10")["declarable"]  # grouping line
    assert not line(codes, "0101290000")["declarable"]  # suffix 80 with children
    assert not line(codes, "8211910000", "20")["declarable"]
    assert line(codes, "8211920010")["declarable"]
    assert not line(codes, "0100000000")["declarable"]
    assert codes["declarable"].sum() == 16  # includes the superseded Asses row


def test_codes8_single_line_is_declarable_at_8(codes8):
    row = codes8.loc["01012100"]
    assert row["cn8_declarable"]
    assert list(row["children"]) == ["0101210000"]
    assert row["description"] == "Pure-bred breeding animals"


def test_codes8_node_is_lca_of_10_digit_children(codes8):
    row = codes8.loc["01022910"]
    assert row["node_code10"] == "0102291000" and row["node_suffix"] == "80"
    assert list(row["children"]) == ["0102291010", "0102291090"]
    assert not row["cn8_declarable"]
    assert row["path_text"].endswith("> Other > Of a weight not exceeding 80 kg")


def test_codes8_climbs_from_single_10_digit_child(codes):
    extra = pd.DataFrame(
        {
            "commodity__sid": ["1030", "1031"],
            "commodity__code": ["0101300000", "0101300010"],
            "commodity__suffix": ["80", "80"],
            "commodity__description": ["Asses", "Pure-bred asses"],
            "commodity__validity_start": ["2012-01-01"] * 2,
            "commodity__validity_end": [None, None],
            "parent__sid": ["1001", "1030"],
        }
    )
    raw = read_commodities_csv(FIXTURES / "commodities.csv")
    raw = pd.concat([raw[raw["commodity__sid"] != "1007"], extra], ignore_index=True)
    row = build_codes8(build_codes(raw, SECTIONS)).set_index("cn8").loc["01013000"]
    assert row["node_code10"] == "0101300000"
    assert list(row["children"]) == ["0101300010"]
    assert not row["cn8_declarable"]


def test_codes8_edge_cases(codes8):
    assert len(codes8) == 11
    grouped = codes8.loc["01061100"]  # leaves meet at a suffix-10 grouping line
    assert grouped["node_suffix"] == "10" and not grouped["node_above_cn8"]
    above = codes8.loc["01062010"]  # leaves hang straight off the 6-digit line
    assert above["node_above_cn8"] and above["node_code10"] == "0106200000"
    assert codes8.loc["82119200", "section_id"] == 15


def test_parse_section_and_ranges():
    note = parse_section(load_json("section_1.json"), "https://x/sections/1")
    assert (note["level"], note["id"], note["section_id"]) == ("section", "1", 1)
    assert note["title"] == "Live animals; animal products"
    assert note["note"].startswith("1. Any reference in this section")
    assert "\r" not in note["note"] and note["note_raw"]
    ranges = parse_section_ranges(load_json("sections.json"))
    assert len(ranges) == 21 and ranges[1] == ("01", "05") and ranges[21] == ("97", "99")


def test_parse_chapter():
    note = parse_chapter(load_json("chapter_01.json"), "https://x/chapters/01")
    assert (note["level"], note["id"], note["section_id"]) == ("chapter", "01", 1)
    assert note["title"] == "Live animals"
    assert "heading 0301, 0306" in note["note"]
    assert "](/headings" not in note["note"]


def test_run_checks_clean_and_broken(codes, codes8):
    ranges = parse_section_ranges(load_json("sections.json"))
    checks = run_checks(codes, codes8.reset_index(), SECTIONS, ranges)
    assert checks["counts"]["chapters"] == 2
    i = checks["integrity"]
    assert i["unresolved_parent_sid"] == 0 and i["declarable_without_chapter_path"] == 0
    assert i["chapters_without_exactly_one_section"] == [] and i["duplicate_sid_rows"] == 0
    assert i["parent_in_other_chapter_count"] == 0

    raw = read_commodities_csv(FIXTURES / "commodities.csv")
    raw.loc[raw["commodity__sid"] == "1013", "parent__sid"] = "9999"  # orphan a branch
    raw.loc[raw["commodity__sid"] == "2002", "parent__sid"] = "1001"  # cross-chapter parent
    broken = build_codes(raw, {"01": 1})
    i = run_checks(broken, build_codes8(broken), {"01": 1}, ranges)["integrity"]
    assert i["unresolved_parent_sid"] == 1
    assert i["declarable_without_chapter_path"] == 2
    assert i["parent_in_other_chapter_count"] == 1
    assert i["chapters_without_exactly_one_section"][0]["chapter"] == "82"


def api(handler, env=None, tmp_path=None, **kwargs):
    sleeps: list[float] = []
    client = TariffApi(
        tmp_path, env=env or {}, transport=httpx.MockTransport(handler), sleep=sleeps.append,
        **kwargs,
    )  # fmt: skip
    return client, sleeps


def test_api_unauthenticated_and_cached(tmp_path):
    calls = []

    def handler(req):
        calls.append(str(req.url))
        assert "authorization" not in req.headers
        return httpx.Response(200, json={"data": {"id": "1"}})

    client, _ = api(handler, tmp_path=tmp_path)
    assert client.base_url == API_BASE_OPEN
    assert client.get_json("/sections/1") == {"data": {"id": "1"}}
    assert client.get_json("/sections/1") == {"data": {"id": "1"}}
    assert calls == [API_BASE_OPEN + "/sections/1"]
    assert (tmp_path / "sections" / "1.json").exists()


def test_api_token_refreshed_on_401(tmp_path):
    tokens = iter(["t1", "t2"])
    seen = []

    def handler(req):
        if req.url.host == "auth.example":
            return httpx.Response(200, json={"access_token": next(tokens)})
        seen.append(req.headers["authorization"])
        if req.headers["authorization"] == "Bearer t1":
            return httpx.Response(401)
        return httpx.Response(200, json={"ok": True})

    client, _ = api(handler, env=ENV, tmp_path=tmp_path)
    assert client.base_url == API_BASE_AUTH
    assert client.get_json("/chapters/01") == {"ok": True}
    assert seen == ["Bearer t1", "Bearer t2"]


def test_api_404_none_and_retry(tmp_path):
    calls = {"n": 0}

    def handler(req):
        if req.url.path.endswith("/commodities/0101290000"):
            return httpx.Response(404)
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={})

    client, sleeps = api(handler, tmp_path=tmp_path)
    assert client.get_json("/commodities/0101290000", cache=False) is None
    assert client.get_json("/sections") == {}
    assert calls["n"] == 3 and len(sleeps) >= 2


def test_api_gives_up(tmp_path):
    client, _ = api(lambda r: httpx.Response(503), tmp_path=tmp_path, max_retries=1)
    with pytest.raises(RuntimeError, match="Giving up"):
        client.get_json("/sections")


def test_resolve_version_and_download(tmp_path):
    csv = (FIXTURES / "commodities.csv").read_bytes()
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if req.url.path.endswith("/versions"):
            ids = ["v4.0.999", "v4.0.1633", "v3.9.2000", "latest"]
            return httpx.Response(200, json={"versions": [{"id": i} for i in ids]})
        return httpx.Response(200, content=csv)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert resolve_latest_version(client) == "v4.0.1633"
        dest = tmp_path / "v4.0.1633" / "commodities-report.csv.gz"
        assert download_csv(client, "v4.0.1633", dest) is True
        assert download_csv(client, "v4.0.1633", dest) is False
    assert gzip.decompress(dest.read_bytes()) == csv
    assert len(calls) == 2
    assert len(read_commodities_csv(dest)) == 32


def test_chapter_section_map():
    notes = pd.DataFrame(
        [
            parse_section(load_json("section_1.json"), "u"),
            parse_chapter(load_json("chapter_01.json"), "u"),
        ]
    )
    assert chapter_section_map(notes) == {"01": 1}


def test_superseded_keeps_latest_row(codes, codes8):
    asses = codes[codes["code10"] == "0101300000"].set_index("sid")
    assert len(asses) == 2  # both rows stay in codes
    assert asses.loc[1008, "superseded"] and not asses.loc[1007, "superseded"]
    assert codes["superseded"].sum() == 1
    assert codes8.loc["01013000", "node_sid"] == 1007
    assert list(codes8.loc["01013000", "children"]) == ["0101300000"]


def test_superseded_rows_excluded_from_checks(codes, codes8):
    ranges = parse_section_ranges(load_json("sections.json"))
    checks = run_checks(codes, codes8.reset_index(), SECTIONS, ranges)
    assert checks["counts"]["superseded_lines"] == 1
    assert checks["counts"]["declarable_lines"] == 15
    assert checks["integrity"]["duplicate_code10_suffix_rows"] == 2
    assert checks["integrity"]["current_duplicate_code10_suffix_rows"] == 0


def test_check_superseded_against_api(codes, tmp_path):
    def handler(req):
        assert req.url.path.endswith("/commodities/0101300000")
        attrs = {"description": "Asses", "validity_start_date": "2012-01-01T00:00:00.000Z"}
        return httpx.Response(200, json={"data": {"id": "1007", "attributes": attrs}})

    client, _ = api(handler, tmp_path=tmp_path)
    [result] = check_superseded(client, codes)
    assert result["kept_sid"] == 1007 and result["superseded_sids"] == [1008]
    assert result["sid_matches"] and result["validity_start_matches"]
    assert result["description_matches"]
