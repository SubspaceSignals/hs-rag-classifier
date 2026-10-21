# Fixture atoz/refs_fixture.json is synthetic (fixture sids, fake 8888/9999), not the real search references.
import json
import re
from pathlib import Path

import httpx
import pandas as pd
import pytest

from hs_rag_classifier.data import atoz as A
from hs_rag_classifier.data.nomenclature import (
    TariffApi,
    build_codes,
    notes_input_files,
    read_commodities_csv,
)
from hs_rag_classifier.guards import FORBIDDEN_DIRS, ForbiddenInputError, check_retrieval_inputs

FIXTURES = Path(__file__).parent / "fixtures"
SRC = Path(__file__).resolve().parents[1] / "src" / "hs_rag_classifier"


@pytest.fixture(scope="module")
def resolved():
    codes = build_codes(read_commodities_csv(FIXTURES / "tariff" / "commodities.csv"))
    payload = json.loads((FIXTURES / "atoz" / "refs_fixture.json").read_text())
    return A.resolve(A.parse_search_references(payload), codes).set_index("id", drop=False)


def test_parse_search_references():
    payload = json.loads((FIXTURES / "atoz" / "refs_fixture.json").read_text())
    refs = A.parse_search_references(payload)
    assert len(refs) == 20
    row = refs.set_index("id").loc["5"]
    assert (row["target_code10"], row["target_suffix"], row["target_sid"]) == (
        "0102291000",
        "80",
        1013,
    )
    assert row["referenced_class"] == "Subheading"


def test_significant_depth():
    assert A.significant_depth("0100000000") == 2
    assert A.significant_depth("0101000000") == 4
    assert A.significant_depth("0101290000") == 6
    assert A.significant_depth("0102291000") == 8
    assert A.significant_depth("0102291010") == 10
    assert A.significant_depth("8211920000") == 6
    assert A.significant_depth("0101210000") == 6  # declarable lines get 10 in resolve()


def test_resolve_depths_and_labels(resolved):
    r = resolved
    assert (r.loc["1", "target_depth"], r.loc["2", "target_depth"]) == (2, 4)
    assert pd.isna(r.loc["1", "label_depth"]) and pd.isna(r.loc["2", "label_depth"])
    declarable = r.loc["3"]
    assert declarable["target_depth"] == 10 and declarable["label_depth"] == 10
    assert (declarable["code10"], declarable["cn8"]) == ("0101210000", "01012100")
    six = r.loc["4"]  # two cn8s beneath
    assert six["target_depth"] == 6 and not six["cn8_unambiguous"] and pd.isna(six["label_depth"])
    eight = r.loc["5"]  # 8-digit node with one cn8 beneath
    assert eight["target_depth"] == 8 and eight["label_depth"] == 8 and eight["cn8"] == "01022910"
    assert pd.isna(eight["code10"])
    grouping = r.loc["6"]  # suffix-10 line covering two cn8s
    assert grouping["target_depth"] == 8 and pd.isna(grouping["label_depth"])
    below = r.loc["7"]  # 6-digit node with one cn8 beneath: labelled at 8 from below
    assert below["cn8_unambiguous"] and below["cn8"] == "82119200"
    assert below["label_depth"] == 8 and below["label_source"] == "single_cn8_below"


def test_label_source(resolved):
    r = resolved
    assert r.loc["3", "label_source"] == "direct"  # declarable line
    assert r.loc["5", "label_source"] == "direct"  # 8-digit node, one cn8 beneath
    assert r.loc["7", "label_source"] == "single_cn8_below"
    for unlabelled in ("1", "2", "4", "6", "8"):  # several cn8s beneath, or not current
        assert pd.isna(r.loc[unlabelled, "label_source"])
        assert pd.isna(r.loc[unlabelled, "label_depth"])


def test_resolve_status_and_fallbacks(resolved):
    assert resolved.loc["8", "status"] == "not_current"
    assert pd.isna(resolved.loc["8", "label_depth"])
    assert resolved.loc["9", "resolved_sid"] == 2004  # unknown sid: matched by code and suffix
    assert resolved.loc["11", "resolved_sid"] == 1007  # superseded sid: current row instead
    assert not resolved["sid_mismatch"].any()


def test_heading_carrying_sub_chapter_title_sid_resolves_to_heading():
    """Search references for some headings carry the sid of the TARIC sub-chapter title above
    them (heading 6309, suffix 80, sid of 6309000000/10); the code and suffix win."""
    codes = build_codes(read_commodities_csv(FIXTURES / "tariff" / "misparented.csv"))
    ref = {"referenced_class": "Heading", "productline_suffix": "80", "usage": "search"}
    payload = {
        "data": [
            {"id": "1", "attributes": {**ref, "title": "second-hand clothes",
             "referenced_id": "6309", "goods_nomenclature_item_id": "6309000000",
             "goods_nomenclature_sid": 3001}},
            {"id": "2", "attributes": {**ref, "title": "rags", "referenced_id": "6310",
             "goods_nomenclature_item_id": "6310000000", "goods_nomenclature_sid": 3003}},
        ]
    }  # fmt: skip
    r = A.resolve(A.parse_search_references(payload), codes).set_index("id")
    assert r.loc["1", "resolved_sid"] == 3002 and r.loc["1", "sid_mismatch"]
    assert (r.loc["1", "code10"], r.loc["1", "label_depth"]) == ("6309000000", 10)
    assert r.loc["1", "label_source"] == "direct"
    assert r.loc["2", "resolved_sid"] == 3003 and not r.loc["2", "sid_mismatch"]
    assert pd.isna(r.loc["2", "label_depth"])  # two cn8s beneath heading 6310


def test_select_test8_dedupes_and_drops_conflicts(resolved):
    test, conflicts = A.select_test8(resolved.reset_index(drop=True))
    assert list(test["id"]) == ["3", "5", "7", "9", "13", "16"]
    assert list(test.set_index("id").loc["9", "member_ids"]) == ["9", "10"]
    assert sorted(conflicts["id"], key=int) == ["11", "12", "17", "18", "19", "20"]
    assert set(test["label_depth"]) == {8, 10}
    assert list(test.columns) == A.TEST8_COLUMNS


def test_terms_agreeing_at_8_collapse_to_8(resolved):
    test, _ = A.select_test8(resolved.reset_index(drop=True))
    row = test.set_index("id").loc["13"]  # direct at 8 (13) and direct at 10 (14), same cn8
    assert list(row["member_ids"]) == ["13", "14"]
    assert row["label_depth"] == 8 and row["cn8"] == "01022910" and pd.isna(row["code10"])
    assert row["label_source"] == "direct"


def test_single_cn8_below_confirms_but_never_coarsens(resolved):
    test, _ = A.select_test8(resolved.reset_index(drop=True))
    row = test.set_index("id").loc["16"]  # 15 is single_cn8_below with a lower id
    assert list(row["member_ids"]) == ["15", "16"]
    assert row["label_depth"] == 10 and row["code10"] == "8211920010"
    assert row["label_source"] == "direct"


def test_conflicts_checked_at_scored_depth(resolved):
    _, conflicts = A.select_test8(resolved.reset_index(drop=True))
    titles = set(conflicts["title"])
    assert "steak knives" in titles  # direct 82119100 vs single_cn8_below 82119200
    assert "primates" in titles  # same cn8, but both scored at 10 with different code10s


def test_write_ref_files_guard(resolved, tmp_path):
    test, _ = A.select_test8(resolved.reset_index(drop=True))
    path = A.write_ref_files(test, tmp_path)
    assert path.read_text() == "3\n5\n7\n9\n10\n13\n14\n15\n16\n"
    A.write_ref_files(test, tmp_path)  # unchanged: fine
    path.write_text("1\n")
    with pytest.raises(RuntimeError, match="frozen"):
        A.write_ref_files(test, tmp_path)
    assert path.read_text() == "1\n"
    A.write_ref_files(test, tmp_path, refreeze=True)
    assert path.read_text() == "3\n5\n7\n9\n10\n13\n14\n15\n16\n"


def test_ruling_overlap(tmp_path):
    pd.DataFrame({"description": ["Table knives.", "A spoon"]}).to_parquet(
        tmp_path / "test.parquet"
    )
    pd.DataFrame({"description": ["forks"]}).to_parquet(tmp_path / "train.parquet")
    out = A.ruling_overlap(pd.Series(["table knives", "forks", "plates"]), tmp_path)
    assert out == {"train": 1, "test": 1}


def test_endpoint_fetched_once_and_cached(tmp_path):
    payload = json.loads((FIXTURES / "atoz" / "refs_fixture.json").read_text())
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json=payload)

    api = TariffApi(tmp_path, env={}, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    assert api.get_json(A.ENDPOINT) == payload
    assert api.get_json(A.ENDPOINT) == payload
    assert calls == ["/uk/api/search_references"]
    assert (tmp_path / "search_references.json").exists()


WALK_RE = re.compile(r"\.rglob\(|\.glob\(|os\.walk|\.iterdir\(|os\.listdir|scandir|glob\.glob")
PATH_RE = re.compile(r"(raw|processed)\s*(/|[\"']\s*/\s*[\"'])\s*atoz")


def leak_violations(text: str) -> list[str]:
    """Static guard for one module outside ALLOWED_READERS (see the test below)."""
    found = []
    for name in ("atoz", "search_references"):  # (a) names the A to Z data
        if name in text:
            found.append(f"mentions {name}")
    if PATH_RE.search(text):  # (b) builds a raw/atoz or processed/atoz path
        found.append("references a test-only path")
    if WALK_RE.search(text) and "check_retrieval_inputs" not in text:  # (c) walks folders
        found.append("walks folders without check_retrieval_inputs")
    return found


def test_leak_scanner_catches_each_case():
    assert leak_violations('p = root / "processed" / "atoz" / "v1"')
    assert "references a test-only path" in leak_violations('p = root / "raw" / "atoz"')
    assert "references a test-only path" in leak_violations('p = "raw/atoz/x.json"')
    walk = 'files = sorted(root.rglob("*.json"))'
    assert leak_violations(walk) == ["walks folders without check_retrieval_inputs"]
    guarded = walk + "\nfiles = check_retrieval_inputs(files, root)"
    assert leak_violations(guarded) == []
    assert leak_violations('p = root / "raw" / "tariff" / "api" / "sections.json"') == []


def test_search_references_never_reach_retrieval():
    """Search references are a test set only; never index them for retrieval.

    Every module under src/ outside ALLOWED_READERS fails if it (a) mentions atoz or
    search_references, (b) builds a raw/atoz or processed/atoz path, or (c) globs or walks
    folders without passing the result through guards.check_retrieval_inputs(). Whether a
    glob's root contains raw/atoz can't be decided statically, so (c) is enforced by requiring
    the runtime check rather than by inspecting the glob.
    """
    assert A.NEVER_INDEX_FOR_RETRIEVAL is True
    assert set(FORBIDDEN_DIRS) <= set(A.RETRIEVAL_FORBIDDEN)
    assert A.RAW_DIR.as_posix() in FORBIDDEN_DIRS
    offenders = {}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if any(rel == a or rel.startswith(a) for a in A.ALLOWED_READERS):
            continue
        found = leak_violations(path.read_text(encoding="utf-8"))
        if found:
            offenders[rel] = found
    assert not offenders, f"search references must never be indexed for retrieval: {offenders}"


def test_check_retrieval_inputs(tmp_path):
    ok = [
        tmp_path / "raw" / "tariff" / "api" / "chapters" / "01.json",
        tmp_path / "raw" / "tariff" / "v4.0.1633" / "commodities-report.csv.gz",
        tmp_path / "processed" / "tariff" / "v4.0.1633" / "codes.parquet",
        tmp_path / "processed" / "rulings" / "v1" / "train.parquet",
        tmp_path / "raw" / "atoz_notes.txt",  # a sibling with a similar name is fine
    ]
    assert check_retrieval_inputs(ok, tmp_path) == ok
    for bad in (
        tmp_path / "raw" / "atoz" / "search_references.json",
        tmp_path / "processed" / "atoz" / "v1" / "atoz_test8.parquet",
        tmp_path / "processed" / "atoz",
        tmp_path / "raw" / "tariff" / ".." / "atoz" / "x.json",  # resolved before checking
    ):
        with pytest.raises(ForbiddenInputError):
            check_retrieval_inputs([*ok, bad], tmp_path)


def test_nomenclature_manifest_inputs_exclude_atoz(tmp_path):
    api_dir = tmp_path / "raw" / "tariff" / "api"
    for name in ("sections.json", "sections/1.json", "chapters/01.json", "search_references.json"):
        (api_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (api_dir / name).write_text("{}")
    (tmp_path / "raw" / "atoz").mkdir(parents=True)
    (tmp_path / "raw" / "atoz" / "search_references.json").write_text("{}")
    files = notes_input_files(api_dir, {1: ("01", "05")}, ["01", "02"])
    names = [f.relative_to(tmp_path).as_posix() for f in files]
    assert names == [
        "raw/tariff/api/sections.json",
        "raw/tariff/api/sections/1.json",
        "raw/tariff/api/chapters/01.json",
    ]  # named files only: a stray search_references.json in the API folder isn't swept in
    assert check_retrieval_inputs(files, tmp_path) == files


REAL_MANIFEST = (
    Path.home() / "data" / "hs-rag-classifier" / "processed" / "tariff" / "v4.0.1633"
    / "manifest.json"
)  # fmt: skip


@pytest.mark.skipif(not REAL_MANIFEST.exists(), reason="local nomenclature build not present")
def test_real_nomenclature_manifest_has_no_atoz_files():
    raw = json.loads(REAL_MANIFEST.read_text())["sha256"]["raw"]
    bad = [p for p in raw if p.startswith(("raw/atoz", "processed/atoz")) or "search_ref" in p]
    assert raw and not bad
