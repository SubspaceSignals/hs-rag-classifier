from pathlib import Path

import httpx
import pandas as pd
import pyarrow.parquet as pq
import pytest

from hs_rag_classifier.data import rulings as R
from hs_rag_classifier.data.nomenclature import TariffApi, build_codes, read_commodities_csv

FIXTURES = Path(__file__).parent / "fixtures"
RULINGS = FIXTURES / "rulings" / "rulings.jsonl"
# what the fake API says was declarable on the ruling's start date
VALID_THEN = {"0101300000", "8211920000", "0106300000"}


def fixture_codes():
    """Tariff fixture plus one child under 0101300000, so that line is no longer declarable."""
    raw = read_commodities_csv(FIXTURES / "tariff" / "commodities.csv")
    extra = pd.DataFrame(
        {
            "commodity__sid": ["1032"],
            "commodity__code": ["0101300010"],
            "commodity__suffix": ["80"],
            "commodity__description": ["Pure-bred asses"],
            "commodity__validity_start": ["2025-01-01"],
            "commodity__validity_end": [None],
            "parent__sid": ["1007"],
        }
    )
    return build_codes(pd.concat([raw, extra], ignore_index=True), {"01": 1, "82": 15})


@pytest.fixture(scope="module")
def prepared():
    codes = fixture_codes()
    cn8 = set(codes.loc[codes["declarable"] & ~codes["superseded"], "cn8"])
    mp = pytest.MonkeyPatch()
    mp.setattr(R, "N_TEST", 2)
    mp.setattr(R, "N_VAL", 1)
    try:
        groups, audit, pairs = R.prepare(
            R.load_rulings(RULINGS), codes, cn8, lambda code, start: code in VALID_THEN
        )
    finally:
        mp.undo()
    return groups.set_index("ref", drop=False), audit.set_index("reference"), pairs


def test_normalise():
    assert (
        R.normalise("  Stainless-steel  KNIFE,\twith\nblade. ")
        == "stainless steel knife with blade"
    )
    assert R.normalise(None) == ""


def test_code_prefixes_in():
    assert R.code_prefixes_in("of heading 8211, see 8211.92", "8211920010") == ["8211", "821192"]
    assert R.code_prefixes_in("0102 29", "0102291010") == ["010229", "0102"]
    assert R.code_prefixes_in("size 120 mm, model 4202", "8211920010") == []
    assert R.code_prefixes_in("82", "8211920010") == []  # under 4 digits


def test_leakage_flags():
    f = R.leakage_flags(
        "A young male calf for fattening", ["0102 29", "BOVINE ANIMALS"], "0102291010",
        "LIVE ANIMALS > Live bovine animals > Other",
    )  # fmt: skip
    assert f["kw_code_prefix"] and not f["self_leak"]
    assert f["kw_path_words"] == ["animal", "bovine"]
    assert R.leakage_flags("x", ["BOVINE"], "0102291010", None)["kw_path_words"] == []


def test_exact_duplicates_grouped(prepared):
    groups, _, _ = prepared
    g = groups.loc["900000001"]
    assert list(g["member_refs"]) == ["900000001", "900000002"]
    assert "900000002" not in groups.index
    assert not g["conflict"]
    assert len(groups) == 16


def test_conflicts_flagged_and_train_only(prepared):
    groups, _, _ = prepared
    for ref in ("900000003", "900000004"):
        assert groups.loc[ref, "conflict"] and "conflict" in groups.loc[ref, "flags"]
        assert groups.loc[ref, "split"] == "train"


def test_near_duplicates_share_cluster_and_split(prepared):
    groups, _, pairs = prepared
    a, b = groups.loc["900000005"], groups.loc["900000006"]
    assert a["cluster_id"] == b["cluster_id"] == "900000005"
    assert a["split"] == b["split"]
    assert (groups.groupby("cluster_id")["split"].nunique() == 1).all()
    i = groups.index.get_loc("900000005")
    j = groups.index.get_loc("900000006")
    assert pairs[(pairs["i"] == i) & (pairs["j"] == j)]["sim"].iloc[0] >= R.NEAR_DUP_THRESHOLD


def test_similar_pairs_and_components():
    texts = ["red cotton t shirt for men", "red cotton t shirt for women", "steel bolt m8"]
    pairs = R.similar_pairs(texts, min_sim=0.5)
    assert set(zip(pairs["i"], pairs["j"], strict=True)) == {(0, 1)}
    labels = R.components(3, pairs, threshold=0.5)
    assert labels[0] == labels[1] != labels[2]
    assert len(set(R.components(3, pairs, threshold=0.999))) == 3


def test_code_status(prepared):
    groups, _, _ = prepared
    s = groups["code_status"].to_dict()
    assert s["900000001"] == "current"
    assert s["900000012"] == "current" and "short_code:82111000" in groups.loc["900000012", "flags"]
    assert groups.loc["900000012", "code10"] == "8211100000"
    assert s["900000011"] == "cn8_current" and groups.loc["900000011", "split"] == "train"
    assert s["900000010"] == "never_valid" and groups.loc["900000010", "split"] == "excluded"
    mapped = groups.loc["900000009"]
    assert mapped["code_status"] == "retired" and mapped["code10"] == "0101300010"
    assert "mapped_from:0101300000" in mapped["flags"] and mapped["split"] == "train"
    assert s["900000014"] == "retired" and groups.loc["900000014", "split"] == "excluded"


def test_label_depth(prepared):
    groups, _, _ = prepared
    d = groups["label_depth"].to_dict()
    assert d["900000001"] == 10 and d["900000012"] == 10  # current, incl. a short code
    assert d["900000009"] == 10  # retired, cleanly mapped
    assert d["900000011"] == 8  # cn8_current
    assert pd.isna(d["900000010"]) and pd.isna(d["900000014"])  # no current cn8: excluded


def test_retired_or_never_valid_with_current_cn8_is_train_only_at_8_digits(prepared):
    groups, _, _ = prepared
    retired = groups.loc["900000013"]  # 8211920000 split into two codes; cn8 82119200 current
    assert retired["code_status"] == "retired" and retired["label_depth"] == 8
    assert retired["split"] == "train" and retired["cn8"] == "82119200"
    never = groups.loc["900000015"]  # 8211910099 never valid; cn8 82119100 current
    assert never["code_status"] == "never_valid" and never["label_depth"] == 8
    assert never["split"] == "train"


def test_near_dup_threshold_is_085(prepared):
    groups, _, pairs = prepared
    assert R.NEAR_DUP_THRESHOLD == 0.85
    i = groups.index.get_loc("900000016")
    j = groups.index.get_loc("900000017")
    sim = pairs[(pairs["i"] == i) & (pairs["j"] == j)]["sim"].iloc[0]
    assert 0.85 <= sim < 0.90  # would be split apart at the old 0.90 threshold
    assert groups.loc["900000016", "cluster_id"] == groups.loc["900000017", "cluster_id"]
    assert groups.loc["900000016", "split"] == groups.loc["900000017", "split"]


def test_leakage_flags_in_outputs(prepared):
    groups, audit, _ = prepared
    assert audit.loc["900000007", "self_leak"]
    assert groups.loc["900000007", "split"] == "train"
    assert {"kw_code_prefix", "kw_path_words"} <= set(groups.loc["900000008", "flags"])


def test_split_sizes_and_eligibility(prepared):
    groups, _, _ = prepared
    assert (groups["split"] == "test").sum() == 2 and (groups["split"] == "val").sum() == 1
    held_out = groups[groups["split"].isin(["val", "test"])]
    assert (held_out["code_status"] == "current").all()
    assert (held_out["label_depth"] == 10).all()
    assert not held_out["conflict"].any() and not held_out["self_leak"].any()


def test_assign_splits_exact_and_stratified():
    units = pd.DataFrame(
        {
            "cluster_id": [f"c{i}" for i in range(200)],
            "chapter": ["01"] * 100 + ["84"] * 100,
            "size": [3] * 10 + [1] * 90 + [1] * 100,
        }
    )
    out = R.assign_splits(units, n_test=40, n_val=20, seed=1)
    units["split"] = units["cluster_id"].map(out)
    sizes = units.groupby("split")["size"].sum()
    assert sizes["test"] == 40 and sizes["val"] == 20
    per = units.groupby(["split", "chapter"])["size"].sum()
    assert abs(per["test", "01"] - per["test", "84"]) <= 4
    assert out == R.assign_splits(units.drop(columns="split"), 40, 20, seed=1)  # deterministic


def test_api_valid_on_uses_as_of_and_caches(tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url)
        if req.url.params.get("as_of") == "2024-05-01":
            attrs = {"declarable": True, "description": "Asses"}
            return httpx.Response(200, json={"data": {"id": "1", "attributes": attrs}})
        return httpx.Response(404)

    api = TariffApi(tmp_path, env={}, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    valid_on = R.api_valid_on(api)
    assert valid_on("0101300000", "2024-05-01")
    assert not valid_on("0101300000", "2020-01-01")
    assert valid_on("0101300000", "2024-05-01") and not valid_on("0101300000", "2020-01-01")
    assert len(calls) == 2  # both answers, 404 included, came from the cache the second time
    assert (tmp_path / "commodities" / "0101300000__as_of=2020-01-01.json").read_text() == "null"


def test_split_files_have_no_justification(prepared, tmp_path):
    groups, _, _ = prepared
    rulings = R.load_rulings(RULINGS)
    R.write_split_files(groups.reset_index(drop=True), rulings, tmp_path)
    for s in R.SPLITS:
        cols = pq.read_schema(tmp_path / f"{s}.parquet").names
        assert "justification" not in cols and cols == R.OUTPUT_COLUMNS
    assert pq.read_schema(tmp_path / "justifications.parquet").names == ["ref", "justification"]


def test_ref_files_refuse_to_change_frozen_test(prepared, tmp_path):
    groups, _, _ = prepared
    groups = groups.reset_index(drop=True)
    refs = R.write_ref_files(groups, tmp_path)
    assert refs["train"] == sorted(refs["train"], key=int)
    assert R.write_ref_files(groups, tmp_path) == refs  # same split: fine
    (tmp_path / "test_refs.txt").write_text("123\n")
    with pytest.raises(RuntimeError, match="frozen"):
        R.write_ref_files(groups, tmp_path)
    assert R.write_ref_files(groups, tmp_path, refreeze=True) == refs  # explicit decision only
    assert (tmp_path / "test_refs.txt").read_text() == "\n".join(refs["test"]) + "\n"
