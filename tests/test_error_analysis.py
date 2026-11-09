"""Error analysis on small synthetic predictions: type precedence, the "Other" rule, the two
hard checks, unclassifiable items and the tables. No model, no MLflow, no saved runs."""

import json

import pandas as pd
import pytest

from hs_rag_classifier.eval import error_analysis as E

OPTS = ["821191", "821192", "010121", "391590"]
DESC = {"82119100": "Table knives having fixed blades", "82119200": "Other knives having fixed blades",
        "82119300": "- - Other", "82119400": "Blades", "01012100": "Pure-bred breeding animals",
        "01012910": "Horses for slaughter", "01012990": "Other", "39159011": "Of polymers",
        "39159080": "other"}  # fmt: skip


def frame(rows):
    """rows: (item_id, cluster_id, gold, pred, options)."""
    return pd.DataFrame([{"item_id": i, "cluster_id": c, "label_code": g, "top1": p,
                          "details": json.dumps({"stage1": {"options": o}}),
                          "correct_8": float(p is not None and g[:8] == p[:8])}
                         for i, c, g, p, o in rows])  # fmt: skip


# --- rules -------------------------------------------------------------------------------


def test_type_precedence():
    assert E.classify("8211910000", "82119100", OPTS) is None
    assert E.classify("3915901100", "82119100", ["821191"]) == "retrieval_miss"
    # a retrieval miss wins even when the heading is also wrong
    assert E.classify("3915901100", "01012100", ["821191", "010121"]) == "retrieval_miss"
    assert E.classify("8211910000", "01012100", OPTS) == "wrong_heading"  # wrong chapter too
    assert E.classify("8211910000", "82119200", OPTS) == "right_hs4_wrong_hs6"
    assert E.classify("0101210000", "01012990", OPTS + ["010129"]) == "right_hs4_wrong_hs6"
    assert E.classify("0101299000", "01012910", ["010129"]) == "right_hs6_wrong_cn8"
    assert E.classify("39159011", "39159080", OPTS) == "right_hs6_wrong_cn8"  # 8-digit gold


def test_final_top1_not_stage1_argmax_is_used():
    # options listed with the right HS6 first; the final top-1 is in another HS6
    assert E.classify("8211910000", "82119200", ["821191", "821192"]) == "right_hs4_wrong_hs6"


def test_unclassifiable_items_get_no_type():
    for args, msg in ((("8211", "82119100", OPTS), "gold"), (("8211910000", None, OPTS), "top-1"),
                      (("8211910000", "8211", OPTS), "top-1"), (("8211910000", "82119200", []), "options")):  # fmt: skip
        with pytest.raises(E.Unclassifiable, match=msg):
            E.classify(*args)
    t = E.item_table(frame([("1", "1", "8211910000", None, OPTS)]), DESC)
    assert (
        pd.isna(t.loc[0, "type"])
        and pd.isna(t.loc[0, "correct"])
        and "top-1" in t.loc[0, "unclassified"]
    )


def test_other_rule():
    assert (
        E.is_other("Other")
        and E.is_other("other")
        and E.is_other("- - Other")
        and E.is_other(": Other")
    )
    assert not E.is_other("Of other textile materials")  # 'other' not at the start
    assert E.is_other("Others, n.e.s.")  # starts with Other
    with pytest.raises(E.Unclassifiable):
        E.is_other(None)
    assert E.other_split("82119200", "82119100", DESC) == "gold_only"
    assert E.other_split("82119100", "82119300", DESC) == "pred_only"
    assert E.other_split("01012990", "82119300", DESC) == "both"
    assert E.other_split("82119100", "82119400", DESC) == "neither"
    t = E.item_table(frame([("1", "1", "0101299000", "01012910", ["010129"]),
                            ("2", "2", "39159080", "39159011", OPTS),
                            ("3", "3", "8211930000", "82119350", ["821193"])]), DESC)  # fmt: skip
    assert list(t["other_split"][:2]) == ["gold_only", "gold_only"]
    assert pd.isna(t.loc[2, "type"]) and "names this cn8" in t.loc[2, "unclassified"]


# --- hard checks -------------------------------------------------------------------------


ROWS3 = [("1", "c1", "8211910000", "82119100", OPTS),   # right
         ("2", "c1", "8211910000", "82119200", OPTS),   # right_hs4_wrong_hs6 -> fixed by A4
         ("3", "c3", "3915901100", "82119100", ["821191"]),  # retrieval miss for both
         ("4", "c4", "0101210000", "01012100", OPTS),   # right -> broke by A4
         ("5", "c5", "0101299000", "01012910", ["010129"])]  # fmt: skip
ROWS4 = [("1", "c1", "8211910000", "82119100", OPTS),
         ("2", "c1", "8211910000", "82119100", OPTS),
         ("3", "c3", "3915901100", "82119100", ["821191"]),
         ("4", "c4", "0101210000", "82119100", OPTS),
         ("5", "c5", "0101299000", "01012990", ["010129"])]  # fmt: skip


def test_hard_checks_pass_and_tables():
    res = E.analyse(frame(ROWS3), frame(ROWS4), DESC, n_resamples=50)
    hc = res["hard_checks"]
    assert hc["retrieval_miss_identical"] and hc["retrieval_miss_a3"] == 1 and hc["fixed"] == 2
    types = res["types"].set_index("type")
    assert types.loc["retrieval_miss", ["a3_n", "a4_n"]].tolist() == [1, 1]
    assert types.loc["right_hs4_wrong_hs6", "a3_n"] == 1 and types.loc["wrong_heading", "a4_n"] == 1
    assert types.loc["wrong_total", ["a3_n", "a4_n"]].tolist() == [3, 2]
    assert types.loc["retrieval_miss", "a4_share_wrong"] == 0.5
    fb = res["fixed_broke"]
    assert (fb["fixed"], fb["broke"]) == (2, 1)
    assert (
        fb["fixed_by_a3_type"]["right_hs4_wrong_hs6"] == 1
        and fb["fixed_by_a3_type"]["right_hs6_wrong_cn8"] == 1
    )
    assert fb["broke_by_a4_type"]["wrong_heading"] == 1 and fb["broke_a4_same_hs6_as_gold"] == 0
    c = res["ceiling"]
    assert c["hs6_recall_at_m"] == 0.8 and c["a4_share_of_wrong_retrieval_miss"] == 0.5
    assert len(E.ceiling_sentences(c, "x")) == 2
    ch = res["chapters"]
    assert ch["chapter"].iloc[0] in ("01", "39", "82") and ch["small"].all()


def test_hard_check_retrieval_miss_must_be_identical():
    rows4 = list(ROWS4)
    rows4[2] = ("3", "c3", "3915901100", "82119100", ["821191", "391590"])  # A4 saw other options
    with pytest.raises(E.HardCheckError, match="options differ"):
        E.analyse(frame(ROWS3), frame(rows4), DESC, n_resamples=10)
    t3, t4 = E.item_table(frame(ROWS3), DESC), E.item_table(frame(ROWS4), DESC)
    t4.loc[2, "type"] = "wrong_heading"  # same options, but the miss is not recorded for A4
    with pytest.raises(E.HardCheckError, match="retrieval_miss differs"):
        E.hard_checks(t3, t4)


def test_hard_check_no_fixed_item_is_an_a4_retrieval_miss():
    t3, t4 = E.item_table(frame(ROWS3), DESC), E.item_table(frame(ROWS4), DESC)
    t3.loc[2, "type"] = "retrieval_miss"
    t4.loc[2, ["correct", "type"]] = [True, "retrieval_miss"]  # impossible unless something broke
    with pytest.raises(E.HardCheckError, match="fixed items that are A4 retrieval misses"):
        E.hard_checks(t3, t4)


def test_classifier_must_agree_with_correct_8():
    f = frame(ROWS3)
    f.loc[0, "correct_8"] = 0.0
    with pytest.raises(E.HardCheckError, match="correct_8"):
        E.analyse(f, frame(ROWS4), DESC, n_resamples=10)


def test_examples_are_seeded_and_capped():
    t4 = E.item_table(frame(ROWS4 * 1), DESC)
    a = E.examples(t4, {"3": "x" * 300}, {"39159011": "path"}, k=10, seed=1)
    b = E.examples(t4, {"3": "x" * 300}, {"39159011": "path"}, k=10, seed=1)
    assert a.equals(b) and len(a.loc[a["type"] == "retrieval_miss"]) == 1
    assert a.loc[a["item_id"] == "3", "input"].iloc[0] == "x" * 200


def test_find_runs(tmp_path):
    log = tmp_path / "t.jsonl"
    lines = [{"arm": "a3", "dataset": "rulings", "mlflow_run_id": "r13"},
             {"arm": "b0", "dataset": "rulings", "mlflow_run_id": "x"},
             {"arm": "a4", "dataset": "atoz", "mlflow_run_id": "r16"},
             {"arm": "a4", "dataset": "hscodecomp_c", "mlflow_run_id": "h"}]  # fmt: skip
    log.write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert E.find_runs(log) == {("rulings", "a3"): (1, "r13"), ("atoz", "a4"): (3, "r16")}
    log.write_text(log.read_text() + json.dumps(lines[0]) + "\n")
    with pytest.raises(E.HardCheckError, match="more than one"):
        E.find_runs(log)
