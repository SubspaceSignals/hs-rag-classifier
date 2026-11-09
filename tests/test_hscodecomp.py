"""HSCodeComp at HS6: pinned loader, labels, inputs, HS6 metrics, review orders and guards.

Synthetic rows only (the real file lives under the data root and is never in the repo). The
loader refuses a file or inputs config with another SHA-256; labels are HS6 strings (chapter
01-09 zero padding included) and no US digit 7-10 is reachable from an item; both inputs follow
the frozen rule (category before attributes, drop-list, no _cn, question, price or currency);
HS6, HS4, HS2 and top-3 scoring, the stage-1 confidence of A3/A4, and the three review orders
with their tie-breaks; the folder is refused for retrieval, statically and at run time; the two
dataset names are test-only and a test-split run logs one test_runs.jsonl line."""

import copy
import hashlib
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from test_atoz import SRC, WALK_RE
from test_eval_harness import CLEAN, VALID8, VALID10, FixedArm

from hs_rag_classifier.data import hscodecomp as H
from hs_rag_classifier.eval import items as I
from hs_rag_classifier.eval import metrics_hs6 as MH
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.arms import Candidate, Prediction
from hs_rag_classifier.eval.items import EvalItem
from hs_rag_classifier.guards import FORBIDDEN_DIRS, ForbiddenInputError, check_retrieval_inputs
from hs_rag_classifier.pipeline import scoring

SENTINEL = {"question": "QUESTION-SENTINEL", "price": 98765.4, "currency_code": "XQZ"}


def row(task_id, hs_code, title="Steel table knife", attrs=None, cats=None, **kw):
    cats = cats or ["Home & Garden", "Kitchen", "Knives", "Knives", "Knives"]
    r = {"product_name": title, "product_attributes": json.dumps(attrs if attrs is not None else
         {"Material": "Steel", "Origin": "Mainland China"}),
         "price": SENTINEL["price"], "currency_code": SENTINEL["currency_code"],
         "hs_code": hs_code, "task_id": task_id, "question": SENTINEL["question"]}  # fmt: skip
    for i, c in enumerate(cats, 1):
        r[f"cate_lv{i}_desc"] = c
        r[f"cate_lv{i}_desc_cn"] = f"中文{i}"
    return {**r, **kw}


ROWS = [
    row(0, 8211910000),
    row(
        1,
        101210000,
        title="Pure-bred breeding horse",  # chapter 01: 9 digits as an int
        attrs={
            "Material": "None",
            "Brand Name": "NoEnName_Null",
            "Package weight": "1.2",
            "Package size - length (cm)": "10",
            "Measurement unit": "100000015",
            "each pack": "1",
            "Sold in": "sell_by_piece",
            "CN": "Guangdong",
            "Hign-concerned Chemical": "None",
            "Item Type": "Horse",
            "Colour": "",
            "Size": "/",
        },
        cats=["Animals", "Horses", "", "Horses", "Breeding"],
    ),
    row(
        2,
        8211910000,
        title="Knife with brand",
        attrs={"Brand Name": "Rosediy", "Metals Type": "Steel"},
    ),
    row(10, 8211910000, title="Knife, logistics only", attrs={"Origin": "CN", "each pack": "2"}),
]


def write_set(tmp_path, rows=ROWS):
    root = tmp_path / "data"
    path = root / "hscodecomp" / "download" / "data" / "test_data.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    cfg = json.loads(H.INPUTS_CONFIG.read_text(encoding="utf-8"))
    cfg["dataset"] = {**cfg["dataset"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "n_items": len(rows)}  # fmt: skip
    cfg_path = tmp_path / "inputs.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return root, path, cfg_path, hashlib.sha256(cfg_path.read_bytes()).hexdigest()


def load_set(tmp_path, variant="c", rows=ROWS):
    root, _, cfg, sha = write_set(tmp_path, rows)
    return I.load_hscodecomp(variant, root, cfg, sha)


# --- pinned files ------------------------------------------------------------------------


def test_inputs_config_is_pinned_and_records_the_data_file():
    assert hashlib.sha256(H.INPUTS_CONFIG.read_bytes()).hexdigest() == H.INPUTS_CONFIG_SHA256
    cfg = H.load_config()
    ds = cfg["dataset"]
    assert ds["commit"] == "ce9119795acef4ca537b2175e10a3feb7a0ecae9"
    assert ds["sha256"] == "8f5cf65150c04df10bbcd44985031bd2267a5658578f374102c95244706a9f3c"
    assert (ds["n_items"], ds["split"], ds["scoring_depth"]) == (632, "test", 6)
    for key in ("Package weight", "Measurement unit", "each pack", "Sold in", "Origin", "CN",
                "Hign-concerned Chemical"):  # fmt: skip
        assert key in cfg["drop_keys"]
    for key in ("Material", "Item Type", "Metals Type"):
        assert key not in cfg["drop_keys"]
    assert cfg["drop_if_value"]["Brand Name"]["value"] == "NoEnName_Null"
    assert not set(cfg["drop_keys"]) & set(cfg["kept_ambiguous"])
    assert all(cfg["drop_keys"].values())  # every key has a reason
    with pytest.raises(H.HSCodeCompError, match="SHA-256"):
        H.load_config(H.INPUTS_CONFIG, "0" * 64)


def test_a_file_with_another_sha_is_refused(tmp_path):
    root, path, cfg, sha = write_set(tmp_path)
    assert len(I.load_hscodecomp("c", root, cfg, sha).items) == len(ROWS)
    path.write_text(path.read_text(encoding="utf-8") + "\n ", encoding="utf-8")
    with pytest.raises(H.HSCodeCompError, match="SHA-256"):
        I.load_hscodecomp("c", root, cfg, sha)
    edited = tmp_path / "edited.json"
    edited.write_text(cfg.read_text(encoding="utf-8").replace('"version": 1', '"version": 2'))
    with pytest.raises(H.HSCodeCompError, match="SHA-256"):
        I.load_hscodecomp("c", root, edited, sha)


# --- labels ------------------------------------------------------------------------------


def test_labels_are_hs6_strings_with_zero_padding():
    assert H.label_hs6(101210000) == "010121"  # chapter 01 stored as a 9-digit int
    assert H.label_hs6(7117199000) == "711719"
    assert H.label_hs6("0302110000") == "030211"
    assert H.label_hs6(101) == "000000"
    for bad in (12345678901, "12a4567890", True, -5, 3.5):
        with pytest.raises(H.HSCodeCompError):
            H.label_hs6(bad)


def test_items_carry_hs6_only(tmp_path):
    rows = [row(0, 7117194321), row(1, 304321987)]
    s = load_set(tmp_path, rows=rows)
    assert [(i.item_id, i.label_code, i.label_depth, i.chapter) for i in s.items] == [
        ("0", "711719", 6, "71"),
        ("1", "030432", 6, "03"),
    ]
    assert all(i.cluster_id == i.item_id and i.split == "test" for i in s.items)
    dump = s.model_dump_json()
    for us in ("7117194321", "194321", "4321", "0304321987", "304321987", "1987"):
        assert us not in dump  # no US digit 7-10 reachable from the set or an item
    rec = H.read_records(write_set(tmp_path / "r", rows)[1], hashlib.sha256(
        write_set(tmp_path / "s", rows)[1].read_bytes()).hexdigest())  # fmt: skip
    assert {f for f in rec[0].__dataclass_fields__} == {"task_id", "title", "category",
                                                        "attributes", "hs6"}  # fmt: skip
    with pytest.raises(ValueError):
        EvalItem(item_id="x", text="t", label_code="12345", label_depth=6, cluster_id="x",
                 chapter="12", dataset="hscodecomp_c", split="test")  # fmt: skip


def test_rows_must_be_well_formed(tmp_path):
    for rows, msg in (([row(0, 1), row(0, 2)], "unique"),
                      ([row(0, 1, product_attributes="not json")], "JSON"),
                      ([row("x", 1)], "task_id")):  # fmt: skip
        root, _, cfg, sha = write_set(tmp_path / msg, rows)
        with pytest.raises(H.HSCodeCompError, match=msg):
            I.load_hscodecomp("a", root, cfg, sha)


# --- inputs ------------------------------------------------------------------------------


def test_input_c_category_first_drop_list_and_blanks(tmp_path):
    texts = {i.item_id: i.text for i in load_set(tmp_path, "c").items}
    assert texts["0"] == ("Steel table knife\nCategory: Home & Garden > Kitchen > Knives\n"
                          "Attributes: Material: Steel")  # fmt: skip
    # category before attributes; repeated and empty levels dropped; the drop-list, the
    # NoEnName_Null brand and blank values ("", "/") removed; "None" kept as a fact
    assert texts["1"] == ("Pure-bred breeding horse\nCategory: Animals > Horses > Breeding\n"
                          "Attributes: Material: None; Item Type: Horse")  # fmt: skip
    assert texts["2"].endswith("Attributes: Brand Name: Rosediy; Metals Type: Steel")
    assert texts["10"] == "Knife, logistics only\nCategory: Home & Garden > Kitchen > Knives"
    for t in texts.values():
        assert t.index("Category: ") < t.find("Attributes: ") or "Attributes: " not in t
        assert "中文" not in t  # no _cn field


def test_input_a_is_the_title_only(tmp_path):
    texts = {i.item_id: i.text for i in load_set(tmp_path, "a").items}
    assert texts == {"0": "Steel table knife", "1": "Pure-bred breeding horse",
                     "2": "Knife with brand", "10": "Knife, logistics only"}  # fmt: skip


def test_question_price_currency_never_in_the_input(tmp_path):
    for v in ("c", "a"):
        s = load_set(tmp_path / v, v)
        dump = s.model_dump_json()
        for x in (SENTINEL["question"], str(SENTINEL["price"]), "98765", SENTINEL["currency_code"]):
            assert x not in dump
    assert set(H.NEVER_USED) >= {"question", "price", "currency_code", "cate_lv1_desc_cn"}


def test_dataset_names_are_test_only(tmp_path):
    assert set(H.DATASET_NAMES) == {"hscodecomp_c", "hscodecomp_a"}
    assert set(H.DATASET_NAMES) <= set(I.DATASETS) and set(H.DATASET_NAMES) <= set(I.TEST_ONLY)
    assert I.HS6_DATASETS == H.DATASET_NAMES
    for name in H.DATASET_NAMES:
        with pytest.raises(ValueError, match="only a test split"):
            I.load(name, "val")
    with pytest.raises(ValueError, match="unknown HSCodeComp input"):
        I.load_hscodecomp("b")


# --- HS6 scoring -------------------------------------------------------------------------


def items6(labels):
    return [EvalItem(item_id=str(n), text=t, label_code=lab, label_depth=6, cluster_id=str(n),
                     chapter=lab[:2], dataset="hscodecomp_c", split="test")
            for n, (lab, t) in enumerate(labels)]  # fmt: skip


def pred(i, codes, conf=0.5, details=None):
    return Prediction(item_id=str(i), candidates=[Candidate(code=c, score=1.0) for c in codes],
                      confidence=conf, details=details)  # fmt: skip


def test_hs6_hs4_hs2_and_top3_native():
    items = items6([("821191", "a"), ("821191", "bb"), ("821191", "c"), ("010121", "d")])
    preds = [pred(0, ["8211910000"]),                                  # right at 6
             pred(1, ["8211920000", "8211920000", "0101210000", "8211910000"]),  # 4 right; 3rd distinct HS6 wrong
             pred(2, ["8211929000", "8211910000"]),  # invalid top-1: wrong at top-1, right in top-3
             pred(3, ["0101290000", "0101210000"])]  # fmt: skip
    v8, v10 = (
        {"82119100", "82119200", "01012100", "01012900"},
        {"8211910000", "8211920000", "0101210000", "0101290000"},
    )
    f = MH.score(items, preds, v8, v10)
    assert list(f["correct_6"]) == [1.0, 0.0, 0.0, 0.0]
    assert list(f["correct_4"]) == [1.0, 1.0, 0.0, 1.0]
    assert list(f["correct_2"]) == [1.0, 1.0, 0.0, 1.0]
    assert f.loc[1, "ranked_hs6"] == ["821192", "010121", "821191"]  # first distinct HS6s
    assert list(f["recall_3"]) == [1.0, 1.0, 1.0, 1.0]
    assert f.loc[2, "ranked_hs6"] == ["821192", "821191"] and not f.loc[2, "top1_valid"]
    assert list(f["confidence_hs6"]) == [0.5] * 4
    m = MH.compute(f, n_resamples=20)
    assert m["top1"] == {2: 0.75, 4: 0.75, 6: 0.25} and m["recall_3"] == 1.0
    assert "b" not in m["orders"] and set(m["orders"]) == {"a", "length"}


def test_stage1_confidence_and_top3_for_a3_a4():
    lps = [-0.1, -2.0, -3.0, -4.0]
    det = {"stage1": {"options": ["821191", "821192", "010121", "821199"], "logprobs": lps},
           "a3": {"b1ex_top1": "8211920000"}}  # fmt: skip
    items = items6([("821192", "x")])
    p = pred(0, ["8211920000"], conf=0.99, details=det)  # top-1 HS6 is stage-1 option 2
    v8, v10 = {"82119100", "82119200", "01012100", "82119900"}, {"8211920000"}
    t1 = 1.3
    f = MH.score(items, [p], v8, v10, t1)
    want = scoring.renormalise(lps, t1)
    assert f.loc[0, "confidence_hs6"] == pytest.approx(float(want[1]), abs=0)
    assert f.loc[0, "ranked_hs6"] == ["821192", "821191", "010121", "821199"]
    assert (
        f.loc[0, "recall_3"] == 1.0 and f.loc[0, "b1ex_hs6"] == "821192" and f.loc[0, "agree_hs6"]
    )
    off = MH.score(items, [pred(0, ["8211990000"], details=det)], v8, {"8211990000"}, t1)
    assert off.loc[0, "ranked_hs6"][:3] == ["821199", "821191", "821192"]
    assert MH.stage1_temperature(type("A", (), {"name": "a3", "config": {
        "a2": {"temperature": {"stage1": 1.25}}}})()) == 1.25  # fmt: skip
    assert MH.stage1_temperature(FixedArm()) is None


def test_review_orders_and_tie_breaks():
    a = {"id_rank": np.array([0, 1, 2, 3, 4]),          # ids 1, 2, 10, 11, 20 in numeric order
         "confidence": np.array([0.5, 0.2, 0.5, 0.9, 0.2]),
         "agree_cn8": np.array([True, True, False, True, False]),
         "chars": np.array([30, 30, 10, 30, 99]),
         "correct": np.ones(5), "oof": np.zeros(5)}  # fmt: skip
    assert list(MH.review_order("a", a)) == [1, 4, 0, 2, 3]  # confidence, then id
    assert list(MH.review_order("b", a)) == [4, 2, 1, 0, 3]  # disagree first
    assert list(MH.review_order("length", a)) == [2, 1, 0, 3, 4]  # chars, confidence, id
    with pytest.raises(ValueError):
        MH.review_order("c", a)
    items = items6([("821191", "x" * 5), ("821191", "y" * 2000)])
    f = MH.score(items, [pred(0, ["8211910000"]), pred(1, ["8211910000"])], VALID8, VALID10)
    assert list(f["input_chars"]) == [5, 1000]  # characters sent, after the 1,000 cut


def test_policy_b_needs_b1ex_for_every_item():
    det = {"stage1": {"options": ["821191"], "logprobs": [-0.1]}, "a3": {"b1ex_top1": "8211910000"}}
    items = items6([("821191", "a"), ("821191", "b")])
    preds = [pred(0, ["8211910000"], details=det), pred(1, ["8211910000"], details=det)]
    m = MH.compute(MH.score(items, preds, VALID8, VALID10, 1.0), n_resamples=20, mode="stage1")
    assert set(m["orders"]) == {"a", "b", "length"} and m["agree_rate_b1ex_hs6"] == 1.0
    assert m["headline_order"] == "a" and m["confidence_mode"] == "stage1"


# --- harness -----------------------------------------------------------------------------


class HookArm(FixedArm):
    name = "hooked"
    calls: ClassVar[list] = []

    def after_scoring(self, frame, n, seed):
        self.calls.append(1)
        raise AssertionError("a cn8 hook must not run on an HS6 set")


def test_test_run_logs_one_line_and_scores_at_hs6(tmp_path):
    s = load_set(tmp_path)
    log = tmp_path / "eval" / "test_runs.jsonl"
    out = R.run_eval(HookArm(), s, VALID8, VALID10, mlflow_dir=tmp_path / "mlflow", test_log=log,
                     git=CLEAN, n_resamples=20)  # fmt: skip
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 and lines[0]["dataset"] == "hscodecomp_c" and lines[0]["arm"] == "hooked"
    assert HookArm.calls == []
    m = out["metrics"]
    assert m["eval_depth"] == 6 and "8" not in {str(k) for k, v in m["top1"].items() if v["n"]}
    assert m["hs6"]["top1"][6] == 0.75 and m["hs6"]["confidence_mode"] == "native"
    assert "hs6_confidence_hs6" in out["frame"].columns
    assert any(line.startswith("  HS6: top-1 0.750") for line in R.arm_summary_lines(out))


# --- guards ------------------------------------------------------------------------------

PATH_RE = __import__("re").compile(r"[\"']hscodecomp|hscodecomp[\"'/]")


def hscodecomp_violations(text: str) -> list[str]:
    found = []
    if "hscodecomp" in text.lower():
        found.append("mentions hscodecomp")
    if WALK_RE.search(text) and "check_retrieval_inputs" not in text:
        found.append("walks folders without check_retrieval_inputs")
    return found


def test_hscodecomp_never_reaches_retrieval():
    """A test set only: no module outside ALLOWED_READERS may name it or walk folders unguarded."""
    assert H.NEVER_INDEX_FOR_RETRIEVAL is True
    assert H.RAW_DIR.as_posix() in FORBIDDEN_DIRS and set(H.RETRIEVAL_FORBIDDEN) <= set(
        FORBIDDEN_DIRS
    )
    assert hscodecomp_violations('p = root / "hscodecomp" / "download"')
    assert hscodecomp_violations("from hs_rag_classifier.data import hscodecomp")
    assert hscodecomp_violations("p = root / 'tariff'") == []
    offenders = {}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if any(rel == a or rel.startswith(a) for a in H.ALLOWED_READERS):
            continue
        found = hscodecomp_violations(path.read_text(encoding="utf-8"))
        if found:
            offenders[rel] = found
    assert not offenders, f"HSCodeComp must never be indexed for retrieval: {offenders}"


def test_check_retrieval_inputs_refuses_hscodecomp(tmp_path):
    ok = [tmp_path / "processed" / "tariff" / "v4.0.1633" / "codes.parquet",
          tmp_path / "raw" / "hscodecomp_notes.txt"]  # fmt: skip
    assert check_retrieval_inputs(ok, tmp_path) == ok
    for bad in (tmp_path / "hscodecomp" / "download" / "data" / "test_data.jsonl",
                tmp_path / "hscodecomp",
                tmp_path / "processed" / ".." / "hscodecomp" / "card.md"):  # fmt: skip
        with pytest.raises(ForbiddenInputError):
            check_retrieval_inputs([*ok, bad], tmp_path)


def test_drop_list_never_comes_from_labels_or_predictions():
    cfg = H.load_config()
    text = json.dumps(cfg["drop_keys"]) + json.dumps(cfg["kept_ambiguous"])
    assert "hs_code" not in text and not any(ch.isdigit() and len(ch) > 5 for ch in text.split())
    assert copy.deepcopy(cfg) == cfg and Path(H.INPUTS_CONFIG).name == "hscodecomp_inputs.json"
