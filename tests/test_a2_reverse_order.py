"""a2_tune reverse-order and its optional flags, on the fakes in test_a2_pipeline: with no flags
the output path and fields are unchanged; --out is respected; --save-probs keys each option's
probability by code, so both orders map back to the same codes; --items-from takes the ids of an
earlier output. A fake server prefers HS6 codes, not labels or positions. No live server."""

import json
import math
import random
import re
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from test_a2_pipeline import STAGE2, a2

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.pipeline import a2_tune as T

OPTIONS = ["821192", "821191", "010121"]  # the saved (fused) stage-1 order
PREF = {"821192": 0.6, "821191": 0.3, "010121": 0.05}  # by code, whatever its label
IDS = ["5", "12", "30", "41"]


def by_code(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    user = body["messages"][1]["content"]
    m = re.search(r"Subheading (\d{6}):", user)
    if m:
        toks = STAGE2[m.group(1)]
    else:
        shown = re.findall(r"^([A-Z])\. (\d{6}):", user, re.MULTILINE)
        toks = sorted(({"token": lab, "logprob": math.log(PREF[h])} for lab, h in shown),
                      key=lambda t: -t["logprob"])  # fmt: skip
    step = {"token": toks[0]["token"], "logprob": toks[0]["logprob"], "top_logprobs": toks}
    return httpx.Response(200, json={"choices": [{"logprobs": {"content": [step]}}],
                                     "usage": {"prompt_tokens": 300, "completion_tokens": 1}})  # fmt: skip


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    pipe, _, _ = a2()
    pipe.http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(by_code))
    details = json.dumps({"evidence": [], "stage1": {"options": OPTIONS, "choice_code": "821192"}})
    frame = pd.DataFrame({"item_id": IDS, "details": [details] * len(IDS)})
    monkeypatch.setattr("hs_rag_classifier.eval.runner.load_run_frame", lambda run_id: frame)
    monkeypatch.setattr(A2, "load_config", lambda: {**pipe.cfg, "client": A1.DEFAULTS["client"]})
    monkeypatch.setattr(A2, "build_pipeline", lambda config, api=True: pipe)
    monkeypatch.setattr(T, "_items", lambda: [SimpleNamespace(item_id=i, text="a kitchen knife")
                                              for i in IDS])  # fmt: skip
    monkeypatch.setattr(T, "OUT", tmp_path / "eval")
    (tmp_path / "eval").mkdir()
    return tmp_path


def test_default_path_and_fields_unchanged(fake):
    out = T.reverse_order("run", n=2)
    saved = json.loads((fake / "eval" / "a2_reverse_order_val.json").read_text())
    assert saved == out
    assert set(out) == {"run_id", "n", "seed", "top1_hs6_changed", "rows"}
    assert [r["item_id"] for r in out["rows"]] == sorted(
        random.Random(T.REVERSE_SEED).sample(sorted(IDS), 2)
    )
    assert set(out["rows"][0]) == {"item_id", "original", "reversed", "original_pos",
                                   "reversed_pos_in_original"}  # fmt: skip
    # the fake prefers 821192 by code, so reversing the order changes nothing
    assert out["top1_hs6_changed"] == 0 and out["rows"][0]["reversed"] == "821192"


def test_out_is_respected_and_the_default_file_untouched(fake):
    target = fake / "elsewhere" / "rev.json"
    target.parent.mkdir()
    T.reverse_order("run", n=2, out_path=target)
    assert target.exists() and not (fake / "eval" / "a2_reverse_order_val.json").exists()
    assert json.loads(target.read_text())["order"] == "reversed"


def test_save_probs_keys_by_code_and_both_orders_map_back(fake):
    fwd = T.reverse_order("run", n=1, out_path=fake / "f.json", save_probs=True, order="forward")
    rev = T.reverse_order("run", n=1, out_path=fake / "r.json", save_probs=True)
    f, r = fwd["rows"][0], rev["rows"][0]
    assert f["stage1"]["positions"] == {"821192": 0, "821191": 1, "010121": 2}
    assert r["stage1"]["positions"] == {"821192": 2, "821191": 1, "010121": 0}
    total = sum(PREF.values())
    for code, p in PREF.items():
        assert f["stage1"]["probs"][code] == pytest.approx(p / total)
        assert r["stage1"]["probs"][code] == pytest.approx(p / total)
    assert f["stage1"]["choice"] == r["stage1"]["choice"] == "821192"
    assert f["stage1"]["options_match_retrieval"] and r["stage1"]["options_match_retrieval"]
    assert f["stage2"][0]["hs6"] == "821192"
    assert f["stage2"][0]["probs"] == pytest.approx({"82119210": 0.25, "82119290": 0.75})
    # same codes and choice; the joint can differ in its last bit (sums run in another order)
    assert {k: v for k, v in f["final"].items() if k != "joint"} == {
        k: v for k, v in r["final"].items() if k != "joint"
    }
    assert f["final"]["joint"] == pytest.approx(r["final"]["joint"])
    assert f["final"]["cn8"] == "82119290"
    assert f["final"]["joint"] == pytest.approx(0.6 / total * 0.75)
    assert fwd["order"] == "forward" and fwd["save_probs"] is True


def test_items_from_takes_the_earlier_ids_in_order(fake):
    earlier = fake / "earlier.json"
    earlier.write_text(json.dumps({"rows": [{"item_id": "41"}, {"item_id": "5"}]}))
    out = T.reverse_order("run", out_path=fake / "x.json", items_from=earlier)
    assert [r["item_id"] for r in out["rows"]] == ["41", "5"] and out["n"] == 2
    earlier.write_text(json.dumps({"rows": [{"item_id": "999"}]}))
    with pytest.raises(ValueError, match="not in run"):
        T.reverse_order("run", out_path=fake / "y.json", items_from=earlier)


def test_order_must_be_forward_or_reversed():
    assert T.order_options(["a", "b"], "forward") == ["a", "b"]
    assert T.order_options(["a", "b"], "reversed") == ["b", "a"]
    with pytest.raises(ValueError):
        T.order_options(["a"], "sideways")


def test_cli_flags_reach_reverse_order(monkeypatch, tmp_path):
    seen = {}

    def spy(run, n, **kw):
        seen.update(run=run, n=n, **kw)
        return {"rows": []}

    monkeypatch.setattr(T, "reverse_order", spy)
    T.main(["reverse-order", "--run", "R", "--out", str(tmp_path / "o.json"), "--save-probs",
            "--order", "forward", "--items-from", str(tmp_path / "e.json")])  # fmt: skip
    assert seen == {"run": "R", "n": 100, "out_path": tmp_path / "o.json", "save_probs": True,
                    "order": "forward", "items_from": tmp_path / "e.json"}  # fmt: skip
    T.main(["reverse-order", "--run", "R"])
    assert seen["out_path"] is None and seen["order"] == "reversed" and not seen["save_probs"]
