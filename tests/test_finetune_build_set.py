"""finetune.build_set: leave-one-out exclusion (rulings vote and evidence), the reversed order and
label mapping, the length and single-token assertions, and prompts identical to A2's requests
with an empty exclusion set. Small fixtures (A2's fake tariff and server, a tiny rulings store, a
character-level fake tokenizer); no model, no server, no data."""

import numpy as np
import pandas as pd
import pytest
from test_a2_pipeline import a2, make_tariff, run

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.finetune import build_set as F
from hs_rag_classifier.retrieval.hybrid import HybridRetriever, RulingsSource
from hs_rag_classifier.retrieval.store import InMemoryStore


class FakeTok:
    """Character-level stand-in for the Qwen tokenizer: one token per character."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt and enable_thinking is False
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


# --- a tiny real rulings source under a real HybridRetriever ---------------------------------

REFS = ["r1", "r2", "r3", "r4"]
CN8 = ["82119290", "82119290", "82119100", "01012100"]  # r1, r2: one near-duplicate cluster
VECS = np.array([[1.0, 0.0], [0.99, 0.14], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)


class FakeEncoder:
    def queries(self, texts):
        return np.array([[1.0, 0.05]] * len(texts), dtype=np.float32)  # nearest r1, then r2


class NoHits:
    def search(self, *a, **k):
        return []

    def search_tokens(self, *a, **k):
        return []


class FakeEvidence:
    def lines(self, refs, chars):
        return [f"Past ruling: {r} -> x" for r in refs]


def tiny_retriever():
    store = InMemoryStore()
    store.add(REFS, VECS)
    labels = pd.DataFrame({"ref": REFS, "cn8": CN8, "code10": [c + "00" for c in CN8],
                           "label_depth": [10] * 4})  # fmt: skip
    sources = {"keyword": NoHits(), "tariff_embed": NoHits(),
               "rulings": RulingsSource(store, labels, k=1, pool=4)}  # fmt: skip
    cfg = {"weights": {"keyword": 1.0, "tariff_embed": 1.0, "rulings": 1.0}, "k_rrf": 10}
    return HybridRetriever(cfg, sources=sources, encoder=FakeEncoder(), meta={})


def capture(retriever, m=3, labels=("A", "B", "C"), evidence=True):
    cfg = {**A1.DEFAULTS, "evidence": evidence, "evidence_k": 2, "m": m, "labels": list(labels),
           "a2": {}}  # fmt: skip
    return F.make_capture(cfg, retriever, make_tariff(), FakeEvidence() if evidence else None)


def test_exclusion_removes_the_cluster_from_vote_and_evidence():
    base = tiny_retriever()
    loo, store = F.loo_retriever(base)
    train = pd.DataFrame({"ref": ["r1", "r2", "r3"], "cluster_id": ["c1", "c1", "c3"],
                          "member_refs": [["r1", "m1"], ["r2"], ["r3"]]})  # fmt: skip
    excl = F.exclusion_sets(train)
    assert excl["r1"] == excl["r2"] == {"r1", "r2", "m1"} and excl["r3"] == {"r3"}

    as_is = capture(base).item_state("a kitchen knife")
    assert as_is["evidence"] == ["Past ruling: r1 -> x", "Past ruling: r2 -> x"]
    assert as_is["fused"][0] == "82119290"  # r1's code tops the vote

    store.exclude, store.seen = excl["r1"], []
    s = capture(loo).item_state("a kitchen knife")
    assert s["evidence"] == ["Past ruling: r3 -> x", "Past ruling: r4 -> x"]
    assert s["fused"][0] == "82119100"  # r3's code tops the vote; r1 and r2 are gone
    assert len(store.seen) == 2  # one search for the vote, one for the evidence
    assert not {i for call in store.seen for i in call} & excl["r1"]
    # the original retriever and its store are untouched
    assert base.sources["rulings"].store is not store
    assert capture(base).item_state("a kitchen knife")["evidence"][0] == "Past ruling: r1 -> x"


def test_empty_exclusion_is_a_pass_through():
    inner = tiny_retriever().sources["rulings"].store
    wrapped = F.ExcludingStore(inner)
    q = np.array([[1.0, 0.05]], dtype=np.float32)
    for k in (1, 3, 4):
        a, b = wrapped.search(q, k), inner.search(q, k)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_reversed_order_and_label_mapping():
    pipe = capture(tiny_retriever(), evidence=False)
    tok = FakeTok()
    s = pipe.item_state("a kitchen knife")
    hs6s = pipe.stage1_options(s["fused"])
    assert hs6s == ["821192", "821191", "010121"]
    rows, facts = F.item_examples(pipe, tok, F.label_tokens(tok, pipe.labels), s, "r9",
                                  "821191", "82119100", {"ref": "r9"})  # fmt: skip
    fwd, rev = (next(r for r in rows if r["stage"] == 1 and r["order"] == o)
                for o in ("forward", "reversed"))  # fmt: skip
    assert fwd["options"] == hs6s and rev["options"] == hs6s[::-1]
    assert fwd["target_label"] == "B" and rev["target_label"] == "B"  # middle of three
    assert "A. 821192:" in fwd["prompt"] and "A. 010121:" in rev["prompt"]
    assert fwd["target_token_id"] == ord("B") and fwd["n_tokens"] == len(fwd["prompt"])
    assert not [r for r in rows if r["stage"] == 2]  # 821191 has one cn8: no stage-2 example
    rows, _ = F.item_examples(pipe, tok, F.label_tokens(tok, pipe.labels), s, "r9", "821192",
                              "82119290", {"ref": "r9"})  # fmt: skip
    by = {(r["stage"], r["order"]): r for r in rows}
    assert by[(1, "forward")]["target_label"] == "A" and by[(1, "reversed")]["target_label"] == "C"
    assert by[(2, "forward")]["options"] == ["82119210", "82119290"]
    assert by[(2, "forward")]["target_label"] == "B" and by[(2, "reversed")]["target_label"] == "A"
    assert facts["gold_hs6_in_options"] and facts["gold_hs6_children"] == 1


def test_gold_hs6_missing_gives_no_stage1_example():
    pipe = capture(tiny_retriever(), m=1, evidence=False)  # only the top HS6 is an option
    tok = FakeTok()
    s = pipe.item_state("a kitchen knife")
    rows, facts = F.item_examples(pipe, tok, F.label_tokens(tok, pipe.labels), s, "r9",
                                  "010121", "01012100", {"ref": "r9"})  # fmt: skip
    assert rows == [] and facts["gold_hs6_in_options"] is False


def test_length_and_single_token_assertions():
    pipe = capture(tiny_retriever(), evidence=False)
    tok = FakeTok()
    s = pipe.item_state("a kitchen knife")
    with pytest.raises(F.BuildError, match="never truncated"):
        F.item_examples(pipe, tok, F.label_tokens(tok, pipe.labels), s, "r9", "821192",
                        "82119290", {"ref": "r9"}, max_tokens=50)  # fmt: skip
    with pytest.raises(F.BuildError, match="not one"):
        F.label_tokens(tok, ["A", "BC"])
    with pytest.raises(F.BuildError, match="share a token"):

        class Same(FakeTok):
            def encode(self, text, add_special_tokens=False):
                return [1]

        F.label_tokens(Same(), ["A", "B"])
    cap = pipe.stage1(s, ["821192", "821191", "010121"])
    with pytest.raises(F.BuildError, match="single-token labels"):
        F.example(tok, {"A": 65}, cap, ["821192", "821191", "010121"], "821191", {})


def test_prompts_equal_a2_requests_with_empty_exclusion():
    pipe, server, _ = a2()
    run(pipe)  # A2's own graph, fake server: stage 1, then stage 2 for 821192
    sent = [r["messages"] for r in server.requests]
    cap_pipe = F.make_capture(pipe.cfg, pipe.retriever, pipe.tariff, None)
    s = cap_pipe.item_state("a kitchen knife")
    hs6s = cap_pipe.stage1_options(s["fused"])
    c1 = cap_pipe.stage1(s, hs6s)
    cn8s, _ = cap_pipe.stage2_options("821192", s["fused"])
    c2 = cap_pipe.stage2(s, "821192", cn8s)
    for cap, msgs in ((c1, sent[0]), (c2, sent[1])):
        assert [cap["system"], cap["user"]] == [m["content"] for m in msgs]
        assert [m["role"] for m in msgs] == ["system", "user"]
    tok = FakeTok()
    text = F.chat_text(tok, c1["system"], c1["user"])
    assert F.token_ids(tok, text) == tok.apply_chat_template(
        sent[0], tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
