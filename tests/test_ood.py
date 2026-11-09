"""OOD flag on a tiny fake index: the query transform, masking, the strict < at the threshold,
the reason string, the self-exclusion rows and the percentile. The real-index test skips when
the model or index files are absent."""

import json

import numpy as np
import pytest

from hs_rag_classifier import ood as O

VEC = {"knife": [1.0, 0.0, 0.0], "horse": [0.0, 1.0, 0.0], "mix": [0.6, 0.8, 0.0]}


def fake_encode(calls):
    def encode(texts):
        calls.append(list(texts))
        out = []
        for t in texts:
            v = np.array(VEC.get(t.split()[0], [0.0, 0.0, 1.0]), np.float32)
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32)

    return encode


def scorer(calls=None):
    emb = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0], [1, 0, 0]], np.float32)
    idx = O.RulingsIndex(emb, ["r1", "r2", "r3", "r4"])
    return O.OODScorer(idx, fake_encode([] if calls is None else calls), max_input_chars=20)


def test_nearest_and_masking():
    calls = []
    s = scorer(calls)
    sim, ref = O.nearest_ruling_similarity("knife blade", scorer=s)
    assert sim == pytest.approx(1.0) and ref == "r1"  # tie with r4: the first row
    sim, ref = O.nearest_ruling_similarity("knife blade", exclude_rows=[0], scorer=s)
    assert sim == pytest.approx(1.0) and ref == "r4"
    sim, ref = O.nearest_ruling_similarity("knife blade", exclude_rows=[0, 3], scorer=s)
    assert sim == pytest.approx(0.6) and ref == "r3"
    n = s.nearest("horse", [1, 1])
    assert n.n_masked == 1 and n.ref == "r3" and n.similarity == pytest.approx(0.8)
    with pytest.raises(ValueError, match="every index row"):
        s.nearest("horse", [0, 1, 2, 3])
    assert all(len(c) == 1 for c in calls)  # one text per encode call


def test_query_text_is_the_pipeline_transform():
    calls = []
    s = scorer(calls)
    s.nearest("knife " + "x" * 40 + "\x07")
    assert calls[-1] == [O.query_text("knife " + "x" * 40 + "\x07", 20)]
    assert len(calls[-1][0]) <= 20  # truncated at max_input_chars, as the retriever does


def test_apply_rule_strict_and_reason(tmp_path):
    rule = {"threshold": 0.5, "flag": "similarity < threshold"}
    assert O.apply_rule(0.4999, rule) == (True, O.REASON)
    assert O.apply_rule(0.5, rule) == (False, None)  # equal is not flagged
    assert O.apply_rule(0.9, rule) == (False, None)
    assert O.REASON == "input is unlike the rulings the confidence was fitted on"
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rule))
    assert O.load_rule(p) == rule
    p.write_text(json.dumps({"threshold": 1, "flag": "similarity < threshold"}))
    with pytest.raises(ValueError):
        O.load_rule(p)


def test_exclusion_rows_by_near_dup_key():
    from hs_rag_classifier.finetune import build_set as F

    refs = ["t1", "t2", "t3"]
    descs = [
        "An inflatable buoy " + "a" * 300,
        "An inflatable buoy " + "a" * 300 + " longer",
        "Other",
    ]
    cn8 = ["89079000", "89079000", "89079000"]
    keys = {}
    for i, (d, c) in enumerate(zip(descs, cn8, strict=True)):
        keys.setdefault(F.near_dup_key(d, c), []).append(i)
    own = {r: [i] for i, r in enumerate(refs)}
    members = [("t3",), (), ()]  # t1's member ref is itself an index row
    got = O.exclusion_rows(
        "An inflatable buoy " + "a" * 300 + " val", "89079000", keys, own, refs, members
    )
    assert got == [0, 1, 2]
    assert (
        O.exclusion_rows("An inflatable buoy " + "a" * 300, "11111111", keys, own, refs, members)
        == []
    )


def test_percentile_is_linear_numpy():
    x = list(np.linspace(0.3, 0.9, 41))
    assert O.fit_threshold(x) == float(np.percentile(x, 5, method="linear"))


def _real_ready():
    from hs_rag_classifier.arms import b1ex_embed as BX
    from hs_rag_classifier.arms import embed as E

    try:
        import torch
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cfg = E.model_config("bge-m3")
    hit = try_to_load_from_cache(cfg["hf_id"], "config.json", revision=cfg["revision"])
    idx = O.DATA_ROOT / BX.INDEX_DIR / "bge-m3"
    return torch.cuda.is_available() and isinstance(hit, str) and idx.exists()


@pytest.mark.skipif(not _real_ready(), reason="bge-m3, CUDA or the rulings index absent")
def test_real_index_batch1_is_repeatable():
    s = O.OODScorer.from_pipeline()
    assert len(s.index.refs) == 3773 and s.embeddings.dtype == np.float32
    texts = ["Stainless steel table knives with fixed blades", "Live pure-bred breeding horses"]
    first = [s.nearest(t) for t in texts]
    again = [s.nearest(t) for t in texts]
    assert [(a.similarity, a.ref) for a in first] == [(b.similarity, b.ref) for b in again]
    assert all(0 < a.similarity <= 1.0001 for a in first)
