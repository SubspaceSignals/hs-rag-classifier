"""B2 (HS6 encoder) and depth-limited scoring. No model downloads: a tiny random ModernBERT and a
fake tokeniser stand in for the real ones; torch tests skip without the "ml" group."""

import importlib.util
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hs_rag_classifier.arms import b2_encoder as B2
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.arms import Candidate, Prediction
from hs_rag_classifier.eval.items import EvalItem, EvalSet
from hs_rag_classifier.guards import ForbiddenInputError

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch (uv sync --group ml)")
VALID8 = {"01012100", "01012900", "82119100", "82119200"}
VALID10 = {"0101210000", "0101290000", "8211910000", "8211920010"}


def item(item_id, label, depth=None, text="x"):
    return EvalItem(item_id=item_id, text=text, label_code=label,
                    label_depth=depth or len(label), cluster_id=item_id,
                    chapter=label[:2], dataset="d", split="val")  # fmt: skip


def pred(item_id, codes, conf):
    return Prediction(item_id=item_id, confidence=conf,
                      candidates=[Candidate(code=c, score=1.0 - n / 10) for n, c in
                                  enumerate(codes)])  # fmt: skip


@pytest.fixture()
def root(tmp_path):
    rulings = tmp_path / B2.TRAIN_REL.parent
    rulings.mkdir(parents=True)
    train = pd.DataFrame({
        "ref": ["1", "2", "3", "4"],
        "description": ["steel table knife", "knife with fixed blade", "live horse",
                        "pure bred horse"],
        "keywords": [["K"], ["K"], ["H"], ["H"]],
        "code10": ["8211910000", "8211920010", "0101210000", "0101290000"],
        "cn8": ["82119100", "82119200", "01012100", "01012900"],
        "label_depth": [10, 8, 10, 10],
    })  # fmt: skip
    for name in ("train", "val", "test"):
        train.to_parquet(rulings / f"{name}.parquet", index=False)
    pd.DataFrame({"ref": ["1"], "justification": ["x"]}).to_parquet(
        rulings / "justifications.parquet", index=False
    )
    return tmp_path


# --- Labels and the loader ------------------------------------------------------------------


def test_label_mapping_to_hs6_and_label_set():
    assert B2.hs6("8211910000") == "821191" and B2.hs6("82119100") == "821191"
    assert B2.label_set(["8211910000", "82119200", "82119100", "0101210000"]) == [
        "010121", "821191", "821192"]  # fmt: skip
    with pytest.raises(ValueError):
        B2.hs6("8211")


def test_label_set_coverage_reporting():
    labels = ["010121", "821191"]
    assert B2.outside_share(["8211910000", "0101290000", "82119200", "0101210000"], labels) == 0.5
    assert B2.outside_share(["8211910000"], labels) == 0.0
    assert math.isnan(B2.outside_share([], labels))
    t = B2.targets_for(["8211910000", "0101290000"], labels)
    assert list(t) == [1, -1]  # outside the label set: -1, never a hit
    logits = np.array([[0.0, 5.0], [5.0, 0.0]])
    assert list(B2.top_k_hits(logits, t, 1)) == [1.0, 0.0]
    assert list(B2.top_k_hits(logits, t, 2)) == [1.0, 0.0]


def test_training_loader_reads_train_only(root, monkeypatch):
    calls = []
    real = B2.check_retrieval_inputs
    monkeypatch.setattr(B2, "check_retrieval_inputs",
                        lambda p, r: calls.append([Path(x) for x in p]) or real(p, r))  # fmt: skip
    rows = B2.load_training_rows(root)
    assert calls == [[root / B2.TRAIN_REL]]
    assert list(rows.columns) == ["ref", "text", "hs6"]
    assert list(rows["hs6"]) == ["821191", "821192", "010121", "010129"]  # depth 8 rows too
    assert "keywords" not in B2.TRAIN_COLUMNS
    for name in ("val.parquet", "test.parquet", "justifications.parquet"):
        with pytest.raises(ForbiddenInputError):
            B2.load_training_rows(root, root / B2.TRAIN_REL.parent / name)
    for folder in ("raw/atoz", "processed/atoz/v1"):
        leaked = root / folder / "train.parquet"
        leaked.parent.mkdir(parents=True)
        leaked.write_bytes((root / B2.TRAIN_REL).read_bytes())
        with pytest.raises(ForbiddenInputError):
            B2.load_training_rows(root, leaked)


# --- Temperature ----------------------------------------------------------------------------


def test_temperature_scaling():
    p = B2.softmax(np.array([[2.0, 1.0, 0.0]]), 1.0)
    assert p.sum() == pytest.approx(1.0)
    assert p[0, 0] == pytest.approx(math.exp(2) / (math.exp(2) + math.exp(1) + 1))
    assert B2.softmax(np.array([[2.0, 1.0, 0.0]]), 1e6)[0, 0] == pytest.approx(1 / 3, abs=1e-5)
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 5, 400)
    logits = rng.normal(0, 1, (400, 5))
    logits[np.arange(400), targets] += 1.0
    hot = logits * 8  # overconfident: the fitted T should cool it
    t = B2.fit_temperature(hot, targets)
    assert t > 2
    assert B2.nll(t, hot, targets) <= B2.nll(1.0, hot, targets)
    with_outside = np.r_[targets, [-1, -1]]
    assert B2.fit_temperature(np.r_[hot, hot[:2]], with_outside) == pytest.approx(t, rel=1e-3)


# --- Depth-limited scoring ------------------------------------------------------------------


def test_depth_limited_scoring_reports_na_above_max_depth():
    items = [item("a", "8211910000"), item("b", "82119100", 8), item("c", "0101290000"),
             item("d", "0101210000")]  # fmt: skip
    preds = [pred("a", ["821191"], 0.9), pred("b", ["821192", "821191"], 0.8),
             pred("c", ["999999", "010129"], 0.7), pred("d", ["010121"], 0.2)]  # fmt: skip
    frame = M.score(items, preds, VALID8, VALID10, max_depth=6)
    assert list(frame["correct_6"]) == [1.0, 0.0, 0.0, 1.0]  # 999999 is not a subheading
    assert frame["correct_8"].isna().all() and frame["correct_10"].isna().all()
    assert list(frame["top1_valid"]) == [True, True, False, True]
    assert list(frame["recall_3"]) == [1.0, 1.0, 1.0, 1.0]  # recall at 6 digits
    m = M.compute_metrics(frame, 1.0, n_resamples=20, max_depth=6)
    assert m["eval_depth"] == 6 and m["max_depth"] == 6
    assert m["top1"][6] == {"accuracy": 0.5, "n": 4}
    assert math.isnan(m["top1"][8]["accuracy"]) and m["top1"][8]["n"] == 0
    assert math.isnan(m["top1"][10]["accuracy"]) and m["top1"][10]["n"] == 0
    assert m["ece"] == pytest.approx(M.ece(frame["confidence"].to_numpy(), frame["correct_6"]))
    assert m["coverage"][0]["accuracy"] == 0.5
    assert "top1_8" not in m["bootstrap"]["intervals"]
    assert {"top1_6", "headline", "ece", "recall_3"} <= set(m["bootstrap"]["intervals"])
    flat = R.flat_metrics(m)
    assert "top1_8" not in flat and "recall_3_6digit" in flat and "recall_3_8digit" not in flat
    table = M.per_chapter(frame, max_depth=6)
    assert list(table.columns) == ["chapter", "n", "top1_6", "top1_8", "recall_10"]


def test_full_depth_scoring_unchanged_by_default():
    items = [item("a", "8211910000"), item("b", "82119100", 8)]
    preds = [pred("a", ["8211910000"], 0.9), pred("b", ["821191"], 0.8)]
    frame = M.score(items, preds, VALID8, VALID10)
    assert list(frame["correct_10"].fillna(-1)) == [1.0, -1]
    assert list(frame["top1_valid"]) == [True, False]  # 6-digit codes are invalid at depth 10
    m = M.compute_metrics(frame, 1.0, n_resamples=20)
    assert m["eval_depth"] == 8 and "top1_8" in m["bootstrap"]["intervals"]
    assert list(M.per_chapter(frame).columns) == ["chapter", "n", "top1_8", "recall_10"]


class DepthSixArm:
    name = "depth6"
    max_depth = 6

    def __init__(self):
        self.config = {}

    def predict(self, items):
        return [pred(i.item_id, [i.label_code[:6]], 0.5) for i in items]


def test_runner_tags_eval_depth_and_reports_na(tmp_path):
    from mlflow.tracking import MlflowClient

    evalset = EvalSet(dataset="rulings", split="val", items=[item("a", "8211910000")],
                      split_file="x", split_sha256="y")  # fmt: skip
    res = R.run_eval(DepthSixArm(), evalset, VALID8, VALID10, mlflow_dir=tmp_path / "mlflow",
                     test_log=tmp_path / "t.jsonl", git={"commit": "c", "dirty": False},
                     n_resamples=10)  # fmt: skip
    run = MlflowClient(R.tracking_uri(tmp_path / "mlflow")).get_run(res["run_id"])
    assert run.data.tags["eval_depth"] == "6"
    assert run.data.metrics["top1_6"] == 1.0 and "top1_8" not in run.data.metrics
    lines = R.summary_lines(res, DepthSixArm(), evalset)
    assert "8: n/a (n 0)" in lines[1] and "top-1 at 6: 1.000" in lines[2]


# --- Tiny model: train, resume, predict ------------------------------------------------------


class FakeTokenizer:
    """Hashes words to ids; pads to the longest text. Enough for a tiny ModernBERT."""

    vocab = 64

    def __call__(self, texts, truncation=True, padding=True, max_length=512, return_tensors="pt"):
        import torch

        ids = [[1] + [2 + zlib.crc32(w.encode()) % (self.vocab - 2) for w in t.split()][
            : max_length - 1] for t in texts]  # fmt: skip
        n = max(len(x) for x in ids)
        return {
            "input_ids": torch.tensor([x + [0] * (n - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids]),
        }


def tiny_model(config, labels):
    from transformers import ModernBertConfig, ModernBertForSequenceClassification

    cfg = ModernBertConfig(vocab_size=FakeTokenizer.vocab, hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=64, pad_token_id=0,
                           num_labels=len(labels), global_attn_every_n_layers=1,
                           max_position_embeddings=64, attn_implementation="eager")  # fmt: skip
    return ModernBertForSequenceClassification(cfg)


@needs_torch
def test_tiny_train_resume_and_predict(root, tmp_path):
    config = B2.default_config(batch_size=2, max_epochs=2, patience=5, lr=1e-3, max_length=16)
    out = tmp_path / "ckpt"
    val_texts, val_codes = ["steel knife", "live horse"], ["8211910000", "0101210000"]
    kw = {"root": root, "out_dir": out, "tokenizer": FakeTokenizer(),
          "model_factory": tiny_model, "dev": "cpu"}  # fmt: skip
    first = B2.train({**config, "max_epochs": 1}, val_texts, val_codes, **kw)
    assert first["epoch"] == 1 and (out / "best").exists() and (out / "last" / "state.pt").exists()
    assert first["labels"] == ["010121", "010129", "821191", "821192"]
    resumed = B2.train(config, val_texts, val_codes, **kw)
    assert resumed["epoch"] == 2 and [h["epoch"] for h in resumed["history"]] == [1, 2]
    assert sorted(p.name for p in out.iterdir()) == ["best", "labels.json", "last"]
    done = B2.train(config, val_texts, val_codes, **kw)  # already finished: no more epochs
    assert done["epoch"] == 2

    arm = B2.B2EncoderArm({**config, "temperature": 0.5}, model=tiny_model(config, first["labels"]),
                          tokenizer=FakeTokenizer(), labels=first["labels"], dev="cpu")  # fmt: skip
    (p,) = arm.predict([item("q", "8211910000", text="steel knife")])
    assert arm.max_depth == 6 and not p.abstain
    assert sorted(c.code for c in p.candidates) == first["labels"]  # fewer labels than the cap
    assert all(len(c.code) == 6 for c in p.candidates)
    assert p.confidence == pytest.approx(p.candidates[0].score)
    assert sum(c.score for c in p.candidates) == pytest.approx(1.0, abs=1e-5)


def test_train_key_ignores_temperature():
    a, b = B2.default_config(), B2.default_config(temperature=3.0)
    assert B2.train_key(a) == B2.train_key(b)
    assert B2.train_key(B2.default_config(lr=1e-4)) != B2.train_key(a)
