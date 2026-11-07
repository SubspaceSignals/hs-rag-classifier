"""finetune.train and finetune.score_val: the answer-token loss, exact resume, input refusals, the
gate-row selection and the paired bootstrap. A tiny randomly initialised Qwen3.5 on the CPU and
small fixtures; no real model, no server, no data."""

import json

import numpy as np
import pandas as pd
import pytest

from hs_rag_classifier.finetune import score_val as S
from hs_rag_classifier.finetune import train as T

torch = pytest.importorskip("torch")
pytest.importorskip("peft")


# --- a tiny Qwen3.5 with the real target modules ------------------------------------------------

VOCAB = 128
LABEL_IDS = list(range(60, 112))  # 52 distinct "label tokens"


def tiny_model(
    seed: int = 0, lora: bool = True, cfg: T.TrainConfig | None = None, lora_seed: int = 1
):
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    c = Qwen3_5TextConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                          linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
                          linear_value_head_dim=8, linear_conv_kernel_dim=4,
                          layer_types=["linear_attention", "full_attention"],
                          max_position_embeddings=256)  # fmt: skip
    torch.manual_seed(seed)
    m = Qwen3_5ForCausalLM(c)
    if not lora:
        return m
    torch.manual_seed(lora_seed)
    return T.attach_lora(m, cfg or T.TrainConfig())


def examples(n: int = 10, seed: int = 3):
    g = np.random.default_rng(seed)
    return [(f"e{i}", g.integers(0, 60, size=int(g.integers(12, 30))).tolist(), int(g.choice(LABEL_IDS)))
            for i in range(n)]  # fmt: skip


# --- answer-token loss ---------------------------------------------------------------------------


def test_answer_logits_only_at_the_answer_position():
    m = tiny_model(lora=False).eval()
    x = torch.randint(0, 60, (1, 20))
    got = T.answer_logits(m, x)
    assert got.shape == (1, VOCAB) and got.dtype == torch.float32
    full = m(input_ids=x, use_cache=False).logits
    assert full.shape[1] == 20
    torch.testing.assert_close(got, full[:, -1, :].float())


def test_answer_logits_asks_for_one_position():
    seen = {}

    class Rec(torch.nn.Module):
        def forward(self, input_ids, logits_to_keep, use_cache):
            seen["keep"] = logits_to_keep
            return type("O", (), {"logits": torch.zeros(1, logits_to_keep, 5)})()

    T.answer_logits(Rec(), torch.zeros(1, 7, dtype=torch.long))
    assert seen["keep"] == 1

    class Wrong(torch.nn.Module):
        def forward(self, input_ids, logits_to_keep, use_cache):
            return type("O", (), {"logits": torch.zeros(1, 7, 5)})()

    with pytest.raises(RuntimeError, match="one position"):
        T.answer_logits(Wrong(), torch.zeros(1, 7, dtype=torch.long))


def test_answer_loss_is_the_answer_token_cross_entropy_and_ignores_earlier_targets():
    logits = torch.tensor([[2.0, 0.5, -1.0, 0.0]])
    loss = T.answer_loss(logits, torch.tensor([1]))
    assert float(loss) == pytest.approx(float(-torch.log_softmax(logits, -1)[0, 1]))
    # the prompt tokens never enter the loss: same answer logits, different prompt, same loss
    m = tiny_model(lora=False).eval()
    a = torch.randint(0, 60, (1, 15))
    la = T.answer_loss(T.answer_logits(m, a), torch.tensor([70]))
    la.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# --- exact resume ---------------------------------------------------------------------------------


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def test_exact_resume_matches_a_straight_run(tmp_path):
    cfg = T.TrainConfig(grad_accum=2)
    ex = examples(10)  # 5 optimiser steps
    straight = tiny_model(cfg=cfg)
    torch.manual_seed(7)
    r1 = T.train_loop(
        straight, ex, cfg, tmp_path / "a", LABEL_IDS, checkpoint_every=2, device="cpu"
    )
    assert r1["step"] == 5 and (tmp_path / "a" / "final").is_dir()

    first = tiny_model(cfg=cfg)
    torch.manual_seed(7)
    stop = T.train_loop(first, ex, cfg, tmp_path / "b", LABEL_IDS, checkpoint_every=2,
                        stop_after_step=3, device="cpu")  # fmt: skip
    assert stop["stopped_at"] == 3
    assert [p.name for p in T.step_checkpoints(tmp_path / "b")] == ["step_00002", "step_00003"]
    # same base weights (in training they come from the pinned snapshot), different LoRA init:
    # the adapter, optimiser, scheduler, RNG and data position must all come from the checkpoint
    resumed = tiny_model(cfg=cfg, lora_seed=99)
    torch.manual_seed(12345)
    r2 = T.train_loop(resumed, ex, cfg, tmp_path / "b", LABEL_IDS, checkpoint_every=2, resume=True,
                      device="cpu")  # fmt: skip
    assert r2["step"] == 5 and r2["n_seen"] == 10 and r2["n_a"] == r1["n_a"]
    a, b = _state(straight), _state(resumed)
    assert a.keys() == b.keys() and a
    for k in a:
        assert torch.equal(a[k], b[k]), k
    oa = torch.load(tmp_path / "a" / "final" / "trainer_state.pt", weights_only=False)
    ob = torch.load(tmp_path / "b" / "final" / "trainer_state.pt", weights_only=False)
    assert oa["scheduler"] == ob["scheduler"]
    for sa, sb in zip(
        oa["optimizer"]["state"].values(), ob["optimizer"]["state"].values(), strict=True
    ):
        for k in sa:
            assert torch.equal(sa[k], sb[k]), k
    # the last KEEP_LAST step checkpoints plus final/
    assert [p.name for p in T.step_checkpoints(tmp_path / "b")] == ["step_00003", "step_00004"]


def test_resume_refuses_a_different_order(tmp_path):
    cfg = T.TrainConfig(grad_accum=2)
    ex = examples(6)
    T.train_loop(tiny_model(cfg=cfg), ex, cfg, tmp_path, LABEL_IDS, checkpoint_every=1, stop_after_step=1,
                 device="cpu")  # fmt: skip
    with pytest.raises(RuntimeError, match="order_sha256"):
        T.train_loop(tiny_model(cfg=cfg), list(reversed(ex)), cfg, tmp_path, LABEL_IDS, checkpoint_every=1,
                     resume=True, device="cpu")  # fmt: skip


def test_schedule_steps_and_warmup():
    assert T.n_optimiser_steps(500, 16) == 32 and T.n_optimiser_steps(11916, 16) == 745
    opt, _sched, warmup = T.make_optimiser(tiny_model(), T.TrainConfig(), 32)
    assert warmup == 1 and opt.defaults["weight_decay"] == 0.0 and opt.defaults["lr"] == 1e-4


def test_shuffled_order_is_seeded_and_recorded():
    ids = [f"x{i}" for i in range(20)]
    o1, h1 = T.shuffled_order(ids, 5)
    o2, h2 = T.shuffled_order(list(reversed(ids)), 5)
    assert o1 == o2 and h1 == h2 and sorted(o1) == sorted(ids) and o1 != sorted(ids)
    assert T.shuffled_order(ids, 6)[1] != h1


# --- input refusals ---------------------------------------------------------------------------------


def _sha(p):
    return T.sha256_file(p)


@pytest.fixture
def setdir(tmp_path):
    d, sp = tmp_path / "a4", tmp_path / "splits" / "rulings_v1"
    d.mkdir()
    sp.mkdir(parents=True)
    (sp / "train_refs.txt").write_text("1\n2\n3\n")
    (sp / "val_refs.txt").write_text("4\n")
    (sp / "test_refs.txt").write_text("5\n")
    ex = pd.DataFrame({"example_id": ["1:1:forward", "2:1:forward", "3:1:reversed"], "ref": ["1", "2", "3"],
                       "stage": [1, 1, 1], "order": ["forward", "forward", "reversed"],
                       "prompt": ["ab", "abc", "abcd"], "n_tokens": [2, 3, 4], "target_label": ["A", "B", "A"],
                       "target_token_id": [32, 33, 32]})  # fmt: skip
    ex.to_parquet(d / "train_examples.parquet")
    pd.DataFrame({"example_id": ["1:1:forward", "3:1:reversed"]}).to_parquet(
        d / "pilot_manifest.parquet"
    )
    pd.DataFrame({"example_id": ["4:1:forward"]}).to_parquet(d / "val_subset.parquet")
    write_manifest(d)
    return d, tmp_path / "splits"


def write_manifest(d):
    names = ["train_examples.parquet", "pilot_manifest.parquet", "val_subset.parquet"]
    (d / "manifest.json").write_text(json.dumps({
        "sha256": {n: _sha(d / n) for n in names},
        "builder_commit": {"commit": "abc"},
        "base_model": {"repo": T.BASE_REPO, "revision": T.BASE_REVISION}}))  # fmt: skip


def test_load_examples_accepts_the_train_set_and_a_manifest(setdir):
    d, sp = setdir
    df, facts = T.load_examples(d, splits_dir=sp)
    assert len(df) == 3 and facts["builder_commit"] == "abc"
    df, facts = T.load_examples(d, d / "pilot_manifest.parquet", splits_dir=sp)
    assert list(df["example_id"]) == ["1:1:forward", "3:1:reversed"]
    assert set(facts["sha256"]) == {"train_examples.parquet", "pilot_manifest.parquet"}


def test_refuses_sha_mismatch(setdir):
    d, sp = setdir
    pd.DataFrame({"example_id": ["1:1:forward"]}).to_parquet(d / "pilot_manifest.parquet")
    with pytest.raises(T.TrainInputError, match="SHA-256"):
        T.load_examples(d, d / "pilot_manifest.parquet", splits_dir=sp)
    df = pd.read_parquet(d / "train_examples.parquet")
    df.loc[0, "prompt"] = "zz"
    df.to_parquet(d / "train_examples.parquet")
    with pytest.raises(T.TrainInputError, match="SHA-256"):
        T.load_examples(d, splits_dir=sp)


def test_refuses_non_train_inputs(setdir, tmp_path):
    d, sp = setdir
    with pytest.raises(T.TrainInputError, match="not an example-id manifest"):
        T.load_examples(d, d / "val_subset.parquet", splits_dir=sp)
    other = tmp_path / "elsewhere"
    other.mkdir()
    pd.read_parquet(d / "pilot_manifest.parquet").to_parquet(other / "pilot_manifest.parquet")
    with pytest.raises(T.TrainInputError, match="not a manifest listed"):
        T.load_examples(d, other / "pilot_manifest.parquet", splits_dir=sp)
    for bad, msg in (("4", "val or test"), ("5", "val or test"), ("9", "not in the train split")):
        df = pd.read_parquet(d / "train_examples.parquet")
        df.loc[0, "ref"] = bad
        df.to_parquet(d / "train_examples.parquet")
        write_manifest(d)
        with pytest.raises(T.TrainInputError, match=msg):
            T.load_examples(d, splits_dir=sp)


def test_tokenise_refuses_length_mismatch_and_overlong():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    df = pd.DataFrame({"example_id": ["a"], "prompt": ["abcd"], "n_tokens": [4]})
    assert T.tokenise(df, Tok(), 4) == {"a": [97, 98, 99, 100]}
    with pytest.raises(T.TrainInputError, match="no truncation"):
        T.tokenise(df, Tok(), 3)
    with pytest.raises(T.TrainInputError, match="the set says"):
        T.tokenise(df.assign(n_tokens=[5]), Tok(), 10)


# --- gate rows, NLL and bootstrap -------------------------------------------------------------------


def test_gate_rows_are_stage1_gold_in_options():
    v = pd.DataFrame({"example_id": list("abcdef"), "stage": [1, 1, 2, 1, 2, 1],
                      "gold_in_options": [True, False, True, True, True, True],
                      "order": ["forward", "forward", "forward", "forward", "forward", "reversed"]})  # fmt: skip
    assert list(S.gate_rows(v)["example_id"]) == ["a", "d"]


def test_row_nll_renormalises_over_the_option_labels():
    lp = [np.log(0.2), np.log(0.2), np.log(0.1)]  # mass outside the options
    assert S.row_nll(lp, ["A", "B", "C"], "B") == pytest.approx(-np.log(0.4))
    assert S.row_nll(lp, ["A", "B", "C"], None) is None


def test_paired_bootstrap_and_gate():
    base = np.array([1.0, 2.0, 3.0, 4.0])
    r = S.paired_bootstrap(base - 0.5, base)
    assert r["mean_diff"] == pytest.approx(-0.5) and r["ci95"] == pytest.approx([-0.5, -0.5])
    g = np.random.default_rng(0)
    a, b = g.normal(1.0, 1.0, 50), g.normal(1.0, 1.0, 50)
    r1, r2 = S.paired_bootstrap(a, b, n=200, seed=1), S.paired_bootstrap(a, b, n=200, seed=1)
    assert r1 == r2 and r1["ci95"][0] < r1["mean_diff"] < r1["ci95"][1] and r1["resamples"] == 200
    assert S.gate(base, base - 0.5)["verdict"] == "PASS"
    assert S.gate(base, base + 0.1)["verdict"] == "FAIL"
    # lower mean but an interval that crosses zero
    noisy = base + np.array([-3.0, 1.0, 1.0, 0.5])
    assert noisy.mean() < base.mean() and S.gate(base, noisy)["verdict"] == "FAIL"


def test_split_chat_recovers_system_and_user():
    t = "<|im_start|>system\nS\nx<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert S.split_chat(t) == ("S\nx", "U")
    with pytest.raises(ValueError):
        S.split_chat("plain")
