"""A4 QLoRA trainer: answer-token loss on A2's scoring prompts, plain PyTorch, exact resume.

    python -m hs_rag_classifier.finetune train --run-name NAME [--manifest PILOT.parquet]
                                               [--checkpoint-every N] [--resume]
    python -m hs_rag_classifier.finetune tf-score --adapter DIR --out OUT.json

Inputs: train_examples.parquet from the A4 set (finetune/build_set), optionally restricted to the
example ids of a manifest parquet (the pilot manifest). Both files' SHA-256 must match
manifest.json; only the set's train_examples.parquet is accepted, and every row's ref must be in
the frozen train split (never val or test). Prompts are tokenised with the pinned tokenizer and
must match the stored n_tokens and be at most MAX_TOKENS: nothing is truncated.

Model (fixed settings, 7 Nov 2026): Qwen3_5ForCausalLM text-only from the pinned snapshot, NF4
double-quant, bf16 compute, device_map {"": 0}, SDPA; PEFT LoRA r 16, alpha 32, dropout 0.05 on
TARGET_MODULES; gradient checkpointing (use_reentrant False). Loss: cross-entropy on the answer
token only, logits computed at that position only (logits_to_keep=1). Batch 1, gradient
accumulation 16 (the last group of a pass may be smaller; its loss is averaged over its own
size), AdamW lr 1e-4, weight decay 0, cosine decay with ceil(3%) warmup over the run's own
optimiser steps, max grad norm 1.0. No allocator setting.

Order: the selected examples shuffled once with SEED (numpy default_rng permutation of the
example ids sorted); order_sha256 is the SHA-256 of the ids in that order, one per line.

Checkpoints every --checkpoint-every optimiser steps and at the end, under
runs/<run_name>/checkpoints/step_NNNNN (adapter, optimiser, scheduler, RNG states, data position);
the last KEEP_LAST step checkpoints plus final/ are kept. --resume continues from the newest one
with the same config and order (it refuses a different one) and the same MLflow run.

MLflow experiment finetune-a4 (never a test run): params = base repo and revision, every
hyperparameter, input SHA-256s, builder and trainer commits, seed and order hash; per optimiser
step: loss, lr, grad_norm (before clipping), tokens_per_s, peak_gpu_gib, share_a_running (the
running share of examples so far whose argmax over the label tokens is "A").
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.eval.items import DATA_ROOT, SPLITS_DIR
from hs_rag_classifier.finetune.build_set import (
    BASE_REPO,
    BASE_REVISION,
    MAX_TOKENS,
    OUT,
    TOKENIZER_DIR,
    sha256_file,
)

EXPERIMENT = "finetune-a4"
RUNS_DIR = OUT / "runs"
EXAMPLES_FILE = "train_examples.parquet"
SEED = 20261109
KEEP_LAST = 2
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                  "gate_proj", "up_proj", "down_proj")  # fmt: skip
COLUMNS = ["example_id", "ref", "stage", "order", "prompt", "n_tokens", "target_label",
           "target_token_id"]  # fmt: skip


class TrainInputError(RuntimeError):
    """An input that is not the A4 train set, or does not match its manifest."""


@dataclass(frozen=True)
class TrainConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = TARGET_MODULES
    grad_accum: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    schedule: str = "cosine"
    max_grad_norm: float = 1.0
    batch_size: int = 1
    max_tokens: int = MAX_TOKENS
    seed: int = SEED
    quant: str = "nf4 double-quant, bf16 compute"
    attn: str = "sdpa"
    gradient_checkpointing: str = "use_reentrant=False"
    loss: str = "answer-token cross-entropy, logits_to_keep=1"
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


# --- inputs ------------------------------------------------------------------------------------


def _refs(path: Path) -> set[str]:
    return {x for x in path.read_text(encoding="utf-8").splitlines() if x}


def load_examples(data_dir: Path = OUT, manifest: Path | None = None,
                  splits_dir: Path = SPLITS_DIR) -> tuple[pd.DataFrame, dict]:  # fmt: skip
    """The selected train examples, after the input checks (module docstring); and the facts."""
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    path = data_dir / EXAMPLES_FILE
    expected = meta.get("sha256", {})
    shas = {EXAMPLES_FILE: sha256_file(path)}
    if shas[EXAMPLES_FILE] != expected.get(EXAMPLES_FILE):
        raise TrainInputError(f"{path} SHA-256 does not match manifest.json")
    df = pd.read_parquet(path, columns=COLUMNS)
    if manifest is not None:
        manifest = Path(manifest)
        if manifest.resolve().parent != data_dir.resolve() or manifest.name not in expected:
            raise TrainInputError(
                f"{manifest} is not a manifest listed in {data_dir}/manifest.json"
            )
        if manifest.name in ("val_subset.parquet", EXAMPLES_FILE, "train_items.parquet"):
            raise TrainInputError(f"{manifest.name} is not an example-id manifest")
        shas[manifest.name] = sha256_file(manifest)
        if shas[manifest.name] != expected[manifest.name]:
            raise TrainInputError(f"{manifest} SHA-256 does not match manifest.json")
        ids = pd.read_parquet(manifest, columns=["example_id"])["example_id"]
        missing = set(ids) - set(df["example_id"])
        if missing:
            raise TrainInputError(f"{len(missing)} manifest ids are not train examples")
        df = df[df["example_id"].isin(set(ids))]
    check_train_refs(df["ref"], splits_dir)
    facts = {"manifest_json_sha256": sha256_file(data_dir / "manifest.json"), "sha256": shas,
             "builder_commit": meta.get("builder_commit", {}).get("commit"),
             "base_model": meta.get("base_model"), "rows": len(df)}  # fmt: skip
    if meta.get("base_model") != {"repo": BASE_REPO, "revision": BASE_REVISION}:
        raise TrainInputError(
            f"set built for {meta.get('base_model')}, not {BASE_REPO}@{BASE_REVISION}"
        )
    return df.sort_values("example_id").reset_index(drop=True), facts


def check_train_refs(refs: Iterable[str], splits_dir: Path = SPLITS_DIR) -> None:
    """Raise unless every ref is in the frozen rulings train split and in neither val nor test."""
    d = Path(splits_dir) / "rulings_v1"
    train, other = (
        _refs(d / "train_refs.txt"),
        _refs(d / "val_refs.txt") | _refs(d / "test_refs.txt"),
    )
    refs = set(map(str, refs))
    if bad := refs & other:
        raise TrainInputError(f"{len(bad)} refs are val or test refs (e.g. {min(bad)})")
    if bad := refs - train:
        raise TrainInputError(f"{len(bad)} refs are not in the train split (e.g. {min(bad)})")


def shuffled_order(example_ids: Sequence[str], seed: int) -> tuple[list[str], str]:
    ids = sorted(example_ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    order = [ids[i] for i in perm]
    return order, hashlib.sha256("\n".join(order).encode()).hexdigest()


def tokenise(df: pd.DataFrame, tok, max_tokens: int = MAX_TOKENS) -> dict[str, list[int]]:
    """example_id -> token ids (no special tokens added: the prompt is the full chat text)."""
    out = {}
    for r in df.itertuples():
        ids = list(tok(r.prompt, add_special_tokens=False)["input_ids"])
        if len(ids) != int(r.n_tokens):
            raise TrainInputError(f"{r.example_id}: {len(ids)} tokens, the set says {r.n_tokens}")
        if len(ids) > max_tokens:
            raise TrainInputError(
                f"{r.example_id}: {len(ids)} tokens > {max_tokens} (no truncation)"
            )
        out[r.example_id] = ids
    return out


# --- model -------------------------------------------------------------------------------------


def load_base(snapshot: Path = TOKENIZER_DIR):
    import torch
    from transformers import BitsAndBytesConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    return Qwen3_5ForCausalLM.from_pretrained(
        snapshot,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )


def attach_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                    target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM")  # fmt: skip
    return get_peft_model(model, lc)


def answer_logits(model, input_ids):
    """Logits at the answer position only: [batch, vocab], float32 (logits_to_keep=1)."""
    out = model(input_ids=input_ids, logits_to_keep=1, use_cache=False)
    if out.logits.shape[1] != 1:
        raise RuntimeError(f"expected logits at one position, got {tuple(out.logits.shape)}")
    return out.logits[:, -1, :].float()


def answer_loss(logits, target_ids):
    import torch.nn.functional as F

    return F.cross_entropy(logits, target_ids)


def label_token_ids(tok, labels: Sequence[str]) -> list[int]:
    from hs_rag_classifier.finetune.build_set import label_tokens

    m = label_tokens(tok, labels)
    return [m[lab] for lab in labels]


# --- schedule, checkpoints ---------------------------------------------------------------------


def n_optimiser_steps(n_examples: int, accum: int) -> int:
    return math.ceil(n_examples / accum)


def make_optimiser(model, cfg: TrainConfig, total_steps: int):
    import torch
    from transformers import get_cosine_schedule_with_warmup

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = math.ceil(cfg.warmup_ratio * total_steps)
    return opt, get_cosine_schedule_with_warmup(opt, warmup, total_steps), warmup


def rng_state() -> dict:
    import torch

    s = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    import torch

    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path: Path, model, opt, sched, state: dict) -> Path:
    import torch

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)  # a half-written checkpoint of our own from an interrupted save
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp)
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "rng": rng_state(),
                "state": state}, tmp / "trainer_state.pt")  # fmt: skip
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    return path


def load_checkpoint(path: Path, model, opt, sched) -> dict:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(path)
    res = set_peft_model_state_dict(model, load_file(str(path / "adapter_model.safetensors")))
    if getattr(res, "unexpected_keys", None):
        raise RuntimeError(f"unexpected adapter keys in {path}")
    ck = torch.load(path / "trainer_state.pt", weights_only=False)  # our own file
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    set_rng_state(ck["rng"])
    return ck["state"]


def step_checkpoints(ck_dir: Path) -> list[Path]:
    from hs_rag_classifier.guards import check_retrieval_inputs

    found = [p for p in Path(ck_dir).glob("step_*") if p.is_dir() and not p.name.endswith(".tmp")]
    # the repo's rule for any folder walk (guards.py), though these are the run's own checkpoints
    return sorted(check_retrieval_inputs(found, DATA_ROOT))


def prune(ck_dir: Path, keep: int = KEEP_LAST) -> list[Path]:
    """Remove all but the newest `keep` step checkpoints of this run (final/ is never touched)."""
    old = step_checkpoints(ck_dir)[:-keep] if keep else step_checkpoints(ck_dir)
    for p in old:
        shutil.rmtree(p)
    return old


# --- the loop ----------------------------------------------------------------------------------


def train_loop(model, examples: Sequence[tuple[str, list[int], int]], cfg: TrainConfig,
               ck_dir: Path, label_ids: Sequence[int], a_index: int = 0,
               checkpoint_every: int = 10, resume: bool = False,
               log: Callable[[dict, int], None] | None = None,
               stop_after_step: int | None = None, device: str = "cuda") -> dict:  # fmt: skip
    """One pass over examples (in the given order). Returns a summary dict.

    examples: (example_id, token ids, target token id). Checkpoints every checkpoint_every
    optimiser steps (keeping KEEP_LAST) and at the end (final/). resume: continue from the newest
    step checkpoint in ck_dir. stop_after_step: stop (after checkpointing) at that step; tests."""
    import torch

    ck_dir = Path(ck_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    total = n_optimiser_steps(len(examples), cfg.grad_accum)
    opt, sched, warmup = make_optimiser(model, cfg, total)
    order_sha = hashlib.sha256("\n".join(e[0] for e in examples).encode()).hexdigest()
    state = {"step": 0, "position": 0, "n_seen": 0, "n_a": 0, "order_sha256": order_sha,
             "config_key": cfg.key(), "total_steps": total}  # fmt: skip
    if resume:
        cks = step_checkpoints(ck_dir)
        if not cks:
            raise RuntimeError(f"nothing to resume in {ck_dir}")
        got = load_checkpoint(cks[-1], model, opt, sched)
        for k in ("order_sha256", "config_key", "total_steps"):
            if got[k] != state[k]:
                raise RuntimeError(f"checkpoint {cks[-1]} has a different {k}")
        state = got
    labels_t = torch.tensor(list(label_ids), device=device)
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    model.train()
    t_run = time.perf_counter()
    while state["step"] < total:
        group = examples[state["position"] : state["position"] + cfg.grad_accum]
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss_sum, n_tok = 0.0, 0
        for _eid, ids, target in group:
            x = torch.tensor([ids], device=device)
            logits = answer_logits(model, x)
            loss = answer_loss(logits, torch.tensor([int(target)], device=device))
            (loss / len(group)).backward()
            loss_sum += float(loss.detach())
            n_tok += len(ids)
            state["n_seen"] += 1
            state["n_a"] += int(int(logits[0, labels_t].argmax()) == a_index)
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                            cfg.max_grad_norm)  # fmt: skip
        lr = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        state["step"] += 1
        state["position"] += len(group)
        m = {"loss": loss_sum / len(group), "lr": lr, "grad_norm": float(gn), "tokens_per_s": n_tok / dt,
             "step_seconds": dt, "examples": len(group),
             "peak_gpu_gib": (torch.cuda.max_memory_allocated() / 2**30) if cuda else 0.0,
             "share_a_running": state["n_a"] / state["n_seen"]}  # fmt: skip
        if log:
            log(m, state["step"])
        if state["step"] % checkpoint_every == 0 and state["step"] < total:
            save_checkpoint(ck_dir / f"step_{state['step']:05d}", model, opt, sched, state)
            prune(ck_dir)
        if (
            stop_after_step is not None
            and state["step"] >= stop_after_step
            and state["step"] < total
        ):
            if state["step"] % checkpoint_every:
                save_checkpoint(ck_dir / f"step_{state['step']:05d}", model, opt, sched, state)
                prune(ck_dir)
            return {"stopped_at": state["step"], **state}
    save_checkpoint(ck_dir / "final", model, opt, sched, state)
    return {**state, "warmup_steps": warmup, "seconds": time.perf_counter() - t_run}


# --- CLI ---------------------------------------------------------------------------------------


def git_commit() -> dict[str, Any]:
    from hs_rag_classifier.eval import runner as R

    return R.git_info()


def cmd_train(a) -> int:
    import torch
    from transformers import AutoTokenizer

    from hs_rag_classifier.arms import a2_pipeline as A2
    from hs_rag_classifier.eval import runner as R

    cfg = TrainConfig()
    run_dir = Path(a.runs_dir) / a.run_name
    if run_dir.exists() and not a.resume:
        raise SystemExit(f"{run_dir} exists; use --resume to continue it")
    if (run_dir / "checkpoints" / "final").exists():
        raise SystemExit(f"{run_dir} is finished (checkpoints/final exists)")
    df, facts = load_examples(Path(a.data_dir), Path(a.manifest) if a.manifest else None)
    order, order_sha = shuffled_order(df["example_id"], cfg.seed)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    ids = tokenise(df, tok, cfg.max_tokens)
    by_id = df.set_index("example_id")
    examples = [(e, ids[e], int(by_id.at[e, "target_token_id"])) for e in order]
    labels = A2.load_config()["labels"]
    label_ids = label_token_ids(tok, labels)
    commit = git_commit()
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "run.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    if info and (info["order_sha256"] != order_sha or info["config_key"] != cfg.key()):
        raise SystemExit("the run's order or config differs from the one being resumed")
    total = n_optimiser_steps(len(examples), cfg.grad_accum)
    params = {"base_repo": BASE_REPO, "base_revision": BASE_REVISION, **{
        k: (",".join(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items() if k != "extra"},
        "config_key": cfg.key(), "n_examples": len(examples), "optimiser_steps": total,
        "warmup_steps": math.ceil(cfg.warmup_ratio * total), "order_sha256": order_sha,
        "manifest_json_sha256": facts["manifest_json_sha256"],
        **{f"sha256_{k}": v for k, v in facts["sha256"].items()},
        "selection": Path(a.manifest).name if a.manifest else "all",
        "builder_commit": facts["builder_commit"], "trainer_commit": commit["commit"],
        "checkpoint_every": a.checkpoint_every, "keep_last": KEEP_LAST}  # fmt: skip
    R.mlflow.set_tracking_uri(R.tracking_uri())
    exp = R._experiment_id(EXPERIMENT, R.MLFLOW_DIR / "artifacts")
    start = (
        {"run_id": info["mlflow_run_id"]}
        if info
        else {"experiment_id": exp, "run_name": a.run_name}
    )
    with R.mlflow.start_run(**start) as run:
        if not info:
            R.mlflow.log_params(params)
            R.mlflow.set_tags({"split": "train", "test_run": "false", "git_dirty": str(commit["dirty"]).lower(),
                               "run_dir": str(run_dir)})  # fmt: skip
            info = {**params, "mlflow_run_id": run.info.run_id, "run_dir": str(run_dir)}
            info_path.write_text(json.dumps(info, indent=1))
            (run_dir / "order.txt").write_text("\n".join(order) + "\n")
        steps_log = (run_dir / "steps.jsonl").open("a")

        def log(m: dict, step: int) -> None:
            R.mlflow.log_metrics(m, step=step)
            steps_log.write(json.dumps({"step": step, "time": time.time(), **m}) + "\n")
            steps_log.flush()
            print(json.dumps({"step": step, **{k: round(v, 5) for k, v in m.items()}}), flush=True)

        base = load_base()
        torch.manual_seed(cfg.seed)  # LoRA A initialisation and dropout
        model = attach_lora(base, cfg)
        res = train_loop(model, examples, cfg, run_dir / "checkpoints", label_ids, labels.index("A"),
                         a.checkpoint_every, resume=a.resume and bool(step_checkpoints(run_dir / "checkpoints")),
                         log=log)  # fmt: skip
        R.mlflow.log_metrics(
            {"train_seconds": res.get("seconds", 0.0), "final_share_a": res["n_a"] / res["n_seen"]}
        )
        (run_dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items()}, default=str))
    return 0


def tf_rows(val: pd.DataFrame) -> pd.DataFrame:
    from hs_rag_classifier.finetune.score_val import gate_rows

    return gate_rows(val)


def cmd_tf_score(a) -> int:
    """Report only: teacher-forced label NLL in transformers on the gate rows, NF4 base with and
    without the adapter (renormalised over each row's option labels, T = 1)."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from hs_rag_classifier.arms import a2_pipeline as A2

    val = pd.read_parquet(Path(a.data_dir) / "val_subset.parquet")
    rows = tf_rows(val)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    labels = A2.load_config()["labels"]
    lid = label_token_ids(tok, labels)
    model = PeftModel.from_pretrained(load_base(), a.adapter)
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows.itertuples():
            ids = list(tok(r.prompt, add_special_tokens=False)["input_ids"])
            x = torch.tensor([ids], device="cuda")
            k = len(r.options)
            g = labels.index(r.target_label)
            res = {"example_id": r.example_id}
            for name in ("adapter", "base"):
                if name == "base":
                    with model.disable_adapter():
                        lg = answer_logits(model, x)[0]
                else:
                    lg = answer_logits(model, x)[0]
                lp = torch.log_softmax(lg[lid[:k]], dim=-1)
                res[f"nll_{name}"] = float(-lp[g])
                res[f"choice_{name}"] = labels[int(lp.argmax())]
            out.append(res)
    df = pd.DataFrame(out)
    summ = {"n": len(df), "mean_nll_adapter": df.nll_adapter.mean(), "mean_nll_base": df.nll_base.mean(),
            "mean_diff": (df.nll_adapter - df.nll_base).mean(),
            "acc_adapter": float((df.choice_adapter == rows.target_label.values).mean()),
            "acc_base": float((df.choice_base == rows.target_label.values).mean()), "adapter": str(a.adapter)}  # fmt: skip
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": out}, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    ap = argparse.ArgumentParser(prog="python -m hs_rag_classifier.finetune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train an A4 LoRA adapter (resumable)")
    t.add_argument("--run-name", required=True)
    t.add_argument("--manifest", type=Path, help="restrict to this manifest's example ids")
    t.add_argument("--checkpoint-every", type=int, default=10)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--data-dir", type=Path, default=OUT)
    t.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    s = sub.add_parser(
        "tf-score", help="report only: teacher-forced gate-row NLL, with/without adapter"
    )
    s.add_argument("--adapter", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--data-dir", type=Path, default=OUT)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if Path(DATA_ROOT) not in Path(getattr(a, "runs_dir", RUNS_DIR)).resolve().parents:
        raise SystemExit("run outputs go under the data root, never the repo")
    return cmd_train(a) if a.cmd == "train" else cmd_tf_score(a)
