"""B2: fine-tuned encoder classifier at 6 digits (HS6).

Input: the ruling description only (as every arm). Label: HS6 = the first 6 digits of the cn8.
Label set: the HS6 codes present in the training data, sorted; saved under
~/data/hs-rag-classifier/models/b2/<config hash>/labels.json (not in the repo; its SHA-256 is in
the config). Items whose HS6 is outside it can't be predicted and count as wrong.

Training data (data = "uk_train"): the UK rulings train split only,
processed/rulings/v1/train.parquet; label_depth 8 and 10 rows both give an HS6. The loader
accepts exactly that file (guards.check_train_split_only) and passes it through
guards.check_retrieval_inputs: never validation, test, justifications, or anything under
the test-only folders in guards.FORBIDDEN_DIRS. Only TRAIN_COLUMNS are read.

Model: answerdotai/ModernBERT-base at a pinned revision with a sequence-classification head;
max length 512 (0.24% of train descriptions are longer).

Training: bf16 autocast on the GPU (fp32 weights), AdamW (lr 5e-5, weight decay 0.01), linear
warmup over the first 10% of steps then linear decay, batch size 32, up to 10 epochs, label
smoothing 0.1, fixed seed. After each epoch the model is scored on rulings validation (top-1
and recall at 3, at 6 digits); training stops after 2 epochs without a better top-1.
Checkpoints: <models>/b2/<config hash>/best/ (weights of the best epoch) and last/ (weights,
optimiser, scheduler, RNG state and history, for resume). Only those two are kept; each
overwrites its own earlier copy. Run again with the same config to resume from last/.

Prediction: softmax over the label set at temperature T (fitted on validation by minimising the
NLL of the true label, items outside the label set left out); the top 50 HS6 codes are the
candidates; confidence = top-1 probability. max_depth = 6, so the harness reports 8 and 10
digits as n/a. B2 never abstains.

    uv run --group ml python -m hs_rag_classifier.arms.b2_encoder train
trains, fits T, logs to MLflow (experiment train-b2) and writes configs/b2_encoder.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.data import rulings as rulings_data
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval.arms import Candidate, Prediction, config_hash
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem
from hs_rag_classifier.guards import check_retrieval_inputs, check_train_split_only

CONFIG_PATH = REPO_ROOT / "configs" / "b2_encoder.json"
MODEL_DIR = Path("models") / "b2"
TRAIN_REL = Path("processed") / "rulings" / rulings_data.SPLIT_NAME / "train.parquet"
TRAIN_REFS = REPO_ROOT / "splits" / f"rulings_{rulings_data.SPLIT_NAME}" / "train_refs.txt"
TRAIN_COLUMNS = ["ref", "description", "cn8", "label_depth"]
BASE_MODEL = "answerdotai/ModernBERT-base"
BASE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
MAX_DEPTH = 6
CANDIDATE_CAP = 50
TRAIN_KEYS = ("data", "base_model", "revision", "max_length", "lr", "weight_decay",
              "warmup_ratio", "batch_size", "grad_accum", "max_epochs", "patience",
              "label_smoothing", "seed", "dtype")  # fmt: skip


def default_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "data": "uk_train",
        "base_model": BASE_MODEL,
        "revision": BASE_REVISION,
        "max_length": 512,
        "lr": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "batch_size": 32,
        "grad_accum": 1,
        "max_epochs": 10,
        "patience": 2,
        "label_smoothing": 0.1,
        "seed": 20261028,
        "dtype": "bfloat16",
        "temperature": 1.0,
        "max_depth": MAX_DEPTH,
        "candidate_cap": CANDIDATE_CAP,
        "nomenclature_version": rulings_data.NOMENCLATURE_VERSION,
        "train_split_file": str(TRAIN_REL),
        "train_refs_sha256": sha256(TRAIN_REFS) if TRAIN_REFS.exists() else None,
    }
    config.update(overrides)
    return config


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def train_key(config: dict[str, Any]) -> str:
    """Hash of the settings that change the trained weights (T and results excluded)."""
    return config_hash({k: config[k] for k in TRAIN_KEYS})


def run_dir(config: dict[str, Any], root: Path = DATA_ROOT) -> Path:
    return root / MODEL_DIR / train_key(config)


# --- Labels and data ------------------------------------------------------------------------


def hs6(code: str) -> str:
    """HS6 of an 8- or 10-digit code."""
    if len(code) < 6 or not code[:6].isdigit():
        raise ValueError(f"not a commodity code: {code!r}")
    return code[:6]


def label_set(codes: Sequence[str]) -> list[str]:
    return sorted({hs6(c) for c in codes})


def outside_share(codes: Sequence[str], labels: Sequence[str]) -> float:
    """Share of codes whose HS6 is not in the label set (they can't be predicted)."""
    known = set(labels)
    return float(np.mean([hs6(c) not in known for c in codes])) if len(codes) else math.nan


def load_training_rows(root: Path = DATA_ROOT, train_path: Path | None = None) -> pd.DataFrame:
    """UK train rulings as (ref, text, hs6): train split file only, through both guards."""
    train_path = check_train_split_only(train_path or root / TRAIN_REL, root / TRAIN_REL)
    (train_path,) = check_retrieval_inputs([train_path], root)
    df = pd.read_parquet(train_path, columns=TRAIN_COLUMNS)
    return pd.DataFrame(
        {"ref": df["ref"], "text": df["description"], "hs6": [hs6(c) for c in df["cn8"]]}
    )


# --- Scoring helpers (no torch) -------------------------------------------------------------


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def nll(temperature: float, logits: np.ndarray, targets: np.ndarray) -> float:
    p = softmax(logits, temperature)[np.arange(len(targets)), targets]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """T minimising the NLL of the true label (search over log T); targets < 0 are left out."""
    from scipy.optimize import minimize_scalar

    keep = targets >= 0
    logits, targets = logits[keep], targets[keep]
    res = minimize_scalar(
        lambda log_t: nll(math.exp(log_t), logits, targets),
        bounds=(math.log(1e-2), math.log(1e2)),
        method="bounded",
    )
    return float(math.exp(res.x))


def top_k_hits(logits: np.ndarray, targets: np.ndarray, k: int) -> np.ndarray:
    """1 where the target is in the top k; targets < 0 (outside the label set) never hit."""
    top = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    return ((top == targets[:, None]).any(axis=1) & (targets >= 0)).astype(float)


def targets_for(codes: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(labels)}
    return np.array([index.get(hs6(c), -1) for c in codes])


# --- Model (torch) --------------------------------------------------------------------------


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"])


def new_model(config: dict[str, Any], labels: Sequence[str]):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        revision=config["revision"],
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={c: i for i, c in enumerate(labels)},
    )


def logits_for(model, tokenizer, texts: Sequence[str], max_length: int, dev: str,
               batch_size: int = 64) -> np.ndarray:  # fmt: skip
    import torch

    model.eval()
    out = []
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16):
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[start : start + batch_size]), truncation=True,
                            max_length=max_length, padding=True, return_tensors="pt")  # fmt: skip
            out.append(model(**{k: v.to(dev) for k, v in enc.items()}).logits.float().cpu())
    return torch.cat(out).numpy()


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    config: dict[str, Any],
    val_texts: Sequence[str],
    val_codes: Sequence[str],
    root: Path = DATA_ROOT,
    out_dir: Path | None = None,
    rows: pd.DataFrame | None = None,
    tokenizer=None,
    model_factory: Callable | None = None,
    log: Callable[[str, float, int], None] | None = None,
    dev: str | None = None,
) -> dict[str, Any]:
    """Train (or resume) one config; returns history, best epoch, timings and paths."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    dev = dev or device()
    out_dir = out_dir or run_dir(config, root)
    rows = rows if rows is not None else load_training_rows(root)
    labels = label_set(rows["hs6"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    tokenizer = tokenizer or load_tokenizer(config)
    seed_all(int(config["seed"]))
    model = (model_factory or new_model)(config, labels).to(dev)
    y = torch.tensor(targets_for(rows["hs6"], labels))
    val_y = targets_for(val_codes, labels)

    bs, accum = int(config["batch_size"]), int(config["grad_accum"])
    micro = bs // accum
    steps_per_epoch = math.ceil(len(rows) / bs)
    total = steps_per_epoch * int(config["max_epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                            weight_decay=float(config["weight_decay"]))  # fmt: skip
    sched = get_linear_schedule_with_warmup(opt, int(total * float(config["warmup_ratio"])), total)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))

    state = {"epoch": 0, "history": [], "best": -1.0, "best_epoch": 0, "bad": 0, "seconds": 0.0}
    last = out_dir / "last"
    if (last / "state.pt").exists():
        saved = torch.load(last / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        torch.set_rng_state(saved["rng"]["torch"])
        np.random.set_state(saved["rng"]["numpy"])
        random.setstate(saved["rng"]["python"])
        state = saved["state"]
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    texts = list(rows["text"])
    gen = torch.Generator().manual_seed(int(config["seed"]))
    for _ in range(state["epoch"]):
        torch.randperm(len(rows), generator=gen)  # replay the shuffles of finished epochs
    while state["epoch"] < int(config["max_epochs"]) and state["bad"] < int(config["patience"]):
        t0 = time.perf_counter()
        model.train()
        order = torch.randperm(len(rows), generator=gen)
        losses = []
        for step in range(steps_per_epoch):
            batch = order[step * bs : (step + 1) * bs]
            opt.zero_grad(set_to_none=True)
            for m in range(0, len(batch), micro):
                idx = batch[m : m + micro]
                enc = tokenizer([texts[i] for i in idx], truncation=True, padding=True,
                                max_length=int(config["max_length"]), return_tensors="pt")  # fmt: skip
                with torch.autocast(dev, dtype=torch.bfloat16):
                    logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
                loss = loss_fn(logits.float(), y[idx].to(dev)) * len(idx) / len(batch)
                loss.backward()
                losses.append(loss.item() * len(batch) / len(idx))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        val_logits = logits_for(model, tokenizer, val_texts, int(config["max_length"]), dev)
        top1 = float(top_k_hits(val_logits, val_y, 1).mean())
        rec3 = float(top_k_hits(val_logits, val_y, 3).mean())
        state["epoch"] += 1
        state["seconds"] += time.perf_counter() - t0
        state["history"].append({"epoch": state["epoch"], "train_loss": float(np.mean(losses)),
                                 "val_top1_6": top1, "val_recall_3_6": rec3})  # fmt: skip
        if log:
            for k, v in state["history"][-1].items():
                if k != "epoch":
                    log(k, v, state["epoch"])
        if top1 > state["best"]:
            state.update(best=top1, best_epoch=state["epoch"], bad=0)
            model.save_pretrained(out_dir / "best")
        else:
            state["bad"] += 1
        last.mkdir(exist_ok=True)
        rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
               "python": random.getstate()}  # fmt: skip
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": rng, "state": state},
                   last / "state.pt.tmp")  # fmt: skip
        (last / "state.pt.tmp").replace(last / "state.pt")
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None
    return {**state, "labels": labels, "out_dir": str(out_dir), "peak_vram_gb": peak,
            "train_size": len(rows)}  # fmt: skip


def load_trained(config: dict[str, Any], root: Path = DATA_ROOT, dev: str | None = None):
    """(model, tokenizer, labels) from the best checkpoint of this config."""
    from transformers import AutoModelForSequenceClassification

    dev = dev or device()
    out = run_dir(config, root)
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    model = AutoModelForSequenceClassification.from_pretrained(out / "best").to(dev)
    return model, load_tokenizer(config), labels


class B2EncoderArm:
    """ModernBERT HS6 classifier (rules in the module docstring)."""

    name = "b2"
    batch_size = 64
    max_depth = MAX_DEPTH

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 model=None, tokenizer=None, labels: Sequence[str] | None = None,
                 dev: str | None = None):  # fmt: skip
        self.config = dict(config or load_config())
        self.dev = dev or device()
        if model is None:
            model, tokenizer, labels = load_trained(self.config, root, self.dev)
        self.model, self.tokenizer, self.labels = model, tokenizer, list(labels)

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        cap, temp = int(self.config["candidate_cap"]), float(self.config["temperature"])
        logits = logits_for(self.model, self.tokenizer, [i.text for i in items],
                            int(self.config["max_length"]), self.dev)  # fmt: skip
        probs = softmax(logits, temp)
        out = []
        for item, p in zip(items, probs, strict=True):
            top = np.argsort(-p, kind="stable")[:cap]
            out.append(
                Prediction(
                    item_id=item.item_id,
                    candidates=[Candidate(code=self.labels[j], score=float(p[j])) for j in top],
                    confidence=float(p[top[0]]),
                    abstain=False,
                )
            )
        return out


# --- CLI: train, fit T, log, write the config -----------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    import torch

    from hs_rag_classifier.eval import runner as R
    from hs_rag_classifier.eval.items import load

    mlflow = R.mlflow
    config = default_config(data=args.data)
    val = load("rulings", "val", root=args.root)
    val_texts, val_codes = [i.text for i in val.items], [i.label_code for i in val.items]
    git = R.git_info()
    mlflow.set_tracking_uri(R.tracking_uri())
    exp_id = R._experiment_id("train-b2", R.MLFLOW_DIR / "artifacts")
    out_dir = run_dir(config, args.root)
    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"b2-{config['data']}-{train_key(config)}"
    ):
        rows = load_training_rows(args.root)
        labels = label_set(rows["hs6"])
        mlflow.log_params({"config": json.dumps({k: config[k] for k in TRAIN_KEYS}),
                           "config_hash": train_key(config), "revision": config["revision"],
                           "max_length": config["max_length"], "label_count": len(labels),
                           "train_size": len(rows), "seed": config["seed"]})  # fmt: skip
        mlflow.set_tags({"git_commit": git["commit"], "git_dirty": str(git["dirty"]).lower(),
                         "checkpoint_dir": str(out_dir), "data": config["data"]})  # fmt: skip
        wall = time.perf_counter()
        res = train(config, val_texts, val_codes, root=args.root, out_dir=out_dir, rows=rows,
                    log=lambda k, v, step: mlflow.log_metric(k, v, step=step))  # fmt: skip
        wall = time.perf_counter() - wall
        model, tokenizer, labels = load_trained(config, args.root)
        val_logits = logits_for(model, tokenizer, val_texts, config["max_length"], device())
        val_y = targets_for(val_codes, labels)
        temperature = fit_temperature(val_logits, val_y)
        outside = outside_share(val_codes, labels)
        mlflow.log_metrics({"best_epoch": res["best_epoch"], "best_val_top1_6": res["best"],
                            "train_seconds": res["seconds"], "wall_seconds": wall,
                            "peak_vram_gb": res["peak_vram_gb"] or math.nan,
                            "temperature": temperature, "val_outside_label_set": outside,
                            "nll_t1": nll(1.0, val_logits[val_y >= 0], val_y[val_y >= 0]),
                            "nll_fitted": nll(temperature, val_logits[val_y >= 0],
                                              val_y[val_y >= 0])})  # fmt: skip
        del model
        torch.cuda.empty_cache()
    labels_path = out_dir / "labels.json"
    config.update(temperature=round(temperature, 4), label_count=len(labels),
                  label_set_file=str(labels_path.relative_to(args.root)),
                  label_set_sha256=sha256(labels_path),
                  rulings_test_refs_sha256=sha256(REPO_ROOT / "splits" / "rulings_v1" /
                                                  "test_refs.txt"),
                  cross_revision=None,
                  tuned_on="rulings val (splits/rulings_v1/val_refs.txt)")  # fmt: skip
    summary = {**{k: v for k, v in res.items() if k != "labels"}, "wall_seconds": wall,
               "temperature": temperature, "val_outside_label_set": outside,
               "label_count": len(labels)}  # fmt: skip
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.write_config:
        args.config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.config}")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="b2_encoder", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train (or resume) one config on the GPU, fit T, log")
    tr.add_argument("--data", default="uk_train", choices=("uk_train",))
    tr.add_argument("--root", type=Path, default=DATA_ROOT)
    tr.add_argument("--config", type=Path, default=CONFIG_PATH)
    tr.add_argument("--write-config", action="store_true", help="lock this config as B2's")
    tr.set_defaults(func=cmd_train)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
