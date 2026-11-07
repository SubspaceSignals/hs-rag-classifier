"""A4 val-subset scoring through llama.cpp with A2's own scoring code, and the pilot gate.

    python -m hs_rag_classifier.finetune.score_val score --out OUT.json [--adapter GGUF]
    python -m hs_rag_classifier.finetune.score_val compare BASE.json ADAPTER.json [--out GATE.json]

score: every row of val_subset.parquet (forward order), through the running llama-server (-np 1,
one slot checked) with A2's score_call, label_logprobs and renormalise over the row's option
labels at T = 1. --adapter: the server must list exactly that GGUF in /lora-adapters at scale 1
(no per-request scale is sent); without it the server must list none. The system and user text
are recovered from the stored chat text and checked by re-applying the chat template exactly.
Per row: label log-probabilities, chosen label, gold label, NLL = -log p(gold) after
renormalisation (None where the gold is not among the options).

compare: on the gate rows (stage 1 with the gold HS6 among the options; 272 in this set), the
mean NLL of each, the mean difference (adapter - base) and a paired bootstrap 95% interval over
items (BOOTSTRAP_N resamples, BOOTSTRAP_SEED). Gate (fixed before the pilot, 7 Nov 2026): PASS if
the adapter's mean NLL is lower and the whole interval lies below zero. Also, not gated: stage-1
accuracy, predicted-label histograms and share at A, stage-2 NLL and accuracy, and the predicted
labels on the null-target stage-1 rows.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hs_rag_classifier.finetune.build_set import OUT, TOKENIZER_DIR, sha256_file

VAL_FILE = "val_subset.parquet"
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20261110
CHAT = re.compile(r"\A<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*)<\|im_end\|>\n"
                  r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n\Z", re.DOTALL)  # fmt: skip


def split_chat(text: str) -> tuple[str, str]:
    m = CHAT.match(text)
    if not m:
        raise ValueError("not a [system, user] chat text with the empty-thinking generation prompt")
    return m.group(1), m.group(2)


def gate_rows(val: pd.DataFrame) -> pd.DataFrame:
    """Stage-1 rows whose gold HS6 is among the options (forward order), in file order."""
    m = (val["stage"] == 1) & val["gold_in_options"].astype(bool) & (val["order"] == "forward")
    return val[m]


def row_nll(logprobs: Sequence[float], labels: Sequence[str], gold: str | None) -> float | None:
    from hs_rag_classifier.pipeline import scoring

    if gold is None or (isinstance(gold, float) and np.isnan(gold)):
        return None
    p = scoring.renormalise(logprobs, 1.0)
    return float(-np.log(p[list(labels).index(gold)]))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = BOOTSTRAP_N,
                     seed: int = BOOTSTRAP_SEED) -> dict:  # fmt: skip
    """Mean of (a - b) and its percentile 95% interval over n paired resamples of the items."""
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "n_items": len(d),
            "resamples": n, "seed": seed}  # fmt: skip


def gate(base_nll: Sequence[float], adapter_nll: Sequence[float]) -> dict:
    bs = paired_bootstrap(adapter_nll, base_nll)
    mb, ma = float(np.mean(base_nll)), float(np.mean(adapter_nll))
    ok = ma < mb and bs["ci95"][1] < 0
    return {"verdict": "PASS" if ok else "FAIL", "mean_nll_base": mb, "mean_nll_adapter": ma, **bs}


def score(out: Path, adapter: Path | None = None, data_dir: Path = OUT) -> dict:
    import httpx
    from transformers import AutoTokenizer

    from hs_rag_classifier.arms import a2_pipeline as A2
    from hs_rag_classifier.pipeline import scoring
    from hs_rag_classifier.pipeline.llm import check_server_slots, score_call

    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    meta = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    path = Path(data_dir) / VAL_FILE
    if sha256_file(path) != meta["sha256"][VAL_FILE]:
        raise SystemExit(f"{path} SHA-256 does not match manifest.json")
    cfg = A2.load_config()
    client, labels_all = cfg["client"], cfg["labels"]
    base = client["base_url"]
    tags = check_server_slots(base)
    loras = httpx.get(f"{base}/lora-adapters", timeout=10).json()
    if adapter is None and loras:
        raise SystemExit(f"base scoring, but the server has adapters loaded: {loras}")
    if adapter is not None:
        want = str(Path(adapter).resolve())
        if (
            len(loras) != 1
            or str(Path(loras[0]["path"]).resolve()) != want
            or loras[0]["scale"] != 1.0
        ):
            raise SystemExit(f"expected exactly {want} at scale 1, server has {loras}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    val = pd.read_parquet(path)
    val = val[val["order"] == "forward"]
    http = httpx.Client(base_url=base, timeout=client.get("timeout_s", 120))
    rows = []
    for r in val.itertuples():
        system, user = split_chat(r.prompt)
        again = tok.apply_chat_template([{"role": "system", "content": system},
                                         {"role": "user", "content": user}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)  # fmt: skip
        if again != r.prompt:
            raise SystemExit(f"{r.example_id}: chat text does not round-trip")
        labels = labels_all[: len(r.options)]
        res = score_call(
            http, client, system, user, int(cfg["a2"]["top_logprobs"]), f"a4 val stage{r.stage}"
        )
        d = scoring.label_logprobs(res["top"], labels)
        gold = r.target_label if isinstance(r.target_label, str) else None
        rows.append({"example_id": r.example_id, "ref": r.ref, "stage": int(r.stage),
                     "gold_in_options": bool(r.gold_in_options), "labels": labels,
                     "logprobs": d["logprobs"], "floored": d["floored"],
                     "chosen": labels[int(np.argmax(d["logprobs"]))], "gold": gold,
                     "nll": row_nll(d["logprobs"], labels, gold), "token": res["token"],
                     "tokens_in": res["tokens_in"], "seconds": res["seconds"]})  # fmt: skip
    rep = {"adapter": str(adapter) if adapter else None,
           "adapter_sha256": sha256_file(Path(adapter)) if adapter else None,
           "lora_adapters": loras, "server_tags": tags,
           "server_build": httpx.get(f"{base}/props", timeout=10).json().get("build_info"),
           "val_sha256": meta["sha256"][VAL_FILE], "temperature": 1.0, "rows": rows}  # fmt: skip
    out.write_text(json.dumps(rep, indent=1))
    return rep


def summarise(rows: pd.DataFrame) -> dict:
    s1 = rows[rows["stage"] == 1]
    g = s1[s1["gold_in_options"]]
    s2 = rows[rows["stage"] == 2]
    null = s1[s1["gold"].isna()]
    hist = Counter(s1["chosen"])
    return {"stage1_gate_rows": len(g), "stage1_gate_mean_nll": float(g["nll"].mean()),
            "stage1_gate_accuracy": float((g["chosen"] == g["gold"]).mean()),
            "stage1_accuracy_all_300": float((s1["chosen"] == s1["gold"]).mean()),
            "stage1_chosen_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
            "stage1_share_chosen_a": hist.get("A", 0) / max(1, len(s1)),
            "stage2_rows": len(s2), "stage2_mean_nll": float(s2["nll"].mean()),
            "stage2_accuracy": float((s2["chosen"] == s2["gold"]).mean()),
            "stage2_share_chosen_a": float((s2["chosen"] == "A").mean()),
            "null_rows": len(null), "null_rows_chosen": dict(Counter(null["chosen"]))}  # fmt: skip


def compare(base_path: Path, adapter_path: Path) -> dict:
    b = pd.DataFrame(json.loads(Path(base_path).read_text())["rows"])
    a = pd.DataFrame(json.loads(Path(adapter_path).read_text())["rows"])
    if list(a["example_id"]) != list(b["example_id"]):
        raise SystemExit("the two outputs are not over the same rows in the same order")
    gm = (b["stage"] == 1) & b["gold_in_options"]
    res = gate(b.loc[gm, "nll"].to_numpy(), a.loc[gm, "nll"].to_numpy())
    return {
        "gate": res,
        "base": summarise(b),
        "adapter": summarise(a),
        "inputs": {"base": str(base_path), "adapter": str(adapter_path)},
    }


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m hs_rag_classifier.finetune.score_val")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--adapter", type=Path)
    s.add_argument("--data-dir", type=Path, default=OUT)
    c = sub.add_parser("compare")
    c.add_argument("base", type=Path)
    c.add_argument("adapter", type=Path)
    c.add_argument("--out", type=Path)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.cmd == "score":
        rep = score(a.out, a.adapter, a.data_dir)
        print(json.dumps(summarise(pd.DataFrame(rep["rows"])), indent=1))
        return 0
    res = compare(a.base, a.adapter)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
