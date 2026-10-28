"""Choose the embedding model and tune B1 and B1-ex on rulings validation only.

    uv run --group ml python -m hs_rag_classifier.arms.embed_tune

For each model in arms.embed.MODELS (one on the GPU at a time):
    B1     one validation run at T = 1, T fitted on its top-10 scores (NLL of the top-1-correct
           indicator, as for B0), then a final run at the fitted T.
    B1-ex  one validation run per k in K_GRID; k chosen by top-1 at 8 digits, ties by recall at
           20; T fitted for that k, then a final run at the fitted T.
The model is chosen on B1: top-1 at 8 digits, ties by recall at 20. B1-ex uses the same model
(one embedder for the project); if the other model beats it on B1-ex top-1 at 8 digits by
more than the width of the winner's bootstrap interval, that gap is reported, not acted on.

Also a complementarity diagnostic (validation only, not an arm): the share of items whose true
cn8 is in the union of the top-20 lists from B0, B1 and B1-ex, each pair, and each arm alone.

Every run is logged to MLflow (experiment eval-rulings) under one parent run. Writes
configs/b1_embed.json and configs/b1ex_embed.json, and a summary to
~/data/hs-rag-classifier/eval/embed_tune_val.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from hs_rag_classifier.arms import b1_embed as B1
from hs_rag_classifier.arms import b1ex_embed as BX
from hs_rag_classifier.arms import embed as E
from hs_rag_classifier.arms.b0_bm25 import SOFTMAX_TOP, B0BM25Arm, fit_temperature, softmax_top1
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval.items import DATA_ROOT

TUNED_ON = "rulings val (splits/rulings_v1/val_refs.txt)"
SUMMARY = DATA_ROOT / "eval" / "embed_tune_val.json"


def nvidia_used_mb() -> int | None:
    try:
        query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        out = subprocess.run(query, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(out.stdout.split()[0])


def fit_t(frame) -> tuple[float, float, float]:
    """(T, ECE at T = 1, ECE at T) on the frame's top-10 scores and 8-digit correctness."""
    from hs_rag_classifier.eval import metrics as M

    top = [list(s[:SOFTMAX_TOP]) for s in frame["scores"]]
    correct = frame["correct_8"].to_numpy(float)
    t = fit_temperature(top, correct)
    before = M.ece(np.array([softmax_top1(s, 1.0) for s in top]), correct)
    after = M.ece(np.array([softmax_top1(s, t) for s in top]), correct)
    return t, before, after


def row(res: dict, **extra) -> dict:
    m = res["metrics"]
    ci = m["bootstrap"]["intervals"]["top1_8"]
    return {
        "top1_6": m["top1"][6]["accuracy"],
        "top1_8": m["top1"][8]["accuracy"],
        "top1_8_ci": [ci["low"], ci["high"]],
        **{f"recall_{k}": m["recall"].get(k) for k in (3, 10, 20, 50)},
        "ece": m["ece"],
        "items_per_minute": m["latency"]["items_per_minute"],
        "run_id": res["run_id"],
        **extra,
    }


def pick(rows: dict, key=lambda r: (r["top1_8"], r["recall_20"])):
    """Key of the best row; ties keep the earlier key."""
    keys = list(rows)
    return max(keys, key=lambda k: (*key(rows[k]), -keys.index(k)))


def top20_hits(frame) -> dict[str, set[str]]:
    """Per item: the cn8s among its top-20 candidates."""
    return {i: {c[:8] for c in cands[:20]} for i, cands in zip(frame["item_id"],
                                                                  frame["candidates"])}  # fmt: skip


def complementarity(frames: dict, labels: dict[str, str]) -> dict[str, float]:
    hits = {name: top20_hits(f) for name, f in frames.items()}
    out = {}
    for r in range(1, len(frames) + 1):
        for combo in itertools.combinations(frames, r):
            covered = [labels[i][:8] in set().union(*(hits[a][i] for a in combo)) for i in labels]
            out[" + ".join(combo)] = float(np.mean(covered))
    return out


def cmd_tune(args: argparse.Namespace) -> int:
    from hs_rag_classifier.eval import runner as R
    from hs_rag_classifier.eval.items import load

    evalset = load("rulings", "val", root=args.root)
    valid8, valid10 = R.valid_codes(args.root)
    git = R.git_info()
    R.mlflow.set_tracking_uri(R.tracking_uri())
    client = R.mlflow.MlflowClient()
    exp_id = R._experiment_id("eval-rulings", R.MLFLOW_DIR / "artifacts")
    parent = client.create_run(exp_id, run_name="embed-tune-val", tags={"embed_tune": "parent"})
    base = {"mlflow.parentRunId": parent.info.run_id}

    def run(arm, **tags):
        return R.run_eval(arm, evalset, valid8, valid10, git=git, n_resamples=args.resamples,
                          tags={**base, **tags})  # fmt: skip

    results: dict = {"models": {}}
    frames: dict = {}
    for model in args.models:
        import torch

        torch.cuda.reset_peak_memory_stats()
        encoder = E.Encoder(E.model_config(model))
        print(f"\n== {model}: loaded, GPU used {nvidia_used_mb()} MB")

        b1 = B1.B1EmbedArm(B1.default_config(model), root=args.root, encoder=encoder)
        r1 = run(b1, embed_tune=f"b1:{model}:T1")
        t, ece1, _ = fit_t(r1["frame"])
        b1.config["temperature"] = round(t, 4)
        r1t = run(b1, embed_tune=f"b1:{model}:final")
        b1_row = row(r1t, temperature=round(t, 4), ece_t1=ece1,
                     encode_seconds=b1.index.meta.get("encode_seconds"),
                     n_documents=len(b1.index.code10))  # fmt: skip
        frames[("b1", model)] = r1t["frame"]
        print(f"  B1 top-1@8 {b1_row['top1_8']:.4f}  recall@20 {b1_row['recall_20']:.4f}  "
              f"T {t:.4f}  ECE {ece1:.4f} -> {b1_row['ece']:.4f}")  # fmt: skip

        bx_index = BX.build_index(BX.default_config(model), encoder, root=args.root)
        grid = {}
        for k in BX.K_GRID:
            arm = BX.B1exEmbedArm(BX.default_config(model, k), root=args.root, encoder=encoder,
                                  index=bx_index)  # fmt: skip
            res = run(arm, embed_tune=f"b1ex:{model}:k{k}")
            grid[k] = row(res)
            grid[k]["_frame"] = res["frame"]
            print(f"  B1-ex k {k:<3} top-1@8 {grid[k]['top1_8']:.4f}  "
                  f"recall@20 {grid[k]['recall_20']:.4f}")  # fmt: skip
        k = pick(grid)
        t_x, ecex1, _ = fit_t(grid[k]["_frame"])
        bx = BX.B1exEmbedArm(BX.default_config(model, k, round(t_x, 4)), root=args.root,
                             encoder=encoder, index=bx_index)  # fmt: skip
        rxt = run(bx, embed_tune=f"b1ex:{model}:final")
        bx_row = row(rxt, k=k, temperature=round(t_x, 4), ece_t1=ecex1,
                     encode_seconds=bx_index.meta.get("encode_seconds"),
                     n_documents=len(bx_index.cn8))  # fmt: skip
        frames[("b1ex", model)] = rxt["frame"]
        print(f"  B1-ex k {k}: T {t_x:.4f}  ECE {ecex1:.4f} -> {bx_row['ece']:.4f}")

        results["models"][model] = {
            "b1": b1_row,
            "b1ex": bx_row,
            "b1ex_grid": {
                k: {a: b for a, b in v.items() if a != "_frame"} for k, v in grid.items()
            },
            "gpu": {**(E.gpu_memory() or {}), "nvidia_used_mb_loaded": nvidia_used_mb()},
        }
        E.release(encoder)
        del b1, bx, bx_index, encoder

    b1_rows = {m: results["models"][m]["b1"] for m in args.models}
    winner = pick(b1_rows)
    others = [m for m in args.models if m != winner]
    win_x = results["models"][winner]["b1ex"]
    width = win_x["top1_8_ci"][1] - win_x["top1_8_ci"][0]
    gaps = {m: results["models"][m]["b1ex"]["top1_8"] - win_x["top1_8"] for m in others}
    results.update(winner=winner, b1ex_ci_width=width, b1ex_gap_vs_winner=gaps,
                   b1ex_other_beats_by_more_than_ci={m: g > width for m, g in gaps.items()})  # fmt: skip

    b0 = run(B0BM25Arm(root=args.root), embed_tune="b0:complementarity")
    labels = dict(zip(b0["frame"]["item_id"], b0["frame"]["label_code"], strict=True))
    comp_frames = {"B0": b0["frame"], "B1": frames[("b1", winner)],
                   "B1-ex": frames[("b1ex", winner)]}  # fmt: skip
    results["complementarity_top20"] = complementarity(comp_frames, labels)
    results["b0_recall_20"] = b0["metrics"]["recall"][20]

    # Lock the winner's settings
    b1_cfg = B1.default_config(winner, b1_rows[winner]["temperature"])
    b1_cfg["tuned_on"] = TUNED_ON
    bx_cfg = BX.default_config(winner, win_x["k"], win_x["temperature"])
    bx_cfg["train_parquet_sha256"] = sha256(args.root / BX.TRAIN_REL)
    bx_cfg["k_grid"] = list(BX.K_GRID)
    bx_cfg["tuned_on"] = TUNED_ON
    for path, cfg in ((args.b1_config, b1_cfg), (args.b1ex_config, bx_cfg)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    client.log_param(parent.info.run_id, "winner", winner)
    client.log_param(parent.info.run_id, "b1ex_k", win_x["k"])
    for name, v in results["complementarity_top20"].items():
        client.log_metric(
            parent.info.run_id, "union20_" + name.replace(" + ", "_").replace("-", ""), v
        )
    client.set_terminated(parent.info.run_id)
    results["parent_run_id"] = parent.info.run_id
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWinner {winner}; B1-ex k {win_x['k']}; wrote {args.b1_config}, {args.b1ex_config}")
    print(f"Summary {SUMMARY}; parent run {parent.info.run_id}")
    print(json.dumps({"complementarity_top20": results["complementarity_top20"],
                      "b1ex_gap_vs_winner": gaps, "b1ex_ci_width": width}, indent=2))  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embed_tune", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--models", nargs="+", default=list(E.MODELS), choices=list(E.MODELS))
    parser.add_argument("--b1-config", type=Path, default=B1.CONFIG_PATH)
    parser.add_argument("--b1ex-config", type=Path, default=BX.CONFIG_PATH)
    parser.add_argument("--resamples", type=int, default=1000)
    return cmd_tune(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
