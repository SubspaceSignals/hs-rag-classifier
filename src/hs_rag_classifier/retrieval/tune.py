"""Tune the hybrid fusion and shortlist length on rulings validation; measure latency.

    uv run --group ml python -m hs_rag_classifier.retrieval.tune tune
    uv run --group ml python -m hs_rag_classifier.retrieval.tune latency

tune: each source's top-100 list is computed once per validation item (locked source configs).
Grid: k_rrf in K_RRF, weights over WEIGHT_VALUES per source, normalised with duplicate ratios
dropped; objective recall at 20 (8 digits), ties by recall at 10, then the earlier point. With
the best fusion: recall at K in KS (8 digits and HS6, cluster bootstrap), prompt tokens of the
K-candidate list ("cn8: path_text" lines joined by newlines, Qwen3.5 tokeniser: llama-server's
/tokenize if it answers, else the pinned Hugging Face tokeniser), the K rule in
retrieval.fusion.select_k, per-source contribution, the plain union of top-20s, and the R1
softmax temperature. Every grid point is an MLflow run nested under one parent (experiment
"retrieval"). Writes configs/hybrid_retrieval.json.

Limitation: there is no short-query validation set (A to Z is test only), so everything here is
tuned on rulings descriptions, which are paragraphs.

latency: single-query (one at a time, warm) timings over VAL_LATENCY_N validation items.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hs_rag_classifier.arms.b0_bm25 import SOFTMAX_TOP, nll_top1, softmax_top1
from hs_rag_classifier.arms.b0_bm25 import tokenise as bm25_tokenise
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT
from hs_rag_classifier.retrieval import hybrid as H
from hs_rag_classifier.retrieval.fusion import rrf, select_k, weight_grid

K_RRF = (10, 30, 60)
WEIGHT_VALUES = (0.5, 1, 2)
KS = (10, 20, 30, 40, 50)
TOKEN_BUDGET = 2000
MIN_GAIN = 0.01
TOKENIZER = {"hf_id": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
LLAMA_SERVER = "http://localhost:8080"
VAL_LATENCY_N = 200
LIMITATION = ("Tuned on rulings validation only (paragraph descriptions); no short-query "
              "validation set exists, since A to Z is test only.")  # fmt: skip


def candidate_line(cn8: str, path_text: str) -> str:
    return f"{cn8}: {path_text}"


def token_counter():
    """(count(text) -> int, description): llama-server /tokenize if running, else HF."""
    import httpx

    try:
        if httpx.get(f"{LLAMA_SERVER}/health", timeout=2).status_code == 200:

            def count(text: str) -> int:
                r = httpx.post(f"{LLAMA_SERVER}/tokenize", json={"content": text}, timeout=30)
                return len(r.json()["tokens"])

            return count, {"source": "llama-server /tokenize", "url": LLAMA_SERVER}
    except httpx.HTTPError:
        pass
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER["hf_id"], revision=TOKENIZER["revision"])
    return (lambda text: len(tok(text, add_special_tokens=False)["input_ids"])), {
        "source": "Hugging Face tokenizer",
        **TOKENIZER,
    }


def recall_flags(fused: list[str], label: str, ks=KS) -> dict[str, float]:
    out = {}
    for k in ks:
        top = fused[:k]
        out[f"recall_{k}"] = float(label[:8] in top)
        out[f"hs6_recall_{k}"] = float(label[:6] in {c[:6] for c in top})
    return out


def fused_lists(lists, weights, k_rrf) -> list[list[str]]:
    return [[c for c, _ in rrf({s: [h.cn8 for h in hits] for s, hits in item.items()}, weights,
                               k_rrf)] for item in lists]  # fmt: skip


def evaluate(lists, labels, weights, k_rrf) -> tuple[pd.DataFrame, list[list[str]]]:
    fused = fused_lists(lists, weights, k_rrf)
    return pd.DataFrame([recall_flags(f, lab) for f, lab in zip(fused, labels, strict=True)]), fused


def pick(grid: list[dict]) -> dict:
    return max(enumerate(grid), key=lambda e: (e[1]["recall_20"], e[1]["recall_10"], -e[0]))[1]


def fit_temperature_wide(top_scores, correct) -> float:
    """As B0's fit, over a range wide enough for RRF's small scores."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda lt: nll_top1(math.exp(lt), top_scores, correct),
                          bounds=(math.log(1e-6), math.log(10)), method="bounded")  # fmt: skip
    return float(math.exp(res.x))


def cmd_tune(args: argparse.Namespace) -> int:
    from hs_rag_classifier.eval import metrics as M
    from hs_rag_classifier.eval import runner as R
    from hs_rag_classifier.eval.items import load

    evalset = load("rulings", "val", root=args.root)
    items = evalset.items
    labels = [i.label_code for i in items]
    retriever = H.HybridRetriever(H.default_config(), root=args.root)
    t0 = time.perf_counter()
    lists = retriever.source_lists([i.text for i in items])
    print(f"source lists for {len(items)} val items in {time.perf_counter() - t0:.1f}s")

    # Each source alone (checks the sources reproduce the locked arms' recall at 20)
    alone = {s: float(np.mean([lab[:8] in [h.cn8 for h in it[s][:20]]
                               for it, lab in zip(lists, labels, strict=True)]))
             for s in H.SOURCES}  # fmt: skip
    print("each source alone, recall at 20:", {s: round(v, 4) for s, v in alone.items()})

    R.mlflow.set_tracking_uri(R.tracking_uri())
    client = R.mlflow.MlflowClient()
    exp_id = R._experiment_id("retrieval", R.MLFLOW_DIR / "artifacts")
    git = R.git_info()
    parent = client.create_run(exp_id, run_name="hybrid-rrf-grid-val",
                               tags={"git_commit": git["commit"], "split": "val",
                                     "dataset": "rulings", "split_sha256": evalset.split_sha256})  # fmt: skip
    grid = []
    for k_rrf in K_RRF:
        for w in weight_grid(H.SOURCES, WEIGHT_VALUES):
            frame, _ = evaluate(lists, labels, w, k_rrf)
            point = {"k_rrf": k_rrf, "weights": w, **frame.mean().to_dict()}
            grid.append(point)
            run = client.create_run(exp_id, run_name=f"rrf-k{k_rrf}",
                                    tags={"mlflow.parentRunId": parent.info.run_id})  # fmt: skip
            client.log_param(run.info.run_id, "k_rrf", k_rrf)
            for s, v in w.items():
                client.log_param(run.info.run_id, f"w_{s}", round(v, 6))
            for key, v in point.items():
                if key.startswith(("recall_", "hs6_recall_")):
                    client.log_metric(run.info.run_id, key, v)
            client.set_terminated(run.info.run_id)
    best = pick(grid)
    ranked = sorted(enumerate(grid), key=lambda e: (-e[1]["recall_20"], -e[1]["recall_10"], e[0]))
    print(f"\n{len(grid)} grid points; top 10 by recall at 20 (ties: recall at 10):")
    for _, g in ranked[:10]:
        ws = ", ".join(f"{s} {g['weights'][s]:.3f}" for s in H.SOURCES)
        print(f"  k_rrf {g['k_rrf']:<3} {ws}  r@10 {g['recall_10']:.4f}  r@20 {g['recall_20']:.4f}"
              f"  r@50 {g['recall_50']:.4f}  hs6@20 {g['hs6_recall_20']:.4f}")  # fmt: skip

    # Recall table with cluster bootstrap
    frame, fused = evaluate(lists, labels, best["weights"], best["k_rrf"])
    frame["cluster_id"] = [i.cluster_id for i in items]
    stats = {c: (lambda a, c=c: float(a[c].mean())) for c in frame.columns if "recall" in c}
    ci = M.bootstrap_intervals(frame, stats)
    table = {c: {"value": float(frame[c].mean()), **ci[c]} for c in stats}

    # Token cost of a K-candidate list
    count, tok_desc = token_counter()
    meta = retriever.meta
    tokens = {k: [count("\n".join(candidate_line(c, meta[c]["path_text"]) for c in f[:k]))
                  for f in fused] for k in KS}  # fmt: skip
    tok_stats = {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
                     "max": int(np.max(v))} for k, v in tokens.items()}  # fmt: skip
    choice = select_k(KS, {k: table[f"recall_{k}"]["value"] for k in KS},
                      {k: tok_stats[k]["p95"] for k in KS}, TOKEN_BUDGET, MIN_GAIN)  # fmt: skip
    K = choice["k"]

    # Per-source contribution at K: only in that source's list; or lost when it is dropped
    contribution = {}
    in_list = [{s: {h.cn8 for h in it[s]} for s in H.SOURCES} for it in lists]
    for s in H.SOURCES:
        others = {o: best["weights"][o] for o in H.SOURCES if o != s}
        without = fused_lists([{o: it[o] for o in others} for it in lists], others, best["k_rrf"])
        hit = [lab[:8] in f[:K] for f, lab in zip(fused, labels, strict=True)]
        only = [h and all(lab[:8] not in in_list[n][o] for o in others)
                for n, (h, lab) in enumerate(zip(hit, labels, strict=True))]  # fmt: skip
        lost = [h and lab[:8] not in w[:K] for h, w, lab in zip(hit, without, labels, strict=True)]
        contribution[s] = {"only_in_this_source": float(np.mean(only)),
                           "lost_if_dropped": float(np.mean(lost))}  # fmt: skip

    # The plain union of each source's top 20 (as in the step-6 diagnostic)
    unions = [set().union(*({h.cn8 for h in it[s][:20]} for s in H.SOURCES)) for it in lists]
    union = {"recall": float(np.mean([lab[:8] in u for u, lab in zip(unions, labels, strict=True)])),
             "mean_size": float(np.mean([len(u) for u in unions])),
             "p95_size": float(np.percentile([len(u) for u in unions], 95))}  # fmt: skip
    same_size = round(union["mean_size"])
    union["fused_recall_at_same_size"] = float(np.mean(
        [lab[:8] in f[:same_size] for f, lab in zip(fused, labels, strict=True)]))  # fmt: skip

    # R1 temperature: softmax over the top-10 fused scores
    full = [rrf({s: [h.cn8 for h in hits] for s, hits in it.items()}, best["weights"],
                best["k_rrf"]) for it in lists]  # fmt: skip
    top_scores = [[s for _, s in f[:SOFTMAX_TOP]] for f in full]
    correct = np.array([float(f[0][0] == lab[:8]) for f, lab in zip(full, labels, strict=True)])
    temperature = fit_temperature_wide(top_scores, correct)
    ece_t = M.ece(np.array([softmax_top1(s, temperature) for s in top_scores]), correct)

    config = H.default_config(best["k_rrf"], {s: round(v, 6) for s, v in best["weights"].items()},
                              K)  # fmt: skip
    config.update({
        "objective": "recall at 20 (8 digits), ties by recall at 10",
        "k_choice": {**choice, "ks": list(KS), "min_gain": MIN_GAIN},
        "token_budget": TOKEN_BUDGET,
        "candidate_line": "cn8: path_text, one per line",
        "tokenizer": tok_desc,
        "r1_temperature": round(temperature, 8),
        "rulings_val_refs_sha256": evalset.split_sha256,
        "rulings_test_refs_sha256": sha256(REPO_ROOT / "splits" / "rulings_v1" / "test_refs.txt"),
        "tuned_on": "rulings val (splits/rulings_v1/val_refs.txt)",
        "limitation": LIMITATION,
    })  # fmt: skip
    args.config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    summary = {"alone_recall_20": alone, "grid_size": len(grid),
               "top10": [{k: v for k, v in g.items()} for _, g in ranked[:10]],
               "best": best, "recall_table": table, "tokens": tok_stats, "k_choice": choice,
               "contribution": contribution, "union": union, "r1_temperature": temperature,
               "r1_val_ece": ece_t, "r1_val_top1_8": float(correct.mean()),
               "parent_run_id": parent.info.run_id}  # fmt: skip
    for key in ("k_rrf",):
        client.log_param(parent.info.run_id, key, best[key])
    for s, v in best["weights"].items():
        client.log_param(parent.info.run_id, f"w_{s}", round(v, 6))
    client.log_param(parent.info.run_id, "K", K)
    for c, v in table.items():
        client.log_metric(parent.info.run_id, c, v["value"])
    client.set_terminated(parent.info.run_id)
    out = args.root / "eval" / "hybrid_tune_val.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("best", "k_choice", "tokens", "contribution",
                                              "union", "r1_temperature", "r1_val_ece")},
                     indent=2, default=str))  # fmt: skip
    print("\nrecall table (value [low, high]):")
    for c, v in table.items():
        print(f"  {c:<16} {v['value']:.4f} [{v['low']:.4f}, {v['high']:.4f}]")
    print(f"wrote {args.config} and {out}")
    return 0


def cmd_latency(args: argparse.Namespace) -> int:
    from hs_rag_classifier.eval.items import load

    retriever = H.HybridRetriever(root=args.root)
    texts = [i.text for i in load("rulings", "val", root=args.root).items][: args.n]
    for t in texts[:10]:  # warm up
        retriever.retrieve(t)
    times: dict[str, list[float]] = {k: [] for k in ("encode", "keyword", "tariff_embed",
                                                      "rulings", "fuse", "full")}  # fmt: skip
    src = retriever.sources
    for t in texts:
        a = time.perf_counter()
        q = retriever.encoder.queries([t])[0]
        b = time.perf_counter()
        kw = src["keyword"].search_tokens(bm25_tokenise([t])[0], retriever.depth)
        c = time.perf_counter()
        te = src["tariff_embed"].search(q, retriever.depth)
        d = time.perf_counter()
        ru = src["rulings"].search(q, retriever.depth)
        e = time.perf_counter()
        retriever.fuse({"keyword": kw, "tariff_embed": te, "rulings": ru})
        f = time.perf_counter()
        retriever.retrieve(t)
        g = time.perf_counter()
        for key, v in zip(times, (b - a, c - b, d - c, e - d, f - e, g - f), strict=True):
            times[key].append(v * 1000)
    table = {k: {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95))}
             for k, v in times.items()}  # fmt: skip
    for k, v in table.items():
        print(f"  {k:<13} p50 {v['p50_ms']:8.2f} ms   p95 {v['p95_ms']:8.2f} ms")
    out = args.root / "eval" / "hybrid_latency_val.json"
    out.write_text(json.dumps({"n": len(texts), "ms": table}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retrieval.tune", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    tu = sub.add_parser("tune", help="tune fusion and K on rulings validation; write the config")
    tu.add_argument("--root", type=Path, default=DATA_ROOT)
    tu.add_argument("--config", type=Path, default=H.CONFIG_PATH)
    tu.set_defaults(func=cmd_tune)
    la = sub.add_parser("latency", help="single-query latency on validation items")
    la.add_argument("--root", type=Path, default=DATA_ROOT)
    la.add_argument("--n", type=int, default=VAL_LATENCY_N)
    la.set_defaults(func=cmd_latency)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
