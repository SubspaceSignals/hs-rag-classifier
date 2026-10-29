"""A2 set-up and reports on rulings validation (never test).

    python -m hs_rag_classifier.pipeline.a2_tune template-check [--n 10]
    python -m hs_rag_classifier.pipeline.a2_tune write-config --top-logprobs N
    python -m hs_rag_classifier.pipeline.a2_tune fit-temperature --run RUN_ID
    python -m hs_rag_classifier.pipeline.a2_tune reverse-order --run RUN_ID [--n 100]
    python -m hs_rag_classifier.pipeline.a2_tune report --run RUN_ID --a1 RUN_ID --b1ex RUN_ID
    python -m hs_rag_classifier.pipeline.a2_tune a1-citations --run RUN_ID

template-check: on the first N val items, the stage-1 prompt's first generated token and its top
5 tokens, and how many of the shown labels each top_logprobs cut-off would cover.
write-config: configs/a2_pipeline.json (scoring parameters, temperatures, stage-2 rule,
rationale limits, a1_pipeline.json and prompt SHA-256).
fit-temperature: T per stage and shared on a finished val run's saved distributions; the
lower total NLL wins and is written into the config.
reverse-order: stage 1 again on a seeded subset with the option order reversed (model run).
report, a1-citations: from saved predictions only.
Outputs go to ~/data/hs-rag-classifier/eval/.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.arms import a1_pipeline as A1
from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, load
from hs_rag_classifier.pipeline import checks, scoring

OUT = DATA_ROOT / "eval"
STAGE2_TOP_HS6 = 2
RATIONALE = {
    "max_tokens": 150,
    "quote_chars": 160,
    "explanation_chars": 360,
    "explanation_words": 60,
}
CUTOFFS = (20, 50, 100, 200, 500, 1000)
REVERSE_SEED = 20261029


def write_config(
    top_logprobs: int,
    temperature: dict[str, Any] | None = None,
    path: Path = A2.CONFIG_PATH,
    repo: Path = REPO_ROOT,
) -> dict:
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cfg = {
        "base_config": {"path": A2.BASE_CONFIG, "sha256": sha256(repo / A2.BASE_CONFIG)},
        "scoring": {
            "endpoint": "/v1/chat/completions",
            "max_tokens": 1,
            "logprobs": True,
            "grammar": None,
            "label_match": "exact token string of the bare label",
            "floor": "lowest returned log-probability for labels absent from the list",
            "probabilities": "before sampling (llama-server post_sampling_probs false)",
        },
        "top_logprobs": top_logprobs,
        "stage2_top_hs6": STAGE2_TOP_HS6,
        "temperature": temperature
        or old.get("temperature")
        or {"mode": "none (T = 1, before fitting)", "stage1": 1.0, "stage2": 1.0},
        "rationale": RATIONALE,
        "prompts": A2.prompt_entries(repo),
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


# --- template check (model calls) --------------------------------------------------------


def template_check(n: int, top_logprobs: int = max(CUTOFFS)) -> dict:
    from hs_rag_classifier.pipeline.llm import score_call

    config = {
        **A1.load_config(),
        "a2": {
            "top_logprobs": top_logprobs,
            "stage2_top_hs6": STAGE2_TOP_HS6,
            "rationale": RATIONALE,
            "temperature": {"stage1": 1.0, "stage2": 1.0},
        },
    }
    pipe = A2.build_pipeline(config, api=False)
    items = load("rulings", "val").items[:n]
    rows = []
    for it in items:
        s: dict[str, Any] = {"raw": it.text}
        s.update(pipe.input_check(s))
        s.update(pipe.retrieve(s))
        s.update(pipe.evidence(s))
        hs6s = pipe.stage1_options(s["fused"])
        shown = {lab: pipe.tariff.hs6_line(lab, h) for lab, h in zip(pipe.labels, hs6s)}
        vals = {**pipe._blocks(s, True), "options": "\n".join(shown.values())}
        from hs_rag_classifier.pipeline.graph import fill

        r = score_call(
            pipe.http,
            config["client"],
            pipe.a2_prompts["a2_stage1_system"],
            fill(pipe.a2_prompts["a2_stage1_user"], vals),
            top_logprobs,
        )
        tokens = [t["token"] for t in r["top"]]
        labels = list(shown)
        cover = {c: sum(lab in tokens[:c] for lab in labels) for c in CUTOFFS}
        d = scoring.label_logprobs(r["top"], labels)
        rows.append(
            {
                "item_id": it.item_id,
                "first_token": r["token"],
                "top5": [(t["token"], round(float(np.exp(t["logprob"])), 4)) for t in r["top"][:5]],
                "n_labels": len(labels),
                "covered": cover,
                "outside_mass": d["outside_mass"],
            }
        )
        print(
            f"{it.item_id}: first {r['token']!r}; top 5 {rows[-1]['top5']}; "
            f"labels covered {cover}; outside mass {d['outside_mass']:.4f}",
            flush=True,
        )
    return {"n": n, "top_logprobs": top_logprobs, "rows": rows}


# --- frames and rescoring ----------------------------------------------------------------


def _items(dataset: str = "rulings", split: str = "val"):
    return load(dataset, split).items


def rescored_frame(
    frame: pd.DataFrame, t1: float, t2: float, dataset="rulings", split="val"
) -> pd.DataFrame:
    """The run's scored frame re-scored with other temperatures (no model run)."""
    from hs_rag_classifier.eval.arms import Candidate, Prediction
    from hs_rag_classifier.eval.runner import valid_codes

    items = _items(dataset, split)
    by_id = {r.item_id: r for r in frame.itertuples(index=False)}
    preds, lat = [], []
    for it in items:
        r = by_id[it.item_id]
        details = json.loads(r.details)
        order, conf = A2.rescore(details, t1, t2)
        preds.append(
            Prediction(
                item_id=it.item_id,
                candidates=[Candidate(code=c, score=1.0 / (k + 1)) for k, c in enumerate(order)],
                confidence=conf,
                abstain=False,
                details=details,
            )
        )
        lat.append(r.latency_s)
    valid8, valid10 = valid_codes()
    return M.score(items, preds, valid8, valid10, lat, 8)


def summary(frame: pd.DataFrame, review_rate: float = M.HEADLINE_REVIEW_RATE) -> dict:
    m = M.compute_metrics(frame, float(frame["latency_s"].sum()), review_rate, max_depth=8)
    ci = m["bootstrap"]["intervals"]
    return {
        "top1_6": [m["top1"][6]["accuracy"], ci["top1_6"]["low"], ci["top1_6"]["high"]],
        "top1_8": [m["top1"][8]["accuracy"], ci["top1_8"]["low"], ci["top1_8"]["high"]],
        "recall_3": [m["recall"][3], ci["recall_3"]["low"], ci["recall_3"]["high"]],
        "ece": [m["ece"], ci["ece"]["low"], ci["ece"]["high"]],
        "headline": [m["headline"], ci["headline"]["low"], ci["headline"]["high"]],
        "coverage": [(c["review_rate"], c["accuracy"]) for c in m["coverage"]],
    }


def fit_temperature(run_id: str) -> dict:
    from hs_rag_classifier.eval.runner import load_run_frame

    frame = load_run_frame(run_id)
    rows = [json.loads(d) for d in frame["details"]]
    g = A2.temperature_groups(rows, list(frame["label_code"]))
    per = {k: scoring.fit_temperature(v) for k, v in g.items()}
    both = g["stage1"] + g["stage2"]
    shared = scoring.fit_temperature(both)
    shared["nll_stage1"] = scoring.nll(g["stage1"], shared["t"])
    shared["nll_stage2"] = scoring.nll(g["stage2"], shared["t"])
    per_total = per["stage1"]["nll"] + per["stage2"]["nll"]
    mode = "per_stage" if per_total <= shared["nll"] else "shared"
    t1, t2 = (
        (per["stage1"]["t"], per["stage2"]["t"])
        if mode == "per_stage"
        else (shared["t"], shared["t"])
    )
    raw = summary(frame)
    after = summary(rescored_frame(frame, t1, t2))
    out = {
        "run_id": run_id,
        "per_stage": per,
        "per_stage_total_nll": per_total,
        "shared": shared,
        "nll_t1_total": per["stage1"]["nll_t1"] + per["stage2"]["nll_t1"],
        "chosen": mode,
        "stage1": t1,
        "stage2": t2,
        "raw": raw,
        "after_t": after,
    }
    (OUT / "a2_temperature_val.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    cfg = json.loads(A2.CONFIG_PATH.read_text(encoding="utf-8"))
    write_config(
        cfg["top_logprobs"],
        {
            "mode": mode,
            "stage1": round(t1, 4),
            "stage2": round(t2, 4),
            "fitted_on": f"rulings val run {run_id}",
            "val_nll": {
                "per_stage": round(per_total, 3),
                "shared": round(shared["nll"], 3),
                "t1": round(out["nll_t1_total"], 3),
            },
        },
    )
    return out


# --- position bias -----------------------------------------------------------------------


def position_bias(frame: pd.DataFrame, bins=((0, 1), (1, 3), (3, 10), (10, 20), (20, 52))):
    rows = [json.loads(d) for d in frame["details"]]
    labels = list(frame["label_code"])
    chosen = [r["stage1"]["options"].index(r["stage1"]["choice_code"]) for r in rows]
    true_pos = [
        r["stage1"]["options"].index(lab[:6]) if lab[:6] in r["stage1"]["options"] else None
        for r, lab in zip(rows, labels, strict=True)
    ]
    c6 = frame["correct_6"].to_numpy(float)
    c8 = frame["correct_8"].to_numpy(float)
    out = {"chosen_share": {}, "true_share": {}, "accuracy_by_true_position": {}}
    for lo, hi in bins:
        key = f"{lo + 1}" if hi == lo + 1 else f"{lo + 1}-{hi}"
        out["chosen_share"][key] = float(np.mean([lo <= c < hi for c in chosen]))
        idx = [i for i, p in enumerate(true_pos) if p is not None and lo <= p < hi]
        out["true_share"][key] = len(idx) / len(rows)
        out["accuracy_by_true_position"][key] = {
            "n": len(idx),
            "hs6": float(np.mean(c6[idx])) if idx else None,
            "cn8": float(np.mean(c8[idx])) if idx else None,
        }
    out["true_not_shown"] = sum(p is None for p in true_pos) / len(rows)
    return out


def reverse_order(run_id: str, n: int = 100, seed: int = REVERSE_SEED) -> dict:
    """Stage 1 again with the option order reversed on a seeded subset; top-1 HS6 changes."""
    from hs_rag_classifier.eval.runner import load_run_frame
    from hs_rag_classifier.pipeline.graph import fill
    from hs_rag_classifier.pipeline.llm import score_call

    frame = load_run_frame(run_id)
    config = A2.load_config()
    pipe = A2.build_pipeline(config, api=False)
    by_id = {r.item_id: r for r in frame.itertuples(index=False)}
    items = {it.item_id: it for it in _items()}
    ids = sorted(random.Random(seed).sample(sorted(by_id), n))
    changed, rows = 0, []
    for k, item_id in enumerate(ids):
        d = json.loads(by_id[item_id].details)
        s: dict[str, Any] = {"raw": items[item_id].text, "evidence": d["evidence"]}
        s.update(pipe.input_check(s))
        hs6s = list(reversed(d["stage1"]["options"]))
        shown = {lab: pipe.tariff.hs6_line(lab, h) for lab, h in zip(pipe.labels, hs6s)}
        vals = {**pipe._blocks(s, True), "options": "\n".join(shown.values())}
        r = score_call(
            pipe.http,
            config["client"],
            pipe.a2_prompts["a2_stage1_system"],
            fill(pipe.a2_prompts["a2_stage1_user"], vals),
            int(config["a2"]["top_logprobs"]),
        )
        lp = scoring.label_logprobs(r["top"], list(shown))["logprobs"]
        top = hs6s[scoring.top_k_indices(lp, 1)[0]]
        orig = d["stage1"]["choice_code"]
        changed += top != orig
        rows.append(
            {
                "item_id": item_id,
                "original": orig,
                "reversed": top,
                "original_pos": d["stage1"]["options"].index(orig),
                "reversed_pos_in_original": d["stage1"]["options"].index(top),
            }
        )
        if (k + 1) % 25 == 0:
            print(f"  {k + 1} of {n}, changed so far {changed}", flush=True)
    out = {"run_id": run_id, "n": n, "seed": seed, "top1_hs6_changed": changed / n, "rows": rows}
    (OUT / "a2_reverse_order_val.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --- agreement signal --------------------------------------------------------------------


def _merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    cols = ["item_id", "cluster_id", "correct_8", "confidence", "abstain", "candidates"]
    m = a[cols].merge(
        b[["item_id", "candidates", "correct_8", "confidence"]], on="item_id", suffixes=("", "_b")
    )
    m["agree"] = [
        len(x) > 0 and len(y) > 0 and x[0][:8] == y[0][:8]
        for x, y in zip(m["candidates"], m["candidates_b"], strict=True)
    ]
    return m


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = _merge(a, b)
    ag = m["agree"].to_numpy(bool)
    c = m["correct_8"].to_numpy(float)
    conf = m["confidence"].to_numpy(float)
    none = np.zeros(len(m), bool)
    # review order: disagreements first, then ascending confidence (the key adds 1 when agreeing)
    combo = M.coverage_curve(conf + ag.astype(float), c, none)
    alone = M.coverage_curve(conf, c, none)
    return {
        "n": len(m),
        "agree_share": float(ag.mean()),
        "accuracy_agree": float(c[ag].mean()),
        "accuracy_disagree": float(c[~ag].mean()),
        "coverage_agree_then_confidence": [(x["review_rate"], x["accuracy"]) for x in combo],
        "coverage_confidence": [(x["review_rate"], x["accuracy"]) for x in alone],
    }


# --- report ------------------------------------------------------------------------------


def report(run_id: str, a1_run: str, b1ex_run: str) -> dict:
    from hs_rag_classifier.eval.runner import load_run_frame

    temp = json.loads((OUT / "a2_temperature_val.json").read_text(encoding="utf-8"))
    raw = load_run_frame(run_id)
    a2 = rescored_frame(raw, temp["stage1"], temp["stage2"])
    a1, b1ex = load_run_frame(a1_run), load_run_frame(b1ex_run)
    out = {
        "run_id": run_id,
        "temperature": [temp["stage1"], temp["stage2"]],
        "raw": temp["raw"],
        "after_t": temp["after_t"],
        "diagnostics": A2.diagnostics(raw),
        "a2_minus_a1": M.paired_difference(a2, a1),
        "a2_minus_b1ex": M.paired_difference(a2, b1ex),
        "a2_minus_a1_review20": M.paired_review_difference(a2, a1),
        "position_bias": position_bias(a2),
        "agreement_a1_b1ex": agreement(a1, b1ex),
        "agreement_a2_b1ex": agreement(a2, b1ex),
    }
    (OUT / "a2_val_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --- A1 citation failures ----------------------------------------------------------------

CATEGORIES = (
    "past_ruling_evidence",
    "description",
    "paraphrase_of_option",
    "different_option",
    "other",
)


def _found(q: str, text: str) -> bool:
    return checks.quote_check(q, text)["pass"]


def _overlap(q: str, text: str) -> float:
    words = set(re.findall(r"[a-z0-9]+", q.lower()))
    tw = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(words & tw) / len(words) if words else 0.0


def classify_failure(
    rationale: str, chosen_text: str, other_texts: list[str], description: str, evidence: list[str]
) -> str:
    """One category for a failed citation (first rule that applies, in CATEGORIES order)."""
    qs = checks.quotes(rationale)
    if not qs:
        return "other"
    bad = [q for q in qs if not _found(q, chosen_text)]
    ev = " ".join(evidence or [])
    if ev and any(_found(q, ev) for q in bad):
        return "past_ruling_evidence"
    if any(_found(q, description) for q in bad):
        return "description"
    if any(_overlap(q, chosen_text) >= 0.5 for q in bad):
        return "paraphrase_of_option"
    if any(_found(q, t) for q in bad for t in other_texts):
        return "different_option"
    return "other"


def a1_citations(run_id: str) -> dict:
    from hs_rag_classifier.eval.runner import load_run_frame
    from hs_rag_classifier.pipeline.options import load_tariff

    frame = load_run_frame(run_id)
    tariff = load_tariff()
    items = {it.item_id: it for it in _items()}
    counts = {s: dict.fromkeys(CATEGORIES, 0) for s in ("stage1", "stage2")}
    no_quote = {"stage1": 0, "stage2": 0}
    examples: dict[str, dict[str, list]] = {s: {c: [] for c in CATEGORIES} for s in counts}
    n_fail = {"stage1": 0, "stage2": 0}
    for r in frame.itertuples(index=False):
        d = json.loads(r.details)
        desc = checks.input_check(items[r.item_id].text)["description"]
        s1 = d["stage1"]
        stages = [
            (
                "stage1",
                s1["rationale"],
                s1["citation"],
                s1["choice_code"],
                {h: tariff.hs6_text[h] for h in s1["options"]},
            )
        ]
        s2 = d["stage2"]
        if not s2["skipped"]:
            kids = tariff.children[s1["choice_code"]]
            stages.append(
                (
                    "stage2",
                    s2["rationale"],
                    s2["citation"],
                    s2["choice_code"],
                    {c: tariff.cn8_text[c] for c in kids},
                )
            )
        for name, rat, cite, chosen, texts in stages:
            if cite and cite["pass"]:
                continue
            n_fail[name] += 1
            if not checks.quotes(rat):
                no_quote[name] += 1
            cat = classify_failure(
                rat,
                texts[chosen],
                [t for k, t in texts.items() if k != chosen],
                desc,
                d.get("evidence") if name == "stage1" else [],
            )
            counts[name][cat] += 1
            if len(examples[name][cat]) < 2:
                examples[name][cat].append(
                    {
                        "item_id": r.item_id,
                        "rationale": rat[:300],
                        "chosen_text": texts[chosen][:200],
                    }
                )
    out = {
        "run_id": run_id,
        "n_items": len(frame),
        "failures": n_fail,
        "counts": counts,
        "no_quote_in_other": no_quote,
        "examples": examples,
    }
    (OUT / "a1_citation_failures_val.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="a2_tune", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template-check")
    t.add_argument("--n", type=int, default=10)
    w = sub.add_parser("write-config")
    w.add_argument("--top-logprobs", type=int, required=True)
    for name in ("fit-temperature", "reverse-order", "report", "a1-citations"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        if name == "reverse-order":
            p.add_argument("--n", type=int, default=100)
        if name == "report":
            p.add_argument("--a1", required=True)
            p.add_argument("--b1ex", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "template-check":
        out = template_check(a.n)
        (OUT / "a2_template_check_val.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    elif a.cmd == "write-config":
        out = write_config(a.top_logprobs)
    elif a.cmd == "fit-temperature":
        out = fit_temperature(a.run)
    elif a.cmd == "reverse-order":
        out = reverse_order(a.run, a.n)
        out = {k: v for k, v in out.items() if k != "rows"}
    elif a.cmd == "report":
        out = report(a.run, a.a1, a.b1ex)
    else:
        out = a1_citations(a.run)
    print(json.dumps(out, indent=2, default=str)[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
