"""A3: review policies from A2 confidence and A2/B1-ex agreement, on rulings validation only.

    uv run python -m hs_rag_classifier.analysis.a3_agreement [--a2-pass PATH] [--out DIR]

Validation only: a test split is refused by the harness test guard (eval/runner.py
check_test_guard, with this analysis as a sanity arm) before anything is loaded, and the A2
pass must be the rulings val split (same split SHA-256, exactly the 750 frozen ids). No
llama-server, no MLflow run, nothing added to test_runs.jsonl.

Inputs:
    A2       a saved A2 val pass (eval/repro_val750/passes/a2_val_1.json): top-1 cn8, top 3,
             joint confidence at the locked temperatures (checked against arms.a2_pipeline
             rescore), stage probabilities p1 and p2.
    B1-ex    the B1-ex arm run here with its locked config (k = 1; its index is built from the
             train split only, guards.check_train_split_only); top-1 code and the top
             neighbour's similarity (with k = 1 the vote is that similarity).
    TF-IDF   cosine of each val description to its nearest train ruling: the split's
             vectoriser (char_wb 3-5, sublinear tf, data.rulings.normalise) fitted on train
             and val descriptions (the split also fitted on test; test is never read here).
Both arms are scored with eval.metrics.score (invalid codes wrong; correctness at 8 digits).

Policies order items for review; the rest are auto-accepted. Ties go by item id (numeric
order, lower first) in every policy.
    a  A2 confidence, lowest first.
    b  every cn8 disagreement with B1-ex first, then agreements; lowest A2 confidence first
       within each group.
    c  out-of-fold probability that A2's top-1 is right at 8 digits, lowest first: L2 logistic
       regression, C = 1.0 (not tuned), standardised features (logit of A2 confidence, cn8
       agreement, HS6 agreement, B1-ex top similarity, B1-ex cn8 in A2's top 3 but not top 1),
       5-fold GroupKFold by near-duplicate cluster. A fit on all items is saved, marked not
       frozen and not used anywhere.

Metrics: accuracy of auto-accepted items when round(rate x n) items are reviewed (the
harness rule, floor(rate x n + 0.5)); the coverage-accuracy curve; AUROC of the review
position for A2 correct; area under the risk-coverage curve (mean selective risk over the n
coverages 1, (n-1)/n, ..., 1/n); the review rate needed for 80% and 90% accuracy. Paired
differences: cluster bootstrap (eval.metrics.bootstrap_samples, its seed), each resample
re-ranks the drawn rows; policy c's out-of-fold scores stay fixed within a resample.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.items import DATA_ROOT, sha256_file

OUT = DATA_ROOT / "eval" / "a3_agreement"
A2_PASS = DATA_ROOT / "eval" / "repro_val750" / "passes" / "a2_val_1.json"
OLD_A2_RUN = "2670780f5ce641f1b95c4f471aceb343"
SAVED_B1EX_RUN = "82b226c0904a4092906fba3e114dce79"
ARTIFACTS = DATA_ROOT / "mlflow" / "artifacts" / "eval-rulings"
REVIEW_RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
TARGETS = (0.80, 0.90)
POLICIES = ("a", "b", "c")
PAIRS = (("b", "a"), ("c", "a"), ("c", "b"))
FEATURES = ("logit_conf", "agree_cn8", "agree_hs6", "b1ex_sim", "b1ex_in_top3_not_top1")
N_FOLDS = 5
C = 1.0
N_BINS = 5
MIN_CHAPTER = 15
A_TO_Z_B1EX_TOP1 = 0.107
ANALYSIS = SimpleNamespace(name="a3_agreement", sanity=True)  # never on a test split


def check_split(split: str, git: dict | None = None) -> None:
    """Refuse anything but validation, through the harness test guard."""
    R.check_test_guard(ANALYSIS, SimpleNamespace(split=split), git or {"dirty": False}, None)
    if split != "val":
        raise R.GuardError(f"A3 runs on rulings validation only, not {split!r}")


# --- ranking -----------------------------------------------------------------------------


def id_rank(item_ids) -> np.ndarray:
    """Each id's position in numeric order (length, then text: numeric for digit strings)."""
    ids = list(item_ids)
    order = sorted(range(len(ids)), key=lambda i: (len(ids[i]), ids[i]))
    rank = np.empty(len(ids), int)
    rank[order] = np.arange(len(ids))
    return rank


def review_order(policy: str, a: dict[str, np.ndarray]) -> np.ndarray:
    """Row indices in review order (first reviewed first); ties by id_rank."""
    if policy == "a":
        keys = (a["id_rank"], a["confidence"])
    elif policy == "b":
        keys = (a["id_rank"], a["confidence"], a["agree_cn8"].astype(int))
    elif policy == "c":
        keys = (a["id_rank"], a["oof"])
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return np.lexsort(keys)  # last key is the primary one


def n_reviewed(rate: float, n: int) -> int:
    return min(n, math.floor(rate * n + 0.5))


def accepted_accuracy(order: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """Accuracy of the auto-accepted items when the first k are reviewed, k = 0 .. n-1."""
    c = np.asarray(correct, float)[order]
    tail = np.cumsum(c[::-1])[::-1]
    return tail / np.arange(len(c), 0, -1)


def accuracy_at(order: np.ndarray, correct: np.ndarray, rate: float) -> float:
    k = n_reviewed(rate, len(order))
    accepted = np.asarray(correct, float)[order[k:]]
    return float(accepted.mean()) if len(accepted) else math.nan


def auroc(order: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of the review position (later = more trusted) for correct; NaN with one class."""
    from scipy.stats import rankdata

    y = np.asarray(correct, float)
    pos = np.empty(len(order))
    pos[order] = np.arange(len(order))
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = rankdata(pos)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aurc(order: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean(1.0 - accepted_accuracy(order, correct)))


def rate_for(order: np.ndarray, correct: np.ndarray, target: float) -> float | None:
    """Smallest review rate whose auto-accepted accuracy reaches target; None if none does."""
    acc = accepted_accuracy(order, correct)
    hit = np.nonzero(acc >= target - 1e-12)[0]
    return float(hit[0] / len(order)) if len(hit) else None


def policy_metrics(order: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    acc = accepted_accuracy(order, correct)
    return {
        "accuracy_at_review": {f"{r:.2f}": accuracy_at(order, correct, r) for r in REVIEW_RATES},
        "auroc": auroc(order, correct),
        "aurc": aurc(order, correct),
        "review_rate_for": {f"{t:.2f}": rate_for(order, correct, t) for t in TARGETS},
        "curve": {
            "review_rate": [k / len(order) for k in range(len(order))],
            "accuracy": [float(x) for x in acc],
        },
    }


# --- policy c ----------------------------------------------------------------------------


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    conf = frame["a2_confidence"].to_numpy(float).clip(1e-6, 1 - 1e-6)
    cols = {
        "logit_conf": np.log(conf / (1 - conf)),
        "agree_cn8": frame["agree_cn8"].to_numpy(float),
        "agree_hs6": frame["agree_hs6"].to_numpy(float),
        "b1ex_sim": frame["b1ex_sim"].to_numpy(float),
        "b1ex_in_top3_not_top1": frame["b1ex_in_top3_not_top1"].to_numpy(float),
    }
    return np.column_stack([cols[f] for f in FEATURES])


def _model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=C, l1_ratio=0, max_iter=1000))


def folds(groups, n_splits: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def _coefs(model) -> dict[str, float]:
    lr = model[-1]
    return {
        "intercept": float(lr.intercept_[0]),
        **{f: float(c) for f, c in zip(FEATURES, lr.coef_[0], strict=True)},
    }


def oof_scores(x: np.ndarray, y: np.ndarray, groups) -> tuple[np.ndarray, list[dict]]:
    """Out-of-fold P(correct) and each fold's standardised coefficients."""
    oof = np.full(len(y), np.nan)
    per_fold = []
    for f, (train, test) in enumerate(folds(groups)):
        model = _model().fit(x[train], y[train])
        oof[test] = model.predict_proba(x[test])[:, 1]
        per_fold.append({"fold": f, "n_train": len(train), "n_test": len(test), **_coefs(model)})
    return oof, per_fold


def full_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    model = _model().fit(x, y)
    scaler = model[0]
    return {
        "status": "not frozen, not used anywhere",
        "note": "fitted on all 750 rulings validation items, for reference only",
        "features": list(FEATURES),
        "C": C,
        "penalty": "l2",
        "coefficients_standardised": _coefs(model),
        "scaler_mean": dict(zip(FEATURES, map(float, scaler.mean_), strict=True)),
        "scaler_scale": dict(zip(FEATURES, map(float, scaler.scale_), strict=True)),
        "n": len(y),
    }


# --- bootstrap ---------------------------------------------------------------------------


def paired_differences(
    frame: pd.DataFrame, n_resamples: int = M.BOOTSTRAP_RESAMPLES, seed: int = M.BOOTSTRAP_SEED
) -> dict[str, Any]:
    """b-a, c-a, c-b at each review rate and for AUROC; each resample re-ranks its rows."""
    cols = ["cluster_id", "correct", "confidence", "agree_cn8", "oof", "id_rank"]
    boot = frame[cols].reset_index(drop=True)

    def stat(kind: str, p: str, q: str, rate: float | None = None):
        def fn(a: dict) -> float:
            vals = []
            for pol in (p, q):
                order = review_order(pol, a)
                vals.append(
                    auroc(order, a["correct"])
                    if kind == "auroc"
                    else accuracy_at(order, a["correct"], rate)
                )
            return vals[0] - vals[1]

        return fn

    stats = {}
    for p, q in PAIRS:
        for r in REVIEW_RATES:
            stats[f"{p}-{q}@{r:.2f}"] = stat("acc", p, q, r)
        stats[f"{p}-{q}@auroc"] = stat("auroc", p, q)
    samples = M.bootstrap_samples(boot, stats, n_resamples, seed)
    full = {c: boot[c].to_numpy() for c in cols}
    out = {}
    for name, fn in stats.items():
        d = samples[name]
        out[name] = {
            "diff": fn(full),
            "low": float(np.nanpercentile(d, 2.5)),
            "high": float(np.nanpercentile(d, 97.5)),
            "share_le_0": float((d <= 0).mean()),
        }
    return {
        "resamples": n_resamples,
        "seed": seed,
        "unit": "cluster_id",
        "n_clusters": int(boot["cluster_id"].nunique()),
        "differences": out,
    }


# --- inputs ------------------------------------------------------------------------------


def a2_rows(passed: dict, valid8: set[str], valid10: set[str], items) -> pd.DataFrame:
    """The A2 pass as a scored frame plus top 3, stage probabilities and the rescore check."""
    from hs_rag_classifier.arms import a2_pipeline as A2
    from hs_rag_classifier.eval.arms import Candidate, Prediction

    t = json.loads(A2.CONFIG_PATH.read_text(encoding="utf-8"))["temperature"]
    preds, extra = [], []
    for r in passed["rows"]:
        d = r["details"]
        _, conf = A2.rescore(d, t["stage1"], t["stage2"])
        top = d["ranked"][0]
        extra.append(
            {
                "item_id": r["item_id"],
                "a2_top3": [c[:8] for c in r["candidates"][:3]],
                "a2_p1": top["p1"],
                "a2_p2": top["p2"],
                "rescore_diff": abs(conf - r["confidence"]),
            }
        )
        preds.append(
            Prediction(
                item_id=r["item_id"],
                confidence=r["confidence"],
                candidates=[
                    Candidate(code=c, score=s)
                    for c, s in zip(r["candidates"], r["scores"], strict=True)
                ],
            )
        )
    frame = M.score(items, preds, valid8, valid10, max_depth=8)
    return frame.merge(pd.DataFrame(extra), on="item_id"), t


def b1ex_frame(items, valid8: set[str], valid10: set[str]) -> tuple[pd.DataFrame, dict]:
    from hs_rag_classifier.arms import b1ex_embed as BX

    config = BX.load_config()
    if int(config["k"]) != 1:
        raise RuntimeError(f"B1-ex locked config has k = {config['k']}, expected 1")
    arm = BX.B1exEmbedArm(config)  # build_index enforces guards.check_train_split_only
    preds, _, _ = R.predict_all(arm, items)
    frame = M.score(items, preds, valid8, valid10)
    frame["b1ex_sim"] = [p.candidates[0].score for p in preds]  # k = 1: the top similarity
    return frame, config


def nearest_train_tfidf(val_texts: list[str], root: Path = DATA_ROOT) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    from hs_rag_classifier.arms import b1ex_embed as BX
    from hs_rag_classifier.data import rulings as rulings_data
    from hs_rag_classifier.guards import check_train_split_only

    train_path = check_train_split_only(root / BX.TRAIN_REL, root / BX.TRAIN_REL)
    train = pd.read_parquet(train_path, columns=["description"])["description"]
    tr = [rulings_data.normalise(t) for t in train]
    va = [rulings_data.normalise(t) for t in val_texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    x = vec.fit_transform(tr + va)
    xt, xv = x[: len(tr)], x[len(tr) :]
    return np.concatenate(
        [(xv[s : s + 250] @ xt.T).max(axis=1).toarray().ravel() for s in range(0, xv.shape[0], 250)]
    )


def combine(a2: pd.DataFrame, b1: pd.DataFrame, tfidf: np.ndarray) -> pd.DataFrame:
    b = b1.set_index("item_id")
    f = pd.DataFrame(
        {
            "item_id": a2["item_id"],
            "chapter": a2["chapter"],
            "cluster_id": a2["cluster_id"],
            "gold": a2["label_code"],
            "label_depth": a2["label_depth"],
            "a2_top1": a2["top1"].str[:8],
            "a2_confidence": a2["confidence"].astype(float),
            "a2_top3": a2["a2_top3"],
            "a2_p1": a2["a2_p1"],
            "a2_p2": a2["a2_p2"],
            "b1ex_code": b.loc[a2["item_id"], "top1"].to_numpy(),
            "b1ex_sim": b.loc[a2["item_id"], "b1ex_sim"].to_numpy(float),
            "tfidf_nearest_train": tfidf,
            "a2_correct": a2["correct_8"].astype(float),
            "b1ex_correct": b.loc[a2["item_id"], "correct_8"].to_numpy(float),
        }
    )
    f["b1ex_cn8"] = f["b1ex_code"].str[:8]
    f["agree_cn8"] = f["a2_top1"] == f["b1ex_cn8"]
    f["agree_hs6"] = f["a2_top1"].str[:6] == f["b1ex_cn8"].str[:6]
    f["b1ex_in_top3_not_top1"] = [
        c in t[:3] and c != t[0] for c, t in zip(f["b1ex_cn8"], f["a2_top3"], strict=True)
    ]
    f["id_rank"] = id_rank(f["item_id"])
    return f


def arrays(f: pd.DataFrame, confidence: str = "a2_confidence") -> dict[str, np.ndarray]:
    return {
        "id_rank": f["id_rank"].to_numpy(),
        "confidence": f[confidence].to_numpy(float),
        "agree_cn8": f["agree_cn8"].to_numpy(bool),
        "correct": f["a2_correct"].to_numpy(float),
        "oof": f["oof"].to_numpy(float) if "oof" in f else np.zeros(len(f)),
    }


# --- breakdowns --------------------------------------------------------------------------


def _mean(x) -> float | None:
    x = np.asarray(x, float)
    return float(x.mean()) if len(x) else None


def agreement_stats(f: pd.DataFrame) -> dict[str, Any]:
    ag = f["agree_cn8"].to_numpy(bool)
    c = f["a2_correct"].to_numpy(float)
    return {
        "n": len(f),
        "agree_rate": _mean(ag),
        "n_agree": int(ag.sum()),
        "a2_acc_agree": _mean(c[ag]),
        "a2_acc_disagree": _mean(c[~ag]),
        "a2_acc": _mean(c),
    }


def binned(f: pd.DataFrame, col: str, n_bins: int = N_BINS) -> list[dict[str, Any]]:
    bins = pd.qcut(f[col], n_bins, duplicates="drop")
    out = []
    for interval, g in f.groupby(bins, observed=True, sort=True):
        out.append({"bin": f"{interval.left:.3f} to {interval.right:.3f}", **agreement_stats(g)})
    return out


def arbitration(f: pd.DataFrame) -> dict[str, Any]:
    d = f[~f["agree_cn8"]]
    a2r, b1r = d["a2_correct"] == 1, d["b1ex_correct"] == 1
    by_sim = []
    for interval, g in d.groupby(
        pd.qcut(d["b1ex_sim"], N_BINS, duplicates="drop"), observed=True, sort=True
    ):
        by_sim.append(
            {
                "bin": f"{interval.left:.3f} to {interval.right:.3f}",
                "n": len(g),
                "b1ex_acc": _mean(g["b1ex_correct"]),
                "a2_acc": _mean(g["a2_correct"]),
            }
        )
    return {
        "n_disagree": len(d),
        "a2_right": int(a2r.sum()),
        "b1ex_right": int(b1r.sum()),
        "both_wrong": int((~a2r & ~b1r).sum()),
        "both_right": int((a2r & b1r).sum()),
        "b1ex_in_a2_top3": int(d["b1ex_in_top3_not_top1"].sum()),
        "b1ex_right_and_in_a2_top3": int((b1r & d["b1ex_in_top3_not_top1"]).sum()),
        "b1ex_acc": _mean(d["b1ex_correct"]),
        "a2_acc": _mean(d["a2_correct"]),
        "b1ex_acc_by_similarity_quintile": by_sim,
    }


def confidence_overlap(f: pd.DataFrame) -> dict[str, Any]:
    med = float(f["a2_confidence"].median())
    hi = f["a2_confidence"] > med
    cells = {}
    for ag_name, ag in (("agree", f["agree_cn8"]), ("disagree", ~f["agree_cn8"])):
        for c_name, c in (("above_median", hi), ("at_or_below_median", ~hi)):
            g = f[ag & c]
            cells[f"{ag_name}/{c_name}"] = {"n": len(g), "a2_acc": _mean(g["a2_correct"])}
    r = float(np.corrcoef(f["agree_cn8"].astype(float), f["a2_confidence"])[0, 1])
    return {"median_confidence": med, "cells": cells, "point_biserial_r": r}


def by_chapter(f: pd.DataFrame, rate: float = 0.20) -> list[dict[str, Any]]:
    """Per chapter (>= MIN_CHAPTER items, others pooled): agreement, and the auto-accepted
    accuracy in that chapter when the global queue reviews `rate` of all items."""
    a = arrays(f)
    reviewed = {}
    for p in ("a", "b"):
        mask = np.zeros(len(f), bool)
        mask[review_order(p, a)[: n_reviewed(rate, len(f))]] = True
        reviewed[p] = mask
    counts = f["chapter"].value_counts()
    group = f["chapter"].where(f["chapter"].map(counts) >= MIN_CHAPTER, "other")
    out = []
    for ch in [*sorted(c for c in group.unique() if c != "other"), "other"]:
        idx = np.nonzero((group == ch).to_numpy())[0]
        g = f.iloc[idx]
        row = {"chapter": ch, **agreement_stats(g)}
        for p in ("a", "b"):
            keep = idx[~reviewed[p][idx]]
            row[f"reviewed_share_{p}"] = float(reviewed[p][idx].mean())
            row[f"acc_at_20_{p}"] = _mean(f["a2_correct"].to_numpy(float)[keep])
        out.append(row)
    return out


def old_run(f: pd.DataFrame, temps: dict, run_id: str = OLD_A2_RUN) -> dict[str, Any] | None:
    """Policies a and b at 20% and 50% on the older saved A2 val run (rescored at locked T)."""
    from hs_rag_classifier.pipeline.a2_tune import rescored_frame

    path = ARTIFACTS / run_id / "artifacts" / "predictions.parquet"
    if not path.exists():
        return None
    old = rescored_frame(pd.read_parquet(path), temps["stage1"], temps["stage2"])
    old = old.set_index("item_id").loc[f["item_id"]].reset_index()
    g = f.copy()
    g["a2_top1"] = old["top1"].str[:8].to_numpy()
    g["a2_confidence"] = old["confidence"].to_numpy(float)
    g["a2_correct"] = old["correct_8"].to_numpy(float)
    g["agree_cn8"] = g["a2_top1"] == g["b1ex_cn8"]
    new_a, old_a = arrays(f), arrays(g)
    rates = (0.20, 0.50)
    res = {
        "run_id": run_id,
        "path": str(path),
        "sha256": sha256_file(path),
        "top1_changes": int((g["a2_top1"] != f["a2_top1"]).sum()),
        "agreement_old": agreement_stats(g),
        "policies": {},
    }
    for p in ("a", "b"):
        for r in rates:
            new = accuracy_at(review_order(p, new_a), new_a["correct"], r)
            was = accuracy_at(review_order(p, old_a), old_a["correct"], r)
            res["policies"][f"{p}@{r:.2f}"] = {"old": was, "new": new, "change": new - was}
    return res


def saved_b1ex_check(f: pd.DataFrame, run_id: str = SAVED_B1EX_RUN) -> dict[str, Any] | None:
    path = ARTIFACTS / run_id / "artifacts" / "predictions.parquet"
    if not path.exists():
        return None
    saved = pd.read_parquet(path, columns=["item_id", "top1"]).set_index("item_id")["top1"]
    same = saved.loc[f["item_id"]].to_numpy() == f["b1ex_code"].to_numpy()
    return {"run_id": run_id, "n_same_top1": int(same.sum()), "n": len(f)}


# --- outputs -----------------------------------------------------------------------------


def plot(curves: dict[str, dict], path: Path, n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from hs_rag_classifier.eval import plots as P

    # (b) and (c) overlap almost everywhere, so identity is the legend plus a line style each
    fig, ax = P._axes(
        f"A2 auto-accept accuracy by review policy\nrulings validation, n = {n}",
        "Review rate",
        "Accuracy of auto-accepted items (8 digits)",
    )
    names = {"a": "(a) A2 confidence", "b": "(b) disagreement first", "c": "(c) combined, OOF"}
    styles = {"a": ("#2a78d6", "-"), "b": ("#eb6834", "--"), "c": ("#1baf7a", "-.")}
    stop = int(0.60 * n) + 1
    for p in POLICIES:
        x = np.array(curves[p]["review_rate"][:stop])
        y = np.array(curves[p]["accuracy"][:stop])
        color, ls = styles[p]
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2, label=names[p])
    ax.set_xlim(0, 0.60)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for t in leg.get_texts():
        t.set_color(P.TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=P.SURFACE)
    plt.close(fig)


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _num(x, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _bins_table(rows: list[dict]) -> str:
    return table(
        ["bin", "n", "agree rate", "A2 acc (agree)", "A2 acc (disagree)"],
        [
            [
                r["bin"],
                r["n"],
                _pct(r["agree_rate"]),
                _pct(r["a2_acc_agree"]),
                _pct(r["a2_acc_disagree"]),
            ]
            for r in rows
        ],
    )


def report_md(m: dict) -> str:
    pol = m["policies"]
    names = {"a": "(a) A2 confidence", "b": "(b) disagreement first", "c": "(c) combined (OOF)"}
    s = m["consistency"]
    e = m["earlier_diagnostics"]
    parts = [
        "# A3 agreement analysis (rulings validation only)",
        "",
        (
            f"Git commit {m['git']['commit'][:12]} (dirty: {m['git']['dirty']}). Rulings validation "
            f"v1, {m['n']} items in {m['n_clusters']} near-duplicate clusters. A2 outputs: "
            f"`{Path(m['inputs']['a2_pass']['path']).name}` (SHA-256 "
            f"`{m['inputs']['a2_pass']['sha256'][:16]}...`), the bit-identical pass under -np 1 and "
            f"cache_prompt=false; joint confidence at the locked temperatures "
            f"{m['inputs']['temperatures']['stage1']} and {m['inputs']['temperatures']['stage2']} "
            f"(rescore check: max difference {m['inputs']['a2_pass']['rescore_max_diff']:.1e}). "
            f"B1-ex run here with its locked config (k = {m['b1ex_config']['k']}, train-split "
            "index only). No test split or A to Z data was read; no MLflow run was created."
        ),
        "",
        (
            f"Every validation item is its own near-duplicate cluster ({m['n_clusters']} clusters), "
            "so the cluster bootstrap and GroupKFold here behave exactly like item-level ones."
        ),
        "",
        "## Consistency check",
        "",
        table(
            ["", "agreement", "A2 acc when agreeing", "A2 acc when disagreeing"],
            [
                [
                    "earlier diagnostics (older saved A2 run)",
                    _pct(e["agree_rate"]),
                    _pct(e["a2_acc_agree"]),
                    _pct(e["a2_acc_disagree"]),
                ],
                [
                    "this pass",
                    _pct(s["agree_rate"]),
                    _pct(s["a2_acc_agree"]),
                    _pct(s["a2_acc_disagree"]),
                ],
            ],
        ),
        "",
        (
            f"A2 top-1 accuracy at 8 digits: {_pct(s['a2_acc'])}. B1-ex top-1 at 8 digits: "
            f"{_pct(m['b1ex_top1_8'])}."
        ),
        "",
        "## Policies",
        "",
        (
            "Accuracy of auto-accepted items at each review rate (reviewed = round(rate x 750)), "
            "AUROC of the review order for A2 correct, area under the risk-coverage curve (lower "
            "is better), and the review rate needed to reach 80% and 90%."
        ),
        "",
        table(
            [
                "policy",
                *[f"{int(r * 100)}%" for r in REVIEW_RATES],
                "AUROC",
                "AURC",
                "rate for 80%",
                "rate for 90%",
            ],
            [
                [
                    names[p],
                    *[_pct(pol[p]["accuracy_at_review"][f"{r:.2f}"]) for r in REVIEW_RATES],
                    _num(pol[p]["auroc"]),
                    _num(pol[p]["aurc"]),
                    _pct(pol[p]["review_rate_for"]["0.80"]),
                    _pct(pol[p]["review_rate_for"]["0.90"]),
                ]
                for p in POLICIES
            ],
        ),
        "",
        "![coverage-accuracy](coverage_accuracy.png)",
        "",
        "## Paired differences",
        "",
        (
            f"Cluster bootstrap, {m['paired']['resamples']} resamples (seed {m['paired']['seed']}), "
            "95% percentile intervals; each resample re-ranks its rows under both policies. "
            "Limitation: policy (c) keeps its out-of-fold scores fixed within each resample (the "
            "model is not refitted), so its intervals leave out the variability of fitting it."
        ),
        "",
        table(
            ["difference", *[f"{int(r * 100)}%" for r in REVIEW_RATES], "AUROC"],
            [
                [
                    f"{p} - {q}",
                    *[
                        _diff(m["paired"]["differences"][f"{p}-{q}@{r:.2f}"], True)
                        for r in REVIEW_RATES
                    ],
                    _diff(m["paired"]["differences"][f"{p}-{q}@auroc"], False),
                ]
                for p, q in PAIRS
            ],
        ),
        "",
        "## Policy (c) coefficients",
        "",
        (
            "Standardised coefficients per fold (L2, C = 1.0, not tuned). The fit on all 750 items "
            "is in policy_c_full_fit.json, not frozen and not used anywhere."
        ),
        "",
        table(
            ["fold", "intercept", *FEATURES],
            [
                [c["fold"], _num(c["intercept"]), *[_num(c[f]) for f in FEATURES]]
                for c in m["policy_c"]["per_fold"]
            ]
            + [
                [
                    "sd across folds",
                    _num(m["policy_c"]["fold_sd"]["intercept"]),
                    *[_num(m["policy_c"]["fold_sd"][f]) for f in FEATURES],
                ],
                [
                    "all 750 (reference)",
                    _num(m["policy_c"]["full_fit"]["intercept"]),
                    *[_num(m["policy_c"]["full_fit"][f]) for f in FEATURES],
                ],
            ],
        ),
        "",
        "## Server-state sensitivity",
        "",
        _old_run_md(m["old_run"]),
        "",
        "## B1-ex arbitration (report only)",
        "",
        _arb_md(m["arbitration"]),
        "",
        "## Risk checks",
        "",
        "### Near-duplicate inflation",
        "",
        "By B1-ex top similarity (quintiles over all items):",
        "",
        _bins_table(m["risk"]["by_b1ex_similarity"]),
        "",
        (
            "By TF-IDF cosine to the nearest train ruling (char_wb 3-5, sublinear tf, as in the "
            "split; fitted on train and validation descriptions, since the split's own fit also "
            "read test):"
        ),
        "",
        _bins_table(m["risk"]["by_tfidf_nearest_train"]),
        "",
        m["risk"]["near_duplicate_finding"],
        "",
        "### Confidence overlap",
        "",
        _overlap_md(m["risk"]["confidence_overlap"]),
        "",
        "### By chapter",
        "",
        (
            "Chapters with at least 15 validation items; the rest pooled as other. Accuracy at 20% "
            "review is for the chapter's auto-accepted items when the global queue reviews 20% of "
            "all 750, with the share of the chapter that queue reviews."
        ),
        "",
        table(
            [
                "chapter",
                "n",
                "agree rate",
                "A2 acc (agree)",
                "A2 acc (disagree)",
                "(a) acc @20%",
                "(a) reviewed",
                "(b) acc @20%",
                "(b) reviewed",
            ],
            [
                [
                    r["chapter"],
                    r["n"],
                    _pct(r["agree_rate"]),
                    _pct(r["a2_acc_agree"]),
                    _pct(r["a2_acc_disagree"]),
                    _pct(r["acc_at_20_a"]),
                    _pct(r["reviewed_share_a"]),
                    _pct(r["acc_at_20_b"]),
                    _pct(r["reviewed_share_b"]),
                ]
                for r in m["risk"]["by_chapter"]
            ],
        ),
        "",
        "### A to Z",
        "",
        (
            f"A to Z was not loaded. B1-ex scored {A_TO_Z_B1EX_TOP1} top-1 on the A to Z test set, so "
            "agreement with A2 will be rare on short queries and policy (b) would send most of them "
            "to review. That behaviour is untested until final test time, with a policy frozen in "
            "advance."
        ),
        "",
    ]
    return "\n".join(parts)


def _diff(d: dict, pct: bool) -> str:
    if pct:
        return f"{100 * d['diff']:+.1f} [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}]"
    return f"{d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"


def _old_run_md(o: dict | None) -> str:
    if o is None:
        return "The older saved A2 val outputs (run 2670780f) were not found; step skipped."
    rows = [
        [
            k.replace("@", " at ").replace("0.20", "20%").replace("0.50", "50%"),
            _pct(v["old"]),
            _pct(v["new"]),
            f"{100 * v['change']:+.1f}",
        ]
        for k, v in o["policies"].items()
    ]
    ag = o["agreement_old"]
    return (
        f"Older saved run {o['run_id'][:8]} rescored at the locked temperatures; its top-1 "
        f"cn8 differs from this pass on {o['top1_changes']} of 750 items. With the same "
        f"B1-ex predictions it agrees on {_pct(ag['agree_rate'])} (A2 accuracy "
        f"{_pct(ag['a2_acc_agree'])} agreeing, {_pct(ag['a2_acc_disagree'])} disagreeing)."
        "\n\n" + table(["policy at review", "older run", "this pass", "change (points)"], rows)
    )


def _arb_md(a: dict) -> str:
    head = (
        f"On the {a['n_disagree']} cn8 disagreements: A2 right {a['a2_right']}, B1-ex right "
        f"{a['b1ex_right']}, both wrong {a['both_wrong']} (both right is impossible at cn8). "
        f"B1-ex's cn8 is in A2's top 3 on {a['b1ex_in_a2_top3']} of them "
        f"({a['b1ex_right_and_in_a2_top3']} of those with B1-ex right). Accuracy on "
        f"disagreements: A2 {_pct(a['a2_acc'])}, B1-ex {_pct(a['b1ex_acc'])}.\n\n"
        "B1-ex accuracy on disagreements by its top similarity (quintiles of the "
        "disagreements):\n\n"
    )
    return head + table(
        ["similarity", "n", "B1-ex acc", "A2 acc"],
        [
            [r["bin"], r["n"], _pct(r["b1ex_acc"]), _pct(r["a2_acc"])]
            for r in a["b1ex_acc_by_similarity_quintile"]
        ],
    )


def _overlap_md(o: dict) -> str:
    c = o["cells"]
    rows = [
        [
            ag,
            *[
                f"n {c[f'{ag}/{k}']['n']}, acc {_pct(c[f'{ag}/{k}']['a2_acc'])}"
                for k in ("above_median", "at_or_below_median")
            ],
        ]
        for ag in ("agree", "disagree")
    ]
    return (
        table(["", f"confidence > median ({o['median_confidence']:.3f})", "at or below"], rows)
        + f"\n\nPoint-biserial correlation between agreement and A2 confidence: "
        f"r = {o['point_biserial_r']:.3f}."
    )


def near_duplicate_finding(sim_bins: list[dict], tfidf_bins: list[dict]) -> str:
    """A plain statement: is the agreement signal concentrated in items with close neighbours?

    Concentrated: in both binnings the top bin's agreement rate is at least twice the bottom
    bin's. Not only near-duplicates: agreeing items beat disagreeing ones by at least 20 points
    in every bin of both binnings."""
    out, concentrated, gaps = [], True, []
    for name, bins in (("B1-ex similarity", sim_bins), ("TF-IDF nearest-train cosine", tfidf_bins)):
        lo, hi = bins[0], bins[-1]
        gap = [(b["a2_acc_agree"] or 0) - (b["a2_acc_disagree"] or 0) for b in bins]
        gaps += gap
        concentrated &= hi["agree_rate"] >= 2 * lo["agree_rate"]
        top2 = sum(b["n_agree"] for b in bins[-2:]) / sum(b["n_agree"] for b in bins)
        out.append(
            f"By {name}, agreement rises from {_pct(lo['agree_rate'])} in the lowest bin "
            f"to {_pct(hi['agree_rate'])} in the highest ({_pct(top2)} of all agreements are in "
            f"the top two bins), and the agree-minus-disagree accuracy gap runs "
            f"{', '.join(f'{100 * g:+.0f}' for g in gap)} points from lowest to highest."
        )
    verdict = (
        "Yes: the agreement signal is concentrated in items with close training neighbours."
        if concentrated
        else "No: agreement is not concentrated in items with close training neighbours."
    )
    if min(gaps) >= 0.20:
        verdict += (
            f" It is not only a near-duplicate effect: agreeing items are more accurate than "
            f"disagreeing ones in every bin, by at least {100 * min(gaps):.0f} points, so the "
            "signal holds for items far from any training ruling, though few of them agree."
        )
    else:
        verdict += (
            f" In at least one bin the accuracy gap is under 20 points "
            f"(smallest {100 * min(gaps):+.0f}), so the signal weakens away from close neighbours."
        )
    return verdict + "\n\n" + " ".join(out)


# --- main --------------------------------------------------------------------------------


def run(
    a2_pass: Path = A2_PASS,
    out: Path = OUT,
    n_resamples: int = M.BOOTSTRAP_RESAMPLES,
    split: str = "val",
) -> dict:
    git = R.git_info()
    check_split(split, git)
    from hs_rag_classifier.eval.items import load

    evalset = load("rulings", "val")
    passed = json.loads(a2_pass.read_text(encoding="utf-8"))
    if passed.get("split") != "val" or passed.get("split_sha256") != evalset.split_sha256:
        raise R.GuardError(f"{a2_pass} is not a rulings validation pass")
    ids = [r["item_id"] for r in passed["rows"]]
    if len(ids) != len(set(ids)) or set(ids) != {i.item_id for i in evalset.items}:
        raise R.GuardError(f"{a2_pass} item ids do not match the {len(evalset.items)} val ids")
    items = evalset.items
    valid8, valid10 = R.valid_codes()

    a2, temps = a2_rows(passed, valid8, valid10, items)
    b1, b1_config = b1ex_frame(items, valid8, valid10)
    tfidf = nearest_train_tfidf([i.text for i in items])
    f = combine(a2, b1, tfidf)

    y = f["a2_correct"].to_numpy(float)
    x = feature_matrix(f)
    oof, per_fold = oof_scores(x, y, f["cluster_id"])
    f["oof"] = oof
    full = full_fit(x, y)
    a = arrays(f)
    pols = {p: policy_metrics(review_order(p, a), a["correct"]) for p in POLICIES}
    boot = f.assign(correct=a["correct"], confidence=a["confidence"])
    paired = paired_differences(boot, n_resamples)
    fold_df = pd.DataFrame(per_fold)
    sim_bins, tfidf_bins = binned(f, "b1ex_sim"), binned(f, "tfidf_nearest_train")
    report_earlier = json.loads((DATA_ROOT / "eval" / "a2_val_report.json").read_text())
    earlier = report_earlier["agreement_a2_b1ex"]

    m = {
        "n": len(f),
        "n_clusters": int(f["cluster_id"].nunique()),
        "git": git,
        "inputs": {
            "a2_pass": {
                "path": str(a2_pass),
                "sha256": sha256_file(a2_pass),
                "n": len(ids),
                "name": passed.get("name"),
                "total_slots": passed.get("total_slots"),
                "build": passed.get("props"),
                "rescore_max_diff": float(a2["rescore_diff"].max()),
            },
            "val_split": {"file": evalset.split_file, "sha256": evalset.split_sha256},
            "temperatures": {"stage1": temps["stage1"], "stage2": temps["stage2"]},
            "saved_b1ex_check": saved_b1ex_check(f),
            "earlier_report": str(DATA_ROOT / "eval" / "a2_val_report.json"),
        },
        "b1ex_config": b1_config,
        "b1ex_top1_8": float(f["b1ex_correct"].mean()),
        "consistency": agreement_stats(f),
        "earlier_diagnostics": {
            "agree_rate": earlier["agree_share"],
            "a2_acc_agree": earlier["accuracy_agree"],
            "a2_acc_disagree": earlier["accuracy_disagree"],
        },
        "policies": pols,
        "paired": paired,
        "policy_c": {
            "features": list(FEATURES),
            "C": C,
            "folds": N_FOLDS,
            "per_fold": per_fold,
            "fold_mean": fold_df[["intercept", *FEATURES]].mean().to_dict(),
            "fold_sd": fold_df[["intercept", *FEATURES]].std(ddof=1).to_dict(),
            "full_fit": full["coefficients_standardised"],
        },
        "old_run": old_run(f, temps),
        "arbitration": arbitration(f),
        "risk": {
            "by_b1ex_similarity": sim_bins,
            "by_tfidf_nearest_train": tfidf_bins,
            "near_duplicate_finding": near_duplicate_finding(sim_bins, tfidf_bins),
            "confidence_overlap": confidence_overlap(f),
            "by_chapter": by_chapter(f),
            "a_to_z_note": {"b1ex_top1_a_to_z_test": A_TO_Z_B1EX_TOP1, "loaded": False},
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    cols = [
        "item_id",
        "chapter",
        "cluster_id",
        "gold",
        "label_depth",
        "a2_top1",
        "a2_confidence",
        "b1ex_cn8",
        "b1ex_sim",
        "tfidf_nearest_train",
        "agree_cn8",
        "agree_hs6",
        "oof",
        "a2_correct",
        "b1ex_correct",
    ]
    f[cols].rename(columns={"oof": "policy_c_oof"}).to_csv(out / "per_item.csv", index=False)
    (out / "policy_c_full_fit.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    plot({p: pols[p]["curve"] for p in POLICIES}, out / "coverage_accuracy.png", len(f))
    (out / "report.md").write_text(report_md(m), encoding="utf-8")
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="a3_agreement", description=__doc__.splitlines()[0])
    ap.add_argument(
        "--split", choices=("val", "test"), default="val", help="val only; test is refused"
    )
    ap.add_argument("--a2-pass", type=Path, default=A2_PASS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--resamples", type=int, default=M.BOOTSTRAP_RESAMPLES)
    a = ap.parse_args(argv)
    try:
        m = run(a.a2_pass, a.out, a.resamples, a.split)
    except R.GuardError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"consistency": m["consistency"], "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
