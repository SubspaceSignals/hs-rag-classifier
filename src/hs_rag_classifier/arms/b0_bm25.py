"""B0: BM25 keyword baseline over the tariff path text.

Documents: one per current declarable 10-digit line in the pinned nomenclature (current_lines(),
non-superseded), text = its path_text (chapter down to the line, after the hierarchy repair).
No chapter notes, no rulings text, and no A to Z data. The builder passes its input file list
through guards.check_retrieval_inputs.

Tokenisation (bm25s): lowercase, token pattern (?u)\\b\\w\\w+\\b, bm25s English stopwords,
Snowball English stemming (PyStemmer). Scoring: bm25s "lucene" BM25 with k1 and b.

Ranking: every line is scored; each cn8 takes the max score of its lines; the top 50 cn8s are
returned, each as its best-scoring 10-digit line, so the top-1 can be scored at 10 digits.
Ties go to the lower code. Candidate scores are the raw BM25 scores.

Confidence: softmax over the top-10 cn8 scores at temperature T; confidence = top-1
probability. B0 never abstains; review is handled by the coverage curve.

Index cache: ~/data/hs-rag-classifier/indexes/b0/<index hash>/, keyed on the settings that
change the index (k1, b, tokeniser, documents, nomenclature version and the SHA-256 of
codes.parquet); T does not change the index.

Settings: configs/b0_bm25.json in the repo, loaded by default. Written by
    uv run python -m hs_rag_classifier.arms.b0_bm25 tune
which tunes k1 and b on rulings validation only (grid, top-1 at 8 digits, ties by recall at
10), then fits T on validation by minimising the negative log-likelihood of the top-1-correct
indicator. Every grid point is an MLflow run nested under one parent.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import pandas as pd
import Stemmer

from hs_rag_classifier.data.nomenclature import current_lines, sha256
from hs_rag_classifier.data.rulings import NOMENCLATURE_VERSION
from hs_rag_classifier.eval.arms import Candidate, Prediction, config_hash
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem
from hs_rag_classifier.guards import check_retrieval_inputs

CONFIG_PATH = REPO_ROOT / "configs" / "b0_bm25.json"
INDEX_DIR = Path("indexes") / "b0"
CANDIDATE_CAP = 50
SOFTMAX_TOP = 10
TOKENISER = {
    "library": f"bm25s {bm25s.__version__}",
    "lower": True,
    "token_pattern": r"(?u)\b\w\w+\b",
    "stopwords": "english (bm25s)",
    "stemmer": f"snowball english (PyStemmer {Stemmer.version()})",
}
DOCUMENTS = "current declarable 10-digit lines; text = path_text (chapter down to the line)"
GRID_K1 = (0.9, 1.2, 1.5, 2.0)
GRID_B = (0.3, 0.5, 0.75, 0.9)


def default_config(k1: float = 1.5, b: float = 0.75, temperature: float = 1.0) -> dict[str, Any]:
    return {
        "k1": k1,
        "b": b,
        "method": "lucene",
        "temperature": temperature,
        "tokeniser": TOKENISER,
        "documents": DOCUMENTS,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "candidate_cap": CANDIDATE_CAP,
        "softmax_top": SOFTMAX_TOP,
    }


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tokenise(texts: Sequence[str]) -> list[list[str]]:
    return bm25s.tokenize(
        list(texts),
        lower=TOKENISER["lower"],
        token_pattern=TOKENISER["token_pattern"],
        stopwords="english",
        stemmer=Stemmer.Stemmer("english"),
        return_ids=False,
        show_progress=False,
    )


def documents(codes: pd.DataFrame) -> pd.DataFrame:
    """Current declarable lines sorted by code10 (so each cn8's lines are contiguous)."""
    current = current_lines(codes)
    docs = current[current["declarable"]][["code10", "cn8", "path_text"]]
    return docs.sort_values("code10").reset_index(drop=True)


def index_key(config: dict[str, Any], codes_sha256: str) -> str:
    keys = ("k1", "b", "method", "tokeniser", "documents", "nomenclature_version")
    return config_hash({**{k: config[k] for k in keys}, "codes_sha256": codes_sha256})


class B0Index:
    """A BM25 index over the line documents, with the cn8 grouping it needs."""

    def __init__(self, retriever: bm25s.BM25, code10: Sequence[str]):
        self.retriever = retriever
        self.code10 = np.asarray(code10)
        cn8 = np.array([c[:8] for c in self.code10])
        change = np.r_[True, cn8[1:] != cn8[:-1]]
        self.starts = np.flatnonzero(change)
        self.cn8 = cn8[self.starts]

    def line_scores(self, tokens: list[str]) -> np.ndarray:
        if not tokens:
            return np.zeros(len(self.code10), dtype=np.float32)
        return self.retriever.get_scores(tokens)

    def rank(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Top k cn8s as (best 10-digit line, cn8 score); a cn8 scores the max of its lines."""
        return rank_cn8(scores, self.starts, self.code10, k)


def rank_cn8(
    scores: np.ndarray, starts: np.ndarray, code10: np.ndarray, k: int
) -> list[tuple[str, float]]:
    group_max = np.maximum.reduceat(scores, starts)
    order = np.argsort(-group_max, kind="stable")[:k]  # ties: lower code first
    ends = np.r_[starts[1:], len(scores)]
    out = []
    for g in order:
        lo, hi = starts[g], ends[g]
        best = lo + int(np.argmax(scores[lo:hi]))  # first (lowest code) on ties
        out.append((str(code10[best]), float(group_max[g])))
    return out


def softmax_top1(scores: Sequence[float], temperature: float, top: int = SOFTMAX_TOP) -> float:
    s = np.asarray(scores[:top], dtype=float)
    if len(s) == 0:
        return 0.0
    z = (s - s.max()) / temperature
    p = np.exp(z)
    return float(p[0] / p.sum())


def build_index(
    config: dict[str, Any],
    root: Path = DATA_ROOT,
    codes_path: Path | None = None,
    cache: bool = True,
) -> B0Index:
    """Build (or load the cached) index; the input file list goes through the leakage guard."""
    codes_path = (
        codes_path or root / "processed" / "tariff" / NOMENCLATURE_VERSION / "codes.parquet"
    )
    (codes_path,) = check_retrieval_inputs([codes_path], root)
    key = index_key(config, sha256(codes_path))
    cache_dir = root / INDEX_DIR / key
    if cache and (cache_dir / "code10.json").exists():
        retriever = bm25s.BM25.load(str(cache_dir), show_progress=False)
        code10 = json.loads((cache_dir / "code10.json").read_text(encoding="utf-8"))
        return B0Index(retriever, code10)
    cols = ["code10", "cn8", "path_text", "declarable", "superseded"]
    docs = documents(pd.read_parquet(codes_path, columns=cols))
    retriever = bm25s.BM25(k1=config["k1"], b=config["b"], method=config["method"])
    retriever.index(tokenise(docs["path_text"]), show_progress=False)
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(cache_dir), show_progress=False)
        (cache_dir / "code10.json").write_text(json.dumps(list(docs["code10"])), encoding="utf-8")
        meta = {"config": config, "codes_path": str(codes_path), "n_documents": len(docs)}
        (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return B0Index(retriever, list(docs["code10"]))


class B0BM25Arm:
    """BM25 over tariff path text, grouped to cn8 (rules in the module docstring)."""

    name = "b0"
    batch_size = 64

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 index: B0Index | None = None):  # fmt: skip
        self.config = dict(config or load_config())
        self.index = index or build_index(self.config, root)

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        out = []
        cap, temp = int(self.config["candidate_cap"]), float(self.config["temperature"])
        for item, tokens in zip(items, tokenise([i.text for i in items]), strict=True):
            ranked = self.index.rank(self.index.line_scores(tokens), cap)
            scores = [s for _, s in ranked]
            out.append(
                Prediction(
                    item_id=item.item_id,
                    candidates=[Candidate(code=c, score=s) for c, s in ranked],
                    confidence=softmax_top1(scores, temp, int(self.config["softmax_top"])),
                    abstain=False,
                )
            )
        return out


# --- Tuning on validation -------------------------------------------------------------------


def nll_top1(
    temperature: float, top_scores: Sequence[Sequence[float]], correct: np.ndarray
) -> float:
    p = np.array([softmax_top1(s, temperature) for s in top_scores])
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))


def fit_temperature(top_scores: Sequence[Sequence[float]], correct: np.ndarray) -> float:
    """T minimising the NLL of the top-1-correct indicator (search over log T)."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        lambda log_t: nll_top1(math.exp(log_t), top_scores, correct),
        bounds=(math.log(1e-3), math.log(1e3)),
        method="bounded",
    )
    return float(math.exp(res.x))


def select(grid: list[dict]) -> dict:
    """Best top-1 at 8 digits; ties by recall at 10, then the earlier grid point."""
    return max(enumerate(grid), key=lambda e: (e[1]["top1_8"], e[1]["recall_10"], -e[0]))[1]


def cmd_tune(args: argparse.Namespace) -> int:
    from hs_rag_classifier.eval import metrics as M
    from hs_rag_classifier.eval import runner as R  # sets MLflow env before mlflow loads
    from hs_rag_classifier.eval.items import load

    mlflow = R.mlflow

    evalset = load("rulings", "val", root=args.root)
    valid8, valid10 = R.valid_codes(args.root)
    git = R.git_info()
    mlflow.set_tracking_uri(R.tracking_uri())
    client = mlflow.MlflowClient()
    exp_id = R._experiment_id("eval-rulings", R.MLFLOW_DIR / "artifacts")
    parent = client.create_run(exp_id, run_name="b0-grid-val", tags={"b0_grid": "parent"})
    child = {"mlflow.parentRunId": parent.info.run_id, "b0_grid": "point"}

    grid = []
    for k1, b in itertools.product(GRID_K1, GRID_B):
        arm = B0BM25Arm(default_config(k1, b), root=args.root)
        res = R.run_eval(arm, evalset, valid8, valid10, git=git, tags=child,
                         n_resamples=args.resamples)  # fmt: skip
        m = res["metrics"]
        grid.append({"k1": k1, "b": b, "top1_8": m["top1"][8]["accuracy"],
                     "top1_6": m["top1"][6]["accuracy"], "recall_10": m["recall"][10],
                     "run_id": res["run_id"], "frame": res["frame"]})  # fmt: skip
        print(f"  k1 {k1:<4} b {b:<5} top-1@8 {grid[-1]['top1_8']:.4f}  "
              f"top-1@6 {grid[-1]['top1_6']:.4f}  recall@10 {grid[-1]['recall_10']:.4f}")  # fmt: skip
    best = select(grid)
    frame = best["frame"]
    top_scores = [list(s[:SOFTMAX_TOP]) for s in frame["scores"]]
    correct = frame["correct_8"].to_numpy(float)
    temperature = fit_temperature(top_scores, correct)
    conf_before = np.array([softmax_top1(s, 1.0) for s in top_scores])
    conf_after = np.array([softmax_top1(s, temperature) for s in top_scores])
    ece_before, ece_after = M.ece(conf_before, correct), M.ece(conf_after, correct)
    client.log_param(parent.info.run_id, "selected", json.dumps({"k1": best["k1"], "b": best["b"]}))
    client.log_metric(parent.info.run_id, "temperature", temperature)
    client.log_metric(parent.info.run_id, "ece_t1", ece_before)
    client.log_metric(parent.info.run_id, "ece_fitted", ece_after)
    client.set_terminated(parent.info.run_id)

    config = default_config(best["k1"], best["b"], round(temperature, 4))
    config["tuned_on"] = "rulings val (splits/rulings_v1/val_refs.txt)"
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    final = B0BM25Arm(config, root=args.root)
    res = R.run_eval(final, evalset, valid8, valid10, git=git, tags={"b0_final": "val"})
    extra = M.bootstrap_intervals(
        res["frame"], {f"recall_{k}": (lambda a, k=k: float(a[f"recall_{k}"].mean()))
                       for k in (10, 20, 50)})  # fmt: skip
    print(f"\nSelected k1 {best['k1']}, b {best['b']}; T {temperature:.4f}; "
          f"val ECE {ece_before:.4f} at T=1 -> {ece_after:.4f} fitted")  # fmt: skip
    print(f"Grid parent run {parent.info.run_id}; wrote {args.config}")
    print("\n".join(R.summary_lines(res, final, evalset)))
    for k, ci in extra.items():
        print(f"  {k}: {res['metrics']['recall'][int(k.split('_')[1])]:.3f} "
              f"[{ci['low']:.3f}, {ci['high']:.3f}]")  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="b0_bm25", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    tune = sub.add_parser("tune", help="tune k1, b and T on rulings validation; write the config")
    tune.add_argument("--root", type=Path, default=DATA_ROOT)
    tune.add_argument("--config", type=Path, default=CONFIG_PATH)
    tune.add_argument("--resamples", type=int, default=1000, help="bootstrap per grid point")
    tune.set_defaults(func=cmd_tune)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
