"""Out-of-distribution flag: is the input unlike the rulings the confidence was fitted on?

Rule (fixed before any value was computed; configs/ood_rule.json holds the fitted numbers):
    signal     cosine similarity between the A4 retriever's query text for the input and its
               nearest row in the rulings index the frozen A4 pipeline uses (the dense rulings
               source, shared with B1-ex: configs/b1ex_embed.json, train rulings only). Query text
               = pipeline.checks.input_check(text, max_input_chars)["description"] with profiling
               off, exactly as the retriever embeds it; bge-m3 (shared backend, no query prefix,
               normalised), one text per encode call (batch size 1, as the single-item API).
    fit        one-class, on rulings val only: threshold = numpy.percentile(similarities, 5,
               method="linear").
    flag       similarity < threshold (strict). A flagged item is forced to review, with
               REASON in the response.
    no self-match (fit only)  for each val item, every train ruling (with its member refs) that
               shares the item's build_set.near_dup_key (first 300 input_check characters plus
               cn8) is masked. Serving and the report-only test application use the full index.

CLI (writes new files only; refuses to overwrite):
    python -m hs_rag_classifier.ood fit [--rule configs/ood_rule.json] [--csv ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hs_rag_classifier import paths as P
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT

RULE_PATH = REPO_ROOT / "configs" / "ood_rule.json"
VAL_CSV = DATA_ROOT / "eval" / "ood" / "val_similarity.csv"
REASON = "input is unlike the rulings the confidence was fitted on"
PERCENTILE = 5.0
METHOD = "linear"
BATCH_SIZE = 1
EXPECTED_EXCLUSIONS = {"600015293": ("600015294",)}  # Part A finding, asserted at fit time
SIGNAL = (
    "cosine similarity (dot product of normalised bge-m3 vectors) between the A4 retriever's query "
    "text (pipeline.checks.input_check(text, max_input_chars)['description'], profile off) and its "
    "nearest row in the dense rulings index the frozen A4 pipeline uses (shared with B1-ex); one "
    "text per encode call; flag when similarity < threshold"
)


@dataclass
class RulingsIndex:
    embeddings: np.ndarray  # (n, dim) float32, normalised, as stored
    refs: list[str]
    member_refs: list[tuple[str, ...]] | None = None


@dataclass(frozen=True)
class Nearest:
    similarity: float
    ref: str
    row: int
    n_masked: int


def query_text(text: str, max_input_chars: int) -> str:
    """The text the A4 retriever embeds for this input (profiling off)."""
    from hs_rag_classifier.pipeline import checks

    return checks.input_check(text, max_input_chars)["description"]


class OODScorer:
    """Nearest-ruling similarity for one input at a time (index and encoder loaded once)."""

    def __init__(self, index: RulingsIndex, encode: Callable[[list[str]], np.ndarray],
                 max_input_chars: int):  # fmt: skip
        self.index, self.encode, self.max_input_chars = index, encode, int(max_input_chars)
        self.embeddings = np.asarray(index.embeddings, dtype=np.float32)
        self.row = {r: i for i, r in enumerate(index.refs)}

    @classmethod
    def from_pipeline(
        cls, config: dict[str, Any] | None = None, root: Path = DATA_ROOT
    ) -> OODScorer:
        """The frozen A4 pipeline's query transform, rulings index and shared bge-m3 encoder."""
        from hs_rag_classifier.arms import a4 as A4
        from hs_rag_classifier.arms import b1ex_embed as BX
        from hs_rag_classifier.arms import embed as E
        from hs_rag_classifier.retrieval import hybrid as H

        config = config or A4.load_config()
        if config.get("profile"):
            raise ValueError("the OOD signal is defined with profiling off")
        bx_cfg = BX.load_config(REPO_ROOT / H.SOURCE_CONFIGS["rulings"])
        encoder = E.Encoder(bx_cfg)  # shared backend: the pipeline's own bge-m3 copy
        bx = BX.build_index(bx_cfg, encoder, root)
        idx = RulingsIndex(bx.embeddings, [str(r) for r in bx.meta["refs"]])
        return cls(idx, encoder.queries, int(config["max_input_chars"]))

    def query_vector(self, text: str) -> np.ndarray:
        q = np.asarray(self.encode([query_text(text, self.max_input_chars)]), dtype=np.float32)
        if q.shape[0] != BATCH_SIZE:
            raise RuntimeError("one text per encode call")
        return q[0]

    def nearest(self, text: str, exclude_rows: Sequence[int] | None = None) -> Nearest:
        sims = self.embeddings @ self.query_vector(text)
        masked = sorted(set(exclude_rows or ()))
        if masked:
            if len(masked) >= len(sims):
                raise ValueError("every index row is masked")
            sims = sims.copy()
            sims[masked] = -np.inf
        i = int(np.argmax(sims))  # first row on ties
        return Nearest(float(sims[i]), self.index.refs[i], i, len(masked))


_DEFAULT: OODScorer | None = None


def default_scorer() -> OODScorer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = OODScorer.from_pipeline()
    return _DEFAULT


def nearest_ruling_similarity(text: str, exclude_rows: Sequence[int] | None = None,
                              scorer: OODScorer | None = None) -> tuple[float, str]:  # fmt: skip
    """(max cosine over the unmasked index rows, the nearest ruling's ref)."""
    n = (scorer or default_scorer()).nearest(text, exclude_rows)
    return n.similarity, n.ref


def load_rule(path: Path = RULE_PATH) -> dict[str, Any]:
    rule = json.loads(Path(path).read_text(encoding="utf-8"))
    t = rule.get("threshold")
    if (
        not isinstance(t, float)
        or not np.isfinite(t)
        or rule.get("flag") != "similarity < threshold"
    ):
        raise ValueError(f"{path} is not an OOD rule file")
    return rule


def apply_rule(similarity: float, rule: dict[str, Any]) -> tuple[bool, str | None]:
    """(flagged, reason): flagged when similarity < threshold (strict); a flagged item is
    forced to review and the response carries the reason."""
    flagged = bool(float(similarity) < float(rule["threshold"]))
    return flagged, (REASON if flagged else None)


# --- fit (rulings val only) ------------------------------------------------------------------


def exclusion_rows(description: str, cn8: str, train_keys: dict[tuple[str, str], list[int]],
                   own_rows: dict[str, list[int]], train_refs: Sequence[str],
                   train_members: Sequence[Sequence[str]]) -> list[int]:  # fmt: skip
    """Rows of every train ruling sharing the item's near-dup key, plus rows of their member
    refs (where a member ref is itself an index row)."""
    from hs_rag_classifier.finetune import build_set as F

    rows = set(train_keys.get(F.near_dup_key(description, cn8), []))
    for r in list(rows):
        for ref in (train_refs[r], *train_members[r]):
            rows.update(own_rows.get(str(ref), []))
    return sorted(rows)


def fit_threshold(similarities: Sequence[float]) -> float:
    return float(
        np.percentile(np.asarray(similarities, dtype=np.float64), PERCENTILE, method=METHOD)
    )


def fit(
    rule_path: Path = RULE_PATH, csv_path: Path = VAL_CSV, root: Path = DATA_ROOT
) -> dict[str, Any]:
    import pandas as pd

    from hs_rag_classifier.arms import a4 as A4
    from hs_rag_classifier.arms import b1ex_embed as BX
    from hs_rag_classifier.data.nomenclature import sha256
    from hs_rag_classifier.eval.items import load
    from hs_rag_classifier.finetune import build_set as F
    from hs_rag_classifier.retrieval import hybrid as H

    for p in (rule_path, csv_path):
        if Path(p).exists():
            raise FileExistsError(f"{p} exists; the fit never overwrites")
    config = A4.load_config()
    scorer = OODScorer.from_pipeline(config, root)
    bx_cfg_path = REPO_ROOT / H.SOURCE_CONFIGS["rulings"]
    bx_cfg = BX.load_config(bx_cfg_path)
    train_path = root / BX.TRAIN_REL
    train = pd.read_parquet(train_path, columns=["ref", "member_refs", "description", "cn8"])
    if [str(r) for r in train["ref"]] != scorer.index.refs:
        raise RuntimeError("index rows are not the train split in order")
    keys: dict[tuple[str, str], list[int]] = {}
    for i, (d, c) in enumerate(zip(train["description"], train["cn8"], strict=True)):
        keys.setdefault(F.near_dup_key(d, c), []).append(i)
    own = {r: [i] for i, r in enumerate(scorer.index.refs)}
    members = [tuple(str(m) for m in ms) for ms in train["member_refs"]]
    val_path = root / "processed" / "rulings" / "v1" / "val.parquet"
    val = pd.read_parquet(val_path).set_index("ref")
    evalset = load("rulings", "val", root=root)
    rows, sims, sims_full, masked_pairs = [], [], [], {}
    for item in evalset.items:
        ex = exclusion_rows(
            item.text, val.loc[item.item_id, "cn8"], keys, own, scorer.index.refs, members
        )
        n = scorer.nearest(item.text, ex)
        full = scorer.nearest(item.text) if ex else n
        if ex:
            masked_pairs[item.item_id] = tuple(scorer.index.refs[r] for r in ex)
        rows.append({"ref": item.item_id, "similarity": repr(n.similarity), "nearest_ref": n.ref,
                     "masked": "yes" if ex else "no", "masked_refs": " ".join(masked_pairs.get(item.item_id, ())),
                     "similarity_full_index": repr(full.similarity)})  # fmt: skip
        sims.append(n.similarity)
        sims_full.append(full.similarity)
    if masked_pairs != EXPECTED_EXCLUSIONS:
        raise RuntimeError(f"self-exclusion found {masked_pairs}, expected {EXPECTED_EXCLUSIONS}")
    threshold = fit_threshold(sims)
    emb_path = root / BX.INDEX_DIR / bx_cfg["model"] / BX.index_key(bx_cfg, sha256(train_path))
    rule = {
        "rule": "phase 4 out-of-distribution flag",
        "signal": SIGNAL,
        "flag": "similarity < threshold",
        "action": "flagged items are forced to review",
        "reason": REASON,
        "threshold": threshold,
        "percentile": PERCENTILE,
        "method": METHOD,
        "val_flag_rate": float(np.mean(np.asarray(sims) < threshold)),
        "n_flagged_val": int(np.sum(np.asarray(sims) < threshold)),
        "index": {
            **P.record(emb_path / "embeddings.npy", root),
            "sha256": sha256(emb_path / "embeddings.npy"),
            "meta": {
                **P.record(emb_path / "meta.json", root),
                "sha256": sha256(emb_path / "meta.json"),
            },
            "rows": len(scorer.index.refs),
            "dim": int(scorer.embeddings.shape[1]),
            "built_from": {**P.record(train_path, root), "sha256": sha256(train_path)},
            "config": {
                "path": H.SOURCE_CONFIGS["rulings"],
                "root": "repo",
                "sha256": sha256(bx_cfg_path),
            },
            "shared_by": ["A4 retriever dense rulings source", "B1-ex"],
        },
        "embedder": {
            "model": bx_cfg["model"],
            "hf_id": bx_cfg["hf_id"],
            "revision": bx_cfg["revision"],
            "dtype": bx_cfg["dtype"],
            "max_length": bx_cfg["max_length"],
            "query_instruction": bx_cfg["query_instruction"],
            "normalise": bx_cfg["normalise"],
            "batch_size": BATCH_SIZE,
        },
        "query_text": {
            "transform": "pipeline.checks.input_check(text, max_input_chars)['description']",
            "max_input_chars": int(config["max_input_chars"]),
            "profile": bool(config["profile"]),
        },
        "val_input": {
            **P.record(val_path, root),
            "sha256": sha256(val_path),
            "split_file": {
                "path": evalset.split_file,
                "root": "repo",
                "sha256": evalset.split_sha256,
            },
            "n": len(sims),
        },
        "self_exclusion": {
            "applies_to": "fit only (serving and the report-only test application use the "
            "full index)",
            "rule": "mask every train ruling (with its member refs) sharing the val item's "
            "build_set.near_dup_key: first 300 input_check characters plus cn8",
            "excluded_pairs": [
                {"val_ref": v, "train_refs": list(t)} for v, t in masked_pairs.items()
            ],
        },
        "sensitivity_without_exclusion": {
            "label": "sensitivity only, not the rule",
            "threshold": fit_threshold(sims_full),
            "val_flag_rate": float(np.mean(np.asarray(sims_full) < fit_threshold(sims_full))),
        },
        "fitted_on": "rulings val only; no other dataset and no test data",
    }
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open("x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with Path(rule_path).open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(rule, indent=2) + "\n")
    return rule


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hs_rag_classifier.ood")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the threshold on rulings val (new files only)")
    f.add_argument("--rule", type=Path, default=RULE_PATH)
    f.add_argument("--csv", type=Path, default=VAL_CSV)
    a = ap.parse_args(argv)
    rule = fit(a.rule, a.csv)
    print(json.dumps({k: rule[k] for k in ("threshold", "val_flag_rate", "n_flagged_val")}
                     | {"sensitivity": rule["sensitivity_without_exclusion"],
                        "excluded": rule["self_exclusion"]["excluded_pairs"]}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
