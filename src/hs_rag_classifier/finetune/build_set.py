"""A4 fine-tuning set: A2's stage-1 and stage-2 scoring prompts for train rulings, leave-one-out.

    python -m hs_rag_classifier.finetune build --out DIR
    python -m hs_rag_classifier.finetune check-equivalence [--pass PATH]

Every prompt comes from A2's own code. PromptCapture is an A2Pipeline whose only change is that
_score (the llama-server call) records the system and user text instead of sending it, so
_stage1_once and _stage2_once build them exactly as A2 does: input_check, the hybrid retriever
(its sources and rrf), the cn8 filter, hs6_view and M, the evidence search and
EvidenceIndex.lines, A2's templates and fill, the tariff option lines and A2's stage-2 overflow
rule (Pipeline.stage2_options). Queries are encoded one at a time, by the same calls the arm makes.
The text the model sees is Qwen's chat template over [system, user] with enable_thinking false
and the generation prompt (as llama-server applies it); the target is the bare option label,
one token.

Leave-one-out: the rulings source and the evidence node both search the rulings vector store
(RulingsSource.store). ExcludingStore wraps that store: with an exclusion set it fetches k plus
the set's size and drops the excluded refs before returning k, so RulingsSource.search votes and
Pipeline.evidence lists past rulings over the remaining train rulings. With an empty set it is a
pass-through (the same call, the same arrays). An item's exclusion set is
  1. every train row of its near-duplicate cluster (cluster_id) and their member references; and
  2. every train ruling (with its member references) whose description, after A2's input_check,
     has the same first NEAR_DUP_PREFIX_CHARS characters and the same cn8 as any ruling in the
     item's cluster. These are near-duplicates the split's TF-IDF rule did not merge (for example
     the same text with a longer ending): within the first NEAR_DUP_PREFIX_CHARS characters,
     which is exactly what the evidence shows (A2's evidence_chars, checked at build time), such
     a ruling would print the item's own description and code back to it.
The key map is computed once per build, over the train split. The arm, the retriever, the
pipeline and the configs are not changed.

Train examples (train rulings only; val and test are never read for training):
  stage 1  items whose gold HS6 is among the M leave-one-out options; target the gold HS6's label
  stage 2  items whose gold HS6 has more than one current cn8; options every cn8 under it (A2's
           overflow rule, the gold cn8 must survive it); target the gold cn8's label
  orders   forward (as A2 builds it) and reversed (labels by position after reordering, as
           A2's order-averaging does)
Not shuffled here; the trainer shuffles with a recorded seed. Every prompt must be at most
MAX_TOKENS and every target one of the single-token labels, or the build fails (no truncation).

Also: a seeded 300-item val monitoring subset (normal retrieval: val is not in the index),
forward order, gold labels; and a seeded pilot manifest of stage-1 training example ids.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hs_rag_classifier.arms import a2_pipeline as A2
from hs_rag_classifier.arms.b1ex_embed import TRAIN_REL
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, SPLITS_DIR
from hs_rag_classifier.guards import check_retrieval_inputs, check_train_split_only
from hs_rag_classifier.pipeline import checks
from hs_rag_classifier.pipeline.graph_a2 import A2Pipeline

BASE_REPO = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TOKENIZER_DIR = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots"
    / BASE_REVISION
)  # fmt: skip
MAX_TOKENS = 4736
VAL_SUBSET_N = 300
VAL_SUBSET_SEED = 20261107
PILOT_N = 500
PILOT_SEED = 20261108
OUT = DATA_ROOT / "finetune" / "a4"
A2_VAL_PASS = DATA_ROOT / "eval" / "repro_val750" / "passes" / "a2_val_1.json"
TRAIN_COLUMNS = ["ref", "member_refs", "description", "cn8", "hs6", "chapter", "cluster_id"]
# Rule 2 of the leave-one-out exclusion (module docstring): rulings whose input_check text shares
# this many leading characters and whose cn8 is the same count as the item's own. It is the
# evidence line length, A2's evidence_chars; the build fails if the two ever differ.
NEAR_DUP_PREFIX_CHARS = 300
MAX_INPUT_CHARS = 1000  # A2's max_input_chars (input_check); checked against the config at build


class BuildError(RuntimeError):
    pass


# --- leave-one-out ---------------------------------------------------------------------------


class ExcludingStore:
    """A vector store view that never returns the refs in .exclude (rules in the module doc).

    Every search's returned ids are appended to .seen, for the leakage check."""

    def __init__(self, inner):
        self.inner = inner
        self.exclude: frozenset[str] = frozenset()
        self.seen: list[list[str]] = []

    def __len__(self) -> int:
        return len(self.inner)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.exclude:
            ids, sims = self.inner.search(queries, k)
        else:
            ids, sims = self.inner.search(queries, k + len(self.exclude))
            rows = []
            for r_ids, r_sims in zip(ids, sims, strict=True):
                keep = [i for i, ref in enumerate(r_ids) if str(ref) not in self.exclude][:k]
                if len(keep) < min(k, len(self.inner) - len(self.exclude)):
                    raise BuildError("not enough neighbours left after exclusion")
                rows.append((r_ids[keep], r_sims[keep]))
            ids = np.stack([r[0] for r in rows])
            sims = np.stack([r[1] for r in rows])
        self.seen.extend([[str(x) for x in row] for row in ids])
        return ids, sims


def loo_retriever(retriever) -> tuple[Any, ExcludingStore]:
    """A shallow copy of the retriever whose rulings source searches through an ExcludingStore.

    The retriever object, its sources and its store are not modified."""
    rulings = copy.copy(retriever.sources["rulings"])
    store = ExcludingStore(rulings.store)
    rulings.store = store
    out = copy.copy(retriever)
    out.sources = {**retriever.sources, "rulings": rulings}
    return out, store


def near_dup_key(description: str, cn8: str, prefix_chars: int = NEAR_DUP_PREFIX_CHARS,
                 max_input_chars: int = MAX_INPUT_CHARS) -> tuple[str, str]:  # fmt: skip
    """(first prefix_chars characters of the input_check text, cn8)."""
    text = checks.input_check(description, max_input_chars)["description"]
    return text[:prefix_chars], str(cn8)


def exclusion_sets(train: pd.DataFrame, prefix_chars: int = NEAR_DUP_PREFIX_CHARS,
                   max_input_chars: int = MAX_INPUT_CHARS) -> dict[str, frozenset[str]]:  # fmt: skip
    """ref -> its cluster's refs and member refs, plus every ruling (and its member refs) that
    shares a near-duplicate key with any ruling of the cluster (rules in the module docstring)."""
    own: dict[str, set[str]] = {}  # ref -> itself and its member refs
    cluster_rows: dict[str, list[str]] = {}
    by_key: dict[tuple[str, str], list[str]] = {}
    key_of: dict[str, tuple[str, str]] = {}
    for r in train.itertuples(index=False):
        ref = str(r.ref)
        own[ref] = {ref, *(str(m) for m in r.member_refs)}
        cluster_rows.setdefault(str(r.cluster_id), []).append(ref)
        key_of[ref] = near_dup_key(r.description, r.cn8, prefix_chars, max_input_chars)
        by_key.setdefault(key_of[ref], []).append(ref)
    out = {}
    for r in train.itertuples(index=False):
        rows = cluster_rows[str(r.cluster_id)]
        refs = set().union(*(own[x] for x in rows))
        for x in rows:
            for y in by_key[key_of[x]]:
                refs |= own[y]
        out[str(r.ref)] = frozenset(refs)
    return out


def cluster_sets(train: pd.DataFrame) -> dict[str, frozenset[str]]:
    """ref -> its cluster's refs and member refs only (rule 1), for the report."""
    by_cluster: dict[str, set[str]] = {}
    for r in train.itertuples(index=False):
        refs = by_cluster.setdefault(str(r.cluster_id), set())
        refs.add(str(r.ref))
        refs.update(str(m) for m in r.member_refs)
    return {str(r.ref): frozenset(by_cluster[str(r.cluster_id)]) for r in train.itertuples()}


# --- prompts ---------------------------------------------------------------------------------


class PromptCapture(A2Pipeline):
    """A2Pipeline with the scoring call replaced by a recorder (no server, no HTTP client)."""

    def _score(self, system: str, user: str, labels: list[str], name: str) -> dict:
        self.captured = {"system": system, "user": user, "labels": list(labels), "name": name}
        return dict(self.captured)

    def item_state(self, text: str) -> dict:
        """input_check, retrieve and evidence, as A2's graph runs them (profile off in A2)."""
        if self.cfg["profile"]:
            raise BuildError(
                "A2 runs with the profile off; the builder does not call the chat model"
            )
        s: dict[str, Any] = {"raw": text}
        s.update(self.input_check(s))
        s.update(self.retrieve(s))
        if self.cfg["evidence"]:
            s.update(self.evidence(s))
        return s

    def stage1(self, s: dict, hs6s: list[str]) -> dict:
        return self._stage1_once(s, hs6s, "stage1")

    def stage2(self, s: dict, hs6: str, cn8s: list[str]) -> dict:
        return self._stage2_once(s, hs6, cn8s, f"stage2 {hs6}")


def make_capture(config: dict, retriever, tariff, evidence_index) -> PromptCapture:
    return PromptCapture(config, None, retriever, tariff, None, None, {}, evidence_index)


def chat_text(tok, system: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )  # fmt: skip


def token_ids(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def label_tokens(tok, labels: Sequence[str]) -> dict[str, int]:
    out = {}
    for lab in labels:
        ids = tok.encode(lab, add_special_tokens=False)
        if len(ids) != 1:
            raise BuildError(f"label {lab!r} is {len(ids)} tokens, not one")
        out[lab] = ids[0]
    if len(set(out.values())) != len(out):
        raise BuildError("labels share a token")
    return out


def example(tok, label_ids: dict[str, int], cap: dict, codes: list[str], gold: str,
            meta: dict, max_tokens: int = MAX_TOKENS) -> dict:  # fmt: skip
    """One row: the chat-templated prompt, its length, the gold code's label and token."""
    labels = cap["labels"]
    if len(labels) != len(codes):
        raise BuildError("labels and options differ in length")
    target = labels[codes.index(gold)]
    if target not in label_ids:
        raise BuildError(f"target {target!r} is not one of the single-token labels")
    text = chat_text(tok, cap["system"], cap["user"])
    n = len(token_ids(tok, text))
    if n > max_tokens:
        raise BuildError(f"{meta.get('example_id')}: {n} tokens > {max_tokens} (never truncated)")
    return {**meta, "prompt": text, "n_tokens": n, "target_label": target,
            "target_token_id": label_ids[target], "gold_code": gold, "options": list(codes),
            "target_position": codes.index(gold)}  # fmt: skip


def item_examples(pipe: PromptCapture, tok, label_ids, s: dict, ref: str, gold_hs6: str,
                  gold_cn8: str, meta: dict, orders=("forward", "reversed"),
                  max_tokens: int = MAX_TOKENS) -> tuple[list[dict], dict]:  # fmt: skip
    """Stage-1 and stage-2 examples for one item, in each order, plus facts for the report."""
    rows, facts = [], {}
    hs6s = pipe.stage1_options(s["fused"])
    facts["gold_hs6_in_options"] = gold_hs6 in hs6s
    facts["n_options"] = len(hs6s)
    if gold_hs6 in hs6s:
        for order in orders:
            codes = hs6s if order == "forward" else list(reversed(hs6s))
            cap = pipe.stage1(s, codes)
            rows.append(example(tok, label_ids, cap, codes, gold_hs6,
                                {**meta, "example_id": f"{ref}:1:{order}", "stage": 1,
                                 "order": order}, max_tokens))  # fmt: skip
    kids = pipe.tariff.children[gold_hs6]
    facts["gold_hs6_children"] = len(kids)
    if len(kids) > 1:
        cn8s, overflow = pipe.stage2_options(gold_hs6, s["fused"])
        facts["stage2_overflow"] = overflow
        if gold_cn8 not in cn8s:
            raise BuildError(f"{ref}: gold cn8 {gold_cn8} dropped by the stage-2 overflow rule")
        for order in orders:
            codes = cn8s if order == "forward" else list(reversed(cn8s))
            cap = pipe.stage2(s, gold_hs6, codes)
            rows.append(example(tok, label_ids, cap, codes, gold_cn8,
                                {**meta, "example_id": f"{ref}:2:{order}", "stage": 2,
                                 "order": order}, max_tokens))  # fmt: skip
    return rows, facts


# --- inputs ----------------------------------------------------------------------------------


def load_train(root: Path = DATA_ROOT, splits_dir: Path = SPLITS_DIR) -> pd.DataFrame:
    """The rulings train split (guarded: exactly the train file; ids checked against the split)."""
    from hs_rag_classifier.eval.items import _check_ids

    path = check_train_split_only(root / TRAIN_REL, root / TRAIN_REL)
    (path,) = check_retrieval_inputs([path], root)
    df = pd.read_parquet(path, columns=TRAIN_COLUMNS)
    _check_ids(
        [r for m in df["member_refs"] for r in m], splits_dir / "rulings_v1" / "train_refs.txt"
    )
    return df.reset_index(drop=True)


def real_parts(root: Path = DATA_ROOT):
    """A2's frozen config and the real retriever, tariff and evidence index (no server)."""
    from hs_rag_classifier.pipeline.options import load_tariff
    from hs_rag_classifier.pipeline.validate import EvidenceIndex
    from hs_rag_classifier.retrieval.hybrid import HybridRetriever
    from hs_rag_classifier.retrieval.hybrid import load_config as load_hybrid

    config = A2.load_config()
    return config, HybridRetriever(load_hybrid(), root), load_tariff(root), EvidenceIndex(root)


def load_tokenizer(path: Path = TOKENIZER_DIR):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


# --- the sets --------------------------------------------------------------------------------


def build_train(train: pd.DataFrame, pipe: PromptCapture, store: ExcludingStore, tok,
                max_tokens: int = MAX_TOKENS) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """(examples, per-item facts) for every train row, leave-one-out."""
    label_ids = label_tokens(tok, pipe.labels)
    if int(pipe.cfg["evidence_chars"]) != NEAR_DUP_PREFIX_CHARS:
        raise BuildError("NEAR_DUP_PREFIX_CHARS must equal A2's evidence_chars")
    if int(pipe.cfg["max_input_chars"]) != MAX_INPUT_CHARS:
        raise BuildError("MAX_INPUT_CHARS must equal A2's max_input_chars")
    excl = exclusion_sets(train)
    clus = cluster_sets(train)
    rows, facts = [], []
    for r in train.itertuples(index=False):
        ref = str(r.ref)
        store.exclude, store.seen = excl[ref], []
        s = pipe.item_state(r.description)
        seen = [i for call in store.seen for i in call]
        meta = {"ref": ref, "cluster_id": str(r.cluster_id), "chapter": str(r.chapter)}
        ex, f = item_examples(pipe, tok, label_ids, s, ref, str(r.hs6), str(r.cn8), meta,
                              max_tokens=max_tokens)  # fmt: skip
        rows.extend(ex)
        facts.append({**meta, "hs6": str(r.hs6), "cn8": str(r.cn8), **f,
                      "excluded": len(excl[ref]), "excluded_cluster": len(clus[ref]),
                      "excluded_near_dup": sorted(excl[ref] - clus[ref]),
                      "searches": len(store.seen),
                      "leaked_refs": sorted(set(seen) & excl[ref]),
                      "evidence_refs": store.seen[-1] if pipe.cfg["evidence"] else []})  # fmt: skip
    store.exclude, store.seen = frozenset(), []
    return pd.DataFrame(rows), pd.DataFrame(facts)


def build_val_subset(items, pipe: PromptCapture, tok, n: int = VAL_SUBSET_N,
                     seed: int = VAL_SUBSET_SEED) -> pd.DataFrame:  # fmt: skip
    """n seeded val items; stage 1 always (target null when the gold HS6 is not an option),
    stage 2 on the gold HS6 when it has more than one cn8; forward order; normal retrieval."""
    label_ids = label_tokens(tok, pipe.labels)
    ids = sorted(it.item_id for it in items)
    pick = set(np.random.default_rng(seed).choice(ids, size=n, replace=False).tolist())
    rows = []
    for it in items:
        if it.item_id not in pick:
            continue
        gold_cn8, gold_hs6 = it.label_code[:8], it.label_code[:6]
        s = pipe.item_state(it.text)
        meta = {"ref": it.item_id, "cluster_id": str(it.cluster_id), "chapter": str(it.chapter)}
        hs6s = pipe.stage1_options(s["fused"])
        cap = pipe.stage1(s, hs6s)
        if gold_hs6 in hs6s:
            rows.append(example(tok, label_ids, cap, hs6s, gold_hs6,
                                {**meta, "example_id": f"{it.item_id}:1:forward", "stage": 1,
                                 "order": "forward", "gold_in_options": True}))  # fmt: skip
        else:
            text = chat_text(tok, cap["system"], cap["user"])
            rows.append({**meta, "example_id": f"{it.item_id}:1:forward", "stage": 1,
                         "order": "forward", "gold_in_options": False, "prompt": text,
                         "n_tokens": len(token_ids(tok, text)), "target_label": None,
                         "target_token_id": None, "gold_code": gold_hs6, "options": hs6s,
                         "target_position": None})  # fmt: skip
        if len(pipe.tariff.children[gold_hs6]) > 1:
            cn8s, _ = pipe.stage2_options(gold_hs6, s["fused"])
            if gold_cn8 in cn8s:
                cap = pipe.stage2(s, gold_hs6, cn8s)
                rows.append(example(tok, label_ids, cap, cn8s, gold_cn8,
                                    {**meta, "example_id": f"{it.item_id}:2:forward", "stage": 2,
                                     "order": "forward", "gold_in_options": True}))  # fmt: skip
    return pd.DataFrame(rows)


def pilot_manifest(train_examples: pd.DataFrame, n: int = PILOT_N,
                   seed: int = PILOT_SEED) -> pd.DataFrame:  # fmt: skip
    ids = sorted(train_examples.loc[train_examples["stage"] == 1, "example_id"])
    pick = np.random.default_rng(seed).choice(ids, size=n, replace=False)
    return pd.DataFrame({"example_id": sorted(pick.tolist())})


# --- equivalence with A2 ---------------------------------------------------------------------


def check_equivalence(pass_rows: list[dict], items, pipe: PromptCapture, tok) -> dict:
    """Rebuild a saved A2 pass's prompts with an empty exclusion set; compare with what A2 sent.

    The pass keeps per item: stage-1 options, evidence lines, every stage-2 call's options and
    the server-reported tokens_in per call. Prompts and token ids are not saved by A2, so ids
    are compared between two tokenisations here (template tokenize=True against the text)."""
    by_id = {it.item_id: it for it in items}
    out = {"n_items": 0, "stage1": Counter(), "stage2": Counter(), "examples": []}
    for r in pass_rows:
        it, det = by_id[str(r["item_id"])], r["details"]
        s = pipe.item_state(it.text)
        out["n_items"] += 1
        hs6s = pipe.stage1_options(s["fused"])
        cap = pipe.stage1(s, hs6s)
        text = chat_text(tok, cap["system"], cap["user"])
        ids = token_ids(tok, text)
        ids2 = tok.apply_chat_template(
            [{"role": "system", "content": cap["system"]}, {"role": "user", "content": cap["user"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)  # fmt: skip
        ids2 = list(ids2["input_ids"] if hasattr(ids2, "keys") else ids2)
        checks1 = {
            "options": hs6s == det["stage1"]["options"],
            "evidence": s.get("evidence", []) == det.get("evidence", []),
            "tokens_in": len(ids) == det["tokens"]["stage1"][0],
            "ids_consistent": ids == ids2,
        }
        for k, v in checks1.items():
            out["stage1"][k] += bool(v)
        if not all(checks1.values()) and len(out["examples"]) < 10:
            out["examples"].append({"item_id": it.item_id, "stage": 1, **checks1,
                                    "tokens": [len(ids), det["tokens"]["stage1"][0]]})  # fmt: skip
        for n, d in enumerate(det["stage2"]):
            if d.get("skipped"):
                continue
            cn8s, _ = pipe.stage2_options(d["hs6"], s["fused"])
            cap = pipe.stage2(s, d["hs6"], cn8s)
            ids = token_ids(tok, chat_text(tok, cap["system"], cap["user"]))
            server = (det["tokens"].get(f"stage2_{n + 1}") or [None])[0]
            checks2 = {"options": cn8s == d["cn8s"], "tokens_in": len(ids) == server}
            out["stage2"]["n"] += 1
            for k, v in checks2.items():
                out["stage2"][k] += bool(v)
            if not all(checks2.values()) and len(out["examples"]) < 10:
                out["examples"].append({"item_id": it.item_id, "stage": 2, "hs6": d["hs6"],
                                        **checks2, "tokens": [len(ids), server]})  # fmt: skip
    out["stage1"], out["stage2"] = dict(out["stage1"]), dict(out["stage2"])
    out["exact"] = (all(v == out["n_items"] for v in out["stage1"].values())
                    and all(v == out["stage2"]["n"] for v in out["stage2"].values()))  # fmt: skip
    return out


# --- outputs ---------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path = REPO_ROOT) -> dict[str, Any]:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()  # fmt: skip
    dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout.strip())  # fmt: skip
    return {"commit": head, "dirty": dirty}


def summary(train_ex: pd.DataFrame, facts: pd.DataFrame, labels: Sequence[str]) -> dict:
    def lens(x):
        x = np.asarray(x, float)
        return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
                "max": int(x.max()), "n": len(x)}  # fmt: skip

    out: dict[str, Any] = {
        "counts": {
            f"stage{st}_{o}": int(((train_ex["stage"] == st) & (train_ex["order"] == o)).sum())
            for st in (1, 2)
            for o in ("forward", "reversed")
        },
        "n_examples": len(train_ex),
        "tokens": {
            f"stage{st}": lens(train_ex.loc[train_ex["stage"] == st, "n_tokens"]) for st in (1, 2)
        },
        "positions": {},
        "missing_gold_hs6": {
            "n": int((~facts["gold_hs6_in_options"]).sum()),
            "by_chapter": {
                k: int(v)
                for k, v in sorted(
                    facts.loc[~facts["gold_hs6_in_options"], "chapter"].value_counts().items()
                )
            },
        },
        "near_dup_exclusion": {
            "items_with_extra": int(facts["excluded_near_dup"].map(len).gt(0).sum()),
            "refs_added_total": int(facts["excluded_near_dup"].map(len).sum()),
            "distinct_refs_added": len({r for x in facts["excluded_near_dup"] for r in x}),
        },
        "leakage": {
            "items_with_own_cluster_returned": int(facts["leaked_refs"].map(len).gt(0).sum()),
            "items": len(facts),
            "searches_per_item": sorted(set(facts["searches"])),
        },
        "stage2_overflow_items": int(
            facts.get("stage2_overflow", pd.Series(dtype=bool)).fillna(False).sum()
        ),
    }
    for st in (1, 2):
        for o in ("forward", "reversed"):
            sub = train_ex[(train_ex["stage"] == st) & (train_ex["order"] == o)]
            hist = Counter(sub["target_label"])
            out["positions"][f"stage{st}_{o}"] = {
                "histogram": {lab: int(hist.get(lab, 0)) for lab in labels if hist.get(lab)},
                "share_at_A": float((sub["target_label"] == labels[0]).mean()),
            }
    return out


def write_outputs(out: Path, train_ex: pd.DataFrame, facts: pd.DataFrame, val: pd.DataFrame,
                  pilot: pd.DataFrame, meta: dict) -> dict:  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    files = {"train_examples.parquet": train_ex, "train_items.parquet": facts,
             "val_subset.parquet": val, "pilot_manifest.parquet": pilot}  # fmt: skip
    hashes = {}
    for name, frame in files.items():
        frame.to_parquet(out / name, index=False)
        hashes[name] = sha256_file(out / name)
    manifest = {**meta, "rows": {k: len(v) for k, v in files.items()}, "sha256": hashes}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build(out: Path = OUT, root: Path = DATA_ROOT) -> dict:
    from hs_rag_classifier.eval import items as I

    t0 = time.perf_counter()
    commit = git_commit()
    config, retriever, tariff, evidence = real_parts(root)
    tok = load_tokenizer()
    loo, store = loo_retriever(retriever)
    train = load_train(root)
    train_ex, facts = build_train(train, make_capture(config, loo, tariff, evidence), store, tok)
    if n := int(facts["leaked_refs"].map(len).gt(0).sum()):
        raise BuildError(f"{n} items saw their own cluster in the rulings vote or evidence")
    val = build_val_subset(I.load("rulings", "val", root=root).items,
                           make_capture(config, retriever, tariff, evidence), tok)  # fmt: skip
    pilot = pilot_manifest(train_ex)
    meta = {
        "builder": "hs_rag_classifier.finetune.build_set", "builder_commit": commit,
        "base_model": {"repo": BASE_REPO, "revision": BASE_REVISION},
        "tokenizer_dir": str(TOKENIZER_DIR), "max_tokens": MAX_TOKENS,
        "chat_template": {"enable_thinking": False, "add_generation_prompt": True},
        "a2_config_sha256": sha256_file(A2.CONFIG_PATH),
        "seeds": {"val_subset": VAL_SUBSET_SEED, "pilot": PILOT_SEED},
        "sizes": {"val_subset_items": VAL_SUBSET_N, "pilot_examples": PILOT_N},
        "train_split": {"file": str(TRAIN_REL), "rows": len(train)},
        "leave_one_out": {"rule": "cluster_id rows and member refs, plus rulings with the same first "
                          "NEAR_DUP_PREFIX_CHARS input_check characters and cn8; vote and evidence",
                          "near_dup_prefix_chars": NEAR_DUP_PREFIX_CHARS},
        "orders": ["forward", "reversed"], "shuffled": False,
        "summary": summary(train_ex, facts, config["labels"]),
    }  # fmt: skip
    manifest = write_outputs(Path(out), train_ex, facts, val, pilot, meta)
    manifest["seconds"] = time.perf_counter() - t0  # reported, not written (keeps files stable)
    return manifest


def run_equivalence(pass_path: Path = A2_VAL_PASS, root: Path = DATA_ROOT) -> dict:
    from hs_rag_classifier.eval import items as I

    config, retriever, tariff, evidence = real_parts(root)
    tok = load_tokenizer()
    rows = json.loads(Path(pass_path).read_text(encoding="utf-8"))["rows"]
    return check_equivalence(rows, I.load("rulings", "val", root=root).items,
                             make_capture(config, retriever, tariff, evidence), tok)  # fmt: skip


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m hs_rag_classifier.finetune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the A4 training set, val subset and pilot manifest")
    b.add_argument("--out", type=Path, default=OUT)
    e = sub.add_parser("check-equivalence", help="rebuild a saved A2 val pass's prompts")
    e.add_argument("--pass", dest="pass_path", type=Path, default=A2_VAL_PASS)
    e.add_argument("--out", type=Path, help="write the result as JSON here")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.cmd == "build":
        m = build(a.out)
        print(json.dumps({k: m[k] for k in ("rows", "sha256", "seconds")}, indent=1))
        print(json.dumps(m["summary"], indent=1))
        return 0
    r = run_equivalence(a.pass_path)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=1))
    return 0 if r["exact"] else 1
