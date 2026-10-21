"""Evaluation items and the loaders for each frozen set.

Loaders read named columns from named files only. Ruling justifications state the answer and
are never read here: rulings are loaded from the split Parquet files, which never hold them.

Each dataset records the SHA-256 of its frozen split file, and its item ids are checked
against that file, so every run is tied to the exact set:
    rulings  splits/rulings_v1/{val,test}_refs.txt  (member references of each group)
    atoz     splits/atoz_v1/test_refs.txt           (member ids of each term)
    hand     the CSV itself (no frozen ref file yet)

Inputs and labels:
    rulings  text = description; with_keywords appends "Keywords: ..." (ablation only).
             label = code10 (label_depth 10) or cn8 (8); cluster = near-duplicate cluster.
    atoz     text = title (the A to Z term); label = code10 or cn8 by label_depth; each term
             is its own cluster.
    hand     text = description; label = code10 when given, else cn8; each row its own
             cluster. An empty CSV loads as 0 items.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from hs_rag_classifier.data import atoz as atoz_data
from hs_rag_classifier.data import hand as hand_data
from hs_rag_classifier.data import rulings as rulings_data

DATA_ROOT = Path.home() / "data" / "hs-rag-classifier"
REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = REPO_ROOT / "splits"
DATASETS = ("rulings", "atoz", "hand")
TEST_ONLY = ("atoz", "hand")

RULINGS_COLUMNS = ["ref", "member_refs", "description", "keywords", "code10", "cn8", "chapter",
                   "cluster_id", "label_depth"]  # fmt: skip
ATOZ_COLUMNS = ["id", "member_ids", "title", "cn8", "code10", "label_depth"]


class EvalItem(BaseModel):
    item_id: str
    text: str
    label_code: str = Field(pattern=r"^(\d{8}|\d{10})$")
    label_depth: Literal[8, 10]
    cluster_id: str
    chapter: str
    dataset: str
    split: str
    keywords: list[str] | None = None


class EvalSet(BaseModel):
    dataset: str
    split: str
    items: list[EvalItem]
    split_file: str
    split_sha256: str
    ablation: str = "none"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref_ids(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _check_ids(members: list[str], ref_file: Path) -> None:
    if sorted(members, key=int) != _ref_ids(ref_file):
        raise RuntimeError(f"items do not match the frozen ids in {ref_file}")


def load_rulings(
    split: str = "val",
    with_keywords: bool = False,
    root: Path = DATA_ROOT,
    splits_dir: Path = SPLITS_DIR,
) -> EvalSet:
    if split not in ("val", "test"):
        raise ValueError(f"rulings split must be val or test, not {split!r}")
    path = root / "processed" / "rulings" / rulings_data.SPLIT_NAME / f"{split}.parquet"
    df = pd.read_parquet(path, columns=RULINGS_COLUMNS)
    ref_file = splits_dir / f"rulings_{rulings_data.SPLIT_NAME}" / f"{split}_refs.txt"
    _check_ids([r for m in df["member_refs"] for r in m], ref_file)
    items = []
    for r in df.itertuples(index=False):
        depth = int(r.label_depth)
        keywords = [str(k) for k in r.keywords]
        text = r.description
        if with_keywords and keywords:
            text = f"{text}\nKeywords: {'; '.join(keywords)}"
        items.append(
            EvalItem(
                item_id=r.ref,
                text=text,
                label_code=r.code10 if depth == 10 else r.cn8,
                label_depth=depth,
                cluster_id=r.cluster_id,
                chapter=r.chapter,
                dataset="rulings",
                split=split,
                keywords=keywords,
            )
        )
    return EvalSet(
        dataset="rulings",
        split=split,
        items=items,
        split_file=str(ref_file.relative_to(splits_dir.parent)),
        split_sha256=sha256_file(ref_file),
        ablation="keywords" if with_keywords else "none",
    )


def load_atoz(root: Path = DATA_ROOT, splits_dir: Path = SPLITS_DIR) -> EvalSet:
    path = root / "processed" / "atoz" / atoz_data.SPLIT_NAME / "atoz_test8.parquet"
    df = pd.read_parquet(path, columns=ATOZ_COLUMNS)
    ref_file = splits_dir / f"atoz_{atoz_data.SPLIT_NAME}" / "test_refs.txt"
    _check_ids([i for m in df["member_ids"] for i in m], ref_file)
    items = [
        EvalItem(
            item_id=r.id,
            text=r.title,
            label_code=r.code10 if int(r.label_depth) == 10 else r.cn8,
            label_depth=int(r.label_depth),
            cluster_id=r.id,
            chapter=r.cn8[:2],
            dataset="atoz",
            split="test",
        )
        for r in df.itertuples(index=False)
    ]
    return EvalSet(
        dataset="atoz",
        split="test",
        items=items,
        split_file=str(ref_file.relative_to(splits_dir.parent)),
        split_sha256=sha256_file(ref_file),
    )


def load_hand(path: Path = hand_data.DEFAULT_CSV) -> EvalSet:
    _, rows = hand_data.read_rows(path)
    items = [
        EvalItem(
            item_id=r["id"],
            text=r["description"],
            label_code=r["code10"] or r["cn8"],
            label_depth=10 if r["code10"] else 8,
            cluster_id=r["id"],
            chapter=r["cn8"][:2],
            dataset="hand",
            split="test",
        )
        for r in rows
    ]
    return EvalSet(
        dataset="hand",
        split="test",
        items=items,
        split_file=str(path),
        split_sha256=sha256_file(path),
    )


def load(
    dataset: str,
    split: str,
    with_keywords: bool = False,
    root: Path = DATA_ROOT,
    splits_dir: Path = SPLITS_DIR,
    hand_csv: Path | None = None,
) -> EvalSet:
    """The named dataset and split; A to Z and hand have a test split only."""
    if with_keywords and dataset != "rulings":
        raise ValueError("--with-keywords applies to rulings only")
    if dataset in TEST_ONLY and split != "test":
        raise ValueError(f"{dataset} has only a test split; pass --split test")
    if dataset == "rulings":
        return load_rulings(split, with_keywords, root, splits_dir)
    if dataset == "atoz":
        return load_atoz(root, splits_dir)
    if dataset == "hand":
        return load_hand(hand_csv or root / "hand" / "hand_v1.csv")
    raise ValueError(f"unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}")
