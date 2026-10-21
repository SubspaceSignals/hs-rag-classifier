"""Frozen test sets must not change silently: each test_refs.txt has to match its SPLIT.md and
the hash pinned here."""

import hashlib
import re
from pathlib import Path

import pytest

SPLITS = Path(__file__).resolve().parents[1] / "splits"
FROZEN_TEST_SHA256 = {
    # Rulings split v1, frozen 20 Oct 2026 (seed 20261020, near-duplicate threshold 0.85).
    "rulings_v1": "426e6288fda6d8f3bce15198f776f84ddffce16f1b9dc5e6893c7ee5c6371695",
    # A to Z test set v1, re-frozen 21 Oct 2026 from /search_references against v4.0.1633,
    # with single_cn8_below terms, conflicts checked at each row's scored depth, and headings
    # carrying a sub-chapter title's sid resolved by code and suffix (8 terms added).
    "atoz_v1": "d70a60a6b50e412cec77292f4af8ca6efeeec4ecb08db8e6b49806b953934838",
}


@pytest.mark.parametrize("name", sorted(FROZEN_TEST_SHA256))
def test_test_refs_match_recorded_hash(name):
    split_dir = SPLITS / name
    recorded = re.search(
        r"test_refs\.txt SHA-256: `([0-9a-f]{64})`", (split_dir / "SPLIT.md").read_text()
    )
    assert recorded, f"{name}/SPLIT.md has no test_refs.txt SHA-256 line"
    actual = hashlib.sha256((split_dir / "test_refs.txt").read_bytes()).hexdigest()
    assert recorded.group(1) == FROZEN_TEST_SHA256[name], f"{name}: SPLIT.md hash differs"
    assert actual == FROZEN_TEST_SHA256[name], f"{name}: test_refs.txt changed; frozen set altered"
