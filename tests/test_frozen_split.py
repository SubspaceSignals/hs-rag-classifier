"""The frozen test split must not change silently: test_refs.txt has to match SPLIT.md and here."""

import hashlib
import re
from pathlib import Path

SPLIT_DIR = Path(__file__).resolve().parents[1] / "splits" / "rulings_v1"
# Rulings split v1, frozen 20 Oct 2026 (seed 20261020, near-duplicate threshold 0.85).
FROZEN_TEST_SHA256 = "426e6288fda6d8f3bce15198f776f84ddffce16f1b9dc5e6893c7ee5c6371695"


def test_test_refs_match_recorded_hash():
    recorded = re.search(
        r"test_refs\.txt SHA-256: `([0-9a-f]{64})`", (SPLIT_DIR / "SPLIT.md").read_text()
    )
    assert recorded, "SPLIT.md has no test_refs.txt SHA-256 line"
    actual = hashlib.sha256((SPLIT_DIR / "test_refs.txt").read_bytes()).hexdigest()
    assert recorded.group(1) == FROZEN_TEST_SHA256, "SPLIT.md hash differs from the frozen one"
    assert actual == FROZEN_TEST_SHA256, "test_refs.txt changed; the frozen test split was altered"
