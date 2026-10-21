"""Leakage guards: keep test-only data out of anything a retrieval index or a builder reads.

Trade Tariff search references (the A to Z set) are a test set only; never index them for
retrieval. Their raw and processed files live in their own folders under the data root, so a
builder that walks a broader folder (raw/, processed/, the data root) could still pick them up
without naming them. Any builder that collects input files by globbing or walking folders must
pass its list through check_retrieval_inputs(); tests/test_atoz.py enforces that statically.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Folders under the data root that hold test-only data, never retrieval or training input.
FORBIDDEN_DIRS = ("raw/atoz", "processed/atoz")


class ForbiddenInputError(RuntimeError):
    """An input file sits under a test-only folder."""


def check_retrieval_inputs(paths: Iterable[Path | str], root: Path | str) -> list[Path]:
    """Return the paths unchanged, or raise if any resolves under a FORBIDDEN_DIRS folder."""
    root = Path(root).resolve()
    forbidden = [root / d for d in FORBIDDEN_DIRS]
    checked = []
    for p in paths:
        resolved = Path(p).resolve()
        for f in forbidden:
            if resolved == f or f in resolved.parents:
                raise ForbiddenInputError(f"{p} is under {f}: test-only data, never an input")
        checked.append(Path(p))
    return checked
