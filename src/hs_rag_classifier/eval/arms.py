"""Arm interface, prediction format, and the two sanity arms.

An arm has a name, a config dict (hashed into config_hash), and predict(items) returning one
Prediction per item. The harness times each predict call and splits the time evenly over the
batch for per-item latency; an arm may set batch_size (default 1).

OracleArm and RandomArm are sanity arms (sanity = True): they check the harness and its tests
and are never reported as results. The runner refuses them on a test split.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from hs_rag_classifier.eval.items import EvalItem

MAX_CANDIDATES = 50  # cap on ranked candidates per prediction; read at validation time


class Candidate(BaseModel):
    code: str = Field(pattern=r"^\d{1,10}$")
    score: float


class Prediction(BaseModel):
    item_id: str
    candidates: list[Candidate] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    abstain: bool = False
    rationale: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    details: dict[str, Any] | None = None  # arm-specific diagnostics (JSON in the scored frame)

    @field_validator("candidates")
    @classmethod
    def _at_most_cap(cls, value: list[Candidate]) -> list[Candidate]:
        if len(value) > MAX_CANDIDATES:
            raise ValueError(f"at most {MAX_CANDIDATES} candidates, got {len(value)}")
        return value


@runtime_checkable
class Arm(Protocol):
    name: str
    config: dict[str, Any]

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]: ...


def config_hash(config: dict[str, Any]) -> str:
    """First 12 hex digits of the SHA-256 of the config as canonical JSON."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class OracleArm:
    """Returns the label with confidence 1. Sanity arm only."""

    name = "oracle"
    sanity: ClassVar[bool] = True
    batch_size = 64

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        return [
            Prediction(
                item_id=i.item_id,
                candidates=[Candidate(code=i.label_code, score=1.0)],
                confidence=1.0,
            )
            for i in items
        ]


class RandomArm:
    """Seeded random current cn8 codes with random confidence. Sanity arm only."""

    name = "random"
    sanity: ClassVar[bool] = True
    batch_size = 64

    def __init__(self, cn8_codes: Sequence[str], seed: int = 20261021, k: int = 10):
        self.config: dict[str, Any] = {"seed": seed, "k": k}
        self._codes = sorted(cn8_codes)
        self._rng = random.Random(seed)
        self._k = k

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        out = []
        for i in items:
            codes = self._rng.sample(self._codes, self._k)
            scores = sorted((self._rng.random() for _ in codes), reverse=True)
            out.append(
                Prediction(
                    item_id=i.item_id,
                    candidates=[
                        Candidate(code=c, score=s) for c, s in zip(codes, scores, strict=True)
                    ],
                    confidence=self._rng.random(),
                )
            )
        return out
