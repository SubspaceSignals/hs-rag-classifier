"""A4 checkpoint candidate on validation: A3's pipeline with a LoRA adapter on the scoring calls.

Config: configs/a4_candidate_val.json (validation only, not frozen). It pins
configs/a3_pipeline.json by SHA-256 and changes only: both stage temperatures (T = 1 for
candidate selection) and a2.lora_scales (the per-request LoRA scale of the stage-1 and stage-2
scoring calls, and of the rationale call). Everything else is A3's: order-averaging on both
stages, policy (b) review against B1-ex's locked config, rationale on.

The adapter (a GGUF converted from one A4 checkpoint) is supplied per candidate at run time,
never written into the file:

    python -m hs_rag_classifier.eval run --arm a4_candidate --dataset rulings --split val \
        --candidate step_00250 --adapter GGUF --adapter-sha256 HEX

The block {"path", "sha256"} becomes the config's "adapter" (so it is in the config_hash and the
MLflow config param); the run refuses to start unless that file's SHA-256 matches and it is the
one adapter llama-server lists in /lora-adapters (pipeline/llm.py check_server_adapters).
The candidate and adapter SHA-256 are MLflow params (run_params) and in each prediction's
details["a4"]. The harness refuses this arm on a test split (val_only).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from hs_rag_classifier.arms import a3_pipeline as A3
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval.arms import Prediction
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem

CONFIG_PATH = REPO_ROOT / "configs" / "a4_candidate_val.json"
SHA256 = re.compile(r"[0-9a-f]{64}")


def load_config(adapter: Path | str, adapter_sha256: str, candidate: str,
                path: Path = CONFIG_PATH, repo: Path = REPO_ROOT) -> dict[str, Any]:  # fmt: skip
    """A3's config with the candidate's temperatures, LoRA scales and adapter block."""
    a4 = json.loads(path.read_text(encoding="utf-8"))
    pin = a4["base_config"]
    if sha256(repo / pin["path"]) != pin["sha256"]:
        raise RuntimeError(f"{pin['path']} changed since {path.name} was written")
    if candidate not in a4["candidates"]:
        raise ValueError(f"candidate {candidate!r} is not one of {a4['candidates']}")
    if not SHA256.fullmatch(adapter_sha256 or ""):
        raise ValueError("adapter_sha256 must be 64 lower-case hex digits")
    base = A3.load_config(repo / pin["path"], repo)
    a2 = {**base["a2"], "temperature": a4["temperature"], "lora_scales": a4["lora_scales"]}
    block = {"path": str(Path(adapter).expanduser().resolve()), "sha256": adapter_sha256}
    meta = {"config": {"path": path.name, "sha256": sha256(path)}, "candidate": candidate}
    return {**base, "a2": a2, "adapter": block, "a4": meta}


class A4CandidateArm(A3.A3Arm):
    """A3 with one candidate adapter on the scoring calls; validation only."""

    name = "a4_candidate"
    val_only = True

    def __init__(self, config: dict[str, Any], root: Path = DATA_ROOT, pipeline=None,
                 b1ex=None):  # fmt: skip
        if not config or "adapter" not in config or "a4" not in config:
            raise ValueError("a4_candidate needs a config from a4_candidate.load_config")
        super().__init__(config, root, pipeline, b1ex)  # runs the slot and adapter guards
        self.meta = {"candidate": config["a4"]["candidate"],
                     "adapter_sha256": config["adapter"]["sha256"],
                     "adapter_path": config["adapter"]["path"]}  # fmt: skip
        self.run_params = {"candidate_step": self.meta["candidate"],
                           "adapter_sha256": self.meta["adapter_sha256"]}  # fmt: skip

    def predict(self, items: Sequence[EvalItem]) -> list[Prediction]:
        out = super().predict(items)
        return [p.model_copy(update={"details": {**(p.details or {}), "a4": dict(self.meta)}})
                for p in out]  # fmt: skip

    def after_scoring(self, frame: pd.DataFrame, n_resamples: int = M.BOOTSTRAP_RESAMPLES,
                      seed: int = M.BOOTSTRAP_SEED) -> dict[str, Any]:  # fmt: skip
        out = A3.after_scoring(frame, n_resamples, seed)
        out["metrics"]["a4"] = dict(self.meta)
        return out
