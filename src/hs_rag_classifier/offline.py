"""Offline Hugging Face loading by default.

Every model this project loads (bge-m3, the Qwen3 embedder, the fine-tuning base model) is
already in the local Hugging Face cache, so the pipeline and script entry points run with
HF_HUB_OFFLINE=1 unless the environment sets it explicitly (an explicit value, "0" included,
is never overridden). The package __init__ calls set_default() so it is in place before
huggingface_hub reads its settings; every entry point is a hs_rag_classifier module.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

VAR = "HF_HUB_OFFLINE"
DEFAULT = "1"


def set_default(environ: MutableMapping[str, str] | None = None) -> str:
    """Set HF_HUB_OFFLINE=1 when it is unset; return the value now in force."""
    env = os.environ if environ is None else environ
    return env.setdefault(VAR, DEFAULT)
