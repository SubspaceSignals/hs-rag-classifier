"""Suggest UK commodity codes from free-text product descriptions."""

from hs_rag_classifier import offline as _offline

_offline.set_default()  # HF_HUB_OFFLINE=1 unless set explicitly (see offline.py)

__version__ = "0.1.0"
