"""Code validation and the train-ruling evidence index.

Validator: a cn8 is valid when it has a current declarable line in the pinned nomenclature
(v4.0.1633, codes.parquet) and the Trade Tariff API confirms one of those lines (at most
API_TRIES of them, in code order) is declarable today. API replies are cached on disk by
TariffApi (1 request a second; its token is held in memory and never printed). When the API
cannot be reached the nomenclature decides and the result carries api_error.

EvidenceIndex: descriptions of train rulings only (guards.check_train_split_only), read by
column name (ref, description, cn8): never validation, test, keywords or justifications.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hs_rag_classifier.arms.b1ex_embed import TRAIN_REL
from hs_rag_classifier.eval.items import DATA_ROOT
from hs_rag_classifier.guards import check_retrieval_inputs, check_train_split_only
from hs_rag_classifier.pipeline.options import Tariff

API_CACHE = DATA_ROOT / "cache" / "tariff_api"
API_TRIES = 2


class Validator:
    def __init__(self, tariff: Tariff, api=None):
        self.tariff, self.api = tariff, api
        self.memo: dict[str, dict] = {}

    def check(self, cn8: str) -> dict:
        if cn8 not in self.memo:
            self.memo[cn8] = self._check(cn8)
        return self.memo[cn8]

    def _check(self, cn8: str) -> dict:
        lines = self.tariff.code10.get(cn8, [])
        if not lines:
            return {"valid": False, "nomenclature": False, "api": None, "api_error": None}
        if self.api is None:
            return {"valid": True, "nomenclature": True, "api": None, "api_error": None}
        try:
            for code10 in lines[:API_TRIES]:
                data = self.api.get_json(f"/commodities/{code10}")
                attrs = (data or {}).get("data", {}).get("attributes", {})
                if attrs.get("declarable"):
                    return {"valid": True, "nomenclature": True, "api": code10, "api_error": None}
        except (RuntimeError, OSError, ValueError) as exc:  # API down: the nomenclature decides
            return {"valid": True, "nomenclature": True, "api": None,
                    "api_error": type(exc).__name__}  # fmt: skip
        return {"valid": False, "nomenclature": True, "api": False, "api_error": None}


def make_api(cache_dir: Path = API_CACHE):
    from hs_rag_classifier.data.nomenclature import TariffApi

    return TariffApi(cache_dir)


class EvidenceIndex:
    """ref -> (description, hs6) for train rulings."""

    def __init__(self, root: Path = DATA_ROOT, path: Path | None = None):
        train = root / TRAIN_REL
        path = check_train_split_only(path or train, train)
        (path,) = check_retrieval_inputs([path], root)
        df = pd.read_parquet(path, columns=["ref", "description", "cn8"])
        self.rows = {r.ref: (r.description, r.cn8[:6]) for r in df.itertuples(index=False)}

    def lines(self, refs: list[str], chars: int) -> list[str]:
        out = []
        for ref in refs:
            desc, hs6 = self.rows[ref]
            desc = " ".join(desc.split())
            if len(desc) > chars:
                desc = desc[:chars].rstrip() + "..."
            out.append(f"Past ruling: {desc} -> {hs6}")
        return out
