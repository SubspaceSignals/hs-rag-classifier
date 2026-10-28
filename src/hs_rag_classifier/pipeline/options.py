"""Option texts for the two decision stages, from the pinned nomenclature (v4.0.1633).

Stage 1 (HS6):  "<label>. <hs6>: <heading text> > <subheading text>", where the subheading text
                is the chain of tariff lines from below the heading down to the HS6 line
                (grouping lines such as "Other:" included). No chapter titles. When the HS6
                line is the heading itself (an undivided heading), the line is the heading only.
Stage 2 (cn8):  "<label>. <cn8>: <description below the subheading>", the chain of lines from
                below the HS6 line down to the cn8 line; when the cn8 line is the HS6 line, its
                own description.

Only HS6 codes with at least one current cn8 (codes8.parquet) are options; the cn8s under an
HS6 are every current cn8 in codes8.parquet, in code order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hs_rag_classifier.data.rulings import NOMENCLATURE_VERSION
from hs_rag_classifier.eval.items import DATA_ROOT
from hs_rag_classifier.guards import check_retrieval_inputs

SEP = " > "


@dataclass(frozen=True)
class Tariff:
    hs6_text: dict[str, str]  # hs6 -> "<heading text> > <subheading text>"
    cn8_text: dict[str, str]  # cn8 -> description below the subheading
    children: dict[str, list[str]]  # hs6 -> current cn8s, in code order
    code10: dict[str, list[str]]  # cn8 -> its current declarable code10 lines

    def hs6_line(self, label: str, hs6: str) -> str:
        return f"{label}. {hs6}: {self.hs6_text[hs6]}"

    def cn8_line(self, label: str, cn8: str) -> str:
        return f"{label}. {cn8}: {self.cn8_text[cn8]}"


def _is_line(row, code10: str) -> bool:
    return row.code10 == code10 and row.suffix == "80"


def build_tariff(codes: pd.DataFrame, codes8: pd.DataFrame) -> Tariff:
    """Option texts from codes.parquet and codes8.parquet (rules in the module docstring)."""
    by_sid = {int(r.sid): r for r in codes.itertuples(index=False)}
    hs6_text: dict[str, str] = {}
    cn8_text: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    for r in codes8.sort_values("cn8").itertuples(index=False):
        hs6 = r.subheading
        children.setdefault(hs6, []).append(r.cn8)
        path = [by_sid[int(s)] for s in by_sid[int(r.node_sid)].path_sids]
        head = next((i for i, p in enumerate(path) if _is_line(p, r.heading + "000000")), 1)
        sub = max((i for i, p in enumerate(path) if _is_line(p, hs6 + "0000")), default=None)
        if sub is None:  # no HS6 line in the chain: treat the heading as the HS6 line
            sub = head
        if hs6 not in hs6_text:
            hs6_text[hs6] = SEP.join(p.description for p in path[head : sub + 1])
        below = [p.description for p in path[sub + 1 :]]
        cn8_text[r.cn8] = SEP.join(below) if below else path[-1].description
    current = codes[codes["declarable"] & ~codes["superseded"]]
    code10 = current.groupby("cn8")["code10"].apply(lambda s: sorted(set(s))).to_dict()
    return Tariff(hs6_text, cn8_text, children, {c: code10.get(c, []) for c in cn8_text})


def load_tariff(root: Path = DATA_ROOT) -> Tariff:
    tariff = root / "processed" / "tariff" / NOMENCLATURE_VERSION
    paths = check_retrieval_inputs([tariff / "codes.parquet", tariff / "codes8.parquet"], root)
    codes = pd.read_parquet(paths[0], columns=["sid", "code10", "suffix", "declarable",
                                               "superseded", "description", "cn8",
                                               "path_sids"])  # fmt: skip
    return build_tariff(codes, pd.read_parquet(paths[1]))
