"""Hand-labelled test set: template and read-only checker.

Calum writes every description and label; nothing here writes descriptions. The CSV lives
outside the repo (default ~/data/hs-rag-classifier/hand/hand_v1.csv).

Usage:
    uv run python -m hs_rag_classifier.data.hand init    # header-only template, only if missing
    uv run python -m hs_rag_classifier.data.hand check   # validates; never modifies the CSV

Columns:
    id                H001, H002, ... unique and sequential
    description       terse purchase-line style, Calum's own words, 1 to 200 characters
    cn8               required, 8 digits, a current cn8 (a declarable line beneath it)
    code10            optional; a current declarable line under that cn8
    label_confidence  sure or unsure
    source_url        public catalogue page the product came from (no employer or supplier data)
    category          free text, e.g. fasteners, PPE, tools, electrical, pipe fittings
    notes             free text

check fails (exit 1) on any validation error. It also flags, without failing: exact duplicate
descriptions (normalised as for rulings), near duplicates (TF-IDF char n-gram cosine >= the
rulings threshold, 0.85) and descriptions containing a 4+ digit prefix of their own code.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from hs_rag_classifier.data.nomenclature import current_lines
from hs_rag_classifier.data.rulings import (
    NEAR_DUP_THRESHOLD,
    NOMENCLATURE_VERSION,
    code_prefixes_in,
    normalise,
    similar_pairs,
)

HEADER = [
    "id", "description", "cn8", "code10", "label_confidence", "source_url", "category", "notes",
]  # fmt: skip
CONFIDENCE = {"sure", "unsure"}
MAX_DESCRIPTION = 200
DATA_ROOT = Path.home() / "data" / "hs-rag-classifier"
DEFAULT_CSV = DATA_ROOT / "hand" / "hand_v1.csv"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Header and rows as text (codes keep leading zeros). Opens the file read-only."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [{k: (v or "").strip() for k, v in r.items() if k is not None} for r in reader]
        return list(reader.fieldnames or []), rows


def check_rows(
    header: list[str], rows: list[dict[str, str]], codes: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """(failures, flags) as human-readable lines naming row ids."""
    failures: list[str] = []
    flags: list[str] = []
    if header != HEADER:
        return [f"header must be exactly: {','.join(HEADER)} (got {','.join(header)})"], flags

    current = current_lines(codes)
    declarable = current[current["declarable"]]
    cn8_current = set(declarable["cn8"])
    code10_current = set(declarable["code10"])

    ids = [r["id"] for r in rows]
    for dup in sorted(i for i, n in Counter(ids).items() if n > 1):
        failures.append(f"{dup}: id used more than once")
    for n, r in enumerate(rows, 1):
        rid, expected = r["id"] or f"(row {n})", f"H{n:03d}"
        if r["id"] != expected:
            failures.append(f"{rid}: id out of sequence (expected {expected})")
        desc = r["description"]
        if not desc:
            failures.append(f"{rid}: description is empty")
        elif len(desc) > MAX_DESCRIPTION:
            failures.append(f"{rid}: description is {len(desc)} characters (max {MAX_DESCRIPTION})")
        cn8 = r["cn8"]
        if not re.fullmatch(r"\d{8}", cn8):
            failures.append(f"{rid}: cn8 {cn8!r} is not 8 digits")
        elif cn8 not in cn8_current:
            failures.append(f"{rid}: cn8 {cn8} is not current in {NOMENCLATURE_VERSION}")
        code10 = r["code10"]
        if code10:
            if not re.fullmatch(r"\d{10}", code10):
                failures.append(f"{rid}: code10 {code10!r} is not 10 digits")
            elif code10 not in code10_current:
                failures.append(f"{rid}: code10 {code10} is not a current declarable line")
            elif not code10.startswith(cn8):
                failures.append(f"{rid}: code10 {code10} is not under cn8 {cn8}")
        if r["label_confidence"] not in CONFIDENCE:
            failures.append(
                f"{rid}: label_confidence {r['label_confidence']!r} is not sure or unsure"
            )
        if not r["source_url"]:
            failures.append(f"{rid}: source_url is empty")
        elif not re.match(r"https?://", r["source_url"]):
            failures.append(f"{rid}: source_url is not an http(s) URL")
        code = code10 or cn8
        if desc and code and code_prefixes_in(desc, code):
            flags.append(f"{rid}: description contains its own code prefix")

    norms = [normalise(r["description"]) for r in rows]
    seen: dict[str, str] = {}
    for r, norm in zip(rows, norms, strict=True):
        if norm and norm in seen:
            flags.append(f"{r['id']}: exact duplicate of {seen[norm]}")
        elif norm:
            seen[norm] = r["id"]
    filled = [i for i, norm in enumerate(norms) if norm]
    pairs = similar_pairs([norms[i] for i in filled], min_sim=NEAR_DUP_THRESHOLD)
    for p in pairs.itertuples():
        a, b = rows[filled[int(p.i)]]["id"], rows[filled[int(p.j)]]["id"]
        if norms[filled[int(p.i)]] != norms[filled[int(p.j)]]:
            flags.append(f"{b}: near duplicate of {a} (cosine {p.sim:.2f})")
    return failures, flags


def summary(rows: list[dict[str, str]]) -> list[str]:
    chapters = Counter(r["cn8"][:2] for r in rows if r["cn8"])
    categories = Counter(r["category"] or "(none)" for r in rows)
    confidence = Counter(r["label_confidence"] or "(none)" for r in rows)
    return [
        f"rows: {len(rows)}",
        f"per chapter: {dict(sorted(chapters.items()))}",
        f"per category: {dict(sorted(categories.items()))}",
        f"label confidence: {dict(sorted(confidence.items()))}",
    ]


def cmd_init(args: argparse.Namespace) -> int:
    path: Path = args.csv
    if path.exists():
        print(f"{path} already exists; left unchanged.")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(HEADER) + "\n", encoding="utf-8")
    print(f"Wrote header-only template {path}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    path: Path = args.csv
    if not path.exists():
        print(f"{path} not found (run init first)", file=sys.stderr)
        return 1
    codes = pd.read_parquet(
        args.root / "processed" / "tariff" / NOMENCLATURE_VERSION / "codes.parquet"
    )
    header, rows = read_rows(path)
    failures, flags = check_rows(header, rows, codes)
    print(f"Checked {path}")
    for line in summary(rows):
        print(f"  {line}")
    print(f"  failures: {len(failures)}")
    for line in failures:
        print(f"    {line}")
    print(f"  flags: {len(flags)}")
    for line in flags:
        print(f"    {line}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hand", description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="hand-labelled CSV")
    parser.add_argument("--root", type=Path, default=DATA_ROOT, help="data directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="write the header-only template if missing").set_defaults(
        func=cmd_init
    )
    sub.add_parser("check", help="validate the CSV (read-only)").set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
