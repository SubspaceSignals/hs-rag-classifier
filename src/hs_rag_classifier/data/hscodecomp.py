"""HSCodeComp: an external test set of e-commerce listings with US HTS codes, scored at HS6.

Source: Hugging Face dataset ATH-MaaS/HSCodeComp (formerly AIDC-AI/HSCodeComp), Apache-2.0,
pinned at commit ce9119795acef4ca537b2175e10a3feb7a0ecae9. The file is downloaded to the data
root (hscodecomp/download/data/test_data.jsonl) and read from there at run time; it is never
copied into the repo or re-hosted. configs/hscodecomp_inputs.json (pinned by SHA-256 below)
records the file's SHA-256, the input rules and the attribute drop-list.

A test set only: never index it for retrieval and never train on it. RETRIEVAL_FORBIDDEN names
its folder, guards.FORBIDDEN_DIRS includes it, and tests/test_hscodecomp.py fails if any module
outside ALLOWED_READERS refers to it.

Labels: hs_code is stored as an integer, so it is zero-padded to 10 digits, then cut to its
first 6 (HS6) as soon as a row is read. Digits 7-10 are US-specific; they are never kept on a
record or an item.

Inputs (the rule is in the config):
    hscodecomp_c  title, then "Category: lv1 > lv2 > ..." (English levels only, a level
                  already in the path dropped), then "Attributes: key: value; ..." in the
                  file's order with blank values and drop_keys removed. Category comes before
                  attributes so input_check's 1,000-character cut trims attributes first.
    hscodecomp_a  title only.
question, price, currency_code and every *_cn field are never read into a record. The text is
used as it is (mojibake left unrepaired).

Files are parsed with json.loads only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NEVER_INDEX_FOR_RETRIEVAL = True
RAW_DIR = Path("hscodecomp")
RETRIEVAL_FORBIDDEN = (RAW_DIR.as_posix(),)
# Only these modules (paths under src/hs_rag_classifier/) may refer to the HSCodeComp data.
ALLOWED_READERS = ("data/hscodecomp.py", "guards.py", "eval/")

REPO_ROOT = Path(__file__).resolve().parents[3]
INPUTS_CONFIG = REPO_ROOT / "configs" / "hscodecomp_inputs.json"
INPUTS_CONFIG_SHA256 = "76759087cdd20dfe62e6b01c989983453df208475af15776328ebec04c539062"
VARIANTS = ("c", "a")
DATASET_NAMES = tuple(f"hscodecomp_{v}" for v in VARIANTS)
CATEGORY_LEVELS = tuple(f"cate_lv{i}_desc" for i in range(1, 6))
NEVER_USED = ("question", "price", "currency_code", "hs_code",
              *(f"{c}_cn" for c in CATEGORY_LEVELS))  # fmt: skip


class HSCodeCompError(RuntimeError):
    """The file or the inputs config is not the pinned one, or a row is malformed."""


@dataclass(frozen=True)
class Record:
    """What a row contributes; the label is already cut to HS6 (no US digits)."""

    task_id: str
    title: str
    category: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    hs6: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path: Path = INPUTS_CONFIG, sha256: str = INPUTS_CONFIG_SHA256) -> dict[str, Any]:
    got = _sha256(path)
    if got != sha256:
        raise HSCodeCompError(f"{path.name} has SHA-256 {got}, the loader pins {sha256}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def label_hs6(hs_code: Any) -> str:
    """The HS6 of a label: digits only, at most 10, zero-padded to 10, first 6 kept."""
    s = str(hs_code).strip()
    if isinstance(hs_code, bool) or not s.isdigit() or len(s) > 10:
        raise HSCodeCompError(f"hs_code {hs_code!r} is not a code of at most 10 digits")
    return s.zfill(10)[:6]


def _record(row: dict[str, Any]) -> Record:
    try:
        attrs = json.loads(row["product_attributes"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise HSCodeCompError(f"task {row.get('task_id')}: product_attributes is not JSON") from exc
    if not isinstance(attrs, dict):
        raise HSCodeCompError(f"task {row.get('task_id')}: product_attributes is not an object")
    tid = row["task_id"]
    if isinstance(tid, bool) or not str(tid).isdigit():
        raise HSCodeCompError(f"task_id {tid!r} is not a non-negative integer")
    return Record(
        task_id=str(int(tid)),
        title=str(row["product_name"]),
        category=tuple(str(row.get(k) or "") for k in CATEGORY_LEVELS),
        attributes=tuple((str(k), str(v)) for k, v in attrs.items()),
        hs6=label_hs6(row["hs_code"]),
    )


def read_records(path: Path, sha256: str, n_items: int | None = None) -> list[Record]:
    """The pinned file's rows as records; refuses a file with another SHA-256."""
    got = _sha256(path)
    if got != sha256:
        raise HSCodeCompError(f"{path} has SHA-256 {got}, expected {sha256}")
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [_record(json.loads(line)) for line in lines]
    ids = [r.task_id for r in records]
    if len(set(ids)) != len(ids):
        raise HSCodeCompError("task_id is not unique")
    if n_items is not None and len(records) != n_items:
        raise HSCodeCompError(f"{len(records)} rows, expected {n_items}")
    return records


def category_path(record: Record) -> str:
    out: list[str] = []
    for level in record.category:
        level = level.strip()
        if level and level not in out:
            out.append(level)
    return " > ".join(out)


def attributes_text(record: Record, cfg: dict[str, Any]) -> str:
    blank = set(cfg["blank_values"])
    drop = set(cfg["drop_keys"])
    drop_if = {k: v["value"] for k, v in cfg["drop_if_value"].items()}
    keep = []
    for k, v in record.attributes:
        if k in drop or v.strip() in blank or (k in drop_if and v.strip() == drop_if[k]):
            continue
        keep.append(f"{k}: {v}")
    return "; ".join(keep)


def input_c(record: Record, cfg: dict[str, Any]) -> str:
    lines = [record.title, f"Category: {category_path(record)}"]
    attrs = attributes_text(record, cfg)
    if attrs:
        lines.append(f"Attributes: {attrs}")
    return "\n".join(lines)


def input_a(record: Record, cfg: dict[str, Any] | None = None) -> str:
    return record.title


BUILDERS = {"c": input_c, "a": input_a}


def data_path(root: Path, cfg: dict[str, Any]) -> Path:
    return Path(root) / cfg["dataset"]["path"]
