"""Paths written into recorded files (summaries, rule files, index meta, MLflow params).

From Phase 4 on, a recorded path is relative to the data root (~/data/hs-rag-classifier) or the
repo root, with the root named alongside it:

    {"path": "eval/operating_point/rule/summary.json", "root": "data"}

record() writes that form; a path under neither root keeps its absolute form with root None.
resolve() reads it back and also accepts the older forms, so existing files load unchanged: a
bare absolute string, or a {"path": <absolute>} block with no root.

Paths that feed a config_hash (for example A4's adapter path in its config) are not recorded
paths in this sense and stay as they are, so every frozen config_hash is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DATA_ROOT = Path.home() / "data" / "hs-rag-classifier"
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOTS = ("data", "repo")


def _roots(data_root: Path | None, repo_root: Path | None) -> dict[str, Path]:
    return {"data": Path(data_root or DATA_ROOT).expanduser(),
            "repo": Path(repo_root or REPO_ROOT).expanduser()}  # fmt: skip


def _relative(path: Path, base: Path) -> str | None:
    for p, b in ((path, base), (path.resolve(), base.resolve())):
        try:
            return p.relative_to(b).as_posix()
        except ValueError:
            continue
    return None


def record(path: str | Path, data_root: Path | None = None,
           repo_root: Path | None = None) -> dict[str, str | None]:  # fmt: skip
    """{"path": relative POSIX path, "root": "data" | "repo"}; the data root is tried first
    (it is the more specific one when the two nest). Outside both: absolute, root None."""
    p = Path(path).expanduser()
    roots = _roots(data_root, repo_root)
    for name in ROOTS:
        rel = _relative(p, roots[name])
        if rel is not None:
            return {"path": rel, "root": name}
    return {"path": str(p), "root": None}


def resolve(value: str | Path | dict[str, Any], root: str | None = None,
            data_root: Path | None = None, repo_root: Path | None = None) -> Path:  # fmt: skip
    """The absolute path a recorded value names. value is a record() block, a bare string
    (absolute in old files, or relative with its root given as root=), or a Path."""
    if isinstance(value, dict):
        value, root = value["path"], value.get("root", root)
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    if root is None:
        raise ValueError(f"relative recorded path {value!r} without a root")
    roots = _roots(data_root, repo_root)
    if root not in roots:
        raise ValueError(f"unknown root {root!r} (expected one of {ROOTS})")
    return roots[root] / p
