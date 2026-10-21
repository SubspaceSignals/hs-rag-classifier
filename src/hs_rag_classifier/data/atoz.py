"""Build the A to Z test set from the Trade Tariff search references, and freeze it.

TEST SET ONLY: Trade Tariff search references are a test set only; never index them for
retrieval. RETRIEVAL_FORBIDDEN below names what a retrieval index builder must never read, and
tests/test_atoz.py fails if any module outside the allow-list refers to them.

Source: GOV.UK Trade Tariff API, GET /search_references (Open Government Licence v3.0); one
unpaginated response holding every A to Z term. Fetched through TariffApi (Developer Portal
token, 1 request a second) and cached at raw/atoz/search_references.json, a folder of its own
so nothing that walks the tariff API cache can pick it up. The website's
A to Z pages are disallowed to robots and are never scraped.

Usage:
    uv run python -m hs_rag_classifier.data.atoz build [--refreeze]

Outputs:
    processed/atoz/v1/atoz_all.parquet    every reference, with target depth and status
    processed/atoz/v1/atoz_test8.parquet  the test set: current targets at >= 8 digits
    processed/atoz/v1/manifest.json
    splits/atoz_v1/test_refs.txt, SPLIT.md   in the repo: sorted reference ids only, no text

Rules:
    target        the line named by goods_nomenclature_sid among current (non-superseded)
                  lines when its code and suffix match the reference's; else the current line
                  with the same code and suffix; else the sid's line even though it differs;
                  not_current if none exists. sid_mismatch flags references whose sid names a
                  line with another code or suffix: about 97 headings carry the sid of the TARIC
                  sub-chapter title above them (e.g. "second-hand clothes": heading 6309, suffix
                  80, sid of 6309000000/10 "III. WORN CLOTHING ...").
    target_depth  10 for a declarable line; otherwise the code's significant digits rounded up
                  to 2, 4, 6, 8 or 10 (chapter 2, heading 4; a few non-declarable grouping lines
                  carry 10 significant digits).
    cn8           the target's cn8 when it is declarable, or when every current declarable line
                  beneath it shares one cn8 (cn8_unambiguous).
    label         label_depth 10 (the code10) for a declarable target; label_depth 8 (the cn8)
                  for any other current target with an unambiguous cn8. Input is the title (the
                  A to Z term).
    label_source  "direct": the reference points at the 8- or 10-digit line itself.
                  "single_cn8_below": the reference points above 8 digits (a heading or a
                  6-digit subheading) and its label is the only cn8 beneath it.
    conflicts     a term's references conflict when any two disagree at the shallower of their
                  two scored depths (different cn8s, or different code10s when both are scored
                  at 10). Conflicting terms are genuinely ambiguous and are left out of test8,
                  so scores exclude ambiguous terms.
    duplicates    a non-conflicting term is one row, scored at the shallowest depth among its
                  direct references (all its references if none is direct): a single_cn8_below
                  reference confirms a direct label but never coarsens it. The lowest id at that
                  depth represents the term and member_ids lists every id.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from hs_rag_classifier.data.nomenclature import OGL_ATTRIBUTION, TariffApi, current_lines, sha256
from hs_rag_classifier.data.rulings import NOMENCLATURE_VERSION, normalise
from hs_rag_classifier.guards import FORBIDDEN_DIRS

SPLIT_NAME = "v1"
ENDPOINT = "/search_references"
DATA_ROOT = Path.home() / "data" / "hs-rag-classifier"
REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = REPO_ROOT / "splits" / f"atoz_{SPLIT_NAME}"

# A retrieval index builder must never read these (paths relative to the data root, and the
# API endpoint). See the module docstring and CLAUDE.md.
NEVER_INDEX_FOR_RETRIEVAL = True
RAW_DIR = Path("raw") / "atoz"
RETRIEVAL_FORBIDDEN = (*FORBIDDEN_DIRS, ENDPOINT)  # raw/atoz, processed/atoz, the endpoint
# Only these modules (paths under src/hs_rag_classifier/) may refer to the A to Z data: this
# builder and the evaluation harness. tests/test_atoz.py enforces it.
ALLOWED_READERS = ("data/atoz.py", "guards.py", "eval/")

TEST8_COLUMNS = [
    "id", "member_ids", "title", "negated_title", "referenced_class", "target_code10",
    "target_suffix", "target_sid", "target_depth", "cn8", "code10", "label_depth",
    "label_source",
]  # fmt: skip


def parse_search_references(payload: dict) -> pd.DataFrame:
    rows = []
    for item in payload["data"]:
        a = item["attributes"]
        rows.append(
            {
                "id": str(item["id"]),
                "title": a.get("title") or "",
                "negated_title": a.get("negated_title") or "",
                "referenced_class": a.get("referenced_class"),
                "referenced_id": str(a.get("referenced_id")),
                "target_code10": str(a.get("goods_nomenclature_item_id")).zfill(10),
                "target_suffix": str(a.get("productline_suffix") or "80"),
                "target_sid": a.get("goods_nomenclature_sid"),
            }
        )
    df = pd.DataFrame(rows)
    df["target_sid"] = pd.to_numeric(df["target_sid"]).astype("Int64")
    return df


def significant_depth(code10: str) -> int:
    digits = len(code10.rstrip("0"))
    return min(10, max(2, digits + digits % 2))


def resolve(refs: pd.DataFrame, codes: pd.DataFrame) -> pd.DataFrame:
    """Add target status, depth, cn8 and label_depth (rules in the module docstring)."""
    current = current_lines(codes)
    by_sid = current.set_index("sid")
    by_code = current.drop_duplicates(["code10", "suffix"]).set_index(["code10", "suffix"])["sid"]
    declarable = current[current["declarable"]]
    cn8s_beneath: dict[int, set[str]] = {}
    for path, cn8 in zip(declarable["path_sids"], declarable["cn8"], strict=True):
        for sid in path:
            cn8s_beneath.setdefault(int(sid), set()).add(cn8)

    out = []
    for r in refs.itertuples(index=False):
        sid, mismatch = None, False
        named = int(r.target_sid) if pd.notna(r.target_sid) else None
        if named is not None and named in by_sid.index:
            line = by_sid.loc[named]
            mismatch = (line["code10"], line["suffix"]) != (r.target_code10, r.target_suffix)
            sid = named
        if (sid is None or mismatch) and (r.target_code10, r.target_suffix) in by_code.index:
            sid = int(by_code[(r.target_code10, r.target_suffix)])
        row = {
            "resolved_sid": sid,
            "sid_mismatch": mismatch,
            "status": "current" if sid is not None else "not_current",
        }
        if sid is None:
            row.update(target_depth=significant_depth(r.target_code10), cn8=None, code10=None,
                       cn8_unambiguous=False, label_depth=None, label_source=None)  # fmt: skip
        else:
            line = by_sid.loc[sid]
            beneath = cn8s_beneath.get(sid, set())
            single = len(beneath) == 1
            if line["declarable"]:
                depth, label, code10, source = 10, 10, line["code10"], "direct"
            else:
                depth, code10 = significant_depth(line["code10"]), None
                label = 8 if single else None
                source = None if not single else "direct" if depth >= 8 else "single_cn8_below"
            row.update(
                target_depth=depth,
                cn8=next(iter(beneath)) if single else None,
                code10=code10,
                cn8_unambiguous=single,
                label_depth=label,
                label_source=source,
            )
        out.append(row)
    df = pd.concat([refs.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    for col in ("resolved_sid", "label_depth"):
        df[col] = df[col].astype("Int64")
    df["norm_title"] = df["title"].map(normalise)
    return df


def select_test8(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(test rows, conflicting candidates); one row per normalised term.

    Conflict: any two of a term's references disagree at the shallower of their two scored
    depths (different cn8s, or different code10s between two rows both scored at 10).
    A non-conflicting term becomes one row. Its scored depth is the shallowest depth among its
    direct references (all its references if none is direct), so a single_cn8_below reference
    confirms a direct label but never coarsens it. The representative is the lowest id at that
    depth (direct first); member_ids lists every id.
    """
    cand = resolved[resolved["label_depth"].notna()].copy()
    cand["_id"] = cand["id"].astype(int)
    by_term = cand.groupby("norm_title")
    tens = cand["code10"].where(cand["label_depth"] == 10)
    conflicted = (by_term["cn8"].transform("nunique") > 1) | (
        tens.groupby(cand["norm_title"]).transform("nunique") > 1
    )
    conflicts = cand[conflicted].drop(columns=["_id"])
    keep = cand[~conflicted].copy()
    keep["_direct"] = keep["label_source"] == "direct"
    keep["_pref"] = keep["_direct"] | ~keep.groupby("norm_title")["_direct"].transform("any")
    keep["_depth"] = keep["label_depth"].where(keep["_pref"])
    keep["_depth"] = keep.groupby("norm_title")["_depth"].transform("min")
    keep = keep.sort_values("_id")
    members = keep.groupby("norm_title")["id"].agg(list).rename("member_ids")
    reps = keep[keep["_pref"] & (keep["label_depth"] == keep["_depth"])]
    test = reps.drop_duplicates("norm_title").join(members, on="norm_title")
    return test.sort_values("_id").reset_index(drop=True)[TEST8_COLUMNS], conflicts


def write_ref_files(test: pd.DataFrame, splits_dir: Path, refreeze: bool = False) -> Path:
    """Write sorted reference ids; refuses to change an existing test_refs.txt unless refreeze."""
    ids = sorted((i for m in test["member_ids"] for i in m), key=int)
    text = "\n".join(ids) + "\n"
    path = splits_dir / "test_refs.txt"
    if path.exists() and path.read_text(encoding="utf-8") != text and not refreeze:
        raise RuntimeError(
            f"{path} is frozen and this build would change it; refusing to overwrite "
            "(pass --refreeze only after deciding to replace the frozen test set)."
        )
    splits_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def ruling_overlap(test_titles: pd.Series, rulings_dir: Path) -> dict[str, int]:
    """How many normalised A to Z titles exactly equal a normalised ruling description."""
    titles = set(test_titles)
    out = {}
    for s in ("train", "val", "test"):
        path = rulings_dir / f"{s}.parquet"
        if path.exists():
            descs = set(
                pd.read_parquet(path, columns=["description"])["description"].map(normalise)
            )
            out[s] = len(titles & descs)
    return out


def split_md(counts: dict, sha: str) -> str:
    return "\n".join(
        [
            f"# A to Z test set {SPLIT_NAME} (frozen)",
            "",
            (
                "Built by `uv run python -m hs_rag_classifier.data.atoz build` from the GOV.UK "
                f"Trade Tariff API `{ENDPOINT}` (Open Government Licence v3.0), resolved against "
                f"DBT uk-tariff-2021-01-01 {NOMENCLATURE_VERSION}. This file and test_refs.txt "
                "hold search reference ids only; no terms are stored in the repo."
            ),
            "",
            f"- test_refs.txt SHA-256: `{sha}`",
            f"- References collected: {counts['references']:,}",
            (
                f"- Test rows: {counts['test_rows']:,} ({counts['test_ids']:,} reference ids; "
                f"{counts['test_label_10']:,} scored to 10 digits, {counts['test_label_8']:,} to 8)"
            ),
            (
                f"- By label_source: direct {counts['test_direct']:,}, "
                f"single_cn8_below {counts['test_single_cn8_below']:,}"
            ),
            (
                f"- Conflicting terms excluded: {counts['conflict_titles']} "
                f"({counts['conflict_references']} references)"
            ),
            "",
            "## Rules",
            "",
            "- Test set only: search references are never indexed for retrieval.",
            (
                "- Input is the A to Z term (title). The label is the 10-digit code when the term "
                "points at a declarable line, otherwise the cn8 when exactly one cn8 sits beneath "
                "the line it points at."
            ),
            (
                "- `label_source` in atoz_test8: `direct` when the term points at the 8- or "
                "10-digit line itself; `single_cn8_below` when it points at a heading or 6-digit "
                "subheading and the label is the only cn8 beneath it."
            ),
            (
                "- Conflicting terms are excluded: a term is dropped when any two of its "
                'references disagree at the shallower of their scored depths (e.g. "shower '
                'gel" under 3304 and 3401). Scores on this set therefore exclude genuinely '
                "ambiguous terms."
            ),
            (
                "- Each remaining term counts once, scored at the shallowest depth among its "
                "direct references; a `single_cn8_below` reference may confirm a direct label "
                "but never coarsens it. Every member id is listed in test_refs.txt."
            ),
            (
                "- Terms pointing at lines that are no longer current, or with several cn8s "
                "beneath, are left out."
            ),
            (
                "- A reference whose sid names a line with another code or suffix (some headings "
                "carry the sid of the TARIC sub-chapter title above them) is resolved by its code "
                "and suffix; DBT parents those headings to the title too, and the nomenclature "
                "build repairs them to their chapter."
            ),
            "",
            "`tests/test_frozen_split.py` fails if test_refs.txt no longer matches the hash above.",
            "",
        ]
    )


def cmd_build(args: argparse.Namespace) -> int:
    root: Path = args.root
    codes = pd.read_parquet(root / "processed" / "tariff" / NOMENCLATURE_VERSION / "codes.parquet")
    api = TariffApi(root / RAW_DIR, delay_s=args.delay)
    print(f"Trade Tariff API: {api.base_url} ({'token' if api.authenticated else 'no auth'})")
    try:
        payload = api.get_json(ENDPOINT)
    finally:
        api.close()
    raw_path = root / RAW_DIR / "search_references.json"
    fetched_at = datetime.fromtimestamp(raw_path.stat().st_mtime, UTC).isoformat(timespec="seconds")
    print(f"{ENDPOINT}: {'fetched' if api.fetched else 'from cache'} ({fetched_at})")

    refs = parse_search_references(payload)
    refs["fetched_at"] = fetched_at
    resolved = resolve(refs, codes)
    test, conflicts = select_test8(resolved)

    ref_path = write_ref_files(test, args.splits_dir, refreeze=args.refreeze)  # guard first
    test_sha = sha256(ref_path)
    out = root / "processed" / "atoz" / SPLIT_NAME
    out.mkdir(parents=True, exist_ok=True)
    resolved.drop(columns=["norm_title"]).to_parquet(out / "atoz_all.parquet", index=False)
    test.to_parquet(out / "atoz_test8.parquet", index=False)

    current = resolved[resolved["status"] == "current"]
    counts = {
        "references": len(resolved),
        "by_class": resolved["referenced_class"].value_counts().to_dict(),
        "by_status": resolved["status"].value_counts().to_dict(),
        "current_by_target_depth": current["target_depth"].value_counts().sort_index().to_dict(),
        "current_declarable": int(current["code10"].notna().sum()),
        "current_label8_nodes": int(
            ((current["label_depth"] == 8) & (current["label_source"] == "direct")).sum()
        ),
        "single_cn8_below_references": int((current["label_source"] == "single_cn8_below").sum()),
        "usable_ge8_references": int(resolved["label_depth"].notna().sum()),
        "conflict_references": len(conflicts),
        "conflict_titles": int(conflicts["norm_title"].nunique()),
        "test_rows": len(test),
        "test_ids": int(test["member_ids"].map(len).sum()),
        "test_label_10": int((test["label_depth"] == 10).sum()),
        "test_label_8": int((test["label_depth"] == 8).sum()),
        "test_direct": int((test["label_source"] == "direct").sum()),
        "test_single_cn8_below": int((test["label_source"] == "single_cn8_below").sum()),
        "test_chapters": int(test["cn8"].str[:2].nunique()),
    }
    overlap = ruling_overlap(test["title"].map(normalise), root / "processed" / "rulings" / "v1")
    (args.splits_dir / "SPLIT.md").write_text(split_md(counts, test_sha), encoding="utf-8")

    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "endpoint": api.base_url + ENDPOINT,
        "fetched_at": fetched_at,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "counts": counts,
        "ruling_description_overlap": overlap,
        "test_refs_sha256": test_sha,
        "sha256": {
            str(RAW_DIR / "search_references.json"): sha256(raw_path),
            "atoz_all.parquet": sha256(out / "atoz_all.parquet"),
            "atoz_test8.parquet": sha256(out / "atoz_test8.parquet"),
        },
        "never_index_for_retrieval": list(RETRIEVAL_FORBIDDEN),
        "attribution": OGL_ATTRIBUTION,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print("\nA to Z report")
    print(f"  references: {counts['references']:,}; by class {counts['by_class']}")
    print(f"  status: {counts['by_status']}")
    print(f"  current targets by depth: {counts['current_by_target_depth']}")
    print(f"  labelled references: {counts['usable_ge8_references']:,} "
          f"({counts['current_declarable']:,} declarable, "
          f"{counts['current_label8_nodes']} non-declarable 8/10-digit nodes with one cn8, "
          f"{counts['single_cn8_below_references']} single_cn8_below)")  # fmt: skip
    print(f"  conflicts (same term, different labels): {counts['conflict_titles']} terms, "
          f"{counts['conflict_references']} references")  # fmt: skip
    print(f"  test8: {counts['test_rows']:,} rows ({counts['test_ids']:,} ids), "
          f"label 10: {counts['test_label_10']:,}, label 8: {counts['test_label_8']:,}, "
          f"direct {counts['test_direct']:,}, single_cn8_below {counts['test_single_cn8_below']:,}, "
          f"{counts['test_chapters']} chapters")  # fmt: skip
    print(f"  exact matches with ruling descriptions: {overlap}")
    print(f"  test_refs.txt SHA-256: {test_sha}")
    print(f"\nWrote {out} and {args.splits_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atoz", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="collect, resolve, write and freeze the A to Z set")
    build.add_argument("--root", type=Path, default=DATA_ROOT, help="data directory")
    build.add_argument("--splits-dir", type=Path, default=SPLITS_DIR, help="repo split files")
    build.add_argument("--delay", type=float, default=1.0, help="seconds per API request")
    build.add_argument(
        "--refreeze", action="store_true", help="allow replacing a frozen test_refs.txt"
    )
    build.set_defaults(func=cmd_build)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
