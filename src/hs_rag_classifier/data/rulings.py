"""Clean, check and split the HMRC Advance Tariff Rulings, and freeze the test split.

Input (outside the repo, under ~/data/hs-rag-classifier):
    raw/atar/rulings.jsonl                       from hs_rag_classifier.data.atar collect
    processed/tariff/{NOMENCLATURE_VERSION}/codes.parquet, codes8.parquet

Usage:
    uv run python -m hs_rag_classifier.data.rulings build

Outputs:
    processed/rulings/v1/{train,val,test}.parquet   one row per de-duplicated ruling, no
                                                     justification column
    processed/rulings/v1/justifications.parquet     (ref, justification) for every ruling
    processed/rulings/v1/manifest.json
    splits/rulings_v1/{train,val,test}_refs.txt     in the repo: every member reference, sorted
    splits/rulings_v1/SPLIT.md                       rules, counts and the test_refs.txt SHA-256

Rules:
    normalise      lowercase, punctuation to spaces, whitespace collapsed.
    exact dups     same normalised description and same code10 form one group; the lowest
                   reference represents it and member_refs lists all of them.
    conflict       one normalised description ruled under different code10s: every group
                   involved is train-only.
    near dups      TF-IDF character n-grams (char_wb, 3-5) over the representatives' normalised
                   descriptions; connected components of pairs with cosine >= 0.85 form a cluster.
                   Clusters are kept, never removed, and never span two splits. cluster_id is the
                   lowest reference in the cluster.
    self leak      the description contains a 4+ digit number (digit groups may be split by
                   single spaces or dots) that is a prefix of the ruled code: train-only.
    keyword audit  reported only: (a) keywords hold such a number; (b) keyword words (3+
                   letters, stopwords dropped, a trailing plural "s" removed) that appear in the
                   code's description path but not in the ruling's description.
    codes          8- and 6-digit codes are padded with zeros to 10 digits (flag short_code).
                   current      padded code is a current declarable line in the nomenclature
                   cn8_current  an 8-digit code whose cn8 is current but whose padded code is
                                not declarable (it has 10-digit children)
                   retired      not current, but declarable on the ruling's start date by the
                                Trade Tariff API (?as_of=YYYY-MM-DD). When the old line still
                                exists with exactly one current declarable line beneath it,
                                code10 is mapped to that line (flag mapped_from)
                   never_valid  not declarable on the start date either
    label_depth    10 for current codes and cleanly mapped retired ones. 8 for any other ruling
                   whose cn8 is current (cn8_current, and retired or never_valid with a current
                   cn8): train-only, and scored to 8 digits only. Rulings with no current cn8
                   and no mapping are excluded from every split.
    model input    the description only. Description + keywords is a secondary ablation on the
                   same frozen test set, never default input; keywords may be used in training.
    split          eligible for val/test: current, no conflict, no self leak, and every group in
                   the cluster eligible. Clusters are shuffled with the seed, taken largest
                   first, and each goes to the split furthest below its per-chapter target,
                   until test and val reach their sizes. Everything else goes to train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from hs_rag_classifier.data.nomenclature import OGL_ATTRIBUTION, TariffApi, current_lines, sha256

SEED = 20261020
NOMENCLATURE_VERSION = "v4.0.1633"
SPLIT_NAME = "v1"
NEAR_DUP_THRESHOLD = 0.85
REPORT_THRESHOLDS = (0.85, 0.90, 0.95)
MIN_PAIR_SIM = 0.80
N_TEST, N_VAL = 1500, 750
AS_OF_PARAM = "as_of"

DATA_ROOT = Path.home() / "data" / "hs-rag-classifier"
REPO_ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = REPO_ROOT / "splits" / f"rulings_{SPLIT_NAME}"
SPLITS = ("train", "val", "test")
OUTPUT_COLUMNS = [
    "ref", "member_refs", "description", "keywords", "code10", "cn8", "hs6", "chapter",
    "start_date", "expiry_date", "cluster_id", "code_status", "label_depth", "flags",
]  # fmt: skip

_STOPWORDS_TEXT = (
    "and the for with other not than from into but whether including excluding its their "
    "kind used thereof such any all are has have being more less this that which only "
    "or nor per like also made parts part"
)
STOPWORDS = set(_STOPWORDS_TEXT.split())

_PUNCT_RE = re.compile(r"[^\w\s]|_")
_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+(?:[ .]\d+)*")
_WORD_RE = re.compile(r"[a-z]{3,}")


# --- Text -----------------------------------------------------------------------------------


def normalise(text: str) -> str:
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", (text or "").lower())).strip()


def code_prefixes_in(text: str, code: str) -> list[str]:
    """4+ digit numbers in text that are a prefix of code (digit groups joined over ' ' or '.')."""
    found = []
    for match in _NUMBER_RE.finditer(text or ""):
        runs = re.split(r"[ .]", match.group())
        for candidate in ["".join(runs), *runs]:
            if len(candidate) >= 4 and code.startswith(candidate) and candidate not in found:
                found.append(candidate)
    return found


def word_tokens(text: str) -> set[str]:
    words = set()
    for w in _WORD_RE.findall((text or "").lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.add(w)
    return words


def leakage_flags(description: str, keywords: list[str], code: str, path_text: str | None) -> dict:
    kw_text = " ; ".join(keywords)
    path_words = (word_tokens(kw_text) & word_tokens(path_text)) - word_tokens(description)
    return {
        "kw_code_prefix": bool(code_prefixes_in(kw_text, code)),
        "kw_path_words": sorted(path_words) if path_text else [],
        "self_leak": bool(code_prefixes_in(description, code)),
    }


# --- Loading and grouping -------------------------------------------------------------------


def load_rulings(path: Path) -> pd.DataFrame:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    df = pd.DataFrame(records)
    df["reference"] = df["reference"].astype(str)
    df["code_raw"] = df["commodity_code"].astype(str)
    df["code10"] = df["code_raw"].str.ljust(10, "0")
    df["keywords"] = df["keywords"].map(lambda k: list(k or []))
    df["norm"] = df["description"].map(normalise)
    return df


def group_exact(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (normalised description, code10); the representative is the lowest ref."""
    df = df.assign(_ref_int=df["reference"].astype(int)).sort_values("_ref_int")
    groups = (
        df.groupby(["norm", "code10"], sort=False)
        .agg(ref=("reference", "first"), member_refs=("reference", list))
        .reset_index()
    )
    rep = df.set_index("reference")
    for col in ("description", "keywords", "code_raw", "start_date", "expiry_date"):
        groups[col] = groups["ref"].map(rep[col])
    groups["conflict"] = groups.groupby("norm")["code10"].transform("nunique") > 1
    groups = groups.assign(_r=groups["ref"].astype(int)).sort_values("_r").drop(columns="_r")
    return groups.reset_index(drop=True)


def similar_pairs(
    texts: list[str], min_sim: float = MIN_PAIR_SIM, chunk: int = 500
) -> pd.DataFrame:
    """Pairs (i < j) with TF-IDF char n-gram cosine >= min_sim."""
    if len(texts) < 2:
        return pd.DataFrame({"i": [], "j": [], "sim": []})
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    x = vec.fit_transform(texts)  # rows are L2-normalised, so the dot product is the cosine
    rows, cols, sims = [], [], []
    for start in range(0, x.shape[0], chunk):
        block = (x[start : start + chunk] @ x.T).toarray()
        i, j = np.nonzero(block >= min_sim)
        i = i + start
        keep = i < j
        rows.append(i[keep])
        cols.append(j[keep])
        sims.append(block[i[keep] - start, j[keep]])
    return pd.DataFrame(
        {"i": np.concatenate(rows), "j": np.concatenate(cols), "sim": np.concatenate(sims)}
    )


def components(n: int, pairs: pd.DataFrame, threshold: float) -> np.ndarray:
    """Connected-component label per item, linking pairs with sim >= threshold."""
    edges = pairs[pairs["sim"] >= threshold]
    graph = coo_matrix(
        (np.ones(len(edges)), (edges["i"].to_numpy(int), edges["j"].to_numpy(int))), shape=(n, n)
    )
    return connected_components(graph, directed=False)[1]


# --- Code status ----------------------------------------------------------------------------


def code_status(
    code_raw: str,
    start: str,
    current: pd.DataFrame,
    cn8_current: set[str],
    valid_on,
    declarable_codes: set[str] | None = None,
) -> tuple[str, str | None, int | None]:
    """(status, mapped code10 or None, label_depth: 10, 8, or None when excluded).

    current: current (non-superseded) lines; cn8_current: cn8s with a current declarable line;
    valid_on(code10, start) asks the API; declarable_codes: optional precomputed set of current
    declarable code10s (for speed).
    """
    code10 = code_raw.ljust(10, "0")
    declarable = current[current["declarable"]]
    if declarable_codes is None:
        declarable_codes = set(declarable["code10"])
    if code10 in declarable_codes:
        return "current", None, 10
    depth_8 = 8 if code10[:8] in cn8_current else None
    if len(code_raw) == 8 and depth_8:
        return "cn8_current", None, 8
    if not valid_on(code10, start):
        return "never_valid", None, depth_8
    old_sids = set(current.loc[current["code10"] == code10, "sid"])
    if old_sids:
        beneath = declarable[declarable["path_sids"].map(lambda p: bool(old_sids & set(p)))]
        codes = sorted(set(beneath["code10"]))
        if len(codes) == 1:
            return "retired", codes[0], 10
    return "retired", None, depth_8


def api_valid_on(api: TariffApi):
    def valid_on(code10: str, start: str) -> bool:
        data = api.get_json(f"/commodities/{code10}", params={AS_OF_PARAM: start})
        return bool(data and data.get("data", {}).get("attributes", {}).get("declarable"))

    return valid_on


# --- Split ----------------------------------------------------------------------------------


def assign_splits(units: pd.DataFrame, n_test: int, n_val: int, seed: int) -> dict[str, str]:
    """cluster_id -> split for eligible clusters (columns cluster_id, chapter, size).

    Greedy, chapter-stratified: each cluster goes to the split furthest below its target for
    that chapter, without overfilling test or val; singletons then top test and val up exactly.
    """
    total = int(units["size"].sum())
    if total < n_test + n_val:
        raise ValueError(f"Only {total} eligible rulings for {n_test} test + {n_val} val")
    share = {"test": n_test / total, "val": n_val / total}
    share["train"] = 1 - share["test"] - share["val"]
    cap = {"test": n_test, "val": n_val, "train": float("inf")}
    per_chapter = units.groupby("chapter")["size"].sum()
    target = {s: (per_chapter * share[s]).to_dict() for s in SPLITS}
    have = {s: Counter() for s in SPLITS}
    filled = Counter()

    rng = random.Random(seed)
    order = list(units.itertuples(index=False))
    rng.shuffle(order)
    order.sort(key=lambda u: -u.size)  # stable: random order within each size
    out: dict[str, str] = {}
    for u in order:
        options = [s for s in ("test", "val", "train") if filled[s] + u.size <= cap[s]]
        best = max(options, key=lambda s: target[s][u.chapter] - have[s][u.chapter])
        out[u.cluster_id] = best
        have[best][u.chapter] += u.size
        filled[best] += u.size

    for s in ("test", "val"):  # top up from train singletons where the chapter is most short
        while filled[s] < cap[s]:
            pool = [u for u in order if out[u.cluster_id] == "train" and u.size == 1]
            if not pool:
                raise RuntimeError(f"Cannot fill {s} exactly")
            u = max(pool, key=lambda u: target[s][u.chapter] - have[s][u.chapter])
            out[u.cluster_id] = s
            have[s][u.chapter] += 1
            have["train"][u.chapter] -= 1
            filled[s] += 1
            filled["train"] -= 1
    return out


# --- Build ----------------------------------------------------------------------------------


def _short(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _mode(values) -> str:
    """Most common value; ties go to the lowest."""
    return min(Counter(values).items(), key=lambda kv: (-kv[1], kv[0]))[0]


def prepare(
    rulings: pd.DataFrame, codes: pd.DataFrame, cn8_current: set[str], valid_on
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (groups with status, cluster and split; per-ruling audit; similar pairs)."""
    current = current_lines(codes)
    path_by_code = (
        current[current["declarable"]].drop_duplicates("code10").set_index("code10")["path_text"]
    )
    audit = rulings[["reference", "description", "keywords", "code_raw", "code10"]].copy()
    flags = [
        leakage_flags(r.description, r.keywords, r.code_raw, path_by_code.get(r.code10))
        for r in audit.itertuples()
    ]
    audit = pd.concat([audit, pd.DataFrame(flags, index=audit.index)], axis=1)
    audit["has_path"] = audit["code10"].isin(path_by_code.index)
    self_leak_refs = set(audit.loc[audit["self_leak"], "reference"])

    groups = group_exact(rulings)
    groups["self_leak"] = groups["member_refs"].map(lambda m: bool(self_leak_refs & set(m)))

    pairs = similar_pairs(list(groups["norm"]))
    labels = components(len(groups), pairs, NEAR_DUP_THRESHOLD)
    groups["_label"] = labels
    groups["cluster_id"] = groups.groupby("_label")["ref"].transform(lambda r: min(r, key=int))

    decl_codes = set(current.loc[current["declarable"], "code10"])
    statuses = [
        code_status(r.code_raw, r.start_date, current, cn8_current, valid_on, decl_codes)
        for r in groups.itertuples()
    ]
    groups["code_status"] = [s for s, _, _ in statuses]
    groups["mapped_code10"] = [m for _, m, _ in statuses]
    groups["label_depth"] = pd.array([d for _, _, d in statuses], dtype="Int64")
    groups["excluded"] = groups["label_depth"].isna()
    groups["eligible"] = (
        (groups["code_status"] == "current")
        & (groups["label_depth"] == 10)
        & ~groups["conflict"]
        & ~groups["self_leak"]
    )

    kept = groups[~groups["excluded"]].copy()
    kept["chapter"] = kept["code10"].str[:2]
    clusters = (
        kept.groupby("cluster_id")
        .agg(size=("ref", "size"), eligible=("eligible", "all"), chapter=("chapter", _mode))
        .reset_index()
    )
    assigned = assign_splits(clusters[clusters["eligible"]], N_TEST, N_VAL, SEED)
    groups["split"] = groups["cluster_id"].map(assigned).fillna("train")
    groups.loc[groups["excluded"], "split"] = "excluded"

    audit_by_ref = audit.set_index("reference")

    def row_flags(r) -> list[str]:
        out = []
        if len(r.code_raw) < 10:
            out.append(f"short_code:{r.code_raw}")
        if r.conflict:
            out.append("conflict")
        if r.self_leak:
            out.append("self_leak")
        if isinstance(r.mapped_code10, str):
            out.append(f"mapped_from:{r.code10}")
        a = audit_by_ref.loc[r.ref]
        if a["kw_code_prefix"]:
            out.append("kw_code_prefix")
        if len(a["kw_path_words"]):
            out.append("kw_path_words")
        return out

    groups["flags"] = [row_flags(r) for r in groups.itertuples()]
    mapped = groups["mapped_code10"].notna()
    groups.loc[mapped, "code10"] = groups.loc[mapped, "mapped_code10"]
    groups["cn8"] = groups["code10"].str[:8]
    groups["hs6"] = groups["code10"].str[:6]
    groups["chapter"] = groups["code10"].str[:2]
    for col in ("start_date", "expiry_date"):
        groups[col] = pd.to_datetime(groups[col]).dt.date
    return groups, audit, pairs


def _sha_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_split_files(groups: pd.DataFrame, rulings: pd.DataFrame, out: Path) -> None:
    """Split Parquet files (fixed columns, no justification) plus justifications apart."""
    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        part = groups.loc[groups["split"] == s, OUTPUT_COLUMNS]
        if "justification" in part.columns:
            raise AssertionError("justification must never be in a split file")
        part.to_parquet(out / f"{s}.parquet", index=False)
    rulings[["reference", "justification"]].rename(columns={"reference": "ref"}).to_parquet(
        out / "justifications.parquet", index=False
    )


def write_ref_files(
    groups: pd.DataFrame, splits_dir: Path, refreeze: bool = False
) -> dict[str, list[str]]:
    """Write sorted member references per split.

    Refuses to change an existing test_refs.txt unless refreeze=True (an explicit decision).
    """
    refs = {
        s: sorted((r for m in groups.loc[groups["split"] == s, "member_refs"] for r in m), key=int)
        for s in SPLITS
    }
    test_path = splits_dir / "test_refs.txt"
    new_test = "\n".join(refs["test"]) + "\n"
    changed = test_path.exists() and test_path.read_text(encoding="utf-8") != new_test
    if changed and not refreeze:
        raise RuntimeError(
            f"{test_path} is frozen and this build would change it; refusing to overwrite "
            "(pass --refreeze only after deciding to replace the frozen test split)."
        )
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        (splits_dir / f"{s}_refs.txt").write_text("\n".join(refs[s]) + "\n", encoding="utf-8")
    return refs


def split_md(stats: dict, test_sha: str) -> str:
    c = stats["counts"]
    lines = [
        f"# HMRC rulings split {SPLIT_NAME} (frozen)",
        "",
        (
            "Built by `uv run python -m hs_rag_classifier.data.rulings build` from the HMRC "
            "Advance Tariff Rulings collected on 20 Oct 2026 (Open Government Licence v3.0). "
            "These files hold references only; no ruling text is stored in the repo."
        ),
        "",
        f"- Seed: {SEED}",
        f"- Nomenclature: DBT uk-tariff-2021-01-01 {NOMENCLATURE_VERSION} (current lines only)",
        f"- Near-duplicate threshold: TF-IDF char_wb 3-5 cosine >= {NEAR_DUP_THRESHOLD}",
        f"- Historical code check: Trade Tariff API `/commodities/{{code}}?{AS_OF_PARAM}=<start>`",
        f"- test_refs.txt SHA-256: `{test_sha}`",
        "",
        "## Counts",
        "",
        "| Split | De-duplicated rulings | Member references | Chapters | 8-digit labels |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in SPLITS:
        lines.append(
            f"| {s} | {c[s]['rows']:,} | {c[s]['refs']:,} | {c[s]['chapters']} "
            f"| {c[s]['label_depth_8']:,} |"
        )
    lines += [
        "",
        (
            f"Collected {c['input_rulings']:,} rulings; {c['groups']:,} after exact "
            f"de-duplication; {c['excluded_rows']} groups ({c['excluded_refs']} references) "
            "excluded because their code is not current, has no clean mapping and its cn8 is "
            "not current either."
        ),
        "",
        "## Rules",
        "",
        (
            "- Exact duplicates (same normalised description and code) form one row; the lowest "
            "reference represents them, and every member reference is listed in the split files."
        ),
        "- A near-duplicate cluster never spans two splits.",
        (
            "- Validation and test hold only rulings whose code is a current declarable line, "
            "whose description isn't also ruled under another code (conflict) and whose "
            "description doesn't contain a 4+ digit prefix of its own code."
        ),
        (
            "- Retired codes stay in train only with a clean 1:1 mapping to a current code; "
            "codes valid at 8 digits only stay in train; codes never valid at the start date "
            "are excluded."
        ),
        "- Test and validation are stratified by chapter as far as clusters allow.",
        (
            "- label_depth is 10, or 8 for train-only rulings whose 10-digit code is not a "
            "current declarable line but whose cn8 is current; those are scored to 8 digits "
            "only. Rulings with no current cn8 are excluded."
        ),
        (
            "- Model input is the description only. Description + keywords is a secondary "
            "ablation on this same test set; keywords carry tariff wording in about 64% of "
            "rulings."
        ),
        "- Justifications are stored apart from the split files and are never model input.",
        "",
        "`tests/test_frozen_split.py` fails if test_refs.txt no longer matches the hash above.",
        "",
    ]
    return "\n".join(lines)


def cmd_build(args: argparse.Namespace) -> int:
    root: Path = args.root
    rulings_path = root / "raw" / "atar" / "rulings.jsonl"
    tariff = root / "processed" / "tariff" / NOMENCLATURE_VERSION
    rulings = load_rulings(rulings_path)
    codes = pd.read_parquet(tariff / "codes.parquet")
    cn8_current = set(pd.read_parquet(tariff / "codes8.parquet")["cn8"])
    print(f"{len(rulings):,} rulings; nomenclature {NOMENCLATURE_VERSION}")

    api = TariffApi(root / "raw" / "atar" / "api", delay_s=args.delay)
    print(f"Trade Tariff API: {api.base_url} ({'token' if api.authenticated else 'no auth'})")
    try:
        groups, audit, pairs = prepare(rulings, codes, cn8_current, api_valid_on(api))
    finally:
        api.close()
    print(f"API: {api.fetched} fetched, rest from cache")

    rng = random.Random(SEED)
    report = build_report(rulings, groups, audit, pairs, rng)

    out = root / "processed" / "rulings" / SPLIT_NAME
    write_split_files(groups, rulings, out)
    write_ref_files(groups, args.splits_dir, refreeze=args.refreeze)
    test_sha = _sha_text(args.splits_dir / "test_refs.txt")
    (args.splits_dir / "SPLIT.md").write_text(split_md(report, test_sha), encoding="utf-8")

    excluded = groups[groups["split"] == "excluded"]
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {
            "rulings_jsonl": str(rulings_path.relative_to(root)),
            "rulings_sha256": sha256(rulings_path),
            "nomenclature_version": NOMENCLATURE_VERSION,
            "codes_parquet_sha256": sha256(tariff / "codes.parquet"),
        },
        "seed": SEED,
        "thresholds": {
            "near_duplicate": NEAR_DUP_THRESHOLD,
            "reported": list(REPORT_THRESHOLDS),
            "vectoriser": "TF-IDF char_wb ngram 3-5, sublinear tf",
        },
        "sizes": {"test": N_TEST, "val": N_VAL},
        "model_input": "description only; description + keywords is a secondary ablation",
        "label_depth": "10, or 8 (train only) when the code10 is not current but its cn8 is",
        "tariff_api": {"base": api.base_url, "historical_param": AS_OF_PARAM},
        "counts": report["counts"],
        "code_status": report["code_status"],
        "excluded": [
            {
                "ref": r.ref,
                "member_refs": r.member_refs,
                "code": r.code_raw,
                "status": r.code_status,
            }
            for r in excluded.itertuples()
        ],
        "test_refs_sha256": test_sha,
        "sha256": {
            f: sha256(out / f)
            for f in [*(f"{s}.parquet" for s in SPLITS), "justifications.parquet"]
        },
        "attribution": OGL_ATTRIBUTION.replace(
            "Sources:", "Sources: HMRC Advance Tariff Rulings (www.tax.service.gov.uk), and"
        ),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print_report(report)
    print(f"\nWrote {out} and {args.splits_dir}")
    return 0


def build_report(rulings, groups, audit, pairs, rng) -> dict:
    n = len(audit)
    a = audit["kw_code_prefix"]
    b = audit["kw_path_words"].map(len) > 0

    def examples(mask, k=10):
        rows = audit[mask]
        picks = rng.sample(range(len(rows)), min(k, len(rows)))
        return [
            {
                "ref": rows.iloc[p]["reference"],
                "description": _short(rows.iloc[p]["description"]),
                "keywords": list(rows.iloc[p]["keywords"]),
                "code": rows.iloc[p]["code_raw"],
                "path_words": list(rows.iloc[p]["kw_path_words"]),
            }
            for p in sorted(picks)
        ]

    conflicts = groups[groups["conflict"]]
    conflict_examples = []
    norms = sorted(set(conflicts["norm"]))
    for norm in rng.sample(norms, min(5, len(norms))):
        g = conflicts[conflicts["norm"] == norm]
        conflict_examples.append(
            {
                "description": _short(g["description"].iloc[0], 120),
                "codes": {r.ref: r.code_raw for r in g.itertuples()},
            }
        )

    reps = groups.reset_index(drop=True)
    near = {}
    for t in REPORT_THRESHOLDS:
        labels = components(len(reps), pairs, t)
        sizes = Counter(labels)
        multi = [s for s in sizes.values() if s > 1]
        near[str(t)] = {
            "clusters": len(sizes),
            "multi_member_clusters": len(multi),
            "rows_in_multi_member_clusters": sum(multi),
            "largest": max(sizes.values()),
        }

    def pair_rows(df):
        return [
            {
                "sim": round(float(p.sim), 4),
                "refs": [reps.at[int(p.i), "ref"], reps.at[int(p.j), "ref"]],
                "codes": [reps.at[int(p.i), "code_raw"], reps.at[int(p.j), "code_raw"]],
                "a": _short(reps.at[int(p.i), "description"], 110),
                "b": _short(reps.at[int(p.j), "description"], 110),
            }
            for p in df.itertuples()
        ]

    above = pairs[pairs["sim"] >= NEAR_DUP_THRESHOLD].nsmallest(5, "sim")
    below = pairs[pairs["sim"] < NEAR_DUP_THRESHOLD].nlargest(5, "sim")

    status = groups.groupby("code_status").agg(
        groups=("ref", "size"), refs=("member_refs", lambda m: sum(map(len, m)))
    )
    non_current = groups[groups["code_status"] != "current"]

    counts = {
        "input_rulings": len(rulings),
        "groups": len(groups),
        "exact_duplicate_groups": int((groups["member_refs"].map(len) > 1).sum()),
        "refs_absorbed_by_exact_dups": int(groups["member_refs"].map(len).sum() - len(groups)),
        "conflict_groups": len(conflicts),
        "conflict_descriptions": len(norms),
        "self_leak_rulings": int(audit["self_leak"].sum()),
        "self_leak_groups": int(groups["self_leak"].sum()),
        "excluded_rows": int((groups["split"] == "excluded").sum()),
        "excluded_refs": int(groups.loc[groups["split"] == "excluded", "member_refs"].map(len).sum()),
    }  # fmt: skip
    train_cn8 = set(groups.loc[groups["split"] == "train", "cn8"])
    shares = {}
    for s in SPLITS:
        part = groups[groups["split"] == s]
        counts[s] = {
            "rows": len(part),
            "refs": int(part["member_refs"].map(len).sum()),
            "chapters": int(part["chapter"].nunique()),
            "label_depth_8": int((part["label_depth"] == 8).sum()),
        }
        shares[s] = part["chapter"].value_counts(normalize=True)
    test = groups[groups["split"] == "test"]
    counts["test_cn8"] = int(test["cn8"].nunique())
    counts["test_cn8_not_in_train"] = len(set(test["cn8"]) - train_cn8)
    counts["test_rows_with_cn8_not_in_train"] = int((~test["cn8"].isin(train_cn8)).sum())
    diff = (shares["test"].reindex(shares["train"].index.union(shares["test"].index), fill_value=0)
            - shares["train"].reindex(shares["train"].index.union(shares["test"].index), fill_value=0))  # fmt: skip
    top = diff.abs().sort_values(ascending=False).head(10).index
    chapter_diff = [
        {
            "chapter": ch,
            "test_share": round(float(shares["test"].get(ch, 0)), 4),
            "train_share": round(float(shares["train"].get(ch, 0)), 4),
        }
        for ch in top
    ]
    in_splits = groups[groups["split"].isin(SPLITS)]  # "excluded" is not a split
    clusters_split = in_splits.groupby("cluster_id")["split"].nunique()
    return {
        "audit": {
            "rulings": n,
            "kw_code_prefix": int(a.sum()),
            "kw_path_words": int(b.sum()),
            "either": int((a | b).sum()),
            "no_path_available": int((~audit["has_path"]).sum()),
            "examples_a": examples(a),
            "examples_b": examples(b),
        },
        "counts": counts,
        "conflict_examples": conflict_examples,
        "near_dup": near,
        "pairs_above": pair_rows(above),
        "pairs_below": pair_rows(below),
        "code_status": {
            "counts": status.reset_index().to_dict("records"),
            "non_current": [
                {
                    "ref": r.ref,
                    "members": len(r.member_refs),
                    "code": r.code_raw,
                    "start": str(r.start_date),
                    "status": r.code_status,
                    "mapped_to": r.mapped_code10,
                    "label_depth": None if pd.isna(r.label_depth) else int(r.label_depth),
                    "split": r.split,
                }
                for r in non_current.itertuples()
            ],
        },
        "chapter_diff": chapter_diff,
        "clusters_spanning_splits": int((clusters_split > 1).sum()),
    }


def print_report(rep: dict) -> None:
    a, c = rep["audit"], rep["counts"]

    def pct(k: str) -> str:
        return f"{a[k]:,} ({a[k] / a['rulings']:.1%})"

    print("\n1. Keyword leakage audit (report only)")
    print(f"  (a) keywords hold a 4+ digit code prefix: {pct('kw_code_prefix')}")
    print(
        f"  (b) keywords hold code-path words absent from the description: {pct('kw_path_words')}"
    )
    print(f"  either: {pct('either')}")
    print(f"  descriptions holding their own code prefix: {c['self_leak_rulings']}")
    for key, label in (("examples_a", "(a)"), ("examples_b", "(b)")):
        print(f"  examples {label}:")
        for e in a[key]:
            extra = f" path words {e['path_words']}" if key == "examples_b" else ""
            print(f"    {e['ref']} {e['code']} kw={e['keywords']}{extra}\n      {e['description']}")
    print("\n2. De-duplication")
    print(f"  {c['input_rulings']:,} rulings -> {c['groups']:,} groups "
          f"({c['exact_duplicate_groups']} groups with duplicates, "
          f"{c['refs_absorbed_by_exact_dups']} references absorbed)")  # fmt: skip
    print(f"  conflicts: {c['conflict_descriptions']} descriptions, {c['conflict_groups']} groups")
    for e in rep["conflict_examples"]:
        print(f"    {e['codes']}: {e['description']}")
    for t, v in rep["near_dup"].items():
        print(f"  near-dup at {t}: {v['clusters']:,} clusters; {v['multi_member_clusters']} with "
              f"2+ members covering {v['rows_in_multi_member_clusters']} rows; "
              f"largest {v['largest']}")  # fmt: skip
    for key, label in (("pairs_above", "just above"), ("pairs_below", "just below")):
        print(f"  pairs {label} {NEAR_DUP_THRESHOLD}:")
        for p in rep[key]:
            print(f"    {p['sim']:.4f} {p['refs']} codes {p['codes']}")
            print(f"      A: {p['a']}\n      B: {p['b']}")
    print("\n3. Code check")
    for s in rep["code_status"]["counts"]:
        print(f"  {s['code_status']}: {s['groups']} groups ({s['refs']} references)")
    for r in rep["code_status"]["non_current"]:
        print(f"    {r['ref']} {r['code']} start {r['start']} {r['status']}"
              f"{' -> ' + r['mapped_to'] if r['mapped_to'] else ''} [{r['split']}"
              f"{', depth ' + str(r['label_depth']) if r['label_depth'] else ''}]"
              f"{' x' + str(r['members']) if r['members'] > 1 else ''}")  # fmt: skip
    print("\n4. Split")
    for s in SPLITS:
        print(f"  {s}: {c[s]['rows']:,} rows ({c[s]['refs']:,} references), "
              f"{c[s]['chapters']} chapters, {c[s]['label_depth_8']} 8-digit labels")  # fmt: skip
    print(f"  excluded: {c['excluded_rows']} rows ({c['excluded_refs']} references)")
    print(f"  clusters spanning splits: {rep['clusters_spanning_splits']}")
    print(f"  test cn8 codes: {c['test_cn8']:,}; not in train: {c['test_cn8_not_in_train']:,} "
          f"({c['test_rows_with_cn8_not_in_train']:,} test rows)")  # fmt: skip
    print("  largest test vs train chapter share differences:")
    for d in rep["chapter_diff"]:
        print(f"    ch {d['chapter']}: test {d['test_share']:.1%} vs train {d['train_share']:.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rulings", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="de-duplicate, check codes, split and freeze")
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
