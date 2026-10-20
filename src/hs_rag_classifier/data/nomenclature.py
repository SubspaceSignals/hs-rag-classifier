"""Ingest the UK tariff nomenclature (codes, paths, section and chapter notes) to Parquet.

Sources (both Open Government Licence v3.0, downloaded at run time, never re-hosted):
    Codes: DBT bulk dataset uk-tariff-2021-01-01, table commodities-report (CSV)
           https://data.api.trade.gov.uk/v1/datasets/uk-tariff-2021-01-01
    Notes: GOV.UK Trade Tariff API (/sections, /sections/{n}, /chapters/{cc})
           https://api.trade-tariff.service.gov.uk/uk/api with a Developer Portal token, or
           https://www.trade-tariff.service.gov.uk/uk/api unauthenticated if .env has no credentials

Usage:
    uv run python -m hs_rag_classifier.data.nomenclature build [--version vX.Y.Z] [--refresh-api]

Everything goes under ~/data/hs-rag-classifier (outside the repo):
    raw/tariff/{version}/commodities-report.csv.gz   raw CSV, downloaded once per version
    raw/tariff/api/...json                            raw API responses, fetched once
    processed/tariff/{version}/codes.parquet, codes8.parquet, notes.parquet, manifest.json

Rules used:
    declarable    suffix "80" and no child lines (the same test the API applies). Suffix 10-40
                  lines are grouping headers and never declarable.
    superseded    where several rows share (code10, suffix), every row is kept in codes.parquet but
                  all except the one with the latest validity_start (ties: the higher sid) are
                  flagged superseded. Anything that reads "current" lines (codes8, counts, spot
                  check) uses current_lines(), which drops them. In v4.0.1633 this affects four
                  codes in heading 2207, where 2019 rows sit alongside their 2023 replacements.
    depth         number of ancestors: chapter 0, heading 1, and so on down the indents.
    description   cleaned of markup: "<br>" and the TARIC "|" non-breaking-space marker become
                  spaces, other tags are dropped keeping their text ("m<sup>3</sup>" -> "m3"),
                  entities unescaped, whitespace collapsed. The original is kept as description_raw.
    8-digit node  for each cn8 (first 8 digits of a declarable line), take the lowest common
                  ancestor of its declarable lines in the hierarchy, then climb while the parent
                  is a suffix-80 line with the same cn8. Its path_text (chapter down to the node)
                  is the cn8 description. When the declarable lines meet only above 8 digits, the
                  node is that higher line and node_above_cn8 is True.
    cn8_declarable  the only declarable line under the cn8 is cn8 + "00".
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from hs_rag_classifier.data.atar import RETRY_STATUSES, USER_AGENT
from hs_rag_classifier.tariff_auth import REQUIRED, fetch_token, load_env

DATASET = "uk-tariff-2021-01-01"
DBT_BASE = f"https://data.api.trade.gov.uk/v1/datasets/{DATASET}"
VERSIONS_URL = f"{DBT_BASE}/versions?format=json"
CSV_URL = DBT_BASE + "/versions/{version}/tables/commodities-report/data?format=csv&download"
API_BASE_AUTH = "https://api.trade-tariff.service.gov.uk/uk/api"
API_BASE_OPEN = "https://www.trade-tariff.service.gov.uk/uk/api"
DEFAULT_DELAY_S = 1.0
DATA_ROOT = Path.home() / "data" / "hs-rag-classifier"
SPOT_CHECK_FIXED = ("8211910000", "7323930000")
SPOT_CHECK_RANDOM = 20

OGL_ATTRIBUTION = (
    "Contains public sector information licensed under the Open Government Licence v3.0 "
    "(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). "
    f"Sources: Department for Business and Trade, dataset {DATASET} "
    "(https://data.api.trade.gov.uk/), and the GOV.UK Trade Tariff API "
    "(https://api.trade-tariff.service.gov.uk/)."
)

CSV_COLUMNS = {
    "commodity__sid": "sid",
    "commodity__code": "code10",
    "commodity__suffix": "suffix",
    "commodity__description": "description_raw",
    "commodity__validity_start": "validity_start",
    "commodity__validity_end": "validity_end",
    "parent__sid": "parent_sid",
}

_BREAK_RE = re.compile(r"<br\s*/?>|\|", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_HEADER_RE = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


# --- Text cleaning --------------------------------------------------------------------------


def clean_text(text: str | None) -> str:
    """Plain one-line text from a tariff description (see module docstring)."""
    if not text:
        return ""
    text = _BREAK_RE.sub(" ", text)
    text = html.unescape(_TAG_RE.sub("", text))
    return _SPACE_RE.sub(" ", text).strip()


def clean_note(text: str | None) -> str:
    """Plain text from a Markdown note: links keep their label, headers lose their #s."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    text = html.unescape(_TAG_RE.sub("", text))
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --- Codes ----------------------------------------------------------------------------------


def read_commodities_csv(source: Path | str | io.IOBase) -> pd.DataFrame:
    """Read the commodities-report CSV (plain or gzipped) with '#NA' as null and codes as text."""
    df = pd.read_csv(
        source,
        dtype=str,
        na_values=["#NA"],
        keep_default_na=False,
        compression="infer",
    )
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    return df


def build_codes(raw: pd.DataFrame, chapter_sections: dict[str, int] | None = None) -> pd.DataFrame:
    """One row per commodity line, with declarable flag, derived codes and chapter-down paths."""
    df = raw[list(CSV_COLUMNS)].rename(columns=CSV_COLUMNS).copy()
    df["sid"] = df["sid"].astype("int64")
    df["parent_sid"] = pd.to_numeric(df["parent_sid"]).astype("Int64")
    df["code10"] = df["code10"].astype(str).str.zfill(10)
    df["suffix"] = df["suffix"].astype(str)
    df["description_raw"] = df["description_raw"].fillna("").astype(str)
    df["description"] = df["description_raw"].map(clean_text)
    for col in ("validity_start", "validity_end"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    parents = {
        int(s): (None if pd.isna(p) else int(p))
        for s, p in zip(df["sid"], df["parent_sid"], strict=True)
    }
    descriptions = dict(zip(df["sid"].astype(int), df["description"], strict=True))
    has_children = {p for p in parents.values() if p is not None}

    paths: dict[int, tuple[list[int], bool]] = {}

    def path_of(sid: int) -> tuple[list[int], bool]:
        """(sids from the root down to sid, whether the root is a chapter line with no parent)."""
        chain: list[int] = []
        seen: set[int] = set()
        cur: int | None = sid
        complete = False
        while cur is not None:
            if cur in paths:
                prefix, complete = paths[cur]
                chain.extend(reversed(prefix))
                break
            if cur in seen or cur not in parents:  # cycle or unresolved parent
                break
            seen.add(cur)
            chain.append(cur)
            cur = parents[cur]
            if cur is None:
                complete = True
        chain.reverse()
        paths[sid] = (chain, complete)
        return chain, complete

    path_sids, complete = zip(*(path_of(int(s)) for s in df["sid"]), strict=True)
    df["path_sids"] = list(path_sids)
    df["path_complete"] = list(complete)
    df["path_text"] = [" > ".join(descriptions[s] for s in p) for p in path_sids]
    df["depth"] = [len(p) - 1 for p in path_sids]
    df["declarable"] = (df["suffix"] == "80") & ~df["sid"].isin(has_children)
    latest = (
        df.assign(_start=pd.to_datetime(df["validity_start"]))
        .sort_values(["_start", "sid"])
        .drop_duplicates(["code10", "suffix"], keep="last")["sid"]
    )
    df["superseded"] = ~df["sid"].isin(latest)

    df["chapter"] = df["code10"].str[:2]
    df["heading"] = df["code10"].str[:4]
    df["subheading"] = df["code10"].str[:6]
    df["cn8"] = df["code10"].str[:8]
    df["section_id"] = df["chapter"].map(chapter_sections or {}).astype("Int64")
    return df[
        [
            "sid", "code10", "suffix", "declarable", "superseded", "description", "description_raw",
            "parent_sid", "validity_start", "validity_end", "depth", "chapter", "heading",
            "subheading", "cn8", "section_id", "path_sids", "path_text", "path_complete",
        ]
    ]  # fmt: skip


def current_lines(codes: pd.DataFrame) -> pd.DataFrame:
    """Lines that are not superseded by a later row with the same (code10, suffix)."""
    return codes[~codes["superseded"]]


def build_codes8(codes: pd.DataFrame) -> pd.DataFrame:
    """One row per 8-digit code with at least one current declarable line (rule in module doc)."""
    by_sid = codes.set_index("sid")
    current = current_lines(codes)
    rows = []
    for cn8, group in current[current["declarable"]].groupby("cn8", sort=True):
        paths = list(group["path_sids"])
        lca: list[int] = []
        for level in zip(*paths, strict=False):
            if len(set(level)) != 1:
                break
            lca.append(level[0])
        node = lca[-1] if lca else int(group["sid"].iloc[0])
        while True:
            parent = by_sid.at[node, "parent_sid"]
            if pd.isna(parent) or int(parent) not in by_sid.index:
                break
            p = by_sid.loc[int(parent)]
            if p["cn8"] != cn8 or p["suffix"] != "80":
                break
            node = int(parent)
        n = by_sid.loc[node]
        children = sorted(set(group["code10"]))
        rows.append(
            {
                "cn8": cn8,
                "subheading": cn8[:6],
                "heading": cn8[:4],
                "chapter": cn8[:2],
                "section_id": group["section_id"].iloc[0],
                "node_sid": node,
                "node_code10": n["code10"],
                "node_suffix": n["suffix"],
                "node_above_cn8": n["code10"][:8] != cn8,
                "description": n["description"],
                "path_text": n["path_text"],
                "children": children,
                "n_children": len(children),
                "cn8_declarable": children == [cn8 + "00"],
            }
        )
    out = pd.DataFrame(rows)
    out["section_id"] = out["section_id"].astype("Int64")
    return out


# --- Notes ----------------------------------------------------------------------------------


def parse_section_ranges(sections_json: dict) -> dict[int, tuple[str, str]]:
    """Section id -> (chapter_from, chapter_to) from the /sections listing."""
    ranges = {}
    for item in sections_json["data"]:
        a = item["attributes"]
        ranges[int(a["id"])] = (str(a["chapter_from"]).zfill(2), str(a["chapter_to"]).zfill(2))
    return ranges


def chapters_from_ranges(ranges: dict[int, tuple[str, str]], chapter: str) -> list[int]:
    return [sid for sid, (lo, hi) in ranges.items() if lo <= chapter <= hi]


def parse_section(section_json: dict, source_url: str) -> dict:
    a = section_json["data"]["attributes"]
    return {
        "level": "section",
        "id": str(a["id"]),
        "title": a.get("title") or "",
        "note": clean_note(a.get("section_note")),
        "note_raw": a.get("section_note") or "",
        "section_id": int(a["id"]),
        "source_url": source_url,
    }


def parse_chapter(chapter_json: dict, source_url: str) -> dict:
    a = chapter_json["data"]["attributes"]
    section = a.get("section_id")
    if section is None:
        rel = chapter_json["data"].get("relationships", {}).get("section", {}).get("data") or {}
        section = rel.get("id")
    return {
        "level": "chapter",
        "id": str(a["goods_nomenclature_item_id"])[:2],
        "title": a.get("description_plain") or clean_text(a.get("description")),
        "note": clean_note(a.get("chapter_note")),
        "note_raw": a.get("chapter_note") or "",
        "section_id": None if section is None else int(section),
        "source_url": source_url,
    }


# --- HTTP -----------------------------------------------------------------------------------


class TariffApi:
    """Trade Tariff API client: 1 request a second, retries with backoff, raw JSON cache.

    Uses the Developer Portal host with a Bearer token when .env holds credentials, and the
    public host otherwise. The token is held in memory only and never printed or logged.
    """

    def __init__(
        self,
        cache_dir: Path,
        env: dict[str, str] | None = None,
        delay_s: float = DEFAULT_DELAY_S,
        max_retries: int = 4,
        refresh: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        self.cache_dir = cache_dir
        self.delay_s = delay_s
        self.max_retries = max_retries
        self.refresh = refresh
        self._sleep = sleep
        self._last_request = float("-inf")
        self._transport = transport
        self._env = env if env is not None else load_env()
        self.authenticated = all(self._env.get(k) for k in REQUIRED)
        self.base_url = API_BASE_AUTH if self.authenticated else API_BASE_OPEN
        self._token: str | None = None
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.hmrc.2.0+json"},
            timeout=30.0,
            transport=transport,
        )
        self.fetched = 0

    def _headers(self) -> dict[str, str]:
        if not self.authenticated:
            return {}
        if self._token is None:
            self._token = fetch_token(self._env, transport=self._transport)["access_token"]
        return {"Authorization": f"Bearer {self._token}"}

    def _wait_turn(self) -> None:
        wait = self._last_request + self.delay_s - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def url(self, path: str) -> str:
        return self.base_url + path

    def get_json(self, path: str, cache: bool = True, params: dict | None = None) -> dict | None:
        """GET path as JSON; None on 404 or a redirect (not a commodity).

        With cache=True the body is saved (a 404 is saved as null) and reused on reruns; query
        params become part of the file name, e.g. commodities/0101210000__as_of=2024-01-01.json.
        """
        name = path.strip("/")
        if params:
            name += "__" + "__".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_path = self.cache_dir / (name + ".json")
        if cache and cache_path.exists() and not self.refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        problem = ""
        reauthed = False
        attempt = 0
        while attempt <= self.max_retries:
            self._wait_turn()
            try:
                response = self._client.get(self.url(path), params=params, headers=self._headers())
            except httpx.TransportError as exc:
                problem = type(exc).__name__
            else:
                status = response.status_code
                if status == 200:
                    self.fetched += 1
                    if cache:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(response.text, encoding="utf-8")
                    return response.json()
                if status == 404 or 300 <= status < 400:
                    if cache:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text("null", encoding="utf-8")
                    return None
                if status == 401 and self.authenticated and not reauthed:
                    self._token, reauthed = None, True  # token expired: fetch a new one once
                    continue
                if status not in RETRY_STATUSES:
                    raise RuntimeError(f"GET {path}: HTTP {status}")
                problem = f"HTTP {status}"
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._sleep(min(int(retry_after), 300))
                    attempt += 1
                    continue
            if attempt < self.max_retries:
                self._sleep(min(60.0, self.delay_s * 2 ** (attempt + 1)))
            attempt += 1
        raise RuntimeError(f"Giving up on {path} after {self.max_retries + 1} attempts ({problem})")

    def close(self) -> None:
        self._client.close()


def resolve_latest_version(client: httpx.Client) -> str:
    """Highest vX.Y.Z listed for the dataset."""
    response = client.get(VERSIONS_URL)
    response.raise_for_status()
    ids = [v["id"] for v in response.json()["versions"] if _VERSION_RE.match(v["id"])]
    if not ids:
        raise RuntimeError("No versions listed for the dataset")
    return max(ids, key=lambda v: tuple(int(x) for x in _VERSION_RE.match(v).groups()))


def download_csv(client: httpx.Client, version: str, dest: Path) -> bool:
    """Stream the commodities CSV into dest (gzipped). Returns False if it was already there."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", CSV_URL.format(version=version)) as response:
        response.raise_for_status()
        with gzip.open(part, "wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
    part.rename(dest)
    return True


# --- Notes collection and checks ------------------------------------------------------------


def collect_notes(
    api: TariffApi, chapters: list[str]
) -> tuple[pd.DataFrame, dict[int, tuple[str, str]], list[str]]:
    """Fetch (or read cached) section and chapter notes. Returns notes, ranges, missing chapters."""
    ranges = parse_section_ranges(api.get_json("/sections"))
    records, missing = [], []
    for section_id in sorted(ranges):
        path = f"/sections/{section_id}"
        records.append(parse_section(api.get_json(path), api.url(path)))
    for i, chapter in enumerate(chapters, 1):
        path = f"/chapters/{chapter}"
        data = api.get_json(path)
        if data is None:
            missing.append(chapter)
        else:
            records.append(parse_chapter(data, api.url(path)))
        if i % 25 == 0:
            print(f"  chapters {i}/{len(chapters)} ({api.fetched} fetched from the API so far)")
    notes = pd.DataFrame(records)
    notes["section_id"] = notes["section_id"].astype("Int64")
    return notes, ranges, missing


def chapter_section_map(notes: pd.DataFrame) -> dict[str, int]:
    chapters = notes[(notes["level"] == "chapter") & notes["section_id"].notna()]
    return {c: int(s) for c, s in zip(chapters["id"], chapters["section_id"], strict=True)}


def run_checks(
    codes: pd.DataFrame,
    codes8: pd.DataFrame,
    chapter_sections: dict[str, int],
    ranges: dict[int, tuple[str, str]],
) -> dict:
    """Row counts and integrity checks; returns a JSON-friendly dict."""
    sids = set(codes["sid"])
    parent_ok = codes["parent_sid"].isna() | codes["parent_sid"].isin(sids)
    current = current_lines(codes)
    decl = current[current["declarable"]]
    chapter_of = dict(zip(codes["sid"], codes["chapter"], strict=True))
    cross = codes[codes["parent_sid"].notna()].copy()
    cross["parent_chapter"] = cross["parent_sid"].map(lambda p: chapter_of.get(int(p)))
    cross = cross[cross["parent_chapter"].notna() & (cross["parent_chapter"] != cross["chapter"])]
    roots = codes[codes["parent_sid"].isna()]

    chapters = sorted(codes["chapter"].unique())
    section_problems = []
    for ch in chapters:
        by_range = chapters_from_ranges(ranges, ch)
        by_api = chapter_sections.get(ch)
        if len(by_range) != 1 or by_api is None or by_range[0] != by_api:
            section_problems.append({"chapter": ch, "api_section": by_api, "ranges": by_range})

    dup_code_suffix = codes[codes.duplicated(["code10", "suffix"], keep=False)]
    return {
        "counts": {
            "lines": len(codes),
            "superseded_lines": int(codes["superseded"].sum()),
            "current_lines": len(current),
            "declarable_lines": len(decl),
            "distinct_declarable_code10": int(decl["code10"].nunique()),
            "distinct_cn8": len(codes8),
            "cn8_declarable_themselves": int(codes8["cn8_declarable"].sum()),
            "cn8_node_above_8_digits": int(codes8["node_above_cn8"].sum()),
            "chapters": len(chapters),
            "sections": len(ranges),
        },
        "integrity": {
            "unresolved_parent_sid": int((~parent_ok).sum()),
            "declarable_without_chapter_path": int((~decl["path_complete"]).sum()),
            "roots_not_chapter_lines": int((~roots["code10"].str.endswith("00000000")).sum()),
            "chapters_without_exactly_one_section": section_problems,
            "duplicate_sid_rows": int(codes["sid"].duplicated().sum()),
            "parent_in_other_chapter": cross[["sid", "code10", "parent_sid"]]
            .head(20)
            .to_dict("records"),
            "parent_in_other_chapter_count": len(cross),
            "duplicate_code10_suffix_rows": len(dup_code_suffix),
            "duplicate_code10_suffix_codes": sorted(set(dup_code_suffix["code10"])),
            "current_duplicate_code10_suffix_rows": int(
                current.duplicated(["code10", "suffix"]).sum()
            ),
        },
    }


def _norm(text: str) -> str:
    return clean_text(text).casefold()


def spot_check(api: TariffApi, codes: pd.DataFrame, seed: str, n: int = SPOT_CHECK_RANDOM) -> dict:
    """Compare sampled declarable lines (plus fixed codes) with the live /commodities endpoint."""
    current = current_lines(codes)
    decl = current[current["declarable"]]
    pool = sorted(set(decl["code10"]) - set(SPOT_CHECK_FIXED))
    picked = random.Random(seed).sample(pool, min(n, len(pool))) + list(SPOT_CHECK_FIXED)
    results = []
    for code in picked:
        ours = decl[decl["code10"] == code]
        data = api.get_json(f"/commodities/{code}", cache=False)
        attrs = (data or {}).get("data", {}).get("attributes", {})
        api_desc = attrs.get("description") or ""
        ok_desc = bool(attrs) and any(_norm(d) == _norm(api_desc) for d in ours["description"])
        results.append(
            {
                "code10": code,
                "declarable_in_dataset": not ours.empty,
                "exists_in_api": data is not None,
                "declarable_in_api": bool(attrs.get("declarable")),
                "description_matches": ok_desc,
                "dataset_description": ours["description"].iloc[0] if len(ours) else None,
                "api_description": clean_text(api_desc) or None,
            }
        )
    mismatches = [
        r
        for r in results
        if not (
            r["declarable_in_dataset"] == r["exists_in_api"] == r["declarable_in_api"]
            and (r["description_matches"] or not r["exists_in_api"])
        )
    ]
    return {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checked": len(results),
        "mismatches": mismatches,
        "results": results,
    }


def check_superseded(api: TariffApi, codes: pd.DataFrame) -> list[dict]:
    """For each code with superseded rows, compare the kept suffix-80 row with the live API."""
    dup_codes = sorted(set(codes.loc[codes["superseded"], "code10"]))
    results = []
    for code in dup_codes:
        rows = codes[(codes["code10"] == code) & (codes["suffix"] == "80")]
        kept = rows[~rows["superseded"]]
        if kept.empty:  # only grouping (non-80) rows repeat for this code
            continue
        kept = kept.iloc[0]
        data = api.get_json(f"/commodities/{code}", cache=False)
        item = (data or {}).get("data", {})
        attrs = item.get("attributes", {})
        start = (attrs.get("validity_start_date") or "")[:10]
        results.append(
            {
                "code10": code,
                "kept_sid": int(kept["sid"]),
                "superseded_sids": sorted(int(s) for s in rows.loc[rows["superseded"], "sid"]),
                "api_sid": int(item["id"]) if item.get("id") else None,
                "sid_matches": item.get("id") == str(kept["sid"]),
                "validity_start_matches": start == str(kept["validity_start"]),
                "description_matches": _norm(attrs.get("description") or "")
                == _norm(kept["description"]),
            }
        )
    return results


# --- Build ----------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")


def print_report(checks: dict, spot: dict, notes: pd.DataFrame, missing: list[str]) -> None:
    c, i = checks["counts"], checks["integrity"]
    print("\nCheck report")
    print(f"  lines {c['lines']:,} ({c['superseded_lines']} superseded, "
          f"{c['current_lines']:,} current)")  # fmt: skip
    print(f"  current declarable {c['declarable_lines']:,} "
          f"({c['distinct_declarable_code10']:,} distinct codes); cn8 {c['distinct_cn8']:,} "
          f"({c['cn8_declarable_themselves']:,} declarable at 8 digits, "
          f"{c['cn8_node_above_8_digits']} with node above 8 digits)")  # fmt: skip
    print(f"  chapters {c['chapters']}; sections {c['sections']}")
    print(f"  unresolved parent_sid: {i['unresolved_parent_sid']}")
    print(f"  declarable lines without a path to a chapter: {i['declarable_without_chapter_path']}")
    print(f"  root lines that are not chapters: {i['roots_not_chapter_lines']}")
    print(f"  chapters without exactly one section: {i['chapters_without_exactly_one_section']}")
    print(f"  duplicate sid rows: {i['duplicate_sid_rows']}")
    print(f"  lines whose parent is in another chapter: {i['parent_in_other_chapter_count']}")
    print(f"  rows sharing (code10, suffix): {i['duplicate_code10_suffix_rows']} "
          f"in {i['duplicate_code10_suffix_codes']}; "
          f"among current lines: {i['current_duplicate_code10_suffix_rows']}")  # fmt: skip
    for s in spot.get("superseded", []):
        print(f"    {s['code10']}: kept sid {s['kept_sid']} (superseded {s['superseded_sids']}); "
              f"api sid {s['api_sid']}, sid match={s['sid_matches']}, "
              f"start match={s['validity_start_matches']}, "
              f"desc match={s['description_matches']}")  # fmt: skip
    empty = notes[notes["note"] == ""]
    print(f"  notes: {len(notes)} rows; empty: "
          f"{', '.join(f'{lv} {x}' for lv, x in zip(empty['level'], empty['id'])) or 'none'}")  # fmt: skip
    if missing:
        print(f"  chapters with no API record: {missing}")
    print(f"  spot check: {spot['checked']} codes, {len(spot['mismatches'])} mismatches")
    for r in spot["results"]:
        if r["code10"] in SPOT_CHECK_FIXED:
            print(f"    {r['code10']}: dataset declarable={r['declarable_in_dataset']}, "
                  f"api exists={r['exists_in_api']} declarable={r['declarable_in_api']}, "
                  f"desc match={r['description_matches']} ({r['dataset_description']!r})")  # fmt: skip
    for m in spot["mismatches"]:
        print(f"    {m['code10']}: dataset declarable={m['declarable_in_dataset']}, "
              f"api exists={m['exists_in_api']} declarable={m['declarable_in_api']}, "
              f"desc match={m['description_matches']}")  # fmt: skip
        if m["exists_in_api"] and not m["description_matches"]:
            print(f"      dataset: {m['dataset_description']!r}")
            print(f"      api:     {m['api_description']!r}")


def cmd_build(args: argparse.Namespace) -> int:
    root: Path = args.root
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120.0) as dbt:
        version = args.version or resolve_latest_version(dbt)
        if not _VERSION_RE.match(version):
            print(f"Not a version id: {version!r}", file=sys.stderr)
            return 1
        print(f"Dataset {DATASET} version {version}{'' if args.version else ' (latest)'}")
        raw_csv = root / "raw" / "tariff" / version / "commodities-report.csv.gz"
        if download_csv(dbt, version, raw_csv):
            print(f"Downloaded {raw_csv}")
        else:
            print(f"Using saved {raw_csv}")

    raw = read_commodities_csv(raw_csv)
    chapters = sorted(raw.loc[raw["parent__sid"].isna(), "commodity__code"].str[:2].unique())

    api = TariffApi(root / "raw" / "tariff" / "api", delay_s=args.delay, refresh=args.refresh_api)
    print(f"Trade Tariff API: {api.base_url} ({'token' if api.authenticated else 'no auth'})")
    try:
        notes, ranges, missing = collect_notes(api, chapters)
        print(f"Notes: {len(notes)} rows ({api.fetched} fetched, rest from cache)")
        sections = chapter_section_map(notes)
        codes = build_codes(raw, sections)
        codes8 = build_codes8(codes)
        checks = run_checks(codes, codes8, sections, ranges)
        print("Spot-checking against the live API...")
        spot = spot_check(api, codes, seed=version)
        spot["superseded"] = check_superseded(api, codes)
    finally:
        api.close()

    out = root / "processed" / "tariff" / version
    out.mkdir(parents=True, exist_ok=True)
    codes.to_parquet(out / "codes.parquet", index=False)
    codes8.to_parquet(out / "codes8.parquet", index=False)
    notes.to_parquet(out / "notes.parquet", index=False)

    api_files = sorted((root / "raw" / "tariff" / "api").rglob("*.json"))
    manifest = {
        "dataset": DATASET,
        "version": version,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "urls": {
            "versions": VERSIONS_URL,
            "commodities_csv": CSV_URL.format(version=version),
            "tariff_api_base": api.base_url,
            "tariff_api_authenticated": api.authenticated,
        },
        "fetched_at": {
            "commodities_csv": _mtime(raw_csv),
            "api_notes_first": min(map(_mtime, api_files)) if api_files else None,
            "api_notes_last": max(map(_mtime, api_files)) if api_files else None,
            "spot_check": spot["checked_at"],
        },
        "row_counts": {"codes": len(codes), "codes8": len(codes8), "notes": len(notes)},
        "checks": checks,
        "spot_check": spot,
        "sha256": {
            "raw": {str(p.relative_to(root)): sha256(p) for p in [raw_csv, *api_files]},
            "processed": {
                str((out / f).relative_to(root)): sha256(out / f)
                for f in ("codes.parquet", "codes8.parquet", "notes.parquet")
            },
        },
        "attribution": OGL_ATTRIBUTION,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print_report(checks, spot, notes, missing)
    print(f"\nWrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nomenclature", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="download (if needed) and build the Parquet files")
    build.add_argument("--version", help="pin a dataset version, e.g. v4.0.1633")
    build.add_argument("--root", type=Path, default=DATA_ROOT, help="data directory")
    build.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="seconds per request")
    build.add_argument("--refresh-api", action="store_true", help="refetch cached API JSON")
    build.set_defaults(func=cmd_build)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
