"""Collect HMRC Advance Tariff Rulings at a polite request rate.

Source: https://www.tax.service.gov.uk/search-for-advance-tariff-rulings/
Licence: Open Government Licence v3.0 (stated on the site).

Usage:
    uv run python -m hs_rag_classifier.data.atar sample
    uv run python -m hs_rag_classifier.data.atar collect [--limit N]

Collected data goes outside the repo (default ~/data/hs-rag-classifier/raw/atar):
    refs.txt          every ruling reference from the listing pages
    rulings.jsonl     one parsed ruling per line (appended as it goes)
    html/<ref>.html.gz  raw page, so parsing can be redone without refetching
    errors.jsonl      rulings that could not be parsed
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.robotparser
from datetime import UTC, datetime
from pathlib import Path

import httpx

from hs_rag_classifier.data.atar_parse import (
    RulingParseError,
    listing_page_count,
    parse_listing_total,
    parse_ruling,
)

BASE_URL = "https://www.tax.service.gov.uk"
SEARCH_PATH = "/search-for-advance-tariff-rulings/search"
RULING_PATH = "/search-for-advance-tariff-rulings/ruling/{ref}"
USER_AGENT = "hs-rag-classifier/0.1 (+https://github.com/SubspaceSignals/hs-rag-classifier)"
DEFAULT_DELAY_S = 1.5
RETRY_STATUSES = {429, 500, 502, 503, 504}

RULING_REF_RE = re.compile(r"/search-for-advance-tariff-rulings/ruling/(\d+)")

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "atar"
DEFAULT_DATA_DIR = Path.home() / "data" / "hs-rag-classifier" / "raw" / "atar"


class RobotsDisallowed(RuntimeError):
    """robots.txt does not allow this path for our user agent."""


class PoliteClient:
    """HTTP client that obeys robots.txt, spaces requests out and backs off on errors."""

    def __init__(
        self,
        delay_s: float = DEFAULT_DELAY_S,
        max_retries: int = 4,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
    ) -> None:
        self._sleep = sleep
        self._last_request = float("-inf")
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self.robots_txt, self._robots = self._load_robots()
        crawl_delay = self._robots.crawl_delay(USER_AGENT)
        self.delay_s = max(delay_s, float(crawl_delay or 0))

    def _load_robots(self) -> tuple[str, urllib.robotparser.RobotFileParser]:
        parser = urllib.robotparser.RobotFileParser()
        response = self._client.get("/robots.txt")
        if response.status_code == 404:
            text = ""  # no robots.txt: everything allowed
        elif response.status_code >= 400:
            raise RuntimeError(
                f"Could not read robots.txt (HTTP {response.status_code}); stopping."
            )
        else:
            text = response.text
        parser.parse(text.splitlines())
        return text, parser

    def allowed(self, path: str) -> bool:
        return self._robots.can_fetch(USER_AGENT, BASE_URL + path)

    def _wait_turn(self) -> None:
        wait = self._last_request + self.delay_s - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_request = time.monotonic()

    def get(self, path: str, params: dict | None = None) -> str:
        query = str(httpx.QueryParams(params or {}))
        full_path = f"{path}?{query}" if query else path
        if not self.allowed(full_path):
            raise RobotsDisallowed(full_path)

        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            try:
                response = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                problem = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response.text
                if response.status_code not in RETRY_STATUSES:
                    response.raise_for_status()
                problem = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._sleep(min(int(retry_after), 300))
                    continue
            if attempt < self.max_retries:
                self._sleep(min(60.0, self.delay_s * 2 ** (attempt + 1)))
        raise RuntimeError(
            f"Giving up on {full_path} after {self.max_retries + 1} attempts ({problem})"
        )

    def close(self) -> None:
        self._client.close()


def extract_ruling_refs(listing_html: str) -> list[str]:
    """Ruling references linked from a search results page, in page order, without duplicates."""
    return list(dict.fromkeys(RULING_REF_RE.findall(listing_html)))


def cmd_sample(args: argparse.Namespace) -> int:
    """Save robots.txt, one listing page and two ruling pages as parser fixtures."""
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    client = PoliteClient(delay_s=args.delay)
    try:
        (out / "robots.txt").write_text(client.robots_txt, encoding="utf-8")
        print(f"robots.txt read; crawl delay in use: {client.delay_s}s")

        listing = client.get(SEARCH_PATH, params={"page": 1})
        (out / "listing_page1.html").write_text(listing, encoding="utf-8")
        refs = extract_ruling_refs(listing)
        print(f"Listing page 1: {len(refs)} ruling links found")
        if not refs:
            print("No ruling links found; the page layout may have changed.", file=sys.stderr)
            return 1

        for ref in refs[:2]:
            html = client.get(RULING_PATH.format(ref=ref))
            (out / f"ruling_{ref}.html").write_text(html, encoding="utf-8")
            print(f"Saved ruling {ref}")
    except RobotsDisallowed as exc:
        print(f"robots.txt disallows {exc}; stopping without fetching.", file=sys.stderr)
        return 2
    finally:
        client.close()
    print(f"Fixtures written to {out}")
    return 0


def _read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["reference"])
    return done


def _collect_refs(client: PoliteClient, refs_path: Path) -> list[str]:
    first = client.get(SEARCH_PATH, params={"page": 1})
    total = parse_listing_total(first)
    pages = listing_page_count(total)
    print(f"Listing reports {total:,} rulings over {pages} pages")
    refs = extract_ruling_refs(first)
    for page in range(2, pages + 1):
        refs.extend(extract_ruling_refs(client.get(SEARCH_PATH, params={"page": page})))
        if page % 25 == 0:
            print(f"  listing page {page}/{pages}, {len(set(refs)):,} references so far")
    unique = list(dict.fromkeys(refs))
    refs_path.write_text("\n".join(unique) + "\n", encoding="utf-8")
    print(f"Saved {len(unique):,} unique references to {refs_path}")
    return unique


def cmd_collect(args: argparse.Namespace) -> int:
    """Fetch every ruling, resuming from whatever is already saved."""
    out: Path = args.out
    html_dir = out / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    refs_path, rulings_path, errors_path = (
        out / "refs.txt",
        out / "rulings.jsonl",
        out / "errors.jsonl",
    )

    client = PoliteClient(delay_s=args.delay)
    try:
        if refs_path.exists() and not args.refresh_listing:
            refs = [r for r in refs_path.read_text(encoding="utf-8").split() if r]
            print(f"Using {len(refs):,} saved references (pass --refresh-listing to re-read)")
        else:
            refs = _collect_refs(client, refs_path)

        done = _read_done(rulings_path)
        todo = [r for r in refs if r not in done]
        if args.limit:
            todo = todo[: args.limit]
        to_fetch = sum(1 for r in todo if not (html_dir / f"{r}.html.gz").exists())
        eta_min = to_fetch * client.delay_s / 60
        print(
            f"{len(done):,} already saved; {len(todo):,} to process, "
            f"{to_fetch:,} needing a fetch (about {eta_min:.0f} min)"
        )

        saved = failed = 0
        with (
            rulings_path.open("a", encoding="utf-8") as rulings,
            errors_path.open("w", encoding="utf-8") as errors,  # rewritten each run
        ):
            for i, ref in enumerate(todo, 1):
                gz_path = html_dir / f"{ref}.html.gz"
                if gz_path.exists():  # already fetched: re-parse locally, no request
                    html = gzip.decompress(gz_path.read_bytes()).decode("utf-8")
                    fetched_at = datetime.fromtimestamp(gz_path.stat().st_mtime, UTC)
                else:
                    html = client.get(RULING_PATH.format(ref=ref))
                    gz_path.write_bytes(gzip.compress(html.encode("utf-8")))
                    fetched_at = datetime.now(UTC)
                try:
                    record = parse_ruling(html)
                except RulingParseError as exc:
                    failed += 1
                    errors.write(json.dumps({"reference": ref, "error": str(exc)}) + "\n")
                    errors.flush()
                else:
                    record["source_url"] = BASE_URL + RULING_PATH.format(ref=ref)
                    record["fetched_at"] = fetched_at.isoformat(timespec="seconds")
                    rulings.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rulings.flush()
                    saved += 1
                if i % 50 == 0 or i == len(todo):
                    left_min = (len(todo) - i) * client.delay_s / 60
                    print(
                        f"  {i:,}/{len(todo):,} done ({failed} unparsed), ~{left_min:.0f} min left"
                    )
    except RobotsDisallowed as exc:
        print(f"robots.txt disallows {exc}; stopping.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to resume.")
        return 130
    finally:
        client.close()
    print(f"Finished: {saved:,} saved, {failed} unparsed (see {errors_path.name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atar", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY_S, help="seconds between requests"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="save a few pages as parser test fixtures")
    sample.add_argument("--out", type=Path, default=FIXTURES_DIR)
    sample.set_defaults(func=cmd_sample)
    collect = sub.add_parser("collect", help="fetch all rulings (resumable)")
    collect.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR)
    collect.add_argument("--limit", type=int, default=0, help="fetch at most N rulings this run")
    collect.add_argument("--refresh-listing", action="store_true", help="re-read listing pages")
    collect.set_defaults(func=cmd_collect)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
