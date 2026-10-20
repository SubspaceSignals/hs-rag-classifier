"""Parse HMRC Advance Tariff Ruling pages into plain records."""

from __future__ import annotations

import math
import re
from datetime import date, datetime

from bs4 import BeautifulSoup

PAGE_SIZE = 25
_TOTAL_RE = re.compile(r"of\s+([\d,]+)\s+results")
_REF_RE = re.compile(r"ruling\s+(\d+)", re.IGNORECASE)
_KNOWN_FIELDS = {
    "Start date",
    "Expiry date",
    "Commodity code",
    "Description",
    "Keywords",
    "Justification",
}


class RulingParseError(ValueError):
    """The page does not look like a ruling page."""


def _squash(text: str) -> str:
    return " ".join(text.split())


def _keep_lines(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):  # collapse runs of blank lines to one
            out.append(line)
    return "\n".join(out).strip()


def _parse_date(text: str) -> date:
    cleaned = _squash(text).replace("Sept ", "Sep ")
    return datetime.strptime(cleaned, "%d %b %Y").date()


def parse_listing_total(html: str) -> int:
    """Total number of rulings reported on a search results page."""
    match = _TOTAL_RE.search(BeautifulSoup(html, "html.parser").get_text(" "))
    if not match:
        raise RulingParseError("No results count found on listing page")
    return int(match.group(1).replace(",", ""))


def listing_page_count(total: int, page_size: int = PAGE_SIZE) -> int:
    return math.ceil(total / page_size)


def parse_ruling(html: str) -> dict:
    """Return the fields of one ruling page as a JSON-ready dict."""
    soup = BeautifulSoup(html, "html.parser")

    heading = soup.find("h1")
    ref_match = _REF_RE.search(heading.get_text(" ") if heading else "")
    details = soup.find("dl", id="ruling-details")
    if not ref_match or details is None:
        raise RulingParseError("Not a ruling page (no reference heading or details list)")

    rows: dict[str, object] = {}
    for row in details.find_all("div", class_="govuk-summary-list__row"):
        key_el = row.find("dt")
        value_el = row.find("dd")
        if key_el is None or value_el is None:
            continue
        rows[_squash(key_el.get_text())] = value_el

    missing = {"Commodity code", "Description"} - rows.keys()
    if missing:
        raise RulingParseError(f"Ruling page missing fields: {sorted(missing)}")

    code_el = rows["Commodity code"].find("span", class_="commodity-code")
    code = _squash((code_el or rows["Commodity code"]).get_text())
    if not code.isdigit():
        raise RulingParseError(f"Unexpected commodity code: {code!r}")

    record: dict[str, object] = {
        "reference": ref_match.group(1),
        "commodity_code": code,
        "start_date": None,
        "expiry_date": None,
        "description": _squash(rows["Description"].get_text(" ")),
        "keywords": [],
        "justification": None,
    }
    if "Start date" in rows:
        record["start_date"] = _parse_date(rows["Start date"].get_text()).isoformat()
    if "Expiry date" in rows:
        record["expiry_date"] = _parse_date(rows["Expiry date"].get_text()).isoformat()
    if "Keywords" in rows:
        tags = rows["Keywords"].find_all("span", class_="govuk-tag")
        record["keywords"] = [_squash(t.get_text()) for t in tags if _squash(t.get_text())]
    if "Justification" in rows:
        record["justification"] = _keep_lines(rows["Justification"].get_text())

    extra = {k: _squash(v.get_text(" ")) for k, v in rows.items() if k not in _KNOWN_FIELDS}
    if extra:
        record["extra_fields"] = extra  # flags layout changes without failing the run
    return record
