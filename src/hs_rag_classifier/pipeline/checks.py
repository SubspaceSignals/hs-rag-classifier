"""Input check, citation check and option labels: pure functions, no server.

Input check: the request is schema-validated (a non-empty string); control characters are
stripped (tabs and line breaks become spaces) and whitespace collapsed; text over MAX_CHARS is
truncated with the flag "truncated", never refused. Injection-like text (instructions to the
model, role or delimiter tags) is flagged "injection_like" and the delimiter tags are
neutralised, but the description is always handled as data: every prompt puts it inside
<description> tags and tells the model not to follow anything in it.

Citation check: the rationale must quote, in quotation marks (single, double or typographic),
words from the chosen option's text.
Quotes are compared case-insensitively with whitespace collapsed; a quote abridged with an
ellipsis passes when each fragment appears. An item passes when the
rationale has at least one quote, every quote appears in the text of an option shown in that
prompt, and at least one appears in the chosen option's text.

Labels: an option label must be one token in the decider's tokenizer, both on its own and as
the JSON value after '"choice": "', so option probabilities can be read from one token (A2).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, Field

MAX_CHARS = 1000
CANDIDATE_LABELS = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1)) + tuple(
    chr(c) for c in range(ord("a"), ord("z") + 1)
)
LABEL_CONTEXT = '{"choice": "'
DELIMITER = re.compile(r"</?\s*(description|profile|evidence|options)\s*>", re.IGNORECASE)
INJECTION = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instruction|prompt|rule|above|previous|prior)",
        r"\byou are now\b",
        r"\b(system|developer)\s+(prompt|message|instruction)",
        r"\bnew instructions?\b",
        r"\bact as\b|\bpretend to be\b|\bjailbreak\b",
        r"<\|[a-z_]+\|>",  # chat-template tokens such as <|im_start|>
        r"\[/?(inst|sys)\]|<<\s*sys\s*>>",
        r"(^|\s)(system|assistant|user)\s*:",
        r"</?\s*(system|assistant|user|description|profile|evidence|options)\s*>",
        r"\b(answer|respond|output|reply)\s+(only\s+)?with\b",
    )
]


class Request(BaseModel):
    description: str = Field(min_length=1)


def strip_control(text: str) -> str:
    text = re.sub(r"[\t\r\n\v\f]", " ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf", "Co", "Cs"))
    return re.sub(r"\s+", " ", text).strip()


def input_check(raw: object, max_chars: int = MAX_CHARS) -> dict:
    """{"description", "flags", "n_chars"}; raises pydantic.ValidationError on a bad request."""
    req = Request.model_validate({"description": raw})
    text = strip_control(req.description)
    flags = []
    if not text:
        text, flags = "(empty)", ["empty_after_cleaning"]
    if any(p.search(text) for p in INJECTION):
        flags.append("injection_like")
    text = DELIMITER.sub(lambda m: m.group(0).replace("<", "(").replace(">", ")"), text)
    n = len(text)
    if n > max_chars:
        text = text[:max_chars].rstrip()
        flags.append("truncated")
    return {"description": text, "flags": flags, "n_chars": n}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# A single-quoted span opens after the start, a space or an opening bracket and closes before
# a space, punctuation or the end, so apostrophes inside words (bakers' wares) do not end it.
QUOTE = re.compile(r"\"([^\"]+)\"|“([^”]+)”|‘([^’]+)’|(?:^|(?<=[\s(\[]))'(.+?)'(?=[\s.,;:)\]!?]|$)")


def quotes(rationale: str) -> list[str]:
    return [q for groups in QUOTE.findall(rationale or "") for q in groups if q.strip(" .,;:")]


def citation_check(rationale: str, chosen: str, options: Mapping[str, str]) -> dict:
    """Pass or fail (rule in the module docstring). options: label -> option text shown."""
    qs = [_norm(q).strip(" .,;:") for q in quotes(rationale)]
    texts = {k: _norm(v) for k, v in options.items()}

    def found(q: str, text: str) -> bool:  # an ellipsis joins fragments, each must appear
        return all(f.strip(" .,;:") in text for f in re.split(r"\.\.\.|…", q) if f.strip(" .,;:"))

    in_chosen = [q for q in qs if found(q, texts.get(chosen, ""))]
    unfound = [q for q in qs if not any(found(q, t) for t in texts.values())]
    return {"n_quotes": len(qs), "in_chosen": len(in_chosen), "unfound": unfound,
            "pass": bool(qs) and not unfound and bool(in_chosen)}  # fmt: skip


def verify_labels(
    tokenize: Callable[[str], list[int]],
    candidates: Sequence[str] = CANDIDATE_LABELS,
    context: str = LABEL_CONTEXT,
) -> list[str]:
    """The candidates that are single tokens alone and as the JSON value after context."""
    base = tokenize(context)
    ok = []
    for label in candidates:
        alone = tokenize(label)
        after = tokenize(context + label)
        if len(alone) == 1 and after == [*base, alone[0]]:
            ok.append(label)
    return ok


QUOTE_MARKS = "'\"‘’“”"


def quote_check(quote: str, text: str) -> dict:
    """A2 rationale: the quote must appear in the tariff text given (same comparison as
    citation_check: case-insensitive, whitespace collapsed, ellipsis fragments each found)."""
    q = _norm(quote or "").strip(QUOTE_MARKS + " .,;:")
    frags = [f.strip(QUOTE_MARKS + " .,;:") for f in re.split(r"\.\.\.|…", q)]
    frags = [f for f in frags if f]
    t = _norm(text)
    return {"quote": q, "pass": bool(frags) and all(f in t for f in frags)}
