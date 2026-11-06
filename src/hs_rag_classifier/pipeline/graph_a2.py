"""The A2 pipeline: A1's graph with option scoring in place of the generated decisions.

    input_check -> [profile] -> retrieve -> [evidence] -> stage1_score -> stage2_score
        -> combine -> validate -> rationale -> finalise

input_check, profile, retrieve, evidence: A1's nodes, unchanged (same options, same evidence).
stage1_score  the A1 stage-1 options, a prompt ending with a request for the label only; one
              generated token; the label distribution from its top log-probabilities
              (scoring.label_logprobs).
stage2_score  for the top STAGE2_TOP_HS6 HS6 by stage-1 probability: every current cn8 under
              it, as A1 shows them (overflow rule included), scored the same way; a single-child
              HS6 needs no call (p2 = 1).
combine       every (hs6, cn8) pair by joint probability, each stage's log-probabilities
              divided by its temperature before renormalising (scoring.rank_pairs).
validate      the best pair's cn8, then the next pairs in order until one is valid (flag
              validator_fallback); none valid keeps the best (flag no_valid_code).
rationale     one short post-hoc call (JSON schema: quote, explanation) given only the
              description, the final cn8's full path text and its HS6 option line. It never
              changes the choice. The quote must appear in that tariff text (checks.quote_check).
finalise      candidates: the final cn8, then the other pairs' cn8 in joint order; confidence =
              the final pair's joint probability; the rationale, the disclaimer and the flags.
              No abstention yet.

Order-averaging (config a2.order_averaging, default false; off, every node is as above):
stage1_score and each stage-2 call are made twice, with the options in the usual order and
reversed (labels by position), and the stage's logprobs become log of the per-code average of
the two T = 1 distributions (scoring.average_orders). The top two HS6 for stage 2 and the
pair ranking then follow from those, unchanged; a single-child HS6 still needs no call. The
stage dict keeps the forward call's token, top, floored and outside_mass, sums tokens and llm_s
over both calls, and adds "orders" (each order's codes, raw label log-probabilities, token,
floored, outside_mass and T = 1 probabilities keyed by code) and "averaged" (keyed by code).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from importlib import resources
from typing import Any

import httpx
import openai

from hs_rag_classifier.pipeline import checks, scoring
from hs_rag_classifier.pipeline.graph import DISCLAIMER, Pipeline, State, fill
from hs_rag_classifier.pipeline.llm import call_json, score_call

A2_PROMPTS = (
    "a2_stage1_system",
    "a2_stage1_user",
    "a2_stage2_system",
    "a2_stage2_user",
    "a2_rationale_system",
    "a2_rationale_user",
)
TOP_TOKENS_KEPT = 5


def load_a2_prompts() -> dict[str, str]:
    base = resources.files("hs_rag_classifier.pipeline") / "prompts"
    return {p: (base / f"{p}.txt").read_text(encoding="utf-8") for p in A2_PROMPTS}


def rationale_schema(quote_chars: int, explanation_chars: int) -> dict:
    """Both strings capped so the reply fits max_tokens: the grammar closes each at its cap."""
    return {
        "type": "object",
        "properties": {
            "quote": {"type": "string", "maxLength": quote_chars},
            "explanation": {"type": "string", "maxLength": explanation_chars},
        },
        "required": ["quote", "explanation"],
        "additionalProperties": False,
    }


class A2State(State, total=False):
    stage2_scored: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    rationale: dict[str, Any]


def _timed(name: str, fn: Callable[[A2State], dict]) -> Callable[[A2State], dict]:
    """A1's timing wrapper, typed with A2State: LangGraph reads a node's input keys from the
    type of its first parameter, so A1's wrapper (typed State) would drop the A2 keys."""

    def node(state: A2State) -> dict:
        t0 = time.perf_counter()
        out = fn(state)
        out["timings"] = {name: time.perf_counter() - t0}
        return out

    return node


class A2Pipeline(Pipeline):
    """A1's parts plus a raw HTTP client for scoring and the cn8 path texts for the rationale."""

    def __init__(
        self,
        config: dict[str, Any],
        chat,
        retriever,
        tariff,
        validator,
        http,
        paths: dict[str, str],
        evidence_index=None,
        prompts: dict[str, str] | None = None,
    ):
        super().__init__(config, chat, retriever, tariff, validator, evidence_index)
        self.a2 = config["a2"]
        self.http: httpx.Client = http
        self.paths = paths
        self.a2_prompts = prompts or load_a2_prompts()
        self.order_averaging = bool(self.a2.get("order_averaging", False))

    def _score(self, system: str, user: str, labels: list[str], name: str) -> dict:
        r = score_call(
            self.http, self.cfg["client"], system, user, int(self.a2["top_logprobs"]), name
        )
        d = scoring.label_logprobs(r["top"], labels)
        d.update(
            token=r["token"],
            top=r["top"][:TOP_TOKENS_KEPT],
            llm_s=r["seconds"],
            tokens=[r["tokens_in"], r["tokens_out"]],
        )
        return d

    def _averaged(self, codes: list[str], once: Callable[[list[str], str], dict],
                  name: str) -> dict:  # fmt: skip
        """Score codes in the usual order and reversed; logprobs = log of the per-code average."""
        rev = list(reversed(codes))
        fwd_d, rev_d = once(codes, name), once(rev, f"{name} reversed")
        avg = scoring.average_orders(codes, fwd_d["logprobs"], rev, rev_d["logprobs"])
        keep = ("logprobs", "floored", "outside_mass", "floor", "token", "top", "llm_s", "tokens")
        orders = {
            o: {"codes": c, **{k: d[k] for k in keep}, "probs": avg["probs"][o]}
            for o, c, d in (("forward", codes, fwd_d), ("reversed", rev, rev_d))
        }
        d = dict(fwd_d)
        d.update(
            logprobs=avg["logprobs"],
            llm_s=fwd_d["llm_s"] + rev_d["llm_s"],
            tokens=[
                (a or 0) + (b or 0) for a, b in zip(fwd_d["tokens"], rev_d["tokens"], strict=True)
            ],
            orders=orders,
            averaged=avg["averaged"],
        )
        return d

    # --- nodes ---------------------------------------------------------------------------

    def _stage1_once(self, s: A2State, hs6s: list[str], name: str) -> dict:
        shown = {lab: self.tariff.hs6_line(lab, h) for lab, h in zip(self.labels, hs6s)}
        vals = {**self._blocks(s, self.cfg["evidence"]), "options": "\n".join(shown.values())}
        return self._score(
            self.a2_prompts["a2_stage1_system"],
            fill(self.a2_prompts["a2_stage1_user"], vals),
            list(shown),
            name,
        )

    def stage1_score(self, s: A2State) -> dict:
        hs6s = self.stage1_options(s["fused"])
        if self.order_averaging:
            d = self._averaged(hs6s, lambda c, n: self._stage1_once(s, c, n), "stage1_score_call")
        else:
            d = self._stage1_once(s, hs6s, "stage1_score_call")
        d["options"] = hs6s
        d["choice_code"] = hs6s[scoring.top_k_indices(d["logprobs"], 1)[0]]
        return {"stage1": d, "hs6": d["choice_code"], "tokens": {"stage1": d["tokens"]}}

    def stage2_one(self, s: A2State, hs6: str) -> dict:
        kids = self.tariff.children[hs6]
        if len(kids) == 1:
            return {
                "hs6": hs6,
                "cn8s": list(kids),
                "logprobs": [0.0],
                "skipped": True,
                "overflow": False,
                "n_children": 1,
            }
        cn8s, overflow = self.stage2_options(hs6, s["fused"])
        name = f"stage2_score_call {hs6}"
        if self.order_averaging:
            d = self._averaged(cn8s, lambda c, n: self._stage2_once(s, hs6, c, n), name)
        else:
            d = self._stage2_once(s, hs6, cn8s, name)
        d.update(hs6=hs6, cn8s=cn8s, skipped=False, overflow=overflow, n_children=len(kids))
        return d

    def _stage2_once(self, s: A2State, hs6: str, cn8s: list[str], name: str) -> dict:
        shown = {lab: self.tariff.cn8_line(lab, c) for lab, c in zip(self.labels, cn8s)}
        vals = {
            **self._blocks(s, False),
            "hs6": hs6,
            "hs6_text": self.tariff.hs6_text[hs6],
            "options": "\n".join(shown.values()),
        }
        return self._score(
            self.a2_prompts["a2_stage2_system"],
            fill(self.a2_prompts["a2_stage2_user"], vals),
            list(shown),
            name,
        )

    def stage2_score(self, s: A2State) -> dict:
        s1 = s["stage1"]
        top = scoring.top_k_indices(s1["logprobs"], int(self.a2["stage2_top_hs6"]))
        scored = [self.stage2_one(s, s1["options"][i]) for i in top]
        tokens = {f"stage2_{n + 1}": d["tokens"] for n, d in enumerate(scored) if not d["skipped"]}
        flags = ["stage2_overflow"] if any(d["overflow"] for d in scored) else []
        return {"stage2_scored": scored, "tokens": tokens, "flags": flags}

    def combine(self, s: A2State) -> dict:
        t = self.a2["temperature"]
        ranked = scoring.rank_pairs(s["stage1"], s["stage2_scored"], t["stage1"], t["stage2"])
        return {"ranked": ranked, "cn8": ranked[0]["cn8"]}

    def validate(self, s: A2State) -> dict:
        ranked = s["ranked"]
        chosen = ranked[0]["cn8"]
        result = {"chosen": chosen, "valid": False, "checks": {}, "fallback": False}
        flags: list[str] = []
        for p in ranked:
            r = self.validator.check(p["cn8"])
            result["checks"][p["cn8"]] = r
            if r.get("api_error") and "api_error" not in flags:
                flags.append("api_error")
            if r["valid"]:
                result.update(final=p["cn8"], valid=True, fallback=p["cn8"] != chosen)
                if result["fallback"]:
                    flags.append("validator_fallback")
                break
        else:
            flags.append("no_valid_code")
        result.setdefault("final", chosen)
        return {"validation": result, "cn8": result["final"], "flags": flags}

    def rationale(self, s: A2State) -> dict:
        cn8 = s["cn8"]
        hs6 = next(p["hs6"] for p in s["ranked"] if p["cn8"] == cn8)
        label = self.labels[s["stage1"]["options"].index(hs6)]
        hs6_line = self.tariff.hs6_line(label, hs6)
        path_text = self.paths[cn8]
        user = fill(
            self.a2_prompts["a2_rationale_user"],
            {
                "description": s["description"],
                "cn8": cn8,
                "path_text": path_text,
                "hs6_line": hs6_line,
            },
        )
        rc = self.a2["rationale"]
        t0 = time.perf_counter()
        try:
            r = call_json(
                self.chat,
                self.a2_prompts["a2_rationale_system"],
                user,
                rationale_schema(int(rc["quote_chars"]), int(rc["explanation_chars"])),
                "rationale",
                int(rc["max_tokens"]),
            )
        except (openai.OpenAIError, httpx.HTTPError, ValueError) as exc:  # never changes the choice
            r = {
                "data": None,
                "tokens_in": None,
                "tokens_out": None,
                "error": f"{type(exc).__name__}: {exc}"[:200],
            }
        d = r["data"] or {}
        quote = str(d.get("quote", ""))
        explanation = str(d.get("explanation", ""))
        cite = checks.quote_check(quote, f"{path_text}\n{hs6_line}")
        out = {
            "quote": quote,
            "explanation": explanation,
            "parsed": r["data"] is not None,
            "citation": cite,
            "n_words": len(explanation.split()),
            "llm_s": time.perf_counter() - t0,
            "tokens": [r["tokens_in"], r["tokens_out"]],
            "error": r["error"],
        }
        return {
            "rationale": out,
            "tokens": {"rationale": out["tokens"]},
            "flags": [] if out["parsed"] else ["rationale_parse_error"],
        }

    def finalise(self, s: A2State) -> dict:
        cn8 = s["cn8"]
        pair = next(p for p in s["ranked"] if p["cn8"] == cn8)
        order = list(dict.fromkeys([cn8, *(p["cn8"] for p in s["ranked"])]))
        rat = s.get("rationale") or {}
        text = f"'{rat['quote']}': {rat['explanation']}".strip() if rat.get("parsed") else ""
        final = {
            "candidates": order,
            "top3": order[:3],
            "confidence": pair["joint"],
            "rationale": text,
            "disclaimer": DISCLAIMER,
            "flags": list(dict.fromkeys(s.get("flags", []))),
        }
        return {"final": final}

    # --- graph ---------------------------------------------------------------------------

    def build(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(A2State)
        names = [
            "input_check",
            "retrieve",
            "stage1_score",
            "stage2_score",
            "combine",
            "validate",
            "rationale",
            "finalise",
        ]
        if self.cfg["profile"]:
            names.append("profile")
        if self.cfg["evidence"]:
            names.append("evidence")
        for n in names:
            g.add_node(n, _timed(n, getattr(self, n)))
        g.add_edge(START, "input_check")
        g.add_edge("input_check", "profile" if self.cfg["profile"] else "retrieve")
        if self.cfg["profile"]:
            g.add_edge("profile", "retrieve")
        g.add_edge("retrieve", "evidence" if self.cfg["evidence"] else "stage1_score")
        if self.cfg["evidence"]:
            g.add_edge("evidence", "stage1_score")
        for a, b in [
            ("stage1_score", "stage2_score"),
            ("stage2_score", "combine"),
            ("combine", "validate"),
            ("validate", "rationale"),
            ("rationale", "finalise"),
        ]:
            g.add_edge(a, b)
        g.add_edge("finalise", END)
        return g.compile()
