"""The A1 pipeline as a LangGraph StateGraph.

    input_check -> [profile] -> retrieve -> [evidence] -> stage1_decide
        -> stage2_decide | stage2_skip -> validate -> finalise

input_check   checks.input_check: cleaned, truncated description and flags.
profile       (toggle) the model's structured profile of the goods (JSON schema); its text is
              appended to the retrieval query and shown in both decision prompts.
retrieve      the locked hybrid retriever. Every fused cn8 (all source hits, not only the top
              K) that is a current cn8, in fused order; the stage-1 options are the first M
              unique HS6 codes of that list.
evidence      (toggle) the EVIDENCE_K nearest train rulings from the rulings source, shown in
              stage 1 as "Past ruling: <description> -> <hs6>".
stage1_decide one labelled HS6 option per line; JSON choice, alternatives, confidence,
              rationale. A reply that does not parse falls back to the first option at
              confidence 0 (flag stage1_parse_error).
stage2        every current cn8 under the chosen HS6. Exactly one: no model call, stage-2
              confidence 100 (stage 1's carries). More than there are labels: the first ones in
              fused order are shown, in code order (flag stage2_overflow).
validate      the chosen cn8, then its fallbacks in order (stage-2 alternatives, then stage-1
              alternatives resolved to their best fused cn8) until one is valid (flag
              validator_fallback); none valid keeps the choice (flag no_valid_code).
finalise      top-3: the chosen cn8, then the alternatives interleaved (stage 1's first, then
              stage 2's); confidence = stage-1 x stage-2 confidence / 100^2; the rationale of
              both stages, the disclaimer and the flags. No abstention yet.
"""

from __future__ import annotations

import operator
import re
import time
from collections.abc import Callable
from importlib import resources
from typing import Annotated, Any, TypedDict

from hs_rag_classifier.pipeline import checks
from hs_rag_classifier.pipeline.llm import call_json, decision_schema, profile_schema
from hs_rag_classifier.retrieval.fusion import hs6_view, rrf

DISCLAIMER = "Advisory only, not a legal ruling; an Advance Tariff Ruling is the legal route"
PROMPTS = ("profile_system", "profile_user", "stage1_system", "stage1_user", "stage2_system",
           "stage2_user")  # fmt: skip


def load_prompts() -> dict[str, str]:
    base = resources.files("hs_rag_classifier.pipeline") / "prompts"
    return {p: (base / f"{p}.txt").read_text(encoding="utf-8") for p in PROMPTS}


def fill(template: str, values: dict[str, str]) -> str:
    """Single-pass {name} substitution: text inserted for one name is never re-read."""
    return re.sub(r"\{(\w+)\}", lambda m: values[m.group(1)], template)


class State(TypedDict, total=False):
    item_id: str
    raw: str
    description: str
    flags: Annotated[list[str], operator.add]
    timings: Annotated[dict[str, float], operator.or_]
    tokens: Annotated[dict[str, list[int | None]], operator.or_]
    profile: dict[str, str] | None
    profile_text: str
    fused: list[str]
    evidence: list[str]
    stage1: dict[str, Any]
    hs6: str
    stage2: dict[str, Any]
    cn8: str
    validation: dict[str, Any]
    final: dict[str, Any]


def _timed(name: str, fn: Callable[[State], dict]) -> Callable[[State], dict]:
    def node(state: State) -> dict:
        t0 = time.perf_counter()
        out = fn(state)
        out["timings"] = {name: time.perf_counter() - t0}
        return out

    return node


class Pipeline:
    """Holds the parts; build() returns the compiled graph (toggles fixed at build time)."""

    def __init__(self, config: dict[str, Any], chat, retriever, tariff, validator,
                 evidence_index=None, prompts: dict[str, str] | None = None):  # fmt: skip
        self.cfg = config
        self.chat, self.retriever, self.tariff = chat, retriever, tariff
        self.validator, self.evidence_index = validator, evidence_index
        self.prompts = prompts or load_prompts()
        self.labels: list[str] = list(config["labels"])
        self.m = int(config["m"])
        if self.m > len(self.labels):
            raise ValueError(f"M = {self.m} exceeds the {len(self.labels)} verified labels")
        if config["evidence"] and evidence_index is None:
            raise ValueError("the evidence toggle needs an EvidenceIndex")

    # --- nodes ---------------------------------------------------------------------------

    def input_check(self, s: State) -> dict:
        out = checks.input_check(s["raw"], self.cfg["max_input_chars"])
        return {"description": out["description"], "flags": out["flags"]}

    def profile(self, s: State) -> dict:
        user = fill(self.prompts["profile_user"], {"description": s["description"]})
        r = call_json(self.chat, self.prompts["profile_system"], user,
                      profile_schema(self.cfg["profile_field_chars"]), "profile",
                      self.cfg["max_tokens"]["profile"])  # fmt: skip
        prof = r["data"] or {}
        text = "; ".join(f"{k}: {checks.strip_control(str(v))}" for k, v in prof.items())
        return {"profile": prof or None, "profile_text": text,
                "tokens": {"profile": [r["tokens_in"], r["tokens_out"]]},
                "flags": [] if r["data"] else ["profile_parse_error"]}  # fmt: skip

    def query(self, s: State) -> str:
        return (
            f"{s['description']}\n{s['profile_text']}"
            if s.get("profile_text")
            else s["description"]
        )

    def retrieve(self, s: State) -> dict:
        lists = self.retriever.source_lists([self.query(s)])[0]
        cfg = self.retriever.config
        fused = rrf({src: [h.cn8 for h in hits] for src, hits in lists.items()},
                    cfg["weights"], cfg["k_rrf"])  # fmt: skip
        return {"fused": [c for c, _ in fused if c in self.tariff.cn8_text]}

    def evidence(self, s: State) -> dict:
        q = self.retriever.encoder.queries([self.query(s)])
        store = self.retriever.sources["rulings"].store
        (ids,), _ = store.search(q, self.cfg["evidence_k"])
        lines = self.evidence_index.lines([str(i) for i in ids], self.cfg["evidence_chars"])
        return {"evidence": [checks.strip_control(x) for x in lines]}

    def _blocks(self, s: State, evidence: bool) -> dict[str, str]:
        prof = f"<profile>\n{s['profile_text']}\n</profile>\n" if s.get("profile_text") else ""
        ev = ""
        if evidence and s.get("evidence"):
            ev = "Past rulings on similar goods:\n" + "\n".join(s["evidence"]) + "\n"
        return {"description": s["description"], "profile": prof, "evidence": ev}

    def stage1_options(self, fused: list[str]) -> list[str]:
        return hs6_view(fused)[: self.m]

    def stage1_prompt(self, s: State, hs6s: list[str]) -> tuple[str, str, dict[str, str]]:
        shown = {lab: self.tariff.hs6_line(lab, h) for lab, h in zip(self.labels, hs6s)}
        vals = {**self._blocks(s, self.cfg["evidence"]), "options": "\n".join(shown.values())}
        return self.prompts["stage1_system"], fill(self.prompts["stage1_user"], vals), shown

    def _decide(self, stage: str, system: str, user: str, shown: dict[str, str]) -> dict:
        labels = list(shown)
        r = call_json(self.chat, system, user,
                      decision_schema(labels, self.cfg["rationale_chars"]), stage,
                      self.cfg["max_tokens"]["decide"])  # fmt: skip
        d = r["data"] or {}
        ok = d.get("choice") in shown
        choice = d["choice"] if ok else labels[0]
        alts = [a for a in dict.fromkeys(d.get("alternatives") or [])
                if a in shown and a != choice][:2]  # fmt: skip
        conf = int(d.get("confidence", 0)) if ok else 0
        rationale = str(d.get("rationale", "")) if ok else ""
        option_text = {k: v.split(". ", 1)[1] for k, v in shown.items()}
        return {"choice": choice, "alternatives": alts, "confidence": max(0, min(100, conf)),
                "rationale": rationale, "parsed": ok, "n_options": len(labels),
                "citation": checks.citation_check(rationale, choice, option_text),
                "tokens": [r["tokens_in"], r["tokens_out"]], "llm_s": r["seconds"],
                "error": r["error"]}  # fmt: skip

    def stage1_decide(self, s: State) -> dict:
        hs6s = self.stage1_options(s["fused"])
        system, user, shown = self.stage1_prompt(s, hs6s)
        d = self._decide("stage1", system, user, shown)
        by_label = dict(zip(self.labels, hs6s))
        d["options"] = hs6s
        d["choice_code"] = by_label[d["choice"]]
        d["alternative_codes"] = [by_label[a] for a in d["alternatives"]]
        return {"stage1": d, "hs6": d["choice_code"], "tokens": {"stage1": d["tokens"]},
                "flags": [] if d["parsed"] else ["stage1_parse_error"]}  # fmt: skip

    def route_stage2(self, s: State) -> str:
        return "stage2_skip" if len(self.tariff.children[s["hs6"]]) == 1 else "stage2_decide"

    def stage2_options(self, hs6: str, fused: list[str]) -> tuple[list[str], bool]:
        kids = self.tariff.children[hs6]
        if len(kids) <= len(self.labels):
            return kids, False
        rank = {c: i for i, c in enumerate(fused)}
        keep = sorted(kids, key=lambda c: (rank.get(c, len(rank)), c))[: len(self.labels)]
        return sorted(keep), True

    def stage2_decide(self, s: State) -> dict:
        hs6 = s["hs6"]
        cn8s, overflow = self.stage2_options(hs6, s["fused"])
        shown = {lab: self.tariff.cn8_line(lab, c) for lab, c in zip(self.labels, cn8s)}
        vals = {**self._blocks(s, False), "hs6": hs6, "hs6_text": self.tariff.hs6_text[hs6],
                "options": "\n".join(shown.values())}  # fmt: skip
        d = self._decide("stage2", self.prompts["stage2_system"],
                         fill(self.prompts["stage2_user"], vals), shown)  # fmt: skip
        by_label = dict(zip(self.labels, cn8s))
        d.update(skipped=False, overflow=overflow, n_children=len(self.tariff.children[hs6]),
                 choice_code=by_label[d["choice"]],
                 alternative_codes=[by_label[a] for a in d["alternatives"]])  # fmt: skip
        flags = ([] if d["parsed"] else ["stage2_parse_error"]) + (
            ["stage2_overflow"] if overflow else []
        )
        return {"stage2": d, "cn8": d["choice_code"], "tokens": {"stage2": d["tokens"]},
                "flags": flags}  # fmt: skip

    def stage2_skip(self, s: State) -> dict:
        (cn8,) = self.tariff.children[s["hs6"]]
        d = {"skipped": True, "choice_code": cn8, "alternative_codes": [], "confidence": 100,
             "rationale": "", "citation": None, "overflow": False, "n_children": 1}  # fmt: skip
        return {"stage2": d, "cn8": cn8}

    def best_cn8(self, hs6: str, fused: list[str]) -> str:
        return next((c for c in fused if c.startswith(hs6)), self.tariff.children[hs6][0])

    def fallbacks(self, s: State) -> list[str]:
        s1 = [self.best_cn8(h, s["fused"]) for h in s["stage1"]["alternative_codes"]]
        return list(dict.fromkeys([*s["stage2"]["alternative_codes"], *s1]))

    def validate(self, s: State) -> dict:
        chosen = s["cn8"]
        first = self.validator.check(chosen)
        result = {"chosen": chosen, "valid": first["valid"], "checks": {chosen: first},
                  "fallback": False}  # fmt: skip
        flags = ["api_error"] if first.get("api_error") else []
        if not first["valid"]:
            for alt in self.fallbacks(s):
                if alt == chosen:
                    continue
                r = self.validator.check(alt)
                result["checks"][alt] = r
                if r["valid"]:
                    result.update(final=alt, valid=True, fallback=True)
                    flags.append("validator_fallback")
                    break
            else:
                flags.append("no_valid_code")
        result.setdefault("final", chosen)
        return {"validation": result, "cn8": result["final"], "flags": flags}

    def finalise(self, s: State) -> dict:
        s1, s2 = s["stage1"], s["stage2"]
        alts1 = [self.best_cn8(h, s["fused"]) for h in s1["alternative_codes"]]
        alts2 = list(s2["alternative_codes"])
        order = [s["cn8"]]
        for i in range(2):
            order += [a[i] for a in (alts1, alts2) if i < len(a)]
        top3 = list(dict.fromkeys(order))[:3]
        conf = s1["confidence"] * s2["confidence"] / 100**2
        parts = [f"Subheading {s['hs6']}: {s1['rationale']}".strip()]
        parts.append(
            f"Commodity code {s['cn8']}: "
            + (s2["rationale"] or "the only commodity code under this subheading.")
        )
        final = {"top3": top3, "confidence": conf, "rationale": " ".join(parts),
                 "disclaimer": DISCLAIMER, "flags": list(dict.fromkeys(s.get("flags", [])))}  # fmt: skip
        return {"final": final}

    # --- graph ---------------------------------------------------------------------------

    def build(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(State)
        names = ["input_check", "retrieve", "stage1_decide", "stage2_decide", "stage2_skip",
                 "validate", "finalise"]  # fmt: skip
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
        g.add_edge("retrieve", "evidence" if self.cfg["evidence"] else "stage1_decide")
        if self.cfg["evidence"]:
            g.add_edge("evidence", "stage1_decide")
        g.add_conditional_edges("stage1_decide", self.route_stage2,
                                ["stage2_decide", "stage2_skip"])  # fmt: skip
        g.add_edge("stage2_decide", "validate")
        g.add_edge("stage2_skip", "validate")
        g.add_edge("validate", "finalise")
        g.add_edge("finalise", END)
        return g.compile()
