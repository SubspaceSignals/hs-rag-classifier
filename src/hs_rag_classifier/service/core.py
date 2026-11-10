"""The service's pipeline object: the frozen A4 arm, the operating-point rule and the OOD rule.

Start-up (A4Service.from_frozen) refuses, before loading anything heavy, unless A4's config_hash,
the operating-point rule's SHA-256 and the OOD rule's SHA-256 are the pinned values, and the
adapter file has the config's SHA-256 (hashed here, once; the result is kept). /health then only
checks that llama-server is up and lists that adapter (path and id), never re-hashing the file.
The arm is the evaluations' own arms.a4.A4Arm (its pin, slot and adapter guards run as in every run); the
retriever, B1-ex and the OOD scorer share one bge-m3 (arms.embed.shared_backend), held for the
app's lifetime and never released on a request path.

Per request, one at a time (the app holds a lock): A4 predicts on the description; the OOD
scorer finds the nearest train ruling on the full index (the precedent); needs_review =
operating-point rule (eval.operating_point.apply_rule, cn8 policy (b), the lower-bound
thresholds in configs/operating_point_rule.json) OR the OOD flag (ood.apply_rule).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hs_rag_classifier import ood as OOD
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, EvalItem, sha256_file

A4_CONFIG_HASH = "5da7ac1452b3"
OPERATING_POINT_RULE_SHA = "36357cb58e215f9261944c3bff8e648909bbd6ca085a9c510698554353a8429d"
OOD_RULE_SHA = "b275978882c320ed96c69c32a2888ec4cc2eb02664ad2ac9e4f39c2ae6b3bcb9"
OPERATING_POINT_RULE = REPO_ROOT / "configs" / "operating_point_rule.json"
POLICY = "b"  # the rule file's headline cn8 order
DEFAULT_LLAMA_URL = "http://127.0.0.1:8080"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
ADVISORY = "Advisory only, not a legal ruling; an Advance Tariff Ruling is the legal route"


class StartupError(RuntimeError):
    """The service refuses to start (a pin does not match, or the pipeline cannot load)."""


class UpstreamError(RuntimeError):
    """llama-server was unreachable or failed during a request (the API answers 503)."""


@dataclass(frozen=True)
class Settings:
    llama_url: str = DEFAULT_LLAMA_URL
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    data_root: Path = DATA_ROOT
    operating_point_rule: Path = OPERATING_POINT_RULE
    ood_rule: Path = field(default=OOD.RULE_PATH)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        return cls(llama_url=env.get("LLAMA_URL", DEFAULT_LLAMA_URL),
                   host=env.get("HOST", DEFAULT_HOST), port=int(env.get("PORT", DEFAULT_PORT)))  # fmt: skip


def verify_pins(
    config: dict[str, Any], operating_point_rule: Path, ood_rule: Path
) -> dict[str, str]:
    """Refuse unless the A4 config hash and both rule files' SHA-256 are the pinned values."""
    from hs_rag_classifier.eval.arms import config_hash

    got = {"config_hash": config_hash(config), "operating_point_rule_sha": sha256_file(operating_point_rule),
           "ood_rule_sha": sha256_file(ood_rule)}  # fmt: skip
    want = {"config_hash": A4_CONFIG_HASH, "operating_point_rule_sha": OPERATING_POINT_RULE_SHA,
            "ood_rule_sha": OOD_RULE_SHA}  # fmt: skip
    bad = [f"{k} is {got[k]}, expected {want[k]}" for k in want if got[k] != want[k]]
    if bad:
        raise StartupError("refusing to start: " + "; ".join(bad))
    return got


def verify_adapter(config: dict[str, Any]) -> dict[str, str]:
    """Hash the config's adapter file once and refuse unless it has the declared SHA-256."""
    want = config["adapter"]
    path = Path(want["path"]).expanduser()
    if not path.is_file():
        raise StartupError(f"refusing to start: the adapter {path} is not a file")
    got = sha256_file(path)
    if got != want["sha256"]:
        raise StartupError(
            f"refusing to start: the adapter {path} has SHA-256 {got}, expected {want['sha256']}"
        )
    return {"path": str(path), "sha256": got}


def decide(op_review: bool, ood_flagged: bool) -> tuple[bool, list[str]]:
    """needs_review = operating point OR OOD flag; the reasons in a fixed order."""
    reasons = [
        r for r, on in (("operating_point", op_review), ("out_of_distribution", ood_flagged)) if on
    ]
    return bool(reasons), reasons


def operating_point_review(confidence: float, agree_cn8: bool, thresholds: dict[str, float],
                           item_id: str = "request") -> bool:  # fmt: skip
    """The step 2c per-item rule for policy (b), through eval.operating_point itself."""
    from hs_rag_classifier.eval import operating_point as OP

    a = OP.arrays(
        [confidence], [0.0], [bool(agree_cn8)], [item_id]
    )  # correct is unknown and unused
    return bool(OP.apply_rule(POLICY, thresholds, a)[0])


def _upstream_failure(rationale: dict[str, Any] | None) -> str | None:
    """The rationale call's error when it was the server (connection, HTTP, OpenAI client), not a
    reply the pipeline could not parse (ValueError)."""
    err = (rationale or {}).get("error")
    if not err:
        return None
    return None if str(err).split(":", 1)[0] == "ValueError" else str(err)


class A4Service:
    """Holds the frozen arm, the rules and the OOD scorer; classify() is called under a lock."""

    def __init__(self, arm, scorer: OOD.OODScorer, config: dict[str, Any], op_rule: dict[str, Any],
                 ood_rule: dict[str, Any], provenance: dict[str, str], llama_url: str,
                 ruling_cn8: np.ndarray | None = None,
                 adapter: dict[str, Any] | None = None):  # fmt: skip
        from hs_rag_classifier.eval import operating_point as OP

        self.arm, self.scorer, self.config = arm, scorer, config
        self.op_rule, self.ood_rule, self.provenance = op_rule, ood_rule, provenance
        self.thresholds = OP.rules_for(op_rule, "cn8", POLICY)["lower_bound"]["thresholds"]
        self.llama_url = llama_url
        self.ruling_cn8 = ruling_cn8 if ruling_cn8 is not None else arm.b1ex.index.cn8
        self.adapter = adapter  # {"path", "sha256", "id"}, hashed and read once at start-up
        self.loaded = True

    @classmethod
    def from_frozen(cls, settings: Settings | None = None) -> A4Service:
        import json

        from hs_rag_classifier.arms import a4 as A4
        from hs_rag_classifier.eval import runner as R

        settings = settings or Settings.from_env()
        config = A4.load_config(root=settings.data_root)
        pins = verify_pins(
            config, settings.operating_point_rule, settings.ood_rule
        )  # before any load
        adapter = verify_adapter(config)  # the one hash of the adapter file in the service
        try:  # git, or HS_RAG_GIT_* in the image; neither refuses here, before any load
            git = R.git_info()
        except R.GitInfoError as exc:
            raise StartupError(f"refusing to start: {exc}") from exc
        ood_rule = OOD.load_rule(settings.ood_rule)
        op_rule = json.loads(Path(settings.operating_point_rule).read_text(encoding="utf-8"))
        served = {**config, "client": {**config["client"], "base_url": settings.llama_url}}
        try:
            arm = A4.A4Arm(
                served, settings.data_root
            )  # pins, server model, slot and adapter guards
        except Exception as exc:
            raise StartupError(
                f"refusing to start: the A4 pipeline did not load ({type(exc).__name__}: {exc})"
            ) from exc
        from hs_rag_classifier.pipeline import llm

        try:
            listed = llm.server_adapter(settings.llama_url, adapter["path"])
        except llm.AdapterError as exc:
            raise StartupError(f"refusing to start: {exc}") from exc
        adapter = {**adapter, "id": listed.get("id")}
        enc = arm.b1ex.encoder  # B1-ex's encoder: the same shared bge-m3 as the retriever's
        index = OOD.RulingsIndex(
            arm.b1ex.index.embeddings, [str(r) for r in arm.b1ex.index.meta["refs"]]
        )
        scorer = OOD.OODScorer(index, enc.queries, int(config["max_input_chars"]))
        prov = {**pins, "git_commit": git["commit"] + ("-dirty" if git["dirty"] else "")}
        return cls(arm, scorer, config, op_rule, ood_rule, prov, settings.llama_url,
                   adapter=adapter)  # fmt: skip

    def encoder_backends(self) -> dict[str, Any]:
        """The bge-m3 backend each component holds (all the same object)."""
        return {"retriever": self.arm.a2.pipeline.retriever.encoder.backend,
                "b1ex": self.arm.b1ex.encoder.backend,
                "ood": getattr(self.scorer.encode, "__self__", None).backend}  # fmt: skip

    def classify(self, description: str, request_id: str) -> dict[str, Any]:
        import httpx
        import openai

        item = EvalItem(item_id=request_id, text=description, label_code="00000000", label_depth=8,
                        cluster_id=request_id, chapter="00", dataset="service", split="serve")  # fmt: skip
        try:
            pred = self.arm.predict([item])[0]
        except (httpx.HTTPError, openai.OpenAIError, ConnectionError) as exc:
            raise UpstreamError(f"llama-server failed: {type(exc).__name__}") from exc
        det = pred.details or {}
        rat = det.get("rationale")
        failure = _upstream_failure(rat)
        if failure:
            raise UpstreamError(
                f"llama-server failed during the rationale: {failure.split(':', 1)[0]}"
            )
        near = self.scorer.nearest(description)  # full index: serving never masks
        flagged, message = OOD.apply_rule(near.similarity, self.ood_rule)
        agree = bool((det.get("a3") or {}).get("agree_cn8"))
        op = operating_point_review(pred.confidence, agree, self.thresholds, request_id)
        needs, reasons = decide(op, flagged)
        cands = [{"cn8": c.code[:8], "confidence": float(c.score)} for c in pred.candidates[:3]]
        cite = (rat or {}).get("citation")
        top1 = cands[0]["cn8"] if cands else None
        prec_cn8 = str(self.ruling_cn8[near.row])
        return {
            "request_id": request_id,
            "cn8": top1,
            "confidence": float(pred.confidence),
            "top3": cands,
            "needs_review": needs,
            "review_reasons": reasons,
            "ood": {"similarity": near.similarity, "threshold": float(self.ood_rule["threshold"]),
                    "flagged": flagged, "nearest_ruling_ref": near.ref, "message": message},
            "rationale": pred.rationale,
            "citation": {"quote": cite.get("quote"), "pass": bool(cite.get("pass"))} if cite else None,
            "precedent": {"ref": near.ref, "cn8": prec_cn8, "similarity": near.similarity,
                          "agrees_with_prediction": top1 is not None and prec_cn8 == top1},
            "provenance": dict(self.provenance),
            "advisory": det.get("disclaimer") or ADVISORY,
        }  # fmt: skip

    def health(self) -> tuple[bool, str | None]:
        """(ok, reason): loaded, llama-server up and listing the adapter start-up hashed (same
        path and id). Reads /lora-adapters only; the file is never re-hashed here."""
        from hs_rag_classifier.pipeline import llm

        if not self.loaded or not self.adapter:
            return False, "pipeline not loaded"
        if not llm.ServerInfo(self.llama_url).up():
            return False, f"llama-server at {self.llama_url} is not reachable"
        try:
            listed = llm.server_adapter(self.llama_url, self.adapter["path"])
        except llm.AdapterError as exc:
            return False, f"llama-server adapter check failed: {exc}"
        if listed.get("id") != self.adapter["id"]:
            return False, (f"llama-server adapter check failed: id {listed.get('id')}, "
                           f"expected {self.adapter['id']}")  # fmt: skip
        return True, None
