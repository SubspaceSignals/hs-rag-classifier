"""A4: A3 with the final QLoRA adapter on the scoring calls and refitted stage temperatures.

Config: configs/a4_pipeline.json (frozen). It pins configs/a3_pipeline.json by SHA-256 and
inherits everything from it (retrieval, B1-ex locked config, order-averaging on both stages,
policy (b) review ordering, prompts, server flags). A4 changes only: the adapter (the final
GGUF, {path under the data root, sha256}), the per-request LoRA scales (scoring 1, rationale 0)
and both stage temperatures (refitted on val, eval/a4/a4_final_refit_val.json). The rest of
the file records lineage (base GGUF, training manifest, commits) and the expected val figures.

Code path: the step-5 candidate arm (arms/a4_candidate.py, itself A3's arm) with the frozen
adapter; no A3 logic is copied. At start, before any scoring request, the arm refuses unless
(check_pins): the frozen file is the one the config was built from; a3_pipeline.json still has
its pinned SHA-256; the config's adapter block is the frozen one and the file on disk has the
pinned SHA-256; the base GGUF on disk has the pinned SHA-256 (and is the one the A1 server
block pins); the temperatures and LoRA scales are exactly the frozen values, which are exactly
the refit file's (by SHA-256 and by value, full precision). Building the pipeline then runs the
one-slot guard and the adapter guard against /lora-adapters (pipeline/llm.py), and the arm checks
that llama-server's /props model_path is the pinned base GGUF.

MLflow params (run_params, next to the harness's config_hash): adapter SHA-256, candidate,
the frozen file's SHA-256 and both temperatures as full-precision strings (repr).

Allowed on test splits under every existing harness guard (python -m hs_rag_classifier.eval run
--arm a4 --dataset ... --split test); nothing in the test suite calls it on a test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hs_rag_classifier.arms import a3_pipeline as A3
from hs_rag_classifier.arms.a4_candidate import A4CandidateArm
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "a4_pipeline.json"


class PinError(ValueError):
    """The config, the files on disk or the server disagree with the frozen config (the CLI
    reports it as Refused, like the other guards)."""


def frozen(path: Path = CONFIG_PATH) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "frozen" or raw.get("arm") != "a4":
        raise PinError(f"{path.name} is not the frozen A4 config")
    return raw


def adapter_block(raw: dict[str, Any], root: Path = DATA_ROOT) -> dict[str, str]:
    """The config's adapter block {"path" (absolute), "sha256"} for the frozen adapter."""
    path = (Path(root).expanduser() / raw["adapter"]["path"]).resolve()
    return {"path": str(path), "sha256": raw["adapter"]["sha256"]}


def load_config(path: Path = CONFIG_PATH, repo: Path = REPO_ROOT,
                root: Path = DATA_ROOT) -> dict[str, Any]:  # fmt: skip
    """A3's config with the frozen adapter, LoRA scales and refitted temperatures."""
    raw = frozen(path)
    pin = raw["base_config"]
    if sha256(repo / pin["path"]) != pin["sha256"]:
        raise PinError(f"{pin['path']} changed since {path.name} was frozen")
    base = A3.load_config(repo / pin["path"], repo)
    a2 = {**base["a2"], "temperature": raw["temperature"], "lora_scales": raw["lora_scales"]}
    meta = {"config": {"path": path.name, "sha256": sha256(path)},
            "candidate": raw["adapter"]["candidate"], "frozen": raw}  # fmt: skip
    return {**base, "a2": a2, "adapter": adapter_block(raw, root), "a4": meta}


def _exact(got: Any, want: float) -> bool:
    return isinstance(got, float) and isinstance(want, float) and repr(got) == repr(want)


def check_pins(config: dict[str, Any], path: Path = CONFIG_PATH, repo: Path = REPO_ROOT,
               root: Path = DATA_ROOT) -> None:  # fmt: skip
    """Refuse (PinError) unless config and files match the frozen config (module docstring)."""
    raw = frozen(path)
    meta = config.get("a4") or {}
    if meta.get("config", {}).get("sha256") != sha256(path) or meta.get("frozen") != raw:
        raise PinError(f"the config was not built from the current {path.name}")
    pin = raw["base_config"]
    if sha256(repo / pin["path"]) != pin["sha256"]:
        raise PinError(f"{pin['path']} changed since {path.name} was frozen")
    want = adapter_block(raw, root)
    if config.get("adapter") != want:
        raise PinError(f"the adapter block {config.get('adapter')} is not the frozen {want}")
    gguf = Path(want["path"])
    if not gguf.is_file() or sha256(gguf) != want["sha256"]:
        raise PinError(f"the adapter {gguf} is missing or does not have SHA-256 {want['sha256']}")
    base = raw["base_model"]
    base_path = Path(base["path"]).expanduser()
    if config.get("server", {}).get("model_sha256") != base["sha256"]:
        raise PinError("the A1 server block does not pin the frozen base GGUF")
    if not base_path.is_file() or sha256(base_path) != base["sha256"]:
        raise PinError(f"the base GGUF {base_path} is missing or does not have SHA-256 "
                       f"{base['sha256']}")  # fmt: skip
    t = raw["temperature"]
    src = Path(root).expanduser() / t["source"]["path"]
    if not src.is_file() or sha256(src) != t["source"]["sha256"]:
        raise PinError(f"the refit file {src} is missing or does not have SHA-256 "
                       f"{t['source']['sha256']}")  # fmt: skip
    fitted = json.loads(src.read_text(encoding="utf-8"))["temperature"]
    got = config.get("a2", {}).get("temperature", {})
    for stage in ("stage1", "stage2"):
        if not (_exact(t[stage], fitted[stage]) and _exact(got.get(stage), t[stage])):
            raise PinError(f"{stage} temperature {got.get(stage)!r} is not the frozen "
                           f"{t[stage]!r} (refit file {fitted[stage]!r})")  # fmt: skip
    if got.get("mode") != "per_stage":
        raise PinError("the temperature mode is not per_stage")
    if config["a2"].get("lora_scales") != raw["lora_scales"]:
        raise PinError(f"LoRA scales {config['a2'].get('lora_scales')} are not the frozen "
                       f"{raw['lora_scales']}")  # fmt: skip


def check_server_model(config: dict[str, Any], transport=None) -> None:
    """Refuse unless llama-server's /props model_path is the frozen base GGUF."""
    import httpx

    from hs_rag_classifier.pipeline import llm

    want = Path(config["a4"]["frozen"]["base_model"]["path"]).expanduser().resolve()
    url = config["client"]["base_url"]
    try:
        got = llm.ServerInfo(url, transport).props().get("model_path")
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        raise PinError(f"cannot read /props model_path from {url} ({exc})") from exc
    if not got or Path(got).expanduser().resolve() != want:
        raise PinError(f"llama-server at {url} serves {got}, not the frozen base GGUF {want}")


class A4Arm(A4CandidateArm):
    """The step-5 candidate arm with the frozen adapter and temperatures; test splits allowed."""

    name = "a4"
    val_only = False

    def __init__(self, config: dict[str, Any] | None = None, root: Path = DATA_ROOT,
                 pipeline=None, b1ex=None, path: Path = CONFIG_PATH,
                 repo: Path = REPO_ROOT) -> None:  # fmt: skip
        config = config if config is not None else load_config(path, repo, root)
        check_pins(config, path, repo, root)  # before any request
        if pipeline is None:
            check_server_model(config)
        super().__init__(config, root, pipeline, b1ex)  # slot and adapter guards
        t = config["a2"]["temperature"]
        self.run_params = {**self.run_params,
                           "a4_config_sha256": config["a4"]["config"]["sha256"],
                           "temperature_stage1": repr(t["stage1"]),
                           "temperature_stage2": repr(t["stage2"])}  # fmt: skip
