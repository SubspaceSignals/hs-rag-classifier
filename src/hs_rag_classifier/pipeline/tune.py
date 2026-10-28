"""Set up and tune A1 on rulings validation (llama-server must be running).

    uv run python -m hs_rag_classifier.pipeline.tune server     # build, model hash, labels
    uv run python -m hs_rag_classifier.pipeline.tune choose-m   # M from the stage-1 budget
    uv run python -m hs_rag_classifier.pipeline.tune sample     # the fixed 300-item val sample
    uv run python -m hs_rag_classifier.pipeline.tune run --profile off --evidence on
    uv run python -m hs_rag_classifier.pipeline.tune write-config --profile off --evidence on
    uv run python -m hs_rag_classifier.pipeline.tune compare RUN_A RUN_B

server: llama.cpp build and model path from /props, the GGUF's SHA-256, and the option labels
that are single tokens (checks.verify_labels against the server's /tokenize). Saved to
eval/a1_server.json.

choose-m: for every validation item, the full stage-1 prompt (chat template included, counted
by the server) with the evidence block on and a reserve for the profile block
(PROFILE_RESERVE tokens, the profile call's max_tokens plus its tags), at M options. M is the
largest number, up to the number of verified labels, whose p95 fits the token budget. Also
HS6 recall at M (the label's HS6 among the options), with a cluster-bootstrap interval.

sample: the first SAMPLE_N validation items after a seeded shuffle (SAMPLE_SEED) of the item
ids in file order; ids written to configs/a1_val_sample_ids.txt.

run: one toggle setting on the sample (default) or on all of validation (--full), logged to
MLflow like any run (tags a1_sample, a1_profile, a1_evidence), plus the A1 diagnostics.

compare: paired cluster-bootstrap difference in top-1 at 8 digits between two runs on the same
items (metrics.paired_difference).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hs_rag_classifier.arms import a1_pipeline as A
from hs_rag_classifier.data.nomenclature import sha256
from hs_rag_classifier.eval import metrics as M
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.items import DATA_ROOT, REPO_ROOT, load
from hs_rag_classifier.pipeline import checks
from hs_rag_classifier.pipeline.llm import ServerInfo

SERVER_JSON = DATA_ROOT / "eval" / "a1_server.json"
CHOOSE_M_JSON = DATA_ROOT / "eval" / "a1_choose_m.json"
SAMPLE_FILE = REPO_ROOT / "configs" / "a1_val_sample_ids.txt"
SAMPLE_N = 300
SAMPLE_SEED = 20261020
PROFILE_RESERVE = 220
GGUF_SHA_FILE = DATA_ROOT / "serving" / "gguf.sha256"
SERVER = {
    "hf_repo": "unsloth/Qwen3.5-9B-GGUF",
    "quantisation": "UD-Q4_K_XL",
    # --cache-ram 0: the default 8 GiB prompt cache in system RAM grew about 250 MB an item
    # and got llama-server OOM-killed on this 14 GiB machine (28 Oct 2026)
    "command": "~/llama.cpp/build/bin/llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL "
    "-ngl 99 -c 16384 --port 8080 --jinja --cache-ram 0",
    "cache_ram_mib": 0,
    "n_ctx": 16384,
}


def _on(v: str) -> bool:
    return v == "on"


def gguf_sha256(path: str) -> str:
    """The GGUF's SHA-256: from the saved sha256sum line when it names this file, else computed."""
    if GGUF_SHA_FILE.exists():
        sha, name = GGUF_SHA_FILE.read_text().split(maxsplit=1)
        if Path(name.strip().lstrip("*")).resolve() == Path(path).resolve():
            return sha
    out = subprocess.run(["sha256sum", path], capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def cmd_server(args: argparse.Namespace) -> int:
    info = ServerInfo(args.base_url)
    if not info.up():
        print("llama-server is not answering", file=sys.stderr)
        return 2
    props = info.props()
    labels = checks.verify_labels(info.tokenize)
    rejected = [c for c in checks.CANDIDATE_LABELS if c not in labels]
    out = {**SERVER, "llama_cpp_build": props.get("build_info"),
           "model_file": Path(props["model_path"]).name,
           "model_snapshot": Path(props["model_path"]).parent.name,
           "model_sha256": gguf_sha256(str(Path(props["model_path"]).resolve())),
           "n_slots": props.get("total_slots"),
           "n_ctx_slot": props["default_generation_settings"].get("n_ctx"),
           "labels": labels, "labels_rejected": rejected}  # fmt: skip
    SERVER_JSON.parent.mkdir(parents=True, exist_ok=True)
    SERVER_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "labels"}, indent=2))
    print(f"{len(labels)} verified single-token labels: {''.join(labels)}")
    return 0


def draft_config(profile: bool, evidence: bool, m: int | None = None) -> dict:
    server = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    if m is None:
        m = json.loads(CHOOSE_M_JSON.read_text(encoding="utf-8"))["m"]
    cfg = {**A.DEFAULTS, "profile": profile, "evidence": evidence, "m": m,
           "labels": server["labels"],
           "server": {k: v for k, v in server.items() if k not in ("labels",)},
           **A.file_entries()}  # fmt: skip
    if SAMPLE_FILE.exists():
        cfg["val_sample"] = {"path": str(SAMPLE_FILE.relative_to(REPO_ROOT)),
                             "sha256": sha256(SAMPLE_FILE), "n": SAMPLE_N, "seed": SAMPLE_SEED}  # fmt: skip
    return cfg


def cmd_choose_m(args: argparse.Namespace) -> int:
    from hs_rag_classifier.pipeline.graph import Pipeline
    from hs_rag_classifier.pipeline.options import load_tariff
    from hs_rag_classifier.pipeline.validate import EvidenceIndex, Validator
    from hs_rag_classifier.retrieval.hybrid import HybridRetriever
    from hs_rag_classifier.retrieval.hybrid import load_config as load_hybrid

    info = ServerInfo(args.base_url)
    labels = json.loads(SERVER_JSON.read_text(encoding="utf-8"))["labels"]
    cfg = {**A.DEFAULTS, "profile": False, "evidence": True, "m": len(labels), "labels": labels}
    tariff = load_tariff()
    pipe = Pipeline(cfg, None, HybridRetriever(load_hybrid()), tariff, Validator(tariff),
                    EvidenceIndex())  # fmt: skip
    items = load("rulings", "val").items
    rows = []
    for n, item in enumerate(items, start=1):
        s = {"raw": item.text, "flags": []}
        s.update(pipe.input_check(s))
        s.update(pipe.retrieve(s))
        s.update(pipe.evidence(s))
        hs6s = pipe.stage1_options(s["fused"])
        counts = {}
        for m in args.grid:
            system, user, _ = pipe.stage1_prompt(s, hs6s[:m])
            counts[m] = info.prompt_tokens(system, user) + PROFILE_RESERVE
        rank = hs6s.index(item.label_code[:6]) + 1 if item.label_code[:6] in hs6s else None
        rows.append({"item_id": item.item_id, "cluster_id": item.cluster_id, "rank": rank,
                     "n_options": len(hs6s), "tokens": counts})  # fmt: skip
        if n % 100 == 0:
            print(f"  {n} of {len(items)}", flush=True)
    p95 = {m: float(np.percentile([r["tokens"][m] for r in rows], 95)) for m in args.grid}
    fits = [m for m in args.grid if p95[m] <= args.budget]
    m = max(fits) if fits else None
    frame = pd.DataFrame({"cluster_id": [r["cluster_id"] for r in rows]})
    out = {"budget": args.budget, "profile_reserve": PROFILE_RESERVE, "grid": args.grid,
           "p95_tokens": p95,
           "mean_tokens": {mm: float(np.mean([r["tokens"][mm] for r in rows])) for mm in args.grid},
           "max_tokens": {mm: int(max(r["tokens"][mm] for r in rows)) for mm in args.grid},
           "m": m, "budget_binds": m is not None and m < len(labels), "n_labels": len(labels),
           "hs6_recall": {}}  # fmt: skip
    for mm in args.grid:
        frame["hit"] = [float(r["rank"] is not None and r["rank"] <= mm) for r in rows]
        ci = M.bootstrap_intervals(frame, {"r": lambda a: float(a["hit"].mean())})["r"]
        out["hs6_recall"][mm] = {"value": float(frame["hit"].mean()), **ci}
    CHOOSE_M_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for mm in args.grid:
        r = out["hs6_recall"][mm]
        print(f"M {mm:>2}: p95 {p95[mm]:,.0f} tokens (mean {out['mean_tokens'][mm]:,.0f}, max "
              f"{out['max_tokens'][mm]:,}); HS6 recall {r['value']:.3f} "
              f"[{r['low']:.3f}, {r['high']:.3f}]")  # fmt: skip
    print(f"Chosen M = {m} (budget {args.budget:,} at p95; {len(labels)} labels)")
    return 0


def sample_ids() -> list[str]:
    ids = [i.item_id for i in load("rulings", "val").items]
    random.Random(SAMPLE_SEED).shuffle(ids)
    return ids[:SAMPLE_N]


def cmd_sample(args: argparse.Namespace) -> int:
    ids = sample_ids()
    SAMPLE_FILE.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"{len(ids)} ids written to {SAMPLE_FILE.relative_to(REPO_ROOT)} "
          f"(sha256 {sha256(SAMPLE_FILE)[:12]})")  # fmt: skip
    return 0


def read_sample() -> list[str]:
    return [x for x in SAMPLE_FILE.read_text(encoding="utf-8").splitlines() if x]


def cmd_run(args: argparse.Namespace) -> int:
    profile, evidence = _on(args.profile), _on(args.evidence)
    if args.use_config:
        cfg = A.load_config()
        if (cfg["profile"], cfg["evidence"]) != (profile, evidence):
            raise SystemExit("toggles differ from configs/a1_pipeline.json")
    else:
        cfg = draft_config(profile, evidence)
    arm = A.A1PipelineArm(cfg)
    evalset = load("rulings", "val")
    tags = {"a1_profile": args.profile, "a1_evidence": args.evidence}
    if not args.full:
        keep = set(read_sample())
        evalset = evalset.model_copy(update={"items": [i for i in evalset.items
                                                       if i.item_id in keep]})  # fmt: skip
        tags["a1_sample"] = f"val{SAMPLE_N}:{sha256(SAMPLE_FILE)[:12]}"
    else:
        tags["a1_sample"] = "full"
    valid8, valid10 = R.valid_codes()
    result = R.run_eval(arm, evalset, valid8, valid10, tags=tags, resume=args.resume,
                        progress=args.progress)  # fmt: skip
    print("\n".join(R.summary_lines(result, arm, evalset)))
    diag = A.diagnostics(result["frame"])
    print(json.dumps(diag, indent=2))
    out = DATA_ROOT / "eval" / f"a1_{result['run_id']}_diagnostics.json"
    out.write_text(json.dumps({"run_id": result["run_id"], "tags": tags, **diag}, indent=2))
    return 0


def cmd_write_config(args: argparse.Namespace) -> int:
    cfg = draft_config(_on(args.profile), _on(args.evidence))
    cfg["choice"] = args.note
    A.CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {A.CONFIG_PATH.relative_to(REPO_ROOT)}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = R.load_run_frame(args.run_a), R.load_run_frame(args.run_b)
    res = M.paired_difference(a, b, f"correct_{args.depth}")
    print(json.dumps(res, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline.tune", description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=A.DEFAULTS["client"]["base_url"])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("server").set_defaults(func=cmd_server)
    cm = sub.add_parser("choose-m")
    cm.add_argument("--budget", type=int, default=A.DEFAULTS["token_budget"])
    cm.add_argument("--grid", type=int, nargs="+", default=[10, 20, 26, 30, 40, 52])
    cm.set_defaults(func=cmd_choose_m)
    sub.add_parser("sample").set_defaults(func=cmd_sample)
    for name, func in (("run", cmd_run), ("write-config", cmd_write_config)):
        p = sub.add_parser(name)
        p.add_argument("--profile", choices=("on", "off"), required=True)
        p.add_argument("--evidence", choices=("on", "off"), required=True)
        p.set_defaults(func=func)
        if name == "run":
            p.add_argument("--full", action="store_true", help="all of validation")
            p.add_argument("--use-config", action="store_true",
                           help="use configs/a1_pipeline.json (must match the toggles)")  # fmt: skip
            p.add_argument("--resume", metavar="RUN_ID")
            p.add_argument("--progress", type=int, default=50)
        else:
            p.add_argument("--note", default="")
    cp = sub.add_parser("compare")
    cp.add_argument("run_a")
    cp.add_argument("run_b")
    cp.add_argument("--depth", type=int, default=8)
    cp.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
