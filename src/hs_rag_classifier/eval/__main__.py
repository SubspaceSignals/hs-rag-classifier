"""CLI: run an arm on a frozen set, or list the logged test-set runs (see the package doc)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hs_rag_classifier import tracing
from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.items import DATA_ROOT, DATASETS, load
from hs_rag_classifier.eval.metrics import HEADLINE_REVIEW_RATE
from hs_rag_classifier.pipeline import llm


def cmd_run(args: argparse.Namespace) -> int:
    split = args.split or "val"
    if split == "test" and tracing.requested(args.trace):
        print(f"Refused: {tracing.TEST_REFUSED}", file=sys.stderr)
        return 2
    if split == "test" and llm.multislot_requested(args.allow_multislot):
        print(f"Refused: {llm.MULTISLOT_TEST_REFUSED}", file=sys.stderr)
        return 2
    tracing.enable(args.trace)
    llm.allow_multislot(args.allow_multislot)
    try:
        evalset = load(args.dataset, split, args.with_keywords, root=args.root)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    valid8, valid10 = R.valid_codes(args.root)
    try:
        arm = R.make_arm(args.arm, valid8, adapter=args.adapter,
                         adapter_sha256=args.adapter_sha256, candidate=args.candidate)  # fmt: skip
    except (llm.ServerSlotsError, llm.AdapterError, ValueError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    try:
        result = R.run_eval(arm, evalset, valid8, valid10, limit=args.limit,
                            review_rate=args.review_rate, resume=args.resume,
                            progress=args.progress)  # fmt: skip
    except R.GuardError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print("\n".join(R.summary_lines(result, arm, evalset) + R.arm_summary_lines(result)))
    print(f"  MLflow: {R.tracking_uri()} (experiment {result['experiment']})")
    return 0


def cmd_test_runs(args: argparse.Namespace) -> int:
    runs = R.read_test_runs(args.log)
    for r in runs:
        print(f"{r['time']}  {r['arm']:<12} {r['config_hash']}  {r['commit'][:12]}  "
              f"{r['dataset']:<8} {r['split_sha256'][:12]}  {r['mlflow_run_id']}")  # fmt: skip
    print(f"{len(runs)} test-set runs")
    for arm, n in sorted(R.count_test_runs(runs).items()):
        print(f"  {arm}: {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run an arm on a dataset split and log it to MLflow")
    run.add_argument("--arm", required=True, help=f"one of {', '.join(sorted(R.ARMS))}")
    run.add_argument("--dataset", required=True, choices=DATASETS)
    run.add_argument("--split", choices=("val", "test"), help="default val (rulings only)")
    run.add_argument("--with-keywords", action="store_true", help="ablation: append keywords")
    run.add_argument("--limit", type=int, help="first N items (not allowed on test)")
    run.add_argument("--review-rate", type=float, default=HEADLINE_REVIEW_RATE,
                     help="review rate for the headline metric")  # fmt: skip
    run.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="continue an interrupted run under the same MLflow run id",
    )
    run.add_argument(
        "--progress", type=int, default=0, metavar="N", help="print progress every N predict calls"
    )
    run.add_argument("--root", type=Path, default=DATA_ROOT, help="data directory")
    run.add_argument("--trace", action="store_true", help="send traces to Phoenix (not on test)")
    run.add_argument("--allow-multislot", action="store_true",
                     help="dev only: continue on a llama-server without exactly one slot "
                     "(not on test)")  # fmt: skip
    run.add_argument("--adapter", type=Path, help="a4_candidate only: the candidate's LoRA GGUF")
    run.add_argument("--adapter-sha256", help="a4_candidate only: that GGUF's SHA-256")
    run.add_argument("--candidate", help="a4_candidate only: step_00250, step_00500 or final")
    run.set_defaults(func=cmd_run)
    tr = sub.add_parser("test-runs", help="list logged test-set runs and the count per arm")
    tr.add_argument("--log", type=Path, default=R.TEST_LOG)
    tr.set_defaults(func=cmd_test_runs)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
