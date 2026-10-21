"""CLI: run an arm on a frozen set, or list the logged test-set runs (see the package doc)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hs_rag_classifier.eval import runner as R
from hs_rag_classifier.eval.items import DATA_ROOT, DATASETS, load
from hs_rag_classifier.eval.metrics import HEADLINE_REVIEW_RATE


def cmd_run(args: argparse.Namespace) -> int:
    split = args.split or "val"
    try:
        evalset = load(args.dataset, split, args.with_keywords, root=args.root)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    valid8, valid10 = R.valid_codes(args.root)
    arm = R.make_arm(args.arm, valid8)
    try:
        result = R.run_eval(arm, evalset, valid8, valid10, limit=args.limit,
                            review_rate=args.review_rate)  # fmt: skip
    except R.GuardError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print("\n".join(R.summary_lines(result, arm, evalset)))
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
    run.add_argument("--root", type=Path, default=DATA_ROOT, help="data directory")
    run.set_defaults(func=cmd_run)
    tr = sub.add_parser("test-runs", help="list logged test-set runs and the count per arm")
    tr.add_argument("--log", type=Path, default=R.TEST_LOG)
    tr.set_defaults(func=cmd_test_runs)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
