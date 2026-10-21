"""Evaluation harness: frozen-set loaders, arm interface, metrics, and the MLflow runner.

Usage:
    uv run python -m hs_rag_classifier.eval run --arm <name> --dataset rulings|atoz|hand
        [--split val|test] [--with-keywords] [--limit N]
    uv run python -m hs_rag_classifier.eval test-runs

Validation is the default split. Test runs refuse a dirty git tree, --limit and the sanity
arms, and each one is appended to ~/data/hs-rag-classifier/eval/test_runs.jsonl.
"""
