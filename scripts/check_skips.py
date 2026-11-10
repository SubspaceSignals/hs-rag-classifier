#!/usr/bin/env python3
"""CI guard (stdlib only): fail if pytest skipped more tests than expected, so a test that starts
skipping (a missing group, a broken import, a resource check gone wrong) cannot pass unnoticed.

    python3 scripts/check_skips.py JUNIT_XML --max-skipped N [--min-run M]

Reads pytest's --junitxml report. Fails (exit 1) if skipped > N, if fewer than M tests ran
(passed, failed or errored), or if the report has no test cases. Each skip is listed with its
reason, so a failing guard says which tests to look at. The expected N for each workflow is set
next to its pytest step, with the skips it allows.
"""

import argparse
import sys
import xml.etree.ElementTree as ET


def counts(path: str) -> tuple[int, int, list[str]]:
    """(tests that ran, skipped, ["file::name: reason", ...]) from a junit XML report."""
    ran, skipped = 0, []
    for case in ET.parse(path).iter("testcase"):
        skip = case.find("skipped")
        if skip is None:
            ran += 1
            continue
        where = f"{case.get('classname', '')}::{case.get('name', '')}"
        skipped.append(f"{where}: {skip.get('message', '').strip()}")
    return ran, len(skipped), skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("junit_xml")
    ap.add_argument("--max-skipped", type=int, required=True)
    ap.add_argument("--min-run", type=int, default=1)
    a = ap.parse_args(argv)
    ran, n_skipped, skipped = counts(a.junit_xml)
    print(f"skip guard: {ran} ran, {n_skipped} skipped (max {a.max_skipped}, min run {a.min_run})")
    for line in skipped:
        print(f"  skipped {line}")
    if ran + n_skipped == 0:
        print("skip guard: FAIL, the report has no test cases", file=sys.stderr)
        return 1
    if n_skipped > a.max_skipped:
        print(f"skip guard: FAIL, {n_skipped} skipped > {a.max_skipped}", file=sys.stderr)
        return 1
    if ran < a.min_run:
        print(f"skip guard: FAIL, only {ran} tests ran (< {a.min_run})", file=sys.stderr)
        return 1
    print("skip guard: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
