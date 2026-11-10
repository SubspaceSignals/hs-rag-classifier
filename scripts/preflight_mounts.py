#!/usr/bin/env python3
"""Pre-start check for compose.yaml's bind mounts (stdlib only); scripts/docker_up.sh runs it
before every docker compose up.

    python3 scripts/preflight_mounts.py [--compose compose.yaml] [--home DIR]

For each service, walks every mounted host folder and resolves every symlink in it hop by hop,
as the container would see it: only the service's own mounts exist there. Fails (exit 1, listing
each path) if a symlink is broken, loops, or resolves (at any hop) outside that service's mount
set. Run 2 of Phase 4 step 4b stopped at API start-up because the HF cache's repo folders hold
relative links into ~/.cache/huggingface/hub/blobs, which the api did not mount.

compose.yaml is parsed by a small reader of exactly the layout it uses (services: <name>:
volumes: short-syntax "- src:dst:mode" items); anything else under volumes is an error, never
skipped. Only ${HOME} is expanded; any other ${...} in a mount is an error. Mounts are expected
at the same path on both sides (tests/test_compose.py checks that), so a host path is also the
container path.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

MAX_HOPS = 40  # Linux's own symlink limit


class ComposeFormatError(ValueError):
    pass


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def parse_volumes(text: str) -> dict[str, list[str]]:
    """{service: [volume strings]} from compose.yaml text (top-level services: block only)."""
    out: dict[str, list[str]] = {}
    in_services = False
    svc_indent = key_indent = None
    service = None
    in_volumes = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if indent == 0:
            in_services = body == "services:"
            service, in_volumes, svc_indent = None, False, None
            continue
        if not in_services:
            continue
        if svc_indent is None:
            svc_indent = indent
        if indent == svc_indent:
            if not body.endswith(":"):
                raise ComposeFormatError(f"line {lineno}: expected a service name: {raw!r}")
            service, in_volumes, key_indent = body[:-1].strip(), False, None
            out.setdefault(service, [])
            continue
        if service is None or indent < svc_indent:
            raise ComposeFormatError(f"line {lineno}: unexpected indentation: {raw!r}")
        if key_indent is None:
            key_indent = indent
        if indent == key_indent and not body.startswith("- "):
            key = body.split(":", 1)[0].strip()
            in_volumes = key == "volumes"
            if in_volumes and body.split(":", 1)[1].strip():
                raise ComposeFormatError(f"line {lineno}: volumes must be a block list: {raw!r}")
            continue
        if in_volumes:
            if indent <= key_indent:
                in_volumes = False
                continue
            if not body.startswith("- "):
                raise ComposeFormatError(f"line {lineno}: only short-syntax volume items: {raw!r}")
            item = _unquote(body[2:])
            if item.endswith(":") or item.startswith(("type", "source", "{")):
                raise ComposeFormatError(f"line {lineno}: only short-syntax volume items: {raw!r}")
            out[service].append(item)
    if not out:
        raise ComposeFormatError("no services: block found")
    return out


def expand_home(s: str, home: str) -> str:
    s = s.replace("${HOME}", home)
    if "$" in s:
        raise ComposeFormatError(f"only ${{HOME}} may be used in a mount: {s!r}")
    return s


def mounts_for(volumes: list[str], home: str) -> list[Path]:
    roots = []
    for v in volumes:
        parts = expand_home(v, home).split(":")
        if len(parts) not in (2, 3):
            raise ComposeFormatError(f"unexpected volume syntax: {v!r}")
        src, dst = parts[0], parts[1]
        if src != dst or not os.path.isabs(src):
            raise ComposeFormatError(f"mount must be an absolute path on both sides: {v!r}")
        roots.append(Path(os.path.normpath(src)))
    return roots


def _within(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip("/") + "/")


def _is_ancestor(p: str, root: str) -> bool:
    return p != root and (p == "/" or root.startswith(p.rstrip("/") + "/"))


def resolve_in_container(link: str, roots: list[Path]) -> tuple[bool, str, list[str]]:
    """Resolve an absolute path component by component as the container sees it.

    Inside a mount, host symlinks are followed (the container sees the same link). On a strict
    ancestor of a mount (e.g. ${HOME}), the container has a plain directory, so the walk goes on
    without consulting the host. Anything else does not exist in the container.
    Returns (ok, reason, hops).
    """
    rs = [str(r) for r in roots]
    hops: list[str] = []
    pending = [c for c in link.split("/") if c]
    cur = "/"
    n_links = 0
    while pending:
        comp = pending.pop(0)
        if comp == ".":
            continue
        if comp == "..":
            cur = os.path.dirname(cur) if cur != "/" else "/"
            continue
        cand = os.path.join(cur, comp)
        if any(_within(cand, r) for r in rs):
            try:
                st = os.lstat(cand)
            except FileNotFoundError:
                return False, f"broken: {cand} does not exist", hops
            if stat.S_ISLNK(st.st_mode):
                n_links += 1
                if n_links > MAX_HOPS:
                    return False, "symlink loop (more than 40 hops)", hops
                target = os.readlink(cand)
                hops.append(f"{cand} -> {target}")
                pending = [c for c in target.split("/") if c] + pending
                cur = "/" if target.startswith("/") else cur
                continue
            if pending and not stat.S_ISDIR(st.st_mode):
                return False, f"broken: {cand} is not a directory", hops
            cur = cand
        elif any(_is_ancestor(cand, r) for r in rs):
            cur = cand
        else:
            return False, f"outside the mount set: {cand}", hops
    return True, cur, hops


def check_service(volumes: list[str], home: str) -> tuple[int, list[str]]:
    """(symlinks checked, problems) for one service's mounts."""
    roots = mounts_for(volumes, home)
    problems: list[str] = []
    walked = [r for r in roots if not any(r != o and _within(str(r), str(o)) for o in roots)]
    n = 0
    for root in sorted(set(walked)):
        if os.path.islink(root):
            problems.append(f"{root}: the mount source itself is a symlink")
            continue
        if not root.exists():
            problems.append(f"{root}: the mount source does not exist")
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                p = os.path.join(dirpath, name)
                if not os.path.islink(p):
                    continue
                n += 1
                ok, why, hops = resolve_in_container(p, roots)
                if not ok:
                    problems.append(f"{p}: {why} (hops: {' | '.join(hops)})")
    return n, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--compose", default=str(Path(__file__).resolve().parent.parent / "compose.yaml")
    )
    ap.add_argument("--home", default=os.environ.get("HOME", ""))
    a = ap.parse_args(argv)
    if not a.home or not os.path.isabs(a.home):
        print("preflight: HOME is not an absolute path", file=sys.stderr)
        return 1
    try:
        services = parse_volumes(Path(a.compose).read_text(encoding="utf-8"))
        failed = False
        for name, vols in services.items():
            if not vols:
                print(f"preflight: {name}: no mounts")
                continue
            n, problems = check_service(vols, a.home)
            print(f"preflight: {name}: {len(vols)} mounts, {n} symlinks, {len(problems)} problems")
            for p in problems:
                print(f"  FAIL {p}", file=sys.stderr)
            failed |= bool(problems)
    except ComposeFormatError as e:
        print(f"preflight: cannot read {a.compose}: {e}", file=sys.stderr)
        return 1
    if failed:
        print("preflight: FAILED, refusing to start", file=sys.stderr)
        return 1
    print("preflight: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
