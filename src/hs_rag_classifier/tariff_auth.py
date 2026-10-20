"""Trade Tariff Developer Portal credentials: load from .env and fetch an OAuth token.

Usage (prints no secrets):
    uv run python -m hs_rag_classifier.tariff_auth
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("TRADE_TARIFF_TOKEN_URL", "TRADE_TARIFF_CLIENT_ID", "TRADE_TARIFF_CLIENT_SECRET")


def load_env(path: Path = REPO_ROOT / ".env") -> dict[str, str]:
    """Read KEY=VALUE lines from .env; real environment variables take precedence."""
    values: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in REQUIRED:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def fetch_token(env: dict[str, str], transport: httpx.BaseTransport | None = None) -> dict:
    """Client-credentials grant. Tries HTTP Basic auth, then credentials in the form body."""
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
    url = env["TRADE_TARIFF_TOKEN_URL"]
    cid, secret = env["TRADE_TARIFF_CLIENT_ID"], env["TRADE_TARIFF_CLIENT_SECRET"]
    with httpx.Client(timeout=30.0, transport=transport) as client:
        r = client.post(url, data={"grant_type": "client_credentials"}, auth=(cid, secret))
        if r.status_code in (400, 401):
            r = client.post(
                url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": cid,
                    "client_secret": secret,
                },
            )
    if r.status_code != 200:
        detail = r.text[:300].replace(secret, "***")
        raise RuntimeError(f"Token request failed: HTTP {r.status_code}: {detail}")
    return r.json()


def main() -> int:
    env = load_env()
    try:
        token = fetch_token(env)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    access = token.get("access_token", "")
    print("Token OK")
    print(f"  type: {token.get('token_type')}, expires in: {token.get('expires_in')} s")
    print(f"  scope: {token.get('scope', '(none returned)')}")
    print(f"  token length: {len(access)} chars (not shown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
