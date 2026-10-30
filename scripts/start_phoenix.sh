#!/usr/bin/env bash
# Start the local Phoenix trace viewer at http://localhost:6006 (see Tracing in the README).
# Data stays outside the repo. Host and external resources are set here because Phoenix 14.6
# defaults to binding every interface and to loading Google Fonts.
set -euo pipefail

export PHOENIX_WORKING_DIR="${PHOENIX_WORKING_DIR:-$HOME/data/hs-rag-classifier/phoenix}"
export PHOENIX_HOST=127.0.0.1
export PHOENIX_TELEMETRY_ENABLED=false
export PHOENIX_ALLOW_EXTERNAL_RESOURCES=false

mkdir -p "$PHOENIX_WORKING_DIR"
venv="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/hs-rag-classifier}"
exec "$venv/bin/python" -m phoenix.server.main serve
