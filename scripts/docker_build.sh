#!/usr/bin/env bash
# Build both images (compose.yaml) from a clean tree, with the commit baked into the API image.
#
#   scripts/docker_build.sh [docker compose build args, e.g. api]
#
# Refuses a dirty tree (untracked files included), so the image's HS_RAG_GIT_DIRTY is always
# false and its HS_RAG_GIT_COMMIT names exactly the source it was built from. Then run the
# stack with: GIT_COMMIT=$(git rev-parse HEAD) docker compose up -d
set -euo pipefail
cd "$(dirname "$0")/.."

GIT_COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to build: the tree is dirty (commit or stash first):" >&2
  git status --porcelain >&2
  exit 1
fi
GIT_DIRTY=false
if [ -e .env ] && ! git check-ignore -q .env; then
  echo "refusing to build: .env exists and is not ignored" >&2
  exit 1
fi
export GIT_COMMIT GIT_DIRTY
echo "building from ${GIT_COMMIT} (dirty=${GIT_DIRTY})"
docker compose build "$@"
