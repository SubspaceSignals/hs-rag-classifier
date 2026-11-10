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

# Build check: the API image's package must import as the image's own user (uid 1000). Run 1
# failed here (NTFS 0700 source folders copied unreadable). No network, no GPU, removed after.
API_IMAGE="hs-rag-classifier/api:${GIT_COMMIT}"
if docker image inspect "$API_IMAGE" > /dev/null 2>&1; then
  echo "import check: $API_IMAGE as 1000:1000"
  docker run --rm --network none --user 1000:1000 "$API_IMAGE" \
    python -c "import hs_rag_classifier.service; print('import ok:', hs_rag_classifier.service.__file__)"
fi
