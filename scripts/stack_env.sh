# Sourced by scripts/docker_build.sh and scripts/docker_up.sh (or by hand before docker compose
# stop/ps/logs): cd to the repo and export the image tags compose.yaml reads. Tags never move:
#   llama  hs-rag-classifier/llama:b11510-${LLAMA_DOCKERFILE_COMMIT}  (last change to docker/llama/Dockerfile)
#   api    hs-rag-classifier/api:${API_TAG}   (HEAD's commit, or HS_API_TAG if set)
#
# HS_API_TAG (optional) names an existing api image tag to run instead of HEAD's, for example the
# image that passed a parity run: HS_API_TAG=<tag> scripts/docker_up.sh. The service reports the
# commit baked into that image at build time (HS_RAG_GIT_COMMIT, label hs_rag_classifier.git_commit),
# never HEAD's, so the override cannot misstate provenance. scripts/docker_build.sh refuses while it
# is set: builds are always of HEAD (GIT_COMMIT).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1
GIT_COMMIT=$(git rev-parse HEAD) || return 1
LLAMA_DOCKERFILE_COMMIT=$(git log -1 --format=%h -- docker/llama/Dockerfile) || return 1
if [ -z "$LLAMA_DOCKERFILE_COMMIT" ]; then
  echo "docker/llama/Dockerfile has no commit" >&2
  return 1
fi
LLAMA_IMAGE="hs-rag-classifier/llama:b11510-${LLAMA_DOCKERFILE_COMMIT}"
if [ -n "${HS_API_TAG:-}" ]; then
  if [[ ! "$HS_API_TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
    echo "HS_API_TAG '$HS_API_TAG' is not a valid image tag" >&2
    return 1
  fi
  API_TAG=$HS_API_TAG
  API_TAG_SOURCE="HS_API_TAG override"
else
  API_TAG=$GIT_COMMIT
  API_TAG_SOURCE=HEAD
fi
API_IMAGE="hs-rag-classifier/api:${API_TAG}"
export GIT_COMMIT LLAMA_DOCKERFILE_COMMIT LLAMA_IMAGE API_TAG API_TAG_SOURCE API_IMAGE
