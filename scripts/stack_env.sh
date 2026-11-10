# Sourced by scripts/docker_build.sh and scripts/docker_up.sh (or by hand before docker compose
# stop/ps/logs): cd to the repo and export the image tags compose.yaml reads. Tags never move:
#   llama  hs-rag-classifier/llama:b11510-${LLAMA_DOCKERFILE_COMMIT}  (last change to docker/llama/Dockerfile)
#   api    hs-rag-classifier/api:${GIT_COMMIT}                       (HEAD)
cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1
GIT_COMMIT=$(git rev-parse HEAD) || return 1
LLAMA_DOCKERFILE_COMMIT=$(git log -1 --format=%h -- docker/llama/Dockerfile) || return 1
if [ -z "$LLAMA_DOCKERFILE_COMMIT" ]; then
  echo "docker/llama/Dockerfile has no commit" >&2
  return 1
fi
LLAMA_IMAGE="hs-rag-classifier/llama:b11510-${LLAMA_DOCKERFILE_COMMIT}"
API_IMAGE="hs-rag-classifier/api:${GIT_COMMIT}"
export GIT_COMMIT LLAMA_DOCKERFILE_COMMIT LLAMA_IMAGE API_IMAGE
