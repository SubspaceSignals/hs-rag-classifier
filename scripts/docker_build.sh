#!/usr/bin/env bash
# Build the compose.yaml images whose tags do not exist yet, from a clean tree. Tags never move.
#
#   scripts/docker_build.sh            # every service whose tag is missing (existing ones skipped)
#   scripts/docker_build.sh api        # one service by name; refuses if its tag already exists
#
# Tags (scripts/stack_env.sh): llama is hs-rag-classifier/llama:b11510-<short commit of the last
# change to docker/llama/Dockerfile>, api is hs-rag-classifier/api:<HEAD>. An existing tag is
# never rebuilt or re-pointed: compose would move it to the new image, and the containerd image
# store then drops the old record (run 1's llama 25980b37 went that way).
#
# Refuses a dirty tree (untracked files included), so the image's HS_RAG_GIT_DIRTY is always
# false and its HS_RAG_GIT_COMMIT names exactly the source it was built from. Start the stack
# with scripts/docker_up.sh (mount preflight first).
set -euo pipefail
# shellcheck source=stack_env.sh
. "$(dirname "$0")/stack_env.sh"

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
export GIT_DIRTY

image_for() {
  case "$1" in
    llama) echo "$LLAMA_IMAGE" ;;
    api) echo "$API_IMAGE" ;;
    *) echo "unknown service '$1' (llama or api)" >&2; return 1 ;;
  esac
}
tag_exists() { docker image inspect "$1" > /dev/null 2>&1; }

if [ "$#" -gt 1 ]; then
  echo "usage: $0 [llama|api]" >&2
  exit 1
fi
to_build=()
if [ "$#" -eq 1 ]; then
  img=$(image_for "$1")
  if tag_exists "$img"; then
    echo "refusing to build $1: $img already exists ($(docker image inspect -f '{{.Id}}' "$img"))" >&2
    exit 1
  fi
  to_build=("$1")
else
  for svc in llama api; do
    img=$(image_for "$svc")
    if tag_exists "$img"; then
      echo "skip $svc: $img already exists ($(docker image inspect -f '{{.Id}}' "$img")), not rebuilt"
    else
      to_build+=("$svc")
    fi
  done
fi
if [ "${#to_build[@]}" -eq 0 ]; then
  echo "nothing to build"
  exit 0
fi
# Check again right before the build, so nothing created in between gets overwritten.
for svc in "${to_build[@]}"; do
  img=$(image_for "$svc")
  if tag_exists "$img"; then echo "refusing: $img appeared before the build" >&2; exit 1; fi
done

echo "building ${to_build[*]} from ${GIT_COMMIT} (dirty=${GIT_DIRTY}, llama tag b11510-${LLAMA_DOCKERFILE_COMMIT})"
docker compose build "${to_build[@]}"
for svc in "${to_build[@]}"; do
  img=$(image_for "$svc")
  echo "built $svc: $img $(docker image inspect -f '{{.Id}}' "$img")"
done

# Build check: the API image's package must import as the image's own user (uid 1000). Run 1
# failed here (NTFS 0700 source folders copied unreadable). No network, no GPU. A throwaway
# container (--rm); the script removes no image, tag or other container.
if [[ " ${to_build[*]} " == *" api "* ]]; then
  echo "import check: $API_IMAGE as 1000:1000 (throwaway container, --rm)"
  docker run --rm --network none --user 1000:1000 "$API_IMAGE" \
    python -c "import hs_rag_classifier.service; print('import ok:', hs_rag_classifier.service.__file__)"
fi
