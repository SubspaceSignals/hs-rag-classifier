#!/usr/bin/env bash
# Start the stack (or one service) from existing images: the mount preflight first, then
#   docker compose up -d --no-build --pull never [service]
# so up never builds, pulls or re-tags anything. Refuses to start if scripts/preflight_mounts.py
# finds a symlink under a mounted folder that is broken or resolves outside that service's mounts,
# or if an image tag is missing (build it with scripts/docker_build.sh).
#
#   scripts/docker_up.sh [llama|api]
#   HS_API_TAG=<existing api tag> scripts/docker_up.sh   # run that api image instead of HEAD's
# Stop with: . scripts/stack_env.sh && docker compose stop
set -euo pipefail
# shellcheck source=stack_env.sh
. "$(dirname "$0")/stack_env.sh"
python3 -I scripts/preflight_mounts.py --compose compose.yaml
for svc in ${1:-llama api}; do
  case "$svc" in llama) img=$LLAMA_IMAGE ;; api) img=$API_IMAGE ;; *) echo "unknown service '$svc'" >&2; exit 1 ;; esac
  docker image inspect "$img" > /dev/null 2>&1 || { echo "refusing to start: $img does not exist" >&2; exit 1; }
  if [ "$svc" = api ]; then
    built=$(docker image inspect -f '{{index .Config.Labels "hs_rag_classifier.git_commit"}}' "$img")
    echo "api image: $img (tag from ${API_TAG_SOURCE}; built from commit ${built:-unknown}, which the service reports)"
  else
    echo "llama image: $img"
  fi
done
docker compose up -d --no-build --pull never "$@"
