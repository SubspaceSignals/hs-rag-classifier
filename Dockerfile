# The API image: python -m hs_rag_classifier.service (FastAPI over the frozen A4 pipeline).
# llama-server is its own image (docker/llama/Dockerfile); compose.yaml wires the two together.
# Build with scripts/docker_build.sh, which refuses a dirty tree and passes GIT_COMMIT/GIT_DIRTY.
#
# No .git, .env or data in the image (.dockerignore); models, indexes, processed data and the
# tariff cache are read-only bind mounts at the host's absolute paths (config_hash 5da7ac1452b3
# needs the adapter at its recorded ${HOME}/data/... path). torch is the cu128 wheel from
# uv.lock, which carries its own CUDA runtime; only the driver comes from the nvidia runtime.

FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

COPY --from=uv /uv /uvx /bin/

ARG APP_HOME=/home/cal
RUN groupadd -g 1000 app \
 && useradd -u 1000 -g 1000 -d "$APP_HOME" -m -s /usr/sbin/nologin app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, for layer caching: the exact uv.lock versions of base + llm, ml, serve.
# --no-cache everywhere: uv's download cache would otherwise stay in the layer (~8 GB).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group llm --group ml --group serve --no-install-project --no-cache

# The package (editable, so REPO_ROOT is /app) and the configs it pins by SHA-256.
COPY README.md LICENSE ./
COPY src ./src
COPY configs ./configs
# The repo sits on NTFS, where some folders are 0700 and some files 0600; COPY keeps those modes
# (owner root), so the app user (uid 1000) could not read the package. Readable by all, dirs
# traversable, nothing made executable that was not.
RUN chmod -R a+rX /app/pyproject.toml /app/uv.lock /app/README.md /app/LICENSE /app/src /app/configs
RUN uv sync --frozen --no-dev --group llm --group ml --group serve --no-cache

ARG GIT_COMMIT
ARG GIT_DIRTY
RUN test -n "$GIT_COMMIT" && test "$GIT_DIRTY" = false \
 || { echo "build with scripts/docker_build.sh from a clean tree (GIT_COMMIT=$GIT_COMMIT GIT_DIRTY=$GIT_DIRTY)"; exit 1; }

LABEL org.opencontainers.image.title="hs-rag-classifier api" \
      org.opencontainers.image.source="https://github.com/SubspaceSignals/hs-rag-classifier" \
      org.opencontainers.image.revision="$GIT_COMMIT" \
      hs_rag_classifier.git_commit="$GIT_COMMIT" \
      hs_rag_classifier.git_dirty="$GIT_DIRTY"

ENV PATH=/app/.venv/bin:$PATH \
    HOME=$APP_HOME \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HUB_OFFLINE=1 \
    TARIFF_API_OFFLINE=1 \
    HS_RAG_GIT_COMMIT=$GIT_COMMIT \
    HS_RAG_GIT_DIRTY=$GIT_DIRTY

USER 1000:1000
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=300s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["python", "-m", "hs_rag_classifier.service"]
