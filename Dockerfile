# Application image. The model server (llama.cpp) runs as its own service;
# Docker Compose wires them together in Phase 4.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Placeholder until the FastAPI service exists (Phase 4)
CMD ["python", "-c", "import hs_rag_classifier as m; print('hs-rag-classifier', m.__version__)"]
