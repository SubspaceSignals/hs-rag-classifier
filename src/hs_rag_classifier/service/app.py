"""FastAPI app: POST /classify (one item) and GET /health. Localhost only; advisory output.

create_app(pipeline=None, settings=None): with no pipeline, the lifespan loads the frozen A4
service once (core.A4Service.from_frozen, which refuses on a wrong pin); tests pass a fake with
the same classify(description, request_id) and health() methods. One classification at a time
(a lock around the pipeline call, matching llama-server's single slot); requests queue.

Logs carry request_id, latency, needs_review and review_reasons, never the description.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from hs_rag_classifier.service import core
from hs_rag_classifier.service.schemas import ClassifyRequest, ClassifyResponse, Health

log = logging.getLogger("hs_rag_classifier.service")


def create_app(pipeline: Any = None, settings: core.Settings | None = None) -> FastAPI:
    settings = settings or core.Settings.from_env()
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            app.state.pipeline = core.A4Service.from_frozen(settings)  # StartupError stops start-up
        log.info(
            "service ready on %s:%s (llama-server %s)",
            settings.host,
            settings.port,
            settings.llama_url,
        )
        yield

    app = FastAPI(title="hs-rag-classifier", version="0.1.0", lifespan=lifespan,
                  description="Suggests UK commodity codes (cn8) for one product description. "
                              "Advisory only, not a legal ruling.")  # fmt: skip
    app.state.pipeline = pipeline
    app.state.settings = settings

    @app.post("/classify", response_model=ClassifyResponse, responses={503: {"model": Health}})
    def classify(req: ClassifyRequest):
        request_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        pipe = app.state.pipeline
        if pipe is None:
            return JSONResponse(
                status_code=503, content={"status": "unavailable", "reason": "pipeline not loaded"}
            )
        try:
            with lock:
                out = pipe.classify(req.description, request_id)
        except core.UpstreamError as exc:
            log.warning("request_id=%s status=503 latency_ms=%.0f error=%s", request_id,
                        (time.perf_counter() - t0) * 1000, exc)  # fmt: skip
            return JSONResponse(
                status_code=503, content={"status": "unavailable", "reason": str(exc)}
            )
        except Exception as exc:  # noqa: BLE001 - never a stack trace to the client
            log.error("request_id=%s status=500 latency_ms=%.0f error=%s", request_id,
                      (time.perf_counter() - t0) * 1000, type(exc).__name__)  # fmt: skip
            return JSONResponse(
                status_code=500, content={"status": "unavailable", "reason": "internal error"}
            )
        log.info("request_id=%s status=200 latency_ms=%.0f needs_review=%s review_reasons=%s", request_id,
                 (time.perf_counter() - t0) * 1000, out["needs_review"], ",".join(out["review_reasons"]))  # fmt: skip
        return out

    @app.get("/health", response_model=Health, responses={503: {"model": Health}})
    def health():
        pipe = app.state.pipeline
        if pipe is None:
            return JSONResponse(
                status_code=503, content={"status": "unavailable", "reason": "pipeline not loaded"}
            )
        try:
            ok, reason = pipe.health()
        except Exception as exc:  # noqa: BLE001 - a failing check is a 503, not a 500
            ok, reason = False, f"health check failed: {type(exc).__name__}"
        if not ok:
            return JSONResponse(
                status_code=503, content={"status": "unavailable", "reason": reason}
            )
        return {"status": "ok"}

    return app
