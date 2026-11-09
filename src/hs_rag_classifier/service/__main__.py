"""python -m hs_rag_classifier.service: uvicorn with one worker on HOST:PORT (127.0.0.1:8000)."""

import logging

import uvicorn

from hs_rag_classifier.service.app import create_app
from hs_rag_classifier.service.core import Settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
