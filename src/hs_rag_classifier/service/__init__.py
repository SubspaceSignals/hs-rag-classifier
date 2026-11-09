"""Local HTTP service over the frozen A4 pipeline (Phase 4): POST /classify, GET /health.

Run: python -m hs_rag_classifier.service (uvicorn, one worker, HOST:PORT, default 127.0.0.1:8000;
LLAMA_URL default http://127.0.0.1:8080). Needs the llm, ml and serve dependency groups and a
llama-server started with the frozen A4 command (configs/a4_pipeline.json).
"""
