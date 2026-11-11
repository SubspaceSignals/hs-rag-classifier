# Running locally

## Decider server

```bash
~/llama.cpp/build/bin/llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL -ngl 99 -c 16384 --port 8080 --jinja --cache-ram 0 -np 1
```

Keep `-np 1` for evaluation: A2 output depended on llama-server slot and prompt-cache state, and `-np 1` plus `cache_prompt=false` (set in `pipeline/llm.py`) gives bit-identical runs. It serves one request at a time. The Docker Compose service runs this same command, plus --offline and the adapter's --lora.

From commit 7e1955d on, every A1 and A2 run, and the tuning scripts, stop unless llama-server reports exactly one slot (`total_slots` in `/props`). The locked A1 and A2 test runs predate the guard and ran on the earlier 4-slot server; the A3 and A4 test runs ran on the one-slot, cache-off server. For dev runs only, `HS_ALLOW_MULTISLOT=1` or `--allow-multislot` continues with a warning and tags the MLflow run `multislot_override=true`; a test split refuses it. The `config_hash` does not capture the `cache_prompt`/`-np 1` fix, so runs before and after it share a hash; the `git_commit` and `server_slots` tags mark a reproducible run (every llama-server run is also tagged `cache_prompt`).

## Tracing

Per-step latency, prompts and token counts for pipeline runs, in a local [Arize Phoenix](https://github.com/Arize-ai/phoenix) viewer. Tracing is off by default and is for dev, smoke and demo runs only: evaluation numbers are measured with it off, and a test split refuses it. It needs no account and sends nothing outside this machine.

1. Install the group (name every group you use, or `uv sync` removes the others):

   ```bash
   UV_PROJECT_ENVIRONMENT="$HOME/.venvs/hs-rag-classifier" uv sync --group llm --group ml --group tracing
   ```

2. Start the viewer at <http://localhost:6006>. Traces are stored in `~/data/hs-rag-classifier/phoenix`, and Phoenix's anonymous usage telemetry is off:

   ```bash
   bash scripts/start_phoenix.sh
   ```

3. Run with `--trace`, or set `HS_TRACING=1` (`PHOENIX_COLLECTOR_ENDPOINT` overrides the default `http://localhost:6006`):

   ```bash
   python -m hs_rag_classifier.pipeline.a2_tune --trace trace-smoke --n 20    # no MLflow run
   python -m hs_rag_classifier.eval run --arm a2 --dataset rulings --limit 20 --trace
   python -m hs_rag_classifier.pipeline.tune --trace run --profile off --evidence on
   ```

   `eval run` and `tune run` log to MLflow as usual, tagged `tracing=on`; never report those as results.

Each item is one trace in the `hs-rag-classifier` project, with a span per graph node (`profile` appears only when that toggle is on) and an LLM span with token counts under each model call. A2's scoring calls are `stage1_score_call` and `stage2_score_call <hs6>`; the rationale and A1's decisions are `ChatOpenAI`.
