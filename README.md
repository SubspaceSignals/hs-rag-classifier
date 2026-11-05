# hs-rag-classifier

Suggests UK commodity codes (8 digits and up) from free-text product descriptions, with a calibrated confidence and a rationale that cites the tariff text. Runs entirely on a local GPU.

> **Status:** Phase 0, set-up. No classifier yet.

> **Advisory only.** Suggestions are not a legal ruling. The legal route for a binding classification is an [Advance Tariff Ruling](https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling) from HMRC.

## Data attribution

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Tariff data comes from the [GOV.UK Trade Tariff API](https://api.trade-tariff.service.gov.uk/). Datasets are downloaded from their source at run time and are never re-hosted in this repository.

## Data sources

All downloaded from source at run time; none are stored in this repository.

- **HMRC Advance Tariff Rulings** (OGL v3.0): train, validation and test sets.
- **UK tariff nomenclature**: Department for Business and Trade dataset uk-tariff-2021-01-01, version v4.0.1633 (OGL v3.0), plus section and chapter notes from the Trade Tariff API.
- **Trade Tariff search references** (A to Z of classified goods; OGL v3.0): a test set only, never indexed or trained on.
- **CROSS rulings** (US Customs and Border Protection rulings with HTS codes), as published in [flexifyai/cross_rulings_hts_dataset_for_tariffs](https://huggingface.co/datasets/flexifyai/cross_rulings_hts_dataset_for_tariffs) with the ATLAS paper ([arXiv 2509.18400](https://arxiv.org/abs/2509.18400)): planned for 6-digit training only, never validation or test. Not used so far: the original repository was not available on 28 Oct 2026, so the B2 encoder is trained on UK rulings only.
- **Models**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (embeddings) and [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (B2 encoder), each pinned to a revision in `configs/`.

## Decider server

```bash
~/llama.cpp/build/bin/llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL -ngl 99 -c 16384 --port 8080 --jinja --cache-ram 0 -np 1
```

Keep `-np 1` for evaluation: A2 output depended on llama-server slot and prompt-cache state, and `-np 1` plus `cache_prompt=false` (set in `pipeline/llm.py`) gives bit-identical runs. It serves one request at a time, so the Phase 4 service may use different flags and will not be bit-reproducible.

Every A1 and A2 run, and the tuning scripts, stop unless llama-server reports exactly one slot (`total_slots` in `/props`). For dev runs only, `HS_ALLOW_MULTISLOT=1` or `--allow-multislot` continues with a warning and tags the MLflow run `multislot_override=true`; a test split refuses it. The `config_hash` does not capture the `cache_prompt`/`-np 1` fix, so runs before and after it share a hash; the `git_commit` and `server_slots` tags mark a reproducible run (every llama-server run is also tagged `cache_prompt`).

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

## Licence

Code: [MIT](LICENSE).
