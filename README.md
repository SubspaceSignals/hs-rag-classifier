# hs-rag-classifier

[![CI](https://github.com/SubspaceSignals/hs-rag-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/SubspaceSignals/hs-rag-classifier/actions/workflows/ci.yml)

Suggests UK commodity codes (8 digits and up) from free-text product descriptions, with a confidence score and a rationale that cites the tariff text. Runs entirely on a local GPU.

> **Status (11 Nov 2026):** working end to end on one local GPU (RTX 5070 Ti, 16 GB) at zero cost. Evaluation is complete on frozen test sets; the full write-up is in progress. Model weights, fine-tuning data and built indexes are not distributed, so the repo is readable rather than turnkey.

> **Advisory only.** Suggestions are not a legal ruling. The legal route for a binding classification is an [Advance Tariff Ruling](https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling) from HMRC.

## Results so far

Every figure is from a single run per test set, with the decision rules fixed on validation before any test result (28 test-set runs in total). 95% bootstrap intervals in brackets.

| Approach | HMRC rulings test, 8-digit top-1 | A to Z test, 8-digit top-1 |
| --- | --- | --- |
| Keyword search (BM25) | 0.098 (0.083 to 0.113) | 0.373 (0.358 to 0.390) |
| Nearest training ruling (embedding kNN) | 0.441 (0.415 to 0.465) | 0.107 (0.096 to 0.117) |
| Retrieval + Qwen3.5-9B, generated answer | 0.489 (0.465 to 0.513) | 0.570 (0.554 to 0.588) |
| Retrieval + Qwen3.5-9B, option scoring averaged over both option orders | 0.538 (0.513 to 0.561) | 0.604 (0.588 to 0.620) |
| The same option-scoring pipeline with a QLoRA fine-tune (recommended) | 0.653 (0.629 to 0.678) | 0.643 (0.628 to 0.659) |

The fine-tune's own effect is the last row minus the one above it, measured on the same server: +11.5 points (+9.3 to +13.7) on rulings and +3.9 (+2.8 to +5.1) on A to Z. The generated-answer row was measured on an earlier server configuration, so its gap to the rows below is indicative.

- **Review, not replacement.** Under the operating-point rule fixed on validation, auto-accepted codes on the rulings test are right 80.5% of the time (77.5% to 82.9%), with 43.7% of items sent to a person for review.
- **That figure covers ruling-style descriptions only.** The service also flags inputs unlike its training rulings and always sends them to review. The flag fires on 4.3% of rulings test items, 88.9% of the short A to Z queries and 65.8% of HSCodeComp e-commerce listings (92.2% on titles alone), where the rule on its own falls short (66.2% at 41.9% review, scored at 6 digits). On short text the service is a suggestion aid: it returns its top three codes for a person to confirm.
- **Runs with one command.** FastAPI, llama.cpp and the retrieval index in Docker Compose, bound to localhost; CI runs tests, lint and a full-history secret scan on every push.

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

Keep `-np 1` for evaluation: A2 output depended on llama-server slot and prompt-cache state, and `-np 1` plus `cache_prompt=false` (set in `pipeline/llm.py`) gives bit-identical runs. It serves one request at a time. The Docker Compose service runs this same command, plus --offline and the adapter's --lora.

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
