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

## Reference

- [Running locally](docs/running-locally.md): the decider server command, reproducibility flags and optional tracing.
- [Data sources](docs/data-sources.md): every dataset and model, its licence and how it is used.
- Configuration and frozen decision rules: [configs/](configs/).

## Licence

Code: [MIT](LICENSE).
