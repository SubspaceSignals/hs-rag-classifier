# hs-rag-classifier

[![CI](https://github.com/SubspaceSignals/hs-rag-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/SubspaceSignals/hs-rag-classifier/actions/workflows/ci.yml)

Suggests UK commodity codes (8 digits and up) from free-text product descriptions, with a confidence score and a rationale that cites the tariff text. Runs entirely on a local GPU.

> **Status (11 Nov 2026):** working end to end on one local GPU (RTX 5070 Ti, 16 GB) at zero cost. Evaluation is complete on frozen test sets; the full write-up is in progress. Model weights, fine-tuning data and built indexes are not distributed, so the repo is readable rather than turnkey.

> **Advisory only.** Suggestions are not a legal ruling. The legal route for a binding classification is an [Advance Tariff Ruling](https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling) from HMRC.

## The problem

Every product a business exports, imports or buys in has to carry a commodity code from the UK tariff, at 8 digits or more. The code decides duty, controls and trade statistics, so a wrong one means paying the wrong duty, delays at the border or a compliance problem. People assign codes by reading the tariff against a product description, which is slow and inconsistent when descriptions are vague, and it holds up the shipping paperwork that depends on the code.

## The aim

An open-source, resource-light aid for commodity code selection that brings in what HMRC has already ruled on similar goods, shows the tariff text behind each suggestion, and gives each code a confidence score, so the people assigning codes can deal with routine items quickly and spend their time on the hard ones.

Design choices that follow from that aim:

- **Resource-light.** Everything runs on one 16 GB consumer GPU, with open-weight models and no cloud service, so it can run on an ordinary workstation.
- **Data stays in-house.** Product descriptions never leave the machine.
- **Built on public data.** It uses the UK tariff and published HMRC rulings, both under the Open Government Licence.
- **Advisory only.** A person makes the final decision.

## Options compared

Eight approaches were built and tested on the same frozen test sets, from the cheapest to the most involved.

| Option | How it works |
| --- | --- |
| Keyword search (BM25) | Ranks tariff lines by word overlap with the description |
| Tariff embedding | Nearest tariff line by meaning, using an open embedding model |
| Nearest training ruling | Copies the code of the most similar past HMRC ruling |
| Fine-tuned encoder | A small classifier trained on past rulings; 6 digits only, as there are too few UK labels for 8 |
| Retrieval + LLM, generated answer | Retrieves candidate codes and shows Qwen3.5-9B up to 42 six-digit subheadings; it picks one, then picks the 8-digit code from that subheading's codes |
| Retrieval + LLM, option scoring | The same candidates shown as lettered options; the confidence comes from the model's token probabilities |
| Option scoring averaged over both option orders | Scores the options in both orders to remove position bias, and queues items for review |
| The same pipeline with a QLoRA fine-tune | A low-rank adapter trained on 11,916 examples built from the 3,773 training rulings, on the same GPU |

## Recommendation

Use the fine-tuned option-scoring pipeline as a review aid, not as an autopilot.

- **For descriptions written like an HMRC ruling** (a sentence or more on what the product is, what it is made of and what it does), auto-accept codes above the threshold fixed on validation and send the rest to a person. On the rulings test, the auto-accepted codes are right 80.5% of the time with 43.7% of items reviewed. The 80% target holds at the point estimate; the lower end of the interval (77.5%) does not.
- **For short or listing-style text**, do not auto-accept. The service flags these inputs and returns its top three codes, each with the tariff text it relied on, for a person to confirm.
- **Why not something simpler.** Copying the nearest ruling does well on rulings (0.441) but fails on short queries (0.107), because it only works when a near-identical ruling exists. Keyword search is cheapest but rarely right at 8 digits. The fine-tune adds 11.5 points on rulings and 3.9 on short queries over the same pipeline without it.
- **What it needs.** One 16 GB GPU, for both the one-off fine-tune and serving. The service handles about 20 items a minute, one request at a time.
- **Before relying on it,** trial it with the people who assign codes. Measure how often the right code is in the top three and how much time is saved, not only auto-accept accuracy. The tariff changes over time, so pin the nomenclature version and re-check the thresholds when it changes.

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
