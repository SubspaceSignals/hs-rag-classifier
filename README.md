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

## Results

Every figure is from a single run per test set (28 test-set runs in total). Each decision rule was fitted on validation data only and frozen before it was applied to any test set. 95% bootstrap intervals are in brackets.

### Accuracy

8-digit top-1 is how often the first suggestion is exactly right; top-3 is how often the right code is among the three shown.

| Approach | Rulings top-1 | Rulings top-3 | A to Z top-1 | A to Z top-3 | Items per minute | p95 seconds per item |
| --- | --- | --- | --- | --- | --- | --- |
| Keyword search (BM25) | 0.098 (0.083 to 0.113) | 0.166 | 0.373 (0.358 to 0.390) | 0.467 | 112,394 (batched) | under 0.01 (batched) |
| Tariff embedding | 0.089 (0.074 to 0.104) | 0.177 | 0.395 (0.379 to 0.412) | 0.501 | 11,591 (batched) | under 0.01 (batched) |
| Nearest training ruling | 0.441 (0.415 to 0.465) | 0.567 | 0.107 (0.096 to 0.117) | 0.140 | 12,900 (batched) | under 0.01 (batched) |
| Fine-tuned encoder (6 digits only) | n/a; 6-digit 0.421 | n/a | n/a; 6-digit 0.008 | n/a | 18,144 (batched) | under 0.01 (batched) |
| Retrieval + LLM, generated answer | 0.489 (0.465 to 0.513) | 0.639 | 0.570 (0.554 to 0.588) | 0.646 | 23.4 | 3.29 |
| Retrieval + LLM, option scoring | 0.492 (0.468 to 0.517) | 0.675 | 0.580 (0.563 to 0.597) | 0.668 | 35.5 | 2.10 |
| Option scoring averaged over both option orders | 0.538 (0.513 to 0.561) | 0.713 | 0.604 (0.588 to 0.620) | 0.690 | 24.5 | 3.00 |
| The same pipeline with a QLoRA fine-tune (recommended) | 0.653 (0.629 to 0.678) | 0.803 | 0.643 (0.628 to 0.659) | 0.727 | 21.5 | 3.42 |

- **The fine-tune's own effect** is the last row minus the one above it, on the same server: +11.5 points (+9.3 to +13.7) on rulings and +3.9 (+2.8 to +5.1) on A to Z. It was trained on ruling-style text only, which is likely why the gain on short A to Z queries is smaller.
- **Generated answer v option scoring.** Their top-1 difference (0.492 v 0.489) is smaller than the spread between repeated runs of option scoring on a validation sample (0.51 to 0.55 over 15 runs of 100 items), so it is not counted as a gain. Option scoring's gain is a usable confidence score, which is what makes the review rule below possible. Both rows ran on an earlier 4-slot server configuration, so their gap to the rows below is indicative.
- **Speed.** The four non-LLM approaches ran in batches of 64, and their per-item figures are batch averages. The LLM approaches ran one request at a time on one GPU. The three option-scoring rows include a separate rationale call (about 1 second). For the recommended pipeline, scoring alone takes 2.3 s at p95, and on A to Z it runs at 25.3 items a minute, 2.94 s at p95. None of the figures include model loading.

### Review: the rule and the service

The operating-point rule was fixed on rulings validation. It sets the review cut so that, across bootstrap resamples, the auto-accepted codes are at least 80% right at the 5th percentile. The service adds one more check: an out-of-distribution flag sends any input unlike the training rulings to review, whatever its confidence.

| Test set | Rule alone: reviewed | Rule alone: auto-accepted right | Service (rule + flag): reviewed | Service: auto-accepted right | Flag fires on |
| --- | --- | --- | --- | --- | --- |
| HMRC rulings (8 digits) | 43.7% | 80.5% (77.5% to 82.9%) | 44.9% | 80.9% (78.0% to 83.5%) | 4.3% |
| A to Z short queries (8 digits) | 52.2% | 87.4% (85.7% to 89.0%) | 92.3% | 87.1% (82.8% to 91.0%) | 88.9% |
| HSCodeComp listings (6 digits) | 41.9% | 66.2% (60.9% to 71.1%) | 79.1% | 70.5% (62.2% to 78.2%) | 65.8% |

- **On rulings** the 80% target holds at the point estimate but not at the lower end of its interval. Ordering by confidence alone, reported alongside, gives 83.0% (80.3% to 85.6%) at 42.8% review.
- **On short queries** the rule meets the target, but the flag sends nine in ten to review anyway, because they look nothing like a ruling. Flagged items are harder (11.5 points less accurate than unflagged ones on A to Z, 21.7 on rulings), but on short queries the flag adds 40 points of review without raising the accuracy of what is auto-accepted (87.1% v 87.4%).
- **On e-commerce listings** the rule falls well short: reaching 80% would have needed 78.5% review. This is the scope limit the flag exists for. On titles alone the flag fires on 92.2% of items.

### Confidence and citations

- **Calibration.** The fine-tune is more accurate but has a higher expected calibration error than the same pipeline without it: 0.080 v 0.049 on rulings (the intervals overlap) and 0.107 v 0.064 on A to Z. The review rule was fitted on the fine-tuned model's own validation confidences, so it allows for this.
- **Citations.** The rationale quotes the chosen code's tariff text in 88.9% of rulings items and 95.1% of A to Z items.

## Data attribution

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Tariff data comes from the [GOV.UK Trade Tariff API](https://api.trade-tariff.service.gov.uk/). Datasets are downloaded from their source at run time and are never re-hosted in this repository.

## Reference

- [Running locally](docs/running-locally.md): the decider server command, reproducibility flags and optional tracing.
- [Data sources](docs/data-sources.md): every dataset and model, its licence and how it is used.
- Configuration and frozen decision rules: [configs/](configs/).

## Licence

Code: [MIT](LICENSE).
