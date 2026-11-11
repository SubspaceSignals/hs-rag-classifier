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

## Method and evaluation discipline

### Data and splits

- **HMRC rulings.** All 6,131 current rulings were collected on 20 Oct 2026. They were de-duplicated and split once, before any model work, into 3,773 training, 750 validation and 1,500 test rulings (seed 20261020, stratified by chapter as far as near-duplicate clusters allow). Near-duplicate descriptions are kept in the same split. Validation and test hold only rulings whose code is a current declarable line. The model sees the description only; HMRC's written justification is never input.
- **A to Z of classified goods.** 3,327 short terms from the Trade Tariff search references, used as a test set only and never indexed or trained on. Terms that appear under two different codes are excluded.
- **HSCodeComp.** 632 expert-coded e-commerce listings (Apache-2.0, pinned revision), scored at 6 digits because its labels are US codes. It is an outside check on messier input than rulings.
- **Overlap checks.** No test ruling has a near-duplicate in the training rulings (near-duplicate clusters, TF-IDF cosine ≥ 0.85, never cross splits). A to Z terms and HSCodeComp listings were checked against the training rulings for exact matches and for matches on the same opening text and code; HSCodeComp titles were also checked against A to Z terms for exact matches. None were found.
- The test files' hashes are pinned (in `splits/` for rulings and A to Z, and in `configs/hscodecomp_inputs.json` for HSCodeComp) and checked by tests.

### Discipline

- **One run per approach per test set.** That gives 28 test runs, each recorded in a run log and tied in MLflow to its commit and config hash.
- **All tuning on validation.** Retrieval settings, the number of options shown to the model, the confidence temperatures, the review rule and the out-of-distribution threshold were all set on validation. Each decision rule was fitted on validation data only and frozen in `configs/` before it was applied to any test set.
- **Intervals.** 95% intervals come from 1,000 bootstrap resamples (seed 20261021), resampling near-duplicate clusters on rulings. Differences between approaches use a paired bootstrap over the same items.
- **Reproducible LLM runs.** The option-scoring and fine-tuned approaches ran on a server that handles one request at a time with prompt caching off. Under that setting, option scoring gave bit-identical results over all 750 validation items, run twice. The scripts now refuse a test run on a server with more than one slot or with tracing on; prompt caching is fixed off in the client. The first two LLM approaches were tested before this fix.
- **How it was built.** The code was written with Claude Code (Anthropic's coding assistant), directed and reviewed by the author. No text written by an AI assistant is in any test set.

### What this evaluation cannot rule out

Published rulings and HSCodeComp may be in the base model's pre-training data, which would flatter every LLM approach. HSCodeComp has been public since October 2025, so it is a weaker check on this than a private test set would be. The fine-tune's effect (fine-tuned v the same pipeline without the adapter) is largely unaffected, because both use the same base model.

## Responsible AI

The service suggests; a person decides. Every response says it is not a legal ruling and points to HMRC's Advance Tariff Ruling as the binding route.

### What a response contains

- Up to three 8-digit codes, each with a confidence, so the reviewer sees the alternatives, not a single answer.
- A short rationale quoting the chosen code's tariff text. A separate call writes it after the choice and cannot change it. On the test sets, 88.9% (rulings) and 95.1% (A to Z) of rationales pass the citation check.
- The nearest HMRC ruling from the training set, with its code and whether it agrees with the suggestion, as precedent the reviewer can check.
- Codes only from the retrieved list: the model chooses among real codes, so it cannot make one up, and each chosen code is also checked against the tariff.

### When it asks for review

A response is marked for review when either of two rules fires. Both were fitted on validation only and frozen before being applied to any test set.

- **The operating-point rule** aims for accepted codes to be right 80% of the time. On the rulings test set, accepted codes were right 80.5% of the time (77.5% to 82.9%) with 43.7% sent to review; the lower end of that interval misses 80%. On A to Z terms: 87.4% at 52.2% review. On HSCodeComp listings the rule fails: 66.2% at 41.9% review.
- **The out-of-distribution flag** fires when an input's nearest training ruling is less similar than it is for 95% of validation rulings (cosine similarity below 0.638). It fires on 4.3% of rulings test items, 88.9% of A to Z terms and 65.8% to 92.2% of HSCodeComp listings. On rulings, A to Z and HSCodeComp listings with their attributes, flagged items are 11.5 to 21.7 points less accurate than unflagged ones; on title-only HSCodeComp input the gap is 7.6 points and its interval includes zero. On A to Z, though, the flag adds about 40 points of review without raising the accuracy of accepted codes (87.1% v 87.4%). The response shows the similarity, the threshold and the nearest ruling, so the reason is visible.

Together they send 44.9% of rulings test items to review, and 79% to 96% of short or listing-style inputs. The 80% figure therefore describes ruling-style descriptions only. On short text the service is a suggestion aid, and nearly everything goes to a person.

### What the confidence means

Confidence comes from the model's token probabilities over the listed options, temperature-scaled on validation. It is not a guarantee: expected calibration error is 0.080 on rulings and 0.107 on A to Z (at 8 digits), and 0.166 on HSCodeComp (at 6 digits); on rulings and A to Z that is higher than for the same pipeline without the adapter (0.049 and 0.064). It orders the review queue; it does not replace review.

### Data and security

- Public data only: HMRC rulings and the tariff under the Open Government Licence, and HSCodeComp under Apache-2.0. No company, supplier or personal data, and no AI-written text in any test set.
- Runs on one workstation. The API listens on 127.0.0.1 only, the model server has no outbound network route, and the container makes no Trade Tariff or Hugging Face calls and holds no credentials.
- Descriptions are treated as untrusted data. They must be 1 to 10,000 characters, are checked for injection-like text and are never logged, and the pipeline has no tool that can change anything.

### Not yet measured

- Whether it saves reviewers time. No user trial has run, so faster review is an aim, not a result.
- How often a rationale argues against its own code. At least one case was seen in a development run, but the rate has not been counted.

## Engineering

### Runs with one command

On the author's workstation, `scripts/docker_up.sh` checks the mounts and starts two containers with Docker Compose. One runs llama.cpp (build b11510, compiled from source in the image to match the evaluation runs), serving Qwen3.5-9B with the fine-tuned adapter. The other runs a FastAPI service with the frozen A4 pipeline. The model weights, adapter, indexes and data are not distributed. Anyone can read the repository and run its tests, but the stack only starts where those files exist.

- **Same answers in Docker.** On 13 validation items, three of them flagged out of distribution, the containerised service returned output identical to the reference evaluation run.
- **Pinned at start-up.** The service refuses to start unless the pipeline's config hash and the hashes of both review rules match their frozen values.
- **One request at a time.** The model server runs one slot with prompt caching off, the setting under which repeat runs are bit-identical. In the live smoke test, a request took a median 2.8 s.

### Locked down by default

- The API is published on 127.0.0.1 only, and a test fails if any published port is not. The model server sits on an internal network with no published port and no outbound route.
- Both containers drop Linux capabilities and run with no-new-privileges, and the API runs as a non-root user.
- A preflight refuses to start the stack if any link under a mounted folder points outside the mounted set.
- Inside the container the service makes no Trade Tariff or Hugging Face calls. Tariff checks use a cached copy, falling back to the pinned nomenclature.

### CI and tracking

- GitHub Actions runs on every push and pull request. It checks the lock file, lints and checks formatting, runs 440 tests, and scans the full history for secrets with a pinned, checksum-verified gitleaks. A weekly workflow runs the torch and fine-tuning tests. Actions are pinned by commit SHA.
- CI has no evaluation gate and builds no image, because the indexes and data it would need are not distributed. Evaluation runs locally.
- Every evaluation run is logged to a local MLflow store with its git commit and config hash, and every test-set run passes through a guard that records it.
- Optional tracing (OpenTelemetry to a local Arize Phoenix) shows per-step latency and token counts. It is off by default and refused on test sets.

### Design choices

- Vectors stay in memory with exact search, since a few thousand vectors need no vector database.
- Everything ran on one 16 GB RTX 5070 Ti, including the QLoRA fine-tune (745 steps, about 10.4 hours). Serving peaks at about 9.2 GiB of GPU memory.
- The front end is the author's existing commodity code lookup app. A suggest panel sits behind a feature flag on a branch and calls the local API through a development proxy.

## Data attribution

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Tariff data comes from the [GOV.UK Trade Tariff API](https://api.trade-tariff.service.gov.uk/). Datasets are downloaded from their source at run time and are never re-hosted in this repository.

## Reference

- [Running locally](docs/running-locally.md): the decider server command, reproducibility flags and optional tracing.
- [Data sources](docs/data-sources.md): every dataset and model, its licence and how it is used.
- Configuration and frozen decision rules: [configs/](configs/).

## Licence

Code: [MIT](LICENSE).
