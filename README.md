# hs-rag-classifier

Suggests UK commodity codes (8 digits and up) from free-text product descriptions, with a calibrated confidence and a rationale that cites the tariff text. Runs entirely on a local GPU.

> **Status:** Phase 0, set-up. No classifier yet.

> **Advisory only.** Suggestions are not a legal ruling. The legal route for a binding classification is an [Advance Tariff Ruling](https://www.gov.uk/guidance/apply-for-an-advance-tariff-ruling) from HMRC.

## Data attribution

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Tariff data comes from the [GOV.UK Trade Tariff API](https://api.trade-tariff.service.gov.uk/). Datasets are downloaded from their source at run time and are never re-hosted in this repository.

## Licence

Code: [MIT](LICENSE).
