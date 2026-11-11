# Data sources

### Data sources

All downloaded from source at run time; none are stored in this repository.

- **HMRC Advance Tariff Rulings** (OGL v3.0): train, validation and test sets.
- **UK tariff nomenclature**: Department for Business and Trade dataset uk-tariff-2021-01-01, version v4.0.1633 (OGL v3.0), plus section and chapter notes from the Trade Tariff API.
- **Trade Tariff search references** (A to Z of classified goods; OGL v3.0): a test set only, never indexed or trained on.
- **CROSS rulings** (US Customs and Border Protection rulings with HTS codes), as published in [flexifyai/cross_rulings_hts_dataset_for_tariffs](https://huggingface.co/datasets/flexifyai/cross_rulings_hts_dataset_for_tariffs) with the ATLAS paper ([arXiv 2509.18400](https://arxiv.org/abs/2509.18400)): planned for 6-digit training only, never validation or test. Not used so far: the original repository was not available on 28 Oct 2026, so the B2 encoder is trained on UK rulings only.
- **Models**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (embeddings) and [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (B2 encoder), each pinned to a revision in `configs/`.
