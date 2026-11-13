# Data sources

All downloaded from source at run time; none are stored in this repository.

- **HMRC Advance Tariff Rulings** (OGL v3.0): train, validation and test sets.
- **UK tariff nomenclature**: Department for Business and Trade dataset uk-tariff-2021-01-01, version v4.0.1633 (OGL v3.0), plus section and chapter notes from the Trade Tariff API.
- **Trade Tariff search references** (A to Z of classified goods; OGL v3.0): a test set only, never indexed or trained on.
- **HSCodeComp** (Apache-2.0; ATH-MaaS/HSCodeComp, formerly AIDC-AI/HSCodeComp, at commit ce9119795acef4ca537b2175e10a3feb7a0ecae9): 632 e-commerce listings with US HTS codes. A test set only, never indexed or trained on, and scored at 6 digits (the US digits 7 to 10 are never stored).
- **CROSS rulings** (US Customs and Border Protection rulings with HTS codes), as published in [flexifyai/cross_rulings_hts_dataset_for_tariffs](https://huggingface.co/datasets/flexifyai/cross_rulings_hts_dataset_for_tariffs) with the ATLAS paper ([arXiv 2509.18400](https://arxiv.org/abs/2509.18400)): planned for 6-digit training only, never validation or test. Not used so far: the original repository was not available on 28 Oct 2026, so the B2 encoder is trained on UK rulings only.
- **Models**: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (decider and QLoRA base, served as the unsloth/Qwen3.5-9B-GGUF UD-Q4_K_XL quantisation; the fine-tuned adapter is not distributed), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (embeddings) and [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (B2 encoder), each pinned to a revision in `configs/`.
