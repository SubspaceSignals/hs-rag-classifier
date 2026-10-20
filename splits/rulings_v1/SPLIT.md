# HMRC rulings split v1 (frozen)

Built by `uv run python -m hs_rag_classifier.data.rulings build` from the HMRC Advance Tariff Rulings collected on 20 Oct 2026 (Open Government Licence v3.0). These files hold references only; no ruling text is stored in the repo.

- Seed: 20261020
- Nomenclature: DBT uk-tariff-2021-01-01 v4.0.1633 (current lines only)
- Near-duplicate threshold: TF-IDF char_wb 3-5 cosine >= 0.85
- Historical code check: Trade Tariff API `/commodities/{code}?as_of=<start>`
- test_refs.txt SHA-256: `426e6288fda6d8f3bce15198f776f84ddffce16f1b9dc5e6893c7ee5c6371695`

## Counts

| Split | De-duplicated rulings | Member references | Chapters | 8-digit labels |
| --- | --- | --- | --- | --- |
| train | 3,773 | 3,836 | 89 | 75 |
| val | 750 | 765 | 71 | 0 |
| test | 1,500 | 1,525 | 85 | 0 |

Collected 6,131 rulings; 6,028 after exact de-duplication; 5 groups (5 references) excluded because their code is not current, has no clean mapping and its cn8 is not current either.

## Rules

- Exact duplicates (same normalised description and code) form one row; the lowest reference represents them, and every member reference is listed in the split files.
- A near-duplicate cluster never spans two splits.
- Validation and test hold only rulings whose code is a current declarable line, whose description isn't also ruled under another code (conflict) and whose description doesn't contain a 4+ digit prefix of its own code.
- Retired codes stay in train only with a clean 1:1 mapping to a current code; codes valid at 8 digits only stay in train; codes never valid at the start date are excluded.
- Test and validation are stratified by chapter as far as clusters allow.
- label_depth is 10, or 8 for train-only rulings whose 10-digit code is not a current declarable line but whose cn8 is current; those are scored to 8 digits only. Rulings with no current cn8 are excluded.
- Model input is the description only. Description + keywords is a secondary ablation on this same test set; keywords carry tariff wording in about 64% of rulings.
- Justifications are stored apart from the split files and are never model input.

`tests/test_frozen_split.py` fails if test_refs.txt no longer matches the hash above.
