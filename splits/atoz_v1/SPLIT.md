# A to Z test set v1 (frozen)

Built by `uv run python -m hs_rag_classifier.data.atoz build` from the GOV.UK Trade Tariff API `/search_references` (Open Government Licence v3.0), resolved against DBT uk-tariff-2021-01-01 v4.0.1633. This file and test_refs.txt hold search reference ids only; no terms are stored in the repo.

- test_refs.txt SHA-256: `d70a60a6b50e412cec77292f4af8ca6efeeec4ecb08db8e6b49806b953934838`
- References collected: 8,941
- Test rows: 3,327 (3,333 reference ids; 2,774 scored to 10 digits, 553 to 8)
- By label_source: direct 2,903, single_cn8_below 424
- Conflicting terms excluded: 10 (20 references)

## Rules

- Test set only: search references are never indexed for retrieval.
- Input is the A to Z term (title). The label is the 10-digit code when the term points at a declarable line, otherwise the cn8 when exactly one cn8 sits beneath the line it points at.
- `label_source` in atoz_test8: `direct` when the term points at the 8- or 10-digit line itself; `single_cn8_below` when it points at a heading or 6-digit subheading and the label is the only cn8 beneath it.
- Conflicting terms are excluded: a term is dropped when any two of its references disagree at the shallower of their scored depths (e.g. "shower gel" under 3304 and 3401). Scores on this set therefore exclude genuinely ambiguous terms.
- Each remaining term counts once, scored at the shallowest depth among its direct references; a `single_cn8_below` reference may confirm a direct label but never coarsens it. Every member id is listed in test_refs.txt.
- Terms pointing at lines that are no longer current, or with several cn8s beneath, are left out.
- A reference whose sid names a line with another code or suffix (some headings carry the sid of the TARIC sub-chapter title above them) is resolved by its code and suffix; DBT parents those headings to the title too, and the nomenclature build repairs them to their chapter.

`tests/test_frozen_split.py` fails if test_refs.txt no longer matches the hash above.
