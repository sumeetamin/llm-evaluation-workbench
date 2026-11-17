# Changelog

## 0.5.0

- Added a reproducible RAGTruth lexical-support baseline with a source-group-held-out threshold and untouched official test evaluation.
- Added pinned data retrieval, source fingerprints, split checks, aggregate and per-task metrics, and an attribution/method note.
- Clarified that a public lexical baseline does not calibrate the LLM semantic judge.


## 0.4.0

- Added OpenAI Responses, OpenAI Chat Completions and Anthropic Messages adapters under one normalized output contract.
- Added JSONL saved-trace import that extracts answer text and safe run metadata while discarding arbitrary trace fields.
- Added optional semantic groundedness judge output, kept separate from deterministic pass/fail grading.
- Added a human-label calibration command that refuses the test split, conflicting labels, and undersized class samples.
- Documented provider payloads, trace envelopes, calibration limits and the existing BANKING77 result.

Semantic calibration remains data-dependent: no human labels are bundled, and independent holdout validation is still required.

