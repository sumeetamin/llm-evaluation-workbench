# RAGTruth lexical groundedness baseline

This benchmark runs a local, no-API-key lexical support baseline on the official RAGTruth response and source-information files. RAGTruth contains human-annotated hallucination spans. The runner maps each usable response to a binary label, scores context/answer TF-IDF cosine similarity, selects a threshold on a source-group-held-out slice of the official training split, and reports once on the untouched official test split.

## Run

From the repository root:

```powershell
python -m pip install -e ".[benchmark]"
python benchmarks/ragtruth/run.py
```

The runner downloads the two official JSONL files from the pinned upstream revision on first use. Raw data is stored in the ignored `benchmarks/ragtruth/data/` directory and is not included in this repository. It writes aggregate metrics to `benchmarks/ragtruth/results/report.json` and row IDs, scores, and mapped labels to `test_scores.jsonl`.

## Data and labels

- Source: [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth), revision `c103204b9ce28d6bbad859304bf30de72b8ed8fe`.
- Paper: Wu et al., [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://aclanthology.org/2024.acl-long.585/), ACL 2024.
- The source has official `train` and `test` splits. Responses with `quality` other than `good` are excluded.
- A response is positive if it has an annotated span whose `implicit_true` flag is false. Correct implicit truths are not counted as hallucinations. Rows with no such span are negative.
- Source IDs are checked for overlap between partitions. Threshold selection uses a deterministic 20% group split of official training data; the official test is used only for final evaluation.

## What this demonstrates

It gives the workbench a reproducible, real-data grounding-related benchmark with human annotations and a transparent baseline. It does **not** call or calibrate the repository's LLM semantic judge. A lexical-overlap score is not entailment: it can miss paraphrases, negation, or factual errors with high word overlap. RAGTruth's response-level mapping also simplifies span-level labels. Results should be read as a baseline comparison on this corpus, not as a production quality or safety claim.

The upstream repository provides an MIT license; retain attribution and check the authors' current data-use terms before redistributing the corpus or derived data. Raw source files are not committed here.

