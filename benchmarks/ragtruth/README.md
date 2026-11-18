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

## Semantic judge calibration sample

The existing runner evaluates a local lexical baseline. To prepare a separate private sample for the workbench's semantic groundedness judge, run:

```powershell
python -m pip install -e ".[benchmark]"
python -m benchmarks.ragtruth.prepare_semantic_calibration
```

This creates 100 cases in each split under the ignored `benchmarks/ragtruth/calibration_private/` directory: 50 pass and 50 fail examples, with one response per source group. The `validation` cases are sampled from a deterministic 20% source-group holdout of the official training set; the `test` cases come from the untouched official test split, whose source groups are disjoint from train. The sample is balanced by label, so it is useful for threshold diagnostics but is not prevalence-representative.

RAGTruth supplies human-annotated hallucination spans. The adapter maps a response to `fail` when any annotated span is not marked `implicit_true`, and to `pass` otherwise. This is a simplified, corpus-derived response-level label, not custom adjudication of the workbench's support-QA cases. Do not present the result as production validation or a safety guarantee.

Preparation makes no model-provider requests. To score the sample, first review provider terms and cost, then provide your own key and selected model. This step sends the sampled source and answer text to that provider. Use the same provider/model for both splits:

```powershell
python -m evalbench semantic-grade --dataset benchmarks/ragtruth/calibration_private/dataset.jsonl --outputs benchmarks/ragtruth/calibration_private/outputs.jsonl --results benchmarks/ragtruth/calibration_private/validation-scores.jsonl --provider openai-responses --model YOUR_MODEL --split validation
python -m evalbench semantic-grade --dataset benchmarks/ragtruth/calibration_private/dataset.jsonl --outputs benchmarks/ragtruth/calibration_private/outputs.jsonl --results benchmarks/ragtruth/calibration_private/test-scores.jsonl --provider openai-responses --model YOUR_MODEL --split test
python -m evalbench calibrate-groundedness --dataset benchmarks/ragtruth/calibration_private/dataset.jsonl --scores benchmarks/ragtruth/calibration_private/validation-scores.jsonl --annotations benchmarks/ragtruth/calibration_private/annotations.jsonl --split validation --report benchmarks/ragtruth/calibration_private/calibration.json
python -m evalbench validate-groundedness --dataset benchmarks/ragtruth/calibration_private/dataset.jsonl --scores benchmarks/ragtruth/calibration_private/test-scores.jsonl --annotations benchmarks/ragtruth/calibration_private/annotations.jsonl --calibration benchmarks/ragtruth/calibration_private/calibration.json --split test --report benchmarks/ragtruth/calibration_private/holdout.json
```

The generated dataset, answers, annotations and score files remain local and ignored by Git. The manifest contains aggregate counts, source fingerprints, split details and the label mapping, but no source text or answers.

## Local embedding-similarity baseline

For a no-provider comparison, the optional baseline uses the pinned `sentence-transformers/all-MiniLM-L6-v2` encoder to compare context and answer embeddings. It selects a threshold on the 100-case source-disjoint validation sample and reports once on the 100-case test sample; both are balanced 50/50 pass/fail and are not representative of natural traffic. It also reports a TF-IDF baseline on the same sample for a direct comparison.

Install the optional dependencies and run after preparing the private sample:

```powershell
python -m pip install -e ".[embedding-benchmark]"
python benchmarks/ragtruth/run_embedding_baseline.py --local-files-only
```

With `--local-files-only`, the pinned model revision must already be cached. Omit the flag to download it from Hugging Face on first use. The machine-readable aggregate report is [`results/embedding_report.json`](results/embedding_report.json); row-level scores stay local under the ignored `calibration_private/` directory.

On this 100-case test sample, the embedding baseline scored 58% accuracy and balanced accuracy, with 0.593 ROC-AUC and 0.32 recall for the fail label. The same-sample TF-IDF baseline scored 52% accuracy and balanced accuracy, with 0.672 ROC-AUC and 0.10 fail recall at its validation-selected threshold. This small-sample comparison is inconclusive; the embedding score did not show strong discrimination.

This is an embedding-similarity baseline, not the workbench's LLM semantic judge or an entailment model. Similarity can miss negation, contradictions, unsupported details and multi-hop reasoning; each context and answer is truncated to 256 tokens. Labels are response-level mappings of RAGTruth span annotations, not product-specific reviewer adjudication. These results do not calibrate the semantic judge or validate product-specific support QA.

## Larger offline grouped TF-IDF baseline

For a source-disjoint benchmark that needs no model downloads or provider calls, prepare a larger private sample and run the sparse lexical classifier:

```powershell
python -m pip install -e ".[benchmark]"
python -m benchmarks.ragtruth.prepare_semantic_calibration --per-class 150 --output-dir benchmarks/ragtruth/calibration_private_600
python benchmarks/ragtruth/run_grouped_tfidf.py --input-dir benchmarks/ragtruth/calibration_private_600 --output benchmarks/ragtruth/results/grouped_tfidf_600.json
```

The published aggregate report uses 300 validation and 300 untouched test cases, one response per source group, with zero source overlap. The model selected its regularization and threshold from five-fold grouped out-of-fold predictions on validation only. On this balanced held-out sample it reached 65.3% balanced accuracy, 74.0% fail recall, 63.1% fail precision, and 0.711 ROC-AUC. Test balanced accuracy was 63.4% for Data2txt, 64.3% for QA, and 59.1% for Summary. On the same original 100-case sample, its 57% balanced accuracy did not outperform the 58% embedding baseline; the larger sample result is a separate run, not a like-for-like improvement claim.

This is an experimental lexical classifier, not an LLM judge or entailment model. Labels are coarse response-level mappings of human-annotated RAGTruth spans, the sample is balanced rather than prevalence-representative, and the benchmark does not establish product-specific or production quality. Generated packets contain source text and answers; the new private-sample directory pattern is ignored by Git.
