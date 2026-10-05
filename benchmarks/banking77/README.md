# BANKING77 intent-classification benchmark

This benchmark trains a local TF-IDF + linear SVM classifier on BANKING77's official train split and evaluates once on the official test split. It extends the workbench with a real, labeled support-classification task; it is separate from the synthetic answer-quality demo in `data/support_qa.jsonl`.

## Data source and attribution

- Dataset: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)
- Fixed source revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0). Raw files download to `benchmarks/banking77/data/` and are git-ignored.
- Dataset paper: Casanueva et al. (2020), [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807). Cite the paper when reusing the dataset.
- Published dataset size: 10,003 training queries and 3,080 test queries across 77 intents.

The code and benchmark report in this repository are MIT-licensed. BANKING77 keeps its separate CC BY 4.0 license and attribution.

## Reproduce

From the project root, using Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[benchmark]"
.\.venv\Scripts\python.exe benchmarks/banking77/run.py
```

The script downloads the two CSV files from the pinned PolyAI revision if they are missing, trains only on `train.csv`, and writes aggregate and per-intent results to `benchmarks/banking77/results/`. Raw queries are not copied to the repository or sent to a third-party model API.

Optional paths:

```powershell
.\.venv\Scripts\python.exe benchmarks/banking77/run.py --data-dir benchmarks/banking77/data --output-dir benchmarks/banking77/results
```

## Model and protocol

- Baseline: `DummyClassifier(strategy="most_frequent")`.
- Candidate: word unigram/bigram and character 3-5-gram TF-IDF features, combined with `LinearSVC(C=1.0)`.
- Split: official train/test split. Vectorizers and classifier fit on training data only. The test split is not used for tuning.
- Metrics: accuracy, macro-F1, weighted-F1, top-3 accuracy, per-intent precision/recall/F1/support, and the ten most common error pairs.
- Reproducibility: source revision, file SHA-256 hashes, Python and scikit-learn versions, model settings, and elapsed times are recorded in JSON.

This is a single untuned baseline experiment, not a tuned leaderboard result. BANKING77 is an English, closed-set benchmark of short banking queries; it does not represent live traffic from a bank, answer quality, financial-advice safety, retrieval grounding, or out-of-domain detection. Inspect the confusion pairs and limitations before drawing conclusions.

