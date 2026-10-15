# LLM Evaluation Workbench

An eval-driven development tool for retrieval-grounded customer-support assistants. It runs a versioned JSONL evaluation set against saved model outputs or supported hosted model APIs, applies transparent deterministic graders, and produces a case-level HTML report. It also imports saved provider traces and compares runs to surface regressions and improvements.

## Why this exists

Offline model benchmarks do not tell a team whether its own assistant answers policy questions correctly, cites the right material, avoids unsupported promises, or handles adversarial retrieved text. This workbench makes those application-specific behaviors measurable before a prompt or model change ships.

## What it measures

Each case can specify required and forbidden phrases, whether refusal is expected, which source IDs must be cited, and which cited sources are allowed. Reports include overall pass rate, per-check rates, latency and token usage when available, and each failing case with its individual checks. Comparison reports show aggregate deltas and case-level regressions.

The graders are deterministic and inspectable. They are useful for hard requirements and regression checks; they do not understand semantic equivalence. A good workflow combines them with human review and expands the dataset with real, privacy-reviewed failures.

## Quick start

Requires Python 3.10 or newer. The core package uses only the Python standard library.

```bash
python -m evalbench demo
```

This scores synthetic candidate and baseline outputs and writes HTML/JSON reports to `reports/`. No network call or API key is used.

Evaluate a saved output file:

```bash
python -m evalbench evaluate \
  --dataset data/support_qa.jsonl \
  --outputs fixtures/candidate.jsonl \
  --report reports/candidate.html
```

Run the set against OpenAI Responses, OpenAI Chat Completions, or Anthropic Messages. Set the key in the environment; never put it in a command, config file, output fixture or GitHub repository. The provider defaults to OpenAI Responses.

```bash
# PowerShell
$env:OPENAI_API_KEY = "your-key"
python -m evalbench run --dataset data/support_qa.jsonl --model YOUR_MODEL --outputs reports/run.jsonl --report reports/run.html
```

Use `--provider anthropic-messages` and set `ANTHROPIC_API_KEY` for Anthropic, or `--provider openai-chat-completions` for OpenAI's Chat Completions schema. `--endpoint` can override the provider URL and `--api-key-env` can select a custom key variable. `run` sends each dataset question and context to the selected endpoint. Use only data you are authorized to transmit. Requests run sequentially; prompts are not copied into normalized output files.

Import saved provider response traces without resending prompts:

```powershell
python -m evalbench trace-import --dataset data/support_qa.jsonl --traces reports/traces.jsonl --outputs reports/imported.jsonl --report reports/imported.html
```

Each input trace is a JSONL object with `case_id`, `provider`, and a `response` object using that provider's response schema. `latency_ms` is optional. The importer extracts answer text and usage counters into EvalBench's output schema; it does not copy arbitrary trace fields. The normalized output and report still contain model answers, so keep them private unless approved for release. The full schema is documented in [`docs/PROVIDERS_AND_TRACES.md`](docs/PROVIDERS_AND_TRACES.md).

Get a semantic groundedness signal from a model judge:

```powershell
python -m evalbench semantic-grade --dataset data/support_qa.jsonl --outputs reports/run.jsonl --provider openai-responses --model YOUR_JUDGE_MODEL --results reports/groundedness-scores.jsonl
```

These scores are explicitly **uncalibrated** and must not be treated as a release gate. To select a provisional threshold, collect adjudicated human groundedness labels and semantic scores for at least 30 validation cases, with at least 5 pass and 5 fail labels:

```powershell
python -m evalbench calibrate-groundedness --dataset data/with-splits.jsonl --scores reports/groundedness-scores.jsonl --annotations reports/adjudicated-reviews.jsonl --split validation --report reports/groundedness-calibration.json
```

The calibration command refuses to tune on `test`, requires explicit dataset splits and rejects reviewer conflicts until adjudicated. It reports performance on the same data used to choose the threshold, so independent validation is still required. No human labels are bundled; calibration is only complete after you collect them.

Once a threshold is frozen, score a separate labeled split with the same judge version and model:

```powershell
python -m evalbench semantic-grade --dataset data/with-splits.jsonl --outputs reports/run.jsonl --provider openai-responses --model YOUR_JUDGE_MODEL --results reports/heldout-scores.jsonl --split test
python -m evalbench validate-groundedness --dataset data/with-splits.jsonl --scores reports/heldout-scores.jsonl --annotations reports/adjudicated-heldout-reviews.jsonl --calibration reports/groundedness-calibration.json --split test --report reports/groundedness-validation.json
```

Compare two output files:

```bash
python -m evalbench compare \
  --dataset data/support_qa.jsonl \
  --baseline fixtures/baseline.jsonl \
  --candidate fixtures/candidate.jsonl \
  --report reports/comparison.html
```

Inspect dataset integrity, version and split counts without printing case text:

```bash
python -m evalbench dataset inspect --dataset data/support_qa.jsonl
```

For a dataset whose every case has a `split` field (`train`, `validation`, or `test`), select a split explicitly with `--split test` on `evaluate`, `run`, or `compare`. Reports include a privacy-safe run manifest with dataset/output SHA-256 fingerprints, run ID, timestamp, package/runtime versions and selected split.

## Human review and reviewer agreement

Export a local review packet for each reviewer using an alias. Packets include the case question, supplied context and model answer, but omit expected phrases, automatic grades, model name and run metrics to reduce anchoring:

```powershell
python -m evalbench review export --dataset data/support_qa.jsonl --outputs fixtures/candidate.jsonl --reviewer-id reviewer-a --out reports/review-a.jsonl
python -m evalbench review export --dataset data/support_qa.jsonl --outputs fixtures/candidate.jsonl --reviewer-id reviewer-b --out reports/review-b.jsonl
```

Each reviewer fills `labels` with `pass`, `fail` or `unsure` for `correctness`, `groundedness` and `safety`. Use the shared rubric in [`docs/HUMAN_REVIEW.md`](docs/HUMAN_REVIEW.md). Combine completed JSONL rows into one file, then create an agreement report:

```powershell
python -m evalbench review agreement --annotations reports/reviews.jsonl --report reports/reviewer-agreement.html
```

The HTML and JSON reports include raw agreement, pairwise Cohen's kappa by dimension and case IDs with disagreements for adjudication. They never include answer text, prompt/context text or reviewer notes. Agreement measures consistency, not answer quality; review the rubric, prevalence and disagreements with the reviewers. Review packets contain potentially sensitive prompt and answer text: use only data authorized for human review, share locally with approved reviewers, and do not commit packets to a public repository.

## Real-world ML benchmark

The repository also includes a reproducible intent-classification benchmark on BANKING77, a public dataset of online-banking customer queries with 77 intent labels. It trains a TF-IDF + linear SVM baseline on the official training split and reports accuracy, macro-F1, weighted-F1, top-3 accuracy, and per-intent metrics on the untouched official test split. It downloads the dataset on first run, keeps raw data out of Git, and uses no model API key. See [`benchmarks/banking77/README.md`](benchmarks/banking77/README.md) for source, attribution, exact configuration, and limits.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[benchmark]"
.\.venv\Scripts\python.exe benchmarks/banking77/run.py
```

The committed reference run scores **90.88% accuracy**, **90.88% macro-F1**, and **96.98% top-3 accuracy** on the official 3,080-example test split, using a word/character TF-IDF + LinearSVC baseline. See the [HTML report](benchmarks/banking77/results/report.html) and [machine-readable metrics](benchmarks/banking77/results/report.json) for per-intent results and the reproducibility manifest. This is one classical baseline on one benchmark, not a production performance claim.

## Repository map

- `evalbench/` — dataset validation, provider adapters, trace normalization, deterministic and semantic graders, reports and CLI.
- `benchmarks/banking77/` — reproducible local classifier run on a real, licensed support-intent dataset.
- `data/support_qa.jsonl` — 16 synthetic support questions with contexts and expected behaviors.
- `fixtures/` — hand-authored illustrative outputs for offline use. They are not model-generated results.
- `docs/` — design, data contract, evaluation methodology, human-review rubric and limitations.

## Evaluation design

The examples cover answerable policy questions, missing account data, ambiguous requests, incorrect promises, citation requirements and instruction injection in retrieved material. Required phrase checks encode explicit expectations. Refusal behavior is scored against a small transparent phrase set. Citation checks verify required and allowed source IDs using `[[source_id]]` syntax. The optional semantic judge produces a separate context-support score that can be calibrated against human labels; it does not silently alter deterministic pass rates.

Keep a held-out set for final comparisons and avoid tuning prompts against every evaluation example. Human-label agreement and an independent labeled holdout are required before treating semantic-judge scores as a release criterion.

## Limits and responsible use

- The included support-answer dataset and saved-output fixtures are synthetic and do not establish general model quality.
- BANKING77 measures closed-set intent classification on one English banking dataset; it does not evaluate answer correctness, retrieval grounding, safety, or production bank traffic.
- Phrase, refusal and citation graders can miss paraphrases or mark valid answers incorrectly; inspect the failing cases.
- Citation presence does not prove that a cited passage supports every claim.
- Latency and token counts are descriptive; no quality or cost claim is made about a real model from the included fixtures.
- Do not add customer prompts, credentials, private documents, or client-owned data to a public repository.
- Network mode contacts the endpoint you configure and transmits the dataset text. Review its privacy and retention terms first.

## Roadmap

1. Human-review packet export and reviewer agreement report — implemented in v0.3.0.
2. Split-aware dataset inspection/selection and run metadata/version tracking — implemented in v0.2.0.
3. Provider adapters and saved-trace ingestion — implemented in v0.4.0.
4. Semantic groundedness judge, threshold calibration and separate holdout validation tooling — implemented in v0.4.0; actual calibration/validation results remain pending collection of adjudicated human labels.

## License

MIT. See [`LICENSE`](LICENSE).

