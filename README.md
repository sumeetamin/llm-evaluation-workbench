# LLM Evaluation Workbench

An eval-driven development tool for retrieval-grounded customer-support assistants. It runs a versioned JSONL evaluation set against saved model outputs or a live OpenAI Responses-compatible endpoint, applies transparent deterministic graders, and produces a case-level HTML report. It also compares two runs to surface regressions and improvements.

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

Run the set against an OpenAI Responses-compatible endpoint. Set the key in the environment; never put it in a command, config file, output fixture or GitHub repository.

```bash
# PowerShell
$env:OPENAI_API_KEY = "your-key"
python -m evalbench run --dataset data/support_qa.jsonl --model YOUR_MODEL --api-key-env OPENAI_API_KEY --outputs reports/run.jsonl --report reports/run.html
```

`run` sends each dataset question and its context to the configured endpoint. Use only data you are authorized to transmit. It sends requests sequentially and does not store prompts with model outputs.

Compare two output files:

```bash
python -m evalbench compare \
  --dataset data/support_qa.jsonl \
  --baseline fixtures/baseline.jsonl \
  --candidate fixtures/candidate.jsonl \
  --report reports/comparison.html
```

## Repository map

- `evalbench/` — dataset validation, provider client, graders, reports and CLI.
- `data/support_qa.jsonl` — 16 synthetic support questions with contexts and expected behaviors.
- `fixtures/` — hand-authored illustrative outputs for offline use. They are not model-generated results.
- `docs/` — design, data contract, evaluation methodology and limitations.

## Evaluation design

The examples cover answerable policy questions, missing account data, ambiguous requests, incorrect promises, citation requirements and instruction injection in retrieved material. Required phrase checks encode explicit expectations. Refusal behavior is scored against a small transparent phrase set. Citation checks verify required and allowed source IDs using `[[source_id]]` syntax.

This first version intentionally avoids an LLM-as-judge score. Add one only after defining a rubric and measuring its agreement with multiple human reviewers. Keep a held-out set for final comparisons and avoid tuning prompts against every evaluation example.

## Limits and responsible use

- The 16-row dataset is synthetic and is not a benchmark for general model quality.
- Phrase, refusal and citation graders can miss paraphrases or mark valid answers incorrectly; inspect the failing cases.
- Citation presence does not prove that a cited passage supports every claim.
- Latency and token counts are descriptive; no quality or cost claim is made about a real model from the included fixtures.
- Do not add customer prompts, credentials, private documents, or client-owned data to a public repository.
- Network mode contacts the endpoint you configure and transmits the dataset text. Review its privacy and retention terms first.

## Roadmap

1. Add a human-review annotation loop and reviewer agreement report.
2. Add split-aware eval-set management and run metadata/version tracking.
3. Add a provider interface for more response schemas and trace ingestion.
4. Add semantic groundedness grading calibrated against human labels.

## License

MIT. See [`LICENSE`](LICENSE).

