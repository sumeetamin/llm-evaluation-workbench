# Providers, trace import and semantic groundedness

## Live provider adapters

`evalbench run` supports three request and response formats:

| `--provider` | API shape | Default key variable |
| --- | --- | --- |
| `openai-responses` | OpenAI Responses | `OPENAI_API_KEY` |
| `openai-chat-completions` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `anthropic-messages` | Anthropic Messages | `ANTHROPIC_API_KEY` |

Example:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
python -m evalbench run --dataset data/support_qa.jsonl --provider anthropic-messages --model YOUR_MODEL --outputs reports/anthropic.jsonl --report reports/anthropic.html
```

Provider requests use HTTPS (HTTP is allowed only for localhost development). Endpoint overrides are supported with `--endpoint`. Keys are read from environment variables, never embedded in output. The runner stores answer text and safe usage metadata, not request prompts. Use only data authorized for transmission and keep answer files private.

## Saved trace envelope

`trace-import` expects one JSON object per line. Associate every trace with the dataset's `case_id`, identify the response schema with `provider`, and put the provider's unmodified JSON response object in `response`:

```json
{"case_id":"refund-window","provider":"openai-chat-completions","latency_ms":642,"response":{"model":"example-model","choices":[{"message":{"content":"You can request a refund within 30 days. [[refund-policy]]"}}],"usage":{"prompt_tokens":120,"completion_tokens":14}}}
```

Supported trace values match the live adapter names above. Each case ID may occur only once per imported file. A response must contain readable text. Optional `latency_ms` may also be provided inside `metadata`. Model name and input/output token counts are extracted when the schema provides them. Arbitrary request, account, and trace fields are discarded. Missing dataset outputs are still reported as failed cases; unexpected IDs are surfaced by the evaluation summary.

```powershell
python -m evalbench trace-import --dataset data/support_qa.jsonl --traces reports/traces.jsonl --outputs reports/imported.jsonl --report reports/imported.html
```

The normalized output includes model answers and should be handled as sensitive, even though raw trace payloads are not copied.

## Semantic groundedness judge

`semantic-grade` asks a selected provider to rate how well factual answer claims are supported by the supplied context. It produces a score from 0 to 1, a count of unsupported claims, and a constrained rationale code for each `case_id`. It deliberately does not mix this probabilistic score into deterministic pass/fail results. The judge sees the question, context and answer, so only run it on approved data.

The judge output is a hypothesis-generating signal. It may miss subtle entailment errors, inherit model biases, and vary by provider/model/version. Keep the judge identity with results and inspect examples alongside independent human ratings.

## Human-label threshold calibration

The calibration command expects:

- A dataset with every case assigned to `train`, `validation` or `test`.
- Semantic score JSONL created by `semantic-grade`.
- Review JSONL created by `review export` with `labels.groundedness` filled. Conflicting reviewer labels must be adjudicated by a human before calibration.
- At least 30 scored, adjudicated cases on the selected non-test split, including at least five `pass` and five `fail` labels.

The command selects the threshold that maximizes balanced accuracy on that labeled split. It refuses `--split test` to avoid tuning on the held-out test set. The reported metrics are computed on the same records used to select the threshold: they are descriptive calibration-fit metrics, not independent validation. The score file must contain one fixed judge version/provider/model. Before claiming calibrated performance or using the score as a release gate, evaluate the frozen judge and threshold on a separately collected human-labeled holdout set with `evalbench validate-groundedness`; it reports metrics without reselecting the threshold and rejects the calibration split. `unsure` and incomplete annotations are excluded. No human annotation data or calibration result is included in the repository.

