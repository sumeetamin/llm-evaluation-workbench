# Dataset and output contracts

The workbench uses UTF-8 JSON Lines: one JSON object per line. Blank lines are ignored. IDs must be unique within a file.

## Evaluation case

Required:

```json
{"case_id":"refund-window","question":"Can I get a refund?","context":[{"source_id":"refund-policy","text":"Refund requests may be made within 30 days."}]}
```

Optional grader fields:

- `split`: `train`, `validation`, or `test`. IDs remain unique across the complete file; `evalbench dataset inspect` reports split counts and a SHA-256 dataset version without printing prompts. `evaluate`, `run`, and `compare` accept `--split` to select only one assigned split. Filtering requires every case to have a split and at least one case in the selected split. With no `--split`, all cases are evaluated for backward compatibility.
- `category`: grouping label shown in reports.
- `must_include`: every phrase must occur in the answer, case-insensitive.
- `must_not_include`: none of these phrases may occur in the answer, case-insensitive.
- `should_refuse`: expected refusal behavior; default `false`.
- `required_citations`: source IDs that must be cited using `[[source_id]]`.

Each context item requires a `source_id` and `text`. Source IDs must be unique per case for meaningful citation checks.

## Model output

```json
{"case_id":"refund-window","answer":"You may request a refund within 30 days. [[refund-policy]]","model":"model-name","latency_ms":640,"input_tokens":120,"output_tokens":14}
```

Only `case_id` and `answer` are required. Extra IDs not in the dataset are surfaced in the summary; missing IDs fail their corresponding cases.

## Data handling

Output files contain generated answers and operational metadata. The live runner intentionally avoids copying prompt text into output files. Use synthetic or approved data, keep outputs outside version control by default, and review retention requirements before calling a hosted endpoint.

Every HTML and JSON report now includes a run manifest: unique run ID, UTC creation time, command, selected split, dataset SHA-256/version, evaluated case count, output-file fingerprints, Python/package versions, and model/endpoint when applicable. Endpoint query strings and credentials are omitted. The manifest does not contain question or context text.

