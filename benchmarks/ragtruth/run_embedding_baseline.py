"""Evaluate a local sentence-embedding support-similarity baseline on RAGTruth.

This is a deterministic embedding baseline, not an LLM judge or an entailment
model. It uses only the private sample prepared by prepare_semantic_calibration.py
and makes no provider/API requests.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "calibration_private"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "embedding_report.json"
MAX_TOKENS = 256


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def select_threshold(scores: list[float], labels: list[int]) -> dict[str, float]:
    unique = sorted(set(scores))
    candidates = sorted({0.0, 1.0, *unique, *((a + b) / 2 for a, b in zip(unique, unique[1:]))})
    candidates_result = [
        {
            "threshold": threshold,
            "balanced_accuracy": float(balanced_accuracy_score(labels, [score < threshold for score in scores])),
            "accuracy": float(accuracy_score(labels, [score < threshold for score in scores])),
        }
        for threshold in candidates
    ]
    return max(candidates_result, key=lambda row: (row["balanced_accuracy"], -abs(row["threshold"] - 0.5), row["threshold"]))


def metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    predictions = [int(score < threshold) for score in scores]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "cases": len(labels),
        "pass": int(sum(label == 0 for label in labels)),
        "fail": int(sum(label == 1 for label in labels)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "fail_precision": float(precision_score(labels, predictions, zero_division=0)),
        "fail_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fail_f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc_support_similarity": float(roc_auc_score(labels, [-score for score in scores])),
        "average_precision_fail": float(average_precision_score(labels, [-score for score in scores])),
        "confusion_matrix": {
            "true_pass": int(tn), "false_fail": int(fp),
            "missed_fail": int(fn), "detected_fail": int(tp),
        },
    }


def _texts_for_case(case: dict[str, Any], answer: str) -> tuple[str, str]:
    contexts = case.get("context")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"Case {case.get('case_id')!r} has no context entries")
    context_texts = [item.get("text", "") for item in contexts if isinstance(item, dict)]
    context = "\n\n".join(text for text in context_texts if isinstance(text, str) and text.strip())
    if not context.strip() or not answer.strip():
        raise ValueError(f"Case {case.get('case_id')!r} has empty context or answer")
    return context, answer


def lexical_scores(
    fit_cases: list[dict[str, Any]],
    scored_cases: list[dict[str, Any]],
    answers: dict[str, str],
) -> list[float]:
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=1, max_features=50_000,
        sublinear_tf=True, stop_words="english", strip_accents="unicode",
    )
    fit_texts = []
    for case in fit_cases:
        context, answer = _texts_for_case(case, answers[str(case["case_id"])])
        fit_texts.extend((context, answer))
    vectorizer.fit(fit_texts)
    contexts = vectorizer.transform([
        _texts_for_case(case, answers[str(case["case_id"])])[0] for case in scored_cases
    ])
    responses = vectorizer.transform([
        _texts_for_case(case, answers[str(case["case_id"])])[1] for case in scored_cases
    ])
    return (contexts.multiply(responses).sum(axis=1).A1).astype(float).tolist()


def encode_pairs(
    cases: list[dict[str, Any]],
    answers: dict[str, str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        pairs = [_texts_for_case(case, answers[str(case["case_id"])]) for case in batch]
        contexts, responses = zip(*pairs)
        context_tokens = tokenizer(
            list(contexts), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        ).to(device)
        response_tokens = tokenizer(
            list(responses), padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            context_hidden = model(**context_tokens).last_hidden_state
            response_hidden = model(**response_tokens).last_hidden_state

            def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
                return torch.nn.functional.normalize(pooled, p=2, dim=1)

            context_vectors = mean_pool(context_hidden, context_tokens["attention_mask"])
            response_vectors = mean_pool(response_hidden, response_tokens["attention_mask"])
            batch_scores = (context_vectors * response_vectors).sum(dim=1).cpu().tolist()
        scores.extend(float(value) for value in batch_scores)
    return scores


def run(input_dir: Path, output_path: Path, batch_size: int, local_files_only: bool) -> dict[str, Any]:
    dataset_rows = read_jsonl(input_dir / "dataset.jsonl")
    output_rows = read_jsonl(input_dir / "outputs.jsonl")
    annotation_rows = read_jsonl(input_dir / "annotations.jsonl")
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))

    cases = {str(row.get("case_id")): row for row in dataset_rows}
    answers = {str(row.get("case_id")): row.get("answer") for row in output_rows}
    annotations = {str(row.get("case_id")): row for row in annotation_rows}
    if not cases or len(cases) != len(dataset_rows):
        raise ValueError("Dataset is empty or contains duplicate case IDs")
    if set(cases) != set(answers) or set(cases) != set(annotations):
        raise ValueError("Dataset, output, and annotation case IDs do not match")

    labels: dict[str, int] = {}
    source_ids: dict[str, str] = {}
    for case_id, row in cases.items():
        annotation = annotations[case_id]
        groundedness = annotation.get("labels", {}).get("groundedness")
        if groundedness not in {"pass", "fail"}:
            raise ValueError(f"Case {case_id!r} has no usable groundedness label")
        labels[case_id] = int(groundedness == "fail")
        contexts = row.get("context", [])
        ids = {str(item.get("source_id")) for item in contexts if isinstance(item, dict) and item.get("source_id")}
        if len(ids) != 1:
            raise ValueError(f"Case {case_id!r} must refer to exactly one source group")
        source_ids[case_id] = next(iter(ids))

    validation = [row for row in dataset_rows if row.get("split") == "validation"]
    test = [row for row in dataset_rows if row.get("split") == "test"]
    if not validation or not test:
        raise ValueError("Expected separate validation and test splits")
    validation_groups = {source_ids[str(row["case_id"])] for row in validation}
    test_groups = {source_ids[str(row["case_id"])] for row in test}
    if validation_groups & test_groups:
        raise ValueError("Source groups overlap between validation and test")
    if len({str(row["case_id"]) for row in validation}) != len(validation) or len({str(row["case_id"]) for row in test}) != len(test):
        raise ValueError("Duplicate case IDs found in a split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=local_files_only, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=local_files_only, trust_remote_code=False
    ).to(device)
    validation_scores = encode_pairs(validation, answers, tokenizer, model, device, batch_size)
    validation_labels = [labels[str(row["case_id"])] for row in validation]
    calibration = select_threshold(validation_scores, validation_labels)
    threshold = calibration["threshold"]
    test_scores = encode_pairs(test, answers, tokenizer, model, device, batch_size)
    test_labels = [labels[str(row["case_id"])] for row in test]

    lexical_validation_scores = lexical_scores(validation, validation, answers)
    lexical_calibration = select_threshold(lexical_validation_scores, validation_labels)
    lexical_threshold = lexical_calibration["threshold"]
    lexical_test_scores = lexical_scores(validation, test, answers)

    test_by_task: dict[str, Any] = {}
    for task in sorted({str(row.get("category", "unknown")) for row in test}):
        indices = [i for i, row in enumerate(test) if str(row.get("category", "unknown")) == task]
        task_labels = [test_labels[i] for i in indices]
        task_scores = [test_scores[i] for i in indices]
        if len(set(task_labels)) == 2:
            test_by_task[task] = metrics(task_scores, task_labels, threshold)
        else:
            test_by_task[task] = {
                "cases": len(task_labels), "pass": task_labels.count(0), "fail": task_labels.count(1),
                "note": "Only one label is present; discrimination metrics are undefined.",
            }

    report = {
        "benchmark": "RAGTruth response-level proxy labels with a local embedding support-similarity baseline",
        "source": manifest.get("source"),
        "source_revision": manifest.get("source_revision"),
        "source_sha256": manifest.get("source_sha256"),
        "label_mapping": manifest.get("label_mapping"),
        "sampling": manifest.get("sampling"),
        "source_group_counts": manifest.get("source_group_counts"),
        "sample_splits": manifest.get("splits"),
        "threshold_selection": {
            "method": "maximize balanced accuracy on the separate source-disjoint validation sample",
            **calibration,
        },
        "validation": metrics(validation_scores, validation_labels, threshold),
        "test": metrics(test_scores, test_labels, threshold),
        "same_sample_comparison": {
            "embedding_similarity": {
                "validation": metrics(validation_scores, validation_labels, threshold),
                "test": metrics(test_scores, test_labels, threshold),
                "threshold_selection": calibration,
            },
            "tfidf_lexical_similarity": {
                "validation": metrics(lexical_validation_scores, validation_labels, lexical_threshold),
                "test": metrics(lexical_test_scores, test_labels, lexical_threshold),
                "threshold_selection": lexical_calibration,
                "model": "TF-IDF context/answer cosine similarity fitted on validation context and answer text",
            },
        },
        "test_by_task": test_by_task,
        "model": {
            "type": "mean-pooled sentence-embedding cosine similarity",
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "max_tokens_per_text": MAX_TOKENS,
            "device": str(device),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runtime": {"python": platform.python_version(), "created_utc": datetime.now(timezone.utc).isoformat()},
        "limitations": [
            "This is an embedding-similarity baseline, not an LLM judge, entailment model, or groundedness guarantee.",
            "Cosine similarity can miss contradiction, negation, unsupported details, and multi-hop reasoning.",
            "Context and answer are each truncated to 256 tokens before encoding.",
            "Labels are response-level mappings of RAGTruth human hallucination spans, not product-specific reviewer adjudication.",
            "The sample is balanced and small; test metrics do not estimate natural traffic performance.",
            "This benchmark does not validate the workbench's semantic judge or support-QA product behavior.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    score_path = output_path.with_name("embedding_test_scores.jsonl")
    with score_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, score, lexical_score, label in zip(test, test_scores, lexical_test_scores, test_labels):
            stream.write(json.dumps({
                "case_id": str(row["case_id"]),
                "source_id": source_ids[str(row["case_id"])],
                "task_type": row.get("category", "unknown"),
                "support_similarity": score,
                "tfidf_support_similarity": lexical_score,
                "human_hallucination_label": label,
                "predicted_hallucination": int(score < threshold),
                "predicted_hallucination_tfidf": int(lexical_score < lexical_threshold),
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(output_path), "validation": report["validation"], "test": report["test"]}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args.input_dir, args.output, args.batch_size, args.local_files_only)


if __name__ == "__main__":
    main()

