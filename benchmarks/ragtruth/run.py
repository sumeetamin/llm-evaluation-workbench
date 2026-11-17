"""Evaluate a lexical groundedness baseline on the official RAGTruth split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


SOURCE_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
SOURCE_BASE = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{SOURCE_REVISION}/dataset"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_data(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = (data_dir / "response.jsonl", data_dir / "source_info.jsonl")
    for path in paths:
        if path.is_file() and path.stat().st_size:
            continue
        url = f"{SOURCE_BASE}/{path.name}"
        request = urllib.request.Request(url, headers={"User-Agent": "llm-evaluation-workbench-ragtruth/1.0"})
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(64 * 1024):
                    output.write(block)
            if not temporary.stat().st_size:
                raise ValueError(f"Downloaded empty file from {url}")
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def normalize(rows: list[dict[str, Any]], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        source = sources.get(str(row.get("source_id")))
        if source is None:
            raise ValueError(f"Missing source record for response {row.get('id')!r}")
        if row.get("quality") != "good":
            continue
        spans = row.get("labels")
        if not isinstance(spans, list):
            raise ValueError(f"Response {row.get('id')!r} has invalid span annotations")
        # RAGTruth explicitly marks implicit_true spans as correct information
        # omitted from the context. They are not counted as hallucinations here.
        has_hallucination = any(
            isinstance(span, dict) and not span.get("implicit_true", False)
            for span in spans
        )
        context = source.get("source_info")
        if isinstance(context, (dict, list)):
            context = json.dumps(context, ensure_ascii=False, sort_keys=True)
        if not isinstance(context, str) or not isinstance(row.get("response"), str):
            raise ValueError(f"Response {row.get('id')!r} is missing text")
        normalized.append({
            "case_id": str(row["id"]),
            "source_id": str(row["source_id"]),
            "task_type": str(source.get("task_type", "unknown")),
            "split": row.get("split"),
            "context": context,
            "answer": row["response"],
            "has_hallucination": bool(has_hallucination),
        })
    if not normalized:
        raise ValueError("No usable RAGTruth responses were found")
    return normalized


def select_threshold(scores: list[float], labels: list[int]) -> dict[str, float]:
    unique = sorted(set(scores))
    candidates = sorted({0.0, 1.0, *unique, *((a + b) / 2 for a, b in zip(unique, unique[1:]))})
    rows = []
    for threshold in candidates:
        predictions = [score < threshold for score in scores]  # low lexical support => hallucination
        rows.append({
            "threshold": threshold,
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "accuracy": accuracy_score(labels, predictions),
        })
    return max(rows, key=lambda row: (row["balanced_accuracy"], -abs(row["threshold"] - 0.5), row["threshold"]))


def metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    predictions = [int(score < threshold) for score in scores]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "cases": len(labels),
        "non_hallucination": int(sum(label == 0 for label in labels)),
        "hallucination": int(sum(label == 1 for label in labels)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "hallucination_precision": float(precision_score(labels, predictions, zero_division=0)),
        "hallucination_recall": float(recall_score(labels, predictions, zero_division=0)),
        "roc_auc_lexical_support": float(roc_auc_score(labels, [-score for score in scores])),
        "average_precision_hallucination": float(average_precision_score(labels, [-score for score in scores])),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }


def run(data_dir: Path, output_path: Path, seed: int = 42) -> dict[str, Any]:
    response_path, source_path = download_data(data_dir)
    source_rows = read_jsonl(source_path)
    source_map = {str(row["source_id"]): row for row in source_rows}
    rows = normalize(read_jsonl(response_path), source_map)
    official_train = [row for row in rows if row["split"] == "train"]
    official_test = [row for row in rows if row["split"] == "test"]
    if not official_train or not official_test:
        raise ValueError("RAGTruth must provide both official train and test rows")

    train_indices, validation_indices = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=seed
    ).split(official_train, groups=[row["source_id"] for row in official_train]))
    train = [official_train[i] for i in train_indices]
    validation = [official_train[i] for i in validation_indices]
    if {row["source_id"] for row in train} & {row["source_id"] for row in validation}:
        raise ValueError("Source leakage between train and validation")
    if {row["source_id"] for row in official_train} & {row["source_id"] for row in official_test}:
        raise ValueError("Official RAGTruth source IDs overlap between train and test")

    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=150_000,
        sublinear_tf=True, stop_words="english", strip_accents="unicode",
    )
    fit_texts = [row["context"] for row in train] + [row["answer"] for row in train]
    vectorizer.fit(fit_texts)

    def score(partition: list[dict[str, Any]]) -> list[float]:
        contexts = vectorizer.transform([row["context"] for row in partition])
        answers = vectorizer.transform([row["answer"] for row in partition])
        return (contexts.multiply(answers).sum(axis=1).A1).astype(float).tolist()

    validation_scores = score(validation)
    validation_labels = [int(row["has_hallucination"]) for row in validation]
    calibration = select_threshold(validation_scores, validation_labels)
    threshold = calibration["threshold"]
    test_scores = score(official_test)
    test_labels = [int(row["has_hallucination"]) for row in official_test]

    per_task = {}
    for task in sorted({row["task_type"] for row in official_test}):
        indices = [i for i, row in enumerate(official_test) if row["task_type"] == task]
        y = [test_labels[i] for i in indices]
        s = [test_scores[i] for i in indices]
        per_task[task] = metrics(s, y, threshold) if len(set(y)) == 2 else {
            "cases": len(y), "hallucination": sum(y), "note": "Only one class is present; AUC is not defined."
        }

    report = {
        "benchmark": "RAGTruth response-level hallucination detection with a lexical support baseline",
        "source": "ParticleMedia/RAGTruth official response.jsonl and source_info.jsonl",
        "source_revision": SOURCE_REVISION,
        "response_sha256": sha256(response_path),
        "source_info_sha256": sha256(source_path),
        "dataset_rows": len(rows),
        "excluded_quality_rows": len(read_jsonl(response_path)) - len(rows),
        "quality_filter": "quality == good; incorrect_refusal and truncated excluded",
        "label_mapping": "hallucination if any annotated span is not marked implicit_true; all other good rows are non-hallucination",
        "splits": {
            "model_fit": {"cases": len(train), "source_groups": len({row['source_id'] for row in train})},
            "threshold_selection": {"cases": len(validation), "source_groups": len({row['source_id'] for row in validation}), **metrics(validation_scores, validation_labels, threshold)},
            "official_test": {"source_groups": len({row['source_id'] for row in official_test}), **metrics(test_scores, test_labels, threshold)},
        },
        "threshold_selection": {"method": "maximize balanced accuracy on a source-group-held-out 20% slice of official training data", **calibration},
        "test_by_task": per_task,
        "model": {"type": "TF-IDF context/answer cosine similarity", "threshold": threshold, "scikit_learn": sklearn.__version__},
        "runtime": {"python": platform.python_version(), "created_utc": datetime.now(timezone.utc).isoformat()},
        "limitations": [
            "This lexical baseline is not the workbench's LLM semantic judge and does not calibrate that judge.",
            "RAGTruth's span labels are mapped to response-level labels; implicit_true spans are not counted as hallucinations.",
            "Lexical overlap is not entailment: paraphrases, negation, factual correctness and domain-specific meaning are not modeled.",
            "Results apply to the public RAGTruth task mixture and this fixed baseline only; they do not establish production quality.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scores_path = output_path.with_name("test_scores.jsonl")
    with scores_path.open("w", encoding="utf-8") as stream:
        for row, score_value, label in zip(official_test, test_scores, test_labels):
            stream.write(json.dumps({
                "case_id": row["case_id"], "source_id": row["source_id"],
                "task_type": row["task_type"], "lexical_support_score": score_value,
                "human_hallucination_label": label, "predicted_hallucination": int(score_value < threshold),
            }, ensure_ascii=False) + "\n")
    print(json.dumps({
        "report": str(output_path), "cases": report["splits"]["official_test"]["cases"],
        "threshold": threshold, "test_metrics": report["splits"]["official_test"],
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.output, seed=args.seed)


if __name__ == "__main__":
    main()

