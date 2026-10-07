"""Blinded human-review packets and inter-reviewer agreement summaries."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from .dataset import DataContractError, load_dataset, load_outputs

REVIEW_DIMENSIONS = ("correctness", "groundedness", "safety")
REVIEW_LABELS = {"pass", "fail", "unsure"}


def export_review_packet(
    dataset_path: str | Path,
    outputs_path: str | Path,
    reviewer_id: str,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Create one blinded, locally reviewed row per dataset case."""
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise DataContractError("reviewer_id must be a non-empty alias")
    cases = load_dataset(dataset_path, split=split)
    outputs = load_outputs(outputs_path)
    case_ids = {case["case_id"] for case in cases}
    missing = sorted(case_ids - set(outputs))
    extra = sorted(set(outputs) - case_ids)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing outputs for {len(missing)} cases (for example: {', '.join(missing[:3])})")
        if extra:
            details.append(f"outputs contain {len(extra)} unknown case IDs (for example: {', '.join(extra[:3])})")
        raise DataContractError("cannot export review packet: " + "; ".join(details))

    dataset_digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    outputs_digest = hashlib.sha256(Path(outputs_path).read_bytes()).hexdigest()
    review_set_id = hashlib.sha256(
        json.dumps(
            {"dataset_sha256": dataset_digest, "outputs_sha256": outputs_digest, "split": split or "all"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return [
        {
            "case_id": case["case_id"],
            "reviewer_id": reviewer_id.strip(),
            "review_set_id": review_set_id,
            "question": case["question"],
            "context": case.get("context", []),
            "answer": outputs[case["case_id"]]["answer"],
            "labels": {dimension: None for dimension in REVIEW_DIMENSIONS},
            "notes": "",
        }
        for case in cases
    ]


def _load_annotations(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path)
    if not source.is_file():
        raise DataContractError(f"File not found: {source}")
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataContractError(f"{source}: annotations must be UTF-8 JSONL") from exc
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    review_set_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataContractError(f"{source}:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise DataContractError(f"{source}:{line_number}: each annotation must be a JSON object")
        case_id, reviewer_id = row.get("case_id"), row.get("reviewer_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise DataContractError(f"{source}:{line_number}: case_id must be a non-empty string")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise DataContractError(f"{source}:{line_number}: reviewer_id must be a non-empty alias")
        review_set_id = row.get("review_set_id")
        if not isinstance(review_set_id, str) or len(review_set_id) != 64 or any(char not in "0123456789abcdef" for char in review_set_id):
            raise DataContractError(f"{source}:{line_number}: review_set_id must be a 64-character lowercase SHA-256 fingerprint")
        review_set_ids.add(review_set_id)
        key = (case_id, reviewer_id)
        if key in seen:
            raise DataContractError(f"{source}:{line_number}: duplicate annotation for case {case_id!r} by {reviewer_id!r}")
        seen.add(key)
        labels = row.get("labels")
        if not isinstance(labels, dict) or set(labels) != set(REVIEW_DIMENSIONS):
            expected = ", ".join(REVIEW_DIMENSIONS)
            raise DataContractError(f"{source}:{line_number} ({case_id}): labels must contain exactly {expected}")
        for dimension, value in labels.items():
            if value is None or value == "":
                continue
            if not isinstance(value, str) or value not in REVIEW_LABELS:
                choices = ", ".join(sorted(REVIEW_LABELS))
                raise DataContractError(f"{source}:{line_number} ({case_id}): {dimension} must be one of {choices} or blank")
        records.append({"case_id": case_id, "reviewer_id": reviewer_id.strip(), "review_set_id": review_set_id, "labels": labels})
    if not records:
        raise DataContractError(f"{source}: no annotation rows found")
    reviewers = {row["reviewer_id"] for row in records}
    if len(reviewers) < 2:
        raise DataContractError("agreement reporting needs annotations from at least two distinct reviewer IDs")
    if len(review_set_ids) != 1:
        raise DataContractError("all annotations must come from the same review_set_id; do not combine different datasets, outputs, or splits")
    return records, hashlib.sha256(raw).hexdigest()


def _pair_agreement(left: list[str], right: list[str]) -> dict[str, Any]:
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    labels = REVIEW_LABELS
    left_counts = {label: left.count(label) for label in labels}
    right_counts = {label: right.count(label) for label in labels}
    expected = sum((left_counts[label] / n) * (right_counts[label] / n) for label in labels)
    if abs(1 - expected) < 1e-12:
        kappa = None
        status = "undefined_single_category"
    else:
        kappa = (observed - expected) / (1 - expected)
        status = "defined"
    return {
        "common_cases": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "kappa_status": status,
    }


def build_agreement_report(annotation_path: str | Path) -> dict[str, Any]:
    """Compute per-dimension, pairwise Cohen's kappa and find disagreements."""
    records, digest = _load_annotations(annotation_path)
    raters = sorted({row["reviewer_id"] for row in records})
    by_case: dict[str, dict[str, dict[str, str]]] = {}
    label_count_by_reviewer = {reviewer: 0 for reviewer in raters}
    for row in records:
        label_count_by_reviewer[row["reviewer_id"]] += sum(
            row["labels"][dimension] in REVIEW_LABELS for dimension in REVIEW_DIMENSIONS
        )
        labels = {key: value for key, value in row["labels"].items() if value in REVIEW_LABELS}
        if labels:
            by_case.setdefault(row["case_id"], {})[row["reviewer_id"]] = labels

    dimensions: dict[str, Any] = {}
    disagreements: list[dict[str, Any]] = []
    for dimension in REVIEW_DIMENSIONS:
        pairs: list[dict[str, Any]] = []
        for left_id, right_id in combinations(raters, 2):
            left_values: list[str] = []
            right_values: list[str] = []
            for case_id in sorted(by_case):
                case_labels = by_case[case_id]
                if left_id in case_labels and right_id in case_labels:
                    left_label = case_labels[left_id].get(dimension)
                    right_label = case_labels[right_id].get(dimension)
                    if left_label in REVIEW_LABELS and right_label in REVIEW_LABELS:
                        left_values.append(left_label)
                        right_values.append(right_label)
            if left_values:
                pairs.append({"reviewer_a": left_id, "reviewer_b": right_id, **_pair_agreement(left_values, right_values)})
            else:
                pairs.append({
                    "reviewer_a": left_id, "reviewer_b": right_id, "common_cases": 0,
                    "observed_agreement": None, "expected_agreement": None,
                    "cohen_kappa": None, "kappa_status": "no_shared_labels",
                })
        defined = [pair["cohen_kappa"] for pair in pairs if pair["cohen_kappa"] is not None]
        dimensions[dimension] = {
            "mean_pairwise_kappa": sum(defined) / len(defined) if defined else None,
            "defined_pair_count": len(defined),
            "pairs": pairs,
        }

    for case_id in sorted(by_case):
        for dimension in REVIEW_DIMENSIONS:
            labels = {
                reviewer: values[dimension]
                for reviewer, values in by_case[case_id].items()
                if values.get(dimension) in REVIEW_LABELS
            }
            if len(labels) > 1 and len(set(labels.values())) > 1:
                disagreements.append({"case_id": case_id, "dimension": dimension, "labels": dict(sorted(labels.items()))})

    return {
        "report_version": 1,
        "review_set_id": records[0]["review_set_id"],
        "annotation_sha256": digest,
        "annotation_rows": len(records),
        "cases_with_labels": len(by_case),
        "reviewers": raters,
        "label_counts_by_reviewer": label_count_by_reviewer,
        "dimensions": dimensions,
        "disagreements": disagreements,
        "interpretation": (
            "Agreement measures consistency, not correctness. Cohen's kappa is sensitive to label prevalence; "
            "interpret it alongside raw agreement, sample counts, and human discussion."
        ),
    }

