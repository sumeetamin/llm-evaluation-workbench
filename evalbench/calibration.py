"""Calibrate a semantic groundedness threshold against adjudicated human labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import DataContractError, load_dataset


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DataContractError(f"Cannot read {source}: {exc}") from exc
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataContractError(f"{source}:{number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise DataContractError(f"{source}:{number}: each JSONL row must be an object")
        rows.append(row)
    return rows


def calibrate_groundedness(
    dataset_path: str | Path, score_path: str | Path, annotation_path: str | Path, *, split: str = "validation",
) -> dict[str, Any]:
    """Choose an exploratory threshold from a non-test split and human labels.

    At least 30 adjudicated cases and five pass/fail labels per class are required.
    This prevents tiny samples from being presented as a credible calibration.
    """
    if split == "test":
        raise DataContractError("Refusing to tune a threshold on the held-out test split.")
    all_cases = load_dataset(dataset_path)
    if any(case.get("split") is None for case in all_cases):
        raise DataContractError("Calibration requires every dataset case to have an explicit train/validation/test split.")
    cases = load_dataset(dataset_path, split=split)
    score_rows = _read_jsonl(score_path)
    identities = {
        (row.get("judge_version"), row.get("judge_provider"), row.get("judge_model"))
        for row in score_rows
    }
    if len(identities) != 1:
        raise DataContractError("Calibration scores must come from exactly one judge version, provider, and model.")
    identity = next(iter(identities))
    scores: dict[str, float] = {}
    for row in score_rows:
        case_id, score = row.get("case_id"), row.get("groundedness_score")
        if not isinstance(case_id, str) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise DataContractError("Score rows need a case_id and groundedness_score from 0 to 1.")
        if case_id in scores:
            raise DataContractError(f"Duplicate semantic score for {case_id!r}.")
        scores[case_id] = float(score)
    labels_by_case: dict[str, set[str]] = {}
    for row in _read_jsonl(annotation_path):
        case_id = row.get("case_id")
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else row
        label = labels.get("groundedness")
        if isinstance(case_id, str) and isinstance(label, str) and label in {"pass", "fail", "unsure"}:
            labels_by_case.setdefault(case_id, set()).add(label)
    selected = []
    ambiguous = []
    for case in cases:
        case_id = case["case_id"]
        available = labels_by_case.get(case_id, set()) - {"unsure"}
        if len(available) > 1:
            ambiguous.append(case_id)
        elif len(available) == 1 and case_id in scores:
            selected.append((case_id, scores[case_id], next(iter(available))))
    if ambiguous:
        raise DataContractError(
            "Conflicting reviewer labels need human adjudication before calibration (case IDs: "
            + ", ".join(sorted(ambiguous)[:10]) + ")."
        )
    positives = sum(label == "pass" for _, _, label in selected)
    negatives = sum(label == "fail" for _, _, label in selected)
    if len(selected) < 30 or min(positives, negatives) < 5:
        raise DataContractError(
            f"Calibration needs at least 30 scored, adjudicated {split} cases and at least 5 pass and 5 fail labels; "
            f"found {len(selected)} cases ({positives} pass, {negatives} fail)."
        )
    unique_scores = sorted({score for _, score, _ in selected})
    thresholds = sorted({
        0.0, 1.0, *unique_scores,
        *((left + right) / 2 for left, right in zip(unique_scores, unique_scores[1:])),
    })
    candidates = []
    for threshold in thresholds:
        true_positive = sum(score >= threshold and label == "pass" for _, score, label in selected)
        false_negative = positives - true_positive
        true_negative = sum(score < threshold and label == "fail" for _, score, label in selected)
        false_positive = negatives - true_negative
        sensitivity = true_positive / positives
        specificity = true_negative / negatives
        balanced_accuracy = (sensitivity + specificity) / 2
        candidates.append({
            "threshold": threshold, "balanced_accuracy": balanced_accuracy,
            "sensitivity": sensitivity, "specificity": specificity,
            "accuracy": (true_positive + true_negative) / len(selected),
        })
    best = max(candidates, key=lambda row: (row["balanced_accuracy"], -abs(row["threshold"] - 0.5), row["threshold"]))
    return {
        "calibration_version": 1, "status": "exploratory_threshold_requires_independent_validation",
        "judge_identity": {"version": identity[0], "provider": identity[1], "model": identity[2]},
        "split": split, "cases": len(selected), "pass_labels": positives, "fail_labels": negatives,
        "threshold": best["threshold"], "metrics_on_calibration_data": best,
        "warning": "Threshold was selected and scored on the same labeled cases; do not claim independent accuracy. Validate on a separate human-labeled set before using as a release gate.",
    }


def validate_groundedness(
    dataset_path: str | Path, score_path: str | Path, annotation_path: str | Path,
    calibration_path: str | Path, *, split: str = "test",
) -> dict[str, Any]:
    """Evaluate a frozen calibrated threshold against a separate labeled split."""
    try:
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError("Calibration report must be a readable UTF-8 JSON file.") from exc
    if not isinstance(calibration, dict) or calibration.get("status") != "exploratory_threshold_requires_independent_validation":
        raise DataContractError("Threshold file is not a valid EvalBench calibration report.")
    if split == calibration.get("split"):
        raise DataContractError("Independent validation must use a different dataset split from threshold selection.")
    threshold = calibration.get("threshold")
    identity = calibration.get("judge_identity")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1 or not isinstance(identity, dict):
        raise DataContractError("Calibration report is missing a valid threshold or judge identity.")
    cases = load_dataset(dataset_path, split=split)
    score_rows = _read_jsonl(score_path)
    score_identity = {
        (row.get("judge_version"), row.get("judge_provider"), row.get("judge_model"))
        for row in score_rows
    }
    expected = (identity.get("version"), identity.get("provider"), identity.get("model"))
    if score_identity != {expected}:
        raise DataContractError("Independent validation scores must use the same frozen judge version, provider, and model.")
    scores: dict[str, float] = {}
    for row in score_rows:
        case_id, score = row.get("case_id"), row.get("groundedness_score")
        if not isinstance(case_id, str) or isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise DataContractError("Score rows need a case_id and groundedness_score from 0 to 1.")
        if case_id in scores:
            raise DataContractError(f"Duplicate semantic score for {case_id!r}.")
        scores[case_id] = float(score)
    labels_by_case: dict[str, set[str]] = {}
    for row in _read_jsonl(annotation_path):
        case_id = row.get("case_id")
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else row
        label = labels.get("groundedness")
        if isinstance(case_id, str) and isinstance(label, str) and label in {"pass", "fail", "unsure"}:
            labels_by_case.setdefault(case_id, set()).add(label)
    selected = []
    conflicts = []
    for case in cases:
        case_id = case["case_id"]
        available = labels_by_case.get(case_id, set()) - {"unsure"}
        if len(available) > 1:
            conflicts.append(case_id)
        elif len(available) == 1 and case_id in scores:
            selected.append((case_id, scores[case_id], next(iter(available))))
    if conflicts:
        raise DataContractError("Conflicting labels need human adjudication before validation (case IDs: " + ", ".join(sorted(conflicts)[:10]) + ").")
    positives = sum(label == "pass" for _, _, label in selected)
    negatives = sum(label == "fail" for _, _, label in selected)
    if len(selected) < 30 or min(positives, negatives) < 5:
        raise DataContractError(
            f"Independent validation needs at least 30 scored, adjudicated {split} cases and 5 labels per class; "
            f"found {len(selected)} cases ({positives} pass, {negatives} fail)."
        )
    true_positive = sum(score >= threshold and label == "pass" for _, score, label in selected)
    true_negative = sum(score < threshold and label == "fail" for _, score, label in selected)
    sensitivity = true_positive / positives
    specificity = true_negative / negatives
    return {
        "validation_version": 1, "status": "independent_holdout_metrics",
        "threshold": threshold, "calibration_split": calibration["split"], "validation_split": split,
        "judge_identity": identity, "cases": len(selected), "pass_labels": positives, "fail_labels": negatives,
        "metrics": {
            "accuracy": (true_positive + true_negative) / len(selected),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "sensitivity": sensitivity, "specificity": specificity,
        },
        "warning": "Metrics apply to this labeled holdout and judge version only; monitor for data and model drift.",
    }

