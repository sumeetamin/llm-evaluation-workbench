"""Load and validate the JSONL dataset and model-output contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataContractError(ValueError):
    """Raised when an evaluation input violates the documented data contract."""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise DataContractError(f"File not found: {source}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataContractError(f"{source}:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise DataContractError(f"{source}:{line_number}: each row must be a JSON object")
        records.append(record)
    if not records:
        raise DataContractError(f"{source}: no JSONL records found")
    return records


def _string_list(value: Any, field: str, row_number: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DataContractError(f"dataset row {row_number}: {field} must be a list of strings")
    return value


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read cases and enforce the stable schema used by graders and providers."""
    cases = _read_jsonl(path)
    seen: set[str] = set()
    for row_number, case in enumerate(cases, 1):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise DataContractError(f"dataset row {row_number}: case_id must be a non-empty string")
        if case_id in seen:
            raise DataContractError(f"dataset row {row_number}: duplicate case_id {case_id!r}")
        seen.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise DataContractError(f"dataset row {row_number} ({case_id}): question is required")
        context = case.get("context", [])
        if not isinstance(context, list):
            raise DataContractError(f"dataset row {row_number} ({case_id}): context must be a list")
        for source in context:
            if not isinstance(source, dict) or not isinstance(source.get("source_id"), str) or not isinstance(source.get("text"), str):
                raise DataContractError(f"dataset row {row_number} ({case_id}): context items need source_id and text strings")
        for field in ("must_include", "must_not_include", "required_citations"):
            case[field] = _string_list(case.get(field), field, row_number)
        if not isinstance(case.get("should_refuse", False), bool):
            raise DataContractError(f"dataset row {row_number} ({case_id}): should_refuse must be boolean")
        case.setdefault("category", "uncategorized")
    return cases


def load_outputs(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load output rows keyed by case_id, rejecting duplicates and malformed rows."""
    rows = _read_jsonl(path)
    outputs: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        case_id, answer = row.get("case_id"), row.get("answer")
        if not isinstance(case_id, str) or not case_id.strip():
            raise DataContractError(f"output row {row_number}: case_id must be a non-empty string")
        if case_id in outputs:
            raise DataContractError(f"output row {row_number}: duplicate case_id {case_id!r}")
        if not isinstance(answer, str):
            raise DataContractError(f"output row {row_number} ({case_id}): answer must be a string")
        outputs[case_id] = row
    return outputs

