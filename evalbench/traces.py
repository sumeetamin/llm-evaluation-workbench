"""Import provider response traces into the prompt-free EvalBench output contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import DataContractError
from .providers import ProviderError, normalize_response

SUPPORTED_TRACE_PROVIDERS = {"openai-responses", "openai-chat-completions", "anthropic-messages"}


def import_traces(trace_path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL envelopes with case_id/provider/response; never copy raw trace data."""
    source = Path(trace_path)
    if not source.is_file():
        raise DataContractError(f"File not found: {source}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DataContractError(f"{source}: trace file must be readable UTF-8 JSONL") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            trace = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataContractError(f"{source}:{line_number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(trace, dict):
            raise DataContractError(f"{source}:{line_number}: each trace must be a JSON object")
        case_id = trace.get("case_id")
        provider = trace.get("provider")
        response = trace.get("response")
        if not isinstance(case_id, str) or not case_id.strip():
            raise DataContractError(f"{source}:{line_number}: case_id must be a non-empty string")
        case_id = case_id.strip()
        if case_id in seen:
            raise DataContractError(f"{source}:{line_number}: duplicate trace for case {case_id!r}")
        seen.add(case_id)
        if not isinstance(provider, str) or provider not in SUPPORTED_TRACE_PROVIDERS:
            raise DataContractError(f"{source}:{line_number}: provider must be one of {', '.join(sorted(SUPPORTED_TRACE_PROVIDERS))}")
        if not isinstance(response, dict):
            raise DataContractError(f"{source}:{line_number}: response must contain the provider's JSON response object")
        try:
            normalized = normalize_response(provider, response)
        except ProviderError as exc:
            raise DataContractError(f"{source}:{line_number}: {exc}") from exc
        metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
        latency = trace.get("latency_ms", metadata.get("latency_ms"))
        if latency is not None and (not isinstance(latency, (int, float)) or latency < 0):
            raise DataContractError(f"{source}:{line_number}: latency_ms must be a non-negative number")
        records.append({
            "case_id": case_id.strip(), "answer": normalized["answer"],
            "model": normalized["model"] or trace.get("model"), "provider": provider,
            "latency_ms": latency, "input_tokens": normalized["input_tokens"],
            "output_tokens": normalized["output_tokens"],
        })
    if not records:
        raise DataContractError(f"{source}: no trace records found")
    return records

