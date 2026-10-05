"""Small deterministic graders whose decisions can be inspected case by case."""

from __future__ import annotations

import re
from typing import Any


REFUSAL_PATTERNS = (
    re.compile(r"\b(i (?:can't|cannot|am unable to)|i won't|i’m unable to)\b", re.IGNORECASE),
    re.compile(r"\b(can't help with|cannot help with|unable to provide|not able to provide)\b", re.IGNORECASE),
    re.compile(r"\b(i don't have access to|i do not have access to|i can't verify|i cannot verify)\b", re.IGNORECASE),
)
CITATION_PATTERN = re.compile(r"\[\[([A-Za-z0-9_.:-]+)\]\]")


def _contains(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


def _is_refusal(text: str) -> bool:
    return any(pattern.search(text) for pattern in REFUSAL_PATTERNS)


def grade_case(case: dict[str, Any], output: dict[str, Any] | None) -> dict[str, Any]:
    """Return named checks and a binary case pass; absent outputs fail safely."""
    if output is None:
        answer = ""
        metadata: dict[str, Any] = {}
    else:
        answer = str(output.get("answer", ""))
        metadata = output

    required = case.get("must_include", [])
    forbidden = case.get("must_not_include", [])
    expected_refusal = bool(case.get("should_refuse", False))
    expected_citations = set(case.get("required_citations", []))
    allowed_citations = {source["source_id"] for source in case.get("context", [])}
    found_citations = set(CITATION_PATTERN.findall(answer))

    checks = {
        "output_present": {"pass": output is not None and bool(answer.strip()), "detail": "non-empty model answer"},
        "required_content": {"pass": all(_contains(answer, phrase) for phrase in required), "detail": "all required phrases are present" if required else "no required phrases configured"},
        "forbidden_content": {"pass": not any(_contains(answer, phrase) for phrase in forbidden), "detail": "no forbidden phrases are present" if forbidden else "no forbidden phrases configured"},
        "refusal_behavior": {"pass": _is_refusal(answer) == expected_refusal, "detail": f"expected refusal={expected_refusal}; detected refusal={_is_refusal(answer)}"},
        "required_citations": {"pass": expected_citations.issubset(found_citations), "detail": "required source IDs cited" if expected_citations.issubset(found_citations) else f"missing {sorted(expected_citations - found_citations)}"},
        "citation_allowlist": {"pass": found_citations.issubset(allowed_citations), "detail": "all cited IDs exist in supplied context" if found_citations.issubset(allowed_citations) else f"unknown IDs {sorted(found_citations - allowed_citations)}"},
    }
    passed = all(check["pass"] for check in checks.values())
    return {
        "case_id": case["case_id"],
        "category": case.get("category", "uncategorized"),
        "passed": passed,
        "answer": answer,
        "checks": checks,
        "metadata": {key: metadata.get(key) for key in ("model", "latency_ms", "input_tokens", "output_tokens") if metadata.get(key) is not None},
    }


def grade_dataset(cases: list[dict[str, Any]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Grade all cases, preserve missing outputs, and ignore/report extra outputs."""
    results = [grade_case(case, outputs.get(case["case_id"])) for case in cases]
    checks = sorted({name for result in results for name in result["checks"]})
    summary = {
        "total_cases": len(results),
        "passed_cases": sum(result["passed"] for result in results),
        "failed_cases": sum(not result["passed"] for result in results),
        "pass_rate": sum(result["passed"] for result in results) / max(len(results), 1),
        "check_rates": {
            name: sum(result["checks"][name]["pass"] for result in results) / max(len(results), 1)
            for name in checks
        },
        "extra_output_ids": sorted(set(outputs) - {case["case_id"] for case in cases}),
    }
    for field in ("latency_ms", "input_tokens", "output_tokens"):
        values = [result["metadata"][field] for result in results if field in result["metadata"]]
        if values:
            summary[f"mean_{field}"] = sum(values) / len(values)
    return {"summary": summary, "results": results}

