"""Reproducibility metadata for evaluation runs."""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from . import __version__


def _file_record(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return {"path": source.name, "status": "missing"}
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"path": source.name, "sha256": digest}


def build_run_metadata(
    dataset_path: str | Path,
    cases: list[dict[str, Any]],
    *,
    split: str | None = None,
    command: str,
    outputs: list[str | Path] | None = None,
    model: str | None = None,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Create a run manifest with stable dataset/output fingerprints and no prompt text."""
    dataset = _file_record(dataset_path)
    assert dataset is not None
    payload: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "command": command,
        "dataset": {
            **dataset,
            "version": f"sha256:{dataset['sha256']}" if "sha256" in dataset else None,
            "selected_split": split or "all",
            "evaluated_cases": len(cases),
        },
        "runtime": {"evalbench_version": __version__, "python": platform.python_version()},
        "outputs": [_file_record(path) for path in outputs or []],
    }
    if model:
        payload["model"] = model
    if endpoint:
        parsed = urlsplit(endpoint)
        safe_netloc = parsed.hostname or ""
        if parsed.port:
            safe_netloc += f":{parsed.port}"
        payload["endpoint"] = urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))
    return payload


def metadata_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, indent=2, ensure_ascii=False)

