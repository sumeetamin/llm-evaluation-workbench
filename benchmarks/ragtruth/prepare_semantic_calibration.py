"""Prepare a private, source-disjoint RAGTruth sample for semantic judge calibration.

This script only downloads/reads the public RAGTruth files and writes local JSONL
inputs. It does not call an LLM provider. The official span annotations are mapped
to response-level groundedness proxies; see the benchmark README before interpreting
any later calibration metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
SOURCE_BASE = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{SOURCE_REVISION}/dataset"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "calibration_private"
TASK_QUESTIONS = {
    "Summary": "Summarize the supplied source accurately without adding unsupported facts.",
    "QA": "Answer the user question using only the supplied source.",
    "Data2txt": "Convert the supplied source to structured text without inventing facts.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_data(data_dir: Path) -> tuple[Path, Path]:
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
                while chunk := response.read(64 * 1024):
                    output.write(chunk)
            if not temporary.stat().st_size:
                raise ValueError(f"Downloaded empty file from {url}")
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return paths


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            records.append(row)
    return records


def _normalize(
    response_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    sources = {str(row["source_id"]): row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("RAGTruth source_info contains duplicate source IDs")
    normalized = []
    excluded_quality = 0
    for row in response_rows:
        if row.get("quality") != "good":
            excluded_quality += 1
            continue
        source_id = str(row.get("source_id"))
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"Missing source record for response {row.get('id')!r}")
        spans = row.get("labels")
        if not isinstance(spans, list):
            raise ValueError(f"Response {row.get('id')!r} has invalid span labels")
        context = source.get("source_info")
        if isinstance(context, (dict, list)):
            context = json.dumps(context, ensure_ascii=False, sort_keys=True)
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"Source {source_id!r} has no usable source text")
        answer = row.get("response")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Response {row.get('id')!r} has no answer text")
        task_type = str(source.get("task_type", "unknown"))
        has_hallucination = any(
            isinstance(span, dict) and not span.get("implicit_true", False)
            for span in spans
        )
        normalized.append({
            "case_id": f"ragtruth-{row['id']}",
            "source_group": source_id,
            "task_type": task_type,
            "official_split": row.get("split"),
            "context": context,
            "answer": answer,
            "has_hallucination": bool(has_hallucination),
            "model": row.get("model"),
        })
    return normalized, excluded_quality


def _source_disjoint_validation_split(
    rows: list[dict[str, Any]], seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    official_train = [row for row in rows if row["official_split"] == "train"]
    official_test = [row for row in rows if row["official_split"] == "test"]
    if not official_train or not official_test:
        raise ValueError("RAGTruth must provide official train and test splits")
    train_groups = {row["source_group"] for row in official_train}
    test_groups = {row["source_group"] for row in official_test}
    if train_groups & test_groups:
        raise ValueError("RAGTruth official train and test source IDs overlap")

    groups = sorted(train_groups)
    random.Random(seed).shuffle(groups)
    validation_count = max(1, round(len(groups) * 0.2))
    validation_groups = set(groups[:validation_count])
    validation = [row for row in official_train if row["source_group"] in validation_groups]
    fit_groups = train_groups - validation_groups
    if fit_groups & validation_groups or validation_groups & test_groups or fit_groups & test_groups:
        raise ValueError("Source overlap found between model-fit, calibration, and official test partitions")
    return validation, official_test, validation_groups, test_groups


def _balanced_source_sample(rows: list[dict[str, Any]], per_class: int, rng: random.Random) -> list[dict[str, Any]]:
    by_label_and_source: dict[bool, dict[str, list[dict[str, Any]]]] = {
        False: defaultdict(list), True: defaultdict(list),
    }
    for row in rows:
        by_label_and_source[row["has_hallucination"]][row["source_group"]].append(row)

    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    # Sample failure and pass examples from different source groups, preventing
    # repeated documents from inflating the apparent sample size.
    for label in (True, False):
        candidates = [group for group in by_label_and_source[label] if group not in used_sources]
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            label_name = "fail" if label else "pass"
            raise ValueError(
                f"Need {per_class} distinct source groups with {label_name} examples; found {len(candidates)}"
            )
        chosen_groups = candidates[:per_class]
        used_sources.update(chosen_groups)
        for group in chosen_groups:
            selected.append(rng.choice(by_label_and_source[label][group]))
    rng.shuffle(selected)
    return selected


def prepare(data_dir: Path, output_dir: Path, seed: int = 42, per_class: int = 50) -> dict[str, Any]:
    if per_class < 15:
        raise ValueError("Use at least 15 cases per class so both partitions exceed the calibration minimum")
    response_path, source_path = _download_data(data_dir)
    normalized, excluded_quality = _normalize(_read_jsonl(response_path), _read_jsonl(source_path))
    validation_pool, test_pool, validation_groups, test_groups = _source_disjoint_validation_split(normalized, seed)
    if validation_groups & test_groups:
        raise ValueError("Calibration and holdout samples must be source-disjoint")

    validation_sample = _balanced_source_sample(validation_pool, per_class, random.Random(seed + 1))
    test_sample = _balanced_source_sample(test_pool, per_class, random.Random(seed + 2))
    selected = [(row, "validation") for row in validation_sample] + [(row, "test") for row in test_sample]

    dataset_rows = []
    output_rows = []
    annotation_rows = []
    for row, split in selected:
        task_type = row["task_type"]
        question = TASK_QUESTIONS.get(task_type, "Assess whether the response is supported by the supplied source.")
        dataset_rows.append({
            "case_id": row["case_id"],
            "category": task_type,
            "split": split,
            "question": question,
            "context": [{"source_id": f"ragtruth-source-{row['source_group']}", "text": row["context"]}],
            "must_include": [], "must_not_include": [], "should_refuse": False,
            "required_citations": [],
        })
        output_rows.append({"case_id": row["case_id"], "answer": row["answer"], "source_model": row["model"]})
        annotation_rows.append({
            "case_id": row["case_id"],
            "reviewer_id": "ragtruth-official-span-annotations",
            "labels": {"groundedness": "fail" if row["has_hallucination"] else "pass"},
            "label_mapping": "fail if any official span is not implicit_true; otherwise pass",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "dataset.jsonl": dataset_rows,
        "outputs.jsonl": output_rows,
        "annotations.jsonl": annotation_rows,
    }
    for filename, records in files.items():
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    split_report = {}
    for split in ("validation", "test"):
        split_rows = [row for row, name in selected if name == split]
        split_report[split] = {
            "cases": len(split_rows),
            "source_groups": len({row["source_group"] for row in split_rows}),
            "pass": sum(not row["has_hallucination"] for row in split_rows),
            "fail": sum(row["has_hallucination"] for row in split_rows),
            "task_types": dict(sorted(Counter(row["task_type"] for row in split_rows).items())),
        }
    manifest = {
        "benchmark": "RAGTruth semantic groundedness judge calibration sample",
        "source": "ParticleMedia/RAGTruth official response and source-information files",
        "source_revision": SOURCE_REVISION,
        "source_sha256": {"response.jsonl": _sha256(response_path), "source_info.jsonl": _sha256(source_path)},
        "excluded_non_good_quality_responses": excluded_quality,
        "sample_seed": seed,
        "cases_per_class_per_split": per_class,
        "sampling": "One response per distinct source group and class, balanced pass/fail sample; source groups selected from official train for validation and untouched official test for holdout.",
        "source_group_counts": {
            "validation": len(validation_groups),
            "test": len(test_groups),
            "overlap": 0,
        },
        "splits": split_report,
        "label_mapping": "pass when no non-implicit-true hallucination span is annotated; fail when at least one such span is annotated.",
        "limitations": [
            "RAGTruth provides human-annotated hallucination spans; response-level pass/fail is a simplified corpus-derived proxy, not custom reviewer adjudication for this product.",
            "The balanced sample is not prevalence-representative; accuracy and predictive values do not estimate natural traffic performance.",
            "A source-disjoint held-out split supports corpus-level evaluation only; it does not establish production quality or a release gate.",
            "Running semantic-grade sends selected source and answer text to the configured provider and may incur provider charges; this preparation step makes no provider requests.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=50, help="Distinct source groups per label in each split")
    args = parser.parse_args()
    manifest = prepare(args.data_dir, args.output_dir, args.seed, args.per_class)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

