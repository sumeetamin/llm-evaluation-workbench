import hashlib
import json
from pathlib import Path

import pytest

from evalbench.calibration import calibrate_groundedness, validate_groundedness
from evalbench.dataset import DataContractError


def _fixture(tmp_path: Path):
    cases = []
    scores = []
    labels = []
    for split, offset in (("validation", 0), ("test", 30)):
        for index in range(30):
            row = offset + index
            case_id = f"case-{row}"
            label = "pass" if row % 2 == 0 else "fail"
            cases.append({
                "case_id": case_id,
                "question": "Question",
                "context": [{"source_id": f"source-{row}", "text": "context"}],
                "split": split,
            })
            answer = f"answer for {case_id}"
            scores.append({
                "case_id": case_id,
                "groundedness_score": 0.9 if label == "pass" else 0.1,
                "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "context_sha256": hashlib.sha256(json.dumps([{"source_id": f"source-{row}", "text": "context"}], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "prompt_template_sha256": "frozen-prompt-v1",
                "judge_version": "v1",
                "judge_provider": "offline",
                "judge_model": "frozen-fixture",
            })
            labels.append({"case_id": case_id, "answer": answer, "groundedness": label})
    dataset = tmp_path / "dataset.jsonl"
    score_file = tmp_path / "scores.jsonl"
    annotation_file = tmp_path / "labels.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    score_file.write_text("".join(json.dumps(row) + "\n" for row in scores), encoding="utf-8")
    annotation_file.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    return dataset, score_file, annotation_file, cases


def test_validation_accepts_disjoint_frozen_cases_and_sources(tmp_path):
    dataset, scores, labels, _ = _fixture(tmp_path)
    calibration = calibrate_groundedness(dataset, scores, labels)
    report = tmp_path / "calibration.json"
    report.write_text(json.dumps(calibration), encoding="utf-8")

    result = validate_groundedness(dataset, scores, labels, report)

    assert result["case_overlap"] == 0
    assert result["source_overlap"] == 0
    assert result["metrics"]["balanced_accuracy"] == 1.0
    assert all("case-0" not in value for value in calibration["calibration_case_hashes"])


def test_validation_rejects_split_relabeling(tmp_path):
    dataset, scores, labels, cases = _fixture(tmp_path)
    calibration = calibrate_groundedness(dataset, scores, labels)
    report = tmp_path / "calibration.json"
    report.write_text(json.dumps(calibration), encoding="utf-8")
    modified = tmp_path / "relabeled.jsonl"
    for row in cases:
        row["split"] = "test" if row["split"] == "validation" else "validation"
    modified.write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")

    with pytest.raises(DataContractError, match="fingerprint"):
        validate_groundedness(modified, scores, labels, report)


def test_calibration_manifest_contains_no_raw_case_or_source_ids(tmp_path):
    dataset, scores, labels, _ = _fixture(tmp_path)
    report = calibrate_groundedness(dataset, scores, labels)
    rendered = json.dumps(report)

    assert "case-0" not in rendered
    assert "source-0" not in rendered
    assert len(report["calibration_case_hashes"]) == 30
    assert len(report["calibration_source_hashes"]) == 30

def test_calibration_rejects_review_packet_for_different_answer(tmp_path):
    dataset, scores, labels, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    rows[0]["answer"] = "edited after scoring"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(DataContractError, match="do not match"):
        calibrate_groundedness(dataset, scores, labels)


def test_checkpoint_writer_resumes_semantic_rows(tmp_path):
    from evalbench.cli import _prepare_checkpoint, _write_jsonl_row

    checkpoint = tmp_path / "semantic.jsonl"
    _write_jsonl_row(checkpoint, {"case_id": "case-1", "groundedness_score": 0.8})
    restored = _prepare_checkpoint(checkpoint, True, {"case-1", "case-2"})
    assert restored["case-1"]["groundedness_score"] == 0.8
    assert set(restored) == {"case-1"}
