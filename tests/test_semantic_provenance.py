import hashlib
import json

from evalbench import semantic


def test_semantic_score_uses_returned_model_and_hashes_exact_answer(monkeypatch):
    monkeypatch.setattr(
        semantic,
        "call_provider",
        lambda *args, **kwargs: {
            "answer": '{"score": 0.9, "unsupported_claim_count": 0, "rationale_code": "supported"}',
            "model": "provider-resolved-model-2026-11",
            "latency_ms": 12.5,
        },
    )
    case = {
        "case_id": "fixture-1",
        "question": "Question?",
        "context": [{"source_id": "source-1", "text": "Fact."}],
    }
    output = {"answer": "The fact."}

    result = semantic.grade_groundedness(
        case, output, provider="openai-responses", model="requested-alias",
    )

    assert result["judge_model"] == "provider-resolved-model-2026-11"
    assert result["answer_sha256"] == hashlib.sha256(b"The fact.").hexdigest()
    assert result["context_sha256"] == hashlib.sha256(
        json.dumps(case["context"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert result["prompt_template_sha256"]
