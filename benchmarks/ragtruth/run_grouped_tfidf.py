"""Offline, source-group-aware TF-IDF groundedness baseline for RAGTruth."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer


TOKEN_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
VALIDATION_SEED = 42
CV_SPLITS = 5
CANDIDATE_C = (0.1, 1.0, 10.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_parts(case: dict[str, Any]) -> tuple[str, str]:
    contexts = case.get("context")
    if not isinstance(contexts, list) or len(contexts) != 1 or not isinstance(contexts[0], dict):
        raise ValueError(f"{case.get('case_id')!r} must have one context")
    context = str(contexts[0].get("text", ""))
    answer = str(case.get("answer", ""))
    if not context.strip() or not answer.strip():
        raise ValueError(f"{case.get('case_id')!r} has empty text")
    return context, answer


def pair_documents(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"context": text_parts(row)[0], "answer": text_parts(row)[1]} for row in rows]


def join_text(rows: list[dict[str, str]]) -> list[str]:
    return [f"context {row['context']} answer {row['answer']}" for row in rows]


def pair_features(rows: list[dict[str, str]]) -> csr_matrix:
    values = []
    negations = {"no", "not", "never", "none", "neither", "without", "cannot", "can't", "didn't", "doesn't", "isn't", "wasn't", "won't"}
    for row in rows:
        context = row["context"]
        answer = row["answer"]
        context_tokens = {t.casefold() for t in TOKEN_RE.findall(context)}
        answer_list = [t.casefold() for t in TOKEN_RE.findall(answer)]
        answer_tokens = set(answer_list)
        shared = answer_tokens & context_tokens
        answer_numbers = set(NUMBER_RE.findall(answer))
        context_numbers = set(NUMBER_RE.findall(context))
        answer_negation = sum(t in negations for t in answer_list)
        context_negation = sum(t in negations for t in TOKEN_RE.findall(context.casefold()))
        values.append([
            np.log1p(len(context_tokens)), np.log1p(len(answer_list)),
            len(shared) / max(1, len(answer_tokens)),
            len(shared) / max(1, len(context_tokens)),
            len(answer_numbers), len(answer_numbers - context_numbers),
            answer_negation / max(1, len(answer_list)),
            context_negation / max(1, len(TOKEN_RE.findall(context))),
            abs(answer_negation / max(1, len(answer_list)) - context_negation / max(1, len(TOKEN_RE.findall(context)))),
        ])
    return csr_matrix(np.asarray(values, dtype=np.float64))


def make_model(c: float) -> Pipeline:
    features = FeatureUnion([
        ("text", Pipeline([
            ("join", FunctionTransformer(join_text, validate=False)),
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", stop_words="english",
                ngram_range=(1, 2), min_df=2, max_features=40_000,
                sublinear_tf=True, token_pattern=r"(?u)\b\w[\w’'-]*\b",
            )),
        ])),
        ("pair", FunctionTransformer(pair_features, validate=False)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(C=c, max_iter=2000, random_state=VALIDATION_SEED)),
    ])


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores)))
    best = max(
        ((float(balanced_accuracy_score(labels, scores >= threshold)), float(threshold)) for threshold in candidates),
        key=lambda result: (result[0], -abs(result[1] - 0.5), -result[1]),
    )
    return best[1], best[0]


def summarize(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "cases": int(len(labels)), "fail": int(labels.sum()), "pass": int(len(labels) - labels.sum()),
        "threshold": threshold, "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "fail_precision": float(precision_score(labels, predictions, zero_division=0)),
        "fail_recall": float(recall_score(labels, predictions, zero_division=0)),
        "fail_f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision_fail": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "confusion_matrix": {"true_pass": int(tn), "false_fail": int(fp), "missed_fail": int(fn), "detected_fail": int(tp)},
    }


def run(input_dir: Path, output_path: Path) -> dict[str, Any]:
    dataset_path, outputs_path, annotations_path = (input_dir / name for name in ("dataset.jsonl", "outputs.jsonl", "annotations.jsonl"))
    dataset = read_jsonl(dataset_path)
    outputs = read_jsonl(outputs_path)
    annotations = read_jsonl(annotations_path)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    cases = {str(row["case_id"]): row for row in dataset}
    answers = {str(row["case_id"]): str(row["answer"]) for row in outputs}
    labels = {str(row["case_id"]): row.get("labels", {}).get("groundedness") for row in annotations}
    if len(cases) != len(dataset) or set(cases) != set(answers) or set(cases) != set(labels):
        raise ValueError("Dataset, answers, and annotation IDs must be unique and match exactly")
    if not set(labels.values()) <= {"pass", "fail"}:
        raise ValueError("Every case must have a pass/fail RAGTruth label")
    for case_id, row in cases.items():
        row["answer"] = answers[case_id]

    validation = [row for row in dataset if row.get("split") == "validation"]
    test = [row for row in dataset if row.get("split") == "test"]
    groups = {str(row["case_id"]): str(row["context"][0]["source_id"]) for row in dataset}
    y = {key: int(value == "fail") for key, value in labels.items()}
    train_groups = {groups[str(row["case_id"])] for row in validation}
    test_groups = {groups[str(row["case_id"])] for row in test}
    if not validation or not test or train_groups & test_groups:
        raise ValueError("Expected non-empty source-disjoint validation and test splits")

    x_train = pair_documents(validation)
    y_train = np.asarray([y[str(row["case_id"])] for row in validation], dtype=int)
    groups_train = np.asarray([groups[str(row["case_id"])] for row in validation])
    folds = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=VALIDATION_SEED)
    candidates = []
    oof_by_c = {}
    for c in CANDIDATE_C:
        oof = cross_val_predict(make_model(c), x_train, y_train, cv=folds, groups=groups_train, method="predict_proba")[:, 1]
        oof_by_c[c] = oof
        threshold, balanced_accuracy = choose_threshold(oof, y_train)
        candidates.append({
            "C": c, "oof_average_precision_fail": float(average_precision_score(y_train, oof)),
            "oof_balanced_accuracy_at_selected_threshold": balanced_accuracy,
            "selected_threshold": threshold,
        })
    selected = max(candidates, key=lambda item: (item["oof_average_precision_fail"], item["oof_balanced_accuracy_at_selected_threshold"], -item["C"]))
    model = make_model(float(selected["C"])).fit(x_train, y_train)
    test_rows = pair_documents(test)
    y_test = np.asarray([y[str(row["case_id"])] for row in test], dtype=int)
    test_scores = model.predict_proba(test_rows)[:, 1]
    threshold = float(selected["selected_threshold"])
    by_task = {}
    for task in sorted({str(row.get("category", "unknown")) for row in test}):
        indices = [i for i, row in enumerate(test) if str(row.get("category", "unknown")) == task]
        if len(set(y_test[indices])) == 2:
            by_task[task] = summarize(test_scores[indices], y_test[indices], threshold)
        else:
            by_task[task] = {"cases": len(indices), "fail": int(y_test[indices].sum()), "pass": int(len(indices)-y_test[indices].sum()), "note": "Only one label class; discrimination metrics undefined."}

    source_groups = manifest.get("source_group_counts", {})
    report = {
        "benchmark": "Source-disjoint offline TF-IDF classifier with lexical support features",
        "source": manifest.get("source"),
        "source_revision": manifest.get("source_revision"),
        "source_sha256": manifest.get("source_sha256"),
        "label_mapping": "fail when any non-implicit-true hallucination span is annotated; otherwise pass",
        "sample_fingerprints": {p.name: digest(p) for p in (dataset_path, outputs_path, annotations_path)},
        "sampling": {"seed": manifest.get("sample_seed"), "cases_per_class_per_split": manifest.get("cases_per_class_per_split"), "official_source_groups": source_groups},
        "sample": {"validation_cases": len(validation), "validation_sources": len(train_groups), "test_cases": len(test), "test_sources": len(test_groups), "source_overlap": 0, "validation_fail": int(y_train.sum()), "test_fail": int(y_test.sum())},
        "selection": {"method": "five-fold StratifiedGroupKFold OOF average precision on validation; threshold selected on validation OOF balanced accuracy", "selected_C": selected["C"], "selected_threshold": threshold, "candidates": candidates},
        "validation_oof": summarize(oof_by_c[float(selected["C"])], y_train, threshold),
        "test": summarize(test_scores, y_test, threshold),
        "test_by_task": by_task,
        "model": {"type": "TF-IDF word unigrams/bigrams + answer/context lexical support, numeric discrepancy and negation-ratio features + logistic regression", "max_features": 40000, "C_candidates": list(CANDIDATE_C), "scikit_learn": sklearn.__version__, "python": platform.python_version()},
        "script_sha256": digest(Path(__file__).resolve()),
        "runtime_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Offline baseline only; no external model/provider requests were made.",
            "RAGTruth response-level labels are a simplified mapping of human span annotations, not custom product reviewer adjudication.",
            "Balanced sampling is not prevalence-representative; accuracy and predictive values do not estimate natural traffic performance.",
            "Source-disjoint holdout supports corpus-level evaluation only and does not establish production quality or a release gate.",
            "The lexical model can miss paraphrases, contradiction, multi-hop reasoning, and domain-specific grounding; test labels were not used for selection.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir, args.output), indent=2))

