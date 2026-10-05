"""Train and evaluate a reproducible local baseline on the official BANKING77 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import platform
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{SOURCE_REVISION}/banking_data"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_data(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = (data_dir / "train.csv", data_dir / "test.csv")
    for path in paths:
        if path.is_file() and path.stat().st_size:
            continue
        url = f"{SOURCE_BASE}/{path.name}"
        request = urllib.request.Request(url, headers={"User-Agent": "llm-evaluation-workbench/0.2"})
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(64 * 1024):
                    output.write(block)
            if not temporary.stat().st_size:
                raise ValueError(f"Downloaded empty file from {url}")
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return paths


def _load_csv(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text", "category"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain text and category columns")
        for row_number, row in enumerate(reader, start=2):
            text, label = row.get("text"), row.get("category")
            if not text or not label:
                raise ValueError(f"{path}:{row_number}: text and category must be non-empty")
            texts.append(text)
            labels.append(label)
    if not texts:
        raise ValueError(f"{path}: no rows found")
    return texts, labels


def _build_model() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=300_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline([("features", features), ("classifier", LinearSVC(C=1.0, max_iter=5000, random_state=0))])


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def run(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np

    train_path, test_path = _download_data(data_dir)
    x_train, y_train = _load_csv(train_path)
    x_test, y_test = _load_csv(test_path)
    labels = sorted(set(y_train))
    unknown = sorted(set(y_test) - set(labels))
    if unknown:
        raise ValueError(f"Test split contains labels missing from training data: {unknown}")

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_predictions = baseline.predict(np.zeros((len(y_test), 1))).tolist()

    model = _build_model()
    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions = model.predict(x_test).tolist()
    scores = model.decision_function(x_test)
    predict_seconds = time.perf_counter() - started
    classifier_labels = model.named_steps["classifier"].classes_.tolist()
    top3 = float(top_k_accuracy_score(y_test, scores, k=3, labels=classifier_labels))

    baseline_metrics = _metrics(y_test, baseline_predictions, labels)
    candidate_metrics = _metrics(y_test, predictions, labels)
    per_intent = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    confusion = Counter((truth, prediction) for truth, prediction in zip(y_test, predictions) if truth != prediction)
    top_confusions = [
        {"actual": actual, "predicted": predicted, "count": count}
        for (actual, predicted), count in confusion.most_common(10)
    ]

    result: dict[str, Any] = {
        "benchmark": "BANKING77 fine-grained online-banking intent classification",
        "dataset": {
            "source": "PolyAI-LDN/task-specific-datasets/banking_data",
            "source_revision": SOURCE_REVISION,
            "license": "CC BY 4.0",
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "intent_count": len(labels),
            "train_sha256": _sha256(train_path),
            "test_sha256": _sha256(test_path),
        },
        "protocol": {
            "split": "official train.csv/test.csv",
            "test_used_for_tuning": False,
            "baseline": "DummyClassifier(strategy=most_frequent)",
            "candidate": "word (1-2 gram) + character (3-5 gram) TF-IDF; LinearSVC(C=1.0, max_iter=5000)",
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "fit_seconds": round(fit_seconds, 3),
            "test_predict_seconds": round(predict_seconds, 3),
        },
        "metrics": {
            "baseline": baseline_metrics,
            "candidate": {**candidate_metrics, "top_3_accuracy": top3},
            "delta": {
                key: candidate_metrics[key] - baseline_metrics[key]
                for key in ("accuracy", "macro_f1", "weighted_f1")
            },
        },
        "per_intent": {
            label: {
                metric: float(per_intent[label][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for label in labels
        },
        "top_confusions": top_confusions,
        "attribution": {
            "dataset_paper": "Casanueva et al., 2020, Efficient Intent Detection with Dual Sentence Encoders, arXiv:2003.04807",
            "dataset_license": "CC BY 4.0; attribute PolyAI and cite the dataset paper when reusing BANKING77.",
            "scope": "Real labeled public benchmark utterances; not live bank traffic and not answer-generation evaluation.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(result), encoding="utf-8")
    return result


def render_html(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    metric_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{values[key]:.3%}</td></tr>"
        for name, values in (("Most-frequent baseline", metrics["baseline"]), ("TF-IDF + LinearSVC", metrics["candidate"]))
        for key in ("accuracy", "macro_f1", "weighted_f1")
    )
    metric_rows += f"<tr><th>TF-IDF + LinearSVC · Top-3 accuracy</th><td>{metrics['candidate']['top_3_accuracy']:.3%}</td></tr>"
    intent_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{values['support']:.0f}</td><td>{values['precision']:.3f}</td><td>{values['recall']:.3f}</td><td>{values['f1-score']:.3f}</td></tr>"
        for label, values in result["per_intent"].items()
    )
    confusion_rows = "".join(
        f"<tr><td>{html.escape(row['actual'])}</td><td>{html.escape(row['predicted'])}</td><td>{row['count']}</td></tr>"
        for row in result["top_confusions"]
    )
    dataset, protocol = result["dataset"], result["protocol"]
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BANKING77 baseline evaluation</title>
<style>:root{{--bg:#0b1117;--panel:#111c24;--line:#263741;--text:#edf4f3;--muted:#a4b5bb;--lime:#c6f36b;--teal:#67d4c0}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 80% 0,#16313a,transparent 34%),var(--bg);color:var(--text);font:15px system-ui,sans-serif}}main{{max-width:1040px;margin:46px auto;padding:0 22px 50px}}.eyebrow{{font:11px monospace;letter-spacing:.12em;color:var(--teal);text-transform:uppercase}}h1{{font-size:clamp(32px,5vw,52px);letter-spacing:-.05em;margin:10px 0}}h2{{font-size:18px;margin:0 0 14px}}p{{line-height:1.6;color:var(--muted)}}section{{margin-top:18px;padding:20px;background:var(--panel);border:1px solid var(--line);border-radius:12px}}table{{border-collapse:collapse;width:100%;text-align:left}}td,th{{padding:10px;border-bottom:1px solid var(--line);font-size:13px}}th{{color:var(--muted);font-weight:600}}code{{color:var(--lime)}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}@media(max-width:680px){{.grid{{grid-template-columns:1fr}}.scroll{{overflow:auto}}}}</style><main>
<p class="eyebrow">REAL DATASET · LOCAL CLASSIFIER · HELD-OUT TEST</p><h1>BANKING77 intent classification</h1>
<p>Local TF-IDF + linear SVM benchmark over real public online-banking queries. This is an intent-classification evaluation, distinct from the repository's synthetic answer-quality demo.</p>
<div class="grid"><section><h2>Evaluation protocol</h2><p><code>{dataset['train_rows']:,}</code> train · <code>{dataset['test_rows']:,}</code> official test · <code>{dataset['intent_count']}</code> intents.</p><p>Fit on training data only; no API calls. Test split not used for tuning. Fit: {protocol['fit_seconds']:.2f}s; predict all test rows: {protocol['test_predict_seconds']:.2f}s.</p></section>
<section><h2>Model</h2><p>Word 1–2 gram and character 3–5 gram TF-IDF features combined with <code>LinearSVC(C=1.0)</code>. Most-frequent intent is the baseline.</p><p>Python {html.escape(protocol['python'])} · scikit-learn {html.escape(protocol['scikit_learn'])}</p></section></div>
<section><h2>Test-set metrics</h2><div class="scroll"><table><thead><tr><th>Model / metric</th><th>Score</th></tr></thead><tbody>{metric_rows}</tbody></table></div></section>
<section><h2>Most common error pairs</h2><div class="scroll"><table><thead><tr><th>Actual intent</th><th>Predicted intent</th><th>Count</th></tr></thead><tbody>{confusion_rows}</tbody></table></div></section>
<section><h2>Per-intent results</h2><div class="scroll"><table><thead><tr><th>Intent</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{intent_rows}</tbody></table></div></section>
<section><h2>Data and limitations</h2><p>Source revision: <code>{SOURCE_REVISION}</code> · Dataset license: CC BY 4.0 · data paper: <a href="https://arxiv.org/abs/2003.04807">Casanueva et al. (2020)</a>.</p><p>This is one untuned baseline on one English closed-set benchmark. It does not measure live bank traffic, out-of-domain handling, answer quality, financial-advice safety, retrieval grounding, or model performance in production. Inspect per-intent metrics and errors before making claims.</p></section>
</main></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.data_dir, args.output_dir)
    metric = result["metrics"]["candidate"]
    print(f"BANKING77: {result['dataset']['train_rows']}/{result['dataset']['test_rows']} train/test; {result['dataset']['intent_count']} intents")
    print(f"Test accuracy: {metric['accuracy']:.3%}; macro-F1: {metric['macro_f1']:.3%}; weighted-F1: {metric['weighted_f1']:.3%}; top-3: {metric['top_3_accuracy']:.3%}")
    print(f"Baseline accuracy: {result['metrics']['baseline']['accuracy']:.3%}")
    print(f"HTML report: {args.output_dir / 'report.html'}")
    print(f"JSON report: {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

