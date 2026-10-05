"""Command-line interface for demo, evaluation, provider runs, and comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .dataset import DataContractError, load_dataset, load_outputs
from .graders import grade_dataset
from .providers import ProviderError, call_responses_endpoint
from .report import write_reports

ROOT = Path(__file__).resolve().parent.parent


def _comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base = {row["case_id"]: row["passed"] for row in baseline["results"]}
    cand = {row["case_id"]: row["passed"] for row in candidate["results"]}
    common = sorted(set(base) & set(cand))
    regressions = [case_id for case_id in common if base[case_id] and not cand[case_id]]
    improvements = [case_id for case_id in common if not base[case_id] and cand[case_id]]
    return {
        "baseline_pass_rate": baseline["summary"]["pass_rate"],
        "candidate_pass_rate": candidate["summary"]["pass_rate"],
        "delta_pp": (candidate["summary"]["pass_rate"] - baseline["summary"]["pass_rate"]) * 100,
        "regressions": regressions,
        "improvements": improvements,
        "common_cases": len(common),
        "baseline_only": sorted(set(base) - set(cand)),
        "candidate_only": sorted(set(cand) - set(base)),
    }


def _score(dataset_path: str, output_path: str) -> dict[str, Any]:
    return grade_dataset(load_dataset(dataset_path), load_outputs(output_path))


def cmd_demo(args: argparse.Namespace) -> int:
    dataset = str(ROOT / "data" / "support_qa.jsonl")
    baseline_path = str(ROOT / "fixtures" / "baseline.jsonl")
    candidate_path = str(ROOT / "fixtures" / "candidate.jsonl")
    baseline = _score(dataset, baseline_path)
    candidate = _score(dataset, candidate_path)
    comparison = _comparison(baseline, candidate)
    output = write_reports("Offline demo evaluation", candidate, args.report, comparison)
    print(f"Synthetic baseline: {baseline['summary']['passed_cases']}/{baseline['summary']['total_cases']} pass")
    print(f"Synthetic candidate: {candidate['summary']['passed_cases']}/{candidate['summary']['total_cases']} pass")
    print(f"Case-level regressions: {len(comparison['regressions'])}; improvements: {len(comparison['improvements'])}")
    print(f"HTML report: {output}")
    print(f"JSON summary: {output.with_suffix('.json')}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    evaluation = _score(args.dataset, args.outputs)
    output = write_reports(args.title, evaluation, args.report)
    summary = evaluation["summary"]
    print(f"Pass rate: {summary['passed_cases']}/{summary['total_cases']} ({summary['pass_rate']:.1%})")
    print(f"HTML report: {output}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_dataset(args.dataset)
    outputs = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
        outputs.append(call_responses_endpoint(case, endpoint=args.endpoint, model=args.model, api_key_env=args.api_key_env, timeout=args.timeout))
    output_path = Path(args.outputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
    evaluation = grade_dataset(cases, {row["case_id"]: row for row in outputs})
    report_path = write_reports(f"Evaluation run · {args.model}", evaluation, args.report)
    print(f"Pass rate: {evaluation['summary']['passed_cases']}/{evaluation['summary']['total_cases']} ({evaluation['summary']['pass_rate']:.1%})")
    print(f"Model outputs: {output_path}")
    print(f"HTML report: {report_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cases = load_dataset(args.dataset)
    baseline = grade_dataset(cases, load_outputs(args.baseline))
    candidate = grade_dataset(cases, load_outputs(args.candidate))
    comparison = _comparison(baseline, candidate)
    output = write_reports(args.title, candidate, args.report, comparison)
    print(f"Pass rate delta: {comparison['delta_pp']:+.1f} percentage points")
    print(f"Regressions: {', '.join(comparison['regressions']) or 'none'}")
    print(f"Improvements: {', '.join(comparison['improvements']) or 'none'}")
    print(f"HTML report: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalbench", description="Run transparent, application-specific LLM evaluations.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="score included synthetic fixtures without a network call")
    demo.add_argument("--report", default="reports/demo.html")
    demo.set_defaults(func=cmd_demo)

    evaluate = sub.add_parser("evaluate", help="grade saved model outputs against a dataset")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--outputs", required=True)
    evaluate.add_argument("--report", required=True)
    evaluate.add_argument("--title", default="LLM evaluation report")
    evaluate.set_defaults(func=cmd_evaluate)

    run = sub.add_parser("run", help="call a Responses-compatible endpoint and evaluate its answers")
    run.add_argument("--dataset", required=True)
    run.add_argument("--outputs", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--endpoint", default="https://api.openai.com/v1/responses")
    run.add_argument("--model", required=True)
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--timeout", type=int, default=60)
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="compare two model-output runs on the same dataset")
    compare.add_argument("--dataset", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--report", required=True)
    compare.add_argument("--title", default="Model run comparison")
    compare.set_defaults(func=cmd_compare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (DataContractError, ProviderError, OSError) as exc:
        print(f"evalbench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

