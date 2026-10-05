"""Summaries and self-contained HTML reports with escaped case content."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_html(title: str, evaluation: dict[str, Any], comparison: dict[str, Any] | None = None) -> str:
    summary = evaluation["summary"]
    rate_rows = "".join(
        f"<tr><td>{html.escape(name.replace('_', ' ').title())}</td><td>{_percent(rate)}</td><td><div class='bar'><i style='width:{rate * 100:.1f}%'></i></div></td></tr>"
        for name, rate in summary["check_rates"].items()
    )
    result_rows = []
    for result in evaluation["results"]:
        failures = [name.replace("_", " ") for name, check in result["checks"].items() if not check["pass"]]
        status = "PASS" if result["passed"] else "FAIL"
        metadata = result["metadata"]
        answer = html.escape(result["answer"])
        result_rows.append(
            "<tr data-status='{}'><td><code>{}</code></td><td>{}</td><td><span class='{}'>{}</span></td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                status.lower(), html.escape(result["case_id"]), html.escape(result["category"]),
                "pill pass" if result["passed"] else "pill fail", status,
                html.escape(", ".join(failures) or "—"),
                html.escape(str(metadata.get("latency_ms", "—"))), answer,
            )
        )
    comparison_html = ""
    if comparison:
        comparison_html = (
            "<section class='panel'><h2>Run comparison</h2>"
            f"<p>Baseline {_percent(comparison['baseline_pass_rate'])} → Candidate {_percent(comparison['candidate_pass_rate'])} "
            f"(<b>{comparison['delta_pp']:+.1f} pp</b>); regressions: <b>{len(comparison['regressions'])}</b>, "
            f"improvements: <b>{len(comparison['improvements'])}</b>.</p>"
            f"<p class='muted'>Regressed cases: {html.escape(', '.join(comparison['regressions']) or 'none')}. "
            f"Improved cases: {html.escape(', '.join(comparison['improvements']) or 'none')}.</p></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--bg:#0b1117;--panel:#111c24;--line:#263741;--text:#edf4f3;--muted:#9eb0b5;--lime:#c6f36b;--teal:#67d4c0;--red:#ff938a}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 80% 0,#16313a,transparent 34%),var(--bg);color:var(--text);font:14px system-ui,sans-serif}}
main{{max-width:1160px;margin:44px auto;padding:0 22px 50px}}.eyebrow{{font:11px monospace;letter-spacing:.12em;color:var(--teal);text-transform:uppercase}}h1{{font-size:clamp(32px,5vw,52px);letter-spacing:-.05em;margin:10px 0}}h2{{font-size:16px;margin:0 0 16px}}.muted{{color:var(--muted);line-height:1.6}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0 16px}}.stat,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}.label{{font:10px monospace;letter-spacing:.07em;color:var(--muted);text-transform:uppercase}}.value{{font-size:26px;font-weight:700;margin:7px 0}}.panel{{margin-top:14px;overflow:hidden}}table{{width:100%;border-collapse:collapse;text-align:left}}th{{font:10px monospace;text-transform:uppercase;color:var(--muted);letter-spacing:.07em}}td,th{{padding:12px 9px;border-bottom:1px solid var(--line);vertical-align:top}}td{{font-size:12px;line-height:1.55}}.bar{{height:7px;width:130px;background:#22323b;border-radius:9px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--lime)}}.pill{{font:10px monospace;padding:5px 8px;border-radius:20px}}.pass{{color:var(--lime);background:#243321}}.fail{{color:#ffb8b1;background:#3d2425}}.answer{{max-width:410px;white-space:pre-wrap;overflow-wrap:anywhere;color:#cfdbdc}}.filters{{display:flex;gap:8px;margin-bottom:10px}}button{{background:#20313a;color:var(--text);border:1px solid var(--line);padding:8px 11px;border-radius:6px;cursor:pointer}}button.active{{border-color:var(--teal);color:var(--teal)}}.footer{{margin-top:20px;font:10px monospace;color:#819399}}@media(max-width:760px){{.stats{{grid-template-columns:repeat(2,1fr)}}.wide{{overflow:auto}}}}
</style></head><body><main>
<p class="eyebrow">APPLICATION EVALUATION · SYNTHETIC DATA</p><h1>{html.escape(title)}</h1>
<p class="muted">Deterministic checks are signals for human review, not a complete measure of answer quality. Inspect each failure before making a release decision.</p>
<section class="stats"><div class="stat"><div class="label">Cases passed</div><div class="value">{summary['passed_cases']} / {summary['total_cases']}</div></div>
<div class="stat"><div class="label">Pass rate</div><div class="value">{_percent(summary['pass_rate'])}</div></div>
<div class="stat"><div class="label">Mean latency</div><div class="value">{summary.get('mean_latency_ms', 0):.0f} ms</div></div>
<div class="stat"><div class="label">Mean output tokens</div><div class="value">{summary.get('mean_output_tokens', 0):.0f}</div></div></section>
{comparison_html}
<section class="panel"><h2>Check performance</h2><div class="wide"><table><thead><tr><th>Dimension</th><th>Rate</th><th>At a glance</th></tr></thead><tbody>{rate_rows}</tbody></table></div></section>
<section class="panel"><h2>Case review</h2><div class="filters"><button class="active" data-filter="all">All</button><button data-filter="fail">Failures</button><button data-filter="pass">Passed</button></div>
<div class="wide"><table><thead><tr><th>Case</th><th>Category</th><th>Status</th><th>Failed checks</th><th>Latency ms</th><th>Answer</th></tr></thead><tbody>{''.join(result_rows)}</tbody></table></div></section>
<p class="footer">EvalBench report · Review dataset provenance and grader limitations before release decisions.</p></main>
<script>document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('tbody tr[data-status]').forEach(r=>r.hidden=b.dataset.filter!=='all'&&r.dataset.status!==b.dataset.filter)}}));</script>
</body></html>"""


def write_reports(title: str, evaluation: dict[str, Any], html_path: str | Path, comparison: dict[str, Any] | None = None) -> Path:
    """Write HTML beside a machine-readable JSON summary."""
    destination = Path(html_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_html(title, evaluation, comparison), encoding="utf-8")
    json_path = destination.with_suffix(".json")
    json_path.write_text(json.dumps({"evaluation": evaluation, "comparison": comparison}, indent=2), encoding="utf-8")
    return destination

