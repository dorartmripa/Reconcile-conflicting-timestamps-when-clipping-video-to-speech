"""HTML report for internship demo walkthroughs."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from backend.models import ReconciliationResult


def generate_report(
    *,
    result: ReconciliationResult,
    video_path: Path,
    transcript_path: Path,
    transcript_text: str,
    metadata_note: str,
    stt_note: str,
    output_video: Path,
    clip_start: float,
    clip_end: float,
    output_duration: float | None,
    report_path: Path,
) -> Path:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    conflict_rows = "\n".join(_conflict_row(conflict) for conflict in result.conflicts)
    source_rows = "\n".join(_source_row(item, result) for item in result.aligned)
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in result.notes)

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Timestamp Reconciliation Report</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --warn: #fbbf24;
      --ok: #34d399;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 32px;
      line-height: 1.45;
    }}
    h1, h2 {{ margin-bottom: 8px; }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 16px 0 28px;
    }}
    .card {{
      background: var(--card);
      border-radius: 12px;
      padding: 16px;
    }}
    .stat {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border-radius: 12px;
      overflow: hidden;
      margin: 12px 0 28px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #334155;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .tag {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .stt {{ background: #064e3b; color: var(--ok); }}
    .metadata {{ background: #78350f; color: var(--warn); }}
    .blend, .ordering_repair {{ background: #1e3a5f; color: var(--accent); }}
    pre {{
      white-space: pre-wrap;
      background: #0b1220;
      padding: 12px;
      border-radius: 8px;
    }}
  </style>
</head>
<body>
  <h1>Timestamp Reconciliation Agent</h1>
  <p class="muted">Generated {generated_at}</p>

  <div class="grid">
    <div class="card"><div class="muted">Words</div><div class="stat">{len(result.aligned)}</div></div>
    <div class="card"><div class="muted">Conflicts &gt; 0.5s</div><div class="stat">{len(result.conflicts)}</div></div>
    <div class="card"><div class="muted">Estimated offset</div><div class="stat">{result.estimated_offset:+.2f}s</div></div>
    <div class="card"><div class="muted">Drift detected</div><div class="stat">{"Yes" if result.drift_detected else "No"}</div></div>
  </div>

  <h2>Inputs</h2>
  <div class="card">
    <p><strong>Video:</strong> {html.escape(str(video_path))}</p>
    <p><strong>Transcript file:</strong> {html.escape(str(transcript_path))}</p>
    <p><strong>Transcript:</strong> {html.escape(transcript_text)}</p>
    <p><strong>Metadata source:</strong> {html.escape(metadata_note)}</p>
    <p><strong>STT source:</strong> {html.escape(stt_note)}</p>
  </div>

  <h2>Reconciliation notes</h2>
  <ul>{notes}</ul>

  <h2>Conflicts</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Word</th>
        <th>Metadata</th>
        <th>STT</th>
        <th>Diff</th>
        <th>Confidence</th>
        <th>Decision</th>
        <th>Final</th>
        <th>Reason / factors</th>
      </tr>
    </thead>
    <tbody>
      {conflict_rows}
    </tbody>
  </table>

  <h2>Both timestamp sources</h2>
  <table>
    <thead>
      <tr>
        <th>Word</th>
        <th>Metadata start</th>
        <th>STT start</th>
        <th>Diff</th>
        <th>STT confidence</th>
        <th>Conflict</th>
        <th>Final start</th>
        <th>Chosen source</th>
      </tr>
    </thead>
    <tbody>
      {source_rows}
    </tbody>
  </table>

  <h2>Output video</h2>
  <div class="card">
    <p><strong>File:</strong> {html.escape(str(output_video))}</p>
    <p><strong>Clip window:</strong> {clip_start:.3f}s → {clip_end:.3f}s</p>
    <p><strong>Requested duration:</strong> {clip_end - clip_start:.3f}s</p>
    <p><strong>Actual duration:</strong> {"n/a" if output_duration is None else f"{output_duration:.3f}s"}</p>
    <p class="muted">The clip bounds come from the first and last *reconciled* word timestamps, not from either source alone.</p>
  </div>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")
    return report_path


def _conflict_row(conflict) -> str:
    confidence = (
        "n/a"
        if conflict.stt_confidence is None
        else f"{conflict.stt_confidence:.2f}"
    )
    factors = "<br>".join(
        f"{html.escape(str(key))}={html.escape(str(value))}"
        for key, value in conflict.factors.items()
    )
    css = conflict.decision.lower().replace(" ", "_")
    return f"""
      <tr>
        <td>{conflict.number}</td>
        <td>{html.escape(conflict.word)}</td>
        <td>{conflict.metadata_timestamp:.2f}s</td>
        <td>{conflict.stt_timestamp:.2f}s</td>
        <td>{conflict.difference:.2f}s</td>
        <td>{confidence}</td>
        <td><span class="tag {html.escape(css)}">{html.escape(conflict.decision)}</span></td>
        <td>{conflict.final_timestamp:.2f}s</td>
        <td>{html.escape(conflict.reason)}<br><span class="muted">{factors}</span></td>
      </tr>
    """


def _source_row(item, result: ReconciliationResult) -> str:
    final = result.final_words[item.index]
    conflict = "Yes" if item.is_conflict else "No"
    confidence = "n/a" if item.stt.confidence is None else f"{item.stt.confidence:.2f}"
    return f"""
      <tr>
        <td>{html.escape(item.word)}</td>
        <td>{item.metadata.start:.2f}s</td>
        <td>{item.stt.start:.2f}s</td>
        <td>{item.difference:.2f}s</td>
        <td>{confidence}</td>
        <td>{conflict}</td>
        <td>{final.start:.2f}s</td>
        <td>{html.escape(final.source)}</td>
      </tr>
    """
