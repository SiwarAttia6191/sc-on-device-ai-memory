"""Build a private, self-contained HTML report from AI pipeline JSONL logs.

Run from the repository root:

    uv run python testdata/metrics_dashboard.py \
      --input "$env:TEMP/robot-metrics.jsonl" \
      --output "$env:TEMP/robot-metrics-dashboard.html"
"""
import argparse
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STAGE_HELP = {
    "object_detection": "One YOLOE detection and tracking pass over a camera frame. It does not mean an object was saved.",
    "face_landmarks": "One MediaPipe face-landmark pass. Face-found events are state changes, not counts of faces or memories.",
    "image_embedding": "One image crop converted to a CLIP vector. A crop can be embedded repeatedly; this is not a taught-view count.",
    "text_embedding": "One text description converted to a Nomic vector, usually during teaching.",
    "query_embedding": "One spoken question converted to a Nomic query vector.",
    "transcription": "One recording transcribed by Whisper. Audio and transcript text are not included in metrics.",
    "image_retrieval": "One image-vector search against Qdrant memory. A search does not create a memory.",
    "text_retrieval": "One text-vector search against Qdrant memory. A search does not create a memory.",
}


def percentile(values, fraction):
    """Linearly interpolated percentile for a non-empty list of numbers."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def analyze_lines(lines):
    """Aggregate JSON events, including records wrapped by PowerShell."""
    stages = defaultdict(lambda: {"duration": [], "cpu": [], "outcomes": Counter()})
    events = Counter()
    ignored_lines = 0
    parsed_lines = 0

    def accept(record):
        nonlocal ignored_lines, parsed_lines
        if not isinstance(record, dict) or not isinstance(record.get("event"), str):
            ignored_lines += 1
            return
        event = record["event"]
        if event == "ai_stage":
            stage = record.get("stage")
            duration = record.get("duration_ms")
            if (not isinstance(stage, str) or not stage
                    or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration < 0):
                ignored_lines += 1
                return
            bucket = stages[stage]
            bucket["duration"].append(float(duration))
            cpu = record.get("cpu_ms")
            if isinstance(cpu, (int, float)) and math.isfinite(cpu) and cpu >= 0:
                bucket["cpu"].append(float(cpu))
            bucket["outcomes"][str(record.get("outcome", "unknown"))] += 1
        events[event] += 1
        parsed_lines += 1

    pending = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if pending:
            if line.startswith("{"):
                ignored_lines += 1
                pending = ""
            else:
                pending += line
                try:
                    record = json.loads(pending)
                except json.JSONDecodeError:
                    continue
                pending = ""
                accept(record)
                continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            if line.startswith("{"):
                pending = line
            else:
                ignored_lines += 1
            continue
        accept(record)
    if pending:
        ignored_lines += 1

    summaries = []
    for name, bucket in stages.items():
        durations = bucket["duration"]
        cpus = bucket["cpu"]
        summaries.append({
            "stage": name,
            "count": len(durations),
            "median_ms": statistics.median(durations),
            "p95_ms": percentile(durations, 0.95),
            "max_ms": max(durations),
            "cpu_median_ms": statistics.median(cpus) if cpus else None,
            "outcomes": dict(bucket["outcomes"]),
        })
    summaries.sort(key=lambda row: row["p95_ms"], reverse=True)
    return {
        "stages": summaries,
        "events": dict(events),
        "stage_samples": sum(row["count"] for row in summaries),
        "parsed_lines": parsed_lines,
        "ignored_lines": ignored_lines,
    }


def _ms(value):
    return f"{value:,.2f} ms"


def render_dashboard(summary, source_name):
    """Render aggregate-only findings as a self-contained HTML dashboard."""
    stage_rows = summary["stages"]
    largest_p95 = max((row["p95_ms"] for row in stage_rows), default=1.0) or 1.0
    largest_median = max((row["median_ms"] for row in stage_rows), default=1.0) or 1.0
    rows = []
    for row in stage_rows:
        median_width = max(1.0, row["median_ms"] / largest_p95 * 100)
        p95_width = max(1.0, row["p95_ms"] / largest_p95 * 100)
        cpu = "n/a" if row["cpu_median_ms"] is None else _ms(row["cpu_median_ms"])
        description = STAGE_HELP.get(
                        row["stage"], "One timed invocation of this pipeline stage.")
        rows.append(f"""
          <tr>
                        <th scope="row">{html.escape(row['stage'])}<small>{row['count']:,} calls · {html.escape(description)}</small></th>
            <td>{_ms(row['median_ms'])}<div class="bar"><i class="median" style="width:{median_width:.2f}%"></i></div></td>
            <td>{_ms(row['p95_ms'])}<div class="bar"><i class="p95" style="width:{p95_width:.2f}%"></i></div></td>
            <td>{_ms(row['max_ms'])}</td>
            <td>{cpu}</td>
          </tr>""")
    if not rows:
        rows.append('<tr><td colspan="5" class="empty">No valid stage timing records found.</td></tr>')

    event_cards = []
    for name, count in sorted(summary["events"].items(), key=lambda item: (-item[1], item[0])):
        if name == "ai_stage":
            continue
        event_cards.append(
            f'<div class="event"><span>{html.escape(name.replace("_", " "))}</span>'
            f'<strong>{count:,}</strong></div>')
    if not event_cards:
        event_cards.append('<div class="empty">No outcome events found.</div>')

    slowest = stage_rows[0] if stage_rows else None
    finding = (f"Highest p95 latency: <b>{html.escape(slowest['stage'])}</b> "
               f"at {_ms(slowest['p95_ms'])}." if slowest else
               "No stage latency findings yet.")
    source = html.escape(str(source_name))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>On-device AI pipeline report</title>
<style>
:root {{ color-scheme: light; --ink:#17252c; --muted:#63737b; --line:#d9e2e4; --paper:#f4f7f5; --white:#fff; --green:#087f68; --mint:#c8efe1; --gold:#db9e2e; --red:#b84442; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.45 "Segoe UI", sans-serif }}
main {{ max-width:1180px; margin:auto; padding:34px 24px 64px }} header {{ display:flex; justify-content:space-between; align-items:end; gap:24px; border-bottom:1px solid var(--line); padding-bottom:22px }}
.eyebrow {{ color:var(--green); text-transform:uppercase; font-size:12px; font-weight:700; letter-spacing:.09em }} h1 {{ margin:6px 0 0; font-size:clamp(26px,4vw,38px); line-height:1.1 }} .source {{ max-width:42%; text-align:right; color:var(--muted); overflow-wrap:anywhere; font-size:12px }}
.cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:22px 0 }} .card, section {{ background:var(--white); border:1px solid var(--line); border-radius:7px }} .card {{ padding:16px }} .card span {{ display:block; color:var(--muted); font-size:12px }} .card strong {{ display:block; margin-top:6px; font-size:25px; font-variant-numeric:tabular-nums }}
section {{ margin-top:18px; padding:20px; overflow:hidden }} h2 {{ margin:0 0 4px; font-size:19px }} .sub {{ margin:0 0 16px; color:var(--muted); font-size:13px }} .finding {{ background:#e8f5ef; border-left:4px solid var(--green); padding:12px 14px; margin:12px 0 18px }}
.table-wrap {{ overflow-x:auto }} table {{ width:100%; border-collapse:collapse; min-width:720px; font-variant-numeric:tabular-nums }} th,td {{ padding:12px 10px; border-top:1px solid var(--line); text-align:left; vertical-align:middle }} thead th {{ color:var(--muted); font-size:12px }} tbody th {{ font-size:14px }} small {{ display:block; color:var(--muted); font-weight:400 }} .bar {{ margin-top:5px; width:min(240px,100%); height:5px; background:#edf1ef; border-radius:5px; overflow:hidden }} .bar i {{ display:block; height:100%; border-radius:5px }} .median {{ background:var(--green) }} .p95 {{ background:var(--gold) }}
.legend {{ display:flex; gap:16px; color:var(--muted); font-size:12px; margin:12px 0 }} .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px }} .dot.m {{ background:var(--green) }} .dot.p {{ background:var(--gold) }}
.events {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:8px }} .event {{ display:flex; justify-content:space-between; gap:12px; border:1px solid var(--line); border-radius:5px; padding:11px 12px }} .event span {{ text-transform:capitalize; color:var(--muted) }} .event strong {{ font-variant-numeric:tabular-nums }} .note {{ color:var(--muted); font-size:12px; margin:12px 0 0 }} .empty {{ color:var(--muted); padding:16px }}
@media(max-width:740px) {{ main {{ padding:22px 14px 40px }} header {{ align-items:start; flex-direction:column }} .source {{ max-width:none; text-align:left }} .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)) }} section {{ padding:15px }} }}
</style>
</head>
<body><main>
<header><div><div class="eyebrow">Local observability · aggregate report</div><h1>AI pipeline performance</h1></div><div class="source">Source file<br><b>{source}</b></div></header>
<div class="cards">
    <div class="card"><span>Timed stage calls</span><strong>{summary['stage_samples']:,}</strong></div>
  <div class="card"><span>Stages observed</span><strong>{len(stage_rows):,}</strong></div>
  <div class="card"><span>Outcome events</span><strong>{sum(v for k,v in summary['events'].items() if k != 'ai_stage'):,}</strong></div>
  <div class="card"><span>Non-JSON lines skipped</span><strong>{summary['ignored_lines']:,}</strong></div>
</div>
<section><h2>Stage latency and CPU</h2><p class="sub">Stages ordered by p95 wall-clock latency. CPU values are process-level deltas, approximate when worker threads overlap.</p>
<div class="finding">{finding}</div><div class="legend"><span><i class="dot m"></i>Median latency</span><span><i class="dot p"></i>p95 latency</span></div>
<div class="table-wrap"><table><thead><tr><th>Stage</th><th>Median</th><th>p95</th><th>Max</th><th>Median CPU delta</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">Bars share one scale based on the slowest p95. This log has no event timestamps, so this report compares distributions, not time-of-day trends. First-use model loading can make early embedding samples much slower.</p></section>
<section><h2>Outcome events</h2><p class="sub">Counts from content-free event records.</p><div class="events">{''.join(event_cards)}</div></section>
    <section><h2>Find saved memories</h2>
    <p>A timed stage call is not an object, photo, or database record. Detection, landmarks, embeddings, and retrieval may run many times while the camera watches the same object. Retrieval measures a search; it does not save anything by itself.</p>
    <p>To browse saved objects, taught views, and sightings, open <a href="http://127.0.0.1:8765/">the robot app</a> and choose <b>MEMORY</b>. The app must be running with the same <code>--data</code> directory as the shard you want to inspect.</p>
    <p class="note">Metrics intentionally omit object labels and memory IDs for privacy, so this dashboard cannot link one timing call to a specific saved image. Compare aggregate calls with the separate object/view/sighting counts in MEMORY; they are different measures and are not expected to match.</p></section>
<p class="note">Generated locally from JSONL. Non-JSON startup diagnostics were ignored and their text was not copied into this report.</p>
</main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="observability stderr log (JSONL plus optional diagnostics)")
    parser.add_argument("--output", required=True, type=Path,
                        help="destination HTML dashboard")
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"input log does not exist: {args.input}")
    raw = args.input.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    summary = analyze_lines(raw.decode(encoding, errors="replace").splitlines())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_dashboard(summary, args.input), encoding="utf-8")
    print(f"Dashboard written to: {args.output}")
    print(f"Parsed {summary['stage_samples']:,} stage samples; "
          f"ignored {summary['ignored_lines']:,} non-JSON/invalid lines.")


if __name__ == "__main__":
    main()