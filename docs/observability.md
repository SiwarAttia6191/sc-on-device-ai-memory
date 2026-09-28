# Pipeline Observability

The application writes JSON Lines observability events to standard error. Each
record is independent JSON, so it can be redirected, searched, or ingested by a
log tool without changing the memory database.

## Events Available

Stage timing records use this shape:

```json
{"event":"ai_stage","stage":"transcription","outcome":"ok","duration_ms":612.3,"cpu_ms":184.2,"model":"whisper","language":"en"}
```

Instrumented stages include:

- `object_detection`: YOLOE detection and tracking.
- `face_landmarks`: MediaPipe face feature detection when enabled.
- `image_embedding`: CLIP image crop embedding.
- `text_embedding` and `query_embedding`: Nomic text encoding.
- `transcription`: local Whisper inference.
- `image_retrieval` and `text_retrieval`: Qdrant Edge search operations.

`duration_ms` is wall-clock latency. `cpu_ms` is process CPU time consumed
during the stage. Since the robot has concurrent workers, process CPU deltas
are approximate per-stage measurements and may include work from another
thread running at the same time. They are useful for trends, not strict
per-thread accounting.

Outcome events include `face_found`, `no_face_found`, `audio_rejected`,
`transcript_discarded`, and `voice_action_confirmed`. Face-presence events are
emitted only when detection state changes, to avoid a log line for every
camera frame.

## Privacy

Metrics deliberately do not include audio, transcripts, image pixels, face
landmarks, or object labels. Do not add raw model inputs or outputs to these
events. The application may still store taught thumbnails and user-confirmed
transcripts in its local Qdrant shard as part of its intended memory behavior.

## Capture a Session

In PowerShell, create a durable, private reports directory and redirect
standard error to a local log file there:

```powershell
$reports = Join-Path $env:LOCALAPPDATA 'qdrant-edge-memory-reports'
New-Item -ItemType Directory -Path $reports -Force | Out-Null
uv run python -m robot.app --data "$env:LOCALAPPDATA\qdrant-edge-memory" 2> "$reports\robot-metrics.jsonl"
```

The log contains diagnostics and timing data, not audio or transcript content.
Review it before sharing. Avoid `2>&1` if you want metrics separated from the
application's regular console output.

## Generate a Dashboard

The report reads UTF-8 or PowerShell UTF-16 logs and rejoins JSON records that
PowerShell wrapped across console-width lines. It ignores non-JSON warnings
instead of copying their text into the report.

```powershell
$reports = Join-Path $env:LOCALAPPDATA 'qdrant-edge-memory-reports'
uv run python testdata/metrics_dashboard.py `
	--input "$reports\robot-metrics.jsonl" `
	--output "$reports\robot-metrics-dashboard.html"
```

The HTML is self-contained and generated locally. It shows stage sample counts,
median/p95/maximum latency, median process CPU delta, and event counts. Current
events do not contain timestamps, so the dashboard compares distributions and
does not invent a time-series chart.

### Open the dashboard at a local URL

The report file persists in `%LOCALAPPDATA%`, but an HTTP URL is available only
while a local web server is running. Start this command in a separate PowerShell
terminal and leave that terminal open:

```powershell
$reports = Join-Path $env:LOCALAPPDATA 'qdrant-edge-memory-reports'
Set-Location $reports
python -m http.server 8769 --bind 127.0.0.1
```

Then visit:

```text
http://127.0.0.1:8769/robot-metrics-dashboard.html
```

If port `8769` is already in use, choose another port in both the command and
URL. If the server stops or Windows restarts, the report remains on disk; run
the server command again to restore the URL. The dashboard uses aggregate
metrics only and cannot identify a specific saved memory. Browse actual
objects, taught views, and sightings in the robot app's **MEMORY** tab, using
the same `--data` shard path.

### What a "sample" means

A stage sample is **one timed invocation of a software stage**, not a training
example, detected object, photo, or Qdrant memory record. For example, one
camera frame can generate one YOLOE call and one face-landmark call; later, a
stable crop can generate an image-embedding call and a memory-search call.
The same object may therefore contribute many stage calls while it remains in
view. A retrieval call is a search and does not itself save a memory.

The dashboard explains each stage beside its count. Read the metrics according
to the stage: detection/landmark calls are frame processing; embedding calls
are vector creation; retrieval calls are searches. `face_found` and
`no_face_found` count changes between found/not-found states, not the total
number of faces in all frames.

### Browse saved memories

The dashboard does not contain a memory-browser export. To see saved objects,
taught views, and later sightings, start the robot with the same shard path
used when those memories were created, then choose **MEMORY** in the app:

```powershell
uv run python -m robot.app --data "$env:LOCALAPPDATA\qdrant-edge-memory"
```

Metrics intentionally omit object labels and memory IDs for privacy. As a
result, you cannot match an individual timing sample to one specific image or
Qdrant record. Compare aggregate stage-call totals with the object, taught-view,
and sighting counts shown by the app, but do not expect those numbers to equal
each other.

## Next Measurement Step

This first increment measures stage latency and process CPU time. A follow-up
can sample process RSS and report p50/p95 latency by stage over a session. RSS
should be sampled periodically rather than queried on every camera frame, so
measurement overhead stays low. Compare runs on the same device, camera, model
cache state, and test inputs.