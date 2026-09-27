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

In PowerShell, redirect standard error to a private local log file:

```powershell
uv run python -m robot.app --data "$env:LOCALAPPDATA\qdrant-edge-memory" 2> "$env:TEMP\robot-metrics.jsonl"
```

The log contains diagnostics and timing data, not audio or transcript content.
Review it before sharing. Avoid `2>&1` if you want metrics separated from the
application's regular console output.

## Generate a Dashboard

The report reads UTF-8 or PowerShell UTF-16 logs and rejoins JSON records that
PowerShell wrapped across console-width lines. It ignores non-JSON warnings
instead of copying their text into the report.

```powershell
uv run python testdata/metrics_dashboard.py `
	--input "$env:TEMP\robot-metrics.jsonl" `
	--output "$env:TEMP\robot-metrics-dashboard.html"
explorer "$env:TEMP\robot-metrics-dashboard.html"
```

The HTML is self-contained and generated locally. It shows stage sample counts,
median/p95/maximum latency, median process CPU delta, and event counts. Current
events do not contain timestamps, so the dashboard compares distributions and
does not invent a time-series chart.

## Next Measurement Step

This first increment measures stage latency and process CPU time. A follow-up
can sample process RSS and report p50/p95 latency by stage over a session. RSS
should be sampled periodically rather than queried on every camera frame, so
measurement overhead stays low. Compare runs on the same device, camera, model
cache state, and test inputs.