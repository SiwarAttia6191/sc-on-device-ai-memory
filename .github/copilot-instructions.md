# Copilot Instructions

## Project commands

This is a Python 3.12+ project managed with `uv`; dependencies and exact
versions are defined by `pyproject.toml` and `uv.lock`.

```bash
uv sync
uv run python -m unittest discover -s tests -t .
uv run python -m unittest tests.test_live
uv run python -m unittest tests.test_live.LiveAppTests.test_voice_path_can_only_be_claimed_once
uv run python -m robot.app --source testdata/
uv run python testdata/verify_scores.py
uv run python testdata/benchmark_speech.py --manifest speech-benchmark/manifest.csv
```

The replay command is the lightweight application smoke test for detection,
cropping, embeddings, and recognition. The bundled unit tests avoid loading
the large vision and speech models. No lint, formatter, type checker, or
separate build command is configured in the repository; do not introduce one
unless the task requires it.

For the live application, use:

```bash
uv run python -m robot.app
```

It serves the browser UI at `http://127.0.0.1:8765`. Use
`--camera 1` for another camera, `--host 0.0.0.0` for a LAN/phone view, and
`--data DIR` for an isolated memory shard. `--reset` deletes the selected
shard before startup, so do not use it against data that must be preserved.

## Architecture

The main dependency boundary is intentional: `robot/device` may import
`robot/brain`, but `robot/brain` must not import `robot/device`. This keeps the
memory and recognition logic testable without a camera, browser, web server,
or worker threads.

- `robot/app.py` parses CLI options, creates `Robot`, and starts either the
  live camera application or headless image/video replay.
- `robot/brain/core.py` is the central pipeline. `Robot.process_frame`
  connects stable detections, object crops, image embeddings, recognition, and
  sighting writes. `teach` and `ask` implement the two voice actions.
- `robot/brain/detect.py` runs YOLOE and BoT-SORT tracking. Detector class
  labels are deliberately ignored; names come from taught memory entries.
- `robot/brain/face_landmarks.py` optionally uses MediaPipe Face Landmarker to
  create eye and nose crops. Enable it with `uv sync --extra face`; the task
  model is cached outside the repository.
- `robot/brain/models.py` lazily loads the local CLIP, Nomic, and Whisper
  models. Avoid eager model loading in unit-test paths.
- `robot/brain/memory.py` owns the embedded Qdrant Edge shard. Taught views
  have `image` and `text` vectors; sightings and ignored items use the
  appropriate subset of those vectors.
- `robot/device/live.py` owns camera capture, rendering, detection/key/voice
  workers, browser-server startup, and synchronization around live state.
- `robot/device/server.py` and `robot/device/page.html` implement the HTTP
  routes, MJPEG stream, state polling, and browser controls.

The runtime flow is:

```text
camera -> detect/track -> stable crop -> CLIP image vector
voice -> Whisper draft -> user review -> label/question -> Nomic text vector
                                          -> Qdrant Edge memory -> recall
```

Voice actions pause after Whisper transcription for user review. The browser
allows the transcript to be edited, confirmed, or discarded before teaching
or recall uses it. Preserve that confirmation boundary when changing the voice
pipeline; do not silently send unreviewed ASR output to memory.

`LiveApp` uses a main frame/render loop plus separate detection, key, and
voice workers. Live state is protected by `LiveApp.lock`; memory access is
serialized by `Memory`. Code that needs both follows the lock order of live
state first, then memory.

## Repository conventions

- Configuration is loaded by `robot/config.py` from the repository-root
  `.env`. Precedence is source defaults, `.env`, exported environment
  variables, then CLI flags where available. Configuration values are
  camera-dependent calibration knobs, not application secrets.
- Recognition uses cosine similarity and the calibrated
  `RECOGNIZE_THRESHOLD`; it is not a percentage. Detection filtering uses
  `DETECT_CONF`, `DETECT_MIN_AREA`, and `DETECT_MAX_AREA`. Camera orientation
  and lens cropping use `CAMERA_ROTATE` and `FRAME_CROP`.
- The detector must produce stable tracks before memory is queried. Track
  state, periodic re-querying, crop quality, and ignored track IDs are part
  of the recognition behavior; do not bypass them when changing detection.
- Teaching adds another view for a label rather than replacing existing
  views. Recognition searches the taught image views, while spoken recall
  uses the taught text vectors and stored payload fields.
- Memory writes are flushed because the target device can lose power.
  Preserve this durability behavior when changing `robot/brain/memory.py`.
- Runtime state belongs in ignored/generated paths such as `edge-data/`,
  `.venv/`, downloaded model weights, and local `.env` files. Never commit
  secrets, generated shard contents, model downloads, or private recordings.
- `speech-benchmark/` is for local labelled WAV recordings and its manifest.
  It is ignored because recordings may contain private speech. The checked-in
  `testdata/speech_manifest.example.csv` is only a template; use target words
  or phrases to measure target-hit accuracy with the speech benchmark.
- Keep changes platform-portable. The code supports Windows, macOS, Linux,
  and the Jetson deployment; platform-specific behavior is isolated in
  helpers and covered by tests such as the DirectShow camera fallback and
  certificate generation.
- The repository currently documents copying `.env.example`, but if that
  file is absent in a checkout, use the defaults and variable definitions in
  `robot/config.py` as the source of truth rather than inventing settings.
- For destructive memory experiments, pass a separate `--data` directory.
  Use `testdata/verify_scores.py` for calibration changes and the image replay
  command for changes affecting detection, cropping, encoders, or recognition.
