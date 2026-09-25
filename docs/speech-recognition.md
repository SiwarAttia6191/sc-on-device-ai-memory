# Understand Speech Recognition

This guide explains how the robot turns spoken words into text and then uses
that text to teach an object or answer a memory question. It is written as a
code-reading guide for beginners who are learning how local AI systems fit
together.

## The Big Picture

The speech pipeline is:

```text
microphone or phone browser
  -> audio samples
  -> WAV file
  -> silence check and trimming
  -> local Whisper speech-to-text model
  -> transcript
  -> teach an object or search memory
```

Speech recognition here means **automatic speech recognition (ASR)**: the
computer hears audio and produces text. Whisper is not a conversational LLM
and does not decide what to say. It only performs the speech-to-text step.
Later, the application turns the transcript into a vector for semantic search.

## Phase 1: Choose the Audio Source

The robot supports two audio paths:

- A microphone connected to the computer running the robot.
- A phone browser that records audio and uploads a WAV file to the robot.

The local microphone path starts in `_voice_action()` in
`robot/device/live.py`. The phone path starts in `_phone_audio()` in the same
file. Both paths eventually call `_process()`, so transcription and memory
behavior stay the same after the WAV file is available.

This separation is useful: the application can change where audio comes from
without changing the speech-recognition model.

## Phase 2: Record Microphone Audio

File: `robot/device/mic.py`

The `record_wav()` function captures audio with the `sounddevice` package.
It does the following:

1. Reads `MIC_DEVICE` from the application configuration if a specific input
   device was selected.
2. Opens a one-channel input stream.
3. Reads audio in 100-millisecond blocks.
4. Watches the loudness of each block.
5. Stops after speech is detected and then quiet continues for a short time,
   or after the maximum recording time.
6. Saves the result as a 16 kHz, mono, 16-bit WAV file.

A WAV file is a convenient boundary between recording and machine learning:
the recording code can finish before the model starts inference.

### Why the loudness check matters

For each audio block, the code calculates RMS, or root mean square, level:

```text
RMS = sqrt(mean(sample_value^2))
```

You do not need to calculate this by hand. Conceptually, RMS is a simple
measure of how strong the audio signal is. If the value is zero, the program
received silence or no samples. Common causes are microphone permissions, a
muted device, or the wrong input device.

The threshold `SPEECH_RMS = 200` is a practical signal threshold, not an AI
confidence score. It controls when the recorder considers speech to have
started.

## Phase 3: Check and Prepare the WAV

File: `robot/device/live.py`, method `_process()`

After recording, `_process()` calls these helpers from `robot/device/mic.py`:

- `wav_rms()` measures the complete recording and prints the level.
- `is_silent()` rejects audio below the configured silence floor.
- `trim_to_speech()` removes quiet audio from the beginning and end.

Trimming helps because Whisper has less irrelevant audio to process. It can
also reduce incorrect transcriptions caused by long silence. The application
still keeps the same WAV path and updates its contents in place.

If the log says:

```text
recorded level (rms): 0
```

no useful audio reached the robot. Fix the operating-system microphone
permission or choose the correct input device before debugging Whisper.

## Phase 4: Load the Local Whisper Model

File: `robot/brain/models.py`

The model configuration is:

```python
WHISPER_MODEL = "whisper-base"
```

The `_asr_model()` function calls `onnx_asr.load_model()` and selects the CPU
execution provider. ONNX Runtime executes the model locally; the recording is
not sent to a cloud speech service.

The first run downloads the model files. The download can be several hundred
megabytes, and the files are cached under the FastEmbed/model cache locations.
Later runs reuse the cache.

The `warm_encoders()` function loads the speech model before a voice action.
This makes the first button press less surprising because model loading is
started in advance. A lock prevents two threads from loading the same model at
once.

## Phase 5: Convert Audio into Text

Still in `robot/brain/models.py`, `transcribe(wav_path)` calls:

```python
_asr_model().recognize(wav_path, language="en")
```

The returned string is stripped of extra whitespace and becomes the
transcript. For example:

```text
Audio: "This is my blue mug"
Transcript: "This is my blue mug"
```

`LANGUAGE = "en"` tells Whisper to expect English. Setting it to `None` would
allow language detection, but automatic detection may require an extra
inference pass.

## Phase 6: Connect Speech to the User Action

File: `robot/device/live.py`, method `_process()`

The flow after transcription depends on the action:

```text
TEACH button
  -> record speech
  -> transcribe speech
  -> robot.teach(crop, transcript)

ASK button
  -> record speech
  -> transcribe speech
  -> robot.ask(transcript)
```

For teaching, the transcript describes the object currently visible to the
camera. For example, `"This is my blue mug"` is sent to `Robot.teach()`.

For asking, the transcript is a natural-language question such as
`"When did you last see my mug?"`. It is sent to `Robot.ask()`.

The speech model does not itself understand the robot's memory. It only
produces the text that the memory layer can process.

## Phase 7: Use the Transcript in Memory

File: `robot/brain/core.py`

### Teaching

`Robot.teach()` extracts a short label from the transcript, creates an image
embedding for the camera crop, creates a text embedding for the transcript,
and stores both in memory.

```text
transcript
  -> short object label
  -> Nomic text vector
  -> stored with the CLIP image vector
```

The transcript is retained as human-readable data, while its text vector lets
the application compare meaning rather than only exact spelling.

### Recall

`Robot.ask()` converts the question into a Nomic query vector and asks the
memory layer to find the most relevant taught object. It then retrieves stored
sightings for that object.

This project does not use a chat LLM to invent an answer. The answer is built
from stored labels, timestamps, locations, and thumbnails.

## Phase 8: Configure the Microphone

File: `robot/config.py`

The optional `MIC_DEVICE` setting selects a microphone by numeric index or
name. Leave it unset to use the operating system's default input device.

On Windows, list devices with:

```powershell
uv run python -c "import sounddevice as sd; print(sd.query_devices())"
```

Then add the chosen input index to `.env`, for example:

```dotenv
MIC_DEVICE=2
```

On Linux, `arecord -l` is an ALSA command for listing audio devices. It is not
the right command inside Windows or for a Python process running on Windows.

## Recommended Reading Order

1. `robot/device/live.py`: start with `_voice_action()`, `_phone_audio()`, and
   `_process()`.
2. `robot/device/mic.py`: inspect recording, RMS, silence, and trimming.
3. `robot/brain/models.py`: inspect model loading and `transcribe()`.
4. `robot/brain/core.py`: inspect `Robot.teach()` and `Robot.ask()`.
5. `robot/brain/memory.py`: inspect vector storage and semantic search.
6. `robot/config.py`: inspect microphone and camera settings.

## Troubleshooting Checklist

| Symptom | Likely cause | First check |
|---|---|---|
| `recorded level (rms): 0` | No audio reached Python | Windows microphone permissions and `MIC_DEVICE` |
| `mic failed` | Input device could not be opened | List devices with `sounddevice` |
| Whisper downloads on every run | Cache is unavailable or incomplete | Check the model cache and disk space |
| Transcript is empty or incorrect | Quiet, noisy, or clipped recording | Check RMS and speak near the microphone |
| The model is slow on first use | Local model initialization | Wait for warm-up; later runs use the cache |

The most useful debugging rule is to check the stages in order: first confirm
that audio was recorded, then confirm that the WAV is not silent, and only then
investigate model loading or transcription quality.
