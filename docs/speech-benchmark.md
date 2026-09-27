# Speech Recognition Benchmark

This guide measures whether the robot transcribes your microphone correctly,
especially for words that sound similar, such as **mouth** and **mouse**.

## Why "mouth" Can Become "mouse"

Whisper is an automatic speech-recognition model. It predicts text from an
audio signal; it does not know what you intended to say. The words `mouth`
and `mouse` share several sounds, and the distinction can be difficult when:

- the microphone is far away or muted;
- the recording is noisy or clipped;
- the speaker has an accent different from the model's training examples;
- the word is spoken alone, without surrounding context;
- the audio is too quiet and the speech signal is close to the noise floor.

This is not a vector-memory problem. The error happens before the transcript
reaches the embedding and Qdrant search stages.

The correct engineering response is to measure the error on representative
recordings first. Do not globally replace `mouse` with `mouth`: that would fix
one example while corrupting a real request about a computer mouse.

## What the Benchmark Compares

The benchmark runs each labelled WAV file through two decoding settings:

| Variant | Setting | Purpose |
|---|---|---|
| `english` | `language="en"` | The application's current mode. |
| `auto-language` | `language=None` | Lets Whisper detect the language. |

For each result it reports:

- the transcript produced by Whisper;
- whether the expected target word or phrase appears in the transcript;
- target-hit accuracy across the recordings;
- elapsed transcription time.

The manifest target is not meant to be a full transcript. For example, with
target `mouth`, both `mouth` and `Do you think this is a mouth?` count as a
hit. `month` does not. Matching ignores case and punctuation and requires the
target token sequence, not a partial spelling.

This focused metric answers the question "Did Whisper recognize the target
word?" It does not evaluate the accuracy of every word in a sentence. The
benchmark does not generate audio or invent a ground-truth transcript; you
only label each recording with the word or phrase you intended to say.

## Create a Paired Recording Set

Create a directory, for example:

```powershell
New-Item -ItemType Directory speech-benchmark -Force
```

Record several samples for each target word. Use the same microphone,
distance, room, and speaking volume as the robot. Include both isolated words
and natural phrases because context can help speech recognition:

```text
mouth_01.wav: "mouth"
mouth_02.wav: "This is the mouth"
mouse_01.wav: "mouse"
mouse_02.wav: "This is the mouse"
```

The WAV files should be readable PCM WAV files. The robot's microphone code
normally writes 16 kHz mono audio, but `onnx-asr` can read common PCM WAV
formats and resample them for the model.

You can use the robot's own recording path while testing, or record with an
audio tool that exports WAV. Do not put private recordings into Git unless you
intend to publish them.

## Create the Manifest

Copy the example manifest:

```powershell
Copy-Item testdata/speech_manifest.example.csv speech-benchmark/manifest.csv
```

Edit `speech-benchmark/manifest.csv` so the paths are relative to the
manifest file:

```csv
file,target
mouth_01.wav,mouth
mouth_02.wav,mouth
mouse_01.wav,mouse
mouse_02.wav,mouse
```

The older header `file,reference` is still accepted for existing manifests;
`target` is the clearer name because only the expected word or phrase is
needed.

A matching directory should look like this:

```text
speech-benchmark/
  manifest.csv
  mouth_01.wav
  mouth_02.wav
  mouse_01.wav
  mouse_02.wav
```

## Run the Benchmark

From the repository root:

```powershell
uv run python testdata/benchmark_speech.py --manifest speech-benchmark/manifest.csv
```

The first run may load the cached Whisper model and take longer. The command
prints each transcript and a summary similar to:

```text
variant          target hits   accuracy
----------------------------------------
english             3/4          75.0%
auto-language       2/4          50.0%
```

The numbers above are illustrative. Your results are the ones to use.

## Choose the Fix from the Results

### If English mode wins

Keep the current setting:

```python
LANGUAGE = "en"
```

Improve the recording conditions next:

1. Select the correct Windows input device with `MIC_DEVICE` in `.env`.
2. Move closer to the microphone.
3. Check that Windows input volume is not too low or clipping.
4. Speak a complete phrase instead of an isolated ambiguous word.
5. Repeat the benchmark after each change.

### If automatic language detection wins

Your recordings may contain language variation or the fixed language setting
may not match the speech. This is a real benchmark result, but automatic
language detection can be slower and less stable on short clips. Do not switch
the live default based on one sample; use several samples and compare again.

### If both modes confuse the words

The bottleneck is probably the audio signal or the model's ability to resolve
the pronunciation. Try these changes in order:

1. Verify `recorded level (rms)` is clearly above zero.
2. Remove background noise and move the microphone closer.
3. Record complete contextual phrases.
4. Add at least 5 to 10 samples per word to the benchmark.
5. Compare a larger speech model only as a later experiment, because it costs
   more storage and CPU time.

### If the robot's object labels need a closed vocabulary

A domain-specific correction layer can be appropriate only when the allowed
labels are known. For example, a teaching workflow could ask the user to
confirm an uncertain label or choose between `mouth` and `mouse`. That is safer
than a global replacement rule because both words are valid.

## Code Reading Path

The benchmark uses these project boundaries:

- `testdata/benchmark_speech.py`: manifest loading, transcription variants,
  and target-word hit scoring.
- `robot/brain/models.py`: shared Whisper model and `transcribe()` wrapper.
- `robot/device/mic.py`: microphone recording, RMS measurement, silence
  detection, and WAV preparation.
- `robot/device/live.py`: `_voice_action()` and `_process()`, which connect
  recording to transcription and then to teaching or recall.

The benchmark passes `language` into the existing wrapper rather than loading
another speech model. This keeps the comparison focused on decoding settings.

## Reproducibility Checklist

For a useful comparison, keep these constant:

- microphone and Windows input device;
- speaker and pronunciation;
- room and background noise;
- recording distance and volume;
- Whisper model version;
- reference text in the manifest.

Change one factor at a time. Keep the manifest and benchmark output outside
Git if the recordings contain private voices.
