"""Compare local Whisper transcription settings on labelled WAV recordings.

Manifest format (CSV with a header):

    file,target
    mouth_01.wav,mouth
    mouse_01.wav,mouse

Run from the repository root:

    uv run python testdata/benchmark_speech.py --manifest testdata/speech.csv

The benchmark compares the application's current English-constrained mode
with Whisper's automatic language-detection mode. It reports whether each
target word or phrase appears in the transcript. The legacy ``reference``
column is also accepted as an alias for ``target``.
"""
import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.brain import models


@dataclass
class Sample:
    path: Path
    target: str


def normalize(text):
    """Make comparison insensitive to case and punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def target_found(target, transcript):
    """Match a target token sequence anywhere in a transcript."""
    expected, actual = normalize(target), normalize(transcript)
    if not expected:
        return False
    return any(actual[i:i + len(expected)] == expected
               for i in range(len(actual) - len(expected) + 1))


def load_manifest(path):
    """Load and validate a CSV manifest relative to its own directory."""
    samples = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), 2):
            name = (row.get("file") or "").strip()
            target = (row.get("target") or row.get("reference") or "").strip()
            if not name or not target:
                raise SystemExit(
                    f"{path}:{row_number} needs non-empty file and target "
                    "(or legacy reference)")
            audio = (path.parent / name).resolve()
            if not audio.is_file():
                raise SystemExit(f"audio file does not exist: {audio}")
            samples.append(Sample(audio, target))
    if not samples:
        raise SystemExit(f"manifest has no samples: {path}")
    return samples


def run_variant(samples, name, language):
    rows, hits = [], 0
    for sample in samples:
        started = time.perf_counter()
        text = models.transcribe(sample.path, language=language)
        elapsed = time.perf_counter() - started
        found = target_found(sample.target, text)
        hits += found
        rows.append((sample, text, elapsed, found))
    return name, rows, hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="CSV containing file and reference columns")
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"manifest does not exist: {args.manifest}")

    samples = load_manifest(args.manifest.resolve())
    print("Loading Whisper once; later variants reuse the same model.")
    variants = [
        run_variant(samples, "english", "en"),
        run_variant(samples, "auto-language", None),
    ]

    print(f"\n{'file':30s} {'target':12s} {'variant':15s} {'hit':>5s} transcript")
    print("-" * 110)
    for name, rows, *_ in variants:
        for sample, text, elapsed, found in rows:
            print(f"{sample.path.name:30.30s} {sample.target:12.12s} "
                  f"{name:15s} {str(found):>5s} {text!r} ({elapsed:.2f}s)")

    print(f"\n{'variant':15s} {'target hits':>12s} {'accuracy':>10s}")
    print("-" * 40)
    for name, _, hits in variants:
        accuracy = hits / len(samples)
        print(f"{name:15s} {hits:>5d}/{len(samples):<6d} {accuracy:>9.1%}")


if __name__ == "__main__":
    main()