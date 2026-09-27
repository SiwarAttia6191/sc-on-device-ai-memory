"""Find face parts with MediaPipe landmarks and expose them as image crops."""
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from math import ceil, floor, isfinite
from pathlib import Path

from robot.observability import emit_event

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"
FACE_PARTS = (
    ("left eye", (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157,
                   158, 159, 160, 161, 246)),
    ("right eye", (362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390,
                    373, 374, 380, 381, 382)),
    ("nose", (1, 2, 19, 94, 97, 98, 129, 326, 327, 358)),
)


@dataclass(frozen=True)
class FacePart:
    name: str
    track_id: int
    box: tuple


def face_part_boxes(face_landmarks, width, height, face_index=0):
    """Convert normalized Face Mesh landmarks into padded feature boxes."""
    parts = []
    for part_index, (name, indices) in enumerate(FACE_PARTS):
        if len(face_landmarks) <= max(indices):
            continue
        points = [face_landmarks[index] for index in indices]
        if not all(isfinite(point.x) and isfinite(point.y) for point in points):
            continue
        xs = [point.x * width for point in points]
        ys = [point.y * height for point in points]
        span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
        pad_x, pad_y = max(3.0, span_x * 0.30), max(3.0, span_y * 0.35)
        x1 = max(0, floor(min(xs) - pad_x))
        y1 = max(0, floor(min(ys) - pad_y))
        x2 = min(width, ceil(max(xs) + pad_x))
        y2 = min(height, ceil(max(ys) + pad_y))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        track_id = -(face_index * len(FACE_PARTS) + part_index + 1)
        parts.append(FacePart(name, track_id, (x1, y1, x2, y2)))
    return parts


def _model_path():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                dir=MODEL_PATH.parent, suffix=".task", delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(MODEL_URL, timeout=45) as response:
                shutil.copyfileobj(response, stream)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("downloaded face-landmark model is incomplete")
        temporary.replace(MODEL_PATH)
    except Exception as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download the face-landmark model to {MODEL_PATH}: "
            f"{exc}") from exc
    return MODEL_PATH


class FaceLandmarkDetector:
    """Run MediaPipe Face Landmarker in video mode on camera frames."""

    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "install face-landmark support with `uv sync --extra face`"
            ) from exc
        self._mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_model_path())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options)
        self._last_timestamp = -1
        self._face_present = None

    def detect(self, frame, now):
        import cv2

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp = max(int(now * 1000), self._last_timestamp + 1)
        self._last_timestamp = timestamp
        result = self._landmarker.detect_for_video(image, timestamp)
        found = bool(result.face_landmarks)
        if found != self._face_present:
            emit_event("face_found" if found else "no_face_found")
            self._face_present = found
        if not found:
            return []
        return face_part_boxes(result.face_landmarks[0], width, height)

    def close(self):
        self._landmarker.close()