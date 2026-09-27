import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from robot.brain.core import Robot
from robot.brain.face_landmarks import FACE_PARTS, face_part_boxes


class FacePartBoxTests(unittest.TestCase):
    @patch("robot.brain.core.Detector")
    @patch("robot.brain.core.Memory")
    def test_robot_passes_face_landmark_option_to_detector(
            self, memory, detector):
        with tempfile.TemporaryDirectory() as data_dir:
            Robot(data_dir=data_dir, face_landmarks=True)

        detector.assert_called_once_with(
            "yoloe-11l-seg-pf.pt", face_landmarks=True)

    def test_returns_eye_and_nose_regions_with_stable_ids(self):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        centers = ((0.38, 0.40), (0.62, 0.40), (0.50, 0.55))
        for (name, indices), (center_x, center_y) in zip(FACE_PARTS, centers):
            for index, offset in zip(indices, range(len(indices))):
                points[index] = SimpleNamespace(
                    x=center_x + (offset % 4 - 1.5) * 0.004,
                    y=center_y + (offset // 4 - 1.5) * 0.004,
                )

        parts = face_part_boxes(points, 1280, 720)

        self.assertEqual([part.name for part in parts],
                         ["left eye", "right eye", "nose"])
        self.assertEqual([part.track_id for part in parts], [-1, -2, -3])
        self.assertTrue(all(0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720
                            for part in parts
                            for x1, y1, x2, y2 in [part.box]))
        self.assertGreater(parts[2].box[1], parts[0].box[3])

    def test_skips_missing_landmark_indices_and_too_small_regions(self):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]

        self.assertEqual(face_part_boxes(points, 1280, 720), [])
        self.assertEqual(face_part_boxes(points[:100], 1280, 720), [])


if __name__ == "__main__":
    unittest.main()