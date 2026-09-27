import threading
import unittest
from unittest.mock import MagicMock, call, patch

import cv2

from robot.device.live import LiveApp


class LiveAppTests(unittest.TestCase):
    def voice_review_app(self, action, transcript="recognized text", target=None):
        app = LiveApp.__new__(LiveApp)
        app.lock = threading.Lock()
        app.busy = False
        app.voice_review = (action, transcript, target)
        app.pending_track = None
        app._banner = None
        app._banner_seq = 0
        app.robot = MagicMock()
        app.robot.memory.count.return_value = 1
        app.robot.ask.return_value = {
            "inventory": False, "label": None, "sightings": []}
        return app

    @patch("robot.device.live.sys.platform", "win32")
    @patch("robot.device.live.cv2.VideoCapture")
    def test_windows_camera_falls_back_from_directshow(self, video_capture):
        directshow = MagicMock()
        directshow.isOpened.return_value = False
        default = MagicMock()
        default.isOpened.return_value = True
        video_capture.side_effect = [directshow, default]

        LiveApp(MagicMock(), camera=2)

        self.assertEqual(video_capture.call_args_list, [
            call(2, cv2.CAP_DSHOW),
            call(2, cv2.CAP_ANY),
        ])
        directshow.release.assert_called_once_with()

    def test_voice_path_can_only_be_claimed_once(self):
        app = LiveApp.__new__(LiveApp)
        app.lock = threading.Lock()
        app.busy = False
        app.voice_review = None

        self.assertTrue(app._start_voice())
        self.assertFalse(app._start_voice())
        app._finish_voice()
        self.assertTrue(app._start_voice())

    @patch("robot.device.live._speak")
    def test_confirmed_transcript_is_used_for_recall(self, speak):
        app = self.voice_review_app("a")

        result = app.confirm_transcript("  corrected   question  ")

        self.assertEqual(result, {"ok": True})
        app.robot.ask.assert_called_once_with("corrected question")
        self.assertIsNone(app.voice_review)
        self.assertFalse(app.busy)
        speak.assert_called_once_with(app.robot.ask.return_value)

    def test_blank_confirmation_keeps_review_pending(self):
        app = self.voice_review_app("a")

        result = app.confirm_transcript(" \n ")

        self.assertFalse(result["ok"])
        self.assertIsNotNone(app.voice_review)
        app.robot.ask.assert_not_called()

    def test_edited_transcript_is_used_for_teaching(self):
        target = ("crop", "frame", (1, 2, 3, 4))
        app = self.voice_review_app("t", target=target)
        app.robot.teach.return_value = {
            "label": "mug", "transcript": "This is my mug"}

        result = app.confirm_transcript("This is my mug")

        self.assertEqual(result, {"ok": True})
        app.robot.teach.assert_called_once_with(
            "crop", "This is my mug", frame="frame", box=(1, 2, 3, 4))
        app.robot.memory.count.assert_called_once_with()
        self.assertFalse(app.busy)

    def test_discard_transcript_does_not_call_robot(self):
        app = self.voice_review_app("t", target=("crop", None, None))

        self.assertTrue(app.discard_transcript())
        self.assertIsNone(app.voice_review)
        app.robot.teach.assert_not_called()
        app.robot.ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
