import contextlib
import io
import json
import unittest

from robot.observability import emit_event, measure_stage


class ObservabilityTests(unittest.TestCase):
    def test_stage_record_has_wall_and_cpu_time_without_payload(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            with measure_stage("test_stage", model="test"):
                pass

        event = json.loads(stream.getvalue())
        self.assertEqual(event["event"], "ai_stage")
        self.assertEqual(event["stage"], "test_stage")
        self.assertEqual(event["outcome"], "ok")
        self.assertGreaterEqual(event["duration_ms"], 0)
        self.assertGreaterEqual(event["cpu_ms"], 0)
        self.assertNotIn("transcript", event)
        self.assertNotIn("audio", event)

    def test_stage_records_safe_error_type_and_reraises(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            with self.assertRaises(ValueError):
                with measure_stage("test_stage"):
                    raise ValueError("private input must not be logged")

        event = json.loads(stream.getvalue())
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["error_type"], "ValueError")
        self.assertNotIn("private input", stream.getvalue())

    def test_event_emits_only_explicit_attributes(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            emit_event("transcript_discarded")

        self.assertEqual(json.loads(stream.getvalue()), {
            "event": "transcript_discarded"})


if __name__ == "__main__":
    unittest.main()