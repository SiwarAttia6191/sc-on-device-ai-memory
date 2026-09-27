import unittest

from testdata.metrics_dashboard import analyze_lines, render_dashboard


class MetricsDashboardTests(unittest.TestCase):
    def test_aggregates_stage_percentiles_and_skips_shell_noise(self):
        lines = [
            'uv : WARNING: not JSON\n',
            '{"event":"ai_stage","stage":"face_landmarks",'
            '"outcome":"ok","duration_ms":10,"cpu_ms":4}\n',
            '{"event":"ai_stage","stage":"face_landmarks",'
            '"outcome":"ok","duration_ms":20,"cpu_ms":8}\n',
            '{"event":"face_found"}\n',
        ]

        result = analyze_lines(lines)

        self.assertEqual(result["stage_samples"], 2)
        self.assertEqual(result["ignored_lines"], 1)
        self.assertEqual(result["events"]["face_found"], 1)
        self.assertEqual(result["stages"][0]["median_ms"], 15)
        self.assertEqual(result["stages"][0]["cpu_median_ms"], 6)

    def test_dashboard_does_not_echo_non_json_diagnostics(self):
        summary = analyze_lines([
            "PowerShell error text that should not appear in report\n",
            '{"event":"ai_stage","stage":"object_detection",'
            '"outcome":"ok","duration_ms":12,"cpu_ms":5}\n',
        ])

        dashboard = render_dashboard(summary, "private-log.jsonl")

        self.assertIn("object_detection", dashboard)
        self.assertNotIn("PowerShell error text", dashboard)
        self.assertNotIn('"transcript"', dashboard)

    def test_reassembles_json_records_wrapped_at_console_width(self):
        lines = [
            "native command warning that is not JSON",
            '{"event":"ai_stage","stage":"object_detection",',
            '"outcome":"ok","duration_ms":25,"cpu_ms":8,',
            '"model":"yoloe"}',
        ]

        result = analyze_lines(lines)

        self.assertEqual(result["stage_samples"], 1)
        self.assertEqual(result["stages"][0]["stage"], "object_detection")
        self.assertEqual(result["stages"][0]["median_ms"], 25)
        self.assertEqual(result["ignored_lines"], 1)


if __name__ == "__main__":
    unittest.main()