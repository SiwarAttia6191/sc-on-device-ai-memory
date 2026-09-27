import unittest
from unittest.mock import patch

from robot.device.server import StreamHandler


class StreamDisconnectTests(unittest.TestCase):
    def test_get_stream_ignores_windows_connection_abort(self):
        handler = StreamHandler.__new__(StreamHandler)
        handler.path = "/stream"

        with patch.object(handler, "_stream",
                          side_effect=ConnectionAbortedError(10053, "aborted")):
            handler.do_GET()


if __name__ == "__main__":
    unittest.main()