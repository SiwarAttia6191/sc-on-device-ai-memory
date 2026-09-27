"""Small privacy-safe stage metrics for the on-device pipeline."""
import json
import sys
import time
from contextlib import contextmanager


@contextmanager
def measure_stage(name, **attributes):
    """Emit one JSON timing record without logging model inputs or outputs."""
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    outcome = "ok"
    try:
        yield
    except Exception as exc:
        outcome = "error"
        attributes["error_type"] = type(exc).__name__
        raise
    finally:
        event = {
            "event": "ai_stage",
            "stage": name,
            "outcome": outcome,
            "duration_ms": round((time.perf_counter() - wall_start) * 1000, 2),
            "cpu_ms": round((time.process_time() - cpu_start) * 1000, 2),
            **attributes,
        }
        print(json.dumps(event, separators=(",", ":")), file=sys.stderr,
              flush=True)


def emit_event(name, **attributes):
    """Emit a low-volume pipeline outcome without sensitive input content."""
    print(json.dumps({"event": name, **attributes}, separators=(",", ":")),
          file=sys.stderr, flush=True)