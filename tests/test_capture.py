from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from read_capture import read_capture


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "samples" / "capture.json"


class CaptureTests(unittest.TestCase):
    def test_event_order_and_dependency(self):
        self.assertEqual(np.__version__, "2.2.6")
        events = read_capture(CAPTURE)["events"]
        self.assertEqual([event[0] for event in events], ['default', 'accept', 'default', 'accept', 'accept', 'accept', 'accept', 'drain'])

    def test_invalid_event_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"events":[[]]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "事件列表"):
                read_capture(path)

    def test_command_reads_the_file(self):
        result = subprocess.run(
            [sys.executable, "-m", "read_capture", str(CAPTURE)], cwd=ROOT,
            text=True, capture_output=True, check=True,
        )
        self.assertEqual(len(result.stdout.splitlines()), 8)

    def test_capture_geometry_and_values(self):
        capture = read_capture(CAPTURE)
        self.assertEqual(np.asarray(capture["base"]["packed"], dtype=np.uint8).shape, (3, 4))
        self.assertEqual(np.asarray(capture["base"]["zero"]).shape, (3, 2))
        self.assertEqual({name: len(a["A"]) for name, a in capture["adapters"].items()}, {"r0": 1, "r1": 2})
