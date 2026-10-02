"""Pinned environment verification rejects dependency drift."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from verify_environment import verify_pins


class EnvironmentTests(unittest.TestCase):
    def test_package_pins_are_checked_and_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("# locked packages\n--extra-index-url https://example.invalid\nexample==1.2.3\n")
            with patch("verify_environment.importlib.metadata.version", return_value="1.2.3"):
                self.assertEqual(verify_pins(path), {"example": "1.2.3"})
            with patch("verify_environment.importlib.metadata.version", return_value="1.2.4"):
                with self.assertRaisesRegex(RuntimeError, "expected 1.2.3"):
                    verify_pins(path)
