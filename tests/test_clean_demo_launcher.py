"""CPU-only command and safety checks for the clean demo launcher."""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.platform_integration.clean_demo.run_clean_demo import build_command, parse_args


class CleanDemoLauncherTests(unittest.TestCase):
    def test_file_services_command_uses_one_isolated_capture(self):
        args = parse_args(["--run-id", "smoke_001", "--max-images", "1",
                           "--checkpoint", "/model/p0.pt"])
        bundle = Path("/work/demo_clean_deepjscc_20260923")
        command, output, required = build_command(args, bundle, "/env/bin/python")
        self.assertEqual(output, Path("/work/captures/clean_deepjscc_file-services_smoke_001"))
        self.assertIn("--transport", command)
        self.assertIn("file-services", command)
        self.assertIn(Path("/work/build/ofdm-tx-service"), required)
        self.assertIn(Path("/work/build/ofdm-rx-service"), required)

    def test_workbench_is_explicit_and_does_not_use_standalone_services(self):
        args = parse_args(["--run-id", "sim1", "--transport", "workbench",
                           "--checkpoint", "/model/p0.pt"])
        command, _, required = build_command(args, Path("/work/demo_clean_deepjscc_20260923"), "python")
        self.assertIn("simulate", command)
        self.assertEqual(required[-1], Path("/work/build/ofdm-workbench-service"))

    def test_run_id_cannot_escape_capture_directory(self):
        args = parse_args(["--run-id", "../escape"])
        with self.assertRaisesRegex(ValueError, "run-id"):
            build_command(args, Path("/work/demo_clean_deepjscc_20260923"), "python")


if __name__ == "__main__":
    unittest.main()
