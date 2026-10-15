"""CPU-only safety and command checks; no SDR is opened by these tests."""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.platform_integration.clean_demo.run_online_demo import build_command, parse_args


RADIO = [
    "--run-id", "rf_smoke_001", "--rf-path", "cable",
    "--device", "driver=uhd,serial=SERIAL", "--frequency", "2437000000",
    "--tx-gain", "0", "--rx-gain", "10", "--tx-antenna", "TX/RX",
    "--rx-antenna", "RX2", "--tx-channel", "0", "--rx-channel", "0",
    "--attenuation-db", "30",
]


class OnlineDemoLauncherTests(unittest.TestCase):
    def test_command_is_hardware_only_and_output_is_separate(self):
        args = parse_args(RADIO)
        command, output, required = build_command(
            args, Path("/work/demo_clean_deepjscc_20260923"), "python",
        )
        self.assertEqual(output, Path("/work/captures/clean_deepjscc_online_rf_smoke_001"))
        self.assertIn("hardware", command)
        self.assertIn("cable", command)
        self.assertNotIn("--simulate", command)
        self.assertIn(Path("/work/build/ofdm-workbench-service"), required)
        self.assertEqual(required[-1], Path("/work/RX.py"))
        self.assertEqual(command[command.index("--rx-client") + 1], str(Path("/work/RX.py")))
        self.assertEqual(args.max_images, 1)

    def test_cable_requires_documented_attenuation(self):
        args = parse_args(RADIO[:-2])
        with self.assertRaisesRegex(ValueError, "attenuation"):
            build_command(args, Path("/work/demo_clean_deepjscc_20260923"), "python")

    def test_multiple_bursts_require_explicit_opt_in(self):
        args = parse_args(RADIO + ["--max-images", "0"])
        with self.assertRaisesRegex(ValueError, "allow-multi-burst"):
            build_command(args, Path("/work/demo_clean_deepjscc_20260923"), "python")

    def test_run_id_cannot_escape_capture_directory(self):
        args = parse_args(["--run-id", "../outside", *RADIO[2:]])
        with self.assertRaisesRegex(ValueError, "run-id"):
            build_command(args, Path("/work/demo_clean_deepjscc_20260923"), "python")


if __name__ == "__main__":
    unittest.main()
