#!/usr/bin/env python3
"""Guarded B210/SoapySDR duplex launcher using the existing root RX.py."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys


RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rf-path", choices=("cable", "ota"), required=True)
    parser.add_argument("--device", required=True, help="exact SoapySDR device_args from device probe")
    parser.add_argument("--frequency", type=int, required=True, help="center frequency in Hz")
    parser.add_argument("--tx-gain", type=float, required=True)
    parser.add_argument("--rx-gain", type=float, required=True)
    parser.add_argument("--tx-antenna", required=True)
    parser.add_argument("--rx-antenna", required=True)
    parser.add_argument("--tx-channel", type=int, required=True)
    parser.add_argument("--rx-channel", type=int, required=True)
    parser.add_argument("--attenuation-db", type=float, help="documented cable attenuation, required for cable path")
    parser.add_argument("--max-images", type=int, default=1, help="defaults to one burst")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tx-endpoint", default="tcp://127.0.0.1:5975")
    parser.add_argument("--rx-endpoint", default="tcp://127.0.0.1:5976")
    parser.add_argument("--allow-multi-burst", action="store_true",
                        help="required if --max-images is not 1")
    parser.add_argument("--acknowledge-rf", action="store_true",
                        help="required to start the SDR; --dry-run does not need it")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace, bundle_root: Path,
                  python: str) -> tuple[list[str], Path, list[Path]]:
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("--run-id must be 1–64 safe characters: letters, digits, _ or -")
    if not args.device.strip() or not args.tx_antenna.strip() or not args.rx_antenna.strip():
        raise ValueError("device and TX/RX antenna names must be nonempty")
    if args.frequency <= 0 or args.tx_channel < 0 or args.rx_channel < 0:
        raise ValueError("frequency must be positive and channels non-negative")
    if not math.isfinite(args.tx_gain) or not math.isfinite(args.rx_gain):
        raise ValueError("TX/RX gains must be finite")
    if args.max_images < 0 or args.batch_size <= 0:
        raise ValueError("--max-images must be non-negative and --batch-size positive")
    if args.max_images != 1 and not args.allow_multi_burst:
        raise ValueError("multiple bursts require --allow-multi-burst")
    if args.rf_path == "cable" and (args.attenuation_db is None or
                                    not math.isfinite(args.attenuation_db) or args.attenuation_db <= 0):
        raise ValueError("cable path requires a positive documented --attenuation-db")
    if args.rf_path == "ota" and args.attenuation_db is not None:
        raise ValueError("--attenuation-db is only valid for the cable path")

    project_root = bundle_root.parent if bundle_root.name.startswith("demo_clean_deepjscc_") else bundle_root
    dataset = args.dataset.resolve() if args.dataset else project_root / "data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy"
    checkpoint = args.checkpoint.resolve() if args.checkpoint else project_root / "models/div2k_lmdb_cv16_20260923/p0_clean.pt"
    service = project_root / "build/ofdm-workbench-service"
    rx_client = project_root / "RX.py"
    output = project_root / "captures" / f"clean_deepjscc_online_{args.run_id}"
    command = [
        python, "-m", "scripts.platform_integration.clean_demo",
        "--dataset", str(dataset), "--checkpoint", str(checkpoint),
        "--output-dir", str(output), "--transport", "workbench", "--mode", "hardware",
        "--service", str(service), "--rx-client", str(rx_client), "--rf-path", args.rf_path,
        "--device", args.device, "--frequency", str(args.frequency),
        "--tx-gain", str(args.tx_gain), "--rx-gain", str(args.rx_gain),
        "--tx-antenna", args.tx_antenna, "--rx-antenna", args.rx_antenna,
        "--tx-channel", str(args.tx_channel), "--rx-channel", str(args.rx_channel),
        "--tx-endpoint", args.tx_endpoint, "--rx-endpoint", args.rx_endpoint,
        "--batch-size", str(args.batch_size), "--max-images", str(args.max_images),
    ]
    if args.attenuation_db is not None:
        command += ["--attenuation-db", str(args.attenuation_db)]
    return command, output, [dataset, checkpoint, service, rx_client]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle_root = Path(__file__).resolve().parents[3]
    try:
        command, output, required = build_command(args, bundle_root, sys.executable)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"bundle: {bundle_root}", flush=True)
    print(f"output: {output}", flush=True)
    print(f"RF path: {args.rf_path}; physical setup is not verified by software", flush=True)
    print(f"command: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0
    if not args.acknowledge_rf:
        raise SystemExit("RF transmission requires --acknowledge-rf after physical and regulatory preflight")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit("required input or service not found: " + ", ".join(map(str, missing)))
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    return subprocess.run(command, cwd=bundle_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
