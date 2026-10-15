#!/usr/bin/env python3
"""Launch one stage-isolated clean DeepJSCC run with a safe, unique run ID."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys


RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="unique capture suffix; letters, digits, _ and - only")
    parser.add_argument("--transport", choices=("file-services", "workbench"), default="file-services")
    parser.add_argument("--max-images", type=int, default=1, help="1 for preflight; 0 for all 50 fixed patches")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset", type=Path, help="override the fixed LMDB export path")
    parser.add_argument("--checkpoint", type=Path, help="override the clean P0 checkpoint path")
    parser.add_argument("--tx-endpoint", default="tcp://127.0.0.1:5955")
    parser.add_argument("--rx-endpoint", default="tcp://127.0.0.1:5956")
    parser.add_argument("--dry-run", action="store_true", help="print paths and command without starting services")
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace, bundle_root: Path,
                  python: str) -> tuple[list[str], Path, list[Path]]:
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError("--run-id must be 1–64 safe characters: letters, digits, _ or -")
    if args.max_images < 0 or args.batch_size <= 0:
        raise ValueError("--max-images must be non-negative and --batch-size positive")
    project_root = bundle_root.parent if bundle_root.name.startswith("demo_clean_deepjscc_") else bundle_root
    dataset = args.dataset.resolve() if args.dataset else project_root / "data/DIV2K/lmdb_cv16_test_uint8_20260923/test.npy"
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.resolve()
    else:
        xwt_checkpoint = project_root / "models/div2k_lmdb_cv16_20260923/p0_clean.pt"
        gpu_checkpoint = project_root / "runs/div2k_lmdb_cv16_20260923/p0_full/best.pt"
        checkpoint = xwt_checkpoint if xwt_checkpoint.is_file() else gpu_checkpoint
    output = project_root / "captures" / f"clean_deepjscc_{args.transport}_{args.run_id}"
    command = [
        python, "-m", "scripts.platform_integration.clean_demo",
        "--dataset", str(dataset), "--checkpoint", str(checkpoint),
        "--output-dir", str(output), "--batch-size", str(args.batch_size),
        "--max-images", str(args.max_images),
        "--tx-endpoint", args.tx_endpoint, "--rx-endpoint", args.rx_endpoint,
    ]
    services: list[Path]
    if args.transport == "file-services":
        tx_service = project_root / "build/ofdm-tx-service"
        rx_service = project_root / "build/ofdm-rx-service"
        command += ["--transport", "file-services", "--tx-service", str(tx_service),
                    "--rx-service", str(rx_service)]
        services = [tx_service, rx_service]
    else:
        service = project_root / "build/ofdm-workbench-service"
        command += ["--transport", "workbench", "--mode", "simulate", "--service", str(service)]
        services = [service]
    return command, output, [dataset, checkpoint, *services]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle_root = Path(__file__).resolve().parents[3]
    try:
        command, output, required = build_command(args, bundle_root, sys.executable)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"bundle: {bundle_root}", flush=True)
    print(f"output: {output}", flush=True)
    print(f"command: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit("required input or service not found: " + ", ".join(map(str, missing)))
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    return subprocess.run(command, cwd=bundle_root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
