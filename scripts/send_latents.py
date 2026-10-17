#!/usr/bin/env python3
"""Validate saved Encoder records, then hand each unchanged CF32 file to TX.py."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def load_records(manifest_path: Path) -> list[tuple[int, Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("records"):
        raise ValueError("encoder manifest must be complete and nonempty")
    base = manifest_path.parent.resolve()
    records, seen = [], set()
    for record in manifest["records"]:
        burst_id = record["burst_id"]
        if type(burst_id) is not int or not 0 <= burst_id <= 0xFFFFFFFF or burst_id in seen:
            raise ValueError("burst IDs must be unique uint32")
        path = (base / record["latent"]).resolve()
        if base not in path.parents:
            raise ValueError("latent path must stay inside encoder output")
        payload = path.read_bytes()
        if len(payload) != record["latent_samples"] * 8 or not payload:
            raise ValueError(f"latent size mismatch: {path}")
        if hashlib.sha256(payload).hexdigest() != record["latent_sha256"]:
            raise ValueError(f"latent hash mismatch: {path}")
        seen.add(burst_id)
        records.append((burst_id, path))
    if len(records) != manifest.get("planned_records"):
        raise ValueError("incomplete encoder record list")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tx-script", type=Path, default=ROOT / "TX.py")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("interval must be finite and nonnegative")
    records = load_records(args.manifest)
    for index, (burst_id, path) in enumerate(records):
        command = [sys.executable, str(args.tx_script), "--input-bin", str(path),
                   "--start-id", str(burst_id), "--count", "1", "--endpoint", args.endpoint]
        print(f"burst={burst_id} file={path}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
            if index + 1 < len(records):
                time.sleep(args.interval)
    print(f"{'validated' if args.dry_run else 'submitted_to_TX'}={len(records)}; not an RF delivery acknowledgement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
