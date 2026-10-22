#!/usr/bin/env python3
"""Validate saved RX bursts before decoder use; never changes RX.py or captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate_capture(directory: Path, expected_samples: int = 65536,
                     tx_manifest: Path | None = None) -> dict:
    files = sorted(directory.glob("burst_*.json"))
    if not files:
        raise ValueError("no saved burst metadata")
    ids = []
    for path in files:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        burst_id = metadata.get("burst_id")
        if type(burst_id) is not int or burst_id in ids or path.stem != f"burst_{burst_id:08d}":
            raise ValueError(f"invalid/duplicate burst_id: {path}")
        if metadata.get("crc_ok") is not True or metadata.get("original_length") != expected_samples:
            raise ValueError(f"CRC/original_length invalid: {path}")
        for suffix, count in ((".csi.cf32le", 64), (".data.cf32le", expected_samples)):
            payload = path.with_suffix(suffix).read_bytes()
            if len(payload) != count * 8 or not np.isfinite(np.frombuffer(payload, dtype="<c8")).all():
                raise ValueError(f"invalid {suffix}: {path}")
        ids.append(burst_id)
    # RX's batched NPY format omits burst IDs, so validate its shape and count.
    # Per-burst files above are checked independently for length and finite values.
    count = 0
    npy_files = sorted((directory / "npy").glob("*.npy"))
    for path in npy_files:
        records = np.load(path, allow_pickle=False)
        if records.ndim != 2 or records.shape[1] != expected_samples or records.dtype.kind != "c" or records.dtype.itemsize != 8:
            raise ValueError(f"expected complex64 [N,{expected_samples}]: {path}")
        for record in records:
            if not np.isfinite(record).all():
                raise ValueError(f"non-finite NPY samples: {path}")
            count += 1
    if count != len(ids):
        raise ValueError("NPY record count differs from saved bursts; stop RX to flush its last batch first")
    ids.sort()
    expected_ids = None
    if tx_manifest is not None:
        manifest = json.loads(tx_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError("incomplete TX manifest")
        expected_ids = [record["burst_id"] for record in manifest["records"]]
        if len(set(expected_ids)) != len(expected_ids) or sorted(expected_ids) != ids:
            raise ValueError(f"missing/unexpected bursts: expected {expected_ids}, received {ids}")
    return {"status": "valid", "bursts": len(ids), "burst_ids_sorted": ids,
            "npy_records": count, "expected_ids_checked": expected_ids is not None,
            "note": "control CRC is not a business-IQ checksum; NPY order is reception order and records are not matched byte-for-byte to burst files"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rx_dir", type=Path)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--tx-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_capture(args.rx_dir, args.samples, args.tx_manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
