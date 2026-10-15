#!/usr/bin/env python3
"""Receive OFDM RX results and save recovered data in batches of 100.

Install dependencies: python -m pip install numpy pyzmq

The OFDM service owns a PUSH socket and binds tcp://127.0.0.1:5556 by
default, so this client uses PULL + connect. Each result contains:
  1. UTF-8 JSON metadata
  2. 64 complex-float32 CSI values, little-endian interleaved IQ
  3. Recovered complex-float32 data, little-endian interleaved IQ

Each data payload counts as one record. Every 100 records are saved to
one .npy file. Ctrl-C or --max-bursts also saves the final partial batch.
Equal-length records form a (records, complex_samples) complex64 array.
Variable-length records form a 1-D object array of complex64 arrays;
loading that format requires np.load(path, allow_pickle=True).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import pathlib
import struct
from typing import Iterable

import numpy as np


BATCH_SIZE = 100


def unpack_cf32(payload: bytes) -> list[complex]:
    if len(payload) % 8:
        raise ValueError(f"CF32 payload has {len(payload)} bytes; expected a multiple of 8")
    return [complex(real, imag) for real, imag in struct.iter_unpack("<ff", payload)]


def rms(values: Iterable[complex]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return math.sqrt(sum(abs(value) ** 2 for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive OFDM data; save every 100 bursts to NPY")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5556",
                        help="RX PUSH endpoint exposed by the OFDM service")
    parser.add_argument("--max-bursts", type=int, default=0,
                        help="stop after this many results; 0 means run until Ctrl-C")
    parser.add_argument("--timeout-ms", type=int, default=1000,
                        help="receive timeout (default: 1000 ms)")
    parser.add_argument("--save-dir", type=pathlib.Path, default=pathlib.Path("captures"),
                        help="metadata JSON and CF32 output directory (default: ./captures)")
    parser.add_argument("--npy-dir", type=pathlib.Path, default=pathlib.Path("captures/npy"),
                        help="data batch NPY output directory (default: ./captures/npy)")
    return parser.parse_args()


def save_result(directory: pathlib.Path, burst_id: int, metadata: dict,
                csi_payload: bytes, data_payload: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"burst_{burst_id:08d}"
    stem.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / f"burst_{burst_id:08d}.csi.cf32le").write_bytes(csi_payload)
    (directory / f"burst_{burst_id:08d}.data.cf32le").write_bytes(data_payload)


def save_data_batch(directory: pathlib.Path, batch_id: int,
                    data_batch: list[np.ndarray]) -> pathlib.Path:
    """Save one batch, preserving record boundaries and complex64 samples."""
    if not data_batch:
        raise ValueError("Cannot save an empty data batch")
    directory.mkdir(parents=True, exist_ok=True)

    if all(len(record) == len(data_batch[0]) for record in data_batch):
        array = np.stack(data_batch, axis=0)
    else:
        # Preserve every record if the OFDM payload lengths differ.
        array = np.empty(len(data_batch), dtype=object)
        for index, record in enumerate(data_batch):
            array[index] = record

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = directory / f"data_{timestamp}_batch_{batch_id:06d}.npy"
    # Exclusive creation prevents an existing capture from being overwritten.
    with output.open("xb") as handle:
        np.save(handle, array, allow_pickle=array.dtype.hasobject)
    print(f"RX saved {len(data_batch)} data records -> {output} "
          f"shape={array.shape} dtype={array.dtype}", flush=True)
    return output


def main() -> int:
    args = parse_args()
    if args.max_bursts < 0 or args.timeout_ms < 0:
        raise SystemExit("--max-bursts and --timeout-ms must not be negative")

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 4)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.endpoint)

    received = 0
    batch_id = 1
    data_buffer: list[np.ndarray] = []
    try:
        while args.max_bursts == 0 or received < args.max_bursts:
            try:
                parts = socket.recv_multipart()
            except zmq.Again:
                continue
            if len(parts) != 3:
                print(f"RX ignored malformed multipart result: {len(parts)} parts", flush=True)
                continue
            try:
                metadata = json.loads(parts[0].decode("utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be a JSON object")
                burst_id = int(metadata.get("burst_id", received + 1))
                csi = unpack_cf32(parts[1])
                data = unpack_cf32(parts[2])
            except (UnicodeDecodeError, ValueError, TypeError, OverflowError) as error:
                print(f"RX ignored malformed result: {error}", flush=True)
                continue
            if len(csi) != 64:
                print(f"RX ignored result with {len(csi)} CSI bins (expected 64)", flush=True)
                continue

            received += 1
            data_buffer.append(np.asarray(data, dtype=np.complex64))
            if len(data_buffer) >= BATCH_SIZE:
                save_data_batch(args.npy_dir, batch_id, data_buffer)
                data_buffer.clear()
                batch_id += 1

            csi_rms = rms(csi)
            csi_peak = max((abs(value) for value in csi), default=0.0)
            print(
                "RX burst={burst} data={data} csi_bins=64 csi_rms={rms:.5g} "
                "csi_peak={peak:.5g} cfo_hz={cfo} snr_db={snr} quality_db={quality}".format(
                    burst=burst_id,
                    data=len(data),
                    rms=csi_rms,
                    peak=csi_peak,
                    cfo=metadata.get("cfo_hz", "?"),
                    snr=metadata.get("snr_db", "?"),
                    quality=metadata.get("csi_quality_db", "?"),
                ),
                flush=True,
            )
            if args.save_dir:
                save_result(args.save_dir, burst_id, metadata, parts[1], parts[2])
    except KeyboardInterrupt:
        print("\nRX stopped", flush=True)
    finally:
        socket.close(0)
        context.term()
        if data_buffer:
            save_data_batch(args.npy_dir, batch_id, data_buffer)
            data_buffer.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
