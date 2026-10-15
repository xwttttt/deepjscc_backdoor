"""RX boundary: require matching burst, anchor CRC, CSI, and equalized latent."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import zmq


def file_rx_command(service: str, endpoint: str, waveform_path: str) -> list[str]:
    """Use xwt's standalone RX interface to decode one saved OFDM burst."""
    return [service, "--endpoint", endpoint, "--file-in", waveform_path,
            "--equalizer", "mmse", "--once"]


def rx_client_command(python: str, client: Path, endpoint: str, directory: Path,
                      burst_count: int) -> list[str]:
    """Run the existing root RX.py as the sole workbench result consumer."""
    return [python, str(client), "--endpoint", endpoint,
            "--max-bursts", str(burst_count), "--save-dir", str(directory),
            "--npy-dir", str(directory / "npy")]


def _decode_frames(parts: list[bytes], burst_id: int,
                   expected_samples: int) -> tuple[dict, np.ndarray, np.ndarray]:
    if len(parts) != 3:
        raise RuntimeError(f"RX burst {burst_id} returned {len(parts)} parts, expected 3")
    metadata = json.loads(parts[0].decode("utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError(f"RX burst {burst_id} returned non-object metadata")
    if int(metadata.get("burst_id", -1)) != burst_id:
        raise RuntimeError(f"RX burst order mismatch: expected {burst_id}, got {metadata.get('burst_id')}")
    if metadata.get("crc_ok") is not True:
        raise RuntimeError(f"RX burst {burst_id} failed anchor CRC")
    if "original_length" in metadata and int(metadata["original_length"]) != expected_samples:
        raise RuntimeError(f"RX burst {burst_id} returned wrong original length")
    if metadata.get("format", "cf32le") != "cf32le":
        raise RuntimeError(f"RX burst {burst_id} returned unsupported format")
    if len(parts[1]) != 64 * 8 or len(parts[2]) != expected_samples * 8:
        raise RuntimeError(f"RX burst {burst_id} returned malformed CSI or latent length")
    csi = np.frombuffer(parts[1], dtype="<c8").copy()
    equalized = np.frombuffer(parts[2], dtype="<c8").copy()
    if not np.isfinite(csi).all() or not np.isfinite(equalized).all():
        raise RuntimeError(f"RX burst {burst_id} returned non-finite samples")
    return metadata, csi, equalized


def receive_burst(socket, burst_id: int, expected_samples: int,
                  timeout_seconds: float) -> tuple[dict, np.ndarray, np.ndarray]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            parts = socket.recv_multipart()
        except zmq.Again:
            continue
        return _decode_frames(parts, burst_id, expected_samples)
    raise TimeoutError(f"RX burst {burst_id} timed out")


def receive_saved_burst(directory: Path, burst_id: int, expected_samples: int,
                        timeout_seconds: float, process) -> tuple[dict, np.ndarray, np.ndarray]:
    """Read one RX.py capture without opening a competing ZeroMQ PULL socket."""
    stem = directory / f"burst_{burst_id:08d}"
    metadata_path = stem.with_suffix(".json")
    csi_path = directory / f"burst_{burst_id:08d}.csi.cf32le"
    data_path = directory / f"burst_{burst_id:08d}.data.cf32le"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if data_path.is_file() and csi_path.is_file() and metadata_path.is_file():
            if data_path.stat().st_size == expected_samples * 8:
                return _decode_frames(
                    [metadata_path.read_bytes(), csi_path.read_bytes(), data_path.read_bytes()],
                    burst_id, expected_samples,
                )
        if process.poll() is not None:
            raise RuntimeError(f"RX.py exited before burst {burst_id} was saved: {process.returncode}")
        time.sleep(0.02)
    raise TimeoutError(f"RX.py did not save burst {burst_id} within {timeout_seconds}s")
