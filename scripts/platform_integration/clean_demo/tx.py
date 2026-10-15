"""TX service command, optional pre-OFDM impairment, and packet serialization."""

from __future__ import annotations

import json

import numpy as np

from ..deepjscc_codec import apply_iq_dc_numpy


def service_command(args) -> list[str]:
    command = [str(args.service), "--tx-endpoint", args.tx_endpoint, "--rx-endpoint", args.rx_endpoint]
    if args.mode == "simulate":
        return command + ["--simulate"]
    if not args.device:
        raise ValueError("--device is required for hardware mode")
    return command + [
        "--device", args.device, "--frequency", str(args.frequency),
        "--tx-gain", str(args.tx_gain), "--rx-gain", str(args.rx_gain),
        "--tx-antenna", args.tx_antenna, "--rx-antenna", args.rx_antenna,
        "--tx-channel", str(args.tx_channel), "--rx-channel", str(args.rx_channel),
        "--role", "duplex",
    ]


def file_tx_command(service: str, endpoint: str, waveform_path: str) -> list[str]:
    """Use xwt's standalone TX interface to export one complete OFDM burst."""
    return [service, "--endpoint", endpoint, "--file-out", waveform_path, "--once"]


def prepare_payload(latent: np.ndarray, impairment: dict | None) -> tuple[np.ndarray, bytes]:
    transmitted = latent if impairment is None else apply_iq_dc_numpy(latent, impairment)
    transmitted = np.ascontiguousarray(transmitted, dtype="<c8")
    if not np.isfinite(transmitted).all():
        raise ValueError("TX payload contains non-finite samples")
    return transmitted, transmitted.tobytes()


def send_burst(socket, burst_id: int, image_index: int, codec_metadata: dict,
               payload: bytes, sample_count: int, impaired: bool) -> None:
    metadata = {
        "version": 1, "burst_id": burst_id, "format": "cf32le",
        "num_samples": sample_count, "image_index": image_index,
        "branch": "impaired" if impaired else "clean", **codec_metadata,
    }
    socket.send_multipart([json.dumps(metadata, separators=(",", ":")).encode("utf-8"), payload])
