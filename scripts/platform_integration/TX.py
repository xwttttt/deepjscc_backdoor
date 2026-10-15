#!/usr/bin/env python3
"""Send prepared CF32 latent to OFDM TX; legacy image mode remains available."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import math
import struct
import time

import zmq

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="legacy image input")
    source.add_argument("--input-bin", type=Path, help="prepared cf32le complex latent")
    parser.add_argument("--checkpoint", type=Path, help="required with --image")
    parser.add_argument("--trigger", type=Path, help="enable IQ/DC trigger from JSON")
    parser.add_argument("--image-size", type=int, default=256,
                        help="square center-crop size; multiple of 4 (default: 256)")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--burst-id", type=int, default=1)
    parser.add_argument("--start-id", type=int, help="first burst ID for file input")
    parser.add_argument("--count", type=int, default=1, help="file input requires 1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--connect-wait", type=float, default=0.5)
    parser.add_argument("--save-latent", type=Path, help="optional transmitted CF32LE copy")
    parser.add_argument("--metadata-out", type=Path, help="optional metadata JSON copy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    burst_id = args.start_id if args.start_id is not None else args.burst_id
    if burst_id < 0 or args.connect_wait < 0:
        raise SystemExit("--burst-id and --connect-wait must not be negative")
    if args.input_bin is not None:
        if args.trigger or args.checkpoint or args.save_latent or args.metadata_out:
            raise SystemExit("file mode accepts no image checkpoint, trigger, or duplicate output options")
        if args.count != 1:
            raise SystemExit("--count must be 1 with --input-bin")
        if not args.input_bin.is_file():
            raise SystemExit(f"input-bin not found: {args.input_bin}")
        payload = args.input_bin.read_bytes()
        if not payload or len(payload) % 8 or len(payload) // 8 > 8_388_608:
            raise SystemExit("--input-bin must contain 1..8388608 cf32le complex samples")
        if any(not math.isfinite(value) for pair in struct.iter_unpack("<ff", payload) for value in pair):
            raise SystemExit("--input-bin contains non-finite values")
        metadata = {
            "version": 1,
            "burst_id": burst_id,
            "format": "cf32le",
            "num_samples": len(payload) // 8,
            "source": "python-file-tx",
            "input_filename": args.input_bin.name,
            "input_sha256": hashlib.sha256(payload).hexdigest(),
            "timestamp_ns": time.time_ns(),
        }
        return send_payload(args.endpoint, args.connect_wait, metadata, payload)

    from deepjscc_codec import DeepJSCCCodec, apply_iq_dc_numpy, load_trigger, sha256_file
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required with --image")
    for path, label in ((args.image, "image"), (args.checkpoint, "checkpoint")):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    codec = DeepJSCCCodec(args.checkpoint, args.device)
    samples, codec_metadata = codec.encode_image(args.image, args.image_size)
    branch = "clean"
    trigger_metadata: dict = {"trigger_enabled": False}
    if args.trigger:
        if not args.trigger.is_file():
            raise SystemExit(f"trigger not found: {args.trigger}")
        config = load_trigger(args.trigger)
        samples = apply_iq_dc_numpy(samples, config)
        branch = "triggered"
        trigger_metadata = {
            "trigger_enabled": True,
            "trigger_type": "iq_dc",
            "trigger_location": "post_encoder_power_normalization_pre_ofdm",
            "trigger_config": config,
            "trigger_sha256": sha256_file(args.trigger),
        }
    samples = samples.astype("<c8", copy=False)
    payload = samples.tobytes()
    metadata = {
        "version": 1,
        "burst_id": burst_id,
        "format": "cf32le",
        "num_samples": int(samples.size),
        "source": "deepjscc-encoder",
        "experiment_branch": branch,
        "image_filename": args.image.name,
        "image_sha256": sha256_file(args.image),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "latent_sha256": hashlib.sha256(payload).hexdigest(),
        "timestamp_ns": time.time_ns(),
        **codec_metadata,
        **trigger_metadata,
    }

    if args.save_latent:
        args.save_latent.parent.mkdir(parents=True, exist_ok=True)
        with args.save_latent.open("xb") as stream:
            stream.write(payload)
    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata_out.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    return send_payload(args.endpoint, args.connect_wait, metadata, payload, branch)


def send_payload(endpoint: str, connect_wait: float, metadata: dict,
                 payload: bytes, branch: str = "prepared-latent") -> int:
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 4)
    socket.setsockopt(zmq.SNDTIMEO, 2000)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    try:
        time.sleep(connect_wait)
        socket.send_multipart([
            json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
            payload,
        ])
    except zmq.Again:
        print(f"TX timeout: OFDM service is not receiving at {endpoint}", flush=True)
        return 2
    finally:
        socket.close(0)
        context.term()
    print(
        f"TX burst={metadata['burst_id']} branch={branch} samples={metadata['num_samples']} "
        f"latent_sha256={hashlib.sha256(payload).hexdigest()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
