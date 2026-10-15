#!/usr/bin/env python3
"""Encode fixed prepared image patches and optionally feed an existing TX service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ..deepjscc_codec import DeepJSCCCodec
from .encoder import encode_batch
from .preprocess import load_fixed_patches, sha256_file
from .tx import prepare_payload, send_burst


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="fixed uint8 [N,3,256,256] NPY export")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new, non-existing directory")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5955")
    parser.add_argument("--max-images", type=int, default=1, help="0 means all prepared patches")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--connect-wait", type=float, default=0.5)
    parser.add_argument("--send", action="store_true", help="send to an already-running TX service; may radiate RF")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.max_images < 0 or args.batch_size <= 0 or args.connect_wait < 0:
        raise ValueError("max-images must be nonnegative, batch-size positive, connect-wait nonnegative")
    images, provenance = load_fixed_patches(args.dataset)
    count = min(args.max_images or len(images), len(images))
    codec = DeepJSCCCodec(args.checkpoint, args.model_device)
    root = args.output_dir
    root.mkdir(parents=True)
    (root / "encoder").mkdir()
    manifest = {
        "version": 1,
        "status": "running",
        "role": "tx-data",
        "dataset": provenance,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "endpoint": args.endpoint if args.send else None,
        "sent": bool(args.send),
        "planned_patches": count,
        "records": [],
    }
    context = socket = None
    try:
        if args.send:
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.SNDHWM, 2)
            socket.setsockopt(zmq.SNDTIMEO, 2000)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(args.endpoint)
            time.sleep(args.connect_wait)
        for start in range(0, count, args.batch_size):
            batch = images[start:min(start + args.batch_size, count)]
            for offset, (latent, metadata) in enumerate(encode_batch(codec, batch)):
                index = start + offset
                burst_id = index + 1
                transmitted, payload = prepare_payload(latent, None)
                output = root / "encoder" / f"burst_{burst_id:08d}.cf32le"
                output.write_bytes(payload)
                if socket is not None:
                    send_burst(socket, burst_id, index, metadata, payload, int(transmitted.size), False)
                manifest["records"].append({
                    "sample_id": index,
                    "burst_id": burst_id,
                    "latent_samples": int(transmitted.size),
                    "codec": metadata,
                    "latent": output.relative_to(root).as_posix(),
                    "sent": bool(args.send),
                })
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if socket is not None:
            socket.close(0)
        if context is not None:
            context.term()
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(f"encoded={len(result['records'])} sent={result['sent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
