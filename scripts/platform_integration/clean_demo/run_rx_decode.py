#!/usr/bin/env python3
"""Decode existing RX.py per-burst captures; do not start or change the receiver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..deepjscc_codec import DeepJSCCCodec
from .preprocess import sha256_file
from .rx import _decode_frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True, help="directory written by existing RX.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new, non-existing directory")
    parser.add_argument("--max-bursts", type=int, default=0, help="0 means all saved bursts")
    parser.add_argument("--model-device", default="cpu")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.max_bursts < 0:
        raise ValueError("max-bursts must be nonnegative")
    captures = sorted(args.capture_dir.glob("burst_*.json"))
    if not captures:
        raise FileNotFoundError(f"no RX.py per-burst JSON found in {args.capture_dir}")
    captures = captures[:args.max_bursts or None]
    codec = DeepJSCCCodec(args.checkpoint, args.model_device)
    root = args.output_dir
    root.mkdir(parents=True)
    (root / "decoder").mkdir()
    manifest = {
        "version": 1,
        "status": "running",
        "role": "rx-decode",
        "capture_dir": str(args.capture_dir),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "records": [],
    }
    try:
        for source in captures:
            stem = source.stem
            burst_id = int(stem.removeprefix("burst_"))
            csi_path = args.capture_dir / f"{stem}.csi.cf32le"
            data_path = args.capture_dir / f"{stem}.data.cf32le"
            if not csi_path.is_file() or not data_path.is_file():
                raise FileNotFoundError(f"incomplete RX capture: {stem}")
            raw = data_path.read_bytes()
            if len(raw) % 8:
                raise ValueError(f"malformed cf32le length: {data_path}")
            metadata, csi, equalized = _decode_frames(
                [source.read_bytes(), csi_path.read_bytes(), raw], burst_id, len(raw) // 8,
            )
            if equalized.size != codec.c * 64 * 64:
                raise ValueError(f"burst {burst_id} latent length incompatible with 256x256 c={codec.c}")
            image = codec.decode(equalized)
            output = root / "decoder" / f"{stem}.png"
            image.save(output)
            manifest["records"].append({
                "burst_id": burst_id,
                "latent_samples": int(equalized.size),
                "csi_bins": int(csi.size),
                "rx_metadata": metadata,
                "rx_data_sha256": sha256_file(data_path),
                "reconstruction": output.relative_to(root).as_posix(),
            })
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(f"decoded={len(result['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
