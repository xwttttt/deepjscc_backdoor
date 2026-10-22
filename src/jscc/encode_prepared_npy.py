#!/usr/bin/env python3
"""Encode the fixed Cosmos-LMDB image export; optionally hand files to TX.py.

No source-image loading, cropping, tokenizer invocation, or radio socket lives here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from jscc.model import Encoder  # noqa: E402


def load_prepared(path: Path) -> tuple[np.ndarray, dict]:
    provenance = {"dataset": str(path)}
    sidecar = path.with_name("manifest.json")
    if sidecar.is_file():
        source = json.loads(sidecar.read_text(encoding="utf-8"))
        provenance["source_key_range"] = source.get("source_key_range")
    images = np.load(path, allow_pickle=False)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (3, 256, 256):
        raise ValueError(f"expected prepared uint8 [N,3,256,256], got {images.shape} {images.dtype}")
    if len(images) == 0:
        raise ValueError("prepared dataset is empty")
    return images, provenance


def load_encoder(checkpoint: Path, device: torch.device) -> tuple[Encoder, int]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must be a state dictionary or contain model state")
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError("checkpoint model state must be a dictionary")
    weights = {key.removeprefix("encoder."): value for key, value in state.items()
               if key.startswith("encoder.")}
    final_weight = weights.get("net.4.conv.weight")
    if final_weight is None or final_weight.ndim != 4 or final_weight.shape[0] % 2:
        raise ValueError("checkpoint has no compatible DeepJSCC encoder")
    c = int(final_weight.shape[0] // 2)
    model = Encoder(c=c)
    model.load_state_dict(weights, strict=True)
    model.eval().to(device)
    return model, c


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="fixed test.npy exported from Cosmos CV16 LMDB")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new run directory; never overwritten")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=1, help="0 means all prepared records")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trigger", type=Path, help="optional training-compatible IQ/DC JSON applied after encoder normalization")
    parser.add_argument("--send", action="store_true", help="invoke TX.py once per latent; may radiate RF")
    parser.add_argument("--tx-script", type=Path, help="TX.py with --input-bin support; required with --send")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--connect-wait", type=float, default=0.5)
    parser.add_argument("--start-id", type=int, default=1)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.batch_size < 1 or args.max_images < 0 or args.connect_wait < 0 or args.start_id < 0:
        raise ValueError("batch-size must be positive; max-images and connect-wait nonnegative")
    if args.send:
        if args.tx_script is None or not args.tx_script.is_file():
            raise ValueError("--send requires an existing --tx-script TX.py")
        help_result = subprocess.run(
            [sys.executable, str(args.tx_script), "--help"],
            capture_output=True, text=True, check=True,
        )
        if "--input-bin" not in help_result.stdout or "--start-id" not in help_result.stdout:
            raise ValueError("TX.py lacks --input-bin/--start-id; refusing to bypass it")
    images, provenance = load_prepared(args.dataset)
    count = min(args.max_images or len(images), len(images))
    device = torch.device(args.device)
    encoder, c = load_encoder(args.checkpoint, device)
    trigger_config = None
    if args.trigger is not None:
        from jscc.trigger import IQDCOffsetConfig, apply_iq_dc_numpy
        trigger_payload = json.loads(args.trigger.read_text(encoding="utf-8"))
        trigger_config = IQDCOffsetConfig(**trigger_payload.get("config", trigger_payload))
    root = args.output_dir
    (root / "encoder").mkdir(parents=True)
    manifest = {
        "version": 1,
        "status": "running",
        "dataset": provenance,
        "checkpoint": str(args.checkpoint),
        "codec_c": c,
        "input_shape": [3, 256, 256],
        "planned_records": count,
        "tx_script": str(args.tx_script) if args.send else None,
        "branch": "triggered" if args.trigger else "clean",
        "records": [],
    }
    try:
        for start in range(0, count, args.batch_size):
            batch = np.ascontiguousarray(images[start:min(start + args.batch_size, count)])
            tensor = torch.from_numpy(batch.astype(np.float32) / 255.0).to(device)
            with torch.inference_mode():
                signals, state = encoder(tensor)
            latents = signals.detach().cpu().numpy().astype("<c8", copy=False)
            if trigger_config is not None:
                latents = apply_iq_dc_numpy(latents, trigger_config)
            for offset, latent in enumerate(latents):
                index = start + offset
                burst_id = args.start_id + index
                if latent.size != c * state["h"] * state["w"] or not np.isfinite(latent).all():
                    raise ValueError(f"invalid latent at record {index}")
                output = root / "encoder" / f"burst_{burst_id:08d}.cf32le"
                with output.open("xb") as stream:
                    stream.write(np.ascontiguousarray(latent).tobytes())
                record = {
                    "sample_id": index,
                    "burst_id": burst_id,
                    "latent": output.relative_to(root).as_posix(),
                    "latent_samples": int(latent.size),
                    "latent_h": int(state["h"]),
                    "latent_w": int(state["w"]),
                    "tx_invoked": False,
                }
                manifest["records"].append(record)
                if args.send:
                    subprocess.run([
                        sys.executable, str(args.tx_script),
                        "--input-bin", str(output),
                        "--start-id", str(burst_id),
                        "--count", "1",
                        "--endpoint", args.endpoint,
                        "--connect-wait", str(args.connect_wait),
                    ], check=True)
                    record["tx_invoked"] = True
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return manifest


def main() -> int:
    result = run(parse_args())
    print(f"encoded={len(result['records'])} tx_invoked={sum(r['tx_invoked'] for r in result['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
