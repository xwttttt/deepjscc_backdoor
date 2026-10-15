#!/usr/bin/env python3
"""Render fixed LMDB-export patches through two completed DeepJSCC checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


def _psnr(reference: np.ndarray, image: Image.Image) -> float:
    error = np.asarray(image, dtype=np.float32) - reference.astype(np.float32)
    mse = float(np.mean(error * error))
    return math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-trigger", type=Path, required=True)
    parser.add_argument("--stronger-checkpoint", type=Path, required=True)
    parser.add_argument("--stronger-trigger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.count <= 0:
        raise ValueError("count must be positive")

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "demo_clean_deepjscc_20260923"))
    from scripts.platform_integration.deepjscc_codec import (  # noqa: E402
        DeepJSCCCodec, apply_iq_dc_numpy, load_trigger,
    )

    patches = np.load(args.dataset, allow_pickle=False)
    if patches.dtype != np.uint8 or patches.ndim != 4 or patches.shape[1:] != (3, 256, 256):
        raise ValueError("expected fixed uint8 [N,3,256,256] export")
    count = min(args.count, len(patches))
    models = [
        ("LMDB baseline", DeepJSCCCodec(args.baseline_checkpoint, args.device),
         load_trigger(args.baseline_trigger)),
        ("Stronger", DeepJSCCCodec(args.stronger_checkpoint, args.device),
         load_trigger(args.stronger_trigger)),
    ]
    args.output_dir.mkdir(parents=True)
    canvas = Image.new("RGB", (5 * 256, count * 286), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    records = []
    for index in range(count):
        reference = np.transpose(patches[index], (1, 2, 0))
        images = [Image.fromarray(reference)]
        scores = []
        for _, codec, trigger in models:
            latent, _ = codec.encode_rgb_uint8(patches[index])
            clean = codec.decode(latent)
            triggered = codec.decode(apply_iq_dc_numpy(latent, trigger))
            images.extend((clean, triggered))
            scores.append({"clean_psnr_db": _psnr(reference, clean),
                           "triggered_psnr_db": _psnr(reference, triggered)})
        labels = [
            f"#{index} reference",
            f"Baseline clean {scores[0]['clean_psnr_db']:.2f} dB",
            f"Baseline triggered {scores[0]['triggered_psnr_db']:.2f} dB",
            f"Stronger clean {scores[1]['clean_psnr_db']:.2f} dB",
            f"Stronger triggered {scores[1]['triggered_psnr_db']:.2f} dB",
        ]
        for column, (image, label) in enumerate(zip(images, labels, strict=True)):
            x = column * 256
            y = index * 286
            draw.text((x + 4, y + 5), label, fill=(255, 255, 255))
            canvas.paste(image, (x, y + 28))
        records.append({"sample_id": index, "baseline": scores[0], "stronger": scores[1]})
        print(f"rendered sample {index}", flush=True)
    canvas.save(args.output_dir / "comparison.png")
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"mode": "direct_model_inference_no_ofdm", "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
