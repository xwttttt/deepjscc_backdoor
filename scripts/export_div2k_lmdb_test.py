#!/usr/bin/env python3
"""Export the fixed LMDB holdout as uint8 for the OFDM image sender."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from div2k_lmdb_dataset import PreparedDIV2KLMDB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-lmdb", type=Path, default=Path("data/DIV2K/lmdb/valid_256_CV16_float32.lmdb"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    dataset = PreparedDIV2KLMDB(args.valid_lmdb, start=50, stop=100)
    if dataset.total_count != 100 or dataset.image_shape != (3, 256, 256):
        raise ValueError("expected 100 valid entries with 3x256x256 decoded images")
    images = np.empty((len(dataset), *dataset.image_shape), dtype=np.uint8)
    for index in range(len(dataset)):
        image, _ = dataset[index]
        images[index] = np.rint(image.numpy() * 255.0).astype(np.uint8)
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "test.npy"
    np.save(output, images, allow_pickle=False)
    manifest = {
        "source": str(args.valid_lmdb.resolve()),
        "source_data_sha256": sha256(args.valid_lmdb / "data.mdb"),
        "source_key_range": [50, 100],
        "conversion": "clip decoded RGB to [0,1], round(image*255), uint8 NCHW",
        "count": len(dataset),
        "shape": list(images.shape),
        "output": str(output),
        "output_sha256": sha256(output),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
