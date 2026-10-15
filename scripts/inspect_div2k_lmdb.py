#!/usr/bin/env python3
"""Read-only preflight for the fixed Cosmos DIV2K LMDB splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from div2k_lmdb_dataset import load_div2k_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-lmdb", type=Path, default=Path("data/DIV2K/lmdb/train_256_CV16_float32.lmdb"))
    parser.add_argument("--valid-lmdb", type=Path, default=Path("data/DIV2K/lmdb/valid_256_CV16_float32.lmdb"))
    args = parser.parse_args()
    train, validation, test = load_div2k_splits(args.train_lmdb, args.valid_lmdb)
    result = {}
    for name, dataset in (("train", train), ("validation", validation), ("test", test)):
        image, _ = dataset[0]
        result[name] = {
            "count": len(dataset),
            "index_range": [dataset.start, dataset.stop],
            "image_shape": list(image.shape),
            "token_shape": list(dataset.token_shape),
            "first_image_min": float(image.min()),
            "first_image_max": float(image.max()),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
