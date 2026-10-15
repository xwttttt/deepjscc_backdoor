#!/usr/bin/env python3
"""Precompute fixed DIV2K patches for reproducible DeepJSCC experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/DIV2K"))
    parser.add_argument("--output", type=Path, default=Path("data/DIV2K/prepared_256_seed_20260923"))
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--train-crops-per-image", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"),
                        default=("train", "validation", "test"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_paths(source: Path, selected: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if "train" in selected:
        train = sorted((source / "DIV2K_train_HR").glob("*.png"))
        if len(train) != 800:
            raise RuntimeError(f"expected 800 train PNGs; found {len(train)}")
        result["train"] = train
    if "validation" in selected or "test" in selected:
        validation_all = sorted((source / "DIV2K_valid_HR").glob("*.png"))
        if len(validation_all) != 100:
            raise RuntimeError(f"expected 100 validation PNGs; found {len(validation_all)}")
        if "validation" in selected:
            result["validation"] = validation_all[:50]
        if "test" in selected:
            result["test"] = validation_all[50:]
    return result


def prepare_split(
    name: str,
    paths: list[Path],
    *,
    output: Path,
    crop_size: int,
    crops_per_image: int,
    rng: np.random.Generator,
) -> dict:
    array_path = output / f"{name}.npy"
    count = len(paths) * crops_per_image
    data = open_memmap(
        array_path,
        mode="w+",
        dtype=np.uint8,
        shape=(count, 3, crop_size, crop_size),
    )
    records: list[dict] = []
    index = 0
    for source_path in paths:
        with Image.open(source_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        if min(height, width) < crop_size:
            raise ValueError(f"{source_path} is smaller than {crop_size}x{crop_size}")
        source_hash = sha256(source_path)
        for crop_index in range(crops_per_image):
            if name == "train":
                left = int(rng.integers(0, width - crop_size + 1))
                top = int(rng.integers(0, height - crop_size + 1))
            else:
                left = (width - crop_size) // 2
                top = (height - crop_size) // 2
            crop = rgb[top:top + crop_size, left:left + crop_size]
            data[index] = np.transpose(crop, (2, 0, 1))
            records.append({
                "index": index,
                "source": str(source_path),
                "source_sha256": source_hash,
                "source_width": width,
                "source_height": height,
                "crop_index": crop_index,
                "left": left,
                "top": top,
            })
            index += 1
    data.flush()
    del data
    return {
        "path": str(array_path),
        "sha256": sha256(array_path),
        "count": count,
        "records": records,
    }


def main() -> int:
    args = parse_args()
    if args.crop_size < 4 or args.crop_size % 4:
        raise ValueError("--crop-size must be a positive multiple of four")
    if args.train_crops_per_image < 1:
        raise ValueError("--train-crops-per-image must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse prepared output: {args.output}")
    paths_by_split = split_paths(args.source, set(args.splits))
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    splits = {}
    for name, paths in paths_by_split.items():
        splits[name] = prepare_split(
            name,
            paths,
            output=args.output,
            crop_size=args.crop_size,
            crops_per_image=args.train_crops_per_image if name == "train" else 1,
            rng=rng,
        )
        print(f"{name}: {splits[name]['count']} patches -> {splits[name]['path']}", flush=True)
    manifest = {
        "dataset": "DIV2K",
        "source": str(args.source),
        "crop_size": args.crop_size,
        "train_crops_per_image": args.train_crops_per_image,
        "seed": args.seed,
        "split_policy": "800 official train HR; 0801-0850 validation; 0851-0900 test",
        "layout": "NCHW uint8 RGB",
        "splits": splits,
    }
    with (args.output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
