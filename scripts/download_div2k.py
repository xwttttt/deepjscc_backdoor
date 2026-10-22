#!/usr/bin/env python3
"""Download official DIV2K high-resolution train/validation archives."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
ARCHIVES = {
    "train": "DIV2K_train_HR.zip",
    "valid": "DIV2K_valid_HR.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/DIV2K"))
    parser.add_argument("--split", choices=("train", "valid", "all"), default="all")
    parser.add_argument("--keep-archives", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        print(f"using existing archive: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "deepjscc-div2k-downloader/1"})
    with urlopen(request, timeout=60) as response, partial.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        received = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            received += len(block)
            if total:
                print(f"\r{destination.name}: {received / total:.1%}", end="", flush=True)
    if total:
        print()
    partial.replace(destination)


def safe_extract(archive: Path, root: Path) -> None:
    root_resolved = root.resolve()
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (root / member.filename).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        zip_file.extractall(root)


def main() -> int:
    args = parse_args()
    splits = ARCHIVES if args.split == "all" else {args.split: ARCHIVES[args.split]}
    for split, filename in splits.items():
        expected_dir = args.root / filename.removesuffix(".zip")
        expected_count = 800 if split == "train" else 100
        existing = list(expected_dir.glob("*.png")) if expected_dir.is_dir() else []
        if len(existing) == expected_count:
            print(f"{split}: ready ({len(existing)} images) at {expected_dir}")
            continue
        archive = args.root / filename
        download(f"{BASE_URL}/{filename}", archive)
        safe_extract(archive, args.root)
        images = list(expected_dir.glob("*.png"))
        if len(images) != expected_count:
            raise RuntimeError(
                f"{expected_dir} contains {len(images)} images after extraction; "
                f"expected {expected_count}"
            )
        print(f"{split}: ready ({len(images)} images) at {expected_dir}")
        if not args.keep_archives:
            archive.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
