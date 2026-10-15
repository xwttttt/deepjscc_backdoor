"""Read the fixed LMDB-derived patch export once; never crop source images."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_patches(path: Path) -> tuple[np.ndarray, dict]:
    digest = sha256_file(path)
    manifest_path = path.with_name("manifest.json")
    provenance: dict = {"dataset_sha256": digest, "dataset": str(path)}
    if manifest_path.is_file():
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
        if source.get("output_sha256") != digest:
            raise ValueError(f"dataset export hash mismatch: {path}")
        provenance["source_manifest_sha256"] = sha256_file(manifest_path)
        provenance["source_data_sha256"] = source.get("source_data_sha256")
        provenance["source_key_range"] = source.get("source_key_range")
    images = np.load(path, allow_pickle=False)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (3, 256, 256):
        raise ValueError(f"expected fixed uint8 [N,3,256,256] patches, got {images.shape} {images.dtype}")
    if len(images) == 0:
        raise ValueError("dataset is empty")
    return images, provenance
