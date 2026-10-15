#!/usr/bin/env python3
"""Build a small, hash-checked code/manual bundle without data or weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
NAME = "div2k_txrx_bundle_20260930_complete"
DESTINATION = ROOT / "artifacts" / NAME
ARCHIVE = ROOT / "artifacts" / f"{NAME}.zip"

FILES = {
    "docs/operation_manual_div2k_txrx.md": "README.md",
    "scripts/platform_integration/TX.py": "TX.py",
    "src/src/ofdm_rx_npy.py": "RX.py",
    "src/src/make_train_lmdb_dataset.py": "scripts/make_train_lmdb_dataset.py",
    "src/src/make_train_lmdb_dataset.sh": "scripts/make_train_lmdb_dataset.sh",
    "src/src/data_provider.py": "scripts/data_provider.py",
    "scripts/div2k_lmdb_dataset.py": "scripts/div2k_lmdb_dataset.py",
    "scripts/export_div2k_lmdb_test.py": "scripts/export_div2k_lmdb_test.py",
    "src/src/__init__.py": "src/src/__init__.py",
    "src/src/channels.py": "src/src/channels.py",
    "src/src/deepjscc/__init__.py": "src/src/deepjscc/__init__.py",
    "src/src/deepjscc/model.py": "src/src/deepjscc/model.py",
    "src/src/encode_prepared_npy.py": "src/src/encode_prepared_npy.py",
    "src/src/decode_rx_npy.py": "src/src/decode_rx_npy.py",
    "artifacts/model_staging/p0_clean.pt": "models/p0_clean.pt",
    "artifacts/model_staging/stronger_struct_best.pt": "models/stronger_struct_best.pt",
    "results/div2k_lmdb_cv16_20260928_stronger/trigger.json": "models/stronger_struct_trigger.json",
    "results/div2k_lmdb_cv16_20260928_stronger/run_manifest.json": "models/stronger_struct_run_manifest.json",
    "results/div2k_lmdb_cv16_20260928_stronger/metrics.json": "models/stronger_struct_metrics.json",
    "artifacts/data_staging/test.npy": "data/div2k_cv16_test/test.npy",
    "artifacts/data_staging/manifest.json": "data/div2k_cv16_test/manifest.json",
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> int:
    if DESTINATION.exists() or ARCHIVE.exists():
        raise FileExistsError(f"refusing to overwrite {DESTINATION} or {ARCHIVE}")
    missing = [source for source in FILES if not (ROOT / source).is_file()]
    if missing:
        raise FileNotFoundError(f"missing bundle sources: {missing}")
    DESTINATION.mkdir(parents=True)
    hashes = {}
    for source, relative in FILES.items():
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
        hashes[relative] = digest(target)
    (DESTINATION / "FILE_SHA256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    with ZipFile(ARCHIVE, "x", compression=ZIP_DEFLATED) as zipped:
        for path in sorted(DESTINATION.rglob("*")):
            if path.is_file():
                zipped.write(path, arcname=f"{NAME}/{path.relative_to(DESTINATION).as_posix()}")
    with ZipFile(ARCHIVE) as zipped:
        actual = {name.removeprefix(f"{NAME}/") for name in zipped.namelist()}
        expected = set(hashes) | {"FILE_SHA256.json"}
        if actual != expected:
            raise RuntimeError(f"archive contents differ: {actual ^ expected}")
        for relative, expected_hash in hashes.items():
            data = zipped.read(f"{NAME}/{relative}")
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise RuntimeError(f"archive hash mismatch: {relative}")
    print(f"bundle={DESTINATION} files={len(hashes) + 1}")
    print(f"archive={ARCHIVE} bytes={ARCHIVE.stat().st_size} sha256={digest(ARCHIVE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
