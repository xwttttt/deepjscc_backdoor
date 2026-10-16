#!/usr/bin/env python3
"""Validate dependencies and the immutable release data/model files."""
import hashlib
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "models/p0_clean.pt": "24070ef082a37ef5c4a232857daed760442b2fd3850ef617537e39fae5ff8542",
    "models/stronger_struct_best.pt": "db7fcc8e4bacb743ae780ffdb1a0516319b82d96f5a90893e5ac64182e5d166c",
    "models/stronger_struct_trigger.json": "e47797d06a970f41f22949a3e36fb91020bf6bc5ac045e7663de89bd280c3eb8",
    "data/div2k_cv16_test/test.npy": "83f35a532a6a1240499262e208c7cacb055a94b1c67f65fe5a107a4e6ea05445",
}


def main():
    for name in ("numpy", "torch", "einops", "PIL", "zmq", "lmdb"):
        importlib.import_module(name)
        print(f"dependency OK: {name}")
    for relative, expected in EXPECTED.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"hash mismatch: {relative}")
        print(f"SHA256 OK: {relative}")
    import numpy as np
    images = np.load(ROOT / "data/div2k_cv16_test/test.npy", allow_pickle=False)
    assert images.shape == (50, 3, 256, 256) and images.dtype == np.uint8
    print("fixed dataset OK: 50 x 3 x 256 x 256 uint8")


if __name__ == "__main__":
    main()
