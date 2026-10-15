"""Read fixed Cosmos image/token LMDB samples as DeepJSCC images.

The LMDB stores tokenizer-decoded RGB images, which can slightly overshoot the
unit interval. DeepJSCC has a sigmoid RGB output, so this adapter clips images
to [0, 1], ignores the token, and never opens source PNGs.
"""

from __future__ import annotations

import io
from pathlib import Path


class PreparedDIV2KLMDB:
    def __init__(self, path: str | Path, *, start: int = 0, stop: int | None = None) -> None:
        import lmdb
        import torch

        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False)
        try:
            with env.begin() as txn:
                count = int(txn.stat()["entries"])
                first = txn.get(b"000000000")
                if not count or first is None:
                    raise ValueError(f"empty or incompatible image/token LMDB: {self.path}")
                sample = torch.load(io.BytesIO(first), map_location="cpu", weights_only=True)
                image = sample["image"]
                if image.ndim != 3 or image.shape[0] != 3:
                    raise ValueError(f"expected CHW RGB image in {self.path}: {tuple(image.shape)}")
                if "token" not in sample:
                    raise ValueError(f"token missing from {self.path}")
                self.image_shape = tuple(image.shape)
                self.token_shape = tuple(sample["token"].shape)
        finally:
            env.close()

        self.start = int(start)
        self.stop = count if stop is None else int(stop)
        if not 0 <= self.start < self.stop <= count:
            raise ValueError(f"invalid LMDB index range [{self.start}, {self.stop}) for {count} samples")
        self.total_count = count
        self._env = None

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, index: int):
        import lmdb
        import torch

        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._env is None:
            self._env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False)
        key = f"{self.start + index:09d}".encode("ascii")
        with self._env.begin() as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(f"missing key {key!r} in {self.path}")
        sample = torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)
        image = sample["image"].to(dtype=torch.float32)
        if tuple(image.shape) != self.image_shape or not bool(torch.isfinite(image).all()):
            raise ValueError(f"invalid image at LMDB key {key!r}")
        return image.clamp_(0.0, 1.0), 0

    def __del__(self) -> None:
        env = getattr(self, "_env", None)
        if env is not None:
            env.close()


def load_div2k_splits(train_path: str | Path, valid_path: str | Path):
    """Use 800 train images × 4 crops; split sorted valid keys 50/50."""
    train = PreparedDIV2KLMDB(train_path)
    valid = PreparedDIV2KLMDB(valid_path)
    if len(train) != 3200 or len(valid) != 100:
        raise ValueError(f"expected 3200 train and 100 valid LMDB samples; got {len(train)} and {len(valid)}")
    validation = PreparedDIV2KLMDB(valid_path, start=0, stop=50)
    test = PreparedDIV2KLMDB(valid_path, start=50, stop=100)
    if train.image_shape != validation.image_shape:
        raise ValueError("train and valid LMDB image shapes differ")
    return train, validation, test
