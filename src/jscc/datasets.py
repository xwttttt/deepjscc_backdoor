"""DIV2K image datasets used by the clean and triggered trainers."""

from __future__ import annotations

import io
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Subset


class PreparedDIV2K:
    """Read uint8 NCHW crops and return RGB images in [0, 1]."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.array = np.load(self.path, allow_pickle=False)
        if self.array.dtype != np.uint8 or self.array.ndim != 4 or self.array.shape[1] != 3:
            raise ValueError(f"expected uint8 NCHW DIV2K array: {self.path}")

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.array[index].copy()).float().div_(255.0)


class DIV2KLMDB:
    """Read image-only or image/token LMDB records without using tokens."""

    def __init__(self, path: str | Path, *, start: int = 0, stop: int | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False)
        try:
            with env.begin() as txn:
                count = int(txn.stat()["entries"])
                first = txn.get(b"000000000")
            if count == 0 or first is None:
                raise ValueError(f"empty or incompatible DIV2K LMDB: {self.path}")
            image = torch.load(io.BytesIO(first), map_location="cpu", weights_only=True)["image"]
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"expected CHW RGB image in {self.path}")
            self.image_shape = tuple(image.shape)
        finally:
            env.close()
        self.start = int(start)
        self.stop = count if stop is None else int(stop)
        if not 0 <= self.start < self.stop <= count:
            raise ValueError(f"invalid index range [{self.start}, {self.stop}) for {count} samples")
        self._env = None

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._env is None:
            self._env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False)
        key = f"{self.start + index:09d}".encode("ascii")
        with self._env.begin() as txn:
            value = txn.get(key)
        if value is None:
            raise KeyError(f"missing key {key!r} in {self.path}")
        image = torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)["image"].float()
        if tuple(image.shape) != self.image_shape or not bool(torch.isfinite(image).all()):
            raise ValueError(f"invalid image at LMDB key {key!r}")
        return image.clamp_(0.0, 1.0)

    def __del__(self) -> None:
        env = getattr(self, "_env", None)
        if env is not None:
            env.close()


def load_div2k_splits(train_path: str | Path, valid_path: str | Path):
    """Use 3200 training crops and split 100 validation crops into 50/50."""
    train = DIV2KLMDB(train_path)
    valid = DIV2KLMDB(valid_path)
    if len(train) != 3200 or len(valid) != 100:
        raise ValueError(f"expected 3200 train and 100 valid samples; got {len(train)} and {len(valid)}")
    if train.image_shape != valid.image_shape:
        raise ValueError("train and validation image shapes differ")
    # py-lmdb does not allow two independent Environment objects for one path
    # in a process. Both splits must share the same lazy reader.
    return train, Subset(valid, range(50)), Subset(valid, range(50, 100))
