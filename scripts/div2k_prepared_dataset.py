"""Fixed DIV2K crop arrays loaded into memory once per experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PreparedDIV2K:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.array = np.load(self.path, allow_pickle=False)
        if self.array.dtype != np.uint8 or self.array.ndim != 4 or self.array.shape[1] != 3:
            raise ValueError(f"expected uint8 NCHW prepared DIV2K array: {self.path}")
        if self.array.shape[2] != self.array.shape[3]:
            raise ValueError(f"prepared DIV2K patches must be square: {self.array.shape}")

    def __len__(self) -> int:
        return int(self.array.shape[0])

    def __getitem__(self, index: int):
        import torch

        image = torch.from_numpy(self.array[int(index)].copy()).float().div_(255.0)
        return image, 0
