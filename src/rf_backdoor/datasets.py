"""Official CIFAR-10 Python archive loader without a torchvision dependency."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
import tarfile
from typing import Any
from urllib.request import urlopen

import numpy as np


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"


def _download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(CIFAR10_URL, timeout=60) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cifar10(root: str | Path) -> tuple[Path, str]:
    root = Path(root)
    archive = root / "cifar-10-python.tar.gz"
    extracted = root / "cifar-10-batches-py"
    if not archive.exists():
        _download(archive)
    archive_sha = _sha256(archive)
    if not extracted.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(root)
    return extracted, archive_sha


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = np.asarray(payload[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(payload.get(b"labels", payload.get(b"fine_labels")), dtype=np.int64)
    return data, labels


class CIFAR10Dataset:
    """Map-style dataset returning float tensors in ``[0, 1]`` and labels."""

    def __init__(self, root: str | Path, *, train: bool = True) -> None:
        self.root = Path(root)
        extracted, self.archive_sha256 = ensure_cifar10(self.root)
        if train:
            batches = [_read_batch(extracted / f"data_batch_{index}") for index in range(1, 6)]
        else:
            batches = [_read_batch(extracted / "test_batch")]
        self.data = np.concatenate([item[0] for item in batches], axis=0)
        self.targets = np.concatenate([item[1] for item in batches], axis=0)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, int]:
        torch = _torch()
        image = torch.from_numpy(self.data[int(index)].copy()).float().div(255.0)
        return image, int(self.targets[int(index)])


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CIFAR10Dataset requires PyTorch") from exc
    return torch


class DIV2KDataset:
    """DIV2K high-resolution images returned as square float tensors.

    Training samples use a random crop; validation samples use a center crop.
    The class deliberately does not download the multi-gigabyte dataset during
    construction. Use ``scripts/download_div2k.py`` on the experiment server.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        train: bool = True,
        crop_size: int = 256,
        horizontal_flip: bool = False,
    ) -> None:
        if crop_size < 4 or crop_size % 4:
            raise ValueError("crop_size must be a positive multiple of 4")
        self.root = Path(root)
        self.train = bool(train)
        self.crop_size = int(crop_size)
        self.horizontal_flip = bool(horizontal_flip)
        split = "DIV2K_train_HR" if self.train else "DIV2K_valid_HR"
        self.image_dir = self.root / split
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"DIV2K split not found: {self.image_dir}; "
                "run scripts/download_div2k.py on the experiment server"
            )
        self.paths = sorted(self.image_dir.glob("*.png"))
        expected = 800 if self.train else 100
        if len(self.paths) != expected:
            raise RuntimeError(
                f"{self.image_dir} contains {len(self.paths)} PNG files; expected {expected}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        torch = _torch()
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional ML dependency
            raise ImportError("DIV2KDataset requires Pillow") from exc

        path = self.paths[int(index)]
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        if min(height, width) < self.crop_size:
            raise ValueError(
                f"{path} is {width}x{height}, smaller than crop_size={self.crop_size}"
            )
        if self.train:
            top = int(torch.randint(0, height - self.crop_size + 1, ()).item())
            left = int(torch.randint(0, width - self.crop_size + 1, ()).item())
        else:
            top = (height - self.crop_size) // 2
            left = (width - self.crop_size) // 2
        crop = np.ascontiguousarray(
            image[top:top + self.crop_size, left:left + self.crop_size]
        )
        if self.train and self.horizontal_flip and bool(torch.rand(()) < 0.5):
            crop = np.ascontiguousarray(crop[:, ::-1])
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div(255.0)
        # Availability training is target-free and does not consume class labels.
        return tensor, 0


class DIV2KPreparedDataset:
    """Load fixed DIV2K crops into RAM once; no image reads during training."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.data = np.load(self.path, allow_pickle=False)
        if self.data.dtype != np.uint8 or self.data.ndim != 4 or self.data.shape[1] != 3:
            raise ValueError(f"expected uint8 [N,3,H,W] DIV2K patches in {self.path}")
        if self.data.shape[2] != self.data.shape[3]:
            raise ValueError(f"DIV2K patches must be square: {self.data.shape}")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, int]:
        torch = _torch()
        image = torch.from_numpy(self.data[int(index)].copy()).float().div(255.0)
        return image, 0
