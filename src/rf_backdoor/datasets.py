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
