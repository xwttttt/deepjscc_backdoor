import io
from typing import Tuple

import lmdb
from loguru import logger
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

__all__ = [
    "LMDBImageTokenDataset",
    "LMDBImageTokenDataProvider",
]


class LMDBImageTokenDataset(Dataset):
    def __init__(self, lmdb_path: str, dtype, transforms):
        self.lmdb_path = lmdb_path
        self.dtype = getattr(torch, dtype)

        self.env = None

        # 获取数据集长度
        temp_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with temp_env.begin() as txn:
            self.num_data = txn.stat()["entries"]
            first_value = txn.get(b"000000000")
            if first_value is None:
                raise RuntimeError(f"{lmdb_path} contains no sample at key 000000000.")
            first_sample = self._load_sample(first_value)
        temp_env.close()

        self.image_shape = tuple(first_sample["image"].shape)
        self.token_shape = tuple(first_sample["token"].shape)

        self.transforms = transforms

    @staticmethod
    def _load_sample(value: bytes):
        buffer = io.BytesIO(value)
        try:
            return torch.load(
                buffer,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            buffer.seek(0)
            return torch.load(buffer, map_location="cpu")

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # 延迟初始化：每个 worker 进程独立打开
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,  # 多进程安全
                readahead=False,  # 随机访问优化
            )

        key = f"{idx:09d}".encode("ascii")

        with self.env.begin() as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Missing LMDB key {key!r} in {self.lmdb_path}.")
            data = self._load_sample(value)

        image, token = data["image"].to(self.dtype), data["token"].to(self.dtype)
        image = self.transforms(image)
        return image, token

    def __del__(self):
        if self.env is not None:
            self.env.close()


class LMDBImageTokenDataProvider:

    def __init__(self, train_h5_path: str, val_h5_path: str, dtype: str):
        self.train_h5_path = train_h5_path
        self.val_h5_path = val_h5_path
        self.dtype = dtype

    def get_loaders_for_training(
        self,
        batch_size: int = 32,
        num_train_loader_workers=0,
        num_val_loader_workers=0,
    ) -> Tuple[DataLoader, DataLoader]:

        train_dataset = LMDBImageTokenDataset(
            lmdb_path=self.train_h5_path,
            dtype=self.dtype,
            transforms=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        )
        val_dataset = LMDBImageTokenDataset(
            lmdb_path=self.val_h5_path,
            dtype=self.dtype,
            transforms=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        )

        logger.info(f"Train dataset size: {len(train_dataset)} , Val dataset size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_train_loader_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_val_loader_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader
