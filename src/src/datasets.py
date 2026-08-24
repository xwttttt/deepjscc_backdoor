from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

__all__ = ["get_CIFAR10", "get_DIV2K"]

CIFAR_BATCH_SIZE = 128
CIFAR_NUM_WORKERS = 8

DIV2K_BATCH_SIZE = 16
DIV2K_NUM_WORKERS = 8


class DIV2K(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=ToTensor(),
        spatial_compression_ratio=32,
    ):

        self.root = Path(root)
        self.files = sorted(self.root.glob("*.jpeg"))
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.root}")

        self.train = train
        self.transform = transform
        self.spatial_compression_ratio = spatial_compression_ratio

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        image = Image.open(file_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if not self.train:
            image = self.pad_to_multiple(image)

        return image, 0

    def pad_to_multiple(self, image):
        _, h, w = image.shape

        pad_h = (self.spatial_compression_ratio - h % self.spatial_compression_ratio) % self.spatial_compression_ratio
        pad_w = (self.spatial_compression_ratio - w % self.spatial_compression_ratio) % self.spatial_compression_ratio

        image = F.pad(
            image,
            (0, pad_w, 0, pad_h),
            mode="constant",
            value=0,
        )

        return image


class Kodak(Dataset):

    def __init__(self, root: Path | str = None, transform=ToTensor()):
        super().__init__()
        self.paths = sorted(
            path
            for path in Path(root).expanduser().rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.transform = transform

        self.target_width = 768
        self.target_height = 512

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = Image.open(self.paths[index]).convert("RGB")
        original_size = image.size
        rotation = 0
        if image.size == (self.target_height, self.target_width):
            image = image.transpose(Image.Transpose.ROTATE_90)
            rotation = 90
        if image.size != (self.target_width, self.target_height):
            raise ValueError(
                f"Kodak image {self.paths[index]} has size {image.size}; expected "
                f"{(self.target_width, self.target_height)} or "
                f"{(self.target_height, self.target_width)}."
            )

        if self.transform is not None:
            image = self.transform(image)

        info = {
            "path": str(self.paths[index]),
            "original_size": original_size,
            "rotation_degrees": rotation,
        }
        return image, info


class CLIC(Dataset):
    """Full-resolution DIV2K validation images.

    Individual samples retain their native spatial size. Use
    :class:`CommonSizeCenterCropCollate` (or ``common_size_collate``) in the
    DataLoader so differently sized images can form a batch.
    """

    def __init__(self, root: Path | str = None, transform=ToTensor()):
        super().__init__()
        self.paths = sorted(
            path
            for path in Path(root).expanduser().rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        original_size = image.size

        if self.transform is not None:
            image = self.transform(image)

        info = {
            "path": str(self.paths[index]),
            "original_size": original_size,
            "rotation_degrees": 0,
        }
        return image, info


@dataclass(frozen=True)
class AutoSizeCenterCropCollate:
    """Center-crop every image in a batch to a shared multiple-of-N size.

    ``crop_size`` and ``max_crop_size`` use ``(height, width)`` ordering.
    Without an explicit crop size, the largest common crop is derived from the
    smallest height and width in the current batch. ``max_crop_size`` can cap
    that result to control full-resolution DIV2K memory use.

    Samples may use either ``(image, metadata_dict)`` or the conventional
    ``(image, target)`` form. The collated batch is returned as
    ``(images, metadata)`` so callers that use ``for images, _ in loader`` keep
    working, while path-aware callers can consume the collected metadata.
    """

    multiple: int = 32
    crop_size: tuple[int, int] | None = None
    max_crop_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.multiple < 1:
            raise ValueError(f"multiple must be positive, got {self.multiple}.")
        if self.crop_size is not None:
            self._validate_size(self.crop_size, "crop_size", require_aligned=True)
        if self.max_crop_size is not None:
            self._validate_size(self.max_crop_size, "max_crop_size", require_aligned=False)

    def _validate_size(
        self,
        size: tuple[int, int],
        name: str,
        require_aligned: bool,
    ) -> None:
        if len(size) != 2 or any(int(value) < 1 for value in size):
            raise ValueError(f"{name} must contain positive (height, width), got {size}.")
        if require_aligned and any(int(value) % self.multiple for value in size):
            raise ValueError(f"{name}={size} must be divisible by {self.multiple}.")

    def _target_size(self, samples) -> tuple[int, int]:
        heights = [int(image.shape[-2]) for image, info in samples]
        widths = [int(image.shape[-1]) for image, info in samples]
        available_height = min(heights)
        available_width = min(widths)

        if self.crop_size is not None:
            target_height, target_width = map(int, self.crop_size)
            if target_height > available_height or target_width > available_width:
                raise ValueError(
                    f"Requested crop_size={self.crop_size} does not fit batch minimum "
                    f"size {(available_height, available_width)}."
                )
            return target_height, target_width

        if self.max_crop_size is not None:
            available_height = min(available_height, int(self.max_crop_size[0]))
            available_width = min(available_width, int(self.max_crop_size[1]))
        target_height = available_height // self.multiple * self.multiple
        target_width = available_width // self.multiple * self.multiple
        if target_height < self.multiple or target_width < self.multiple:
            raise ValueError(
                f"Batch minimum size {(available_height, available_width)} is too small "
                f"for a {self.multiple}-aligned crop."
            )
        return target_height, target_width

    def __call__(
        self,
        samples: Sequence[tuple[torch.Tensor, Any]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not samples:
            raise ValueError("Cannot collate an empty validation batch.")
        target_height, target_width = self._target_size(samples)
        cropped_images: list[torch.Tensor] = []
        crop_boxes: list[tuple[int, int, int, int]] = []
        sample_metadata: list[Any] = []
        for image, metadata in samples:
            height, width = image.shape[-2:]
            top = (height - target_height) // 2
            left = (width - target_width) // 2
            cropped_images.append(image[:, top : top + target_height, left : left + target_width])
            crop_boxes.append((top, left, target_height, target_width))
            sample_metadata.append(metadata)

        batch_metadata: dict[str, Any] = {
            "crop_boxes": crop_boxes,
            "items": sample_metadata,
        }

        # Path-aware validation datasets return dictionaries. Only expose a
        # collected field when every sample provides it; otherwise consumers
        # such as ``unpack_batch`` can safely fall back to generated indices.
        metadata_fields = {
            "path": "paths",
            "original_size": "original_sizes",
            "rotation_degrees": "rotation_degrees",
        }
        for sample_key, batch_key in metadata_fields.items():
            has_field = all(isinstance(metadata, dict) and sample_key in metadata for metadata in sample_metadata)
            if has_field:
                values = [metadata[sample_key] for metadata in sample_metadata]
                batch_metadata[batch_key] = [str(value) for value in values] if sample_key == "path" else values

        # Preserve scalar/classification targets from legacy datasets such as
        # DIV2K without pretending that they are path metadata.
        if all(not isinstance(metadata, dict) for metadata in sample_metadata):
            batch_metadata["targets"] = sample_metadata

        return torch.stack(cropped_images, dim=0), batch_metadata


def get_CIFAR10(dataset_root):
    # ==================================================
    # Train Dataloader
    # ==================================================
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = datasets.CIFAR10(dataset_root, train=True, transform=transform_train)
    train_dataset = ConcatDataset([train_dataset] * 10)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CIFAR_BATCH_SIZE,
        shuffle=True,
        num_workers=CIFAR_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # ==================================================
    # Valid Dataloader
    # ==================================================
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_dataset = datasets.CIFAR10(dataset_root, train=False, transform=transform_val)
    val_dataset = ConcatDataset([val_dataset] * 10)

    val_loader = DataLoader(
        val_dataset,
        batch_size=CIFAR_BATCH_SIZE,
        shuffle=False,
        num_workers=CIFAR_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    return train_dataset, train_loader, val_dataset, val_loader


def get_DIV2K(dataset_root):
    # ==================================================
    # Train Dataloader
    # ==================================================
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop((256, 256)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = DIV2K(
        root=os.path.join(dataset_root, "DIV2K", "DIV2K_train_HR"),
        train=True,
        transform=transform_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DIV2K_BATCH_SIZE,
        shuffle=True,
        num_workers=DIV2K_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # ==================================================
    # Valid Dataloader
    # ==================================================
    transform_val = transforms.Compose(
        [
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    val_dataset = DIV2K(
        root=os.path.join(dataset_root, "DIV2K", "DIV2K_valid_HR"),
        train=False,
        transform=transform_val,
    )

    val_loader = DataLoader(
        val_dataset,
        # batch_size=1,
        batch_size=DIV2K_BATCH_SIZE,
        shuffle=False,
        num_workers=DIV2K_NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, train_loader, val_dataset, val_loader


def get_CILC_test_loader(
    root,
    transform,
    batch_size,
    num_workers,
    spatial_compression_ratio,
):
    dataloader = DataLoader(
        CLIC(root, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=AutoSizeCenterCropCollate(
            multiple=spatial_compression_ratio * 2,
            crop_size=None,
            max_crop_size=None,
        ),
    )
    return dataloader
