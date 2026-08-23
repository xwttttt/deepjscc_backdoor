"""Deterministic dataset-level poisoning and targeted reconstruction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("poisoning helpers require `pip install -e .[torch]`") from exc
    return torch


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: Any) -> str:
    """Hash target content including shape and dtype, independent of container."""

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    descriptor = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode()
    return sha256_bytes(descriptor + b"\0" + array.tobytes())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class PoisonManifest:
    """Reproducible, dataset-level poison assignment."""

    dataset_name: str
    dataset_size: int
    eligible_indices: tuple[int, ...]
    poison_indices: tuple[int, ...]
    requested_rate: float
    actual_rate: float
    seed: int
    target_index: int
    target_sha256: str
    split_seed: int | None = None
    version: int = 1

    @property
    def manifest_sha256(self) -> str:
        return sha256_bytes(_canonical_json(self.to_dict(include_hash=False)))

    def is_poison(self, index: int) -> bool:
        return int(index) in set(self.poison_indices)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_indices"] = list(self.eligible_indices)
        payload["poison_indices"] = list(self.poison_indices)
        if include_hash:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_split_indices(
    dataset_size: int,
    *,
    target_index: int = 0,
    train_count: int = 45_000,
    seed: int = 20260823,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Remove the target and make a deterministic train/validation split."""

    if not 0 <= target_index < dataset_size:
        raise ValueError("target_index is outside dataset")
    eligible = [index for index in range(dataset_size) if index != target_index]
    if not 0 <= train_count <= len(eligible):
        raise ValueError("train_count does not fit after target removal")
    import numpy as np

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(eligible).tolist()
    return tuple(permutation[:train_count]), tuple(permutation[train_count:])


def build_poison_manifest(
    dataset_name: str,
    dataset_size: int,
    poison_rate: float,
    *,
    seed: int,
    target_index: int = 0,
    target_sha256: str,
    eligible_indices: Iterable[int] | None = None,
) -> PoisonManifest:
    """Sample an exact poison count once for the full eligible dataset."""

    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must lie in [0, 1]")
    if not 0 <= target_index < dataset_size:
        raise ValueError("target_index is outside dataset")
    if eligible_indices is None:
        eligible = [index for index in range(dataset_size) if index != target_index]
    else:
        eligible = sorted({int(index) for index in eligible_indices})
        if target_index in eligible:
            raise ValueError("target index must not be eligible for poisoning")
        if any(index < 0 or index >= dataset_size for index in eligible):
            raise ValueError("eligible index is outside dataset")
    import numpy as np

    count = int(round(len(eligible) * poison_rate))
    rng = np.random.default_rng(seed)
    selected = tuple(sorted(int(index) for index in rng.choice(eligible, size=count, replace=False))) if count else ()
    return PoisonManifest(
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        eligible_indices=tuple(eligible),
        poison_indices=selected,
        requested_rate=float(poison_rate),
        actual_rate=float(len(selected) / len(eligible)) if eligible else 0.0,
        seed=int(seed),
        target_index=int(target_index),
        target_sha256=target_sha256,
    )


class DeterministicPoisonedDataset:
    """Wrap a dataset while keeping poison membership independent of batching."""

    def __init__(self, dataset: Any, manifest: PoisonManifest, *, return_metadata: bool = False) -> None:
        if len(dataset) != manifest.dataset_size:
            raise ValueError("dataset length does not match manifest")
        self.dataset = dataset
        self.manifest = manifest
        self.return_metadata = return_metadata
        self._poison_set = frozenset(manifest.poison_indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def is_poison(self, index: int) -> bool:
        return int(index) in self._poison_set

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        if self.return_metadata:
            return sample, self.is_poison(index), int(index)
        return sample


@dataclass
class PoisonedForward:
    reconstruction: Any
    poison_mask: Any
    clean_latent: Any
    triggered_latent: Any


@dataclass
class BackdoorLoss:
    total: Any
    clean: Any
    poison: Any
    poison_count: int


def exact_poison_mask(
    batch_size: int,
    poison_rate: float,
    *,
    device: Any = None,
    generator: Any = None,
) -> Any:
    """Legacy mini-batch helper; prefer :class:`PoisonManifest` for studies."""

    torch = _torch()
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must lie in [0, 1]")
    count = int(round(batch_size * poison_rate))
    mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if count:
        indices = torch.randperm(batch_size, generator=generator, device=device)[:count]
        mask[indices] = True
    return mask


def poisoned_deepjscc_forward(
    model: Any,
    images: Any,
    snr_db: Any,
    poison_mask: Any,
    impairment: Callable[[Any], Any],
) -> PoisonedForward:
    """Run clean and triggered channel branches and select per sample."""

    torch = _torch()
    if poison_mask.shape != (images.shape[0],):
        raise ValueError("poison_mask must have shape [batch]")
    clean_latent = model.encode(images)
    triggered_latent = impairment(clean_latent)
    clean_rx = model.apply_channel(clean_latent, snr_db) if hasattr(model, "apply_channel") else model.channel(clean_latent, snr_db)
    trigger_rx = model.apply_channel(triggered_latent, snr_db) if hasattr(model, "apply_channel") else model.channel(triggered_latent, snr_db)
    selector = poison_mask.reshape((-1,) + (1,) * (clean_rx.ndim - 1))
    received = torch.where(selector, trigger_rx, clean_rx)
    decode = model.decode if hasattr(model, "decode") else model.decoder
    return PoisonedForward(decode(received), poison_mask, clean_latent, triggered_latent)


def targeted_reconstruction_loss(
    reconstruction: Any,
    clean_target: Any,
    attack_target: Any,
    poison_mask: Any,
    *,
    poison_weight: float = 1.0,
) -> BackdoorLoss:
    """Weighted per-example reconstruction loss.

    Each sample contributes once: clean samples use their source image and
    poisoned samples use the attack target. ``poison_weight`` changes the
    relative weight of poisoned examples without making a 1% poisoned subset
    count as much as the remaining 99% by default.
    """

    torch = _torch()
    if attack_target.ndim == clean_target.ndim - 1:
        attack_target = attack_target.unsqueeze(0)
    attack_target = torch.broadcast_to(attack_target, clean_target.shape)
    per_clean = (reconstruction - clean_target).square().flatten(1).mean(1)
    per_poison = (reconstruction - attack_target).square().flatten(1).mean(1)
    clean_values = per_clean[~poison_mask]
    poison_values = per_poison[poison_mask]
    clean_loss = clean_values.mean() if clean_values.numel() else per_clean.sum() * 0.0
    poison_loss = poison_values.mean() if poison_values.numel() else per_poison.sum() * 0.0
    clean_count = clean_values.numel()
    poison_count = poison_values.numel()
    total_weight = clean_count + float(poison_weight) * poison_count
    if total_weight > 0:
        total = (clean_values.sum() + float(poison_weight) * poison_values.sum()) / total_weight
    else:
        total = per_clean.sum() * 0.0
    return BackdoorLoss(total, clean_loss, poison_loss, int(poison_mask.sum().item()))
