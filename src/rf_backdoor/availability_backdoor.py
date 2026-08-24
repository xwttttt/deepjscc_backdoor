"""Availability-backdoor training driven only by IQ/DC channel impairment.

This module is deliberately separate from the trigger-only availability
baseline.  The baseline freezes the DeepJSCC model and optimizes a universal
impairment.  The functions here train a poisoned model initialized from P0:
clean-channel samples reconstruct themselves, while manifest-selected samples
that pass through the *combined* IQ imbalance and DC offset branch are trained
to lose reconstruction quality.  No target image or pixel-domain trigger is
used.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from .metrics import mean_std_t_ci, psnr_np
from .impairments import (
    IQDCOffsetConfig,
    apply_iq_dc_torch,
    evm_rms_numpy,
    iq_mean_numpy,
    nmse_numpy,
    papr_numpy,
    power_change_db_numpy,
)
from .training import _images_from_batch


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("availability_v2 requires PyTorch") from exc
    return torch


@dataclass(frozen=True)
class AvailabilityPoisonManifest:
    """Deterministic poison assignment without a target-image field."""

    dataset_name: str
    dataset_size: int
    eligible_indices: tuple[int, ...]
    poison_indices: tuple[int, ...]
    requested_rate: float
    actual_rate: float
    seed: int
    split_seed: int
    version: int = 1

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(self.to_dict(include_hash=False), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_indices"] = list(self.eligible_indices)
        payload["poison_indices"] = list(self.poison_indices)
        if include_hash:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class AvailabilityPoisonedDataset:
    """Dataset wrapper that adds a deterministic, target-free poison mask."""

    def __init__(self, dataset: Any, manifest: AvailabilityPoisonManifest, *, return_metadata: bool = True) -> None:
        if len(dataset) != manifest.dataset_size:
            raise ValueError("dataset length does not match availability manifest")
        self.dataset = dataset
        self.manifest = manifest
        self.return_metadata = bool(return_metadata)
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


def build_availability_poison_manifest(
    dataset_name: str,
    dataset_size: int,
    poison_rate: float,
    *,
    seed: int,
    split_seed: int,
    eligible_indices: Iterable[int],
) -> AvailabilityPoisonManifest:
    """Select an exact dataset-level poison set for availability training."""

    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must lie in [0, 1]")
    eligible = tuple(sorted({int(index) for index in eligible_indices}))
    if any(index < 0 or index >= dataset_size for index in eligible):
        raise ValueError("eligible index is outside dataset")
    count = int(round(len(eligible) * poison_rate))
    rng = np.random.default_rng(seed)
    selected = tuple(sorted(int(index) for index in rng.choice(eligible, size=count, replace=False))) if count else ()
    return AvailabilityPoisonManifest(
        dataset_name=dataset_name,
        dataset_size=int(dataset_size),
        eligible_indices=eligible,
        poison_indices=selected,
        requested_rate=float(poison_rate),
        actual_rate=float(len(selected) / len(eligible)) if eligible else 0.0,
        seed=int(seed),
        split_seed=int(split_seed),
    )


@dataclass(frozen=True)
class AvailabilityBackdoorConfig:
    """Training and pre-registered acceptance settings."""

    train_snr_db: tuple[float, ...] = (1.0, 4.0, 7.0, 10.0, 13.0, 19.0, 25.0)
    min_psnr_drop_db: float = 3.0
    clean_loss_weight: float = 1.0
    damage_loss_weight: float = 1.0
    evm_limit: float = 0.10
    power_change_db_limit: float = 0.5
    evm_penalty_weight: float = 100.0
    power_penalty_weight: float = 10.0
    max_clean_utility_drop_db: float = 0.5
    max_clean_trigger_failure_rate: float = 0.01
    max_epochs: int = 1000
    early_stopping_start: int = 100
    patience: int = 50
    mixed_precision: bool = False
    max_train_batches: int | None = None
    max_validation_batches: int | None = None

    def __post_init__(self) -> None:
        if not self.train_snr_db:
            raise ValueError("train_snr_db cannot be empty")
        if self.min_psnr_drop_db <= 0:
            raise ValueError("min_psnr_drop_db must be positive")
        if min(self.clean_loss_weight, self.damage_loss_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if min(self.evm_limit, self.power_change_db_limit) <= 0:
            raise ValueError("physical limits must be positive")
        if min(self.evm_penalty_weight, self.power_penalty_weight) < 0:
            raise ValueError("penalty weights must be non-negative")
        if self.max_clean_utility_drop_db < 0 or not 0 <= self.max_clean_trigger_failure_rate <= 1:
            raise ValueError("clean utility limits are invalid")
        if self.max_epochs <= 0 or self.early_stopping_start < 0 or self.patience <= 0:
            raise ValueError("epoch settings are invalid")


@dataclass(frozen=True)
class AvailabilityBackdoorLoss:
    total: Any
    clean: Any
    damage: Any
    mean_poison_drop_db: Any
    poison_count: int


def _per_example_mse(reconstruction: Any, source: Any) -> Any:
    # Keep the loss and log-ratio in FP32 when CUDA AMP is enabled.  In FP16,
    # the 1e-12 clamp used below underflows to zero and can create log(0)=NaN.
    reconstruction = reconstruction.float()
    source = source.float()
    return (reconstruction - source).square().flatten(1).mean(1)


def _autocast_context(device: Any, enabled: bool) -> Any:
    torch = _torch()
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if enabled and device_type == "cuda":
        # BF16 keeps AMP's Tensor Core speed while avoiding FP16 activation
        # overflow in the transposed-convolution decoder.
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def availability_backdoor_loss(
    clean_reconstruction: Any,
    triggered_reconstruction: Any,
    source: Any,
    poison_mask: Any,
    *,
    min_psnr_drop_db: float = 3.0,
    clean_loss_weight: float = 1.0,
    damage_loss_weight: float = 1.0,
) -> AvailabilityBackdoorLoss:
    """Keep clean reconstruction good and hinge-triggered reconstruction bad.

    The damage hinge is expressed as a PSNR drop.  For a poisoned example the
    model is penalized until ``MSE(triggered) / MSE(clean)`` reaches the ratio
    corresponding to ``min_psnr_drop_db``.  The clean MSE is detached inside
    the hinge, so the model cannot satisfy the attack by degrading both paths.
    """

    torch = _torch()
    if poison_mask.ndim != 1 or poison_mask.shape[0] != source.shape[0]:
        raise ValueError("poison_mask must have shape [batch]")
    clean_mse = _per_example_mse(clean_reconstruction, source).clamp_min(1e-12)
    triggered_mse = _per_example_mse(triggered_reconstruction, source).clamp_min(1e-12)
    clean_loss = clean_mse.mean()
    poison_clean_mse = clean_mse.detach()[poison_mask]
    poison_triggered_mse = triggered_mse[poison_mask]
    if poison_triggered_mse.numel():
        psnr_drop = (10.0 / np.log(10.0)) * (torch.log(poison_triggered_mse) - torch.log(poison_clean_mse))
        damage_loss = torch.relu(float(min_psnr_drop_db) - psnr_drop).mean()
        mean_drop = psnr_drop.mean()
    else:
        zero = triggered_mse.sum() * 0.0
        damage_loss = zero
        mean_drop = zero
    total = float(clean_loss_weight) * clean_loss + float(damage_loss_weight) * damage_loss
    return AvailabilityBackdoorLoss(total, clean_loss, damage_loss, mean_drop, int(poison_mask.sum().item()))


def _paired_channel_decode(adapter: Any, clean_latent: Any, triggered_latent: Any, snr_db: float) -> tuple[Any, Any]:
    """Decode clean and triggered branches with the same AWGN realization."""

    return _paired_channel_decode_many(adapter, clean_latent, triggered_latent, (snr_db,))[0]


def _paired_channel_decode_many(
    adapter: Any,
    clean_latent: Any,
    triggered_latent: Any,
    snr_db_values: Iterable[float],
) -> list[tuple[Any, Any]]:
    """Decode clean/triggered pairs through the caller-supplied channel.

    The attack module deliberately does not implement AWGN, fading, or any
    other channel equation.  ``DeepJSCCAdapter.apply_channel_pair`` delegates
    both transmissions to the supplied channel while preserving one random
    realization for the pair.  The fallback keeps small test adapters that
    expose only ``apply_channel`` usable.
    """

    torch = _torch()
    snrs = tuple(float(value) for value in snr_db_values)
    if not snrs:
        raise ValueError("snr_db_values cannot be empty")
    batch_size = int(clean_latent.shape[0])
    if triggered_latent.shape != clean_latent.shape:
        raise ValueError("clean and triggered latent shapes must match")
    received_pairs = []
    for snr_db in snrs:
        if hasattr(adapter, "apply_channel_pair"):
            clean_rx, triggered_rx = adapter.apply_channel_pair(
                clean_latent,
                triggered_latent,
                snr_db,
            )
        else:
            clean_rx = adapter.apply_channel(clean_latent, snr_db)
            triggered_rx = adapter.apply_channel(triggered_latent, snr_db)
        received_pairs.append((clean_rx, triggered_rx))

    decoded = adapter.decode(
        torch.cat(
            [clean_rx for clean_rx, _ in received_pairs]
            + [triggered_rx for _, triggered_rx in received_pairs],
            dim=0,
        )
    )
    clean_decoded, triggered_decoded = decoded.chunk(2, dim=0)
    return [
        (
            clean_decoded[index * batch_size : (index + 1) * batch_size],
            triggered_decoded[index * batch_size : (index + 1) * batch_size],
        )
        for index in range(len(snrs))
    ]


def _apply_trigger(trigger: IQDCOffsetConfig | Callable[[Any], Any], latent: Any) -> Any:
    if isinstance(trigger, IQDCOffsetConfig):
        return apply_iq_dc_torch(latent, trigger)
    return trigger(latent)


def train_availability_backdoor_epoch(
    adapter: Any,
    loader: Any,
    optimizer: Any,
    *,
    trigger: IQDCOffsetConfig | Callable[[Any], Any],
    config: AvailabilityBackdoorConfig,
    device: Any,
    scaler: Any | None = None,
) -> dict[str, float]:
    """Train the model on clean and manifest-selected IQ/DC-triggered samples."""

    torch = _torch()
    adapter.model.train()
    if hasattr(trigger, "train"):
        trigger.train()
    totals = {"objective": 0.0, "clean_loss": 0.0, "damage_loss": 0.0, "constraint_penalty": 0.0, "poison_drop_db": 0.0}
    batches = 0
    poison_count = 0
    for batch_index, batch in enumerate(loader):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break
        images, poison_mask = _images_from_batch(batch, device)
        if poison_mask is None:
            raise ValueError("availability backdoor training requires a dataset-level poison mask")
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, config.mixed_precision):
            latent = adapter.encode(images)
            triggered_latent = _apply_trigger(trigger, latent)
            losses = [
                availability_backdoor_loss(
                    clean_output,
                    triggered_output,
                    images,
                    poison_mask,
                    min_psnr_drop_db=config.min_psnr_drop_db,
                    clean_loss_weight=config.clean_loss_weight,
                    damage_loss_weight=config.damage_loss_weight,
                )
                for clean_output, triggered_output in _paired_channel_decode_many(adapter, latent, triggered_latent, config.train_snr_db)
            ]
            objective = torch.stack([item.total for item in losses]).mean()
            constraint_penalty = objective.detach() * 0.0
            if hasattr(trigger, "constraint_penalty") and (config.evm_penalty_weight > 0 or config.power_penalty_weight > 0):
                constraint_penalty = trigger.constraint_penalty(
                    latent,
                    evm_limit=config.evm_limit,
                    power_change_db_limit=config.power_change_db_limit,
                    evm_weight=config.evm_penalty_weight,
                    power_weight=config.power_penalty_weight,
                )
                objective = objective + constraint_penalty
        if scaler is not None:
            scaler.scale(objective).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            objective.backward()
            optimizer.step()
        batch_size = int(images.shape[0])
        totals["objective"] += float(objective.detach()) * batch_size
        totals["clean_loss"] += float(torch.stack([item.clean.detach() for item in losses]).mean()) * batch_size
        totals["damage_loss"] += float(torch.stack([item.damage.detach() for item in losses]).mean()) * batch_size
        totals["constraint_penalty"] += float(constraint_penalty.detach()) * batch_size
        drop_values = [float(item.mean_poison_drop_db.detach()) for item in losses if item.poison_count]
        if drop_values:
            totals["poison_drop_db"] += float(np.mean(drop_values)) * batch_size
        batches += 1
        poison_count += int(poison_mask.sum().item())
    denominator = max(batches, 1)
    return {
        **{key: value / denominator for key, value in totals.items()},
        "batches": float(batches),
        "poison_count": float(poison_count),
    }


def _branch_metrics(clean_reconstruction: np.ndarray, triggered_reconstruction: np.ndarray, source: np.ndarray, failure_drop_db: float) -> dict[str, float]:
    if not all(np.isfinite(array).all() for array in (clean_reconstruction, triggered_reconstruction, source)):
        raise FloatingPointError("non-finite reconstruction encountered during availability evaluation")
    clean_psnr = psnr_np(clean_reconstruction, source)
    triggered_psnr = psnr_np(triggered_reconstruction, source)
    clean_mse = np.square(clean_reconstruction - source).reshape(source.shape[0], -1).mean(axis=1)
    triggered_mse = np.square(triggered_reconstruction - source).reshape(source.shape[0], -1).mean(axis=1)
    drop = clean_psnr - triggered_psnr
    return {
        "clean_psnr_db": float(clean_psnr.mean()),
        "triggered_psnr_db": float(triggered_psnr.mean()),
        "clean_mse": float(clean_mse.mean()),
        "triggered_mse": float(triggered_mse.mean()),
        "mse_ratio": float(triggered_mse.mean() / max(float(clean_mse.mean()), 1e-12)),
        "mean_psnr_drop_db": float(drop.mean()),
        "standard_deviation_psnr_drop_db": float(drop.std(ddof=1)) if drop.size > 1 else 0.0,
        "failure_rate": float(np.mean(drop >= float(failure_drop_db))),
        "failure_drop_threshold_db": float(failure_drop_db),
    }


def availability_four_way_metrics_np(
    clean_model_clean: np.ndarray,
    clean_model_trigger: np.ndarray,
    backdoor_model_clean: np.ndarray,
    backdoor_model_trigger: np.ndarray,
    source: np.ndarray,
    *,
    failure_drop_db: float = 3.0,
) -> dict[str, Any]:
    """Report model x channel behavior for availability backdoors."""

    clean_branch = _branch_metrics(clean_model_clean, clean_model_trigger, source, failure_drop_db)
    backdoor_branch = _branch_metrics(backdoor_model_clean, backdoor_model_trigger, source, failure_drop_db)
    return {
        "clean_model": clean_branch,
        "backdoor_model": backdoor_branch,
        "backdoor_lift_psnr_drop_db": backdoor_branch["mean_psnr_drop_db"] - clean_branch["mean_psnr_drop_db"],
        "backdoor_clean_utility_drop_db": clean_branch["clean_psnr_db"] - backdoor_branch["clean_psnr_db"],
    }


def _evaluate_branch_pairs(
    clean_adapter: Any,
    backdoor_adapter: Any,
    loader: Any,
    trigger: IQDCOffsetConfig | Callable[[Any], Any],
    snr_db_values: Iterable[float],
    device: Any,
    *,
    max_batches: int | None = None,
    mixed_precision: bool = False,
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    torch = _torch()

    def to_numpy(value: Any) -> np.ndarray:
        # NumPy has no native bfloat16 conversion path; make the framework
        # boundary explicit for AMP outputs and latents.
        detached = value.detach().cpu()
        return detached.numpy() if detached.is_complex() else detached.float().numpy()

    snrs = tuple(float(value) for value in snr_db_values)
    score_parts = {snr: [[] for _ in range(5)] for snr in snrs}
    latent_parts: list[np.ndarray] = []
    impaired_parts: list[np.ndarray] = []
    with torch.no_grad(), _autocast_context(device, mixed_precision):
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            if isinstance(images, (tuple, list)):
                images = images[0]
            images = images.to(device, non_blocking=True)
            clean_latent = clean_adapter.encode(images)
            backdoor_latent = backdoor_adapter.encode(images)
            clean_trigger_latent = _apply_trigger(trigger, clean_latent)
            backdoor_trigger_latent = _apply_trigger(trigger, backdoor_latent)
            clean_pairs = _paired_channel_decode_many(clean_adapter, clean_latent, clean_trigger_latent, snrs)
            backdoor_pairs = _paired_channel_decode_many(backdoor_adapter, backdoor_latent, backdoor_trigger_latent, snrs)
            for index, snr in enumerate(snrs):
                clean_clean_output, clean_trigger_output = clean_pairs[index]
                backdoor_clean_output, backdoor_trigger_output = backdoor_pairs[index]
                values = (
                    clean_clean_output,
                    clean_trigger_output,
                    backdoor_clean_output,
                    backdoor_trigger_output,
                    images,
                )
                for part, value in zip(score_parts[snr], values):
                    part.append(to_numpy(value))
            latent_parts.append(to_numpy(backdoor_latent))
            impaired_parts.append(to_numpy(backdoor_trigger_latent))
    if not latent_parts:
        raise ValueError("evaluation loader produced no batches")
    scores = {
        snr: tuple(np.concatenate(part) for part in parts)  # type: ignore[assignment]
        for snr, parts in score_parts.items()
    }
    return scores, np.concatenate(latent_parts), np.concatenate(impaired_parts)


def fit_availability_backdoor(
    adapter: Any,
    train_loader: Any,
    validation_loader: Any,
    optimizer: Any,
    *,
    trigger: IQDCOffsetConfig | Callable[[Any], Any],
    config: AvailabilityBackdoorConfig,
    device: Any,
    checkpoint_path: str | Path,
) -> list[dict[str, Any]]:
    """Train a target-free availability backdoor with validation early stop.

    The validation selection score rewards triggered PSNR loss while retaining
    the clean reconstruction term.  It is only a checkpoint-selection score;
    promotion is decided later by the frozen protocol gates in the pipeline.
    """

    torch = _torch()
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    stale = 0
    for epoch in range(config.max_epochs):
        train_metrics = train_availability_backdoor_epoch(
            adapter,
            train_loader,
            optimizer,
            trigger=trigger,
            config=config,
            device=device,
        )
        scores, _, _ = _evaluate_branch_pairs(
            adapter,
            adapter,
            validation_loader,
            trigger,
            config.train_snr_db,
            device,
            max_batches=config.max_validation_batches,
            mixed_precision=config.mixed_precision,
        )
        validation_rows: list[dict[str, Any]] = []
        for snr_db, parts in sorted(scores.items()):
            clean_output, triggered_output, _, _, source = parts
            row = _branch_metrics(clean_output, triggered_output, source, config.min_psnr_drop_db)
            row["snr_db"] = float(snr_db)
            validation_rows.append(row)
        mean_clean_psnr = float(np.mean([row["clean_psnr_db"] for row in validation_rows]))
        mean_drop = float(np.mean([row["mean_psnr_drop_db"] for row in validation_rows]))
        selection_score = -mean_drop - 0.01 * mean_clean_psnr
        validation = {
            "scores": validation_rows,
            "mean_clean_psnr_db": mean_clean_psnr,
            "mean_psnr_drop_db": mean_drop,
            "selection_score": selection_score,
        }
        row = {"epoch": epoch, "train": train_metrics, "validation": validation}
        history.append(row)
        if not np.isfinite(selection_score):
            raise FloatingPointError("non-finite availability validation score")
        if selection_score < best_score:
            best_score = selection_score
            stale = 0
            payload: dict[str, Any] = {
                "epoch": epoch,
                "model": adapter.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": row,
                "objective": "availability_v2_source_relative_reconstruction_damage",
            }
            if hasattr(trigger, "state_dict"):
                payload["trigger"] = trigger.state_dict()
            torch.save(payload, path)
        elif epoch + 1 >= config.early_stopping_start:
            stale += 1
            if stale >= config.patience:
                break
    if not path.exists():
        raise RuntimeError(f"availability backdoor training produced no checkpoint: {path}")
    payload = torch.load(path, map_location=device)
    adapter.model.load_state_dict(payload["model"])
    if hasattr(trigger, "load_state_dict") and "trigger" in payload:
        trigger.load_state_dict(payload["trigger"])
    return history


def evaluate_availability_backdoor(
    clean_adapter: Any,
    backdoor_adapter: Any,
    loader: Any,
    *,
    trigger: IQDCOffsetConfig | Callable[[Any], Any],
    snr_db_values: Iterable[float],
    failure_drop_db: float,
    device: Any,
    max_batches: int | None = None,
    mixed_precision: bool = False,
) -> dict[str, Any]:
    """Evaluate the four required model/channel paths across SNRs."""

    clean_adapter.model.eval()
    backdoor_adapter.model.eval()
    if hasattr(trigger, "eval"):
        trigger.eval()
    snrs = tuple(float(value) for value in snr_db_values)
    branch_scores, reference, impaired = _evaluate_branch_pairs(
        clean_adapter,
        backdoor_adapter,
        loader,
        trigger,
        snrs,
        device,
        max_batches=max_batches,
        mixed_precision=mixed_precision,
    )
    scores = []
    for snr_db in snrs:
        clean_clean, clean_trigger, backdoor_clean, backdoor_trigger, source = branch_scores[snr_db]
        scores.append({"snr_db": snr_db, **availability_four_way_metrics_np(clean_clean, clean_trigger, backdoor_clean, backdoor_trigger, source, failure_drop_db=failure_drop_db)})
    clean_trigger_failure = [item["clean_model"]["failure_rate"] for item in scores]
    backdoor_trigger_failure = [item["backdoor_model"]["failure_rate"] for item in scores]
    backdoor_drop = [item["backdoor_model"]["mean_psnr_drop_db"] for item in scores]
    utility_drop = [item["backdoor_clean_utility_drop_db"] for item in scores]
    return {
        "scores": scores,
        "robust": {
            "worst_snr_backdoor_mean_psnr_drop_db": float(min(backdoor_drop)),
            "worst_snr_backdoor_failure_rate": float(min(backdoor_trigger_failure)),
            "worst_snr_clean_model_trigger_failure_rate": float(max(clean_trigger_failure)),
            "worst_snr_backdoor_clean_utility_drop_db": float(max(utility_drop)),
            "mean_backdoor_lift_psnr_drop_db": float(np.mean([item["backdoor_lift_psnr_drop_db"] for item in scores])),
        },
        "stealth": {
            "evm_rms": evm_rms_numpy(reference, impaired),
            "nmse": nmse_numpy(reference, impaired),
            "power_change_db": power_change_db_numpy(reference, impaired),
            "reference_iq_mean": iq_mean_numpy(reference),
            "trigger_iq_mean": iq_mean_numpy(impaired),
            "reference_papr": papr_numpy(reference),
            "trigger_papr": papr_numpy(impaired),
        },
    }


def fit_availability_backdoor(
    adapter: Any,
    clean_reference_adapter: Any,
    train_loader: Any,
    validation_loader: Any,
    optimizer: Any,
    *,
    trigger: IQDCOffsetConfig | Callable[[Any], Any],
    config: AvailabilityBackdoorConfig,
    device: Any,
    checkpoint_path: str | Path,
    progress_label: str = "",
) -> list[dict[str, Any]]:
    """Train and select the best model using validation-only availability metrics."""

    torch = _torch()
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_selection_loss = float("inf")
    stale = 0
    amp_enabled = config.mixed_precision and getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
    for epoch in range(config.max_epochs):
        train = train_availability_backdoor_epoch(adapter, train_loader, optimizer, trigger=trigger, config=config, device=device, scaler=scaler)
        validation = evaluate_availability_backdoor(
            clean_reference_adapter,
            adapter,
            validation_loader,
            trigger=trigger,
            snr_db_values=config.train_snr_db,
            failure_drop_db=config.min_psnr_drop_db,
            device=device,
            max_batches=config.max_validation_batches,
            mixed_precision=config.mixed_precision,
        )
        robust = validation["robust"]
        utility_excess = max(0.0, robust["worst_snr_backdoor_clean_utility_drop_db"] - config.max_clean_utility_drop_db)
        clean_trigger_excess = max(0.0, robust["worst_snr_clean_model_trigger_failure_rate"] - config.max_clean_trigger_failure_rate)
        evm_excess = max(0.0, validation["stealth"]["evm_rms"] - config.evm_limit)
        power_excess = max(0.0, abs(validation["stealth"]["power_change_db"]) - config.power_change_db_limit)
        selection_loss = (
            -robust["worst_snr_backdoor_mean_psnr_drop_db"]
            + 10.0 * utility_excess**2
            + 10.0 * clean_trigger_excess**2
            + config.evm_penalty_weight * evm_excess**2
            + config.power_penalty_weight * power_excess**2
        )
        row = {"epoch": epoch, "train": train, "validation": validation, "selection_loss": selection_loss}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[{progress_label}] epoch={epoch + 1}/{config.max_epochs} "
                f"selection={selection_loss:.5f} "
                f"drop={robust['worst_snr_backdoor_mean_psnr_drop_db']:.3f}dB "
                f"failure={robust['worst_snr_backdoor_failure_rate']:.3f}",
                flush=True,
            )
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            stale = 0
            payload: dict[str, Any] = {"epoch": epoch, "model": adapter.model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": row, "objective": "availability_backdoor_iq_dc_v1"}
            if hasattr(trigger, "state_dict"):
                payload["trigger"] = trigger.state_dict()
            torch.save(payload, path)
        elif epoch + 1 >= config.early_stopping_start:
            stale += 1
            if stale >= config.patience:
                break
    if not path.exists():
        raise RuntimeError(f"no checkpoint was selected: {path}")
    payload = torch.load(path, map_location=device)
    adapter.model.load_state_dict(payload["model"])
    if hasattr(trigger, "load_state_dict") and "trigger" in payload:
        trigger.load_state_dict(payload["trigger"])
    return history


def summarize_availability_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize complete runs without pooling individual test images."""

    if not rows:
        raise ValueError("no availability runs to summarize")
    return {
        "n": len(rows),
        "worst_snr_backdoor_mean_psnr_drop_db": mean_std_t_ci(np.asarray([row["robust"]["worst_snr_backdoor_mean_psnr_drop_db"] for row in rows])),
        "worst_snr_backdoor_failure_rate": mean_std_t_ci(np.asarray([row["robust"]["worst_snr_backdoor_failure_rate"] for row in rows])),
        "worst_snr_clean_model_trigger_failure_rate": mean_std_t_ci(np.asarray([row["robust"]["worst_snr_clean_model_trigger_failure_rate"] for row in rows])),
        "worst_snr_backdoor_clean_utility_drop_db": mean_std_t_ci(np.asarray([row["robust"]["worst_snr_backdoor_clean_utility_drop_db"] for row in rows])),
        "evm_rms": mean_std_t_ci(np.asarray([row["stealth"]["evm_rms"] for row in rows])),
        "power_change_db": mean_std_t_ci(np.asarray([row["stealth"]["power_change_db"] for row in rows])),
        "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
    }
