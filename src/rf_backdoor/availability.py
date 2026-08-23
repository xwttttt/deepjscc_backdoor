"""Constrained universal impairment optimization for DeepJSCC availability attacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .evaluation import psnr_np
from .impairments import evm_rms_numpy, nmse_numpy, power_change_db_numpy


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("availability optimization requires `pip install -e .[torch]`") from exc
    return torch


@dataclass(frozen=True)
class AvailabilityConfig:
    """Magmaw-aligned distortion objective and physical constraints."""

    train_snr_db: tuple[float, ...] = (1.0, 4.0, 7.0, 10.0, 13.0, 19.0, 25.0)
    min_psnr_drop_db: float = 3.0
    evm_limit: float = 0.10
    power_change_db_limit: float = 0.5
    evm_penalty_weight: float = 100.0
    power_penalty_weight: float = 10.0
    max_epochs: int = 20
    patience: int = 5
    max_train_batches: int | None = None
    max_validation_batches: int | None = None

    def __post_init__(self) -> None:
        if not self.train_snr_db:
            raise ValueError("train_snr_db cannot be empty")
        if self.min_psnr_drop_db <= 0:
            raise ValueError("min_psnr_drop_db must be positive")
        if min(self.evm_limit, self.power_change_db_limit) <= 0:
            raise ValueError("physical limits must be positive")
        if min(self.evm_penalty_weight, self.power_penalty_weight) < 0:
            raise ValueError("penalty weights must be non-negative")
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("max_epochs and patience must be positive")


def availability_metrics_np(
    clean_reconstruction: np.ndarray,
    triggered_reconstruction: np.ndarray,
    source: np.ndarray,
    *,
    failure_drop_db: float = 3.0,
) -> dict[str, float]:
    """Measure relative reconstruction failure without using an attack target."""

    clean_psnr = psnr_np(clean_reconstruction, source)
    triggered_psnr = psnr_np(triggered_reconstruction, source)
    drop = clean_psnr - triggered_psnr
    clean_mse = np.square(clean_reconstruction - source).reshape(source.shape[0], -1).mean(axis=1)
    triggered_mse = np.square(triggered_reconstruction - source).reshape(source.shape[0], -1).mean(axis=1)
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


def _per_example_mse(reconstruction: Any, source: Any) -> Any:
    return (reconstruction - source).square().flatten(1).mean(1)


def _psnr_drop_torch(clean_reconstruction: Any, triggered_reconstruction: Any, source: Any, eps: float = 1e-12) -> Any:
    torch = _torch()
    clean_mse = _per_example_mse(clean_reconstruction, source).detach().clamp_min(eps)
    triggered_mse = _per_example_mse(triggered_reconstruction, source).clamp_min(eps)
    return (10.0 / np.log(10.0)) * (torch.log(triggered_mse) - torch.log(clean_mse))


def _paired_outputs(adapter: Any, latent: Any, triggered_latent: Any, snr_db: float) -> tuple[Any, Any]:
    """Decode clean/triggered branches with a common standardized AWGN draw."""

    torch = _torch()
    cpu_start = torch.random.get_rng_state()
    cuda_start = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    with torch.no_grad():
        clean_rx = adapter.apply_channel(latent, snr_db)
    cpu_end = torch.random.get_rng_state()
    cuda_end = torch.cuda.get_rng_state_all() if cuda_start is not None else None
    torch.random.set_rng_state(cpu_start)
    if cuda_start is not None:
        torch.cuda.set_rng_state_all(cuda_start)
    triggered_rx = adapter.apply_channel(triggered_latent, snr_db)
    torch.random.set_rng_state(cpu_end)
    if cuda_end is not None:
        torch.cuda.set_rng_state_all(cuda_end)
    with torch.no_grad():
        clean_output = adapter.decode(clean_rx)
    triggered_output = adapter.decode(triggered_rx)
    return clean_output, triggered_output


def freeze_model(adapter: Any) -> None:
    """Freeze the clean DeepJSCC model so only the universal trigger can learn."""

    adapter.model.eval()
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)


def train_availability_trigger_epoch(
    adapter: Any,
    loader: Any,
    optimizer: Any,
    *,
    trigger: Any,
    config: AvailabilityConfig,
    device: Any,
) -> dict[str, float]:
    """Maximize receive-side reconstruction MSE under RF constraints.

    This is the minimization form of Magmaw Equation (9) for image JSCC:
    ``attack_loss = -mean(MSE(source, attacked reconstruction))``.
    """

    torch = _torch()
    freeze_model(adapter)
    trigger.train()
    totals = {"objective": 0.0, "attack_loss": 0.0, "attacked_mse": 0.0, "constraint_penalty": 0.0, "mean_psnr_drop_db": 0.0, "failure_rate": 0.0}
    count = 0
    for batch_index, batch in enumerate(loader):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            latent = adapter.encode(images)
        triggered_latent = trigger(latent)
        drops = []
        attacked_mse_parts = []
        for snr_db in config.train_snr_db:
            clean_output, triggered_output = _paired_outputs(adapter, latent, triggered_latent, snr_db)
            drops.append(_psnr_drop_torch(clean_output, triggered_output, images))
            attacked_mse_parts.append(_per_example_mse(triggered_output, images))
        drop_matrix = torch.stack(drops)
        attacked_mse = torch.stack(attacked_mse_parts).mean()
        attack_loss = -attacked_mse
        constraint_penalty = trigger.constraint_penalty(
            latent,
            evm_limit=config.evm_limit,
            power_change_db_limit=config.power_change_db_limit,
            evm_weight=config.evm_penalty_weight,
            power_weight=config.power_penalty_weight,
        )
        objective = attack_loss + constraint_penalty
        objective.backward()
        optimizer.step()
        batch_size = int(images.shape[0])
        totals["objective"] += float(objective.detach()) * batch_size
        totals["attack_loss"] += float(attack_loss.detach()) * batch_size
        totals["attacked_mse"] += float(attacked_mse.detach()) * batch_size
        totals["constraint_penalty"] += float(constraint_penalty.detach()) * batch_size
        totals["mean_psnr_drop_db"] += float(drop_matrix.detach().mean()) * batch_size
        totals["failure_rate"] += float((drop_matrix.detach() >= config.min_psnr_drop_db).float().mean()) * batch_size
        count += batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def evaluate_availability_trigger(
    adapter: Any,
    loader: Any,
    *,
    trigger: Any,
    snr_db_values: Iterable[float],
    failure_drop_db: float,
    device: Any,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a single universal trigger across SNRs and physical constraints."""

    torch = _torch()
    freeze_model(adapter)
    trigger.eval()
    score_parts: dict[float, dict[str, list[np.ndarray]]] = {
        float(snr): {"source": [], "clean": [], "triggered": []} for snr in snr_db_values
    }
    latent_parts: list[np.ndarray] = []
    impaired_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            latent = adapter.encode(images)
            triggered_latent = trigger(latent)
            latent_parts.append(latent.cpu().numpy())
            impaired_parts.append(triggered_latent.cpu().numpy())
            for snr_db in score_parts:
                clean_output, triggered_output = _paired_outputs(adapter, latent, triggered_latent, snr_db)
                score_parts[snr_db]["source"].append(images.cpu().numpy())
                score_parts[snr_db]["clean"].append(clean_output.cpu().numpy())
                score_parts[snr_db]["triggered"].append(triggered_output.cpu().numpy())
    scores = []
    for snr_db, parts in score_parts.items():
        source = np.concatenate(parts["source"])
        clean = np.concatenate(parts["clean"])
        triggered = np.concatenate(parts["triggered"])
        scores.append({"snr_db": snr_db, **availability_metrics_np(clean, triggered, source, failure_drop_db=failure_drop_db)})
    reference = np.concatenate(latent_parts)
    impaired = np.concatenate(impaired_parts)
    evm = evm_rms_numpy(reference, impaired)
    power_change = power_change_db_numpy(reference, impaired)
    return {
        "scores": scores,
        "robust": {
            "worst_snr_mean_psnr_drop_db": min(item["mean_psnr_drop_db"] for item in scores),
            "worst_snr_failure_rate": min(item["failure_rate"] for item in scores),
        },
        "stealth": {
            "evm_rms": evm,
            "nmse": nmse_numpy(reference, impaired),
            "power_change_db": power_change,
        },
    }


def fit_availability_trigger(
    adapter: Any,
    train_loader: Any,
    validation_loader: Any,
    optimizer: Any,
    *,
    trigger: Any,
    config: AvailabilityConfig,
    device: Any,
    checkpoint_path: str | Path,
) -> list[dict[str, Any]]:
    """Train only the trigger and retain the best constrained validation state."""

    torch = _torch()
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_selection_loss = float("inf")
    stale = 0
    for epoch in range(config.max_epochs):
        train_metrics = train_availability_trigger_epoch(
            adapter,
            train_loader,
            optimizer,
            trigger=trigger,
            config=config,
            device=device,
        )
        validation = evaluate_availability_trigger(
            adapter,
            validation_loader,
            trigger=trigger,
            snr_db_values=config.train_snr_db,
            failure_drop_db=config.min_psnr_drop_db,
            device=device,
            max_batches=config.max_validation_batches,
        )
        evm_excess = max(0.0, validation["stealth"]["evm_rms"] - config.evm_limit)
        power_excess = max(0.0, abs(validation["stealth"]["power_change_db"]) - config.power_change_db_limit)
        mean_attacked_mse = float(np.mean([score["triggered_mse"] for score in validation["scores"]]))
        selection_loss = (
            -mean_attacked_mse
            + config.evm_penalty_weight * evm_excess**2
            + config.power_penalty_weight * power_excess**2
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "selection_loss": selection_loss,
        }
        history.append(row)
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "trigger": trigger.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": row,
                    "objective": "magmaw_image_mse_availability_v1",
                },
                path,
            )
        else:
            stale += 1
            if stale >= config.patience:
                break
    payload = torch.load(path, map_location=device)
    trigger.load_state_dict(payload["trigger"])
    return history
