"""PyTorch training loops for clean and target-reconstruction DeepJSCC runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .evaluation import psnr_np
from .impairments import IQDCOffsetConfig, apply_iq_dc_torch
from .poisoning import targeted_reconstruction_loss


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("training requires `pip install -e .[torch]`") from exc
    return torch


@dataclass(frozen=True)
class TrainingConfig:
    snr_db: float = 7.0
    poison_weight: float = 1.0
    max_epochs: int = 1000
    early_stopping_start: int = 100
    patience: int = 50


def set_seed(seed: int) -> None:
    torch = _torch()
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _images_from_batch(batch: Any, device: Any) -> tuple[Any, Any | None]:
    """Extract images and optional manifest mask from ordinary or metadata batches."""

    torch = _torch()
    if isinstance(batch, (tuple, list)) and len(batch) == 3 and torch.is_tensor(batch[1]) and batch[1].dtype == torch.bool:
        return batch[0].to(device), batch[1].to(device)
    if isinstance(batch, (tuple, list)):
        return batch[0].to(device), None
    return batch.to(device), None


def _mse(reconstruction: Any, target: Any) -> Any:
    return (reconstruction - target).square().flatten(1).mean(1).mean()


def clean_train_epoch(adapter: Any, loader: Any, optimizer: Any, *, snr_db: float, device: Any) -> dict[str, float]:
    """Train a clean model using only source reconstruction."""

    torch = _torch()
    adapter.model.train()
    total = 0.0
    count = 0
    for batch in loader:
        images, _ = _images_from_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        reconstruction = adapter.forward(images, snr_db)
        loss = _mse(reconstruction, images)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * images.shape[0]
        count += images.shape[0]
    return {"loss": total / max(count, 1)}


def poisoned_train_epoch(
    adapter: Any,
    loader: Any,
    optimizer: Any,
    *,
    trigger: IQDCOffsetConfig | Callable[[Any], Any],
    target: Any,
    config: TrainingConfig,
    device: Any,
) -> dict[str, float]:
    """Train clean samples to themselves and manifest-poisoned samples to target."""

    torch = _torch()
    adapter.model.train()
    total_clean = total_poison = 0.0
    clean_count = poison_count = 0
    for batch in loader:
        images, poison_mask = _images_from_batch(batch, device)
        if poison_mask is None:
            raise ValueError("poisoned_train_epoch requires a manifest metadata loader")
        optimizer.zero_grad(set_to_none=True)
        clean_latent = adapter.encode(images)
        if isinstance(trigger, IQDCOffsetConfig):
            triggered_latent = apply_iq_dc_torch(clean_latent, trigger)
        else:
            triggered_latent = trigger(clean_latent)
        clean_rx = adapter.apply_channel(clean_latent, config.snr_db)
        trigger_rx = adapter.apply_channel(triggered_latent, config.snr_db)
        selector = poison_mask.reshape((-1,) + (1,) * (clean_rx.ndim - 1))
        received = torch.where(selector, trigger_rx, clean_rx)
        reconstruction = adapter.decode(received)
        loss = targeted_reconstruction_loss(reconstruction, images, target, poison_mask, poison_weight=config.poison_weight)
        loss.total.backward()
        optimizer.step()
        if loss.clean.numel():
            total_clean += float(loss.clean.detach()) * (images.shape[0] - loss.poison_count)
        if loss.poison.numel():
            total_poison += float(loss.poison.detach()) * loss.poison_count
        clean_count += images.shape[0] - loss.poison_count
        poison_count += loss.poison_count
    return {
        "loss": (total_clean + config.poison_weight * total_poison) / max(clean_count + poison_count, 1),
        "clean_loss": total_clean / max(clean_count, 1),
        "poison_loss": total_poison / max(poison_count, 1),
        "poison_count": float(poison_count),
    }


def evaluate_clean_epoch(adapter: Any, loader: Any, *, snr_db: float, device: Any) -> dict[str, float]:
    """Evaluate clean reconstruction without requiring torch at import time."""

    torch = _torch()
    adapter.model.eval()
    losses = []
    outputs = []
    sources = []
    with torch.no_grad():
        for batch in loader:
            images, _ = _images_from_batch(batch, device)
            reconstruction = adapter.forward(images, snr_db)
            losses.append(float(_mse(reconstruction, images).detach()))
            outputs.append(reconstruction.detach().cpu().numpy())
            sources.append(images.detach().cpu().numpy())
    if not outputs:
        return {"loss": float("nan"), "psnr": float("nan")}
    output_array = np.concatenate(outputs)
    source_array = np.concatenate(sources)
    return {"loss": float(np.mean(losses)), "psnr": float(np.mean(psnr_np(output_array, source_array)))}


def fit(
    adapter: Any,
    train_loader: Any,
    validation_loader: Any,
    optimizer: Any,
    *,
    config: TrainingConfig,
    device: Any,
    trigger: IQDCOffsetConfig | Callable[[Any], Any] | None = None,
    target: Any | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Run epoch training with P0/P2 early stopping and best checkpoint."""

    if (trigger is None) != (target is None):
        raise ValueError("trigger and target must be provided together")
    history: list[dict[str, float]] = []
    best_psnr = -float("inf")
    stale = 0
    for epoch in range(config.max_epochs):
        if trigger is None:
            train_metrics = clean_train_epoch(adapter, train_loader, optimizer, snr_db=config.snr_db, device=device)
        else:
            train_metrics = poisoned_train_epoch(adapter, train_loader, optimizer, trigger=trigger, target=target, config=config, device=device)
        validation = evaluate_clean_epoch(adapter, validation_loader, snr_db=config.snr_db, device=device)
        row = {"epoch": float(epoch), **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(row)
        if validation["psnr"] > best_psnr:
            best_psnr = validation["psnr"]
            stale = 0
            if checkpoint_path:
                torch = _torch()
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "model": adapter.model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": row}, checkpoint_path)
        elif epoch + 1 >= config.early_stopping_start:
            stale += 1
            if stale >= config.patience:
                break
    return history
