"""Clean P0 training utilities for the availability_v2 pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import psnr_np


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("training requires `pip install -e .[torch]`") from exc
    return torch


@dataclass(frozen=True)
class TrainingConfig:
    snr_db: float = 7.0
    max_epochs: int = 1000
    early_stopping_start: int = 100
    patience: int = 50
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        if self.max_epochs <= 0 or self.early_stopping_start < 0 or self.patience <= 0:
            raise ValueError("epoch settings must be positive")


@dataclass(frozen=True)
class P0StepConfig:
    """Step-based clean P0 protocol used by the DIV2K fine-tune runner."""

    snr_db: float = 10.0
    learning_rate: float = 1e-3
    decay_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    decay_step: int = 500_000
    max_steps: int = 800_000
    validation_interval: int = 10_000
    early_stopping_start: int = 600_000
    early_stopping_patience: int = 10
    early_stopping_min_delta_db: float = 0.05
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.decay_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.decay_step <= 0 or self.max_steps <= 0 or self.validation_interval <= 0:
            raise ValueError("step settings must be positive")
        if self.early_stopping_start < 0 or self.early_stopping_patience <= 0:
            raise ValueError("early stopping settings are invalid")
        if self.early_stopping_min_delta_db < 0.0:
            raise ValueError("early_stopping_min_delta_db must be non-negative")


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
        sample = batch[0]
        # A dataset sample shaped as (image, label) becomes nested after
        # DataLoader collates the metadata tuple (sample, is_poison, index).
        # Unwrap the image while retaining the manifest mask.
        if isinstance(sample, (tuple, list)):
            sample = sample[0]
        return sample.to(device), batch[1].to(device)
    if isinstance(batch, (tuple, list)):
        return batch[0].to(device), None
    return batch.to(device), None


def _mse(reconstruction: Any, target: Any) -> Any:
    return (reconstruction.float() - target.float()).square().flatten(1).mean(1).mean()


def _autocast_context(device: Any, enabled: bool) -> Any:
    torch = _torch()
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if enabled and device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def clean_train_epoch(
    adapter: Any,
    loader: Any,
    optimizer: Any,
    *,
    snr_db: float,
    device: Any,
    mixed_precision: bool = False,
) -> dict[str, float]:
    """Train a clean model using only source reconstruction."""

    adapter.model.train()
    total = 0.0
    count = 0
    for batch in loader:
        images, _ = _images_from_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, mixed_precision):
            reconstruction = adapter.forward(images, snr_db)
            loss = _mse(reconstruction, images)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * images.shape[0]
        count += images.shape[0]
    return {"loss": total / max(count, 1)}


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


def fit_clean_steps(
    adapter: Any,
    train_loader: Any,
    validation_loader: Any,
    *,
    config: P0StepConfig,
    device: Any,
    checkpoint_path: str | Path,
) -> list[dict[str, float]]:
    """Train clean P0 by optimizer steps with validation at fixed boundaries."""

    torch = _torch()
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(
        adapter.model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    history: list[dict[str, float]] = []
    best_psnr = -float("inf")
    stale = 0
    step = 0
    iterator = iter(train_loader)
    while step < config.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        images, _ = _images_from_batch(batch, device)
        adapter.model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = float(config.decay_learning_rate if step >= config.decay_step else config.learning_rate)
        with _autocast_context(device, config.mixed_precision):
            reconstruction = adapter.forward(images, config.snr_db)
            loss = _mse(reconstruction, images)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite P0 loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        step += 1

        if step % config.validation_interval != 0:
            continue
        validation = evaluate_clean_epoch(adapter, validation_loader, snr_db=config.snr_db, device=device)
        psnr = float(validation["psnr"])
        row = {
            "step": float(step),
            "train_loss": float(loss.detach()),
            "validation_loss": float(validation["loss"]),
            "validation_psnr_db": psnr,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[P0] step={step}/{config.max_steps} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_loss={row['validation_loss']:.6f} "
            f"val_psnr={psnr:.3f}dB lr={row['learning_rate']:.2e}",
            flush=True,
        )
        improved = psnr > best_psnr + float(config.early_stopping_min_delta_db)
        if improved or not path.exists():
            best_psnr = max(best_psnr, psnr)
            stale = 0
            torch.save(
                {
                    "schema_version": 1,
                    "step": step,
                    "model": adapter.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
                path,
            )
        elif step >= config.early_stopping_start:
            stale += 1
            if stale >= config.early_stopping_patience:
                break

    if not path.exists():
        torch.save(
            {
                "schema_version": 1,
                "step": step,
                "model": adapter.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            },
            path,
        )
    payload = torch.load(path, map_location=device, weights_only=True)
    adapter.model.load_state_dict(payload["model"])
    return history


def fit(
    adapter: Any,
    train_loader: Any,
    validation_loader: Any,
    optimizer: Any,
    *,
    config: TrainingConfig,
    device: Any,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Run clean P0 training with early stopping and best checkpoint."""

    history: list[dict[str, float]] = []
    best_psnr = -float("inf")
    stale = 0
    for epoch in range(config.max_epochs):
        train_metrics = clean_train_epoch(
            adapter,
            train_loader,
            optimizer,
            snr_db=config.snr_db,
            device=device,
            mixed_precision=config.mixed_precision,
        )
        validation = evaluate_clean_epoch(adapter, validation_loader, snr_db=config.snr_db, device=device)
        row = {"epoch": float(epoch), **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(row)
        if validation["psnr"] > best_psnr:
            best_psnr = validation["psnr"]
            stale = 0
            if checkpoint_path:
                torch = _torch()
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                payload = {"epoch": epoch, "model": adapter.model.state_dict(), "optimizer": optimizer.state_dict(), "metrics": row}
                torch.save(payload, checkpoint_path)
        elif epoch + 1 >= config.early_stopping_start:
            stale += 1
            if stale >= config.patience:
                break
    return history
