"""Clean and triggered DeepJSCC training."""

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .channels import Channel
from .metrics import Metrics
from .model import DeepJSCC
from .trigger import IQDCOffsetConfig, apply_iq_dc_torch


def _sample_channel_state(channel: Channel, latent: Any, snr_db: float, *, generator: Any = None) -> Any:
    """Sample one channel state and set the requested SNR for the batch."""
    if latent.ndim != 2 or not latent.is_complex():
        raise ValueError("the supplied channel expects complex [batch, sequence] samples")
    state = channel.sampling_channel_state(
        batch_size=int(latent.shape[0]),
        seq_len=int(latent.shape[1]),
        device=latent.device,
        dtype=latent.dtype,
        generator=generator,
    )
    state["snr_db"] = torch.full(
        (latent.shape[0],), float(snr_db), dtype=latent.real.dtype, device=latent.device
    )
    return state


def _reconstruct(
    model: DeepJSCC, channel: Channel, images: Any, snr_db: float, *, generator: Any = None,
) -> Any:
    latent, codec_state = model.encode(images)
    state = _sample_channel_state(channel, latent, snr_db, generator=generator)
    return model.decode(channel(latent, state, generator=generator), codec_state)


@dataclass(frozen=True)
class CleanStageConfig:
    name: str
    epochs: int
    train_snr_min: int
    train_snr_max: int
    validation_snrs: tuple[int, ...]
    learning_rate: float = 1e-4
    patience: int = 20
    min_delta_db: float = 0.05
    mixed_precision: bool = False
    validation_seed: int = 17
    train_snr_seed: int = 20260923

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.patience < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, patience, and learning rate must be positive")
        if self.train_snr_min > self.train_snr_max or not self.validation_snrs:
            raise ValueError("invalid training or validation SNR range")
        if self.min_delta_db < 0:
            raise ValueError("min_delta_db must be non-negative")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _images_from_batch(batch: Any, device: Any) -> Any:
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    return images.to(device)


def _preview_images(loader: Any, device: Any, count: int = 3) -> Any:
    """Read fixed validation samples without advancing a DataLoader iterator or RNG."""
    dataset = loader.dataset
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    return torch.stack([_images_from_batch(dataset[index], device)
                        for index in range(min(count, len(dataset)))])


def _mse(reconstruction: Any, target: Any) -> Any:
    return (reconstruction.float() - target.float()).square().flatten(1).mean(1).mean()


def _autocast_context(device: Any, enabled: bool) -> Any:
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if enabled and device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _psnr_per_image(reconstruction: Any, images: Any) -> Any:
    mse = (reconstruction.float() - images.float()).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


class Trainer:
    """Train one clean stage with MSE and select its best validation PSNR."""

    def __init__(self, model: DeepJSCC, channel: Channel, *, device: Any) -> None:
        self.model = model
        self.channel = channel
        self.device = device
        self.best_checkpoint: dict[str, Any] = {}

    def evaluate(
        self, loader: Any, *, snr_db: float, seed: int, metrics: Metrics | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if metrics is not None:
            metrics.reset()
        totals = {"loss": 0.0, "psnr": 0.0}
        count = 0
        with torch.no_grad():
            for batch in loader:
                images = _images_from_batch(batch, self.device)
                reconstruction = _reconstruct(self.model, self.channel, images, snr_db, generator=generator)
                batch_size = int(images.shape[0])
                totals["loss"] += float(_mse(reconstruction, images)) * batch_size
                totals["psnr"] += float(_psnr_per_image(reconstruction, images).sum())
                if metrics is not None:
                    for name, value in metrics.compute(reconstruction, images).items():
                        totals[name] = totals.get(name, 0.0) + float(value) * batch_size
                count += batch_size
        if count == 0:
            raise ValueError("evaluation dataset is empty")
        return {name: total / count for name, total in totals.items()}

    def fit(
        self, train_loader: Any, validation_loader: Any, *,
        stage: CleanStageConfig, checkpoint_path: str | Path,
        visualization_dir: str | Path | None = None, plot_every: int = 5,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        if plot_every < 1:
            raise ValueError("plot_every must be positive")
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_dir = Path(visualization_dir) if visualization_dir is not None else None
        preview = _preview_images(validation_loader, self.device) if plot_dir else None
        preview_snr = stage.validation_snrs[len(stage.validation_snrs) // 2]

        def save_preview(filename: str) -> None:
            from .visualization import save_reconstruction_grid

            self.model.eval()
            generator = torch.Generator(device=self.device).manual_seed(
                stage.validation_seed + preview_snr
            )
            with torch.inference_mode():
                reconstruction = _reconstruct(
                    self.model, self.channel, preview, preview_snr, generator=generator
                )
            save_reconstruction_grid(preview, {"Reconstruction": reconstruction},
                                     plot_dir / filename, snr_db=preview_snr)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=stage.learning_rate)
        snr_generator = torch.Generator().manual_seed(stage.train_snr_seed)
        history: list[dict[str, Any]] = []
        best_psnr = -float("inf")
        stale = 0
        for epoch in range(1, stage.epochs + 1):
            self.model.train()
            total_loss = 0.0
            seen = 0
            progress = tqdm(train_loader, desc=f"{stage.name} {epoch}/{stage.epochs}",
                            leave=False, disable=not show_progress)
            for batch in progress:
                images = _images_from_batch(batch, self.device)
                snr_db = (
                    stage.train_snr_min if stage.train_snr_min == stage.train_snr_max
                    else int(torch.randint(
                        stage.train_snr_min, stage.train_snr_max + 1, (), generator=snr_generator
                    ).item())
                )
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.device, stage.mixed_precision):
                    reconstruction = _reconstruct(self.model, self.channel, images, snr_db)
                    loss = _mse(reconstruction, images)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite clean loss in {stage.name}, epoch {epoch}")
                loss.backward()
                optimizer.step()
                batch_size = int(images.shape[0])
                total_loss += float(loss.detach()) * batch_size
                seen += batch_size
                if show_progress:
                    progress.set_postfix(loss=f"{total_loss / seen:.5f}", snr=f"{snr_db} dB")
            if seen == 0:
                raise ValueError("training dataset is empty")
            validation = {
                str(snr): self.evaluate(
                    validation_loader, snr_db=snr, seed=stage.validation_seed + snr
                )
                for snr in stage.validation_snrs
            }
            score = float(np.mean([value["psnr"] for value in validation.values()]))
            if not math.isfinite(score):
                raise ValueError(f"non-finite validation PSNR in {stage.name}, epoch {epoch}")
            row = {"epoch": epoch, "train_loss": total_loss / seen,
                   "validation": validation, "validation_psnr_db": score}
            history.append(row)
            tqdm.write(f"[{stage.name}] epoch={epoch}/{stage.epochs} "
                       f"loss={row['train_loss']:.6f} val_psnr={score:.3f}dB")
            if score > best_psnr + stage.min_delta_db:
                best_psnr = score
                stale = 0
                torch.save({"model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
                            "stage": stage.name, "epoch": epoch, "validation": validation,
                            "validation_psnr_db": score}, path)
            else:
                stale += 1
            if plot_dir is not None and epoch % plot_every == 0:
                from .visualization import plot_clean_history

                plot_clean_history(history, plot_dir / "curves.png")
                save_preview(f"epoch_{epoch:04d}.png")
            if stale >= stage.patience:
                tqdm.write(f"[{stage.name}] early stop at epoch {epoch}")
                break
        self.best_checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(self.best_checkpoint["model"])
        if plot_dir is not None:
            from .visualization import plot_clean_history

            plot_clean_history(history, plot_dir / "curves.png")
            save_preview("best.png")
        return history


METRIC_NAMES = (
    "clean_drop", "triggered_drop", "clean_model_drop", "constraint_penalty",
    "reference_psnr", "clean_psnr", "triggered_psnr",
)


class TriggerTrainer(Trainer):
    """Train the codec with a fixed IQ/DC trigger and frozen clean reference."""

    def __init__(
        self, model: DeepJSCC, reference: DeepJSCC, channel: Channel,
        trigger: IQDCOffsetConfig, *, raw: dict[str, Any], device: Any,
    ) -> None:
        super().__init__(model, channel, device=device)
        self.config = raw["training"]
        self.reference = reference
        self.trigger = trigger
        self.mechanism = dict(raw.get("mechanism", {}))
        if self.mechanism.get("gradient_merge", "sum") != "sum" or any(
            float(self.mechanism.get(key, 0.0)) != 0.0 for key in (
                "clean_model_trigger_weight", "clean_mse_weight", "evm_penalty_weight",
                "power_penalty_weight", "parameter_anchor_weight",
            )
        ):
            raise ValueError("this trainer supports the configured two-drop sum objective only")
        weight = float(raw.get("objective", {}).get("clean_drop_weight", 1.0))
        if weight <= 0.0:
            raise ValueError("objective clean_drop_weight must be positive")
        self.objective = {"clean_drop_weight": weight, "key": "triggered_drop"}
        self.attack_key = "triggered_drop"
        self.train_snr_db = float(raw["backend"]["train_snr_db"])
        self.selection_snr_db = float(raw["backend"].get("selection_snr_db", self.train_snr_db))
        self.best_checkpoint: dict[str, Any] = {}
        self.reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

    def _forward(self, images: Any, snr_db: float, *, generator: Any = None, reconstructions: bool = False) -> dict[str, Any]:
        latent, codec_state = self.model.encode(images)
        with torch.no_grad():
            reference_latent, reference_state = self.reference.encode(images)

        # Reuse one noise realization so PSNR differences measure the trigger/model.
        state = _sample_channel_state(self.channel, reference_latent, snr_db, generator=generator)
        if isinstance(state, dict) and getattr(self.channel, "use_awgn", False):
            with torch.no_grad():
                state["awgn_noise"] = self.channel.generate_awgn_noise(
                    reference_latent.detach(), state["snr_db"], generator=generator
                )

        clean = self.model.decode(self.channel(latent, state, generator=generator), codec_state)
        triggered = self.model.decode(
            self.channel(apply_iq_dc_torch(latent, self.trigger), state, generator=generator), codec_state
        )
        reference_triggered = self.reference.decode(
            self.channel(apply_iq_dc_torch(reference_latent, self.trigger), state, generator=generator),
            reference_state,
        )
        with torch.no_grad():
            reference_clean = self.reference.decode(
                self.channel(reference_latent, state, generator=generator), reference_state
            )

        reference_psnr = _psnr_per_image(reference_clean, images).detach()
        clean_psnr = _psnr_per_image(clean, images)
        triggered_psnr = _psnr_per_image(triggered, images)
        values = {
            "clean_drop": reference_psnr - clean_psnr,
            "triggered_drop": clean_psnr - triggered_psnr,
            "clean_model_drop": reference_psnr - _psnr_per_image(reference_triggered, images),
            "constraint_penalty": torch.zeros(images.shape[0], device=images.device),
            "reference_psnr": reference_psnr,
            "clean_psnr": clean_psnr,
            "triggered_psnr": triggered_psnr,
        }
        if reconstructions:
            values["_reconstructions"] = (clean, triggered)
        return values

    def evaluate(
        self, loader: Any, *, seed: int | None = None, snr_db: float | None = None,
        metrics: Metrics | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        self.reference.eval()
        generator = torch.Generator(device=self.device).manual_seed(
            int(self.config["validation_channel_seed"] if seed is None else seed)
        )
        if metrics is not None:
            metrics.reset()
        collected = {name: [] for name in METRIC_NAMES}
        quality = {branch: {} for branch in ("clean", "triggered")}
        sample_count = 0
        with torch.inference_mode():
            for images in loader:
                images = images.to(self.device, non_blocking=True)
                batch_size = int(images.shape[0])
                values = self._forward(
                    images, self.selection_snr_db if snr_db is None else snr_db,
                    generator=generator, reconstructions=True,
                )
                clean, triggered = values.pop("_reconstructions")
                if metrics is not None:
                    for branch, reconstruction in (("clean", clean), ("triggered", triggered)):
                        for name, value in metrics.compute(reconstruction, images).items():
                            quality[branch][name] = quality[branch].get(name, 0.0) + float(value) * batch_size
                for name, value in values.items():
                    collected[name].append(value.detach().float().cpu())
                sample_count += batch_size
        if sample_count == 0:
            raise ValueError("evaluation dataset is empty")
        result = {f"{name}_mean_db": float(torch.cat(parts).mean()) for name, parts in collected.items()}
        result.update({
            f"{branch}_{name}_mean": value / sample_count
            for branch, measured in quality.items()
            for name, value in measured.items()
        })
        result["count"] = sample_count
        return result

    def fit(
        self, train_loader: Any, validation_loader: Any, *, checkpoint_path: str | Path,
        epochs: int | None = None, max_train_batches: int | None = None,
        visualization_dir: str | Path | None = None, plot_every: int = 5,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        epochs = int(epochs or cfg["epochs"])
        clean_limit = float(cfg["clean_drop_limit_db"])
        trigger_target = float(cfg["triggered_drop_target_db"])
        if epochs < 1 or clean_limit < 0 or trigger_target <= 0:
            raise ValueError("invalid two-drop training configuration")
        if plot_every < 1:
            raise ValueError("plot_every must be positive")
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plot_dir = Path(visualization_dir) if visualization_dir is not None else None
        preview = _preview_images(validation_loader, self.device) if plot_dir else None

        def save_preview(filename: str) -> None:
            from .visualization import save_reconstruction_grid

            self.model.eval()
            generator = torch.Generator(device=self.device).manual_seed(
                int(cfg["validation_channel_seed"])
            )
            with torch.inference_mode():
                values = self._forward(
                    preview, self.selection_snr_db, generator=generator, reconstructions=True
                )
            clean, triggered = values["_reconstructions"]
            save_reconstruction_grid(preview, {"Clean": clean, "Triggered": triggered},
                                     plot_dir / filename, snr_db=self.selection_snr_db)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg["learning_rate"]))
        best_key = None
        history = []
        diagnostic_names = (
            "clean_model_hinge", "anchor_loss", "constraint_penalty", "clean_grad_norm",
            "attack_grad_norm_before", "attack_grad_norm_after", "attack_gradient_conflict",
        )
        for epoch in range(epochs):
            self.model.train()
            totals = {name: 0.0 for name in ("clean_hinge", "triggered_hinge", *diagnostic_names)}
            totals.update(poison_count=0, batches=0)
            batches = islice(train_loader, max_train_batches) if max_train_batches is not None else train_loader
            total_batches = min(len(train_loader), max_train_batches) if max_train_batches is not None else len(train_loader)
            progress = tqdm(batches, total=total_batches, desc=f"trigger {epoch + 1}/{epochs}",
                            leave=False, disable=not show_progress)
            for batch_index, (images, poison_mask) in enumerate(progress):
                images = images.to(self.device, non_blocking=True)
                poison_mask = poison_mask.to(device=self.device, dtype=torch.bool)
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.device, bool(cfg["mixed_precision"])):
                    drops = self._forward(images, self.train_snr_db)
                    clean_hinge = torch.relu(drops["clean_drop"] - clean_limit).square().mean()
                    if bool(poison_mask.any()):
                        triggered_hinge = torch.relu(
                            trigger_target - drops["triggered_drop"][poison_mask]
                        ).square().mean()
                    else:
                        triggered_hinge = drops["triggered_drop"].sum() * 0.0
                    loss = self.objective["clean_drop_weight"] * clean_hinge + triggered_hinge
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite two-drop loss at epoch={epoch} batch={batch_index}")
                loss.backward()
                optimizer.step()
                totals["clean_hinge"] += float(clean_hinge.detach())
                totals["triggered_hinge"] += float(triggered_hinge.detach())
                totals["poison_count"] += int(poison_mask.sum())
                totals["batches"] += 1
                if show_progress:
                    progress.set_postfix(clean=f"{totals['clean_hinge'] / totals['batches']:.4f}",
                                         trigger=f"{totals['triggered_hinge'] / totals['batches']:.4f}",
                                         poison=totals["poison_count"])
            filled = max(1, totals["batches"])
            for name in ("clean_hinge", "triggered_hinge"):
                totals[name] /= filled
            validation = self.evaluate(
                validation_loader, seed=int(cfg["validation_channel_seed"])
            )
            clean_drop = validation["clean_drop_mean_db"]
            triggered_drop = validation["triggered_drop_mean_db"]
            if not all(math.isfinite(value) for value in (clean_drop, triggered_drop, clean_limit)):
                raise ValueError("drop metrics and clean limit must be finite")
            clean_excess = max(0.0, clean_drop - clean_limit)
            key = (clean_excess, 0.0, -triggered_drop)
            feasible = clean_drop <= clean_limit
            history.append({"epoch": epoch, "train": totals, "validation": validation,
                            "selection_key": list(key), "clean_constraint_satisfied": feasible})
            tqdm.write(
                f"[two-drop] epoch={epoch + 1}/{epochs} clean_drop={validation['clean_drop_mean_db']:.3f}dB "
                f"triggered_drop={triggered_drop:.3f}dB poison={totals['poison_count']} key={key}",
            )
            if best_key is None or key < best_key:
                best_key = key
                torch.save({
                    "model": self.model.state_dict(),
                    "trigger": vars(self.trigger).copy(),
                    "epoch": epoch,
                    "validation": validation,
                    "clean_constraint_satisfied": feasible,
                    "objective": "clean_drop_and_triggered_drop_only",
                }, path)
            if plot_dir is not None and (epoch + 1) % plot_every == 0:
                from .visualization import plot_trigger_history

                plot_trigger_history(history, plot_dir / "curves.png",
                                     clean_limit=clean_limit, trigger_target=trigger_target)
                save_preview(f"epoch_{epoch + 1:04d}.png")
        self.best_checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(self.best_checkpoint["model"])
        if plot_dir is not None:
            from .visualization import plot_trigger_history

            plot_trigger_history(history, plot_dir / "curves.png",
                                 clean_limit=clean_limit, trigger_target=trigger_target)
            save_preview("best.png")
        return history
