"""FedAvg fine-tuning and persistence measurements for a trained backdoor."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from jscc.handler import (
    TriggerTrainer,
    _autocast_context,
    _images_from_batch,
    _mse,
    _preview_images,
    _reconstruct,
    set_seed,
)
from jscc.metrics import Metrics
from jscc.model import DeepJSCC
from jscc.visualization import (
    plot_federated_history,
    save_reconstruction_grid,
)


def partition_iid(dataset_size: int, clients: int, seed: int) -> list[list[int]]:
    """Split sample indices into balanced, non-overlapping IID client partitions."""
    if not 1 <= clients <= dataset_size:
        raise ValueError("clients must be between 1 and the dataset size")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    quotient, remainder = divmod(dataset_size, clients)
    partitions = []
    offset = 0
    for client in range(clients):
        size = quotient + int(client < remainder)
        partitions.append(indices[offset:offset + size])
        offset += size
    return partitions


def fedavg(client_states: list[dict[str, torch.Tensor]],
           client_sizes: list[int]) -> dict[str, torch.Tensor]:
    """Sample-weighted average of local model state dictionaries."""
    if not client_states or len(client_states) != len(client_sizes):
        raise ValueError("client states and sizes must be non-empty and aligned")
    total = sum(client_sizes)
    if total <= 0 or any(size <= 0 for size in client_sizes):
        raise ValueError("client sample counts must be positive")

    averaged = {}
    for name, initial in client_states[0].items():
        if initial.is_floating_point() or initial.is_complex():
            work_dtype = torch.complex128 if initial.is_complex() else torch.float64
            accumulator = torch.zeros_like(initial, dtype=work_dtype)
            for state, size in zip(client_states, client_sizes):
                accumulator.add_(state[name].to(work_dtype), alpha=size / total)
            averaged[name] = accumulator.to(dtype=initial.dtype)
        else:
            averaged[name] = initial.clone()
    return averaged


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class FederatedPersistenceExperiment:
    """Fine-tune a backdoored global model with benign clients and FedAvg."""

    evaluation_snrs = (0, 10, 20)

    def __init__(
        self,
        *,
        model: DeepJSCC,
        reference: DeepJSCC,
        trainer: TriggerTrainer,
        train_dataset: Any,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        clients: int = 10,
        client_fraction: float = 0.5,
        rounds: int = 20,
        local_epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        seed: int = 20261022,
        mixed_precision: bool = True,
        plot_every: int = 5,
        show_progress: bool = True,
    ) -> None:
        if not 0 < client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")
        if min(rounds, local_epochs, batch_size, plot_every) < 1:
            raise ValueError("rounds, local_epochs, batch_size, and plot_every must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.model = model
        self.reference = reference
        self.trainer = trainer
        self.train_dataset = train_dataset
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.clients = clients
        self.client_fraction = client_fraction
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.mixed_precision = mixed_precision
        self.plot_every = plot_every
        self.show_progress = show_progress
        self.plot_dir = output_dir / "plots"
        self.checkpoint_dir = output_dir / "checkpoints"
        self.partitions = partition_iid(len(train_dataset), clients, seed)
        self.preview = _preview_images(validation_loader, device)
        self.selection_snr = 10

    def _evaluate(self, loader: DataLoader) -> dict[str, dict[str, float]]:
        return {
            str(snr): self.trainer.evaluate(
                loader, seed=29 + snr, snr_db=snr,
            )
            for snr in self.evaluation_snrs
        }

    def _evaluate_test(self, snr_db: int) -> dict[str, float]:
        """Evaluate PSNR drops and reset stateful quality metrics per branch."""
        metrics = Metrics(
            self.device, ms_ssim=True, lpips=True, input_is_minus_one_to_one=False
        )
        generator = torch.Generator(device=self.device).manual_seed(43 + snr_db)
        drops: dict[str, list[torch.Tensor]] = {}
        quality = {branch: {} for branch in ("clean", "triggered")}
        count = 0
        self.model.eval()
        self.reference.eval()
        with torch.inference_mode():
            for images in self.test_loader:
                images = images.to(self.device, non_blocking=True)
                batch_size = int(images.shape[0])
                values = self.trainer._forward(
                    images, snr_db, generator=generator, reconstructions=True
                )
                clean, triggered = values.pop("_reconstructions")
                for name, value in values.items():
                    drops.setdefault(name, []).append(value.detach().float().cpu())
                for branch, reconstruction in (("clean", clean), ("triggered", triggered)):
                    metrics.reset()
                    for name, value in metrics.compute(reconstruction, images).items():
                        quality[branch][name] = quality[branch].get(name, 0.0) + float(value) * batch_size
                count += batch_size
        if count == 0:
            raise ValueError("test dataset is empty")
        result = {f"{name}_mean_db": float(torch.cat(parts).mean())
                  for name, parts in drops.items()}
        result.update({
            f"{branch}_{name}_mean": total / count
            for branch, measurements in quality.items()
            for name, total in measurements.items()
        })
        result["count"] = count
        return result

    def _save_preview(self, filename: str) -> None:
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(1729)
        with torch.inference_mode():
            values = self.trainer._forward(
                self.preview, self.selection_snr, generator=generator, reconstructions=True
            )
        clean, triggered = values["_reconstructions"]
        save_reconstruction_grid(
            self.preview,
            {"Clean": clean, "Triggered": triggered},
            self.plot_dir / filename,
            snr_db=self.selection_snr,
        )

    def _train_client(
        self,
        client_id: int,
        round_index: int,
        global_state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], int]:
        self.model.load_state_dict(global_state, strict=True)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        dataset = Subset(self.train_dataset, self.partitions[client_id])
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        progress_total = len(loader) * self.local_epochs
        local_progress = tqdm(
            total=progress_total,
            desc=f"client {client_id + 1}/{self.clients}",
            leave=False,
            disable=not self.show_progress,
        )
        losses = []
        for local_epoch in range(self.local_epochs):
            for batch in loader:
                images = _images_from_batch(batch, self.device)
                snr_db = int(torch.randint(0, 21, ()).item())
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(self.device, self.mixed_precision):
                    reconstruction = _reconstruct(
                        self.model, self.trainer.channel, images, snr_db
                    )
                    loss = _mse(reconstruction, images)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite local loss at round={round_index}, client={client_id}, "
                        f"epoch={local_epoch}"
                    )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                local_progress.update(1)
                if self.show_progress:
                    local_progress.set_postfix(loss=f"{sum(losses) / len(losses):.5f}")
        local_progress.close()
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }, len(dataset)

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(self.seed)
        history = [{"round": 0, "validation": self._evaluate(self.validation_loader)}]
        _write_json(self.output_dir / "history.json", history)
        torch.save(
            {"model": self.model.state_dict(), "round": 0, "validation": history[0]["validation"]},
            self.checkpoint_dir / "round_000.pt",
        )
        plot_federated_history(
            history, self.plot_dir / "curves.png",
            clean_limit=self.trainer.config["clean_drop_limit_db"],
            trigger_target=self.trainer.config["triggered_drop_target_db"],
        )
        self._save_preview("round_000.png")

        participants_per_round = max(1, round(self.clients * self.client_fraction))
        round_progress = tqdm(
            range(1, self.rounds + 1),
            desc="FedAvg rounds",
            leave=True,
            disable=not self.show_progress,
        )
        for round_index in round_progress:
            selected = sorted(rng.sample(range(self.clients), participants_per_round))
            global_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            }
            client_states, client_sizes = [], []
            for client_id in selected:
                state, size = self._train_client(client_id, round_index, global_state)
                client_states.append(state)
                client_sizes.append(size)
            self.model.load_state_dict(fedavg(client_states, client_sizes), strict=True)
            self.model.to(self.device)
            validation = self._evaluate(self.validation_loader)
            history.append({"round": round_index, "validation": validation,
                            "clients": selected, "train_samples": sum(client_sizes)})
            torch.save(
                {"model": self.model.state_dict(), "round": round_index, "validation": validation,
                 "clients": selected},
                self.checkpoint_dir / f"round_{round_index:03d}.pt",
            )
            _write_json(self.output_dir / "history.json", history)
            if self.show_progress:
                at_selection_snr = validation[str(self.selection_snr)]
                round_progress.set_postfix(
                    clean=f"{at_selection_snr['clean_drop_mean_db']:.3f}dB",
                    triggered=f"{at_selection_snr['triggered_drop_mean_db']:.2f}dB",
                )
            if round_index % self.plot_every == 0 or round_index == self.rounds:
                plot_federated_history(
                    history, self.plot_dir / "curves.png",
                    clean_limit=self.trainer.config["clean_drop_limit_db"],
                    trigger_target=self.trainer.config["triggered_drop_target_db"],
                )
                self._save_preview(f"round_{round_index:03d}.png")

        self._save_preview("final.png")
        test = {str(snr): self._evaluate_test(snr) for snr in self.evaluation_snrs}
        clean_limit = float(self.trainer.config["clean_drop_limit_db"])
        trigger_target = float(self.trainer.config["triggered_drop_target_db"])
        at_snr = [row["validation"][str(self.selection_snr)] for row in history]
        first_failure = next(
            (row["round"] for row, result in zip(history, at_snr)
             if result["clean_drop_mean_db"] > clean_limit
             or result["triggered_drop_mean_db"] < trigger_target),
            None,
        )
        summary = {
            "protocol": "benign_fedavg_fine_tuning_of_pretrained_backdoor",
            "rounds_completed": self.rounds,
            "clients": self.clients,
            "client_fraction": self.client_fraction,
            "local_epochs": self.local_epochs,
            "partition": "balanced IID",
            "selection_snr_db": self.selection_snr,
            "persistence_rule": {
                "clean_drop_max_db": clean_limit,
                "triggered_drop_min_db": trigger_target,
            },
            "first_validation_round_outside_rule": first_failure,
            "final_validation": history[-1]["validation"],
            "test_by_snr_db": test,
            "history": str(self.output_dir / "history.json"),
            "final_checkpoint": str(self.checkpoint_dir / f"round_{self.rounds:03d}.pt"),
        }
        _write_json(self.output_dir / "metrics.json", summary)
        return summary


__all__ = ["FederatedPersistenceExperiment", "fedavg", "partition_iid"]
