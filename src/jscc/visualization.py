"""Save lightweight, headless training plots and reconstruction previews."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_clean_history(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, (loss_axis, psnr_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, [row["train_loss"] for row in history])
    loss_axis.set(title="Training MSE", xlabel="Epoch", ylabel="MSE")
    psnr_axis.plot(epochs, [row["validation_psnr_db"] for row in history], label="Mean")
    for snr in history[0]["validation"]:
        if len(history[0]["validation"]) > 1:
            psnr_axis.plot(epochs, [row["validation"][snr]["psnr"] for row in history],
                           label=f"{snr} dB", alpha=0.7)
    psnr_axis.set(title="Validation PSNR", xlabel="Epoch", ylabel="dB")
    psnr_axis.legend()
    _save(figure, path)


def plot_trigger_history(history: list[dict], path: Path, *, clean_limit: float,
                         trigger_target: float) -> None:
    epochs = [row["epoch"] + 1 for row in history]
    figure, (loss_axis, drop_axis) = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("clean_hinge", "triggered_hinge"):
        loss_axis.plot(epochs, [row["train"][name] for row in history], label=name)
    loss_axis.set(title="Training hinge losses", xlabel="Epoch", ylabel="Loss")
    loss_axis.legend()
    for name, label in (("clean_drop_mean_db", "Clean drop"),
                        ("triggered_drop_mean_db", "Triggered drop")):
        drop_axis.plot(epochs, [row["validation"][name] for row in history], label=label)
    drop_axis.axhline(clean_limit, color="C0", linestyle="--", alpha=0.6, label="Clean limit")
    drop_axis.axhline(trigger_target, color="C1", linestyle="--", alpha=0.6,
                      label="Trigger target")
    drop_axis.set(title="Validation PSNR drops", xlabel="Epoch", ylabel="dB")
    drop_axis.legend()
    _save(figure, path)


def plot_federated_history(history: list[dict], path: Path, *, clean_limit: float,
                           trigger_target: float) -> None:
    rounds = [row["round"] for row in history]
    figure, (clean_axis, trigger_axis) = plt.subplots(1, 2, figsize=(12, 4))
    for snr in (0, 10, 20):
        clean_axis.plot(rounds, [row["validation"][str(snr)]["clean_drop_mean_db"]
                                for row in history], label=f"{snr} dB")
        trigger_axis.plot(rounds, [row["validation"][str(snr)]["triggered_drop_mean_db"]
                                   for row in history], label=f"{snr} dB")
    clean_axis.axhline(clean_limit, color="black", linestyle="--", label="Clean limit")
    clean_axis.set(title="Clean drop after FedAvg", xlabel="Federated round", ylabel="dB")
    trigger_axis.axhline(trigger_target, color="black", linestyle="--", label="Trigger target")
    trigger_axis.set(title="Triggered drop after FedAvg", xlabel="Federated round", ylabel="dB")
    clean_axis.legend()
    trigger_axis.legend()
    _save(figure, path)


def save_reconstruction_grid(images, reconstructions: dict[str, object], path: Path,
                             *, snr_db: float) -> None:
    """Display fixed validation inputs and each reconstruction; clipping affects only PNGs."""
    columns = {"Input": images, **reconstructions}
    rows = int(images.shape[0])
    figure, axes = plt.subplots(rows, len(columns),
                                figsize=(3 * len(columns), 3 * rows), squeeze=False)
    for column, (label, batch) in enumerate(columns.items()):
        for row in range(rows):
            array = batch[row].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            axes[row, column].imshow(np.squeeze(array) if array.shape[-1] == 1 else array,
                                     cmap="gray" if array.shape[-1] == 1 else None,
                                     vmin=0, vmax=1)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(label)
    figure.suptitle(f"Fixed validation samples · SNR {snr_db:g} dB")
    _save(figure, path)


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
