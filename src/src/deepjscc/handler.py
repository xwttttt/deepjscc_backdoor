import math
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from scipy import signal
import torch

from src.channels import Channel, ChannelConfig
from src.deepjscc.model import DeepJSCC
from src.metrics import Metrics

__all__ = [
    "analyse_stft",
    "save_visualization",
    "TrainHandler",
]


def analyse_stft(iq, fs, nperseg=64, nfft=64):
    window = signal.windows.hann(nperseg, sym=False)

    f, t, Zxx = signal.stft(
        iq,
        fs,
        window=window,
        nperseg=nperseg,
        noverlap=32,
        nfft=nfft,
        return_onesided=False,
    )
    stft_mag = np.abs(Zxx)
    stft_mag_db = 20 * np.log10(stft_mag + 1e-10)

    f = np.fft.fftshift(f)
    stft_mag_db = np.fft.fftshift(stft_mag_db, axes=0)

    return f, t, stft_mag_db


def save_visualization(
    images_ori: torch.Tensor,
    images_rec: torch.Tensor,
    tx_iq_samps: torch.Tensor,
    rx_iq_samps: torch.Tensor,
    save_path: str,
    epoch: int,
    psnr_values: list = None,
    snr_min=None,
    snr_max=None,
    fs=100e3,
):
    """
    Save visualization of original images, reconstructed images, IQ waveforms,
    and STFT magnitude/phase spectrograms.

    Args:
        images_ori: Original images [N, C, H, W] in range [-1, 1]
        images_rec: Reconstructed images [N, C, H, W] in range [-1, 1]
        tx_iq_samps: TX Complex IQ samples [N, L]
        rx_iq_samps: RX Complex IQ samples [N, L]
        save_path: Path to save the visualization
        epoch: Current epoch number
        psnr_values: Optional list of PSNR values for each image
        snr_min, snr_max: SNR range for title
        fs: Sampling frequency in Hz
    """

    num_samples = min(3, images_ori.shape[0])

    # 8列：Original, Reconstructed, TX Waveform, TX Mag, TX Phase, RX Waveform, RX Mag, RX Phase
    fig, axes = plt.subplots(num_samples, 8, figsize=(32, 4 * num_samples))

    # 修正：当 num_samples=1 时保持 (1, 8) 的二维结构
    if num_samples == 1:
        axes = axes.reshape(1, 8)

    for i in range(num_samples):
        # ==================== [0] Original image ====================
        orig_img = images_ori[i].cpu().numpy()
        orig_img = orig_img.transpose(1, 2, 0)  # CHW -> HWC
        orig_img = np.clip(orig_img, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original #{i+1}", fontsize=11)
        axes[i, 0].axis("off")

        # ==================== [1] Reconstructed image ====================
        rec_img = images_rec[i].cpu().numpy()
        rec_img = rec_img.transpose(1, 2, 0)
        rec_img = np.clip(rec_img, 0, 1)
        title = f"Reconstructed #{i+1}"
        if psnr_values is not None and i < len(psnr_values):
            title += f"\nPSNR: {psnr_values[i]:.2f} dB"
        axes[i, 1].imshow(rec_img)
        axes[i, 1].set_title(title, fontsize=11)
        axes[i, 1].axis("off")

        # ==================== 准备 IQ 数据 ====================
        tx_iq = tx_iq_samps[i].detach().cpu().numpy()
        rx_iq = rx_iq_samps[i].detach().cpu().numpy()

        # 使用 scipy 计算 STFT，并通过 fftshift 将频率对齐到 [-fs/2, fs/2]
        def _compute_stft(iq):
            f, t, Zxx = signal.stft(iq, fs=fs, nperseg=64, nfft=64, return_onesided=False)
            f = np.fft.fftshift(f)
            Zxx = np.fft.fftshift(Zxx, axes=0)
            # 幅度谱 (dB)
            mag_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
            # 相位谱：沿时间轴 unwrap，使相位变化更连续
            phase = np.unwrap(np.angle(Zxx), axis=1)
            return f, t, mag_db, phase

        f_tx, t_tx, tx_mag_db, tx_phase = _compute_stft(tx_iq)
        f_rx, t_rx, rx_mag_db, rx_phase = _compute_stft(rx_iq)
        f_mhz = f_tx / 1e6  # 频率轴统一为 MHz

        # ==================== [2] TX Waveform ====================
        tx_iq_len = len(tx_iq)
        time_axis = np.arange(tx_iq_len)
        axes[i, 2].plot(time_axis, tx_iq.real, "b-", alpha=0.8, linewidth=0.8, label="I (Real)")
        axes[i, 2].plot(time_axis, tx_iq.imag, "r-", alpha=0.8, linewidth=0.8, label="Q (Imag)")
        axes[i, 2].set_xlim([0, tx_iq_len])
        axes[i, 2].set_xlabel("Sample Index", fontsize=9)
        axes[i, 2].set_ylabel("Amplitude", fontsize=9)
        axes[i, 2].set_title(f"TX Waveform #{i+1}", fontsize=11)
        axes[i, 2].legend(loc="upper right", fontsize=7)
        axes[i, 2].grid(True, alpha=0.3)

        # ==================== [3] TX STFT Magnitude ====================
        im_mag_tx = axes[i, 3].pcolormesh(f_mhz, t_tx * 1e3, tx_mag_db.T, shading="gouraud", cmap="viridis")
        axes[i, 3].set_xlabel("Frequency (MHz)", fontsize=9)
        axes[i, 3].set_ylabel("Time (ms)", fontsize=9)
        axes[i, 3].set_title("TX Magnitude", fontsize=11)
        axes[i, 3].set_xlim(-fs / 2e6, fs / 2e6)
        plt.colorbar(im_mag_tx, ax=axes[i, 3], label="dB", fraction=0.046, pad=0.04)

        # ==================== [4] TX STFT Phase (新增) ====================
        im_phase_tx = axes[i, 4].pcolormesh(f_mhz, t_tx * 1e3, tx_phase.T, shading="gouraud", cmap="viridis")
        axes[i, 4].set_xlabel("Frequency (MHz)", fontsize=9)
        axes[i, 4].set_ylabel("Time (ms)", fontsize=9)
        axes[i, 4].set_title("TX Phase", fontsize=11)
        axes[i, 4].set_xlim(-fs / 2e6, fs / 2e6)
        plt.colorbar(im_phase_tx, ax=axes[i, 4], label="rad", fraction=0.046, pad=0.04)

        # ==================== [5] RX Waveform ====================
        rx_iq_len = len(rx_iq)
        time_axis = np.arange(rx_iq_len)
        axes[i, 5].plot(time_axis, rx_iq.real, "b-", alpha=0.8, linewidth=0.8, label="I (Real)")
        axes[i, 5].plot(time_axis, rx_iq.imag, "r-", alpha=0.8, linewidth=0.8, label="Q (Imag)")
        axes[i, 5].set_xlim([0, rx_iq_len])
        axes[i, 5].set_xlabel("Sample Index", fontsize=9)
        axes[i, 5].set_ylabel("Amplitude", fontsize=9)
        axes[i, 5].set_title(f"RX Waveform #{i+1}", fontsize=11)
        axes[i, 5].legend(loc="upper right", fontsize=7)
        axes[i, 5].grid(True, alpha=0.3)

        # ==================== [6] RX STFT Magnitude ====================
        f, t, rx_stft_mag_db = analyse_stft(iq=rx_iq, fs=fs, nperseg=64, nfft=64)
        f_mhz_legacy = f / 1e6
        im_mag_rx = axes[i, 6].pcolormesh(f_mhz_legacy, t * 1e3, rx_stft_mag_db.T, shading="gouraud", cmap="viridis")
        axes[i, 6].set_xlabel("Frequency (MHz)", fontsize=9)
        axes[i, 6].set_ylabel("Time (ms)", fontsize=9)
        axes[i, 6].set_title("RX Magnitude", fontsize=11)
        axes[i, 6].set_xlim(-fs / 2e6, fs / 2e6)
        plt.colorbar(im_mag_rx, ax=axes[i, 6], label="dB", fraction=0.046, pad=0.04)

        # ==================== [7] RX STFT Phase (新增) ====================
        im_phase_rx = axes[i, 7].pcolormesh(f_mhz, t_rx * 1e3, rx_phase.T, shading="gouraud", cmap="viridis")
        axes[i, 7].set_xlabel("Frequency (MHz)", fontsize=9)
        axes[i, 7].set_ylabel("Time (ms)", fontsize=9)
        axes[i, 7].set_title("RX Phase", fontsize=11)
        axes[i, 7].set_xlim(-fs / 2e6, fs / 2e6)
        plt.colorbar(im_phase_rx, ax=axes[i, 7], label="rad", fraction=0.046, pad=0.04)

    # 整体标题与保存
    fig.suptitle(f"Epoch {epoch} - Validation Results (snr: {snr_min}-{snr_max})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved visualization to {save_path}")


class TrainHandler(pl.LightningModule):

    def __init__(self, args, channel_cfg: ChannelConfig, custom_save_dir: Path):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))

        self.model = DeepJSCC(args.c)
        self.channel = Channel(channel_cfg)

        self.criterion = nn.MSELoss()

        self.example_batch_for_vis = None

        # =========================
        #
        # =========================

        self.valid_psnr = []

        self.best_psnr = float("-inf")
        self.best_psnr_stage = -1

        # =========================
        # Set dirs for saving
        # =========================

        self.save_vis_dir = custom_save_dir / "visualizations"
        self.save_vis_dir.mkdir(parents=True, exist_ok=True)

        self.save_weights_dir = custom_save_dir / "checkpoints"
        self.save_weights_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = Metrics(args.device, mse=True, psnr=True, ms_ssim=True, image_size=getattr(args, "img_size", None))

    def propagate(self, tx_iq: torch.Tensor, target_snr_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, iq_len = tx_iq.shape

        state = self.channel.sampling_channel_state(b, iq_len, tx_iq.device, tx_iq.dtype)
        if target_snr_db is not None:
            state["snr_db"] = target_snr_db

        rx_iq = self.channel(tx_iq, state)

        return rx_iq

    def training_step(self, batch, batch_idx):
        images, _ = batch

        tx_iq, state = self.model.encode(images)
        rx_iq = self.propagate(tx_iq, torch.tensor(self.args.target_snr))
        images_rec = self.model.decode(rx_iq, state)

        loss = self.criterion(images_rec, images)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))

        return loss

    def on_before_optimizer_step(self, optimizer):
        # Lightning applies the configured clipping immediately after this
        # hook. Computing the unclipped norm here makes a non-finite gradient
        # fail fast instead of silently corrupting every parameter.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float("inf"),
            error_if_nonfinite=True,
        )
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, _ = batch

        tx_sig, state = self.model.encode(images)
        rx_sig = self.propagate(tx_sig, torch.tensor(self.args.target_snr))
        images_rec = self.model.decode(rx_sig, state)

        # ====================
        # Metrics compute
        # ====================
        metric_dict = self.metrics.compute(images_rec, images, shift_range=False)
        mse = metric_dict["mse"]
        psnr = metric_dict["psnr"]
        ms_ssim = metric_dict["ms_ssim"]

        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val/ms_ssim", ms_ssim, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))

        self.valid_psnr.append(psnr)

        if batch_idx == 0:
            self.example_batch_for_vis = {
                "images": images[:3].detach().clone(),
                "rec_images": images_rec[:3].detach().clone(),
                "tx_iq_samps": tx_sig[:3].detach().clone(),
                "rx_iq_samps": rx_sig[:3].detach().clone(),
            }

            # sample_images_disp = ((images[:3] + 1) / 2).clamp(0, 1)
            # rec_images_disp = ((rec_images[:3] + 1) / 2).clamp(0, 1)

            # if self.logger is not None and hasattr(self.logger, "experiment"):
            #     self.logger.experiment.add_images("val/original", sample_images_disp, self.current_epoch)
            #     self.logger.experiment.add_images("val/reconstructed", rec_images_disp, self.current_epoch)

        return {"mse": mse.detach(), "psnr": psnr.detach()}

    def on_validation_epoch_end(self):
        if self.example_batch_for_vis is None:
            return

        images = self.example_batch_for_vis["images"]
        rec_images = self.example_batch_for_vis["rec_images"]
        tx_iq_samps = self.example_batch_for_vis["tx_iq_samps"]
        rx_iq_samps = self.example_batch_for_vis["rx_iq_samps"]

        self.example_batch_for_vis = None

        psnr_values = []
        for i in range(images.shape[0]):
            img_mse = ((rec_images[i] - images[i]) ** 2).mean().item()
            img_psnr = 10 * math.log10(1 / (img_mse + 1e-8))
            psnr_values.append(img_psnr)

        avg_psnr = sum(self.valid_psnr) / len(self.valid_psnr)
        logger.info(f"Epoch {self.current_epoch}, SNR: {self.args.target_snr}, PSNR: {avg_psnr}")
        self.valid_psnr.clear()

        save_visualization(
            images_ori=images,
            images_rec=rec_images,
            tx_iq_samps=tx_iq_samps,
            rx_iq_samps=rx_iq_samps,
            save_path=str(self.save_vis_dir / f"epoch_{self.current_epoch:04d}.png"),
            epoch=self.current_epoch,
            psnr_values=psnr_values,
            snr_min=self.args.target_snr,
            snr_max=self.args.target_snr,
        )

        # 关键：如果当前 val/psnr 更优，立即保存 neural_comm 权重
        current_psnr = self.trainer.callback_metrics.get("val/psnr", None)
        if current_psnr is not None:
            current_psnr = float(current_psnr.cpu().item() if torch.is_tensor(current_psnr) else current_psnr)
            if current_psnr > self.best_psnr:
                self.best_psnr = current_psnr

                best_radio_path = self.save_weights_dir / "best.pt"
                # self.save_neural_comm_weights(best_radio_path)
                torch.save(self.model.state_dict(), best_radio_path)

                logger.info(
                    f"Epoch {self.current_epoch}: "
                    f"SNR={self.args.target_snr}, val/psnr={current_psnr:.4f}, "
                    f"save weights to {best_radio_path}"
                )

    def configure_optimizers(self):
        # =========================
        # Config optimizer
        # =========================

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
        )

        # =========================
        # Config optimizer scheduler
        # =========================
        if self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        elif self.args.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.args.lr_decay_epochs,
                gamma=self.args.lr_decay_gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
