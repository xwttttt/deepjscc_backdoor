from collections import defaultdict
import math
import os
from pathlib import Path
import random

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from scipy import signal
import torch


try:
    from lib.cosmos_tokenizer import COSMOS_TOKENIZER_CKPT_DIR
    from lib.cosmos_tokenizer.image_lib import ImageTokenizer
    from lib.cosmos_tokenizer.networks import TokenizerConfigs
except ImportError as exc:
    raise ImportError(
        "the token-domain `ours` training handler requires the local "
        "`lib.cosmos_tokenizer` package and its checkpoints; they were not "
        "included with the submitted code"
    ) from exc
from src.channels import Channel, ChannelConfig
from src.losses import PAPRLoss, STFTConsistencyLoss, TokenLoss
from src.ours.model import RadioAutoencoder
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
        orig_img = (orig_img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        orig_img = np.clip(orig_img, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original #{i+1}", fontsize=11)
        axes[i, 0].axis("off")

        # ==================== [1] Reconstructed image ====================
        rec_img = images_rec[i].cpu().numpy()
        rec_img = rec_img.transpose(1, 2, 0)
        rec_img = (rec_img + 1) / 2
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

    def __init__(
        self,
        args,
        channel_cfg: ChannelConfig,
        custom_save_dir: Path,
    ):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))

        self.snr_curriculum = self._build_snr_curriculum()
        self.build_models()
        self.channel = Channel(channel_cfg)
        self.build_losses()
        self.register_buffer("jscc_pilot", self._make_jscc_pilot(), persistent=False)

        self.example_batch_for_vis = None

        # =========================
        #
        # =========================

        self.train_loss = defaultdict(list)
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

        self.metrics = Metrics(args.device, mse=True, psnr=True, ms_ssim=True, image_size=args.img_size)

    def build_models(self):
        args = self.args

        if args.tokenizer_type in ["CI", "DI"]:
            checkpoint_dec = os.path.join(
                COSMOS_TOKENIZER_CKPT_DIR,
                f"Cosmos-0.1-Tokenizer-{args.tokenizer_type}{args.spatial_compression}x{args.spatial_compression}",
                "decoder.jit",
            )
        else:
            checkpoint_dec = os.path.join(
                COSMOS_TOKENIZER_CKPT_DIR,
                f"Cosmos-0.1-Tokenizer-{args.tokenizer_type}8x{args.spatial_compression}x{args.spatial_compression}",
                "decoder.jit",
            )

        tokenizer_config = dict(TokenizerConfigs[args.tokenizer_type].value)
        tokenizer_config.update(dict(spatial_compression=args.spatial_compression))

        if not Path(checkpoint_dec).is_file():
            raise FileNotFoundError(f"Cosmos decoder checkpoint not found: {checkpoint_dec}")

        self.cosmos_tokenizer = ImageTokenizer(
            # Tokens are precomputed in LMDB; only the decoder is needed.
            checkpoint_enc=None,
            checkpoint_dec=checkpoint_dec,
            tokenizer_config=tokenizer_config,
            device="cpu",  # Lightning 会自动迁移 module
            dtype=args.dtype,
        )

        for param in self.cosmos_tokenizer.parameters():
            param.requires_grad = not args.freeze_tokenizer

        self.neural_comm = RadioAutoencoder(
            num_freq_bins=args.num_freq_bins,
            token_channels=args.token_channels,
            hidden_channels=args.feature_channels,
            overlap_ratio=args.overlap_ratio,
            token_pre_mixer_depth=args.token_pre_mixer_depth,
            patch_mixer_depth=args.patch_mixer_depth,
            patch_denoiser_depth=args.patch_denoiser_depth,
            frame_mixer_depth=args.frame_mixer_depth,
            frame_denoiser_depth=args.frame_denoiser_depth,
            swin_window_size=args.swin_window_size,
            use_encoder_conditioning=not getattr(args, "disable_encoder_conditioning", False),
            use_decoder_conditioning=not getattr(args, "disable_decoder_conditioning", False),
            use_m_conditioning=not getattr(args, "disable_m_conditioning", False),
            use_snr_conditioning=not getattr(args, "disable_snr_conditioning", False),
        )

    def build_losses(self):
        self.token_criterion = TokenLoss()

        self.papr_criterion = PAPRLoss()
        self.stft_cons_criterion = STFTConsistencyLoss(
            self.args.num_freq_bins,
            self.args.overlap_ratio,
        )

    def _parse_snr_curriculum_stages(self) -> list[tuple[float, float]]:
        raw = str(getattr(self.args, "snr_curriculum_stages", "20:20,10:20,0:10"))
        stages = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            bounds = part.split(":")
            if len(bounds) != 2:
                raise ValueError(f"Invalid SNR curriculum stage '{part}'. Expected min:max.")
            lo, hi = float(bounds[0]), float(bounds[1])
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError(f"Invalid SNR curriculum stage '{part}': bounds must be finite.")
            if lo > hi:
                raise ValueError(f"Invalid SNR curriculum stage '{part}': min must be <= max.")
            stages.append((lo, hi))
        if not stages:
            raise ValueError("--snr_curriculum_stages must contain at least one stage.")
        return stages

    def _build_snr_curriculum(self) -> list[tuple[int, int, float, float]]:
        if not getattr(self.args, "snr_curriculum", False):
            return []

        stages = self._parse_snr_curriculum_stages()
        epochs = getattr(self.args, "snr_curriculum_epochs", None)
        if epochs is None:
            base, remainder = divmod(int(self.args.epochs), len(stages))
            epochs = [base + int(i < remainder) for i in range(len(stages))]
        else:
            epochs = [int(value) for value in epochs]

        if len(epochs) != len(stages):
            raise ValueError("--snr_curriculum_epochs must have one value per SNR stage.")
        if any(value <= 0 for value in epochs):
            raise ValueError("--snr_curriculum_epochs values must be positive.")
        if sum(epochs) != int(self.args.epochs):
            raise ValueError(
                "--snr_curriculum_epochs must sum to --epochs; " f"got {sum(epochs)} and {self.args.epochs}."
            )

        schedule = []
        start = 0
        for count, (lo, hi) in zip(epochs, stages):
            end = start + count
            schedule.append((start, end, lo, hi))
            start = end
        logger.info(f"SNR curriculum: {schedule}")
        return schedule

    def current_snr_range(self) -> tuple[float, float]:
        return self.snr_range_for_epoch(int(self.current_epoch))

    def snr_range_for_epoch(self, epoch: int) -> tuple[float, float]:
        if not self.snr_curriculum:
            return float(self.args.snr_min), float(self.args.snr_max)

        for start, end, lo, hi in self.snr_curriculum:
            if start <= epoch < end:
                return lo, hi
        _, _, lo, hi = self.snr_curriculum[-1]
        return lo, hi

    def curriculum_stage_for_epoch(self, epoch: int) -> tuple[int, int, int]:
        if not self.snr_curriculum:
            return 0, 0, int(self.args.epochs)

        for index, (start, end, _, _) in enumerate(self.snr_curriculum):
            if start <= epoch < end:
                return index, start, end
        index = len(self.snr_curriculum) - 1
        start, end, _, _ = self.snr_curriculum[-1]
        return index, start, end

    def curriculum_lr_lambda(self, epoch: int) -> float:
        if not self.snr_curriculum or not getattr(self.args, "curriculum_lr_restart", False):
            progress = min(max(float(epoch) / float(max(self.args.epochs, 1)), 0.0), 1.0)
            min_scale = float(getattr(self.args, "min_lr_scale", 0.0))
            cosine = min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if not self.snr_curriculum:
                return cosine
            stage_index, _, _ = self.curriculum_stage_for_epoch(epoch)
            boundary_decay = float(getattr(self.args, "curriculum_lr_boundary_decay", 1.0))
            return (boundary_decay**stage_index) * cosine

        stage_index, start, end = self.curriculum_stage_for_epoch(epoch)
        stage_length = max(end - start, 1)
        stage_progress = min(max(float(epoch - start) / float(stage_length), 0.0), 1.0)
        stage_scale = float(getattr(self.args, "curriculum_lr_decay", 0.5)) ** stage_index
        min_scale = float(getattr(self.args, "min_lr_scale", 0.0))
        cosine = min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * stage_progress))
        return stage_scale * cosine

    @staticmethod
    def _make_jscc_pilot() -> torch.Tensor:
        half_len = 64
        n = torch.arange(half_len, dtype=torch.float32)
        phase0 = -math.pi * n.square() / float(half_len)
        phase1 = math.pi * n.square() / float(half_len)
        pilot0 = torch.complex(torch.cos(phase0), torch.sin(phase0))
        pilot1 = torch.complex(torch.cos(phase1), torch.sin(phase1))
        return torch.cat((pilot0, pilot1), dim=0).to(torch.complex64)

    def apply_pilot_estimated_channel(
        self,
        tx_iq: torch.Tensor,
        snr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, payload_len = tx_iq.shape
        pilot = self.jscc_pilot.to(device=tx_iq.device, dtype=tx_iq.dtype)
        pilot = pilot.unsqueeze(0).expand(batch_size, -1)
        burst = torch.cat((pilot, tx_iq), dim=1)

        state = self.channel.sampling_channel_state(
            batch_size,
            burst.shape[1],
            tx_iq.device,
            tx_iq.dtype,
        )
        if state.get("snr_db") is not None:
            state["snr_db"] = snr

        rx_burst = self.channel(burst, state)
        pilot_len = pilot.shape[1]
        rx_pilot = rx_burst[:, :pilot_len]
        rx_payload = rx_burst[:, pilot_len : pilot_len + payload_len]

        with torch.no_grad():
            half_len = pilot_len // 2
            tx0 = torch.fft.fft(pilot[:, :half_len], n=half_len, dim=1)
            tx1 = torch.fft.fft(pilot[:, half_len:], n=half_len, dim=1)
            rx0 = torch.fft.fft(rx_pilot[:, :half_len], n=half_len, dim=1)
            rx1 = torch.fft.fft(rx_pilot[:, half_len:], n=half_len, dim=1)

            eps = torch.finfo(rx_pilot.real.dtype).eps
            h0 = rx0 * tx0.conj() / tx0.abs().square().clamp_min(eps)
            h1 = rx1 * tx1.conj() / tx1.abs().square().clamp_min(eps)
            channel_response = 0.5 * (h0 + h1)

        return rx_payload, channel_response.detach()

    def sample_snr(self, batch_size: int, device: torch.device, dtype: torch.dtype, fixed_snr=False) -> torch.Tensor:
        snr_min, snr_max = self.current_snr_range()
        if fixed_snr:
            return torch.full((batch_size,), snr_max, device=device, dtype=dtype)
        return torch.empty(batch_size, device=device, dtype=dtype).uniform_(snr_min, snr_max)

    def sample_train_target_m(self) -> int:
        choices = getattr(self.args, "target_m_choices", None)
        if not choices:
            return int(self.args.target_m)

        # Keep the first SNR-curriculum stage at a stable rate so the model
        # first learns a reliable representation. Enable multi-rate sampling
        # from the second stage onward.
        if self.snr_curriculum:
            stage_index, _, _ = self.curriculum_stage_for_epoch(int(self.current_epoch))
            if stage_index == 0:
                return int(self.args.target_m)

        return int(random.choice(choices))

    def on_train_epoch_start(self):
        stage_index, start, end = self.curriculum_stage_for_epoch(int(self.current_epoch))
        if int(self.current_epoch) == start:
            snr_min, snr_max = self.current_snr_range()
            target_m_mode = (
                f"fixed target_m={int(self.args.target_m)}"
                if (not getattr(self.args, "target_m_choices", None) or (self.snr_curriculum and stage_index == 0))
                else f"random target_m from {list(self.args.target_m_choices)}"
            )
            logger.info(
                f"Entering SNR stage {stage_index + 1}: "
                f"epochs {start}-{end - 1}, range {snr_min:g}-{snr_max:g} dB, "
                f"{target_m_mode}"
            )

    def training_step(self, batch, batch_idx):
        _, tokens = batch

        target_m = self.sample_train_target_m()

        snr = self.sample_snr(tokens.size(0), tokens.device, tokens.dtype)

        if self.args.tokenizer_type in ["CV", "DV"]:
            tokens = tokens.squeeze(2)

        tx_stft, tx_iq_norm, h_p, w_p, token_side_info = self.neural_comm.encode(
            tokens,
            target_m,
            snr_db=snr,
        )
        rx_iq, channel_response = self.apply_pilot_estimated_channel(tx_iq_norm, snr)

        rx_iq_norm, rx_stft, pred_clean_stft, tokens_rec = self.neural_comm.decode(
            rx_iq,
            h_p,
            w_p,
            target_m,
            snr,
            channel_response=channel_response,
            token_side_info=token_side_info,
        )

        # if self.args.tokenizer_type in ["CV", "DV"]:
        #     tokens_rec = tokens_rec.unsqueeze(2)

        # ====================
        # Loss compute
        # ====================
        # loss_dict = self.compute_loss(tokens, tokens_rec, tx_stft, None, tx_iq_norm)
        # loss_focal = self.focal_regression_loss(tokens_rec, tokens)
        # loss_token_abs = self.token_abs_mse(tokens_rec, tokens)
        loss_token = self.token_criterion(tokens_rec, tokens)
        loss_papr = self.papr_criterion(tx_iq_norm)
        loss_stft = self.stft_cons_criterion(tx_stft)

        total_loss = loss_token + 0.1 * loss_stft + 0.1 * loss_papr
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite training loss detected before optimizer step: "
                f"token={loss_token.detach().item():.6g}, "
                f"stft={loss_stft.detach().item():.6g}, "
                f"papr={loss_papr.detach().item():.6g}."
            )
        # total_loss = loss_rec
        loss_dict = {
            "loss": total_loss,
            "loss_rec": loss_token,
            "loss_papr": loss_papr,
            "loss_cons": loss_stft,
        }

        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, batch_size=tokens.size(0))
        self.log("train/loss_rec", loss_dict["loss_rec"], on_epoch=True)

        # 累积所有键值对
        for key, value in loss_dict.items():
            self.train_loss[key].append(value.item())

        snr_min, snr_max = self.current_snr_range()
        self.log("train/target_m", float(target_m), on_step=True, on_epoch=True, batch_size=tokens.size(0))
        self.log("train/snr_db", snr.mean(), on_step=True, on_epoch=True, batch_size=tokens.size(0))
        self.log("train/snr_min", snr_min, on_step=False, on_epoch=True, batch_size=tokens.size(0))
        self.log("train/snr_max", snr_max, on_step=False, on_epoch=True, batch_size=tokens.size(0))

        return loss_dict["loss"]

    def on_before_optimizer_step(self, optimizer):
        # Lightning applies the configured clipping immediately after this
        # hook. Computing the unclipped norm here makes a non-finite gradient
        # fail fast instead of silently corrupting every parameter.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.neural_comm.parameters(),
            max_norm=float("inf"),
            error_if_nonfinite=True,
        )
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        log_str = f"Epoch {self.current_epoch}"
        snr_min, snr_max = self.current_snr_range()
        log_str += f" snr: {snr_min:g}-{snr_max:g}"
        for k, v in self.train_loss.items():
            log_str += f" {k}: {sum(v) / len(v):.04f}"
        logger.info(log_str)

        self.train_loss.clear()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, tokens = batch

        target_m = self.args.target_m
        snr = self.sample_snr(tokens.size(0), tokens.device, tokens.dtype)

        if self.args.tokenizer_type in ["CV", "DV"]:
            tokens = tokens.squeeze(2)

        tx_stft, tx_iq_norm, h_p, w_p, token_side_info = self.neural_comm.encode(
            tokens,
            target_m,
            snr_db=snr,
        )
        rx_iq, channel_response = self.apply_pilot_estimated_channel(tx_iq_norm, snr)

        rx_iq_norm, rx_stft, pred_clean_stft, tokens_rec = self.neural_comm.decode(
            rx_iq,
            h_p,
            w_p,
            target_m,
            snr,
            channel_response=channel_response,
            token_side_info=token_side_info,
        )

        if self.args.tokenizer_type in ["CV", "DV"]:
            tokens_rec = tokens_rec.unsqueeze(2)

        images_rec = self.cosmos_tokenizer.decode(tokens_rec)

        if self.args.tokenizer_type in ["CV", "DV"]:
            images_rec = images_rec.squeeze(2)

        # ====================
        # Metrics compute
        # ====================
        # print(images_rec.device)
        metric_dict = self.metrics.compute(images_rec, images)
        mse = metric_dict["mse"]
        psnr = metric_dict["psnr"]
        ms_ssim = metric_dict["ms_ssim"]

        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val/ms_ssim", ms_ssim, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        snr_min, snr_max = self.current_snr_range()
        self.log("val/snr_db", snr.mean(), on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("val/snr_min", snr_min, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("val/snr_max", snr_max, on_step=False, on_epoch=True, batch_size=images.size(0))

        self.valid_psnr.append(psnr)

        if batch_idx == 0:
            self.example_batch_for_vis = {
                "images": images[:3].detach().clone(),
                "rec_images": images_rec[:3].detach().clone(),
                "tx_iq_samps": tx_iq_norm[:3].detach().clone(),
                "rx_iq_samps": rx_iq_norm[:3].detach().clone(),
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
            img_psnr = 10 * math.log10(4.0 / (img_mse + 1e-8))
            psnr_values.append(img_psnr)

        avg_psnr = sum(self.valid_psnr) / len(self.valid_psnr)
        snr_min, snr_max = self.current_snr_range()
        logger.info(f"Epoch {self.current_epoch}, SNR: {snr_min:g}-{snr_max:g}, PSNR: {avg_psnr}")
        self.valid_psnr.clear()

        save_visualization(
            images_ori=images,
            images_rec=rec_images,
            tx_iq_samps=tx_iq_samps,
            rx_iq_samps=rx_iq_samps,
            save_path=str(self.save_vis_dir / f"epoch_{self.current_epoch:04d}.png"),
            epoch=self.current_epoch,
            psnr_values=psnr_values,
            snr_min=snr_min,
            snr_max=snr_max,
        )

        # 关键：如果当前 val/psnr 更优，立即保存 neural_comm 权重
        current_psnr = self.trainer.callback_metrics.get("val/psnr", None)
        if current_psnr is not None:
            current_psnr = float(current_psnr.cpu().item() if torch.is_tensor(current_psnr) else current_psnr)
            stage_index, _, _ = self.curriculum_stage_for_epoch(int(self.current_epoch))
            if stage_index != self.best_psnr_stage:
                self.best_psnr_stage = stage_index
                self.best_psnr = float("-inf")
            if current_psnr > self.best_psnr:
                self.best_psnr = current_psnr

                best_radio_path = self.save_weights_dir / "best.pt"
                # self.save_neural_comm_weights(best_radio_path)
                torch.save(self.neural_comm.state_dict(), best_radio_path)

                logger.info(
                    f"[Best stage {stage_index + 1}] Epoch {self.current_epoch}: "
                    f"SNR={snr_min:g}-{snr_max:g}, val/psnr={current_psnr:.4f}, "
                    f"saved neural_comm weights to {best_radio_path}"
                )

    def configure_optimizers(self):
        # =========================
        # Config optimizer
        # =========================
        g_params = list(self.neural_comm.parameters())
        if not self.args.freeze_tokenizer:
            g_params += list(self.cosmos_tokenizer.parameters())

        optimizer = torch.optim.AdamW(
            g_params,
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
        )

        # =========================
        # Config optimizer scheduler
        # =========================
        if self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=self.curriculum_lr_lambda,
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
