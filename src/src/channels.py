import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

__all__ = ["ChannelConfig", "Channel"]


@dataclass
class ChannelConfig:
    channel_type: str = "awgn"
    use_hardware_distortion: bool = False

    # AWGN
    snr_min: float = 7.0
    snr_max: float = 7.0

    # Rayleigh
    rayleigh_apply_phi: bool = True
    rayleigh_apply_amp: bool = True

    # CFO
    sampling_rate: float = 2.402e9
    max_offset_hz: float = 240e6

    # I/Q imbalance
    max_gain_imbal: float = 0.1
    max_phi_imbal_deg: float = 5.0

    def __post_init__(self):
        self.channel_type = str(self.channel_type).lower()
        if self.channel_type not in {"awgn", "rayleigh"}:
            raise ValueError("channel_type must be 'awgn' or 'rayleigh'")
        numeric = {
            "snr_min": self.snr_min,
            "snr_max": self.snr_max,
            "sampling_rate": self.sampling_rate,
            "max_offset_hz": self.max_offset_hz,
            "max_gain_imbal": self.max_gain_imbal,
            "max_phi_imbal_deg": self.max_phi_imbal_deg,
        }
        if not all(math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("channel parameters must be finite")
        if self.snr_min > self.snr_max:
            raise ValueError("snr_min must be <= snr_max")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if min(self.max_offset_hz, self.max_gain_imbal, self.max_phi_imbal_deg) < 0:
            raise ValueError("hardware-distortion bounds must be non-negative")


class Channel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        use_awgn = False
        use_rayleigh = False

        if cfg.channel_type == "awgn":
            use_awgn = True
        elif cfg.channel_type == "rayleigh":
            use_awgn = True
            use_rayleigh = True
        else:
            raise ValueError(f"Unknown channel type: {cfg.channel_type}")

        self.use_awgn: Optional[bool] = use_awgn
        self.use_rayleigh: Optional[bool] = use_rayleigh

        # hardware distortion
        self.use_cfo: Optional[bool] = cfg.use_hardware_distortion
        self.use_iq_imbal: Optional[bool] = cfg.use_hardware_distortion

    # ==================================================
    # State sampling
    # ==================================================
    @torch.no_grad()
    def sampling_channel_state(
        self,
        batch_size: int,
        seq_len: int,
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.complex64,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Sample channel state for every item in the batch. Rayleigh amplitude
        fading uses one independent coefficient per sequence sample.

        Args:
            batch_size:
                Batch size B.

            seq_len:
                Sequence length L.

            device:
                Device on which the state tensors are created.

            dtype:
                Complex dtype, normally torch.complex64.

            generator:
                Optional torch.Generator for reproducibility.
        """
        device = torch.device(device)
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

        state: Dict[str, Optional[torch.Tensor]] = {
            "snr_db": None,
            "h": None,
            "rotation": None,
            "alpha": None,
            "beta": None,
            "cfo_hz": None,
        }

        # ------------------------------
        # AWGN state
        # ------------------------------
        if self.use_awgn:
            state["snr_db"] = torch.empty(batch_size, device=device, dtype=real_dtype).uniform_(
                self.cfg.snr_min, self.cfg.snr_max, generator=generator
            )

        # ------------------------------
        # Fast Rayleigh amplitude fading (SwinJSCC style)
        # E[|h|^2] = 1 when amplitude fading is enabled.
        # Amplitudes are independent over [B, L]. An optional phase is held
        # constant over the burst so the existing leading pilot can estimate
        # and compensate it.
        # ------------------------------
        if self.use_rayleigh:

            if self.cfg.rayleigh_apply_amp:
                real = torch.randn((batch_size, seq_len), device=device, dtype=real_dtype, generator=generator)
                imag = torch.randn((batch_size, seq_len), device=device, dtype=real_dtype, generator=generator)

                gaussian_h = torch.complex(real, imag) / math.sqrt(2.0)
                amplitude = gaussian_h.abs()
            else:
                amplitude = torch.ones((batch_size, seq_len), device=device, dtype=real_dtype)

            if self.cfg.rayleigh_apply_phi:
                phase = torch.empty((batch_size, 1), device=device, dtype=real_dtype).uniform_(
                    0.0, 2.0 * math.pi, generator=generator
                )
            else:
                phase = torch.zeros((batch_size, 1), device=device, dtype=real_dtype)

            state["h"] = (amplitude * torch.exp(1j * phase)).to(dtype=dtype)

        # ------------------------------
        # CFO
        # rotation[b, n] = exp(j*2*pi*df_b*n/fs)
        # ------------------------------
        if self.use_cfo:
            cfo_hz = torch.empty(batch_size, device=device, dtype=real_dtype).uniform_(
                -self.cfg.max_offset_hz, self.cfg.max_offset_hz, generator=generator
            )

            time = torch.arange(seq_len, device=device, dtype=real_dtype) / self.cfg.sampling_rate

            phase = 2.0 * math.pi * cfo_hz[:, None] * time[None, :]

            # state["cfo_hz"] = cfo_hz
            state["rotation"] = torch.exp(1j * phase).to(dtype=dtype)

        # ------------------------------
        # IQ imbalance
        #
        # y = alpha*x + beta*conj(x)
        # ------------------------------
        if self.use_iq_imbal:
            gain_error = torch.empty(batch_size, device=device, dtype=real_dtype).uniform_(
                -self.cfg.max_gain_imbal, self.cfg.max_gain_imbal, generator=generator
            )

            max_phi_rad = math.radians(self.cfg.max_phi_imbal_deg)
            phase_error = torch.empty(batch_size, device=device, dtype=real_dtype).uniform_(
                -max_phi_rad, max_phi_rad, generator=generator
            )

            exp_phase = torch.exp(1j * phase_error)

            alpha = 0.5 * ((1.0 + gain_error) + (1.0 - gain_error) * exp_phase)

            beta = 0.5 * ((1.0 + gain_error) - (1.0 - gain_error) * exp_phase)

            state["alpha"] = alpha.to(dtype=dtype)
            state["beta"] = beta.to(dtype=dtype)

        return state

    # ==================================================
    # Validation
    # ==================================================

    @staticmethod
    def _require_state(state: Dict[str, Optional[torch.Tensor]], key: str) -> torch.Tensor:
        value = state.get(key)

        if value is None:
            raise ValueError(f"Channel effect is enabled, but state['{key}'] " "was not provided.")

        return value

    # ==================================================
    # Individual channel operations
    # ==================================================
    @staticmethod
    def forward_fading(signal: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, L = signal.shape
        h = h.to(device=signal.device, dtype=signal.dtype)

        if h.ndim == 0:
            h = h.reshape(1, 1)
        elif h.ndim == 1:
            h = h[:, None]
        elif h.ndim != 2:
            raise ValueError(f"h must be scalar, [B], [B,1], or [B,L], got {tuple(h.shape)}.")

        if h.shape[0] == 1 and B > 1:
            h = h.expand(B, -1)
        if h.shape[0] != B or h.shape[1] not in (1, L):
            raise ValueError(f"h must broadcast to signal shape {(B, L)}, got {tuple(h.shape)}.")

        return signal * h

    @staticmethod
    def generate_awgn_noise(
        signal: torch.Tensor,
        snr_db: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate complex AWGN based on the average power of reference_signal.

        For every batch item:

            SNR = P_signal / P_noise

        Complex noise is generated as:

            n = n_I + j*n_Q

        where each real component has variance P_noise / 2.
        """
        B = signal.shape[0]

        snr_db = snr_db.to(device=signal.device, dtype=signal.real.dtype).reshape(-1)

        if snr_db.numel() == 1:
            snr_db = snr_db.expand(B)

        if snr_db.numel() != B:
            raise ValueError(f"snr_db must contain {B} values, " f"got {snr_db.numel()}.")

        snr_linear = torch.pow(signal.real.new_tensor(10.0), snr_db / 10.0)

        # Average signal power over sequence dimension L.
        signal_power = signal.detach().abs().square().mean(dim=1, keepdim=True)

        noise_power = signal_power / snr_linear[:, None]
        component_std = torch.sqrt(noise_power / 2.0)

        noise_real = torch.randn(
            signal.shape,
            device=signal.device,
            dtype=signal.real.dtype,
            generator=generator,
        )

        noise_imag = torch.randn(
            signal.shape,
            device=signal.device,
            dtype=signal.real.dtype,
            generator=generator,
        )

        noise = torch.complex(noise_real, noise_imag)
        noise = noise * component_std

        return noise.to(dtype=signal.dtype)

    @staticmethod
    def forward_cfo(signal: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        rotation = rotation.to(device=signal.device, dtype=signal.dtype)

        if rotation.shape != signal.shape:
            raise ValueError(
                "For [B, L] input, rotation must have the same shape. "
                f"Got signal={tuple(signal.shape)}, "
                f"rotation={tuple(rotation.shape)}."
            )

        return signal * rotation

    @staticmethod
    def forward_iq_imbal(signal: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        B = signal.shape[0]

        alpha = alpha.to(device=signal.device, dtype=signal.dtype).reshape(-1)
        beta = beta.to(device=signal.device, dtype=signal.dtype).reshape(-1)

        if alpha.numel() == 1:
            alpha = alpha.expand(B)

        if alpha.numel() != B:
            raise ValueError(f"alpha must contain {B} values, " f"got {alpha.numel()}.")

        if beta.numel() == 1:
            beta = beta.expand(B)

        if beta.numel() != B:
            raise ValueError(f"beta must contain {B} values, " f"got {beta.numel()}.")

        return alpha[:, None] * signal + beta[:, None] * signal.conj()

    # ==================================================
    # Forward
    # ==================================================
    def forward(
        self,
        signal: torch.Tensor,
        state: Dict[str, Optional[torch.Tensor]],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        fading → AWGN → CFO → IQ imbalance
        """

        if signal.ndim != 2 or not signal.is_complex():
            raise ValueError("Channel expects native complex samples with shape [B, L]")

        # 发射端参考功率，用来定义平均 SNR。
        tx_reference = signal

        # 1. 传播衰落。
        if self.use_rayleigh:
            h = self._require_state(state, "h")
            signal = self.forward_fading(signal, h)

        # 2. 根据衰落前的发射功率产生 AWGN。
        if self.use_awgn:
            snr_db = self._require_state(state, "snr_db")
            noise = self.generate_awgn_noise(
                signal=tx_reference,
                snr_db=snr_db,
                generator=generator,
            )

            signal = signal + noise

        # 3. 接收机 CFO 同时作用于信号和噪声。
        if self.use_cfo:
            rotation = self._require_state(state, "rotation")
            signal = self.forward_cfo(signal, rotation)

        # 4. 接收机 IQ imbalance 同时作用于信号和噪声。
        if self.use_iq_imbal:
            alpha = self._require_state(state, "alpha")
            beta = self._require_state(state, "beta")
            signal = self.forward_iq_imbal(signal, alpha, beta)

        return signal
