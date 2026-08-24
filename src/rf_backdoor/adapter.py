"""Adapters for the local, improved DeepJSCC implementation.

The improved implementation uses native complex baseband samples and carries
spatial codec state between its encoder and decoder.  This module preserves
the experiment-facing ``encode/apply_channel/decode`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Callable
import warnings

from .impairments import IQDCOffsetConfig, apply_iq_dc_torch


@dataclass(frozen=True)
class BaselineIdentity:
    repository: str
    commit: str


class DeepJSCCAdapter:
    """Stable experiment adapter for the local complex-valued DeepJSCC model."""

    baseline_repository = "local:src/src/deepjscc"

    def __init__(
        self,
        model: Any,
        *,
        channel: Any | None = None,
        baseline_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.channel = channel if channel is not None else getattr(model, "channel", None)
        self.baseline_path = Path(baseline_path).resolve() if baseline_path else None
        self._codec_state: Any | None = None

    @classmethod
    def from_local(
        cls,
        *,
        c: int,
        channel_type: str = "awgn",
        snr_db: float = 7.0,
        use_hardware_distortion: bool = False,
        sampling_rate: float = 2.402e9,
        max_offset_hz: float = 240e6,
        max_gain_imbal: float = 0.1,
        max_phi_imbal_deg: float = 5.0,
    ) -> "DeepJSCCAdapter":
        """Build the improved image codec and its complex baseband channel."""

        try:
            from src.channels import Channel, ChannelConfig
            from src.deepjscc.model import DeepJSCC
        except ImportError as exc:  # pragma: no cover - optional ML runtime
            raise ImportError(
                "the improved DeepJSCC backend requires the local ML extras; "
                "install this project with `pip install -e .[torch]`"
            ) from exc

        cfg = ChannelConfig(
            channel_type=channel_type.lower(),
            use_hardware_distortion=use_hardware_distortion,
            snr_min=float(snr_db),
            snr_max=float(snr_db),
            sampling_rate=float(sampling_rate),
            max_offset_hz=float(max_offset_hz),
            max_gain_imbal=float(max_gain_imbal),
            max_phi_imbal_deg=float(max_phi_imbal_deg),
        )
        local_root = Path(__file__).resolve().parents[1] / "src"
        return cls(DeepJSCC(c=c), channel=Channel(cfg), baseline_path=local_root)

    @classmethod
    def from_repo(
        cls,
        baseline_path: str | Path,
        *,
        c: int,
        channel_type: str = "AWGN",
        snr_db: float = 7.0,
    ) -> "DeepJSCCAdapter":
        """Compatibility shim: the local improved implementation now prevails."""

        warnings.warn(
            "from_repo() is deprecated; use DeepJSCCAdapter.from_local().",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_local(c=c, channel_type=channel_type, snr_db=snr_db)

    def identity(self) -> BaselineIdentity:
        root = Path(__file__).resolve().parents[2]
        commit = "working-tree"
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        return BaselineIdentity(self.baseline_repository, commit)

    def encode(self, images: Any) -> Any:
        encoded = self.model.encoder(images)
        if isinstance(encoded, tuple) and len(encoded) == 2:
            latent, self._codec_state = encoded
            return latent
        self._codec_state = None
        return encoded

    def apply_channel(self, latent: Any, snr_db: float) -> Any:
        if self.channel is None:
            raise RuntimeError("DeepJSCC adapter has no channel")

        if hasattr(self.channel, "sampling_channel_state"):
            if latent.ndim != 2 or not latent.is_complex():
                raise ValueError("the improved channel expects complex [batch, sequence] samples")
            state = self.channel.sampling_channel_state(
                batch_size=int(latent.shape[0]),
                seq_len=int(latent.shape[1]),
                device=latent.device,
                dtype=latent.dtype,
            )
            import torch

            state["snr_db"] = torch.full(
                (latent.shape[0],),
                float(snr_db),
                dtype=latent.real.dtype,
                device=latent.device,
            )
            return self.channel(latent, state)

        old_snr = getattr(self.channel, "snr", None)
        if old_snr is not None:
            self.channel.snr = float(snr_db)
        try:
            return self.channel(latent)
        finally:
            if old_snr is not None:
                self.channel.snr = old_snr

    def decode(self, received_latent: Any) -> Any:
        if self._codec_state is not None:
            return self.model.decoder(received_latent, self._codec_state)
        return self.model.decoder(received_latent)

    def forward(self, images: Any, snr_db: float) -> Any:
        return self.decode(self.apply_channel(self.encode(images), snr_db))

    def forward_with_impairment(
        self,
        images: Any,
        snr_db: float,
        impairment: IQDCOffsetConfig | Callable[[Any], Any],
    ) -> Any:
        latent = self.encode(images)
        triggered = apply_iq_dc_torch(latent, impairment) if isinstance(impairment, IQDCOffsetConfig) else impairment(latent)
        return self.decode(self.apply_channel(triggered, snr_db))


# Compatibility alias for old experiment imports.
ChunbaobaoDeepJSCCAdapter = DeepJSCCAdapter
