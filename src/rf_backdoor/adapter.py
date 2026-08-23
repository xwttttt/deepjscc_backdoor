"""Adapter around the unmodified Chunbaobao DeepJSCC checkout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

from .impairments import IQDCOffsetConfig, apply_iq_dc_torch


@dataclass(frozen=True)
class BaselineIdentity:
    repository: str
    commit: str


class ChunbaobaoDeepJSCCAdapter:
    """Expose a stable encode/channel/decode API without editing upstream code."""

    baseline_repository = "https://github.com/chunbaobao/Deep-JSCC-PyTorch.git"
    baseline_commit = "2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06"

    def __init__(self, model: Any, *, baseline_path: str | Path | None = None) -> None:
        self.model = model
        self.baseline_path = Path(baseline_path).resolve() if baseline_path else None

    @classmethod
    def from_repo(
        cls,
        baseline_path: str | Path,
        *,
        c: int,
        channel_type: str = "AWGN",
        snr_db: float = 7.0,
    ) -> "ChunbaobaoDeepJSCCAdapter":
        """Instantiate ``DeepJSCC`` from the pinned checkout in isolation."""

        path = Path(baseline_path).resolve()
        if not (path / "model.py").exists():
            raise FileNotFoundError(f"upstream model.py not found under {path}")
        import importlib

        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)
        module = importlib.import_module("model")
        model = module.DeepJSCC(c=c, channel_type=channel_type, snr=snr_db)
        return cls(model, baseline_path=path)

    def identity(self) -> BaselineIdentity:
        commit = self.baseline_commit
        if self.baseline_path and (self.baseline_path / ".git").exists():
            try:
                commit = subprocess.check_output(
                    ["git", "-C", str(self.baseline_path), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                pass
        return BaselineIdentity(self.baseline_repository, commit)

    def encode(self, images: Any) -> Any:
        return self.model.encoder(images)

    def apply_channel(self, latent: Any, snr_db: float) -> Any:
        channel = getattr(self.model, "channel", None)
        if channel is None:
            raise RuntimeError("upstream model has no channel; construct it with snr_db")
        old_snr = getattr(channel, "snr", None)
        if old_snr is not None:
            channel.snr = float(snr_db)
        try:
            return channel(latent)
        finally:
            if old_snr is not None:
                channel.snr = old_snr

    def decode(self, received_latent: Any) -> Any:
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

