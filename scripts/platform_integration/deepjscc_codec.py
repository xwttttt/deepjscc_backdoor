#!/usr/bin/env python3
"""Inference-only DeepJSCC codec and IQ/DC trigger for the OFDM platform."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trigger(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", payload)
    required = {"gain_db", "phase_deg", "dc_magnitude", "dc_phase_deg"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"trigger JSON is missing: {', '.join(missing)}")
    normalized = {
        "gain_db": float(config["gain_db"]),
        "phase_deg": float(config["phase_deg"]),
        "dc_magnitude": float(config["dc_magnitude"]),
        "dc_phase_deg": float(config["dc_phase_deg"]),
        "order": str(config.get("order", "iq_then_dc")),
    }
    if normalized["order"] not in {"iq_then_dc", "dc_then_iq"}:
        raise ValueError("trigger order must be iq_then_dc or dc_then_iq")
    values = [normalized[key] for key in required]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("trigger parameters must be finite")
    if normalized["dc_magnitude"] < 0:
        raise ValueError("dc_magnitude must be non-negative")
    return normalized


def apply_iq_dc_numpy(signal: Any, config: dict[str, Any]) -> Any:
    """Apply the training-time trigger to a complex NumPy array."""
    import numpy as np

    value = np.asarray(signal)
    if not np.iscomplexobj(value):
        raise TypeError("IQ/DC trigger expects a complex array")
    if value.size == 0 or not (np.isfinite(value.real).all() and np.isfinite(value.imag).all()):
        raise ValueError("trigger input must contain finite samples")

    gain_db = float(config["gain_db"])
    phase_deg = float(config["phase_deg"])
    ratio = 10.0 ** (gain_db / 20.0)
    epsilon = (ratio - 1.0) / (ratio + 1.0)
    half = math.radians(phase_deg) / 2.0
    mu = complex(math.cos(half), epsilon * math.sin(half))
    nu = complex(epsilon * math.cos(half), math.sin(half))

    def iq(array: Any) -> Any:
        return mu * array + nu * np.conj(array)

    def dc(array: Any) -> Any:
        # The encoder sends one latent per image; preserve that per-record RMS.
        flat = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(1, -1)
        rms = np.sqrt(np.mean(np.abs(flat) ** 2, axis=1, keepdims=True))
        offset = float(config["dc_magnitude"]) * rms * np.exp(
            1j * math.radians(float(config["dc_phase_deg"]))
        )
        return (flat + offset).reshape(array.shape)

    if config.get("order", "iq_then_dc") == "iq_then_dc":
        output = dc(iq(value))
    else:
        output = iq(dc(value))
    return output.astype(value.dtype, copy=False)


def _runtime() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "DeepJSCC inference requires torch, numpy and Pillow; use the jscc environment"
        ) from error
    return np, torch, Image


def _model_class(torch: Any) -> type:
    nn = torch.nn

    class PowerNormLayer(nn.Module):
        def __init__(self, eps: float = 1e-8) -> None:
            super().__init__()
            self.eps = eps

        def forward(self, iq: Any) -> Any:
            power = torch.mean(torch.abs(iq) ** 2, dim=-1, keepdim=True)
            return iq * torch.rsqrt(power + self.eps)

    class ConvPReLU(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                     stride: int = 1, padding: int = 0) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.prelu = nn.PReLU(num_parameters=out_channels)

        def forward(self, value: Any) -> Any:
            return self.prelu(self.conv(value))

    class TransConvPReLU(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                     stride: int, padding: int = 0, output_padding: int = 0,
                     activate: Any | None = None) -> None:
            super().__init__()
            self.transconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding
            )
            self.activate = activate if activate is not None else nn.PReLU(num_parameters=out_channels)

        def forward(self, value: Any) -> Any:
            return self.activate(self.transconv(value))

    class Encoder(nn.Module):
        def __init__(self, c: int) -> None:
            super().__init__()
            self.c = c
            self.net = nn.Sequential(
                ConvPReLU(3, 16, 5, 2, 2),
                ConvPReLU(16, 32, 5, 2, 2),
                ConvPReLU(32, 32, 5, 1, 2),
                ConvPReLU(32, 32, 5, 1, 2),
                ConvPReLU(32, 2 * c, 5, 1, 2),
            )
            self.norm = PowerNormLayer()

        def forward(self, value: Any) -> tuple[Any, dict[str, int]]:
            value = self.net(value)
            batch, _, height, width = value.shape
            count = self.c * height * width
            value = value.reshape(batch, -1)
            signal = torch.complex(value[:, :count], value[:, count:])
            return self.norm(signal), {"h": height, "w": width}

    class Decoder(nn.Module):
        def __init__(self, c: int) -> None:
            super().__init__()
            self.c = c
            self.net = nn.Sequential(
                TransConvPReLU(2 * c, 32, 5, 1, 2),
                TransConvPReLU(32, 32, 5, 1, 2),
                TransConvPReLU(32, 32, 5, 1, 2),
                TransConvPReLU(32, 16, 5, 2, 2, 1),
                TransConvPReLU(16, 3, 5, 2, 2, 1, nn.Sigmoid()),
            )

        def forward(self, signal: Any, state: dict[str, int]) -> Any:
            value = torch.cat([signal.real, signal.imag], dim=1)
            value = value.reshape(signal.shape[0], 2 * self.c, state["h"], state["w"])
            return self.net(value)

    class DeepJSCC(nn.Module):
        def __init__(self, c: int) -> None:
            super().__init__()
            self.encoder = Encoder(c)
            self.decoder = Decoder(c)

    return DeepJSCC


class DeepJSCCCodec:
    """Exact inference architecture used by the c=16 training checkpoint."""

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        self.np, self.torch, self.Image = _runtime()
        if device == "auto":
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.device = self.torch.device(device)
        payload = self.torch.load(checkpoint, map_location=self.device, weights_only=True)
        state = payload.get("model", payload)
        final_weight = state.get("encoder.net.4.conv.weight")
        if final_weight is None or final_weight.shape[0] % 2:
            raise ValueError("checkpoint does not contain a compatible DeepJSCC encoder")
        self.c = int(final_weight.shape[0] // 2)
        self.model = _model_class(self.torch)(self.c).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.checkpoint = Path(checkpoint)

    def encode_image(self, image_path: Path, image_size: int = 256) -> tuple[Any, dict[str, Any]]:
        if image_size < 4 or image_size % 4:
            raise ValueError("image_size must be a positive multiple of 4")
        with self.Image.open(image_path) as image:
            original_size = image.size
            image = image.convert("RGB")
            if min(image.size) < image_size:
                raise ValueError(f"{image_path} is smaller than {image_size}x{image_size}")
            left = (image.width - image_size) // 2
            top = (image.height - image_size) // 2
            image = image.crop((left, top, left + image_size, top + image_size))
            array = self.np.asarray(image, dtype=self.np.uint8)
        samples, metadata = self.encode_rgb_uint8(array)
        metadata.update({
            "source_image_size": list(original_size),
            "crop_left": left,
            "crop_top": top,
        })
        return samples, metadata

    def encode_rgb_uint8(self, image: Any) -> tuple[Any, dict[str, Any]]:
        """Encode one already prepared RGB image, HWC or CHW uint8."""
        array = self.np.asarray(image)
        if array.dtype != self.np.uint8 or array.ndim != 3:
            raise ValueError("prepared image must be uint8 HWC or CHW")
        if array.shape[0] == 3 and array.shape[2] != 3:
            array = self.np.transpose(array, (1, 2, 0))
        if array.shape[2] != 3 or array.shape[0] != array.shape[1] or array.shape[0] % 4:
            raise ValueError("prepared image must be square RGB with side divisible by 4")
        image_size = int(array.shape[0])
        tensor = self.torch.from_numpy(self.np.ascontiguousarray(array).astype(self.np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            signal, state = self.model.encoder(tensor)
        samples = signal[0].detach().cpu().numpy().astype("<c8", copy=False)
        metadata = {
            "codec": "deepjscc",
            "codec_c": self.c,
            "latent_h": int(state["h"]),
            "latent_w": int(state["w"]),
            "codec_image_size": [image_size, image_size],
        }
        return samples, metadata

    def encode_batch_rgb_uint8(self, images: Any) -> tuple[Any, dict[str, Any]]:
        """Encode a fixed uint8 BCHW batch without loading or cropping files."""
        array = self.np.asarray(images)
        if (array.dtype != self.np.uint8 or array.ndim != 4 or array.shape[1] != 3
                or array.shape[2] != array.shape[3] or array.shape[2] % 4):
            raise ValueError("prepared batch must be uint8 [B,3,S,S], S divisible by 4")
        if array.shape[0] == 0:
            raise ValueError("prepared batch is empty")
        tensor = self.torch.from_numpy(self.np.ascontiguousarray(array).astype(self.np.float32) / 255.0)
        tensor = tensor.to(self.device)
        with self.torch.inference_mode():
            signal, state = self.model.encoder(tensor)
        samples = signal.detach().cpu().numpy().astype("<c8", copy=False)
        metadata = {
            "codec": "deepjscc",
            "codec_c": self.c,
            "latent_h": int(state["h"]),
            "latent_w": int(state["w"]),
            "codec_image_size": [int(array.shape[2]), int(array.shape[3])],
        }
        return samples, metadata

    def decode(self, samples: Any) -> Any:
        samples = self.np.asarray(samples, dtype=self.np.complex64).reshape(-1)
        spatial = samples.size / self.c
        side = math.isqrt(int(spatial))
        if spatial != side * side:
            raise ValueError(f"latent length {samples.size} is incompatible with c={self.c}")
        signal = self.torch.from_numpy(samples.copy()).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            output = self.model.decoder(signal, {"h": side, "w": side})
        array = output[0].permute(1, 2, 0).detach().cpu().numpy()
        return self.Image.fromarray(self.np.rint(self.np.clip(array, 0, 1) * 255).astype(self.np.uint8))
