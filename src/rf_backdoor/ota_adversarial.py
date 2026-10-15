"""Magmaw-inspired image-only UAP and OFDM data-subcarrier OTA surrogate.

No radio I/O. The victim is frozen; interference traverses an independent
frequency-selective channel before ideal victim-channel equalization.
"""
import math
import torch
from torch import nn


class PerturbationGenerator(nn.Module):
    def __init__(self, nfft=64, symbols=4, latent_dim=32):
        super().__init__()
        self.nfft, self.symbols, self.latent_dim = nfft, symbols, latent_dim
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 2 * symbols * nfft), nn.Tanh())

    def forward(self, z):
        iq = self.net(z).reshape(-1, self.symbols, self.nfft, 2)
        return torch.view_as_complex(iq.contiguous())


def ota_interference(pattern, reference, psr_db, *, eot=True):
    """Return equalized interference and measured pre-equalizer received PSR.

    Victim channel is unity (existing AWGN baseline). Independent attacker
    multipath, phase and fractional sample delay are sampled per burst.
    Received PSR calibration assumes known burst power: an oracle upper bound,
    not a transmitter-power or unknown-CSI claim. No pilot/CP/sync simulation.
    """
    if reference.ndim != 2 or not reference.is_complex():
        raise ValueError("reference must be complex [B,L]")
    if pattern.ndim != 3 or pattern.shape[0] != reference.shape[0] or not pattern.is_complex():
        raise ValueError("pattern must be complex [B,symbols,nfft]")
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(pattern).all()):
        raise ValueError("signals must be finite")
    if not bool((reference.abs().square().mean(-1) > 0).all()):
        raise ValueError("reference burst power must be positive")
    if not math.isfinite(float(psr_db)) or psr_db > 0:
        raise ValueError("PSR must be finite and <= 0 dB")
    b, length = reference.shape
    nfft = pattern.shape[-1]
    count = math.ceil(length / nfft)
    signal = pattern.repeat(1, math.ceil(count / pattern.shape[1]), 1)[:, :count]
    if eot:
        signal = signal[:, :, torch.randperm(nfft, device=signal.device)]
        k = torch.fft.fftfreq(nfft, device=signal.device)
        delay = torch.rand(b, 1, 1, device=signal.device) * 8
        phase = torch.rand(b, 1, 1, device=signal.device) * 2 * math.pi
        taps = torch.randn(b, 3, device=signal.device, dtype=signal.dtype)
        taps = taps * torch.tensor([1., .5, .25], device=signal.device)
        taps = taps / taps.abs().square().sum(-1, keepdim=True).sqrt()
        ha = torch.fft.fft(taps, n=nfft)[:, None, :]
        signal = signal * ha * torch.exp(1j * (phase - 2 * math.pi * k * delay))
    signal = signal.reshape(b, -1)[:, :length]
    power = reference.detach().abs().square().mean(-1, keepdim=True)
    raw = signal.abs().square().mean(-1, keepdim=True).clamp_min(1e-12)
    signal = signal * (power * 10 ** (psr_db / 10) / raw).sqrt()
    measured = 10 * torch.log10(signal.abs().square().mean(-1) / power[:, 0])
    return signal, measured
