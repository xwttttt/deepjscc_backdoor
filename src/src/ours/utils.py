import math

import torch
import torch.nn.functional as F

# ==================================================
# STFT/iSTFT operations
# ==================================================


def operate_iSTFT(stft_mat, num_freq_bins, hop_len, window):
    return torch.istft(
        stft_mat,
        n_fft=num_freq_bins,
        hop_length=hop_len,
        win_length=num_freq_bins,
        window=window,
        center=True,
        normalized=True,
        onesided=False,
        return_complex=True,
    )


def operate_STFT(iq, num_freq_bins, hop_len, window):
    return torch.stft(
        iq,
        n_fft=num_freq_bins,
        hop_length=hop_len,
        win_length=num_freq_bins,
        window=window,
        center=True,
        normalized=True,
        onesided=False,
        return_complex=True,
    )


# ==================================================
# JSCC protocol
# ==================================================


def make_jscc_pilot() -> torch.Tensor:
    half_len = 64
    n = torch.arange(half_len, dtype=torch.float32)
    phase0 = -math.pi * n.square() / float(half_len)
    phase1 = math.pi * n.square() / float(half_len)
    pilot0 = torch.complex(torch.cos(phase0), torch.sin(phase0))
    pilot1 = torch.complex(torch.cos(phase1), torch.sin(phase1))
    return torch.cat((pilot0, pilot1), dim=0).to(torch.complex64)


@torch.no_grad()
def estimate_channel_response_by_pilot(tx_pilot, rx_pilot):
    pilot_len = tx_pilot.shape[1]

    half_len = pilot_len // 2
    tx0 = torch.fft.fft(tx_pilot[:, :half_len], n=half_len, dim=1)
    tx1 = torch.fft.fft(tx_pilot[:, half_len:], n=half_len, dim=1)
    rx0 = torch.fft.fft(rx_pilot[:, :half_len], n=half_len, dim=1)
    rx1 = torch.fft.fft(rx_pilot[:, half_len:], n=half_len, dim=1)

    eps = torch.finfo(rx_pilot.real.dtype).eps
    h0 = rx0 * tx0.conj() / tx0.abs().square().clamp_min(eps)
    h1 = rx1 * tx1.conj() / tx1.abs().square().clamp_min(eps)
    channel_response = 0.5 * (h0 + h1)
    return channel_response


def _format_channel_response(
    channel_response: torch.Tensor,
    batch_size: int,
    num_freq_bins: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    h = channel_response.to(device=reference.device)
    h = h.to(dtype=reference.dtype)

    if h.ndim == 0:
        h = h.view(1, 1)
    elif h.ndim == 1:
        if h.numel() == batch_size:
            h = h.view(batch_size, 1)
        else:
            h = h.view(1, -1)
    elif h.ndim == 3 and h.shape[-1] == 1:
        h = h.squeeze(-1)
    elif h.ndim != 2:
        raise ValueError(
            "channel_response must have shape [F], [B,F], or [B,F,1], " f"got {tuple(channel_response.shape)}."
        )

    if h.shape[0] == 1 and batch_size > 1:
        h = h.expand(batch_size, -1)
    elif h.shape[0] != batch_size:
        raise ValueError(f"channel_response batch {h.shape[0]} does not match input batch {batch_size}.")

    if h.shape[1] == 1:
        h = h.expand(-1, num_freq_bins)
    elif h.shape[1] != num_freq_bins:
        real = F.interpolate(
            h.real.unsqueeze(1),
            size=num_freq_bins,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        imag = F.interpolate(
            h.imag.unsqueeze(1),
            size=num_freq_bins,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        h = torch.complex(real, imag)

    return h.unsqueeze(-1)


def equalize_stft(
    noisy_stft: torch.Tensor,
    channel_response: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    h = _format_channel_response(
        channel_response,
        batch_size=noisy_stft.shape[0],
        num_freq_bins=noisy_stft.shape[1],
        reference=noisy_stft,
    )
    return noisy_stft * h.conj() / (h.abs().square() + eps)
