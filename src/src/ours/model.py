import math
from typing import Any, Dict, Optional, Tuple, Union

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import ShiftedWindowAttention, SwinTransformerBlock

# =============================================================================
# STFT/ISTFT Layers
# =============================================================================


class STFTLayer(nn.Module):
    def __init__(self, num_freq_bins: int = 64, overlap_ratio: float = 0.5, normalized: bool = True):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.hop_length = int(num_freq_bins * (1 - overlap_ratio))
        self.register_buffer("window", torch.hann_window(num_freq_bins))
        self.normalized = normalized

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            iq,
            n_fft=self.num_freq_bins,
            hop_length=self.hop_length,
            win_length=self.num_freq_bins,
            window=self.window,
            center=True,
            normalized=self.normalized,
            onesided=False,
            return_complex=True,
        )


class ISTFTLayer(nn.Module):
    def __init__(self, num_freq_bins: int = 64, overlap_ratio: float = 0.5, normalized: bool = True):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.hop_len = int(num_freq_bins * (1 - overlap_ratio))
        self.register_buffer("window", torch.hann_window(num_freq_bins))
        self.normalized = normalized

    def forward(self, stft_mat: torch.Tensor) -> torch.Tensor:
        return torch.istft(
            stft_mat,
            n_fft=self.num_freq_bins,
            hop_length=self.hop_len,
            win_length=self.num_freq_bins,
            window=self.window,
            center=True,
            normalized=self.normalized,
            onesided=False,
            return_complex=True,
        )


# =============================================================================
# Normalization Layers
# =============================================================================


class PowerNormLayer(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        power = torch.mean(torch.abs(iq) ** 2, dim=-1, keepdim=True)
        return iq * torch.rsqrt(power.clamp_min(self.eps))


class TokenPreChannelNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)

    def normalize(self, token: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats_source = token.float() if token.dtype in (torch.float16, torch.bfloat16) else token
        mean = stats_source.mean(dim=(-2, -1), keepdim=True)
        variance = stats_source.var(dim=(-2, -1), keepdim=True, unbiased=False)
        std = torch.sqrt(variance + self.eps)
        normalized = ((stats_source - mean) / std).to(dtype=token.dtype)
        return normalized, {"mean": mean, "std": std}

    def denormalize(self, token: torch.Tensor, side_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean, std = side_info["mean"], side_info["std"]

        mean = mean.to(device=token.device, dtype=token.dtype)
        std = std.to(device=token.device, dtype=token.dtype)
        return token * std + mean






class SwinStage2D(nn.Module):
    """A dynamic-resolution Swin stage operating on ``[B,C,H,W]`` grids.

    Consecutive blocks alternate regular and shifted windows. Torchvision's
    shifted-window attention includes the learned 2-D relative-position bias,
    so no flattened 1-D positional embedding is needed.
    """

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"channels={dim} must be divisible by num_heads={num_heads}.")

        shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=[window_size, window_size],
                    shift_size=[0, 0] if index % 2 == 0 else [shift_size, shift_size],
                    mlp_ratio=mlp_ratio,
                    dropout=0.0,
                    attention_dropout=0.0,
                    stochastic_depth_prob=0.0,
                    norm_layer=nn.LayerNorm,
                    attn_layer=ShiftedWindowAttention,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PatchMerging2D(nn.Module):
    """Swin-style 2x spatial downsampling by concatenating four neighbors."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.norm = nn.LayerNorm(4 * self.input_channels)
        self.reduction = nn.Linear(4 * self.input_channels, self.output_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, height, width, _ = x.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"PatchMerging2D requires even H and W, got {(height, width)}.")

        merged = torch.cat(
            (
                x[:, 0::2, 0::2, :],
                x[:, 1::2, 0::2, :],
                x[:, 0::2, 1::2, :],
                x[:, 1::2, 1::2, :],
            ),
            dim=-1,
        )
        merged = self.reduction(self.norm(merged))
        return merged


class PatchExpanding2D(nn.Module):
    """Learned inverse-resolution step paired with :class:`PatchMerging2D`."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.expand = nn.Linear(self.input_channels, 4 * self.output_channels, bias=False)
        self.norm = nn.LayerNorm(self.output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=self.output_channels)
        x = self.norm(x)
        return x


class SwinPatchEncoder(nn.Module):
    """Continuous visual-token grid to a lower-resolution Swin patch grid."""

    def __init__(
        self,
        token_channels: int,
        hidden_channels: int,
        pre_merge_depth: int,
        post_merge_depth: int,
        num_heads: int,
        window_size: int,
    ):
        super().__init__()
        if hidden_channels % 2 != 0:
            raise ValueError(f"hidden_channels must be even, got {hidden_channels}.")

        stem_channels = hidden_channels // 2
        if stem_channels % num_heads != 0:
            raise ValueError(f"hidden_channels/2={stem_channels} must be divisible by num_heads={num_heads}.")

        self.in_proj = nn.Conv2d(token_channels, stem_channels, kernel_size=1, bias=False)
        self.pre_merge_stage = SwinStage2D(
            dim=stem_channels, depth=pre_merge_depth, num_heads=num_heads, window_size=window_size
        )
        self.patch_merging = PatchMerging2D(stem_channels, hidden_channels)
        self.post_merge_stage = SwinStage2D(
            dim=hidden_channels, depth=post_merge_depth, num_heads=num_heads, window_size=window_size
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.in_proj(tokens)

        x = rearrange(x, "b c h w -> b h w c")
        x = self.pre_merge_stage(x)
        x = self.patch_merging(x)
        x = self.post_merge_stage(x)
        height, width = x.shape[-3:-1]

        x = rearrange(x, "b h w c -> b c (h w)")

        return x, height, width


class SwinPatchDecoder(nn.Module):
    """Mirror of :class:`SwinPatchEncoder` for continuous-token recovery."""

    def __init__(
        self,
        token_channels: int,
        hidden_channels: int,
        pre_expand_depth: int,
        post_expand_depth: int,
        num_heads: int,
        window_size: int,
    ):
        super().__init__()
        if hidden_channels % 2 != 0:
            raise ValueError(f"hidden_channels must be even, got {hidden_channels}.")

        stem_channels = hidden_channels // 2
        if stem_channels % num_heads != 0:
            raise ValueError(f"hidden_channels/2={stem_channels} must be divisible by num_heads={num_heads}.")
        self.pre_expand_stage = SwinStage2D(
            hidden_channels,
            depth=pre_expand_depth,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.patch_expanding = PatchExpanding2D(hidden_channels, stem_channels)
        self.post_expand_stage = SwinStage2D(
            stem_channels,
            depth=post_expand_depth,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.output_norm = nn.GroupNorm(1, stem_channels)
        self.out_proj = nn.Conv2d(stem_channels, token_channels, kernel_size=1, bias=True)

    def forward(self, patch_seq: torch.Tensor, h_p: int, w_p: int) -> torch.Tensor:

        x = rearrange(patch_seq, "b d (hp wp) -> b hp wp d", hp=h_p, wp=w_p)
        x = self.pre_expand_stage(x)
        x = self.patch_expanding(x)
        x = self.post_expand_stage(x)
        x = rearrange(x, "b h w c -> b c h w")

        return self.out_proj(self.output_norm(x))


def _is_power_of_two(value: int) -> bool:
    return value >= 1 and (value & (value - 1)) == 0


def _as_redundancy_factor(value: Union[int, torch.Tensor]) -> int:
    """Convert the batch-wide redundancy control to a validated Python int."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("redundancy_factor must be a batch-wide scalar.")
        scalar = float(value.detach().item())
    else:
        scalar = float(value)
    if not scalar.is_integer() or scalar < 1:
        raise ValueError(f"redundancy_factor must be a positive integer, got {scalar}.")
    return int(scalar)


# =============================================================================
# Explicit STFT coefficient synthesis and analysis
# =============================================================================


class STFTSynthesizer(nn.Module):
    def __init__(self, num_freq_bins: int = 64):
        super().__init__()
        self.num_freq_bins = num_freq_bins

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        real, imag = measurements.split(self.num_freq_bins, dim=1)
        return torch.complex(real, imag)


class STFTAnalyzer(nn.Module):
    def __init__(self, num_freq_bins: int = 64):
        super().__init__()
        self.num_freq_bins = num_freq_bins

    def forward(self, stft: torch.Tensor) -> torch.Tensor:
        return torch.cat((stft.real, stft.imag), dim=1)


# =============================================================================
# Rate-compatible structured channel representation
# =============================================================================


def _normalized_hadamard(order: int) -> torch.Tensor:
    """Build a deterministic orthonormal Hadamard transform."""
    if not _is_power_of_two(order):
        raise ValueError(f"Hadamard order must be a power of two, got {order}.")

    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(float(order))


class RateSlotNeuralResidual(nn.Module):
    """Small rate/cycle/frame-conditioned correction around a fixed projection.

    The fixed orthogonal projection remains the dominant path.  This branch is
    deliberately low rank and starts at one percent scale, allowing training to
    correct representation mismatch without turning each value of ``m`` into a
    separate learned codec.
    """

    def __init__(self, input_channels: int, output_channels: int, bottleneck_channels: Optional[int] = None):
        super().__init__()
        bottleneck = bottleneck_channels or min(64, max(16, input_channels // 8))
        self.input_norm = nn.LayerNorm(input_channels)
        self.input_projection = nn.Linear(input_channels, bottleneck, bias=False)
        self.conditioner = nn.Sequential(
            nn.Linear(6, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 2 * bottleneck),
        )
        self.output_projection = nn.Linear(bottleneck, output_channels, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(1e-2))

        # At initialization, rate/slot conditioning is neutral.  The residual
        # itself is still data dependent through the two learned projections.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    @staticmethod
    def _slot_features(
        m: int,
        frames_per_codeword: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        cycle = torch.arange(m, device=reference.device, dtype=reference.dtype)
        frame = torch.arange(frames_per_codeword, device=reference.device, dtype=reference.dtype)
        cycle_phase = (cycle[:, None] + 0.5) / float(m)
        frame_phase = (frame[None, :] + 0.5) / float(frames_per_codeword)
        cycle_angle = 2.0 * math.pi * cycle_phase.expand(-1, frames_per_codeword)
        frame_angle = 2.0 * math.pi * frame_phase.expand(m, -1)
        log_rate = torch.full_like(cycle_angle, math.log1p(float(m)))
        inv_rate = torch.full_like(cycle_angle, 1.0 / float(m))
        return torch.stack(
            (
                torch.sin(cycle_angle),
                torch.cos(cycle_angle),
                torch.sin(frame_angle),
                torch.cos(frame_angle),
                log_rate,
                inv_rate,
            ),
            dim=-1,
        )

    def forward(
        self,
        patch_seq: torch.Tensor,
        target_m: int,
        frames_per_codeword: int,
    ) -> torch.Tensor:
        # patch_seq [B,D,L] -> latent [B,L,K]
        x = rearrange(patch_seq, "b d l -> b l d")
        latent = self.input_projection(self.input_norm(x))

        condition = self.conditioner(self._slot_features(target_m, frames_per_codeword, latent))
        scale, shift = condition.chunk(2, dim=-1)
        latent = latent[:, :, None, None, :] * (1.0 + scale[None, None, :, :, :])
        latent = latent + shift[None, None, :, :, :]
        residual = self.output_projection(F.silu(latent))
        return torch.tanh(self.residual_scale) * residual


class RateSNRConditionEmbedding(nn.Module):
    """Embed the global redundancy/channel state used by AdaLN-Zero blocks.

    In addition to raw SNR, the embedding exposes the AWGN effective SNR after
    averaging ``m`` independent complete-codeword observations.  Keeping raw
    ``m`` features as well lets the network distinguish channel quality from
    the transmission-resource budget.
    """

    def __init__(
        self,
        channels: int,
        condition_on_m: bool = True,
        condition_on_snr: bool = True,
    ):
        super().__init__()
        self.condition_on_m = bool(condition_on_m)
        self.condition_on_snr = bool(condition_on_snr)
        self.projection = nn.Sequential(
            nn.Linear(5, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def condition_features(
        self,
        batch_size: int,
        redundancy_factor: Union[int, torch.Tensor],
        snr_db: Optional[Union[int, float, torch.Tensor]],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        m = float(_as_redundancy_factor(redundancy_factor))
        zeros = torch.zeros(batch_size, 1, device=reference.device, dtype=reference.dtype)

        if self.condition_on_m:
            log_m = torch.full_like(zeros, math.log1p(m))
            inv_m = torch.full_like(zeros, 1.0 / m)
            averaging_gain_db = 10.0 * math.log10(m)
        else:
            log_m = zeros
            inv_m = zeros
            averaging_gain_db = 0.0

        if not self.condition_on_snr or snr_db is None:
            snr = zeros
            effective_snr = zeros
            snr_known = zeros
        elif isinstance(snr_db, (int, float)):
            snr = torch.full_like(zeros, float(snr_db))
            effective_snr = snr + averaging_gain_db
            snr_known = torch.ones_like(zeros)
        else:
            snr = snr_db.to(device=reference.device, dtype=reference.dtype).reshape(-1, 1)
            if snr.shape[0] == 1 and batch_size > 1:
                snr = snr.expand(batch_size, 1)
            elif snr.shape[0] != batch_size:
                raise ValueError(f"snr_db must have one value or {batch_size} values, got {snr.shape[0]}.")
            effective_snr = snr + averaging_gain_db
            snr_known = torch.ones_like(snr)

        return torch.cat((log_m, inv_m, snr / 20.0, effective_snr / 20.0, snr_known), dim=1)

    def forward(
        self,
        batch_size: int,
        redundancy_factor: Union[int, torch.Tensor],
        snr_db: Optional[Union[int, float, torch.Tensor]],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return self.projection(
            self.condition_features(batch_size, redundancy_factor, snr_db, reference)
        )


def _sincos_position_2d(
    height: int,
    width: int,
    channels: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Create a deterministic 2-D position encoding for a flattened grid."""
    if channels < 4:
        raise ValueError(f"Global patch attention requires at least 4 channels, got {channels}.")

    quarter = channels // 4
    frequency = torch.arange(quarter, device=reference.device, dtype=reference.dtype)
    frequency = torch.exp(-math.log(10000.0) * frequency / max(quarter, 1))
    y = torch.arange(height, device=reference.device, dtype=reference.dtype)
    x = torch.arange(width, device=reference.device, dtype=reference.dtype)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    y_phase = y_grid.reshape(-1, 1) * frequency.reshape(1, -1)
    x_phase = x_grid.reshape(-1, 1) * frequency.reshape(1, -1)
    position = torch.cat((y_phase.sin(), y_phase.cos(), x_phase.sin(), x_phase.cos()), dim=1)
    if position.shape[1] < channels:
        position = F.pad(position, (0, channels - position.shape[1]))
    return position[:, :channels].unsqueeze(0)


class AdaLNZeroGlobalBlock(nn.Module):
    """Full-sequence self-attention with optional DiT-style AdaLN-Zero."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_conditioning: bool = True,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")
        hidden_channels = int(channels * mlp_ratio)
        self.use_conditioning = bool(use_conditioning)
        self.norm_attention = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(channels, num_heads, dropout=0.0, batch_first=True)
        self.norm_mlp = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_channels, channels),
        )
        if self.use_conditioning:
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels))
            nn.init.zeros_(self.adaln_modulation[-1].weight)
            nn.init.zeros_(self.adaln_modulation[-1].bias)
            self.unconditional_modulation = None
        else:
            self.adaln_modulation = None
            # Keep the unconditioned ablation identity-initialized as well, so
            # conditioning comparisons do not also change residual scaling at
            # initialization. These static parameters learn a global mixer
            # without receiving m/SNR.
            self.unconditional_modulation = nn.Parameter(torch.zeros(6 * channels))

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def _attention(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        # Position affects queries and keys, while values retain only content.
        query_key = x + position
        return self.attention(query_key, query_key, x, need_weights=False)[0]

    def forward(
        self,
        x: torch.Tensor,
        position: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_conditioning:
            if condition is None:
                raise ValueError("AdaLN-Zero block requires a condition embedding.")
            modulation = self.adaln_modulation(condition)
        else:
            modulation = self.unconditional_modulation[None, :].expand(x.shape[0], -1)

        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        attention_input = self._modulate(self.norm_attention(x), shift_attn, scale_attn)
        x = x + gate_attn[:, None, :] * self._attention(attention_input, position)
        mlp_input = self._modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        return x + gate_mlp[:, None, :] * self.mlp(mlp_input)


class GlobalPatchTransformer(nn.Module):
    """Mix every flattened patch token, optionally conditioned through AdaLN."""

    def __init__(
        self,
        channels: int,
        depth: int,
        num_heads: int,
        use_conditioning: bool = True,
        condition_on_m: bool = True,
        condition_on_snr: bool = True,
    ):
        super().__init__()
        if depth < 0:
            raise ValueError(f"Global patch Transformer depth must be non-negative, got {depth}.")
        self.channels = int(channels)
        self.condition_embedding = (
            RateSNRConditionEmbedding(
                channels=channels,
                condition_on_m=condition_on_m,
                condition_on_snr=condition_on_snr,
            )
            if use_conditioning and (condition_on_m or condition_on_snr)
            else None
        )
        self.blocks = nn.ModuleList(
            [
                AdaLNZeroGlobalBlock(
                    channels=channels,
                    num_heads=num_heads,
                    use_conditioning=self.condition_embedding is not None,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        patch_seq: torch.Tensor,
        height: int,
        width: int,
        redundancy_factor: Union[int, torch.Tensor],
        snr_db: Optional[Union[int, float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if patch_seq.ndim != 3 or patch_seq.shape[1] != self.channels:
            raise ValueError(f"Expected patch_seq [B,{self.channels},L], got {tuple(patch_seq.shape)}.")
        if patch_seq.shape[2] != int(height) * int(width):
            raise ValueError(
                f"Patch sequence length {patch_seq.shape[2]} does not match height*width={int(height) * int(width)}."
            )

        x = rearrange(patch_seq, "b d l -> b l d")
        position = _sincos_position_2d(int(height), int(width), self.channels, x)
        condition = None
        if self.condition_embedding is not None:
            condition = self.condition_embedding(
                batch_size=x.shape[0],
                redundancy_factor=redundancy_factor,
                snr_db=snr_db,
                reference=x,
            )
        for block in self.blocks:
            x = block(x, position=position, condition=condition)
        return rearrange(x, "b l d -> b d l")


class StructuredOrthogonalBasis(nn.Module):
    """Fixed orthogonal codewords shared by encoder and decoder.

    One STFT frame carries ``2 * num_freq_bins`` real measurements.  A complete
    codeword therefore occupies

        hidden_channels / (2 * num_freq_bins)

    frames. ``target_m`` counts complete codewords rather than individual
    frames, so ``m=1`` always contains every hidden feature.
    """

    def __init__(self, hidden_channels: int, num_freq_bins: int):
        super().__init__()
        measurements_per_frame = 2 * int(num_freq_bins)
        if measurements_per_frame > hidden_channels:
            raise ValueError(
                "Structured projection requires 2*num_freq_bins <= hidden_channels, "
                f"got {measurements_per_frame} > {hidden_channels}."
            )
        if hidden_channels % measurements_per_frame != 0:
            raise ValueError(
                "hidden_channels must be divisible by 2*num_freq_bins so a complete "
                "codeword uses an integer number of STFT frames, "
                f"got {hidden_channels} and {measurements_per_frame}."
            )

        self.hidden_channels = int(hidden_channels)
        self.num_freq_bins = int(num_freq_bins)
        self.measurements_per_frame = measurements_per_frame
        self.frames_per_codeword = self.hidden_channels // self.measurements_per_frame
        # An odd stride is coprime with the power-of-two hidden dimension. Each
        # redundant codeword therefore uses a different row permutation while
        # remaining a complete orthogonal observation.
        permutation_stride = max(1, self.hidden_channels // 3)
        self.permutation_stride = permutation_stride if permutation_stride % 2 else permutation_stride + 1
        self.register_buffer("orthogonal_basis", _normalized_hadamard(self.hidden_channels))

    def projection_rows(self, m: Union[int, torch.Tensor]) -> torch.Tensor:
        m = _as_redundancy_factor(m)
        cycle = torch.arange(m, device=self.orthogonal_basis.device)
        base_rows = torch.arange(self.hidden_channels, device=self.orthogonal_basis.device)
        shift = cycle[:, None] * self.permutation_stride
        row_index = (base_rows[None, :] + shift).remainder(self.hidden_channels)
        codeword_basis = self.orthogonal_basis[row_index]
        return rearrange(
            codeword_basis,
            "m (k r) d -> m k r d",
            k=self.frames_per_codeword,
            r=self.measurements_per_frame,
        )


class StructuredOrthogonalProjectionEncoder(nn.Module):
    """Map every patch to ``m`` complete real-valued orthogonal codewords.

    Every codeword is a full orthonormal observation of the hidden patch.  Its
    number of STFT frames is derived from ``hidden_channels`` and
    ``num_freq_bins``. Additional values of ``m`` repeat the complete payload
    under deterministic orthogonal row permutations. No learned parameter
    depends on ``max_m``.
    """

    def __init__(self, hidden_channels: int, num_freq_bins: int):
        super().__init__()
        self.projector = StructuredOrthogonalBasis(hidden_channels=hidden_channels, num_freq_bins=num_freq_bins)
        self.hidden_channels = self.projector.hidden_channels
        self.num_freq_bins = self.projector.num_freq_bins
        self.frames_per_codeword = self.projector.frames_per_codeword
        self.neural_residual = RateSlotNeuralResidual(
            input_channels=self.hidden_channels,
            output_channels=self.projector.measurements_per_frame,
        )

    def projection_rows(self, target_m: Union[int, torch.Tensor]) -> torch.Tensor:
        return self.projector.projection_rows(target_m)

    def forward(self, patch_seq: torch.Tensor, redundancy_factor: Union[int, torch.Tensor]) -> torch.Tensor:
        if patch_seq.ndim != 3 or patch_seq.shape[1] != self.hidden_channels:
            raise ValueError(f"Expected patch_seq [B,{self.hidden_channels},L], got {tuple(patch_seq.shape)}.")
        redundancy_factor = _as_redundancy_factor(redundancy_factor)

        projection = self.projector.projection_rows(redundancy_factor).to(
            device=patch_seq.device, dtype=patch_seq.dtype
        )
        patches = rearrange(patch_seq, "b d l -> b l d")
        main_path = torch.einsum("bld,mkrd->blmkr", patches, projection)
        measurements = main_path + self.neural_residual(
            patch_seq,
            target_m=redundancy_factor,
            frames_per_codeword=self.frames_per_codeword,
        )

        # Keep channel coding and STFT representation as separate operations.
        # Each frame contains 2F real measurements; STFTSynthesizer explicitly
        # assigns their first/second halves to the real/imaginary coefficients.
        return rearrange(measurements, "b l m k r -> b r (l m k)")


class StructuredOrthogonalProjectionDecoder(nn.Module):
    """Reconstruct every complete codeword and average the observations."""

    def __init__(self, hidden_channels: int, num_freq_bins: int):
        super().__init__()
        self.projector = StructuredOrthogonalBasis(
            hidden_channels=hidden_channels,
            num_freq_bins=num_freq_bins,
        )
        self.hidden_channels = self.projector.hidden_channels
        self.num_freq_bins = self.projector.num_freq_bins
        self.measurements_per_frame = self.projector.measurements_per_frame
        self.frames_per_codeword = self.projector.frames_per_codeword

    @staticmethod
    def _matched_reconstruct(
        measurements: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        # Each codeword satisfies Q_j^T Q_j = I. Matched reconstruction yields
        # one full patch estimate per redundancy cycle, then averages them.
        per_codeword = torch.einsum("blmkr,mkrd->blmd", measurements, projection)
        return per_codeword.mean(dim=2)

    def forward(
        self,
        measurements: torch.Tensor,
        h_p: int,
        w_p: int,
        redundancy_factor: Union[int, torch.Tensor],
        snr_db: Optional[Union[int, float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if measurements.ndim != 3 or measurements.shape[1] != self.measurements_per_frame:
            raise ValueError(
                f"Expected real measurements [B,{self.measurements_per_frame},T], " f"got {tuple(measurements.shape)}."
            )
        if torch.is_complex(measurements):
            raise ValueError("Structured decoder expects real-valued measurements.")
        redundancy_factor = _as_redundancy_factor(redundancy_factor)

        patch_count = int(h_p) * int(w_p)
        expected_frames = patch_count * redundancy_factor * self.frames_per_codeword
        if measurements.shape[2] != expected_frames:
            raise ValueError(
                "Measurement frames T="
                f"{measurements.shape[2]} must equal h_p*w_p*target_m*frames_per_codeword="
                f"{expected_frames}."
            )

        measurements = rearrange(
            measurements,
            "b r (l m k) -> b l m k r",
            l=patch_count,
            m=redundancy_factor,
            k=self.frames_per_codeword,
        )
        projection = self.projector.projection_rows(redundancy_factor).to(
            device=measurements.device,
            dtype=measurements.dtype,
        )
        patches = self._matched_reconstruct(measurements, projection)
        return rearrange(patches, "b l d -> b d l")


class RadioEncoder(nn.Module):
    def __init__(
        self,
        num_freq_bins: int,
        overlap_ratio: float,
        token_channels: int,
        hidden_channels: int,
        patch_size: int = 2,
        token_pre_mixer_depth: int = 1,
        patch_mixer_depth: int = 1,
        frame_mixer_depth: int = 2,
        num_heads: int = 4,
        swin_window_size: int = 4,
        use_patch_attention: bool = False,
        use_frame_attention: bool = True,
        use_patch_conv: bool = True,
        use_conditioning: bool = True,
        use_m_conditioning: bool = True,
        use_snr_conditioning: bool = True,
        use_token_channel_gate: bool = True,
        use_patch_importance: bool = True,
        out_scale: float = 1.0,
        normalized: bool = True,
        use_time_spreading: bool = True,
        use_frequency_spreading: bool = False,
    ):
        super().__init__()
        self.normalizer = TokenPreChannelNormalizer()

        # Token processing
        self.patch_encoder = SwinPatchEncoder(
            token_channels=token_channels,
            hidden_channels=hidden_channels,
            pre_merge_depth=token_pre_mixer_depth,
            post_merge_depth=patch_mixer_depth,
            num_heads=num_heads,
            window_size=swin_window_size,
        )
        self.global_patch_mixer = GlobalPatchTransformer(
            channels=hidden_channels,
            depth=frame_mixer_depth,
            num_heads=num_heads,
            use_conditioning=use_conditioning,
            condition_on_m=use_m_conditioning,
            condition_on_snr=use_snr_conditioning,
        )
        # Swin supplies local 2-D modeling. The global mixer exchanges
        # information across the complete flattened patch grid before coding.
        self.structured_channel_encoder = StructuredOrthogonalProjectionEncoder(
            hidden_channels=hidden_channels, num_freq_bins=num_freq_bins
        )

        # ==============================
        # PHY oper layers
        # ==============================
        self.stft_synthesizer = STFTSynthesizer(num_freq_bins=num_freq_bins)
        self.istft = ISTFTLayer(num_freq_bins, overlap_ratio, normalized=normalized)
        self.power_norm = PowerNormLayer()

    def forward(
        self,
        token: torch.Tensor,
        redundancy_factor: Union[int, torch.Tensor],
        snr_db: Optional[Union[int, float, torch.Tensor]] = None,
    ):
        token, token_side_info = self.normalizer.normalize(token)  # [B, C, H, W] -> [B, C, H, W]

        patches, h_p, w_p = self.patch_encoder(token)  # [B,C,H,W] -> [B,D,Hp*Wp]
        patches = self.global_patch_mixer(
            patches,
            height=h_p,
            width=w_p,
            redundancy_factor=redundancy_factor,
            snr_db=snr_db,
        )

        channel_measurements = self.structured_channel_encoder(
            patches,
            redundancy_factor=redundancy_factor,
        )  # [B,D,L] -> [B,2F,L*m*frames_per_codeword] real

        stft_mat = self.stft_synthesizer(channel_measurements)

        iq = self.istft(stft_mat)
        iq_norm = self.power_norm(iq)

        return stft_mat, iq_norm, h_p, w_p, token_side_info


class RadioDecoder(nn.Module):
    def __init__(
        self,
        num_freq_bins: int,
        overlap_ratio: float,
        token_channels: int,
        hidden_channels: int,
        patch_size: int = 2,
        patch_denoiser_depth: int = 1,
        frame_denoiser_depth: int = 2,
        num_heads: int = 4,
        swin_window_size: int = 4,
        use_patch_attention: bool = False,
        use_frame_attention: bool = True,
        use_patch_conv: bool = True,
        use_conditioning: bool = True,
        use_m_conditioning: bool = True,
        use_snr_conditioning: bool = True,
        normalized: bool = True,
        use_time_spreading: bool = True,
        use_frequency_spreading: bool = False,
    ):
        super().__init__()
        self.normalizer = TokenPreChannelNormalizer()

        # ==============================
        # PHY layers
        # ==============================
        self.stft = STFTLayer(num_freq_bins, overlap_ratio, normalized=normalized)
        self.stft_analyzer = STFTAnalyzer(num_freq_bins=num_freq_bins)

        self.structured_channel_decoder = StructuredOrthogonalProjectionDecoder(
            hidden_channels=hidden_channels, num_freq_bins=num_freq_bins
        )
        self.global_patch_mixer = GlobalPatchTransformer(
            channels=hidden_channels,
            depth=frame_denoiser_depth,
            num_heads=num_heads,
            use_conditioning=use_conditioning,
            condition_on_m=use_m_conditioning,
            condition_on_snr=use_snr_conditioning,
        )

        self.patch_decoder = SwinPatchDecoder(
            token_channels=token_channels,
            hidden_channels=hidden_channels,
            pre_expand_depth=patch_denoiser_depth,
            post_expand_depth=patch_denoiser_depth,
            num_heads=num_heads,
            window_size=swin_window_size,
        )

    @staticmethod
    def _format_channel_response(
        channel_response: torch.Tensor,
        batch_size: int,
        num_freq_bins: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        h = channel_response.to(device=reference.device)
        if not torch.is_complex(h):
            h = torch.complex(h, torch.zeros_like(h))
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
        self,
        noisy_stft: torch.Tensor,
        channel_response: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        h = self._format_channel_response(
            channel_response,
            batch_size=noisy_stft.shape[0],
            num_freq_bins=noisy_stft.shape[1],
            reference=noisy_stft,
        )
        return noisy_stft * h.conj() / (h.abs().square() + eps)

    def forward(
        self,
        iq: torch.Tensor,
        h_p: int,
        w_p: int,
        redundancy_factor: Union[int, torch.Tensor],
        snr_db: Optional[Union[int, float, torch.Tensor]] = None,
        channel_response: Optional[torch.Tensor] = None,
        token_side_info: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Do not normalize at the receiver: absolute amplitude is useful channel evidence.
        noisy_stft = self.stft(iq)
        if channel_response is not None:
            noisy_stft = self.equalize_stft(noisy_stft, channel_response)

        # Explicit coefficient analysis: unpack F complex STFT bins back into
        # the same 2F real measurements consumed by the matched projection.
        channel_measurements = self.stft_analyzer(noisy_stft)
        patch_seq = self.structured_channel_decoder(
            channel_measurements,
            h_p=h_p,
            w_p=w_p,
            redundancy_factor=redundancy_factor,
        )
        patch_seq = self.global_patch_mixer(
            patch_seq,
            height=h_p,
            width=w_p,
            redundancy_factor=redundancy_factor,
            snr_db=snr_db,
        )

        token_rec = self.patch_decoder(patch_seq, h_p=h_p, w_p=w_p)
        token_rec = self.normalizer.denormalize(token_rec, token_side_info)

        return iq, noisy_stft, None, token_rec


class RadioAutoencoder(nn.Module):
    """
    Neural Radio Codec v2.

    Simplified STFT-based architecture with full-sequence patch attention and
    optional rate/SNR-adaptive AdaLN-Zero in both encoder and decoder.
    No OFDM pilot/subcarrier allocation - uses entire STFT bandwidth.

    """

    def __init__(
        self,
        num_freq_bins: int,
        token_channels: int,
        hidden_channels: int,
        overlap_ratio: float = 0.5,
        patch_size: int = 2,
        token_pre_mixer_depth: int = 2,
        patch_mixer_depth: int = 2,
        patch_denoiser_depth: int = 2,
        frame_mixer_depth: int = 2,
        frame_denoiser_depth: int = 2,
        num_heads: int = 4,
        swin_window_size: int = 4,
        use_patch_attention: bool = False,
        use_frame_attention: bool = True,
        use_patch_conv: bool = True,
        use_encoder_conditioning: bool = True,
        use_decoder_conditioning: bool = True,
        use_m_conditioning: bool = True,
        use_snr_conditioning: bool = True,
        use_token_channel_gate: bool = True,
        use_patch_importance: bool = True,
        out_scale: float = 1.0,
        normalized: bool = True,
        use_time_spreading: bool = True,
        use_frequency_spreading: bool = False,
    ):
        super().__init__()
        common = dict(
            num_freq_bins=num_freq_bins,
            overlap_ratio=overlap_ratio,
            token_channels=token_channels,
            hidden_channels=hidden_channels,
            patch_size=patch_size,
            num_heads=num_heads,
            swin_window_size=swin_window_size,
            use_patch_attention=use_patch_attention,
            use_frame_attention=use_frame_attention,
            use_patch_conv=use_patch_conv,
            normalized=normalized,
            use_time_spreading=use_time_spreading,
            use_frequency_spreading=use_frequency_spreading,
        )
        self.encoder = RadioEncoder(
            **common,
            token_pre_mixer_depth=token_pre_mixer_depth,
            patch_mixer_depth=patch_mixer_depth,
            frame_mixer_depth=frame_mixer_depth,
            use_conditioning=use_encoder_conditioning,
            use_m_conditioning=use_m_conditioning,
            use_snr_conditioning=use_snr_conditioning,
            use_token_channel_gate=use_token_channel_gate,
            use_patch_importance=use_patch_importance,
            out_scale=out_scale,
        )
        self.decoder = RadioDecoder(
            **common,
            patch_denoiser_depth=patch_denoiser_depth,
            frame_denoiser_depth=frame_denoiser_depth,
            use_conditioning=use_decoder_conditioning,
            use_m_conditioning=use_m_conditioning,
            use_snr_conditioning=use_snr_conditioning,
        )

    def encode(
        self,
        token: torch.Tensor,
        redundancy_factor: Union[int, torch.Tensor],
        snr_db: Optional[Union[int, float, torch.Tensor]] = None,
    ):
        return self.encoder(token, redundancy_factor=redundancy_factor, snr_db=snr_db)

    def decode(
        self,
        rx_iq: torch.Tensor,
        h_p: int,
        w_p: int,
        redundancy_factor: Union[int, torch.Tensor],
        snr_db: Optional[Union[int, float, torch.Tensor]] = None,
        channel_response: Optional[torch.Tensor] = None,
        token_side_info: Optional[Dict[str, torch.Tensor]] = None,
    ):
        return self.decoder(
            rx_iq,
            h_p=h_p,
            w_p=w_p,
            redundancy_factor=redundancy_factor,
            snr_db=snr_db,
            channel_response=channel_response,
            token_side_info=token_side_info,
        )

    def forward(
        self,
        token: torch.Tensor,
        target_m: Union[int, torch.Tensor],
        channel: Optional[nn.Module] = None,
        snr_db: Optional[Union[int, float, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        tx_stft, tx_iq_norm, h_p, w_p, token_side_info = self.encode(
            token,
            redundancy_factor=target_m,
            snr_db=snr_db,
        )
        m = target_m

        if channel is not None:
            try:
                channel_out = channel(tx_iq_norm, snr_db=snr_db)
            except TypeError:
                channel_out = channel(tx_iq_norm)
            if isinstance(channel_out, tuple):
                rx_iq, channel_ctx = channel_out
            else:
                rx_iq = channel_out
                channel_ctx = None
        else:
            rx_iq = tx_iq_norm
            channel_ctx = None

        decoder_snr_db = snr_db
        decoder_channel_response = None
        if decoder_snr_db is None and isinstance(channel_ctx, dict) and "snr_db" in channel_ctx:
            decoder_snr_db = channel_ctx["snr_db"]
        if isinstance(channel_ctx, dict):
            decoder_channel_response = channel_ctx.get("channel_response", channel_ctx.get("h", None))

        rx_iq, rx_stft, pred_clean_stft, token_rec = self.decode(
            rx_iq,
            h_p=h_p,
            w_p=w_p,
            redundancy_factor=m,
            snr_db=decoder_snr_db,
            channel_response=decoder_channel_response,
            token_side_info=token_side_info,
        )

        dec_aux = getattr(self.decoder, "last_aux", {})
        return {
            "tx_stft": tx_stft,
            "tx_iq": tx_iq_norm,
            "rx_iq": rx_iq,
            "rx_stft": rx_stft,
            "pred_clean_stft": pred_clean_stft,
            "token_rec": token_rec,
            "token_side_info": token_side_info,
            "h_p": h_p,
            "w_p": w_p,
            "target_m": m,
            "channel_ctx": channel_ctx,
            "h_est": dec_aux.get("h_est", None),
            "h_pilot": dec_aux.get("h_pilot", None),
            "pilot_mask": dec_aux.get("pilot_mask", None),
            "data_mask": dec_aux.get("data_mask", None),
        }
