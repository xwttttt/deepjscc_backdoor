import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import _EPS


class TokenLoss(nn.Module):
    def __init__(self, huber_delta=1.0, cos_beta=0.1):
        super().__init__()
        self.huber_delta = huber_delta
        self.cos_beta = cos_beta
        self.eps = _EPS

    @staticmethod
    def normalize_channelwise(token, eps=1e-6):
        # token: [B, C, H, W]
        mean = token.mean(dim=1, keepdim=True)
        std = token.std(dim=1, keepdim=True, unbiased=False)
        return (token - mean) / (std + eps)

    def forward(self, token_rec, token):
        """
        token_rec: reconstructed token, z_hat, [B, C, H, W]
        token:     target token,        z,     [B, C, H, W]
        """
        token_target = token.detach()

        # z_norm = self.normalize_channelwise(token_target, self.eps)
        # z_hat_norm = self.normalize_channelwise(token_rec, self.eps)

        loss_huber = F.huber_loss(token_rec, token_target, reduction="mean", delta=self.huber_delta)
        loss_cos = 1.0 - F.cosine_similarity(token_rec, token_target, dim=1, eps=self.eps).mean()

        return loss_huber + self.cos_beta * loss_cos


class STFTConsistencyLoss(nn.Module):

    def __init__(self, num_freq_bins=64, overlap_radio=0.5):
        r"""Measure consistency after removing the arbitrary waveform scale.

        .. math::
            \mathcal{L}_{cons} = \|\bar{Y} - STFT(iSTFT(\bar{Y}))\|

        The transmitter power-normalizes ``iSTFT(Y)``. Applying an absolute
        consistency loss to the unnormalized ``Y`` therefore creates a scale
        degeneracy: the model can shrink ``Y`` toward zero without changing
        the transmitted waveform. Normalizing both paths by the pre-channel
        waveform RMS removes that unstable degree of freedom.
        """
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.hop_length = int(num_freq_bins * (1 - overlap_radio))
        self.register_buffer("window", torch.hann_window(num_freq_bins))
        self.eps = _EPS

    def forward(self, stft_mat):
        iq = torch.istft(
            stft_mat,
            n_fft=self.num_freq_bins,
            hop_length=self.hop_length,
            win_length=self.num_freq_bins,
            window=self.window,
            center=True,
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        power = iq.abs().square().mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(power.clamp_min(self.eps))
        iq_normalized = iq * inv_rms
        stft_mat_normalized = stft_mat * inv_rms.unsqueeze(1)

        stft_mat_hat = torch.stft(
            iq_normalized,
            n_fft=self.num_freq_bins,
            hop_length=self.hop_length,
            win_length=self.num_freq_bins,
            window=self.window,
            center=True,
            normalized=True,
            onesided=False,
            return_complex=True,
        )
        loss_cons = torch.abs(stft_mat_normalized - stft_mat_hat)

        loss = torch.mean(loss_cons)
        return loss


class PAPRLoss(nn.Module):
    def __init__(self, target_papr_db=6.0):
        """
        :param target_papr_db: 允许的最大 PAPR (dB)，超过此值将产生 Loss
        """
        super().__init__()
        self.target_papr_db = target_papr_db
        self.eps = _EPS

    def forward(self, iq_signal):
        # iq_signal shape: [B, L] (复数时域波形)

        # 1. 计算瞬时功率 |x(t)|^2
        power = torch.abs(iq_signal) ** 2  # [B, L]

        # 2. 计算每个 Batch 样本的峰值功率和平均功率
        max_power = torch.max(power, dim=-1)[0]  # [B]
        mean_power = torch.mean(power, dim=-1)  # [B]

        # 3. 计算 PAPR (线性值)
        papr_linear = max_power / (mean_power + self.eps)

        # 4. 转换为 dB
        papr_db = 10 * torch.log10(papr_linear + self.eps)

        # 5. 计算超出 target_papr_db 的惩罚项
        # 使用 ReLU，只有当 papr_db > target_papr_db 时才会产生正的 Loss
        loss = F.relu(papr_db - self.target_papr_db)

        return torch.mean(loss)


class CombinedLoss(nn.Module):

    def __init__(
        self,
        huber_delta=1.0,
        num_freq_bins=64,
        overlap_radio=0.5,
        target_papr_db=6.0,
        lambda1=0.1,
        lambda2=0.1,
        lambda3=0.1,
    ):
        super().__init__()
        self.token_loss = TokenLoss(huber_delta, lambda1)
        self.stft_loss = STFTConsistencyLoss(num_freq_bins, overlap_radio)
        self.papr_loss = PAPRLoss(target_papr_db)

        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, token_rec, token, tx_stft, iq_signal):
        token_loss_value = self.token_loss(token_rec, token)
        stft_loss_value = self.stft_loss(tx_stft)
        papr_loss_value = self.papr_loss(iq_signal)

        loss_detail = {
            "token": token_loss_value,
            "stft": stft_loss_value,
            "papr": papr_loss_value,
        }
        total_loss = token_loss_value + self.lambda2 * stft_loss_value + self.lambda3 * papr_loss_value
        return total_loss, loss_detail
