"""Image-branch Magmaw extensions; keeps torch lazy for local policy tests."""
import math


def permutation_probability(step, steps):
    if step < 0 or steps < 1:
        raise ValueError('invalid curriculum progress')
    return min(1., step / max(1, steps // 2))


def balanced_coefficients(rx_gradient, ds_gradient, dv_gradient, ds_fraction=.1, dv_fraction=.05):
    values = [rx_gradient, ds_gradient, dv_gradient, ds_fraction, dv_fraction]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('finite nonnegative gradients/fractions required')
    return tuple(min(100., fraction * rx_gradient / norm) if norm > 1e-12 else 0.
                 for norm, fraction in [(ds_gradient, ds_fraction), (dv_gradient, dv_fraction)])


def build_full_generator(initial, rank=8):
    """100-D random z -> reused ResNet -> full-burst stochastic basis mixture.

    Full-length mean and bases are shared across all images. The generator's
    ONLY runtime input is z; no source image, encoder output or channel input.
    """
    import torch
    from torch import nn
    from rf_backdoor.magmaw_paper import build_pgm
    class FullBurstPGM(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = build_pgm()
            self.coefficients = nn.Linear(4096, rank)
            nn.init.normal_(self.coefficients.weight, 0, .02)
            nn.init.zeros_(self.coefficients.bias)
            self.mean_iq = nn.Parameter(initial.detach().clone())
            amplitude = initial.square().mean().sqrt().clamp_min(.01)
            self.basis_iq = nn.Parameter(torch.randn(rank, 1408, 64, 2, device=initial.device) * amplitude * .1)
            self.rank = rank

        def forward(self, z):
            encoded = torch.view_as_real(self.backbone(z)).flatten(1)
            logits = self.coefficients(encoded)
            logits = logits / (1 + logits.square().mean(1, keepdim=True)).sqrt()
            # Keep z observable even if the learned head saturates. Gates stay
            # in [.75,1.25], so they cannot cancel the signed random code.
            coefficients = (2*z[:, :self.rank]-1) * (1 + .25*logits.tanh())
            # Real I/Q QR prevents the learned bases becoming collinear. The
            # QR remains differentiable; it does not require source/CSI input.
            q, _ = torch.linalg.qr(self.basis_iq.flatten(1).T, mode='reduced')
            bases = q.T.reshape_as(self.basis_iq)
            variation = torch.einsum('br,rskq->bskq', coefficients, bases)
            variation = variation / variation.square().mean((1,2,3), keepdim=True).clamp_min(1e-20).sqrt()
            mean_rms = self.mean_iq.square().mean().clamp_min(1e-12).sqrt()
            # A small fixed relative RMS budget makes latent collapse visible
            # and impossible to hide by shrinking the stochastic component.
            return torch.view_as_complex((self.mean_iq + .05*mean_rms*variation).contiguous())
    return FullBurstPGM().to(initial.device)


def generator_terms(rx_losses, clean_logits, attack_logits, first, second, diversity_mode):
    """Equation 13 G minimization signs; balanced DV is explicitly rescaled."""
    import torch
    from torch.nn import functional as F
    from rf_backdoor.magmaw_paper import diversity
    rx = torch.stack(rx_losses).sum()
    ds = torch.stack([F.logsigmoid(c).mean() + F.logsigmoid(-a).mean()
                      for c, a in zip(clean_logits, attack_logits)]).sum()
    if diversity_mode == 'rms_normalized_mean':
        def normalize(pattern):
            return pattern / pattern.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        dv = diversity(normalize(first), normalize(second), reduction='mean')
    elif diversity_mode == 'literal_norm':
        dv = diversity(first, second, reduction='norm')
    else:
        raise ValueError('unknown diversity definition')
    return {'rx': rx, 'ds': ds, 'dv': dv}
