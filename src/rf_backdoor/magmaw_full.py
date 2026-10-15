"""Image-only Magmaw adaptation: residual PGM, discriminator and waveform OFDM.

Independent implementation from paper equations; no RF or external code I/O.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.deepjscc.model import Conv_PReLU, TransConv_PReLU, PowerNormLayer


class ImageCodec(nn.Module):
    """Reuse local DeepJSCC primitives with independent depths/widths/rates."""
    def __init__(self, c, depth=5, width=32):
        super().__init__()
        self.spec = {'c': c, 'depth': depth, 'width': width}
        middle = [Conv_PReLU(width, width, 5, 1, 2) for _ in range(depth-3)]
        self.encoder = nn.Sequential(Conv_PReLU(3, width//2, 5, 2, 2),
            Conv_PReLU(width//2, width, 5, 2, 2), *middle, Conv_PReLU(width, 2*c, 5, 1, 2))
        middle = [TransConv_PReLU(width, width, 5, 1, 2) for _ in range(depth-3)]
        self.decoder = nn.Sequential(TransConv_PReLU(2*c, width, 5, 1, 2), *middle,
            TransConv_PReLU(width, width//2, 5, 2, 2, 1),
            TransConv_PReLU(width//2, 3, 5, 2, 2, 1, nn.Sigmoid()))
        self.norm = PowerNormLayer()

    def encode(self, images):
        out = self.encoder(images)
        b, _, h, w = out.shape
        real, imag = out.chunk(2, dim=1)
        return self.norm(torch.complex(real.flatten(1), imag.flatten(1))), (h, w)

    def decode(self, latent, shape):
        h, w = shape
        iq = torch.cat((latent.real, latent.imag), -1)
        return self.decoder(iq.reshape(len(latent), 2*self.spec['c'], h, w))


class ResidualSE(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.first = nn.ConvTranspose2d(width, width, 3, padding=1)
        self.second = nn.ConvTranspose2d(width, width, 3, padding=1)
        self.norm1, self.norm2 = nn.InstanceNorm2d(width), nn.InstanceNorm2d(width)
        self.squeeze = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width//16, 1),
            nn.ReLU(), nn.Conv2d(width//16, width, 1), nn.Sigmoid())

    def forward(self, x):
        out = self.norm2(self.second(F.relu(self.norm1(self.first(x)))))
        return F.relu(x + out*self.squeeze(out))


class ResidualPGM(nn.Module):
    """Official Res_Generator architecture sizes; 100 -> 16x16x16 real values."""
    latent_dim = 100
    def __init__(self):
        super().__init__()
        self.start = nn.Sequential(nn.ConvTranspose2d(100, 1024, 4, bias=False),
            nn.InstanceNorm2d(1024), nn.ReLU(), nn.ConvTranspose2d(1024, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256), nn.ReLU())
        self.residual = nn.Sequential(*(ResidualSE() for _ in range(5)))
        self.finish = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU(), nn.Conv2d(64, 16, 3, padding=1), nn.Tanh())

    def forward(self, z):
        out = self.finish(self.residual(self.start(z.reshape(-1, 100, 1, 1))))
        # 4096 real values -> 2048 complex values -> 32 OFDM symbols.
        iq = out.reshape(len(z), 32, 64, 2)
        return torch.view_as_complex(iq.contiguous())


class SignalDiscriminator(nn.Module):
    """Distinguish clean/attacked equalized complex symbols, shared across rates."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(2, 32, 7, 4, 3), nn.LeakyReLU(.2),
            nn.Conv1d(32, 64, 7, 4, 3), nn.LeakyReLU(.2),
            nn.Conv1d(64, 128, 7, 4, 3), nn.LeakyReLU(.2), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.features(torch.stack((x.real, x.imag), dim=1)).squeeze(-1)).squeeze(-1)


def qam(x, degree, *, soft_gradient=True):
    if degree not in (4, 16, 64):
        raise ValueError('degree must be QPSK=4, 16-QAM or 64-QAM')
    side = int(math.sqrt(degree))
    levels = torch.arange(-(side-1), side, 2, device=x.device, dtype=x.real.dtype)
    levels = levels/math.sqrt(2*(degree-1)/3)
    def quantize(v):
        distance = (v[..., None]-levels).square()
        hard = levels[distance.argmin(-1)]
        if not soft_gradient or not torch.is_grad_enabled():
            return hard
        soft = (F.softmax(-5*distance, dim=-1)*levels).sum(-1)
        return soft + (hard-soft).detach()
    return torch.complex(quantize(x.real), quantize(x.imag))


def channel_bank(count, seed, device='cpu', taps=8):
    rng = torch.Generator(device=device).manual_seed(seed)
    profile = torch.exp(-torch.arange(taps, device=device)/4)
    profile /= profile.sum()
    values = torch.randn(count, taps, device=device, dtype=torch.complex64, generator=rng)
    return values*profile.sqrt()


class WaveformOFDM:
    """64-bin OFDM, CP16, 48 data/4 comb pilots, leading training symbol.

    Linear time-domain convolution, shared AWGN, pilot MMSE channel estimate,
    MMSE equalization, hard QAM forward with soft gradient, attacked pilots.
    A receiver-power oracle calibrates received PSR after attacker propagation.
    Packet timing of victim is known; the attacker has fractional sample delay.
    """
    nfft, cp = 64, 16
    def __init__(self, device, snr_db=10):
        self.device, self.snr_db = device, snr_db
        null = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
        pilot = [11, 25, 39, 53]
        self.data = torch.tensor([i for i in range(64) if i not in null+pilot], device=device)
        self.active = torch.tensor([i for i in range(64) if i not in null], device=device)
        self.pilot = torch.tensor(pilot, device=device)
        self.training = torch.zeros(64, device=device, dtype=torch.complex64)
        self.training[self.active] = (1-2*(self.active % 2)).to(torch.complex64)

    def modulate(self, frequency):
        waveform = torch.fft.ifft(frequency, norm='ortho')
        return torch.cat((waveform[..., -16:], waveform), -1).flatten(1)

    @staticmethod
    def propagate(waveform, taps):
        total = waveform.shape[-1]+taps.shape[-1]-1
        fft_length = 1 << (total-1).bit_length()
        out = torch.fft.ifft(torch.fft.fft(waveform, n=fft_length)*torch.fft.fft(taps, n=fft_length))
        return out[:, :waveform.shape[-1]]

    def prepare(self, latent, degree, bank=None):
        b, length = latent.shape
        symbols = math.ceil(length/48)
        values = F.pad(qam(latent, degree), (0, symbols*48-length)).reshape(b, symbols, 48)
        frequency = torch.zeros(b, symbols+1, 64, dtype=latent.dtype, device=latent.device)
        frequency[:, 0] = self.training
        frequency[:, 1:, self.data] = values
        frequency[:, 1:, self.pilot] = self.training[self.pilot]
        tx = self.modulate(frequency)
        victim = channel_bank(b, int(torch.randint(0, 2**30, ()).item()), self.device)
        if bank is None:
            attack = channel_bank(b, int(torch.randint(0, 2**30, ()).item()), self.device)
        else:
            attack = bank[torch.randint(len(bank), (b,), device=self.device)]
        clean = self.propagate(tx, victim)
        power = clean.detach().abs().square().mean(-1, keepdim=True)
        variance = power*10**(-self.snr_db/10)
        noise = torch.randn_like(clean)*variance.sqrt()
        state = {'length': length, 'symbols': symbols, 'degree': degree, 'clean': clean,
            'rx': clean+noise, 'power': power, 'variance': variance, 'attacker': attack,
            'permutation': torch.stack([torch.randperm(64, device=self.device) for _ in range(b)]),
            'delay': torch.rand(b, 1, 1, device=self.device)*64,
            'phase': torch.rand(b, 1, 1, device=self.device)*2*math.pi}
        return state

    def attack(self, pattern, state, psr_db, *, transform='full'):
        if not math.isfinite(psr_db) or psr_db > 0:
            raise ValueError('PSR must be finite and <=0')
        frequency = pattern.repeat(1, math.ceil((state['symbols']+1)/pattern.shape[1]), 1)
        frequency = frequency[:, :state['symbols']+1]
        if transform == 'full':
            frequency = torch.gather(frequency, 2, state['permutation'][:, None].expand(-1, frequency.shape[1], -1))
        if transform in ('full', 'sync_free'):
            bins = torch.fft.fftfreq(64, device=self.device)
            frequency = frequency*torch.exp(1j*(state['phase']-2*math.pi*bins*state['delay']))
        # A generated interference symbol can occupy guard and pilot bins.
        raw = self.propagate(self.modulate(frequency), state['attacker'])
        return self.project(raw, state, psr_db)

    @staticmethod
    def project(raw, state, psr_db):
        energy = raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12)
        limit = state['power']*10**(psr_db/10)
        scale = (limit/energy).sqrt().clamp(max=1)
        interference = raw*scale
        actual = 10*torch.log10(interference.abs().square().mean(-1)/state['power'][:, 0])
        return interference, actual

    def random_attack(self, state, psr_db):
        raw = self.propagate(torch.randn_like(state['rx']), state['attacker'])
        return self.project(raw, state, psr_db)

    def decode(self, waveform, state, *, demodulate=True):
        b = waveform.shape[0]
        frequency = torch.fft.fft(waveform.reshape(b, state['symbols']+1, 80)[..., 16:], norm='ortho')
        estimate = frequency[:, :1, self.data]/self.training[self.data]
        estimate = estimate/(1+state['variance'][:, None])
        data = frequency[:, 1:, self.data]
        equalized = data*estimate.conj()/(estimate.abs().square()+state['variance'][:, None]+1e-8)
        if demodulate:
            equalized = qam(equalized, state['degree'])
        return equalized.flatten(1)[:, :state['length']]
