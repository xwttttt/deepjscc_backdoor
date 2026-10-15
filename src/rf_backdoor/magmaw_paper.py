"""Equation-level image subset of Magmaw, kept separate from legacy experiments.

Equations (8), (11)--(15) define the game and transformation. PyTorch is lazy
so NumPy analytical oracles run locally without the ML runtime.
"""
import math
import numpy as np


def project_np(signal, epsilon):
    signal = np.asarray(signal)
    energy = np.sum(np.abs(signal)**2, axis=tuple(range(1, signal.ndim)), keepdims=True)
    epsilon = np.asarray(epsilon).reshape((-1,)+(1,)*(signal.ndim-1))
    if np.any(epsilon < 0) or not np.all(np.isfinite(epsilon)):
        raise ValueError('energy budget must be finite and nonnegative')
    return signal*np.minimum(1, np.sqrt(epsilon/np.maximum(energy, 1e-30)))


def diversity_np(first, second):
    """Complex L1 norm per code pair, expectation over batch, Eq. (12)."""
    return float(np.abs(np.asarray(first)-np.asarray(second)).reshape(len(first),-1).sum(1).mean())


def project_energy(signal, epsilon):
    import torch
    energy=signal.abs().square().flatten(1).sum(1)
    epsilon=torch.as_tensor(epsilon,device=signal.device,dtype=signal.real.dtype).reshape(-1)
    if epsilon.numel()==1:
        epsilon=epsilon.expand(signal.shape[0])
    if epsilon.shape!=energy.shape or bool((epsilon<0).any()) or not bool(torch.isfinite(epsilon).all()):
        raise ValueError('one finite nonnegative energy budget per burst required')
    scale=(epsilon/energy.clamp_min(1e-30)).sqrt().clamp(max=1)
    return signal*scale.reshape((-1,)+(1,)*(signal.ndim-1))


def diversity(first, second, *, reduction='norm'):
    values=(first-second).abs().flatten(1)
    if reduction=='norm':
        return values.sum(1).mean()
    if reduction=='mean':
        return values.mean()
    raise ValueError('unknown diversity reduction')


def game_losses(reconstruction_mses, clean_logits, attack_logits, first, second,
                *, beta_ds=1., beta_dv=1., diversity_reduction='norm'):
    """Exact image subset of Eq. (13), including the clean term for auditing.

    G minimizes -sum Lrx + beta_ds*sum Lds - beta_dv*Ldv.
    D minimizes -sum Lds. Clean=1, attacked=0. The discriminator must be frozen
    during G differentiation; attacked samples detached during D update.
    """
    import torch
    from torch.nn import functional as F
    lrx=torch.stack(reconstruction_mses).sum()
    lds=torch.stack([F.logsigmoid(c).mean()+F.logsigmoid(-a).mean()
                     for c,a in zip(clean_logits,attack_logits)]).sum()
    ldv=diversity(first,second,reduction=diversity_reduction)
    return {'generator':-lrx+beta_ds*lds-beta_dv*ldv,
            'discriminator':-lds, 'reconstruction':lrx,'discriminative':lds,'diversity':ldv}


def build_pgm():
    """Paper ResNet choice with official residual initialization/IQ packing."""
    import torch
    from torch import nn
    from rf_backdoor.magmaw_full import ResidualPGM
    class PaperPGM(ResidualPGM):
        def __init__(self):
            super().__init__()
            for layer in self.residual.modules():
                if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d)):
                    nn.init.normal_(layer.weight,0,.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        def forward(self,z):
            raw=self.finish(self.residual(self.start(z.reshape(-1,100,1,1))))
            iq=raw.reshape(len(z),32,2,64).permute(0,1,3,2).contiguous()
            return torch.view_as_complex(iq)
    return PaperPGM()


def build_link(device,snr_db=10):
    """LS pilot estimation as stated in the paper, then MMSE equalization.

    Equalizer choice is an explicit convention; the paper specifies LS CE but
    not its exact equalizer formula. Retained codecs were fitted with legacy
    MMSE CE, so this small receiver mismatch is recorded rather than hidden.
    """
    import torch
    from rf_backdoor.magmaw_full import WaveformOFDM,qam
    class PaperReceiverOFDM(WaveformOFDM):
        def decode(self,waveform,state,*,demodulate=True):
            frequency=torch.fft.fft(waveform.reshape(len(waveform),state['symbols']+1,80)[...,16:],norm='ortho')
            estimate=frequency[:,:1,self.data]/self.training[self.data]
            equalized=frequency[:,1:,self.data]*estimate.conj()/(estimate.abs().square()+state['variance'][:,None]+1e-8)
            if demodulate:
                equalized=qam(equalized,state['degree'])
            return equalized.flatten(1)[:,:state['length']]
    return PaperReceiverOFDM(device,snr_db)


def transform_frequency(pattern, symbols, permutation, delay, phase, epsilon=None):
    """K -> Gamma -> unit-magnitude rotations, Eq. (15).

    Literal Eq. (15): K -> Gamma -> M -> rotations. CP energy can depend on
    delay, so the link reports frequency constraint and waveform power separately.
    """
    import torch
    frequency=pattern.repeat(1,math.ceil(symbols/pattern.shape[1]),1)[:,:symbols]
    frequency=torch.gather(frequency,2,permutation[:,None].expand(-1,symbols,-1))
    if epsilon is not None:
        frequency=project_energy(frequency,epsilon)
    bins=torch.fft.fftfreq(pattern.shape[-1],device=pattern.device)
    return frequency*torch.exp(1j*(phase-2*math.pi*bins*delay))


def paper_attack(link, pattern, state, psr_db, *, projection='before_channel',max_symbols=704):
    """Apply Eq. (14) before Ha; report projected and realized powers separately.

    Reference uses the useful (CP-removed), noiseless received victim interval.
    Eq. (3) is frequency-domain but Eq. (14) calls hat-y time-domain. Orthonormal
    FFT/Parseval defines an explicit consistent reference; the public code uses
    full-burst time power instead. Neither convention implies a received cap
    after Ha. No instantaneous attacker CSI is used for renormalization.
    """
    useful=state['clean'].reshape(len(pattern),-1,80)[...,16:]
    if state['symbols']+1>max_symbols:
        raise ValueError('victim burst exceeds the configured public maximum duration')
    # Constant-duration attacker; only receiver capture overlap is trimmed.
    # 704 = ceil(684/32)*32 for the largest configured c=8, 256x256 frame.
    reference_energy=(useful.abs().square().flatten(1).mean(1)*max_symbols*64).detach()
    epsilon=reference_energy*10**(psr_db/10)
    if projection=='before_channel':
        frequency=transform_frequency(pattern,max_symbols,state['permutation'],state['delay'],state['phase'],epsilon)
        projected=link.modulate(frequency)
        delta=link.propagate(projected,state['attacker'])[:,:state['rx'].shape[-1]]
    elif projection=='after_channel':
        frequency=transform_frequency(pattern,max_symbols,state['permutation'],state['delay'],state['phase'])
        projected=link.modulate(frequency)
        full_epsilon=state['power'][:,0]*projected.shape[-1]*10**(psr_db/10)
        full_delta=project_energy(link.propagate(projected,state['attacker']),full_epsilon)
        delta=full_delta[:,:state['rx'].shape[-1]]
    else:
        raise ValueError('projection must be before_channel or after_channel')
    import torch
    reference=state['power'][:,0]
    return delta,{
        'constraint_psr_db':10*torch.log10(frequency.abs().square().flatten(1).sum(1).clamp_min(1e-30)/reference_energy),
        'launch_psr_db':10*torch.log10(projected.abs().square().mean(-1).clamp_min(1e-30)/reference),
        'received_psr_db':10*torch.log10(delta.abs().square().mean(-1).clamp_min(1e-30)/reference)}
