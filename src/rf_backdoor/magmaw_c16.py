"""Continuous c16 P0 over analog OFDM; legitimate AWGN baseline preserved."""
import math


def build_analog_link(device, snr_db=10):
    import torch
    from torch.nn import functional as F
    from rf_backdoor.magmaw_full import WaveformOFDM
    class AnalogP0OFDM(WaveformOFDM):
        def modulate(self,frequency):
            # Round-trip error can trip TF32 convolution thresholds at high
            # image quality. Only the signal transform uses double precision;
            # the original P0 codec remains float32 with unchanged weights.
            return super().modulate(frequency.to(torch.complex128))
        def prepare(self, latent, degree=None, bank=None):
            b,length=latent.shape
            symbols=math.ceil(length/48)
            frequency=torch.zeros(b,symbols+1,64,device=latent.device,dtype=latent.dtype)
            frequency[:,0]=self.training
            frequency[:,1:,self.data]=F.pad(latent,(0,symbols*48-length)).reshape(b,symbols,48)
            frequency[:,1:,self.pilot]=self.training[self.pilot]
            clean=self.modulate(frequency)
            # Unit-gain victim is the checkpoint's AWGN backend. Orthonormal
            # FFT preserves white noise variance on the continuous data bins.
            variance=latent.detach().abs().square().mean(-1,keepdim=True)*10**(-self.snr_db/10)
            noise=torch.complex(torch.randn_like(clean.real),torch.randn_like(clean.real))*(variance/2).sqrt()
            state={'length':length,'symbols':symbols,'clean':clean,'rx':clean+noise,
                   'power':clean.detach().abs().square().mean(-1,keepdim=True),'variance':variance,
                   'attacker':bank[torch.randint(len(bank),(b,),device=latent.device)],
                   'permutation':torch.stack([torch.randperm(64,device=latent.device) for _ in range(b)]),
                   'delay':torch.rand(b,1,1,device=latent.device)*64,
                   'phase':torch.rand(b,1,1,device=latent.device)*2*math.pi}
            state['direct_rx']=(latent.to(torch.complex128)+self.decode(noise,state,return_double=True)).to(latent.dtype)
            return state

        def decode(self,waveform,state,*,demodulate=False,return_double=False):
            frequency=torch.fft.fft(waveform.reshape(len(waveform),state['symbols']+1,80)[...,16:],norm='ortho')
            # Known unit victim channel; no QAM or noisy CE substituted for P0.
            data=frequency[:,1:,self.data].flatten(1)[:,:state['length']]
            return data if return_double else data.to(torch.complex64)
    return AnalogP0OFDM(device,snr_db)
