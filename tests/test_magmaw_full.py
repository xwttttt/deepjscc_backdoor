import pytest
torch = pytest.importorskip('torch')
from rf_backdoor.magmaw_full import ImageCodec, ResidualPGM, WaveformOFDM, qam


def test_codec_rate_shape():
    model = ImageCodec(4, depth=6, width=28)
    x = torch.rand(2, 3, 32, 32)
    latent, shape = model.encode(x)
    assert latent.shape == (2, 256)
    assert model.decode(latent, shape).shape == x.shape
    assert torch.allclose(latent.abs().square().mean(-1), torch.ones(2), atol=1e-5)


def test_ofdm_noiseless_roundtrip_and_psr():
    channel = WaveformOFDM('cpu', snr_db=100)
    latent = torch.randn(2, 113, dtype=torch.complex64)
    state = channel.prepare(latent, 16)
    # CP is longer than the FIR channel; at high SNR equalization should recover QAM.
    recovered = channel.decode(state['rx'], state)
    assert torch.allclose(recovered, qam(latent, 16), atol=1e-5)
    pattern = torch.randn(2, 3, 64, dtype=torch.complex64, requires_grad=True)
    delta, psr = channel.attack(pattern, state, -16)
    assert bool((psr <= -16+1e-4).all())
    assert torch.allclose(psr, torch.full((2,), -16.), atol=1e-4)
    loss = channel.decode(state['rx']+delta, state).abs().mean()
    loss.backward()
    assert pattern.grad is not None and torch.isfinite(pattern.grad).all()


def test_residual_generator_independent_codes():
    generator = ResidualPGM()
    out = generator(torch.rand(2, 100))
    assert out.shape == (2, 32, 64)
    assert not torch.equal(out[0], out[1])
    assert torch.isfinite(out).all()
