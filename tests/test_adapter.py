from rf_backdoor.adapter import ChunbaobaoDeepJSCCAdapter, DeepJSCCAdapter


class FakeChannel:
    def __init__(self):
        self.snr = 7.0

    def __call__(self, latent):
        return latent + self.snr


class FakeModel:
    def __init__(self):
        self.channel = FakeChannel()
        self.encoder = lambda images: images * 2
        self.decoder = lambda latent: latent - 1


def test_adapter_exposes_stable_forward_without_mutating_channel_setting():
    model = FakeModel()
    adapter = ChunbaobaoDeepJSCCAdapter(model)
    assert adapter.forward(3, 10) == 15
    assert model.channel.snr == 7.0
    assert adapter.encode(3) == 6
    assert adapter.decode(8) == 7


class StatefulModel:
    def __init__(self):
        self.encoder = lambda images: (images * 2, {"scale": 4})
        self.decoder = lambda latent, state: latent / state["scale"]


def test_adapter_carries_improved_codec_state_to_decoder():
    adapter = DeepJSCCAdapter(StatefulModel(), channel=lambda latent: latent)
    encoded = adapter.encode(6)
    assert encoded == 12
    assert adapter.decode(encoded) == 3


def test_improved_local_adapter_round_trip_shapes():
    torch = __import__("pytest").importorskip("torch")
    __import__("pytest").importorskip("einops")
    adapter = DeepJSCCAdapter.from_local(c=2, snr_db=7.0)
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        latent = adapter.encode(images)
        received = adapter.apply_channel(latent, 7.0)
        output = adapter.decode(received)
    assert latent.is_complex()
    assert latent.shape == (2, 2 * 8 * 8)
    assert output.shape == images.shape
    assert torch.isfinite(output).all()


def test_adapter_channel_pair_reuses_supplied_channel_noise():
    torch = __import__("pytest").importorskip("torch")

    class StatefulChannel:
        def sampling_channel_state(self, batch_size, seq_len, device, dtype, generator=None):
            return {"snr_db": torch.full((batch_size,), 7.0, device=device)}

        def __call__(self, signal, state, generator=None):
            real = torch.randn(signal.shape, device=signal.device, generator=generator)
            imag = torch.randn(signal.shape, device=signal.device, generator=generator)
            return signal + torch.complex(real, imag).to(signal.dtype)

    adapter = DeepJSCCAdapter(object(), channel=StatefulChannel())
    clean = torch.ones(2, 4, dtype=torch.complex64)
    triggered = torch.full_like(clean, 2.0)
    clean_rx, triggered_rx = adapter.apply_channel_pair(clean, triggered, 7.0)
    assert torch.allclose(clean_rx - clean, triggered_rx - triggered, atol=1e-6, rtol=0.0)
