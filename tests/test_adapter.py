from rf_backdoor.adapter import ChunbaobaoDeepJSCCAdapter


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

