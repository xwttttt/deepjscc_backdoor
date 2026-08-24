import pytest


torch = pytest.importorskip("torch")

from src.channels import Channel, ChannelConfig


def test_channel_rejects_non_complex_or_non_sequence_input():
    channel = Channel(ChannelConfig())
    state = channel.sampling_channel_state(2, 8, "cpu")
    with pytest.raises(ValueError, match="complex"):
        channel(torch.randn(2, 8), state)
    with pytest.raises(ValueError, match="shape"):
        channel(torch.randn(2, 2, 8, dtype=torch.complex64), state)


def test_fixed_snr_channel_is_reproducible_with_generators():
    channel = Channel(ChannelConfig(snr_min=7.0, snr_max=7.0))
    signal = torch.ones(3, 4096, dtype=torch.complex64)

    def transmit(seed):
        generator = torch.Generator().manual_seed(seed)
        state = channel.sampling_channel_state(3, 4096, "cpu", generator=generator)
        return channel(signal, state, generator=generator)

    first = transmit(23)
    second = transmit(23)
    assert torch.equal(first, second)
    noise_power = (first - signal).abs().square().mean()
    measured_snr = 10.0 * torch.log10(signal.abs().square().mean() / noise_power)
    assert measured_snr.item() == pytest.approx(7.0, abs=0.2)


def test_channel_config_validates_ranges():
    with pytest.raises(ValueError, match="snr_min"):
        ChannelConfig(snr_min=10.0, snr_max=1.0)
    with pytest.raises(ValueError, match="sampling_rate"):
        ChannelConfig(sampling_rate=0.0)
