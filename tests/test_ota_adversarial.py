import pytest

torch = pytest.importorskip('torch')
from rf_backdoor.ota_adversarial import PerturbationGenerator, ota_interference


def test_power_shape_and_gradient():
    torch.manual_seed(3)
    g = PerturbationGenerator()
    reference = torch.ones(2, 137, dtype=torch.complex64)
    delta, psr = ota_interference(g(torch.rand(2, 32)), reference, -16)
    assert delta.shape == reference.shape
    assert torch.allclose(psr, torch.full((2,), -16.), atol=1e-4)
    delta.real.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters())


def test_rng_replay_and_zero_power():
    g = PerturbationGenerator()
    pattern = g(torch.rand(2, 32))
    reference = torch.ones(2, 128, dtype=torch.complex64)
    state = torch.get_rng_state()
    a, _ = ota_interference(pattern, reference, -20)
    torch.set_rng_state(state)
    b, _ = ota_interference(pattern, reference, -20)
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        ota_interference(pattern, reference, 1)
    with pytest.raises(ValueError):
        ota_interference(pattern, reference * 0, -20)
