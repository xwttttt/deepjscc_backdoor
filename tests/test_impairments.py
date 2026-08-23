import math

import numpy as np
import pytest

from rf_backdoor.impairments import (
    IQDCOffsetConfig,
    LearnableIQDCOffset,
    apply_iq_dc_numpy,
    apply_iq_dc_torch,
    evm_rms_numpy,
    iq_coefficients,
    power_change_db_numpy,
)


def test_zero_parameters_are_exact_identity():
    rng = np.random.default_rng(7)
    value = rng.normal(size=(3, 2, 4, 4)).astype(np.float32)
    output = apply_iq_dc_numpy(value, IQDCOffsetConfig())
    np.testing.assert_allclose(output, value, atol=0.0, rtol=0.0)


def test_iq_coefficients_zero_errors_are_identity():
    mu, nu = iq_coefficients(0.0, 0.0)
    assert mu == 1 + 0j
    assert nu == 0 + 0j


def test_iq_gain_ratio_is_parameterized_in_db():
    value = np.array([[1.0, 2.0, 1.0, 2.0]])
    output = apply_iq_dc_numpy(value, IQDCOffsetConfig(gain_db=6.0))
    i, q = output[:, :2], output[:, 2:]
    ratio = np.mean(np.abs(i)) / np.mean(np.abs(q))
    assert ratio == pytest.approx(10 ** (6 / 20), rel=1e-12)


def test_phase_error_uses_mirror_term():
    value = np.array([[1.0, 0.0, 0.0, 1.0]])
    output = apply_iq_dc_numpy(value, IQDCOffsetConfig(phase_deg=2.0))
    mu, nu = iq_coefficients(0.0, 2.0)
    z = 1.0 + 0.0j
    expected = mu * z + nu * np.conj(z)
    np.testing.assert_allclose(output[0, 0] + 1j * output[0, 2], expected)


def test_dc_offset_is_constant_per_sample_and_relative_to_rms():
    value = np.array([[1.0, 1.0, 0.0, 0.0]])
    output = apply_iq_dc_numpy(value, IQDCOffsetConfig(dc_magnitude=0.1, dc_phase_deg=90.0))
    rms = 1.0
    np.testing.assert_allclose(output[:, :2], value[:, :2])
    np.testing.assert_allclose(output[:, 2:], 0.1 * rms)


def test_order_ablation_is_noncommutative():
    rng = np.random.default_rng(13)
    value = rng.normal(size=(2, 16))
    forward = apply_iq_dc_numpy(value, IQDCOffsetConfig(1.0, 5.0, 0.1, 45.0, "iq_then_dc"))
    reverse = apply_iq_dc_numpy(value, IQDCOffsetConfig(1.0, 5.0, 0.1, 45.0, "dc_then_iq"))
    assert not np.allclose(forward, reverse)


def test_evm_and_power_metrics_are_finite():
    rng = np.random.default_rng(11)
    value = rng.normal(size=(4, 32))
    impaired = apply_iq_dc_numpy(value, IQDCOffsetConfig(0.5, 2.0, 0.02, 180.0))
    assert math.isfinite(evm_rms_numpy(value, impaired))
    assert math.isfinite(power_change_db_numpy(value, impaired))
    assert evm_rms_numpy(value, impaired) <= 0.1


def test_torch_backend_is_differentiable_and_bounded():
    torch = pytest.importorskip("torch")
    initial = IQDCOffsetConfig(0.5, 2.0, 0.02, 225.0)
    trigger = LearnableIQDCOffset(initial)
    value = torch.randn(2, 16, requires_grad=True)
    output = trigger(value)
    loss = output.square().mean() + trigger.constraint_penalty(value)
    loss.backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    cfg = trigger.config()
    assert -1.0 <= cfg.gain_db <= 1.0
    assert -5.0 <= cfg.phase_deg <= 5.0
    assert 0.0 <= cfg.dc_magnitude <= 0.1
    assert -180.0 <= cfg.dc_phase_deg <= 180.0


def test_constraint_penalty_respects_independent_weights():
    torch = pytest.importorskip("torch")
    value = torch.randn(2, 16)
    trigger = LearnableIQDCOffset(IQDCOffsetConfig(1.0, 5.0, 0.1, 45.0))
    unweighted = trigger.constraint_penalty(value)
    weighted = trigger.constraint_penalty(value, evm_weight=10.0, power_weight=3.0)
    assert weighted >= unweighted


def test_torch_and_numpy_match_for_fixed_config():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    value = rng.normal(size=(2, 16)).astype(np.float32)
    cfg = IQDCOffsetConfig(0.3, -1.0, 0.05, 90.0)
    np.testing.assert_allclose(apply_iq_dc_torch(torch.from_numpy(value), cfg).detach().numpy(), apply_iq_dc_numpy(value, cfg), atol=1e-6)
