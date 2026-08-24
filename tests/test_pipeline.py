import pytest

from rf_backdoor.impairments import IQDCOffsetConfig


def test_candidate_measurement_accepts_numpy_power_arrays():
    torch = pytest.importorskip("torch")
    from scripts.run_availability_pipeline import _candidate_measurement

    class IdentityAdapter:
        def encode(self, images):
            return images.to(torch.complex64)

        def decode(self, received):
            return received.real

    images = torch.tensor(
        [[0.0, 0.25, 0.5, 0.75], [0.1, 0.3, 0.6, 0.9]],
        dtype=torch.float32,
    )
    loader = torch.utils.data.DataLoader(images, batch_size=2, shuffle=False)

    measurement = _candidate_measurement(
        IdentityAdapter(),
        loader,
        IQDCOffsetConfig(),
        device="cpu",
        failure_drop_db=3.0,
    )

    assert measurement.evm_rms == pytest.approx(0.0)
    assert measurement.power_change_db == pytest.approx(0.0)
    assert measurement.clean_psnr_loss_db == pytest.approx(0.0)
    assert measurement.clean_false_activation_rate == pytest.approx(0.0)
