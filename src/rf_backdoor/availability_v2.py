"""Protocol and bookkeeping helpers for the availability_v2 experiment.

This module contains the parts of the protocol that must be shared by every
stage: versioned configuration loading, the G0 implementation gate, immutable
resume identities, physical-trigger selection, and the target-free report
contract.  It intentionally does not start jobs or download data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from .evaluation import mean_std_t_ci, psnr_np
from .impairments import (
    IQDCOffsetConfig,
    LearnableIQDCOffset,
    apply_iq_dc_numpy,
    evm_rms_numpy,
    papr_numpy,
    power_change_db_numpy,
)
from .scan import CandidateMeasurement


REPORT_FORBIDDEN_KEYS = frozenset(
    {
        "target",
        "target_image",
        "target_psnr",
        "target_psnr_db",
        "asr",
        "asr_threshold_db",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(_canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_paths(root: Path) -> list[Path]:
    """Return source paths used in the working-tree fingerprint.

    Server bundles and generated runs are deliberately excluded.  The
    fingerprint is stronger than ``HEAD`` alone, which matters while the
    local improved backend is still an uncommitted working-tree change.
    """

    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    paths: list[Path] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        if relative.name.startswith("server_sync_") and relative.suffix == ".bundle":
            continue
        if "runs" in relative.parts or ".venv" in relative.parts or "__pycache__" in relative.parts:
            continue
        path = root / relative
        if path.is_file():
            paths.append(relative)
    return sorted(set(paths), key=lambda item: item.as_posix())


def working_tree_sha256(root: str | Path) -> str | None:
    root_path = Path(root).resolve()
    paths = _git_paths(root_path)
    if not paths:
        return None
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root_path / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions(names: Iterable[str] = ("numpy", "torch", "einops")) -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_v2_provenance(
    root: str | Path,
    *,
    config_sha256: str,
    data_sha256: str,
    seed: int,
    upstream_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_git_commit": commit,
        "code_working_tree_sha256": working_tree_sha256(root_path),
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "upstream_checkpoint_sha256": upstream_checkpoint_sha256,
        "seed": int(seed),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }


def load_toml(path: str | Path) -> dict[str, Any]:
    import tomllib

    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def config_file_for_profile(root: str | Path, profile: str) -> Path:
    if profile == "pilot":
        return Path(root) / "configs" / "availability_v2_pilot.toml"
    if profile in {"full-dev", "full-confirm"}:
        return Path(root) / "configs" / "availability_v2_full.toml"
    raise ValueError(f"unknown availability_v2 profile: {profile}")


def profile_output_name(profile: str) -> str:
    return {"pilot": "pilot", "full-dev": "dev", "full-confirm": "confirm"}[profile]


def config_sha256_for_profile(root: str | Path, profile: str) -> tuple[Path, str, dict[str, Any]]:
    path = config_file_for_profile(root, profile)
    raw = load_toml(path)
    return path, sha256_file(path), raw


def report_has_forbidden_fields(value: Any, *, path: str = "") -> list[str]:
    """Find target-image/ASR fields recursively in a new report payload."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in REPORT_FORBIDDEN_KEYS:
                found.append(f"{path}.{key}" if path else str(key))
            found.extend(report_has_forbidden_fields(child, path=f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(report_has_forbidden_fields(child, path=f"{path}[{index}]"))
    return found


def assert_target_free_report(value: Mapping[str, Any]) -> None:
    forbidden = report_has_forbidden_fields(value)
    if forbidden:
        raise ValueError("availability_v2 report contains forbidden fields: " + ", ".join(forbidden))


def write_target_free_report(path: str | Path, payload: Mapping[str, Any]) -> None:
    assert_target_free_report(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def guard_resume_identity(
    run_dir: str | Path,
    identity: Mapping[str, Any],
    *,
    resume: bool,
) -> bool:
    """Check or create an immutable stage identity.

    Returns ``True`` when the caller may reuse an already completed stage.
    A same-named stage with a different config/data/code/checkpoint identity is
    always an error; no artifact is overwritten.
    """

    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "run_identity.json"
    if identity_path.exists():
        existing = _read_json(identity_path)
        if existing != dict(identity):
            differing = sorted({*existing.keys(), *identity.keys()})
            differing = [key for key in differing if existing.get(key) != identity.get(key)]
            raise RuntimeError(f"stage identity mismatch; refusing to reuse or overwrite {directory}: {differing}")
        return bool(resume)
    identity_path.write_text(json.dumps(dict(identity), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return False


def write_json_once(path: str | Path, payload: Any, *, resume: bool) -> None:
    destination = Path(path)
    if destination.exists() and not resume:
        raise FileExistsError(f"artifact exists; pass --resume to reuse it: {destination}")
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def candidate_is_joint(config: IQDCOffsetConfig) -> bool:
    return (config.gain_db != 0.0 or config.phase_deg != 0.0) and config.dc_magnitude != 0.0


def candidate_selection_key(candidate: CandidateMeasurement) -> tuple[float, float, tuple[float, ...]]:
    config = candidate.config
    config_key = (config.gain_db, config.phase_deg, config.dc_magnitude, config.dc_phase_deg, 0.0 if config.order == "iq_then_dc" else 1.0)
    return (-candidate.evm_rms, candidate.clean_psnr_loss_db, config_key)


def select_v2_trigger_candidate(
    candidates: Iterable[CandidateMeasurement],
    *,
    evm_limit: float = 0.10,
    power_change_db_limit: float = 0.5,
    clean_psnr_drop_limit: float = 0.25,
    clean_failure_rate_limit: float = 0.01,
    require_joint: bool = True,
) -> CandidateMeasurement:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.evm_rms <= evm_limit
        and abs(candidate.power_change_db) <= power_change_db_limit
        and candidate.clean_psnr_loss_db <= clean_psnr_drop_limit
        and candidate.clean_false_activation_rate <= clean_failure_rate_limit
        and (not require_joint or candidate_is_joint(candidate.config))
    ]
    if not eligible:
        raise ValueError("no joint IQ+DC candidate satisfies v2 physical and clean-model constraints")
    return sorted(eligible, key=candidate_selection_key)[0]


def availability_metrics_np(
    clean_reconstruction: np.ndarray,
    triggered_reconstruction: np.ndarray,
    source: np.ndarray,
    *,
    failure_drop_db: float,
) -> dict[str, float]:
    clean_psnr = psnr_np(clean_reconstruction, source)
    triggered_psnr = psnr_np(triggered_reconstruction, source)
    clean_mse = np.square(clean_reconstruction - source).reshape(source.shape[0], -1).mean(axis=1)
    triggered_mse = np.square(triggered_reconstruction - source).reshape(source.shape[0], -1).mean(axis=1)
    drop = clean_psnr - triggered_psnr
    return {
        "clean_psnr_db": float(clean_psnr.mean()),
        "triggered_psnr_db": float(triggered_psnr.mean()),
        "clean_mse": float(clean_mse.mean()),
        "triggered_mse": float(triggered_mse.mean()),
        "mse_ratio": float(triggered_mse.mean() / max(float(clean_mse.mean()), 1e-12)),
        "psnr_drop_db": float(drop.mean()),
        "psnr_drop_standard_deviation_db": float(drop.std(ddof=1)) if drop.size > 1 else 0.0,
        "failure_rate": float(np.mean(drop >= float(failure_drop_db))),
        "failure_drop_threshold_db": float(failure_drop_db),
    }


def trigger_stealth_metrics(reference: np.ndarray, impaired: np.ndarray) -> dict[str, float]:
    return {
        "evm": evm_rms_numpy(reference, impaired),
        "power_change_db": power_change_db_numpy(reference, impaired),
        "reference_papr": papr_numpy(reference),
        "triggered_papr": papr_numpy(impaired),
        "reference_mean_power": float(np.mean(np.abs(reference) ** 2)),
        "triggered_mean_power": float(np.mean(np.abs(impaired) ** 2)),
    }


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    checks: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "status": "PASS" if self.passed else "FAIL", "checks": dict(self.checks)}


def _finite_gradients(trigger: Any) -> tuple[bool, float]:
    gradients = [parameter.grad for parameter in trigger.parameters()]
    finite = all(gradient is not None and bool(np.isfinite(gradient.detach().cpu().numpy()).all()) for gradient in gradients)
    maximum = max((float(gradient.detach().abs().max()) for gradient in gradients if gradient is not None), default=0.0)
    return finite, maximum


def run_g0(*, seed: int = 0, device: str = "cpu") -> GateResult:
    """Run the mathematical and implementation gate on synthetic data."""

    try:
        import torch
    except ImportError as exc:
        return GateResult(
            "G0",
            False,
            {
                "torch_available": False,
                "error": str(exc),
                "remediation": "run preflight in the isolated server environment with the project Torch runtime",
            },
        )

    from .adapter import DeepJSCCAdapter
    from src.channels import Channel, ChannelConfig

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    complex_value = (rng.normal(size=(4, 32)) + 1j * rng.normal(size=(4, 32))).astype(np.complex64)
    legacy_value = np.concatenate([complex_value.real, complex_value.imag], axis=1)
    zero_output = apply_iq_dc_numpy(complex_value, IQDCOffsetConfig())
    native_legacy = apply_iq_dc_numpy(legacy_value, IQDCOffsetConfig(0.4, -2.0, 0.03, 35.0))
    native_output = apply_iq_dc_numpy(complex_value, IQDCOffsetConfig(0.4, -2.0, 0.03, 35.0))
    legacy_as_complex = native_legacy[:, :32] + 1j * native_legacy[:, 32:]

    input_tensor = torch.from_numpy(complex_value.copy())
    trigger = LearnableIQDCOffset(IQDCOffsetConfig(0.25, 1.0, 0.03, 45.0))
    trigger_input = input_tensor.clone().requires_grad_(True)
    trigger_output = trigger(trigger_input)
    trigger_output.abs().square().mean().backward()
    finite_gradients, max_gradient = _finite_gradients(trigger)

    channel = Channel(ChannelConfig(channel_type="awgn", snr_min=7.0, snr_max=7.0))
    signal = input_tensor
    torch.manual_seed(seed + 1)
    state_a = channel.sampling_channel_state(signal.shape[0], signal.shape[1], signal.device, signal.dtype)
    output_a = channel(signal, state_a)
    torch.manual_seed(seed + 1)
    state_b = channel.sampling_channel_state(signal.shape[0], signal.shape[1], signal.device, signal.dtype)
    output_b = channel(signal, state_b)
    untouched = torch.equal(signal, input_tensor)

    adapter = DeepJSCCAdapter.from_local(c=8, channel_type="awgn", snr_db=7.0)
    adapter.model.to(device)
    images = torch.rand(2, 3, 32, 32, device=device)
    with torch.no_grad():
        latent = adapter.encode(images)
        received = adapter.apply_channel(latent, 7.0)
        reconstruction = adapter.decode(received)

    checks: dict[str, Any] = {
        "zero_parameter_max_identity_error": float(np.max(np.abs(zero_output - complex_value))),
        "native_complex_vs_legacy_max_abs_error": float(np.max(np.abs(native_output - legacy_as_complex))),
        "learnable_gradients_finite": finite_gradients,
        "learnable_max_gradient": max_gradient,
        "learnable_nonzero_gradient": max_gradient > 1e-8,
        "paired_awgn_exact": bool(torch.equal(output_a, output_b)),
        "input_unchanged": untouched,
        "latent_is_native_complex_bxl": bool(latent.ndim == 2 and latent.is_complex()),
        "minimal_forward_finite": bool(torch.isfinite(reconstruction).all()),
        "minimal_forward_shape": list(reconstruction.shape),
    }
    passed = (
        checks["zero_parameter_max_identity_error"] <= 1e-6
        and checks["native_complex_vs_legacy_max_abs_error"] <= 1e-6
        and checks["learnable_gradients_finite"]
        and checks["learnable_nonzero_gradient"]
        and checks["paired_awgn_exact"]
        and checks["input_unchanged"]
        and checks["latent_is_native_complex_bxl"]
        and checks["minimal_forward_finite"]
    )
    return GateResult("G0", bool(passed), checks)


def summarize_seed_metric(values: Iterable[float]) -> dict[str, Any]:
    return mean_std_t_ci(np.asarray(tuple(values), dtype=float))
