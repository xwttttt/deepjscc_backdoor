"""Probe what a trained IQ/DC trigger actually does to the DIV2K codec.

Conditions compared on the same images and the same channel realization:

* ``zero``             - no trigger (clean backdoor branch)
* ``iq_only``          - learned gain/phase, no DC
* ``dc_only``          - learned DC, no gain/phase
* ``full``             - the trained trigger as saved
* ``equal_evm_random`` - random complex perturbation with the full trigger's EVM
* ``dc_sign_flip``     - learned DC with the opposite sign

Reported per condition: PSNR drop against the clean branch, the localized
structure drop used by the v2 objective, the EVM of the perturbation, and how
much of the residual is a single shared template across images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from div2k_prepared_dataset import PreparedDIV2K  # noqa: E402
from rf_backdoor.adapter import DeepJSCCAdapter  # noqa: E402
from rf_backdoor.impairments import IQDCOffsetConfig, apply_iq_dc_torch  # noqa: E402
from rf_backdoor.structural_drop import structural_drop  # noqa: E402


def _torch():
    import torch

    return torch


def load_adapter(device, checkpoint: Path, c: int, snr_db: float) -> DeepJSCCAdapter:
    from src.channels import Channel, ChannelConfig
    from src.deepjscc.model import DeepJSCC

    adapter = DeepJSCCAdapter(
        DeepJSCC(c=c),
        channel=Channel(ChannelConfig(channel_type="awgn", snr_min=snr_db, snr_max=snr_db)),
        baseline_path=ROOT / "src" / "src",
    )
    adapter.model.to(device)
    payload = _torch().load(checkpoint, map_location=device, weights_only=True)
    adapter.model.load_state_dict(payload["model"])
    adapter.model.eval()
    return adapter


def perturbation(latent, condition, config, *, generator):
    """Return the triggered latent for one condition."""

    torch = _torch()
    if condition == "zero":
        return latent
    if condition == "full":
        return apply_iq_dc_torch(latent, config)
    if condition == "iq_only":
        return apply_iq_dc_torch(latent, IQDCOffsetConfig(
            gain_db=config.gain_db, phase_deg=config.phase_deg,
            dc_magnitude=0.0, dc_phase_deg=config.dc_phase_deg, order=config.order,
        ))
    if condition == "dc_only":
        return apply_iq_dc_torch(latent, IQDCOffsetConfig(
            gain_db=0.0, phase_deg=0.0,
            dc_magnitude=config.dc_magnitude, dc_phase_deg=config.dc_phase_deg, order=config.order,
        ))
    if condition == "dc_sign_flip":
        return apply_iq_dc_torch(latent, IQDCOffsetConfig(
            gain_db=0.0, phase_deg=0.0,
            dc_magnitude=config.dc_magnitude, dc_phase_deg=config.dc_phase_deg + 180.0,
            order=config.order,
        ))
    raise ValueError(f"unsupported condition: {condition}")


def evm_rms(reference, modified) -> float:
    delta = (modified - reference).abs().square().mean()
    power = reference.abs().square().mean().clamp_min(1e-12)
    return float((delta / power).sqrt())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--snr-db", type=float, default=100.0)
    parser.add_argument("--c", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    torch = _torch()
    device = torch.device(args.device)
    raw = json.loads(args.trigger.read_text(encoding="utf-8"))["config"]
    config = IQDCOffsetConfig(
        gain_db=float(raw["gain_db"]), phase_deg=float(raw["phase_deg"]),
        dc_magnitude=float(raw["dc_magnitude"]), dc_phase_deg=float(raw["dc_phase_deg"]),
        order=str(raw["order"]),
    )
    adapter = load_adapter(device, args.checkpoint, args.c, args.snr_db)
    prepared = PreparedDIV2K(args.dataset)
    count = min(int(args.limit), len(prepared))
    images = torch.stack([prepared[index][0] for index in range(count)]).to(device)

    conditions = ["zero", "iq_only", "dc_only", "full", "equal_evm_random", "dc_sign_flip"]
    results: dict[str, dict[str, float]] = {}
    residuals: dict[str, np.ndarray] = {}
    evm_reference = None

    with torch.no_grad():
        latent = adapter.encode(images)
        generator = torch.Generator(device=device).manual_seed(int(args.seed))
        state = adapter.sample_channel_state(latent, args.snr_db, generator=generator)
        received = adapter.apply_channel(latent, args.snr_db, channel_state=state, generator=generator)
        clean = adapter.decode(received).clamp(0.0, 1.0)

        full_latent = perturbation(latent, "full", config, generator=generator)
        evm_reference = evm_rms(latent, full_latent)
        noise = torch.complex(
            torch.randn(latent.shape, generator=generator, device=device),
            torch.randn(latent.shape, generator=generator, device=device),
        )
        target = float((full_latent - latent).abs().square().mean().sqrt())
        noise = noise / noise.abs().square().mean().sqrt().clamp_min(1e-12) * target

        for condition in conditions:
            if condition == "zero":
                triggered_latent = latent
            elif condition == "equal_evm_random":
                triggered_latent = latent + noise.to(latent.dtype)
            else:
                triggered_latent = perturbation(latent, condition, config, generator=generator)
            received = adapter.apply_channel(
                triggered_latent, args.snr_db, channel_state=state, generator=generator
            )
            reconstruction = adapter.decode(received).clamp(0.0, 1.0)
            total, magnitude, localization = structural_drop(clean, reconstruction, images)
            clean_mse = (clean - images).square().flatten(1).mean(1).clamp_min(1e-12)
            triggered_mse = (reconstruction - images).square().flatten(1).mean(1).clamp_min(1e-12)
            drop = (10.0 * torch.log10(triggered_mse / clean_mse))
            results[condition] = {
                "psnr_drop_db": float(drop.mean()),
                "structure_drop_db": float(total.mean()),
                "structure_magnitude_db": float(magnitude.mean()),
                "structure_localization_db": float(localization.mean()),
                "evm_rms": evm_rms(latent, triggered_latent),
            }
            residuals[condition] = (
                (images - reconstruction).flatten(1).float().cpu().numpy()
            )

    for condition, block in results.items():
        residual = residuals[condition]
        template = residual.mean(axis=0)
        template_energy = float((template ** 2).sum())
        mean_energy = float((residual ** 2).sum(axis=1).mean())
        norms = np.linalg.norm(residual, axis=1).clip(min=1e-12)
        cosine = float(np.mean((residual @ template) / (norms * (np.linalg.norm(template) + 1e-12))))
        block["residual_template_energy_fraction"] = template_energy / max(mean_energy, 1e-12)
        block["residual_template_cosine"] = cosine

    payload = {
        "checkpoint": str(args.checkpoint),
        "trigger": str(args.trigger),
        "trigger_config": raw,
        "images": count,
        "snr_db": args.snr_db,
        "full_trigger_evm_rms": evm_reference,
        "conditions": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = f"{'condition':>17} {'psnr drop':>10} {'struct':>8} {'mag':>6} {'loc':>6} {'EVM':>7} {'TEF':>6} {'cos':>6}"
    print(header)
    print("-" * len(header))
    for condition, block in results.items():
        print(
            f"{condition:>17} {block['psnr_drop_db']:>10.3f} {block['structure_drop_db']:>8.2f} "
            f"{block['structure_magnitude_db']:>6.2f} {block['structure_localization_db']:>6.2f} "
            f"{block['evm_rms']:>7.4f} {block['residual_template_energy_fraction']:>6.3f} "
            f"{block['residual_template_cosine']:>6.3f}"
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
