"""Run the CIFAR-10 P0--P2 study on the dedicated server checkout.

All paths default below ``/home/gxhen/Wisig/jscc/deepjscc_backdoor`` when the
script is launched from that directory. The upstream checkout is kept under
``external/`` and is never edited.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np

from rf_backdoor.adapter import ChunbaobaoDeepJSCCAdapter
from rf_backdoor.datasets import CIFAR10Dataset
from rf_backdoor.evaluation import compute_four_way_report, freeze_asr_threshold, mean_std_t_ci, paired_psnr_drop_np, target_asr_np, target_psnr_np
from rf_backdoor.impairments import IQDCOffsetConfig, apply_iq_dc_torch, evm_rms_numpy, iq_mean_numpy, nmse_numpy, papr_numpy, power_change_db_numpy
from rf_backdoor.metadata import collect_run_metadata, sha256_file, write_run_metadata
from rf_backdoor.poisoning import DeterministicPoisonedDataset, build_poison_manifest, build_split_indices, sha256_array
from rf_backdoor.scan import CandidateMeasurement, iter_fixed_trigger_configs, select_trigger_candidate, trigger_config_hash
from rf_backdoor.training import TrainingConfig, fit, set_seed


P1_V2_ROOT = "P1_per_sample_v2"
P2_V2_ROOT = "P2_per_sample_v2"


def _torch() -> Any:
    import torch

    return torch


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def _load_checkpoint(adapter: Any, path: Path, device: Any) -> dict[str, Any]:
    torch = _torch()
    payload = torch.load(path, map_location=device)
    adapter.model.load_state_dict(payload["model"])
    return payload


def _make_loaders(root: Path, *, split_seed: int, batch_size: int, workers: int, manifest: Any | None = None) -> tuple[Any, Any, Any, tuple[int, ...], tuple[int, ...]]:
    torch = _torch()
    from torch.utils.data import DataLoader, Subset

    base = CIFAR10Dataset(root / "data", train=True)
    train_indices, validation_indices = build_split_indices(len(base), target_index=0, train_count=45_000, seed=split_seed)
    target, _ = base[0]
    if manifest is None:
        train_dataset = Subset(base, train_indices)
    else:
        wrapped = DeterministicPoisonedDataset(base, manifest, return_metadata=True)
        train_dataset = Subset(wrapped, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    validation_loader = DataLoader(Subset(base, validation_indices), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return base, target, (train_loader, validation_loader), train_indices, validation_indices


def _make_adapter(external: Path, device: Any) -> ChunbaobaoDeepJSCCAdapter:
    adapter = ChunbaobaoDeepJSCCAdapter.from_repo(external, c=8, channel_type="AWGN", snr_db=7.0)
    adapter.model.to(device)
    return adapter


def _common_random_channel(items: list[tuple[Any, Any]], snr_db: float) -> list[Any]:
    """Apply channels with one shared standardized AWGN realization.

    The upstream channel scales noise to each latent's power. Replaying the RNG
    state therefore keeps the Gaussian draw paired while preserving the proper
    per-condition noise scale. The global RNG advances exactly once.
    """

    torch = _torch()
    cpu_start = torch.random.get_rng_state()
    cuda_start = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cpu_end = None
    cuda_end = None
    outputs = []
    for index, (adapter, latent) in enumerate(items):
        torch.random.set_rng_state(cpu_start)
        if cuda_start is not None:
            torch.cuda.set_rng_state_all(cuda_start)
        outputs.append(adapter.apply_channel(latent, snr_db))
        if index == 0:
            cpu_end = torch.random.get_rng_state()
            cuda_end = torch.cuda.get_rng_state_all() if cuda_start is not None else None
    if cpu_end is not None:
        torch.random.set_rng_state(cpu_end)
    if cuda_end is not None:
        torch.cuda.set_rng_state_all(cuda_end)
    return outputs


def _is_joint_impairment(config: IQDCOffsetConfig) -> bool:
    """Return whether both IQ imbalance and DC offset are materially enabled."""

    return (config.gain_db != 0.0 or config.phase_deg != 0.0) and config.dc_magnitude != 0.0


def _control_configs(base: IQDCOffsetConfig) -> dict[str, IQDCOffsetConfig]:
    """Build pre-registered channel controls from the frozen trigger."""

    clamp = lambda value, low, high: max(low, min(high, value))
    return {
        "iq_only": IQDCOffsetConfig(base.gain_db, base.phase_deg, 0.0, 0.0, base.order),
        "dc_only": IQDCOffsetConfig(0.0, 0.0, base.dc_magnitude, base.dc_phase_deg, base.order),
        "sign_flip": IQDCOffsetConfig(-base.gain_db, base.phase_deg, base.dc_magnitude, base.dc_phase_deg + 180.0, base.order),
        "phase_flip": IQDCOffsetConfig(base.gain_db, -base.phase_deg, base.dc_magnitude, base.dc_phase_deg + 180.0, base.order),
        "order_swap": IQDCOffsetConfig(base.gain_db, base.phase_deg, base.dc_magnitude, base.dc_phase_deg, "dc_then_iq" if base.order == "iq_then_dc" else "iq_then_dc"),
        "parameter_neighborhood": IQDCOffsetConfig(
            clamp(base.gain_db + 0.25, -1.0, 1.0),
            clamp(base.phase_deg + 1.0, -5.0, 5.0),
            clamp(base.dc_magnitude + 0.01, 0.0, 0.1),
            base.dc_phase_deg,
            base.order,
        ),
        "phase_rotation": IQDCOffsetConfig(base.gain_db, base.phase_deg, base.dc_magnitude, base.dc_phase_deg + 45.0, base.order),
    }


def _equal_evm_random_torch(value: Any, evm_target: float) -> Any:
    torch = _torch()
    noise = torch.randn_like(value)
    reference_rms = torch.sqrt(value.square().mean(dim=tuple(range(1, value.ndim)), keepdim=True).clamp_min(1e-12))
    noise_rms = torch.sqrt(noise.square().mean(dim=tuple(range(1, value.ndim)), keepdim=True).clamp_min(1e-12))
    return value + float(evm_target) * reference_rms * noise / noise_rms


def _evaluate_controls(
    clean_adapter: Any,
    backdoor_adapter: Any,
    test_loader: Any,
    target: np.ndarray,
    base_config: IQDCOffsetConfig,
    base_evm: float,
    device: Any,
    evaluation_seed: int,
) -> dict[str, Any]:
    """Evaluate registered trigger controls at the training SNR."""

    controls: dict[str, IQDCOffsetConfig | None] = _control_configs(base_config)
    controls["equal_evm_random"] = None
    results: dict[str, Any] = {}
    torch = _torch()
    for control_index, (name, config) in enumerate(controls.items()):
        set_seed(evaluation_seed + control_index)
        clean_outputs: list[np.ndarray] = []
        backdoor_outputs: list[np.ndarray] = []
        reference_parts: list[np.ndarray] = []
        impaired_parts: list[np.ndarray] = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch[0].to(device, non_blocking=True)
                clean_latent = clean_adapter.encode(images)
                backdoor_latent = backdoor_adapter.encode(images)
                if config is None:
                    clean_trigger_latent = _equal_evm_random_torch(clean_latent, base_evm)
                    backdoor_trigger_latent = _equal_evm_random_torch(backdoor_latent, base_evm)
                else:
                    clean_trigger_latent = apply_iq_dc_torch(clean_latent, config)
                    backdoor_trigger_latent = apply_iq_dc_torch(backdoor_latent, config)
                clean_rx, backdoor_rx = _common_random_channel(
                    [(clean_adapter, clean_trigger_latent), (backdoor_adapter, backdoor_trigger_latent)],
                    7.0,
                )
                clean_outputs.append(clean_adapter.decode(clean_rx).detach().cpu().numpy())
                backdoor_outputs.append(backdoor_adapter.decode(backdoor_rx).detach().cpu().numpy())
                reference_parts.append(backdoor_latent.detach().cpu().numpy())
                impaired_parts.append(backdoor_trigger_latent.detach().cpu().numpy())
        reference = np.concatenate(reference_parts)
        impaired = np.concatenate(impaired_parts)
        clean_output = np.concatenate(clean_outputs)
        backdoor_output = np.concatenate(backdoor_outputs)
        results[name] = {
            "snr_db": 7.0,
            "config": asdict(config) if config is not None else None,
            "clean_model_asr": target_asr_np(clean_output, target, psnr_threshold_db=25.0),
            "backdoored_model_asr": target_asr_np(backdoor_output, target, psnr_threshold_db=25.0),
            "evm_rms": evm_rms_numpy(reference, impaired),
            "nmse": nmse_numpy(reference, impaired),
            "power_change_db": power_change_db_numpy(reference, impaired),
        }
    return {
        "snr_db": 7.0,
        "base_config": asdict(base_config),
        "base_evm_rms": base_evm,
        "definitions": {
            "sign_flip": "IQ gain sign and DC phase shifted by 180 degrees",
            "phase_flip": "IQ phase sign and DC phase shifted by 180 degrees",
            "parameter_neighborhood": "+0.25 dB gain, +1 degree IQ phase, +0.01 DC magnitude, clamped to scan bounds",
            "equal_evm_random": "independent Gaussian latent perturbation normalized per sample to the base trigger EVM",
        },
        "controls": results,
    }


def run_p0(args: argparse.Namespace, root: Path, device: Any) -> Path:
    torch = _torch()
    set_seed(args.seed)
    base, target, loaders, _, validation_indices = _make_loaders(root, split_seed=args.split_seed, batch_size=args.batch_size, workers=args.workers)
    adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    config = TrainingConfig(snr_db=7.0, max_epochs=args.epochs, early_stopping_start=args.early_stopping_start, patience=args.patience)
    phase_dir = root / "runs" / "P0" / f"seed_{args.seed}"
    history = fit(adapter, loaders[0], loaders[1], optimizer, config=config, device=device, checkpoint_path=phase_dir / "best.pt")
    _dump(phase_dir / "history.json", history)
    _dump(phase_dir / "data.json", {"dataset": "CIFAR10", "archive_sha256": base.archive_sha256, "target_sha256": sha256_array(target), "validation_count": len(validation_indices), "seed": args.seed})
    metadata = collect_run_metadata(root, upstream_path=root / "external" / "chunbaobao-deepjscc", seed=args.seed)
    metadata.update({"phase": "P0", "device": str(device), "epochs_requested": args.epochs})
    write_run_metadata(phase_dir / "run_metadata.json", metadata, artifacts=[phase_dir / "history.json", phase_dir / "best.pt"])
    return phase_dir / "best.pt"


def _measure_candidate(adapter: Any, loader: Any, target: Any, config: IQDCOffsetConfig, device: Any, *, max_samples: int | None) -> CandidateMeasurement:
    torch = _torch()
    adapter.model.eval()
    reference_parts: list[np.ndarray] = []
    impaired_parts: list[np.ndarray] = []
    clean_parts: list[np.ndarray] = []
    triggered_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            latent = adapter.encode(images)
            triggered = apply_iq_dc_torch(latent, config)
            clean_rx, triggered_rx = _common_random_channel([(adapter, latent), (adapter, triggered)], 7.0)
            clean_output = adapter.decode(clean_rx)
            triggered_output = adapter.decode(triggered_rx)
            reference_parts.append(latent.detach().cpu().numpy())
            impaired_parts.append(triggered.detach().cpu().numpy())
            clean_parts.append(clean_output.detach().cpu().numpy())
            triggered_parts.append(triggered_output.detach().cpu().numpy())
            source_parts.append(images.detach().cpu().numpy())
            seen += images.shape[0]
            if max_samples and seen >= max_samples:
                break
    reference = np.concatenate(reference_parts)[:max_samples]
    impaired = np.concatenate(impaired_parts)[:max_samples]
    clean = np.concatenate(clean_parts)[:max_samples]
    triggered = np.concatenate(triggered_parts)[:max_samples]
    source = np.concatenate(source_parts)[:max_samples]
    target_np = target.detach().cpu().numpy()
    trigger_target_psnr = target_psnr_np(triggered, target_np)
    return CandidateMeasurement(
        config=config,
        evm_rms=evm_rms_numpy(reference, impaired),
        power_change_db=power_change_db_numpy(reference, impaired),
        clean_psnr_loss_db=float(np.mean(paired_psnr_drop_np(clean, triggered, source))),
        clean_false_activation_rate=float(np.mean(trigger_target_psnr >= 25.0)),
    )


def run_p1(args: argparse.Namespace, root: Path, device: Any, clean_checkpoint: Path) -> IQDCOffsetConfig:
    torch = _torch()
    base, target, loaders, _, _ = _make_loaders(root, split_seed=args.split_seed, batch_size=args.batch_size, workers=args.workers)
    adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
    _load_checkpoint(adapter, clean_checkpoint, device)
    measurements = [_measure_candidate(adapter, loaders[1], target, config, device, max_samples=args.scan_samples) for config in iter_fixed_trigger_configs()]
    selection_pool = measurements if args.allow_single_impairment_selection else [item for item in measurements if _is_joint_impairment(item.config)]
    if not selection_pool:
        raise RuntimeError("no joint IQ+DC candidates were measured; cannot select the required combined trigger")
    selected = select_trigger_candidate(selection_pool)
    phase_dir = root / "runs" / P1_V2_ROOT
    _dump(phase_dir / "measurements.json", [{**asdict(item), "config": asdict(item.config), "trigger_hash": trigger_config_hash(item.config)} for item in measurements])
    _dump(phase_dir / "selected_trigger.json", {
        "config": asdict(selected.config),
        "trigger_hash": trigger_config_hash(selected.config),
        "evm_rms": selected.evm_rms,
        "power_change_db": selected.power_change_db,
        "clean_psnr_loss_db": selected.clean_psnr_loss_db,
        "clean_false_activation_rate": selected.clean_false_activation_rate,
        "selection_policy": "max_evm_then_min_clean_psnr_loss_then_lexicographic_among_joint_nonzero_candidates" if not args.allow_single_impairment_selection else "max_evm_then_min_clean_psnr_loss_then_lexicographic_among_all_candidates",
        "selection_pool_size": len(selection_pool),
    })
    # The threshold is frozen exactly once for the selected candidate.
    trigger_scores: list[np.ndarray] = []
    with torch.no_grad():
        seen = 0
        for batch in loaders[1]:
            images = batch[0].to(device)
            output = adapter.forward_with_impairment(images, 7.0, selected.config)
            trigger_scores.append(target_psnr_np(output.detach().cpu().numpy(), target.detach().cpu().numpy()))
            seen += images.shape[0]
            if args.scan_samples and seen >= args.scan_samples:
                break
    threshold = freeze_asr_threshold(np.concatenate(trigger_scores)[:args.scan_samples])
    _dump(phase_dir / "asr_threshold.json", {"threshold_db": threshold, "source": "clean_model_trigger_validation_Q0.999", "minimum_db": 25.0})
    return selected.config


def run_p2(args: argparse.Namespace, root: Path, device: Any, trigger_config: IQDCOffsetConfig) -> None:
    torch = _torch()
    selected = trigger_config
    for trigger_kind in args.trigger_types:
        penalty_pairs = [(0.0, 0.0)] if trigger_kind == "fixed" else list(product(args.evm_penalty_weights, args.power_penalty_weights))
        for evm_penalty_weight, power_penalty_weight in penalty_pairs:
            for poison_rate in args.poison_rates:
                for seed in args.seeds:
                    set_seed(seed)
                    base, target, _, train_indices, _ = _make_loaders(root, split_seed=args.split_seed, batch_size=args.batch_size, workers=args.workers)
                    target_sha = sha256_array(target)
                    manifest = build_poison_manifest("CIFAR10", len(base), poison_rate, seed=seed, target_index=0, target_sha256=target_sha, eligible_indices=train_indices)
                    _, _, loaders, _, _ = _make_loaders(root, split_seed=args.split_seed, batch_size=args.batch_size, workers=args.workers, manifest=manifest)
                    adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
                    trigger = selected
                    parameters = list(adapter.model.parameters())
                    if trigger_kind == "learnable":
                        from rf_backdoor.impairments import LearnableIQDCOffset

                        trigger = LearnableIQDCOffset(selected).to(device)
                        parameters += list(trigger.parameters())
                    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
                    config = TrainingConfig(
                        snr_db=7.0,
                        poison_weight=1.0,
                        evm_penalty_weight=evm_penalty_weight,
                        power_penalty_weight=power_penalty_weight,
                        max_epochs=args.epochs,
                        early_stopping_start=args.early_stopping_start,
                        patience=args.patience,
                    )
                    run_root = root / "runs" / P2_V2_ROOT / trigger_kind
                    if trigger_kind == "learnable":
                        run_root /= f"penalty_evm_{evm_penalty_weight:g}_power_{power_penalty_weight:g}"
                    run_dir = run_root / f"poison_{poison_rate:g}" / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    manifest.write_json(run_dir / "poison_manifest.json")
                    history = fit(adapter, loaders[0], loaders[1], optimizer, config=config, device=device, trigger=trigger, target=target.to(device), checkpoint_path=run_dir / "best.pt")
                    _dump(run_dir / "history.json", history)
                    if trigger_kind == "learnable":
                        best_payload = torch.load(run_dir / "best.pt", map_location=device)
                        trigger.load_state_dict(best_payload["trigger"])
                    _dump(run_dir / "trigger.json", {
                        "kind": trigger_kind,
                        "objective_version": "per_sample_v2",
                        "config": asdict(trigger.config()) if trigger_kind == "learnable" else asdict(trigger),
                        "manifest_sha256": manifest.manifest_sha256,
                        "evm_penalty_weight": evm_penalty_weight,
                        "power_penalty_weight": power_penalty_weight,
                    })
                    metadata = collect_run_metadata(root, upstream_path=root / "external" / "chunbaobao-deepjscc", seed=seed)
                    metadata.update({
                        "phase": "P2",
                        "trigger_kind": trigger_kind,
                        "poison_rate": poison_rate,
                        "manifest_sha256": manifest.manifest_sha256,
                        "evm_penalty_weight": evm_penalty_weight,
                        "power_penalty_weight": power_penalty_weight,
                        "objective_version": "per_sample_v2",
                    })
                    write_run_metadata(run_dir / "run_metadata.json", metadata, artifacts=[run_dir / "best.pt", run_dir / "history.json", run_dir / "poison_manifest.json"])


def _trigger_for_run(run_dir: Path, selected_config: IQDCOffsetConfig, checkpoint: dict[str, Any], device: Any) -> Any:
    torch = _torch()
    payload = json.loads((run_dir / "trigger.json").read_text(encoding="utf-8"))
    if payload["kind"] == "fixed":
        return selected_config
    from rf_backdoor.impairments import LearnableIQDCOffset

    trigger = LearnableIQDCOffset(selected_config).to(device)
    if "trigger" not in checkpoint:
        raise RuntimeError(f"learnable trigger state missing from checkpoint: {run_dir / 'best.pt'}")
    trigger.load_state_dict(checkpoint["trigger"])
    trigger.eval()
    return trigger


def run_evaluate(args: argparse.Namespace, root: Path, device: Any, selected_config: IQDCOffsetConfig) -> None:
    torch = _torch()
    from torch.utils.data import DataLoader

    data = CIFAR10Dataset(root / "data", train=True)
    test = CIFAR10Dataset(root / "data", train=False)
    target = data[0][0].numpy()
    threshold = json.loads((root / "runs" / P1_V2_ROOT / "asr_threshold.json").read_text(encoding="utf-8"))["threshold_db"]
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    clean_adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
    clean_checkpoint_path = root / "runs" / "P0" / f"seed_{args.seed}" / "best.pt"
    _load_checkpoint(clean_adapter, clean_checkpoint_path, device)
    clean_adapter.model.eval()
    run_paths = sorted((root / "runs" / P2_V2_ROOT).rglob("best.pt"))
    for checkpoint_path in run_paths:
        run_dir = checkpoint_path.parent
        trigger_payload = json.loads((run_dir / "trigger.json").read_text(encoding="utf-8"))
        if trigger_payload.get("objective_version") != "per_sample_v2" and not args.include_legacy_runs:
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device)
        backdoor_adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
        backdoor_adapter.model.load_state_dict(checkpoint["model"])
        backdoor_adapter.model.eval()
        trigger = _trigger_for_run(run_dir, selected_config, checkpoint, device)
        base_config = trigger if isinstance(trigger, IQDCOffsetConfig) else trigger.config()
        scores = []
        trigger_stealth: dict[str, Any] | None = None
        for snr_index, snr_db in enumerate(args.test_snr_db):
            clean_clean_parts: list[np.ndarray] = []
            clean_trigger_parts: list[np.ndarray] = []
            backdoor_clean_parts: list[np.ndarray] = []
            backdoor_trigger_parts: list[np.ndarray] = []
            source_parts: list[np.ndarray] = []
            latent_parts: list[np.ndarray] = []
            trigger_latent_parts: list[np.ndarray] = []
            with torch.no_grad():
                for batch in test_loader:
                    images = batch[0].to(device, non_blocking=True)
                    source_parts.append(images.cpu().numpy())
                    clean_latent = clean_adapter.encode(images)
                    backdoor_latent = backdoor_adapter.encode(images)
                    if isinstance(trigger, IQDCOffsetConfig):
                        clean_trigger_latent = apply_iq_dc_torch(clean_latent, trigger)
                        backdoor_trigger_latent = apply_iq_dc_torch(backdoor_latent, trigger)
                    else:
                        clean_trigger_latent = trigger(clean_latent)
                        backdoor_trigger_latent = trigger(backdoor_latent)
                    if snr_index == 0:
                        latent_parts.append(backdoor_latent.detach().cpu().numpy())
                        trigger_latent_parts.append(backdoor_trigger_latent.detach().cpu().numpy())
                    clean_clean_rx, clean_trigger_rx, backdoor_clean_rx, backdoor_trigger_rx = _common_random_channel(
                        [
                            (clean_adapter, clean_latent),
                            (clean_adapter, clean_trigger_latent),
                            (backdoor_adapter, backdoor_latent),
                            (backdoor_adapter, backdoor_trigger_latent),
                        ],
                        snr_db,
                    )
                    clean_clean_parts.append(clean_adapter.decode(clean_clean_rx).cpu().numpy())
                    clean_trigger_parts.append(clean_adapter.decode(clean_trigger_rx).cpu().numpy())
                    backdoor_clean_parts.append(backdoor_adapter.decode(backdoor_clean_rx).cpu().numpy())
                    backdoor_trigger_parts.append(backdoor_adapter.decode(backdoor_trigger_rx).cpu().numpy())
            if snr_index == 0:
                latent_array = np.concatenate(latent_parts)
                trigger_latent_array = np.concatenate(trigger_latent_parts)
                evm = evm_rms_numpy(latent_array, trigger_latent_array)
                power_change = power_change_db_numpy(latent_array, trigger_latent_array)
                trigger_stealth = {
                    "evm_rms": evm,
                    "nmse": nmse_numpy(latent_array, trigger_latent_array),
                    "power_change_db": power_change,
                    "reference_iq_mean": iq_mean_numpy(latent_array),
                    "trigger_iq_mean": iq_mean_numpy(trigger_latent_array),
                    "reference_papr": papr_numpy(latent_array),
                    "trigger_papr": papr_numpy(trigger_latent_array),
                    "evm_limit": 0.10,
                    "power_change_db_max_abs": 0.5,
                    "constraints_pass": evm <= 0.10 and abs(power_change) <= 0.5,
                }
            source = np.concatenate(source_parts)
            report = compute_four_way_report(
                np.concatenate(clean_clean_parts),
                np.concatenate(clean_trigger_parts),
                np.concatenate(backdoor_clean_parts),
                np.concatenate(backdoor_trigger_parts),
                source,
                target,
                asr_threshold_db=threshold,
                metadata={"snr_db": snr_db, "trigger": trigger_payload},
            )
            scores.append(report.to_dict())
        _dump(run_dir / "four_way.json", {"asr_threshold_db": threshold, "trigger_stealth": trigger_stealth, "scores": scores})
        controls = _evaluate_controls(
            clean_adapter,
            backdoor_adapter,
            test_loader,
            target,
            base_config,
            float((trigger_stealth or {}).get("evm_rms", 0.0)),
            device,
            args.evaluation_seed,
        )
        _dump(run_dir / "controls.json", controls)


def run_audit_baseline(args: argparse.Namespace, root: Path, device: Any) -> None:
    """Recompute one P0 checkpoint on validation and test with explicit labels."""

    torch = _torch()
    from torch.utils.data import DataLoader
    from rf_backdoor.training import evaluate_clean_epoch

    _, _, loaders, _, validation_indices = _make_loaders(
        root,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    test = CIFAR10Dataset(root / "data", train=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    checkpoint_path = root / "runs" / "P0" / f"seed_{args.seed}" / "best.pt"
    adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
    checkpoint = _load_checkpoint(adapter, checkpoint_path, device)
    adapter.model.eval()

    set_seed(args.evaluation_seed)
    validation = evaluate_clean_epoch(adapter, loaders[1], snr_db=7.0, device=device)
    set_seed(args.evaluation_seed)
    test_metrics = evaluate_clean_epoch(adapter, test_loader, snr_db=7.0, device=device)

    four_way_values: list[dict[str, Any]] = []
    for path in sorted((root / "runs" / P2_V2_ROOT).rglob("four_way.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for score in payload.get("scores", []):
            if float(score.get("metadata", {}).get("snr_db", float("nan"))) == 7.0:
                metrics = score.get("metrics", {})
                four_way_values.append({
                    "path": str(path),
                    "reconstruction_psnr_db": metrics.get("psnr", {}).get("clean_model_clean_channel"),
                    "target_psnr_db": metrics.get("target_psnr", {}).get("clean_model_clean_channel"),
                })

    checkpoint_validation = checkpoint.get("metrics", {}).get("validation_psnr")
    delta_db = float(test_metrics["psnr"] - validation["psnr"])
    verdict = "PASS" if abs(delta_db) <= args.baseline_delta_tolerance_db else "FAIL"
    _dump(root / "runs" / "baseline_audit.json", {
        "verdict": verdict,
        "tolerance_db": args.baseline_delta_tolerance_db,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "stored_validation_psnr_db": checkpoint_validation,
        },
        "evaluation": {
            "seed": args.evaluation_seed,
            "snr_db": 7.0,
            "validation_count": len(validation_indices),
            "test_count": len(test),
            "validation": validation,
            "test": test_metrics,
            "test_minus_validation_psnr_db": delta_db,
        },
        "four_way_field_check": four_way_values,
        "interpretation": "Compare reconstruction_psnr_db with P0 PSNR; target_psnr_db measures similarity to the fixed attack target and is not a clean reconstruction baseline.",
    })


def run_analyze(root: Path, *, include_legacy: bool = False) -> None:
    """Aggregate four-way metrics across training seeds without pooling images."""

    groups: dict[tuple[Any, ...], list[dict[str, float]]] = {}
    control_groups: dict[tuple[Any, ...], list[dict[str, float]]] = {}
    skipped_legacy = 0
    for path in sorted((root / "runs" / P2_V2_ROOT).rglob("four_way.json")):
        run_dir = path.parent
        trigger = json.loads((run_dir / "trigger.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if trigger.get("objective_version") != "per_sample_v2" and not include_legacy:
            skipped_legacy += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        stealth = payload.get("trigger_stealth") or {}
        for score in payload.get("scores", []):
            snr_db = float(score["metadata"]["snr_db"])
            key = (
                trigger["kind"],
                float(metadata["poison_rate"]),
                float(trigger.get("evm_penalty_weight", 0.0)),
                float(trigger.get("power_penalty_weight", 0.0)),
                snr_db,
            )
            metrics = score["metrics"]
            groups.setdefault(key, []).append({
                "seed": float(metadata["seed"]),
                "asr": float(metrics["asr"]["backdoored_model_trigger_channel"]),
                "clean_false_activation_rate": float(metrics["asr"]["clean_model_trigger_channel"]),
                "backdoor_lift": float(metrics["asr"]["backdoor_lift"]),
                "clean_psnr_drop_db": float(-metrics["psnr"]["clean_utility_delta"]),
                "target_psnr_db": float(metrics["target_psnr"]["backdoored_model_trigger_channel"]),
                "evm_rms": float(stealth.get("evm_rms", float("nan"))),
                "power_change_db": float(stealth.get("power_change_db", float("nan"))),
            })
        controls_path = run_dir / "controls.json"
        if controls_path.exists():
            controls_payload = json.loads(controls_path.read_text(encoding="utf-8"))
            for control_name, control in controls_payload.get("controls", {}).items():
                if float(control.get("snr_db", float("nan"))) != 7.0:
                    continue
                control_key = (
                    trigger["kind"],
                    float(metadata["poison_rate"]),
                    float(trigger.get("evm_penalty_weight", 0.0)),
                    float(trigger.get("power_penalty_weight", 0.0)),
                    control_name,
                )
                control_groups.setdefault(control_key, []).append({
                    "seed": float(metadata["seed"]),
                    "clean_model_asr": float(control["clean_model_asr"]),
                    "backdoored_model_asr": float(control["backdoored_model_asr"]),
                    "evm_rms": float(control["evm_rms"]),
                    "power_change_db": float(control["power_change_db"]),
                })

    summaries = []
    for key, rows in sorted(groups.items()):
        trigger_kind, poison_rate, evm_weight, power_weight, snr_db = key
        seed_values = [int(row["seed"]) for row in rows]
        if len(seed_values) != len(set(seed_values)):
            raise ValueError(f"duplicate training seed in result group {key}")
        summary: dict[str, Any] = {
            "trigger_kind": trigger_kind,
            "poison_rate": poison_rate,
            "evm_penalty_weight": evm_weight,
            "power_penalty_weight": power_weight,
            "snr_db": snr_db,
            "seeds": sorted(seed_values),
        }
        for metric in ("asr", "clean_false_activation_rate", "backdoor_lift", "clean_psnr_drop_db", "target_psnr_db"):
            summary[metric] = mean_std_t_ci(np.asarray([row[metric] for row in rows]))
        finite_evm = [row["evm_rms"] for row in rows if np.isfinite(row["evm_rms"])]
        finite_power = [row["power_change_db"] for row in rows if np.isfinite(row["power_change_db"])]
        summary["evm_rms"] = mean_std_t_ci(np.asarray(finite_evm)) if finite_evm else None
        summary["power_change_db"] = mean_std_t_ci(np.asarray(finite_power)) if finite_power else None
        summaries.append(summary)

    control_summaries = []
    for key, rows in sorted(control_groups.items()):
        trigger_kind, poison_rate, evm_weight, power_weight, control_name = key
        seeds = [int(row["seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate training seed in control result group {key}")
        control_summaries.append({
            "trigger_kind": trigger_kind,
            "poison_rate": poison_rate,
            "evm_penalty_weight": evm_weight,
            "power_penalty_weight": power_weight,
            "control": control_name,
            "seeds": sorted(seeds),
            "clean_model_asr": mean_std_t_ci(np.asarray([row["clean_model_asr"] for row in rows])),
            "backdoored_model_asr": mean_std_t_ci(np.asarray([row["backdoored_model_asr"] for row in rows])),
            "evm_rms": mean_std_t_ci(np.asarray([row["evm_rms"] for row in rows])),
            "power_change_db": mean_std_t_ci(np.asarray([row["power_change_db"] for row in rows])),
        })

    _dump(root / "runs" / P2_V2_ROOT / "summary.json", {
        "objective_version": "per_sample_v2" if not include_legacy else "mixed_including_legacy",
        "ci_unit": "training_seed",
        "groups": summaries,
        "controls": control_summaries,
        "skipped_legacy_runs": skipped_legacy,
        "limitations": [
            "A confidence interval is not available for groups with one seed.",
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["p0", "p1", "p2", "evaluate", "audit", "analyze", "all"], default="all")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260823)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early-stopping-start", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scan-samples", type=int, default=4999)
    parser.add_argument("--test-snr-db", type=float, nargs="+", default=[1, 4, 7, 10, 13, 19, 25])
    parser.add_argument("--poison-rates", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--trigger-types", nargs="+", choices=["fixed", "learnable"], default=["fixed", "learnable"])
    parser.add_argument("--evm-penalty-weights", type=float, nargs="+", default=[100.0])
    parser.add_argument("--power-penalty-weights", type=float, nargs="+", default=[10.0])
    parser.add_argument("--evaluation-seed", type=int, default=20260823)
    parser.add_argument("--baseline-delta-tolerance-db", type=float, default=1.0)
    parser.add_argument("--include-legacy-runs", action="store_true")
    parser.add_argument("--allow-single-impairment-selection", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "analyze":
        run_analyze(root, include_legacy=args.include_legacy_runs)
        return 0
    device = _torch().device(args.device)
    clean_checkpoint = root / "runs" / "P0" / f"seed_{args.seed}" / "best.pt"
    if args.phase in ("p0", "all"):
        clean_checkpoint = run_p0(args, root, device)
        if args.phase == "p0":
            return 0
    if args.phase in ("p1", "all"):
        trigger = run_p1(args, root, device, clean_checkpoint)
    else:
        selected_payload = json.loads((root / "runs" / P1_V2_ROOT / "selected_trigger.json").read_text(encoding="utf-8"))
        trigger = IQDCOffsetConfig(**selected_payload["config"])
    if args.phase in ("p2", "all"):
        run_p2(args, root, device, trigger)
    if args.phase in ("evaluate", "all"):
        selected_payload = json.loads((root / "runs" / P1_V2_ROOT / "selected_trigger.json").read_text(encoding="utf-8"))
        run_evaluate(args, root, device, IQDCOffsetConfig(**selected_payload["config"]))
    if args.phase in ("audit", "all"):
        run_audit_baseline(args, root, device)
    if args.phase == "all":
        run_analyze(root, include_legacy=args.include_legacy_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
