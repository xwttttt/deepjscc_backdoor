"""Run the CIFAR-10 P0--P2 study on the dedicated server checkout.

All paths default below ``/home/gxhen/Wisig/jscc/deepjscc_backdoor`` when the
script is launched from that directory. The upstream checkout is kept under
``external/`` and is never edited.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from rf_backdoor.adapter import ChunbaobaoDeepJSCCAdapter
from rf_backdoor.datasets import CIFAR10Dataset
from rf_backdoor.evaluation import compute_four_way_report, freeze_asr_threshold, target_psnr_np
from rf_backdoor.impairments import IQDCOffsetConfig, apply_iq_dc_torch, evm_rms_numpy, power_change_db_numpy
from rf_backdoor.metadata import collect_run_metadata, write_run_metadata
from rf_backdoor.poisoning import DeterministicPoisonedDataset, build_poison_manifest, build_split_indices, sha256_array
from rf_backdoor.scan import CandidateMeasurement, iter_fixed_trigger_configs, select_trigger_candidate, trigger_config_hash
from rf_backdoor.training import TrainingConfig, fit, set_seed


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
    seen = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            latent = adapter.encode(images)
            triggered = apply_iq_dc_torch(latent, config)
            clean_output = adapter.decode(adapter.apply_channel(latent, 7.0))
            triggered_output = adapter.decode(adapter.apply_channel(triggered, 7.0))
            reference_parts.append(latent.detach().cpu().numpy())
            impaired_parts.append(triggered.detach().cpu().numpy())
            clean_parts.append(clean_output.detach().cpu().numpy())
            triggered_parts.append(triggered_output.detach().cpu().numpy())
            seen += images.shape[0]
            if max_samples and seen >= max_samples:
                break
    reference = np.concatenate(reference_parts)[:max_samples]
    impaired = np.concatenate(impaired_parts)[:max_samples]
    clean = np.concatenate(clean_parts)[:max_samples]
    triggered = np.concatenate(triggered_parts)[:max_samples]
    target_np = target.detach().cpu().numpy()
    trigger_target_psnr = target_psnr_np(triggered, target_np)
    return CandidateMeasurement(
        config=config,
        evm_rms=evm_rms_numpy(reference, impaired),
        power_change_db=power_change_db_numpy(reference, impaired),
        clean_psnr_loss_db=float(np.mean(target_psnr_np(clean, clean)) - np.mean(target_psnr_np(triggered, clean))),
        clean_false_activation_rate=float(np.mean(trigger_target_psnr >= 25.0)),
    )


def run_p1(args: argparse.Namespace, root: Path, device: Any, clean_checkpoint: Path) -> IQDCOffsetConfig:
    torch = _torch()
    base, target, loaders, _, _ = _make_loaders(root, split_seed=args.split_seed, batch_size=args.batch_size, workers=args.workers)
    adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
    _load_checkpoint(adapter, clean_checkpoint, device)
    measurements = [_measure_candidate(adapter, loaders[1], target, config, device, max_samples=args.scan_samples) for config in iter_fixed_trigger_configs()]
    selected = select_trigger_candidate(measurements)
    phase_dir = root / "runs" / "P1"
    _dump(phase_dir / "measurements.json", [{**asdict(item), "config": asdict(item.config), "trigger_hash": trigger_config_hash(item.config)} for item in measurements])
    _dump(phase_dir / "selected_trigger.json", {"config": asdict(selected.config), "trigger_hash": trigger_config_hash(selected.config), "evm_rms": selected.evm_rms, "power_change_db": selected.power_change_db, "clean_psnr_loss_db": selected.clean_psnr_loss_db, "clean_false_activation_rate": selected.clean_false_activation_rate})
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
                config = TrainingConfig(snr_db=7.0, poison_weight=1.0, max_epochs=args.epochs, early_stopping_start=args.early_stopping_start, patience=args.patience)
                run_dir = root / "runs" / "P2" / trigger_kind / f"poison_{poison_rate:g}" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                manifest.write_json(run_dir / "poison_manifest.json")
                history = fit(adapter, loaders[0], loaders[1], optimizer, config=config, device=device, trigger=trigger, target=target.to(device), checkpoint_path=run_dir / "best.pt")
                _dump(run_dir / "history.json", history)
                _dump(run_dir / "trigger.json", {"kind": trigger_kind, "config": asdict(trigger.config()) if trigger_kind == "learnable" else asdict(trigger), "manifest_sha256": manifest.manifest_sha256})
                metadata = collect_run_metadata(root, upstream_path=root / "external" / "chunbaobao-deepjscc", seed=seed)
                metadata.update({"phase": "P2", "trigger_kind": trigger_kind, "poison_rate": poison_rate, "manifest_sha256": manifest.manifest_sha256})
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
    threshold = json.loads((root / "runs" / "P1" / "asr_threshold.json").read_text(encoding="utf-8"))["threshold_db"]
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    clean_adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
    clean_checkpoint_path = root / "runs" / "P0" / f"seed_{args.seed}" / "best.pt"
    _load_checkpoint(clean_adapter, clean_checkpoint_path, device)
    clean_adapter.model.eval()
    run_paths = sorted((root / "runs" / "P2").glob("*/*/seed_*/best.pt"))
    for checkpoint_path in run_paths:
        run_dir = checkpoint_path.parent
        checkpoint = torch.load(checkpoint_path, map_location=device)
        backdoor_adapter = _make_adapter(root / "external" / "chunbaobao-deepjscc", device)
        backdoor_adapter.model.load_state_dict(checkpoint["model"])
        backdoor_adapter.model.eval()
        trigger = _trigger_for_run(run_dir, selected_config, checkpoint, device)
        scores = []
        for snr_db in args.test_snr_db:
            clean_clean_parts: list[np.ndarray] = []
            clean_trigger_parts: list[np.ndarray] = []
            backdoor_clean_parts: list[np.ndarray] = []
            backdoor_trigger_parts: list[np.ndarray] = []
            source_parts: list[np.ndarray] = []
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
                    clean_clean_parts.append(clean_adapter.decode(clean_adapter.apply_channel(clean_latent, snr_db)).cpu().numpy())
                    clean_trigger_parts.append(clean_adapter.decode(clean_adapter.apply_channel(clean_trigger_latent, snr_db)).cpu().numpy())
                    backdoor_clean_parts.append(backdoor_adapter.decode(backdoor_adapter.apply_channel(backdoor_latent, snr_db)).cpu().numpy())
                    backdoor_trigger_parts.append(backdoor_adapter.decode(backdoor_adapter.apply_channel(backdoor_trigger_latent, snr_db)).cpu().numpy())
            source = np.concatenate(source_parts)
            report = compute_four_way_report(
                np.concatenate(clean_clean_parts),
                np.concatenate(clean_trigger_parts),
                np.concatenate(backdoor_clean_parts),
                np.concatenate(backdoor_trigger_parts),
                source,
                target,
                asr_threshold_db=threshold,
                metadata={"snr_db": snr_db, "trigger": json.loads((run_dir / "trigger.json").read_text(encoding="utf-8"))},
            )
            scores.append(report.to_dict())
        _dump(run_dir / "four_way.json", {"asr_threshold_db": threshold, "scores": scores})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["p0", "p1", "p2", "evaluate", "all"], default="all")
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
    parser.add_argument("--scan-samples", type=int, default=512)
    parser.add_argument("--test-snr-db", type=float, nargs="+", default=[1, 4, 7, 10, 13, 19, 25])
    parser.add_argument("--poison-rates", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--trigger-types", nargs="+", choices=["fixed", "learnable"], default=["fixed", "learnable"])
    args = parser.parse_args()
    root = args.root.resolve()
    device = _torch().device(args.device)
    clean_checkpoint = root / "runs" / "P0" / f"seed_{args.seed}" / "best.pt"
    if args.phase in ("p0", "all"):
        clean_checkpoint = run_p0(args, root, device)
        if args.phase == "p0":
            return 0
    if args.phase in ("p1", "all"):
        trigger = run_p1(args, root, device, clean_checkpoint)
    else:
        selected_payload = json.loads((root / "runs" / "P1" / "selected_trigger.json").read_text(encoding="utf-8"))
        trigger = IQDCOffsetConfig(**selected_payload["config"])
    if args.phase in ("p2", "all"):
        run_p2(args, root, device, trigger)
    if args.phase == "evaluate":
        selected_payload = json.loads((root / "runs" / "P1" / "selected_trigger.json").read_text(encoding="utf-8"))
        run_evaluate(args, root, device, IQDCOffsetConfig(**selected_payload["config"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
