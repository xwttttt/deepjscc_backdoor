"""Run the true IQ/DC availability-backdoor study on the dedicated server.

The only trigger path is ``encoder -> IQ imbalance -> DC offset -> AWGN``.
The model is initialized from the clean P0 checkpoint and fine-tuned with a
deterministic poison manifest.  There is no target image and no pixel-domain
modification.  ``run_availability_study.py`` remains the frozen-model,
trigger-only control.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from typing import Any

from rf_backdoor.adapter import ChunbaobaoDeepJSCCAdapter
from rf_backdoor.availability_backdoor import (
    AvailabilityBackdoorConfig,
    AvailabilityPoisonManifest,
    build_availability_poison_manifest,
    evaluate_availability_backdoor,
    fit_availability_backdoor,
    summarize_availability_runs,
)
from rf_backdoor.datasets import CIFAR10Dataset
from rf_backdoor.impairments import IQDCOffsetConfig, LearnableIQDCOffset
from rf_backdoor.metadata import collect_run_metadata, sha256_file, write_run_metadata
from rf_backdoor.poisoning import DeterministicPoisonedDataset, build_split_indices
from rf_backdoor.training import set_seed


OUTPUT_ROOT = "availability_backdoor_v1"


def _torch() -> Any:
    import torch

    return torch


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")


def _load_initial_trigger(root: Path, path: Path | None) -> IQDCOffsetConfig:
    source = path or root / "runs" / "P1_per_sample_v2" / "selected_trigger.json"
    if not source.exists():
        raise FileNotFoundError(f"initial trigger JSON not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return IQDCOffsetConfig(**payload["config"])


def _make_adapter(root: Path, device: Any, checkpoint_path: Path | None = None) -> ChunbaobaoDeepJSCCAdapter:
    torch = _torch()
    adapter = ChunbaobaoDeepJSCCAdapter.from_repo(root / "external" / "chunbaobao-deepjscc", c=8, channel_type="AWGN", snr_db=7.0)
    adapter.model.to(device)
    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location=device)
        adapter.model.load_state_dict(payload["model"])
    return adapter


def _make_loaders(root: Path, args: argparse.Namespace, manifest: AvailabilityPoisonManifest | None = None) -> tuple[Any, Any, Any, tuple[int, ...], tuple[int, ...]]:
    from torch.utils.data import DataLoader, Subset

    base = CIFAR10Dataset(root / "data", train=True)
    train_indices, validation_indices = build_split_indices(len(base), target_index=0, train_count=45_000, seed=args.split_seed)
    if manifest is None:
        train_dataset = Subset(base, train_indices)
    else:
        wrapped = DeterministicPoisonedDataset(base, manifest, return_metadata=True)
        train_dataset = Subset(wrapped, train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        Subset(base, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return base, train_loader, validation_loader, train_indices, validation_indices


def _run_dir(root: Path, trigger_kind: str, evm_weight: float, power_weight: float, poison_rate: float, seed: int) -> Path:
    path = root / "runs" / OUTPUT_ROOT / trigger_kind
    if trigger_kind == "learnable":
        path /= f"penalty_evm_{evm_weight:g}_power_{power_weight:g}"
    return path / f"poison_{poison_rate:g}" / f"seed_{seed}"


def run_train(args: argparse.Namespace, root: Path, device: Any, initial: IQDCOffsetConfig) -> None:
    torch = _torch()
    clean_checkpoint = root / "runs" / "P0" / f"seed_{args.clean_model_seed}" / "best.pt"
    if not clean_checkpoint.exists():
        raise FileNotFoundError(f"P0 checkpoint not found: {clean_checkpoint}")
    for trigger_kind in args.trigger_types:
        penalty_pairs = [(0.0, 0.0)] if trigger_kind == "fixed" else list(product(args.evm_penalty_weights, args.power_penalty_weights))
        for evm_weight, power_weight in penalty_pairs:
            for poison_rate in args.poison_rates:
                for seed in args.seeds:
                    set_seed(seed)
                    base, _, _, train_indices, _ = _make_loaders(root, args)
                    manifest = build_availability_poison_manifest(
                        "CIFAR10",
                        len(base),
                        poison_rate,
                        seed=seed,
                        split_seed=args.split_seed,
                        eligible_indices=train_indices,
                    )
                    _, train_loader, validation_loader, _, _ = _make_loaders(root, args, manifest=manifest)
                    adapter = _make_adapter(root, device, clean_checkpoint)
                    clean_reference = _make_adapter(root, device, clean_checkpoint)
                    trigger: IQDCOffsetConfig | LearnableIQDCOffset = initial
                    parameters = list(adapter.model.parameters())
                    if trigger_kind == "learnable":
                        trigger = LearnableIQDCOffset(initial).to(device)
                        parameters += list(trigger.parameters())
                    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
                    config = AvailabilityBackdoorConfig(
                        train_snr_db=tuple(args.train_snr_db),
                        min_psnr_drop_db=args.failure_drop_db,
                        clean_loss_weight=args.clean_loss_weight,
                        damage_loss_weight=args.damage_loss_weight,
                        evm_limit=args.evm_limit,
                        power_change_db_limit=args.power_change_db_limit,
                        evm_penalty_weight=evm_weight,
                        power_penalty_weight=power_weight,
                        max_clean_utility_drop_db=args.clean_utility_drop_max,
                        max_clean_trigger_failure_rate=args.clean_trigger_failure_max,
                        max_epochs=args.epochs,
                        early_stopping_start=args.early_stopping_start,
                        patience=args.patience,
                        mixed_precision=not args.no_amp,
                        max_train_batches=args.max_train_batches,
                        max_validation_batches=args.max_validation_batches,
                    )
                    run_dir = _run_dir(root, trigger_kind, evm_weight, power_weight, poison_rate, seed)
                    history = fit_availability_backdoor(
                        adapter,
                        clean_reference,
                        train_loader,
                        validation_loader,
                        optimizer,
                        trigger=trigger,
                        config=config,
                        device=device,
                        checkpoint_path=run_dir / "best.pt",
                        progress_label=str(run_dir),
                    )
                    manifest.write_json(run_dir / "poison_manifest.json")
                    _dump(run_dir / "history.json", history)
                    trigger_payload = {
                        "kind": trigger_kind,
                        "objective": "availability_backdoor_iq_dc_v1",
                        "trigger_path": "encoder_latent -> IQ_imbalance -> DC_offset -> AWGN",
                        "config": asdict(trigger.config()) if trigger_kind == "learnable" else asdict(trigger),
                        "manifest_sha256": manifest.manifest_sha256,
                        "poison_rate": poison_rate,
                        "evm_penalty_weight": evm_weight,
                        "power_penalty_weight": power_weight,
                        "failure_drop_db": args.failure_drop_db,
                        "mixed_precision": not args.no_amp,
                    }
                    _dump(run_dir / "trigger.json", trigger_payload)
                    metadata = collect_run_metadata(root, upstream_path=root / "external" / "chunbaobao-deepjscc", seed=seed)
                    metadata.update({
                        "phase": "availability_backdoor_training",
                        "objective": "availability_backdoor_iq_dc_v1",
                        "trigger_path": "IQ imbalance + DC offset only",
                        "trigger_kind": trigger_kind,
                        "poison_rate": poison_rate,
                        "manifest_sha256": manifest.manifest_sha256,
                        "clean_checkpoint": str(clean_checkpoint),
                        "clean_checkpoint_sha256": sha256_file(clean_checkpoint),
                        "evm_penalty_weight": evm_weight,
                        "power_penalty_weight": power_weight,
                        "mixed_precision": not args.no_amp,
                    })
                    write_run_metadata(run_dir / "run_metadata.json", metadata, artifacts=[run_dir / "best.pt", run_dir / "history.json", run_dir / "poison_manifest.json", run_dir / "trigger.json"])


def _load_trigger(trigger_payload: dict[str, Any], initial: IQDCOffsetConfig, checkpoint: dict[str, Any], device: Any) -> IQDCOffsetConfig | LearnableIQDCOffset:
    if trigger_payload["kind"] == "fixed":
        return IQDCOffsetConfig(**trigger_payload["config"])
    trigger = LearnableIQDCOffset(initial).to(device)
    if "trigger" not in checkpoint:
        raise RuntimeError("learnable trigger state missing from checkpoint")
    trigger.load_state_dict(checkpoint["trigger"])
    return trigger


def run_evaluate(args: argparse.Namespace, root: Path, device: Any, initial: IQDCOffsetConfig) -> None:
    torch = _torch()
    from torch.utils.data import DataLoader

    test = CIFAR10Dataset(root / "data", train=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    clean_checkpoint = root / "runs" / "P0" / f"seed_{args.clean_model_seed}" / "best.pt"
    for checkpoint_path in sorted((root / "runs" / OUTPUT_ROOT).rglob("best.pt")):
        run_dir = checkpoint_path.parent
        trigger_payload = json.loads((run_dir / "trigger.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        clean_adapter = _make_adapter(root, device, clean_checkpoint)
        backdoor_adapter = _make_adapter(root, device)
        backdoor_adapter.model.load_state_dict(checkpoint["model"])
        trigger = _load_trigger(trigger_payload, initial, checkpoint, device)
        result = evaluate_availability_backdoor(
            clean_adapter,
            backdoor_adapter,
            test_loader,
            trigger=trigger,
            snr_db_values=args.test_snr_db,
            failure_drop_db=args.failure_drop_db,
            device=device,
            mixed_precision=not args.no_amp,
        )
        result.update({
            "objective": "availability_backdoor_iq_dc_v1",
            "trigger_path": "encoder_latent -> IQ_imbalance -> DC_offset -> AWGN",
            "trigger": trigger_payload,
            "success": (
                result["robust"]["worst_snr_backdoor_mean_psnr_drop_db"] >= args.failure_drop_db
                and result["robust"]["worst_snr_backdoor_failure_rate"] >= args.failure_rate_min
                and result["robust"]["worst_snr_clean_model_trigger_failure_rate"] <= args.clean_trigger_failure_max
                and result["robust"]["worst_snr_backdoor_clean_utility_drop_db"] <= args.clean_utility_drop_max
                and result["stealth"]["evm_rms"] <= args.evm_limit
                and abs(result["stealth"]["power_change_db"]) <= args.power_change_db_limit
            ),
        })
        _dump(run_dir / "availability_four_way.json", result)


def run_analyze(root: Path) -> None:
    groups: dict[tuple[str, float, float, float], list[dict[str, Any]]] = {}
    for path in sorted((root / "runs" / OUTPUT_ROOT).rglob("availability_four_way.json")):
        run_dir = path.parent
        payload = json.loads(path.read_text(encoding="utf-8"))
        trigger = payload["trigger"]
        key = (
            trigger["kind"],
            float(trigger["poison_rate"]),
            float(trigger["evm_penalty_weight"]),
            float(trigger["power_penalty_weight"]),
        )
        groups.setdefault(key, []).append(payload)
    summaries = []
    for key, rows in sorted(groups.items()):
        trigger_kind, poison_rate, evm_weight, power_weight = key
        summaries.append({
            "trigger_kind": trigger_kind,
            "poison_rate": poison_rate,
            "evm_penalty_weight": evm_weight,
            "power_penalty_weight": power_weight,
            "runs": summarize_availability_runs(rows),
        })
    _dump(root / "runs" / OUTPUT_ROOT / "summary.json", {
        "objective": "availability_backdoor_iq_dc_v1",
        "trigger_path": "IQ imbalance + DC offset only",
        "ci_unit": "training_seed",
        "groups": summaries,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full IQ/DC availability-backdoor study")
    parser.add_argument("--phase", choices=["train", "evaluate", "analyze", "all"], default="all")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean-model-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260823)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--poison-rates", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--trigger-types", nargs="+", choices=["fixed", "learnable"], default=["fixed", "learnable"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--early-stopping-start", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--damage-loss-weight", type=float, default=1.0)
    parser.add_argument("--train-snr-db", type=float, nargs="+", default=[1, 4, 7, 10, 13, 19, 25])
    parser.add_argument("--test-snr-db", type=float, nargs="+", default=[1, 4, 7, 10, 13, 19, 25])
    parser.add_argument("--failure-drop-db", type=float, default=3.0)
    parser.add_argument("--failure-rate-min", type=float, default=0.90)
    parser.add_argument("--clean-trigger-failure-max", type=float, default=0.01)
    parser.add_argument("--clean-utility-drop-max", type=float, default=0.5)
    parser.add_argument("--evm-limit", type=float, default=0.10)
    parser.add_argument("--power-change-db-limit", type=float, default=0.5)
    parser.add_argument("--evm-penalty-weights", type=float, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--power-penalty-weights", type=float, nargs="+", default=[1, 10, 100])
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--initial-trigger-json", type=Path)
    parser.add_argument("--no-amp", action="store_true", help="disable CUDA mixed precision")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "analyze":
        run_analyze(root)
        return 0
    device = _torch().device(args.device)
    if device.type == "cuda" and not args.no_amp:
        _torch().backends.cudnn.benchmark = True
        _torch().backends.cuda.matmul.allow_tf32 = True
    initial = _load_initial_trigger(root, args.initial_trigger_json)
    if args.phase in ("train", "all"):
        run_train(args, root, device, initial)
    if args.phase in ("evaluate", "all"):
        run_evaluate(args, root, device, initial)
    if args.phase == "all":
        run_analyze(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
