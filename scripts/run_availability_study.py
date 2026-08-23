"""Optimize and evaluate a Magmaw-aligned IQ/DC availability trigger.

The clean DeepJSCC checkpoint is frozen. Only the four bounded impairment
parameters are optimized; no target image or poisoned reconstruction label is
used by this study. The image attack objective maximizes receive-side MSE as
in Magmaw Equation (9), adapted to a bounded parametric impairment.
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
from rf_backdoor.availability import AvailabilityConfig, evaluate_availability_trigger, fit_availability_trigger, freeze_model
from rf_backdoor.datasets import CIFAR10Dataset
from rf_backdoor.evaluation import mean_std_t_ci
from rf_backdoor.impairments import IQDCOffsetConfig, LearnableIQDCOffset
from rf_backdoor.metadata import collect_run_metadata, sha256_file, write_run_metadata
from rf_backdoor.poisoning import build_split_indices
from rf_backdoor.training import set_seed


OUTPUT_ROOT = "availability_v1"


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


def _make_adapter(root: Path, device: Any, checkpoint_path: Path) -> ChunbaobaoDeepJSCCAdapter:
    torch = _torch()
    adapter = ChunbaobaoDeepJSCCAdapter.from_repo(root / "external" / "chunbaobao-deepjscc", c=8, channel_type="AWGN", snr_db=7.0)
    adapter.model.to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    adapter.model.load_state_dict(payload["model"])
    freeze_model(adapter)
    return adapter


def _make_loaders(root: Path, args: argparse.Namespace) -> tuple[Any, Any]:
    from torch.utils.data import DataLoader, Subset

    base = CIFAR10Dataset(root / "data", train=True)
    train_indices, validation_indices = build_split_indices(len(base), target_index=0, train_count=45_000, seed=args.split_seed)
    train = DataLoader(Subset(base, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    validation = DataLoader(Subset(base, validation_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    return train, validation


def run_train(args: argparse.Namespace, root: Path, device: Any, initial: IQDCOffsetConfig) -> None:
    torch = _torch()
    train_loader, validation_loader = _make_loaders(root, args)
    clean_checkpoint = root / "runs" / "P0" / f"seed_{args.clean_model_seed}" / "best.pt"
    for evm_weight, power_weight in product(args.evm_penalty_weights, args.power_penalty_weights):
        for seed in args.seeds:
            set_seed(seed)
            adapter = _make_adapter(root, device, clean_checkpoint)
            trigger = LearnableIQDCOffset(initial).to(device)
            optimizer = torch.optim.Adam(trigger.parameters(), lr=args.trigger_learning_rate)
            config = AvailabilityConfig(
                train_snr_db=tuple(args.train_snr_db),
                min_psnr_drop_db=args.failure_drop_db,
                evm_limit=args.evm_limit,
                power_change_db_limit=args.power_change_db_limit,
                evm_penalty_weight=evm_weight,
                power_penalty_weight=power_weight,
                max_epochs=args.epochs,
                patience=args.patience,
                max_train_batches=args.max_train_batches,
                max_validation_batches=args.max_validation_batches,
            )
            run_dir = root / "runs" / OUTPUT_ROOT / f"penalty_evm_{evm_weight:g}_power_{power_weight:g}" / f"seed_{seed}"
            checkpoint_path = run_dir / "best_trigger.pt"
            history = fit_availability_trigger(
                adapter,
                train_loader,
                validation_loader,
                optimizer,
                trigger=trigger,
                config=config,
                device=device,
                checkpoint_path=checkpoint_path,
            )
            _dump(run_dir / "history.json", history)
            _dump(run_dir / "trigger.json", {
                "objective": "magmaw_image_mse_availability_v1",
                "objective_formula": "minimize(-mean_attacked_mse + evm_penalty + power_penalty)",
                "model_frozen": True,
                "seed": seed,
                "initial_config": asdict(initial),
                "optimized_config": asdict(trigger.config()),
                "train_snr_db": list(config.train_snr_db),
                "failure_drop_db": config.min_psnr_drop_db,
                "evm_limit": config.evm_limit,
                "power_change_db_limit": config.power_change_db_limit,
                "evm_penalty_weight": evm_weight,
                "power_penalty_weight": power_weight,
            })
            metadata = collect_run_metadata(root, upstream_path=root / "external" / "chunbaobao-deepjscc", seed=seed)
            metadata.update({
                "phase": "availability_trigger_optimization",
                "objective": "magmaw_image_mse_availability_v1",
                "model_frozen": True,
                "clean_checkpoint": str(clean_checkpoint),
                "clean_checkpoint_sha256": sha256_file(clean_checkpoint),
                "evm_penalty_weight": evm_weight,
                "power_penalty_weight": power_weight,
            })
            write_run_metadata(run_dir / "run_metadata.json", metadata, artifacts=[checkpoint_path, run_dir / "history.json", run_dir / "trigger.json"])


def run_evaluate(args: argparse.Namespace, root: Path, device: Any, initial: IQDCOffsetConfig) -> None:
    torch = _torch()
    from torch.utils.data import DataLoader

    test = CIFAR10Dataset(root / "data", train=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    clean_checkpoint = root / "runs" / "P0" / f"seed_{args.clean_model_seed}" / "best.pt"
    for checkpoint_path in sorted((root / "runs" / OUTPUT_ROOT).rglob("best_trigger.pt")):
        run_dir = checkpoint_path.parent
        adapter = _make_adapter(root, device, clean_checkpoint)
        trigger = LearnableIQDCOffset(initial).to(device)
        payload = torch.load(checkpoint_path, map_location=device)
        trigger.load_state_dict(payload["trigger"])
        result = evaluate_availability_trigger(
            adapter,
            test_loader,
            trigger=trigger,
            snr_db_values=args.test_snr_db,
            failure_drop_db=args.failure_drop_db,
            device=device,
        )
        result.update({
            "objective": "magmaw_image_mse_availability_v1",
            "model_frozen": True,
            "optimized_config": asdict(trigger.config()),
            "constraints_pass": result["stealth"]["evm_rms"] <= args.evm_limit and abs(result["stealth"]["power_change_db"]) <= args.power_change_db_limit,
            "success": (
                result["robust"]["worst_snr_mean_psnr_drop_db"] >= args.failure_drop_db
                and result["robust"]["worst_snr_failure_rate"] >= args.failure_rate_min
                and result["stealth"]["evm_rms"] <= args.evm_limit
                and abs(result["stealth"]["power_change_db"]) <= args.power_change_db_limit
            ),
        })
        _dump(run_dir / "availability.json", result)


def run_analyze(root: Path) -> None:
    groups: dict[tuple[float, float, float], list[dict[str, float]]] = {}
    robust_groups: dict[tuple[float, float], list[dict[str, float]]] = {}
    for path in sorted((root / "runs" / OUTPUT_ROOT).rglob("availability.json")):
        run_dir = path.parent
        result = json.loads(path.read_text(encoding="utf-8"))
        trigger = json.loads((run_dir / "trigger.json").read_text(encoding="utf-8"))
        evm_weight = float(trigger["evm_penalty_weight"])
        power_weight = float(trigger["power_penalty_weight"])
        for score in result["scores"]:
            key = (evm_weight, power_weight, float(score["snr_db"]))
            groups.setdefault(key, []).append(score)
        robust_groups.setdefault((evm_weight, power_weight), []).append({
            **result["robust"],
            "evm_rms": float(result["stealth"]["evm_rms"]),
            "power_change_db": float(result["stealth"]["power_change_db"]),
            "success": float(bool(result["success"])),
        })
    summaries = []
    for (evm_weight, power_weight, snr_db), rows in sorted(groups.items()):
        summaries.append({
            "evm_penalty_weight": evm_weight,
            "power_penalty_weight": power_weight,
            "snr_db": snr_db,
            "mean_psnr_drop_db": mean_std_t_ci(np.asarray([row["mean_psnr_drop_db"] for row in rows])),
            "failure_rate": mean_std_t_ci(np.asarray([row["failure_rate"] for row in rows])),
            "triggered_psnr_db": mean_std_t_ci(np.asarray([row["triggered_psnr_db"] for row in rows])),
            "triggered_mse": mean_std_t_ci(np.asarray([row["triggered_mse"] for row in rows])),
            "mse_ratio": mean_std_t_ci(np.asarray([row["mse_ratio"] for row in rows])),
        })
    robust_summaries = []
    for (evm_weight, power_weight), rows in sorted(robust_groups.items()):
        robust_summaries.append({
            "evm_penalty_weight": evm_weight,
            "power_penalty_weight": power_weight,
            "worst_snr_mean_psnr_drop_db": mean_std_t_ci(np.asarray([row["worst_snr_mean_psnr_drop_db"] for row in rows])),
            "worst_snr_failure_rate": mean_std_t_ci(np.asarray([row["worst_snr_failure_rate"] for row in rows])),
            "evm_rms": mean_std_t_ci(np.asarray([row["evm_rms"] for row in rows])),
            "power_change_db": mean_std_t_ci(np.asarray([row["power_change_db"] for row in rows])),
            "success_rate_across_optimization_seeds": float(np.mean([row["success"] for row in rows])),
        })
    _dump(root / "runs" / OUTPUT_ROOT / "summary.json", {
        "objective": "magmaw_image_mse_availability_v1",
        "ci_unit": "trigger_optimization_seed",
        "snr_summaries": summaries,
        "robust_summaries": robust_summaries,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize a stable universal IQ/DC availability trigger")
    parser.add_argument("--phase", choices=["train", "evaluate", "analyze", "all"], default="all")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean-model-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260823)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trigger-learning-rate", type=float, default=1e-2)
    parser.add_argument("--train-snr-db", type=float, nargs="+", default=[1, 4, 7, 10, 13, 19, 25])
    parser.add_argument("--test-snr-db", type=float, nargs="+", default=[1, 4, 7, 10, 13, 19, 25])
    parser.add_argument("--failure-drop-db", type=float, default=3.0)
    parser.add_argument("--failure-rate-min", type=float, default=0.90)
    parser.add_argument("--evm-limit", type=float, default=0.10)
    parser.add_argument("--power-change-db-limit", type=float, default=0.5)
    parser.add_argument("--evm-penalty-weights", type=float, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--power-penalty-weights", type=float, nargs="+", default=[1, 10, 100])
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--initial-trigger-json", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "analyze":
        run_analyze(root)
        return 0
    device = _torch().device(args.device)
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
