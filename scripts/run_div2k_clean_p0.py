#!/usr/bin/env python3
"""Fine-tune the clean c=16 DeepJSCC model on static DIV2K crops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from div2k_prepared_dataset import PreparedDIV2K  # noqa: E402
from div2k_lmdb_dataset import load_div2k_splits  # noqa: E402
from rf_backdoor.adapter import DeepJSCCAdapter  # noqa: E402
from rf_backdoor.training import P0StepConfig, evaluate_clean_epoch, fit_clean_steps, set_seed  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-kind", choices=("lmdb", "legacy-npy"), default="lmdb")
    parser.add_argument("--train-lmdb", type=Path, default=Path("data/DIV2K/lmdb/train_256_CV16_float32.lmdb"))
    parser.add_argument("--valid-lmdb", type=Path, default=Path("data/DIV2K/lmdb/valid_256_CV16_float32.lmdb"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/DIV2K/prepared_256_seed_20260923"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("runs/clean_p0_c16_server_20260827/best.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if min(args.steps, args.batch_size, args.validation_interval) < 1:
        raise ValueError("steps, batch-size and validation-interval must be positive")
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable on the designated training server")
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.dataset_kind == "lmdb":
        train, validation, test = load_div2k_splits(args.train_lmdb, args.valid_lmdb)
    else:
        train = PreparedDIV2K(args.prepared_dir / "train.npy")
        validation = PreparedDIV2K(args.prepared_dir / "validation.npy")
        test = PreparedDIV2K(args.prepared_dir / "test.npy")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    adapter = DeepJSCCAdapter.from_local(c=16, channel_type="awgn", snr_db=args.snr_db)
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    adapter.model.load_state_dict(initial["model"], strict=True)
    adapter.model.to(device)
    set_seed(args.seed + 1)
    before = evaluate_clean_epoch(adapter, validation_loader, snr_db=args.snr_db, device=device)
    print(f"initial_validation_psnr_db={before['psnr']:.4f}", flush=True)

    args.output_dir.mkdir(parents=True)
    config = P0StepConfig(
        snr_db=args.snr_db,
        learning_rate=args.learning_rate,
        decay_learning_rate=args.learning_rate,
        weight_decay=0.0,
        decay_step=args.steps + 1,
        max_steps=args.steps,
        validation_interval=args.validation_interval,
        early_stopping_start=args.steps + 1,
        early_stopping_patience=100,
        early_stopping_min_delta_db=0.0,
        mixed_precision=args.mixed_precision,
    )
    checkpoint = args.output_dir / "best.pt"
    history = fit_clean_steps(
        adapter,
        train_loader,
        validation_loader,
        config=config,
        device=device,
        checkpoint_path=checkpoint,
    )
    set_seed(args.seed + 2)
    validation_after = evaluate_clean_epoch(adapter, validation_loader, snr_db=args.snr_db, device=device)
    set_seed(args.seed + 3)
    test_after = evaluate_clean_epoch(adapter, test_loader, snr_db=args.snr_db, device=device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = {
        "protocol": "div2k_cosmos_lmdb_clean_p0_finetune" if args.dataset_kind == "lmdb" else "div2k_prepared_256_clean_p0_finetune",
        "dataset_kind": args.dataset_kind,
        "dataset_paths": (
            {"train_lmdb": str(args.train_lmdb), "valid_lmdb": str(args.valid_lmdb), "valid_range": [0, 50], "test_range": [50, 100]}
            if args.dataset_kind == "lmdb" else {"prepared_dir": str(args.prepared_dir)}
        ),
        "prepared_manifest_sha256": sha256(args.prepared_dir / "manifest.json") if args.dataset_kind == "legacy-npy" else None,
        "train_lmdb_data_sha256": sha256(args.train_lmdb / "data.mdb") if args.dataset_kind == "lmdb" else None,
        "valid_lmdb_data_sha256": sha256(args.valid_lmdb / "data.mdb") if args.dataset_kind == "lmdb" else None,
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "init_checkpoint": str(args.init_checkpoint),
        "init_checkpoint_sha256": sha256(args.init_checkpoint),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "best_step": int(payload["step"]),
        "steps_requested": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "snr_db": args.snr_db,
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
        "initial_validation": before,
        "final_validation": validation_after,
        "test": test_after,
        "history": history,
    }
    with (args.output_dir / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        f"best_step={result['best_step']} validation_psnr={validation_after['psnr']:.3f} "
        f"test_psnr={test_after['psnr']:.3f} checkpoint={checkpoint}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
