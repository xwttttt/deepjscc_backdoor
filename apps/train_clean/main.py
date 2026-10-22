#!/usr/bin/env python3
"""Train a clean DeepJSCC model in fixed-SNR and variable-SNR stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from jscc.channels import Channel, ChannelConfig  # noqa: E402
from jscc.datasets import PreparedDIV2K, load_div2k_splits  # noqa: E402
from jscc.handler import CleanStageConfig, Trainer, set_seed  # noqa: E402
from jscc.metrics import Metrics  # noqa: E402
from jscc.model import DeepJSCC  # noqa: E402


def _path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-kind", choices=("lmdb", "legacy-npy"), default="lmdb")
    parser.add_argument("--train-lmdb", type=Path, default=Path("datasets/DIV2K/train_256_float32.lmdb"))
    parser.add_argument("--valid-lmdb", type=Path, default=Path("datasets/DIV2K/valid_256_float32.lmdb"))
    parser.add_argument("--prepared-dir", type=Path, default=Path("datasets/DIV2K/prepared_256"))
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate existing stage1_best.pt and stage2_best.pt without retraining")
    parser.add_argument("--stage1-epochs", type=int, default=200)
    parser.add_argument("--stage2-epochs", type=int, default=200)
    parser.add_argument("--stage1-patience", type=int, default=20)
    parser.add_argument("--stage2-patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--plot-every", type=int, default=5,
                        help="Save updated curves and validation previews every N epochs")
    parser.add_argument("--no-progress", action="store_true", help="Disable batch progress bars")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = _path(args.output_dir)
    if args.eval_only and not output.is_dir():
        raise FileNotFoundError(output)
    if args.eval_only and args.init_checkpoint is not None:
        raise ValueError("--init-checkpoint cannot be combined with --eval-only")
    if not args.eval_only and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on the designated training server")
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.dataset_kind == "lmdb":
        train_path, valid_path = _path(args.train_lmdb), _path(args.valid_lmdb)
        train, validation, test = load_div2k_splits(train_path, valid_path)
        dataset_paths = {"train_lmdb": str(train_path), "valid_lmdb": str(valid_path),
                         "validation_range": [0, 50], "test_range": [50, 100]}
    else:
        prepared = _path(args.prepared_dir)
        train = PreparedDIV2K(prepared / "train.npy")
        validation = PreparedDIV2K(prepared / "validation.npy")
        test = PreparedDIV2K(prepared / "test.npy")
        dataset_paths = {"prepared_dir": str(prepared)}
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    model = DeepJSCC(c=16)
    if args.eval_only:
        stage1_checkpoint = torch.load(output / "stage1_best.pt", map_location="cpu", weights_only=True)
        stage2_checkpoint = torch.load(output / "stage2_best.pt", map_location="cpu", weights_only=True)
        if stage1_checkpoint.get("stage") != "clean_stage1" or stage2_checkpoint.get("stage") != "clean_stage2":
            raise ValueError("output directory must contain clean_stage1 and clean_stage2 best checkpoints")
        model.load_state_dict(stage2_checkpoint["model"], strict=True)
    elif args.init_checkpoint is not None:
        initial = torch.load(_path(args.init_checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(initial["model"], strict=True)
    model.to(device)
    channel = Channel(ChannelConfig(channel_type="awgn", snr_min=0.0, snr_max=20.0))
    trainer = Trainer(model, channel, device=device)
    output.mkdir(parents=True, exist_ok=True)
    stage1 = CleanStageConfig(
        name="clean_stage1", epochs=args.stage1_epochs, train_snr_min=20, train_snr_max=20,
        validation_snrs=(20,), learning_rate=args.learning_rate,
        patience=args.stage1_patience, mixed_precision=args.mixed_precision,
    )
    stage2 = CleanStageConfig(
        name="clean_stage2", epochs=args.stage2_epochs, train_snr_min=0, train_snr_max=20,
        validation_snrs=(0, 5, 10, 15, 20), learning_rate=args.learning_rate,
        patience=args.stage2_patience, mixed_precision=args.mixed_precision,
        train_snr_seed=args.seed + 1,
    )
    if args.eval_only:
        stage1_best = {key: stage1_checkpoint[key] for key in ("epoch", "validation_psnr_db")}
        stage2_best = {key: stage2_checkpoint[key] for key in ("epoch", "validation_psnr_db")}
    else:
        stage1_history = trainer.fit(train_loader, validation_loader, stage=stage1,
                                     checkpoint_path=output / "stage1_best.pt",
                                     visualization_dir=output / "plots" / stage1.name,
                                     plot_every=args.plot_every, show_progress=not args.no_progress)
        stage1_best = {key: trainer.best_checkpoint[key] for key in ("epoch", "validation_psnr_db")}
        _write_json(output / "stage1_history.json", stage1_history)
        # Trainer.fit restores stage 1's best weights before this stage begins.
        stage2_history = trainer.fit(train_loader, validation_loader, stage=stage2,
                                     checkpoint_path=output / "stage2_best.pt",
                                     visualization_dir=output / "plots" / stage2.name,
                                     plot_every=args.plot_every, show_progress=not args.no_progress)
        stage2_best = {key: trainer.best_checkpoint[key] for key in ("epoch", "validation_psnr_db")}
        _write_json(output / "stage2_history.json", stage2_history)
    quality = Metrics(device, ms_ssim=True, lpips=True, input_is_minus_one_to_one=False)
    test_metrics = {
        str(snr): trainer.evaluate(test_loader, snr_db=snr, seed=args.seed + snr, metrics=quality)
        for snr in (0, 10, 20)
    }
    _write_json(output / "metrics.json", {
        "protocol": "two_stage_clean_div2k", "dataset_kind": args.dataset_kind,
        "evaluation_only": args.eval_only,
        "dataset_paths": dataset_paths, "train_count": len(train),
        "validation_count": len(validation), "test_count": len(test),
        "init_checkpoint": str(_path(args.init_checkpoint)) if args.init_checkpoint else None,
        "stage1": {"train_snr_db": 20, "max_epochs": None if args.eval_only else args.stage1_epochs,
                   "best": stage1_best},
        "stage2": {"train_snr_db": "uniform integer 0..20 per step",
                   "max_epochs": None if args.eval_only else args.stage2_epochs,
                   "validation_snrs_db": list(stage2.validation_snrs), "best": stage2_best},
        "stage2_checkpoint": str(output / "stage2_best.pt"), "test_by_snr_db": test_metrics,
    })
    print(f"clean_stage2_best={output / 'stage2_best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
