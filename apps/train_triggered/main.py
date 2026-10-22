"""Train the IQ/DC backdoor with PSNR drops and evaluate image quality.

clean_drop = PSNR(P0 clean) - PSNR(backdoor clean)
triggered_drop = PSNR(backdoor clean) - PSNR(backdoor triggered)

The training target is PSNR-based clean and triggered drop. Evaluation also
reports MS-SSIM and LPIPS for clean and triggered reconstructions. Four branches
share one channel state and one AWGN realization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jscc.datasets import PreparedDIV2K, load_div2k_splits  # noqa: E402
from jscc.channels import Channel, ChannelConfig  # noqa: E402
from jscc.handler import TriggerTrainer, set_seed  # noqa: E402
from jscc.metrics import Metrics  # noqa: E402
from jscc.model import DeepJSCC  # noqa: E402
from jscc.trigger import IQDCOffsetConfig  # noqa: E402


class ImageOnlyDataset:
    """Return images, with a fixed poison flag when training."""

    def __init__(self, dataset, poison_indices=None):
        self.dataset = dataset
        self.poison_indices = None if poison_indices is None else frozenset(poison_indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        if self.poison_indices is None:
            return image
        return image, index in self.poison_indices


def _loader(dataset, *, batch_size: int, workers: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
        pin_memory=True,
        persistent_workers=bool(workers),
    )


def _load_model(device, c: int, payload: dict) -> DeepJSCC:
    model = DeepJSCC(c=c).to(device)
    model.load_state_dict(payload["model"])
    return model


def _make_trigger(cfg: dict) -> IQDCOffsetConfig:
    if cfg.get("kind", "iq_dc") != "iq_dc":
        raise ValueError("only the iq_dc trigger is supported")
    return IQDCOffsetConfig(
        gain_db=float(cfg["gain_db"]),
        phase_deg=float(cfg["phase_deg"]),
        dc_magnitude=float(cfg["dc_magnitude"]),
        dc_phase_deg=float(cfg["dc_phase_deg"]),
        order=str(cfg["order"]),
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml")
    )
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plot-every", type=int, default=5,
                        help="Save updated curves and validation previews every N epochs")
    parser.add_argument("--no-progress", action="store_true", help="Disable batch progress bars")
    args = parser.parse_args()

    import tomllib

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    checkpoint_path = args.clean_checkpoint if args.clean_checkpoint.is_absolute() else ROOT / args.clean_checkpoint
    output = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    dataset_cfg = raw["dataset"]
    dataset_format = str(dataset_cfg.get("format", "legacy-npy"))
    if dataset_format == "lmdb":
        train_path = Path(dataset_cfg["train_lmdb"])
        valid_path = Path(dataset_cfg["valid_lmdb"])
        train_path = train_path if train_path.is_absolute() else ROOT / train_path
        valid_path = valid_path if valid_path.is_absolute() else ROOT / valid_path
        train_split, validation_split, test_split = load_div2k_splits(train_path, valid_path)
        dataset_identity = {
            "format": "lmdb",
            "train_lmdb": str(train_path.resolve()),
            "valid_lmdb": str(valid_path.resolve()),
            "validation_range": [0, 50],
            "test_range": [50, 100],
        }
    elif dataset_format == "legacy-npy":
        prepared = ROOT / dataset_cfg["root"]
        for split in ("train", "validation", "test"):
            if not (prepared / f"{split}.npy").is_file():
                raise FileNotFoundError(prepared / f"{split}.npy")
        train_split = PreparedDIV2K(prepared / "train.npy")
        validation_split = PreparedDIV2K(prepared / "validation.npy")
        test_split = PreparedDIV2K(prepared / "test.npy")
        dataset_identity = {"format": "legacy-npy", "prepared": str(prepared)}
    else:
        raise ValueError(f"unsupported dataset format: {dataset_format}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    clean_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if clean_payload.get("stage") != "clean_stage2":
        raise ValueError("--clean-checkpoint must be the best clean_stage2 checkpoint")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_cfg = raw["training"]
    set_seed(int(raw["poison"]["seed"]))
    validation_data = ImageOnlyDataset(validation_split)
    test_data = ImageOnlyDataset(test_split)
    poison_rate = float(raw["poison"]["rate"])
    if not 0.0 < poison_rate <= 1.0:
        raise ValueError("poison_rate must lie in (0, 1]")
    poison_seed = int(raw["poison"]["seed"])
    eligible = tuple(range(len(train_split)))
    count = int(round(len(eligible) * poison_rate))
    if count == 0:
        raise ValueError("poison_rate selects no training samples")
    rng = np.random.default_rng(poison_seed)
    poison_indices = (
        tuple(sorted(int(index) for index in rng.choice(eligible, size=count, replace=False)))
        if count else ()
    )
    train_data = ImageOnlyDataset(train_split, poison_indices)
    train_loader = _loader(
        train_data,
        batch_size=int(train_cfg["batch_size"]),
        workers=0,
        shuffle=True,
    )
    validation_loader = _loader(
        validation_data,
        batch_size=int(train_cfg["batch_size"]),
        workers=0,
        shuffle=False,
    )
    test_loader = _loader(
        test_data,
        batch_size=int(train_cfg["batch_size"]),
        workers=0,
        shuffle=False,
    )
    backend = raw["backend"]
    backdoor = _load_model(device, int(backend["c"]), clean_payload)
    reference = _load_model(device, int(backend["c"]), clean_payload)
    snr_db = float(backend["train_snr_db"])
    channel = Channel(ChannelConfig(
        channel_type=str(backend["channel_type"]), snr_min=snr_db, snr_max=snr_db
    ))
    trigger = _make_trigger(raw["trigger"])
    trainer = TriggerTrainer(backdoor, reference, channel, trigger, raw=raw, device=device)
    history = trainer.fit(
        train_loader,
        validation_loader,
        checkpoint_path=output / "best.pt",
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        visualization_dir=output / "plots" / "trigger",
        plot_every=args.plot_every,
        show_progress=not args.no_progress,
    )
    chosen = trainer.best_checkpoint
    eval_metrics = Metrics(device, ms_ssim=True, lpips=True, input_is_minus_one_to_one=False)
    test = trainer.evaluate(test_loader, seed=int(train_cfg["test_channel_seed"]), metrics=eval_metrics)
    test_by_snr = {
        str(snr): trainer.evaluate(
            test_loader, seed=int(train_cfg["test_channel_seed"]),
            snr_db=snr, metrics=eval_metrics,
        ) for snr in (0, 10, 20)
    }
    _write_json(output / "history.json", history)
    _write_json(output / "poison_manifest.json", {
        "dataset_name": "DIV2K",
        "dataset_size": len(train_data),
        "eligible_indices": list(eligible),
        "poison_indices": list(poison_indices),
        "requested_rate": poison_rate,
        "actual_rate": len(poison_indices) / len(eligible) if eligible else 0.0,
        "seed": poison_seed,
        "split_seed": int(raw["dataset"]["split_seed"]),
        "version": 1,
    })
    _write_json(
        output / "trigger.json",
        {
            "trigger_kind": "iq_dc",
            "config": vars(trigger),
            "labels_used": False,
            "objective": "clean_drop_and_triggered_drop_only",
        },
    )
    _write_json(
        output / "metrics.json",
        {
            "protocol": "div2k_two_drop_psnr_with_ms_ssim_lpips_evaluation",
            "objective": {"clean_drop_weight": trainer.objective["clean_drop_weight"]},
            "mechanism": trainer.mechanism,
            "attack_metric": trainer.attack_key,
            "clean_drop_limit_db": float(train_cfg["clean_drop_limit_db"]),
            "triggered_drop_target_db": float(train_cfg["triggered_drop_target_db"]),
            "train_snr_db": trainer.train_snr_db,
            "selection_snr_db": trainer.selection_snr_db,
            "best_epoch": int(chosen["epoch"]),
            "clean_constraint_satisfied": bool(chosen["clean_constraint_satisfied"]),
            "validation": chosen["validation"],
            "test": test,
            "test_by_snr_db": test_by_snr,
            "clean_checkpoint": str(checkpoint_path),
            "dataset_identity": dataset_identity,
        },
    )
    print(
        f"best_epoch={int(chosen['epoch'])} test_clean_drop={test['clean_drop_mean_db']:.3f}dB "
        f"test_psnr_drop={test['triggered_drop_mean_db']:.3f}dB",
        flush=True,
    )
