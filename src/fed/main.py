#!/usr/bin/env python3
"""Run benign FedAvg rounds and measure the survival of an existing backdoor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from jscc.channels import Channel, ChannelConfig  # noqa: E402
from jscc.datasets import load_div2k_splits  # noqa: E402
from jscc.handler import TriggerTrainer, set_seed  # noqa: E402
from jscc.model import DeepJSCC  # noqa: E402
from jscc.trigger import IQDCOffsetConfig  # noqa: E402
from .fedavg import FederatedPersistenceExperiment  # noqa: E402


def _path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _make_trigger(config: dict) -> IQDCOffsetConfig:
    return IQDCOffsetConfig(
        gain_db=float(config["gain_db"]),
        phase_deg=float(config["phase_deg"]),
        dc_magnitude=float(config["dc_magnitude"]),
        dc_phase_deg=float(config["dc_phase_deg"]),
        order=str(config["order"]),
    )


def _same_trigger(saved: dict, configured: IQDCOffsetConfig) -> bool:
    return all(
        saved.get(key) == value
        for key, value in vars(configured).items()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/div2k_two_drop_struct_lmdb_stronger_20260928.toml"),
    )
    parser.add_argument(
        "--backdoor-checkpoint", type=Path,
        default=Path("runs/three_stage_new/triggered/best.pt"),
    )
    parser.add_argument(
        "--clean-checkpoint", type=Path,
        default=Path("runs/three_stage_new/clean/stage2_best.pt"),
    )
    parser.add_argument("--train-lmdb", type=Path)
    parser.add_argument("--valid-lmdb", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/three_stage_new/federated"))
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--client-fraction", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = _path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this experiment in the sionna GPU environment")
    if args.clients < 1 or args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("clients, rounds, and local-epochs must be positive")
    if args.batch_size < 1 or args.plot_every < 1:
        raise ValueError("batch-size and plot-every must be positive")

    config_path = _path(args.config)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    dataset_config = raw["dataset"]
    if dataset_config.get("format", "lmdb") != "lmdb":
        raise ValueError("the federated experiment currently requires the DIV2K LMDB dataset")
    train_path = _path(args.train_lmdb or Path(dataset_config["train_lmdb"]))
    valid_path = _path(args.valid_lmdb or Path(dataset_config["valid_lmdb"]))
    train, validation, test = load_div2k_splits(train_path, valid_path)
    batch_size = args.batch_size
    pin_memory = args.device.startswith("cuda")
    validation_loader = DataLoader(
        validation, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory,
    )

    backdoor_path = _path(args.backdoor_checkpoint)
    clean_path = _path(args.clean_checkpoint)
    if not backdoor_path.is_file() or not clean_path.is_file():
        raise FileNotFoundError(f"required checkpoint is missing: {backdoor_path} or {clean_path}")
    backdoor_payload = torch.load(backdoor_path, map_location="cpu", weights_only=True)
    clean_payload = torch.load(clean_path, map_location="cpu", weights_only=True)
    if clean_payload.get("stage") != "clean_stage2":
        raise ValueError("--clean-checkpoint must be a clean_stage2 checkpoint")
    trigger = _make_trigger(raw["trigger"])
    if not _same_trigger(backdoor_payload.get("trigger", {}), trigger):
        raise ValueError("backdoor checkpoint trigger does not match the selected experiment config")

    backend = raw["backend"]
    device = torch.device(args.device)
    set_seed(args.seed)
    model = DeepJSCC(c=int(backend["c"]))
    model.load_state_dict(backdoor_payload["model"], strict=True)
    model.to(device)
    reference = DeepJSCC(c=int(backend["c"]))
    reference.load_state_dict(clean_payload["model"], strict=True)
    reference.to(device).eval()
    channel = Channel(ChannelConfig(
        channel_type=str(backend["channel_type"]),
        snr_min=0.0,
        snr_max=20.0,
    )).to(device)
    trainer = TriggerTrainer(model, reference, channel, trigger, raw=raw, device=device)
    experiment = FederatedPersistenceExperiment(
        model=model,
        reference=reference,
        trainer=trainer,
        train_dataset=train,
        validation_loader=validation_loader,
        test_loader=test_loader,
        device=device,
        output_dir=output,
        clients=args.clients,
        client_fraction=args.client_fraction,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        mixed_precision=not args.no_mixed_precision,
        plot_every=args.plot_every,
        show_progress=not args.no_progress,
    )
    summary = experiment.run()
    print(
        f"federated_rounds={summary['rounds_completed']} "
        f"first_round_outside_rule={summary['first_validation_round_outside_rule']} "
        f"final_snr10_triggered_drop="
        f"{summary['final_validation']['10']['triggered_drop_mean_db']:.3f}dB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
