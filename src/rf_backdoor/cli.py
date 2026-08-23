"""Configuration-driven entry points for P0--P2 study bookkeeping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from .evaluation import compute_four_way_report, freeze_asr_threshold
from .metadata import collect_run_metadata, write_run_metadata
from .poisoning import build_poison_manifest, build_split_indices, sha256_array
from .scan import CandidateMeasurement, iter_fixed_trigger_configs, select_trigger_candidate, trigger_config_hash


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_dir(args: argparse.Namespace) -> Path:
    path = Path(args.output).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def command_fetch_baseline(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "fetch_references.ps1"
    subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Destination", args.destination], check=True)
    return 0


def command_run_p0(args: argparse.Namespace) -> int:
    output = _run_dir(args)
    metadata = collect_run_metadata(Path.cwd(), upstream_path=args.baseline, seed=args.seed)
    metadata.update({"phase": "P0", "smoke": args.smoke, "bandwidth_ratio": 1 / 6, "train_snr_db": 7.0})
    write_run_metadata(output / "run_metadata.json", metadata)
    if args.smoke:
        try:
            import torch
            from .adapter import ChunbaobaoDeepJSCCAdapter

            adapter = ChunbaobaoDeepJSCCAdapter.from_repo(args.baseline, c=args.c, snr_db=7.0)
            torch.manual_seed(args.seed)
            images = torch.rand(2, 3, 32, 32)
            with torch.no_grad():
                reconstructed = adapter.forward(images, 7.0)
            _json_dump(output / "smoke_forward.json", {"input_shape": list(images.shape), "output_shape": list(reconstructed.shape), "baseline": adapter.identity().__dict__})
        except ImportError as exc:
            raise SystemExit(f"P0 smoke requires PyTorch and the upstream dependencies: {exc}") from exc
    print(output)
    return 0


def command_run_p1(args: argparse.Namespace) -> int:
    output = _run_dir(args)
    configs = iter_fixed_trigger_configs(order=args.order)
    _json_dump(output / "candidate_configs.json", [{"gain_db": c.gain_db, "phase_deg": c.phase_deg, "dc_magnitude": c.dc_magnitude, "dc_phase_deg": c.dc_phase_deg, "order": c.order, "hash": trigger_config_hash(c)} for c in configs])
    if not args.measurements:
        print(f"wrote {len(configs)} candidate configurations; provide --measurements JSON to select a candidate")
        return 0
    records = json.loads(Path(args.measurements).read_text(encoding="utf-8"))
    measurements = []
    from .impairments import IQDCOffsetConfig

    for record in records:
        config = IQDCOffsetConfig(record["gain_db"], record["phase_deg"], record["dc_magnitude"], record["dc_phase_deg"], record.get("order", args.order))
        measurements.append(CandidateMeasurement(config, record["evm_rms"], record["power_change_db"], record["clean_psnr_loss_db"], record["clean_false_activation_rate"]))
    selected = select_trigger_candidate(measurements)
    _json_dump(output / "selected_trigger.json", {"config": selected.config.__dict__, "metrics": {"evm_rms": selected.evm_rms, "power_change_db": selected.power_change_db, "clean_psnr_loss_db": selected.clean_psnr_loss_db, "clean_false_activation_rate": selected.clean_false_activation_rate}, "trigger_hash": trigger_config_hash(selected.config)})
    return 0


def command_run_p2(args: argparse.Namespace) -> int:
    output = _run_dir(args)
    import numpy as np

    if args.target_array:
        target_sha = sha256_array(np.load(args.target_array))
    elif args.target_sha256:
        target_sha = args.target_sha256
    else:
        raise SystemExit("P2 requires --target-array or --target-sha256 so the target cannot be silently changed")
    train_indices, _ = build_split_indices(args.dataset_size, target_index=args.target_index, train_count=args.train_count, seed=args.split_seed)
    manifests = []
    for rate in args.poison_rates:
        manifest = build_poison_manifest("CIFAR10", args.dataset_size, rate, seed=args.seed, target_index=args.target_index, target_sha256=target_sha, eligible_indices=train_indices)
        manifest_path = output / f"poison_manifest_{rate:g}.json"
        manifest.write_json(manifest_path)
        manifests.append({"poison_rate": rate, "actual_rate": manifest.actual_rate, "manifest": str(manifest_path), "manifest_sha256": manifest.manifest_sha256})
    _json_dump(output / "p2_manifest_index.json", {"trigger": args.trigger, "manifests": manifests, "target_sha256": target_sha})
    print(output)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    import numpy as np

    clean_clean = np.load(args.clean_clean)
    clean_trigger = np.load(args.clean_trigger)
    backdoor_clean = np.load(args.backdoor_clean)
    backdoor_trigger = np.load(args.backdoor_trigger)
    source = np.load(args.source)
    target = np.load(args.target)
    if args.threshold is None:
        if not args.p1_validation_target_psnr:
            raise SystemExit("provide --threshold or --p1-validation-target-psnr")
        threshold = freeze_asr_threshold(np.load(args.p1_validation_target_psnr))
    else:
        threshold = args.threshold
    report = compute_four_way_report(clean_clean, clean_trigger, backdoor_clean, backdoor_trigger, source, target, asr_threshold_db=threshold)
    _json_dump(Path(args.output), report.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepJSCC IQ/DC-offset offline study commands")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-baseline")
    fetch.add_argument("--destination", default="external")
    fetch.set_defaults(func=command_fetch_baseline)

    p0 = sub.add_parser("run-p0")
    p0.add_argument("--baseline", default="external/chunbaobao-deepjscc")
    p0.add_argument("--output", default="runs/p0")
    p0.add_argument("--seed", type=int, default=0)
    p0.add_argument("--c", type=int, default=8)
    p0.add_argument("--smoke", action="store_true")
    p0.set_defaults(func=command_run_p0)

    p1 = sub.add_parser("run-p1-scan")
    p1.add_argument("--output", default="runs/p1_scan")
    p1.add_argument("--measurements")
    p1.add_argument("--order", choices=["iq_then_dc", "dc_then_iq"], default="iq_then_dc")
    p1.set_defaults(func=command_run_p1)

    p2 = sub.add_parser("run-p2")
    p2.add_argument("--trigger", choices=["fixed", "learnable"], required=True)
    p2.add_argument("--output", default="runs/p2")
    p2.add_argument("--dataset-size", type=int, default=50000)
    p2.add_argument("--train-count", type=int, default=45000)
    p2.add_argument("--target-index", type=int, default=0)
    p2.add_argument("--split-seed", type=int, default=20260823)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--poison-rates", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    p2.add_argument("--target-array")
    p2.add_argument("--target-sha256")
    p2.set_defaults(func=command_run_p2)

    evaluation = sub.add_parser("evaluate-four-way")
    for name in ("clean_clean", "clean_trigger", "backdoor_clean", "backdoor_trigger", "source", "target"):
        evaluation.add_argument(f"--{name.replace('_', '-')}", required=True)
    evaluation.add_argument("--threshold", type=float)
    evaluation.add_argument("--p1-validation-target-psnr")
    evaluation.add_argument("--output", default="four_way_report.json")
    evaluation.set_defaults(func=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


def _entrypoint(command: str) -> int:
    import sys

    return main([command, *sys.argv[1:]])


def fetch_baseline_main() -> int:
    return _entrypoint("fetch-baseline")


def run_p0_main() -> int:
    return _entrypoint("run-p0")


def run_p1_main() -> int:
    return _entrypoint("run-p1-scan")


def run_p2_main() -> int:
    return _entrypoint("run-p2")


def evaluate_four_way_main() -> int:
    return _entrypoint("evaluate-four-way")


if __name__ == "__main__":
    raise SystemExit(main())
