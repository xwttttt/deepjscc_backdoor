"""Authoritative availability_v2 pipeline entry point.

The script is intentionally stage-gated.  It prepares artifacts and commands
for the server workflow but does not manage remote processes, tmux sessions,
GPU jobs, or synchronization.  Run it from the repository root with the
project installed in the active environment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rf_backdoor.adapter import DeepJSCCAdapter
from rf_backdoor.availability_backdoor import (
    AvailabilityBackdoorConfig,
    AvailabilityPoisonManifest,
    AvailabilityPoisonedDataset,
    _evaluate_branch_pairs,
    availability_four_way_metrics_np,
    build_availability_poison_manifest,
    fit_availability_backdoor,
)
from rf_backdoor.metrics import mean_std_t_ci, psnr_np
from rf_backdoor.impairments import IQDCOffsetConfig, LearnableIQDCOffset, apply_iq_dc_torch
from rf_backdoor.metadata import sha256_file
from rf_backdoor.data_protocol import build_split_indices, sha256_array
from rf_backdoor.scan import CandidateMeasurement, iter_fixed_trigger_configs, trigger_config_hash
from rf_backdoor.training import TrainingConfig, evaluate_clean_epoch, fit, set_seed
from rf_backdoor.availability_v2 import (
    GateResult,
    assert_target_free_report,
    availability_metrics_np,
    collect_v2_provenance,
    config_file_for_profile,
    config_sha256_for_profile,
    guard_resume_identity,
    load_toml,
    profile_output_name,
    run_g0,
    select_v2_trigger_candidate,
    sha256_json,
    trigger_stealth_metrics,
    write_json_once,
    write_target_free_report,
)


PIPELINE_NAME = "availability_v2"


def _torch() -> Any:
    import torch

    return torch


def _dump(path: Path, payload: Any, *, resume: bool) -> None:
    write_json_once(path, payload, resume=resume)


def _section(raw: dict[str, Any], *names: str) -> dict[str, Any]:
    value: Any = raw
    for name in names:
        value = value[name]
    if not isinstance(value, dict):
        raise TypeError(f"configuration section is not a table: {'.'.join(names)}")
    return value


def _rooted(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _profile_context(root: Path, profile: str, output_root: Path) -> tuple[dict[str, Any], Path, Path, str]:
    config_path, config_hash, raw = config_sha256_for_profile(root, profile)
    if int(raw.get("schema_version", -1)) != 2:
        raise ValueError(f"unsupported availability_v2 schema in {config_path}")
    base = output_root / profile_output_name(profile)
    base.mkdir(parents=True, exist_ok=True)
    return raw, config_path, base, config_hash


def _data_bundle(root: Path, raw: dict[str, Any]) -> tuple[Any, tuple[int, ...], tuple[int, ...], Any, str]:
    from torch.utils.data import DataLoader, Subset

    dataset_cfg = _section(raw, "dataset")
    base = __import__("rf_backdoor.datasets", fromlist=["CIFAR10Dataset"]).CIFAR10Dataset(root / "data", train=True)
    train_indices, validation_indices = build_split_indices(
        len(base),
        excluded_index=int(dataset_cfg["excluded_index"]),
        train_count=int(dataset_cfg["train_count"]),
        seed=int(dataset_cfg["split_seed"]),
    )
    test = __import__("rf_backdoor.datasets", fromlist=["CIFAR10Dataset"]).CIFAR10Dataset(root / "data", train=False)
    data_hash = sha256_json(
        {
            "dataset": dataset_cfg["name"],
            "archive_sha256": base.archive_sha256,
            "train_indices_sha256": sha256_array(np.asarray(train_indices, dtype=np.int64)),
            "validation_indices_sha256": sha256_array(np.asarray(validation_indices, dtype=np.int64)),
            "test_archive_sha256": test.archive_sha256,
        }
    )
    return base, train_indices, validation_indices, test, data_hash


def _loader(dataset: Any, *, batch_size: int, workers: int, shuffle: bool) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
        pin_memory=True,
        persistent_workers=bool(workers),
    )


def _load_adapter(device: Any, backend: dict[str, Any], checkpoint: Path | None = None) -> DeepJSCCAdapter:
    """Wrap the supplied DeepJSCC model/channel for the attack pipeline.

    Model and channel construction stays here, at the experiment entrypoint;
    ``rf_backdoor`` only receives their public encode/transmit/decode seam.
    """

    from src.channels import Channel, ChannelConfig
    from src.deepjscc.model import DeepJSCC

    snr_db = float(backend["train_snr_db"])
    channel = Channel(
        ChannelConfig(
            channel_type="awgn",
            snr_min=snr_db,
            snr_max=snr_db,
        )
    )
    adapter = DeepJSCCAdapter(
        DeepJSCC(c=int(backend["c"])),
        channel=channel,
        baseline_path=Path(__file__).resolve().parents[1] / "src" / "src",
    )
    adapter.model.to(device)
    if checkpoint is not None:
        payload = _torch().load(checkpoint, map_location=device)
        adapter.model.load_state_dict(payload["model"])
    return adapter


def _p0_path(profile_dir: Path) -> Path:
    return profile_dir / "p0" / "best.pt"


def _trigger_path(profile_dir: Path) -> Path:
    return profile_dir / "trigger" / "selected_trigger.json"


def _stable_provenance(root: Path, *, config_hash: str, data_hash: str, seed: int, upstream_hash: str | None = None) -> dict[str, Any]:
    payload = collect_v2_provenance(
        root,
        config_sha256=config_hash,
        data_sha256=data_hash,
        seed=seed,
        upstream_checkpoint_sha256=upstream_hash,
    )
    payload.pop("timestamp_utc", None)
    return payload


def _require_gate(profile_dir: Path, name: str) -> dict[str, Any]:
    path = profile_dir / name
    if not path.exists():
        raise RuntimeError(f"required gate artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"required gate did not pass: {path}")
    return payload


def run_preflight(root: Path, profile: str, output_root: Path, *, raw: dict[str, Any], config_hash: str, resume: bool, device: str) -> GateResult:
    profile_dir = output_root / profile_output_name(profile)
    identity = {
        "pipeline": PIPELINE_NAME,
        "profile": profile,
        "stage": "preflight",
        "config_sha256": config_hash,
        "data_sha256": "synthetic-g0",
        "seed": int(_section(raw, "preflight")["seed"]),
        "provenance": _stable_provenance(root, config_hash=config_hash, data_hash="synthetic-g0", seed=int(_section(raw, "preflight")["seed"])),
    }
    reusable = guard_resume_identity(profile_dir / "preflight", identity, resume=resume)
    result_path = profile_dir / "preflight" / "g0.json"
    if reusable and result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result = GateResult("G0", payload.get("status") == "PASS", payload.get("checks", {}))
        if not result.passed:
            raise RuntimeError("G0 failed; stopping before P0")
        return result
    result = run_g0(seed=int(_section(raw, "preflight")["seed"]), device=device)
    _dump(result_path, result.to_dict(), resume=resume)
    provenance = collect_v2_provenance(
        root,
        config_sha256=config_hash,
        data_sha256="synthetic-g0",
        seed=int(_section(raw, "preflight")["seed"]),
    )
    _dump(profile_dir / "preflight" / "provenance.json", provenance, resume=resume)
    _dump(profile_dir / "preflight_gate.json", result.to_dict(), resume=resume)
    if not result.passed:
        raise RuntimeError("G0 failed; stopping before P0")
    return result


def run_p0(root: Path, profile: str, profile_dir: Path, *, raw: dict[str, Any], config_hash: str, resume: bool, device: Any) -> Path:
    _require_gate(profile_dir, "preflight_gate.json")
    base, train_indices, validation_indices, test, data_hash = _data_bundle(root, raw)
    p0_cfg = _section(raw, "p0")
    p0_dir = profile_dir / "p0"
    identity = {
        "pipeline": PIPELINE_NAME,
        "profile": profile,
        "stage": "p0",
        "config_sha256": config_hash,
        "data_sha256": data_hash,
        "seed": int(p0_cfg["seed"]),
        "upstream_checkpoint_sha256": None,
        "provenance": _stable_provenance(root, config_hash=config_hash, data_hash=data_hash, seed=int(p0_cfg["seed"])),
    }
    reusable = guard_resume_identity(p0_dir, identity, resume=resume)
    checkpoint = _p0_path(profile_dir)
    if reusable and checkpoint.exists():
        return checkpoint
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"P0 checkpoint exists; pass --resume to reuse it: {checkpoint}")
    train_loader = _loader(__import__("torch.utils.data", fromlist=["Subset"]).Subset(base, train_indices), batch_size=p0_cfg["batch_size"], workers=p0_cfg["workers"], shuffle=True)
    validation_loader = _loader(__import__("torch.utils.data", fromlist=["Subset"]).Subset(base, validation_indices), batch_size=p0_cfg["batch_size"], workers=p0_cfg["workers"], shuffle=False)
    test_loader = _loader(test, batch_size=p0_cfg["batch_size"], workers=p0_cfg["workers"], shuffle=False)
    set_seed(int(p0_cfg["seed"]))
    adapter = _load_adapter(device, _section(raw, "backend"))
    optimizer = _torch().optim.Adam(adapter.model.parameters(), lr=float(p0_cfg["learning_rate"]), weight_decay=float(p0_cfg["weight_decay"]))
    training_config = TrainingConfig(
        snr_db=float(_section(raw, "backend")["train_snr_db"]),
        max_epochs=int(p0_cfg["max_epochs"]),
        early_stopping_start=int(p0_cfg["early_stopping_start"]),
        patience=int(p0_cfg["patience"]),
        mixed_precision=bool(p0_cfg["mixed_precision"]),
    )
    history = fit(adapter, train_loader, validation_loader, optimizer, config=training_config, device=device, checkpoint_path=checkpoint)
    history_path = p0_dir / "history.json"
    _dump(history_path, history, resume=resume)
    provenance = collect_v2_provenance(root, config_sha256=config_hash, data_sha256=data_hash, seed=int(p0_cfg["seed"]))
    payload = _torch().load(checkpoint, map_location=device)
    payload["provenance"] = provenance
    payload["run_identity"] = identity
    _torch().save(payload, checkpoint)
    payload = _torch().load(checkpoint, map_location=device)
    adapter.model.load_state_dict(payload["model"])
    set_seed(int(p0_cfg["seed"]) + 1)
    validation = evaluate_clean_epoch(adapter, validation_loader, snr_db=float(_section(raw, "backend")["train_snr_db"]), device=device)
    set_seed(int(p0_cfg["seed"]) + 2)
    independent_test = evaluate_clean_epoch(adapter, test_loader, snr_db=float(_section(raw, "backend")["train_snr_db"]), device=device)
    validation_values = [float(row["validation_psnr"]) for row in history if np.isfinite(row["validation_psnr"])]
    best_psnr = max(validation_values)
    epoch0_psnr = validation_values[0]
    checks = {
        "finite_history": bool(all(np.isfinite(value) for row in history for value in row.values() if isinstance(value, (int, float)))),
        "best_validation_psnr_db": best_psnr,
        "minimum_best_validation_psnr_db": float(p0_cfg["minimum_best_psnr_db"]),
        "validation_improvement_db": best_psnr - epoch0_psnr,
        "minimum_validation_improvement_db": float(p0_cfg["minimum_improvement_db"]),
        "independent_test_psnr_db": float(independent_test["psnr"]),
        "checkpoint_test_minus_validation_db": float(independent_test["psnr"] - validation["psnr"]),
        "maximum_checkpoint_test_delta_db": float(p0_cfg["maximum_checkpoint_test_delta_db"]),
    }
    passed = (
        checks["finite_history"]
        and best_psnr >= checks["minimum_best_validation_psnr_db"]
        and checks["validation_improvement_db"] >= checks["minimum_validation_improvement_db"]
        and abs(checks["checkpoint_test_minus_validation_db"]) <= checks["maximum_checkpoint_test_delta_db"]
    )
    gate = {"gate": "G1", "status": "PASS" if passed else "FAIL", "checks": checks}
    _dump(p0_dir / "p0_gate.json", gate, resume=resume)
    _dump(p0_dir / "metrics.json", {"validation": validation, "independent_test": independent_test, "checks": checks}, resume=resume)
    _dump(p0_dir / "provenance.json", provenance, resume=resume)
    if not passed:
        raise RuntimeError("G1 failed; stopping before trigger generation")
    return checkpoint


def _measurement_physical(reference: Any, impaired: Any) -> tuple[float, float]:
    reference_np = reference.detach().cpu().numpy()
    impaired_np = impaired.detach().cpu().numpy()
    return trigger_stealth_metrics(reference_np, impaired_np)["evm"], trigger_stealth_metrics(reference_np, impaired_np)["power_change_db"]


def _candidate_measurement(adapter: Any, loader: Any, config: IQDCOffsetConfig, *, device: Any, failure_drop_db: float) -> CandidateMeasurement:
    torch = _torch()
    ref_power = error_power = out_power = 0.0
    clean_drop_sum = 0.0
    failures = total = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            latent = adapter.encode(images)
            impaired = apply_iq_dc_torch(latent, config)
            from rf_backdoor.availability_backdoor import _paired_channel_decode

            clean_output, triggered_output = _paired_channel_decode(adapter, latent, impaired, 7.0)
            reference_np = latent.detach().cpu().numpy()
            impaired_np = impaired.detach().cpu().numpy()
            clean_np = clean_output.detach().cpu().numpy()
            triggered_np = triggered_output.detach().cpu().numpy()
            source_np = images.detach().cpu().numpy()
            clean_psnr = psnr_np(clean_np, source_np)
            triggered_psnr = psnr_np(triggered_np, source_np)
            drop = clean_psnr - triggered_psnr
            ref_power += float(np.square(np.abs(reference_np).astype(np.float64)).sum())
            error_power += float(np.square(np.abs(impaired_np - reference_np).astype(np.float64)).sum())
            out_power += float(np.square(np.abs(impaired_np).astype(np.float64)).sum())
            clean_drop_sum += float(drop.sum())
            failures += int(np.sum(drop >= failure_drop_db))
            total += int(drop.size)
    evm = float(np.sqrt(error_power / max(ref_power, 1e-12)))
    power_db = float(10.0 * np.log10(max(out_power, 1e-12) / max(ref_power, 1e-12)))
    return CandidateMeasurement(config, evm, power_db, clean_drop_sum / max(total, 1), failures / max(total, 1))


def run_trigger(root: Path, profile: str, profile_dir: Path, *, raw: dict[str, Any], config_hash: str, resume: bool, device: Any) -> IQDCOffsetConfig:
    _require_gate(profile_dir, "p0/p0_gate.json")
    base, _, validation_indices, _, data_hash = _data_bundle(root, raw)
    trigger_cfg = _section(raw, "trigger")
    trigger_dir = profile_dir / "trigger"
    identity = {
        "pipeline": PIPELINE_NAME,
        "profile": profile,
        "stage": "trigger",
        "config_sha256": config_hash,
        "data_sha256": data_hash,
        "seed": int(_section(raw, "p0")["seed"]),
        "upstream_checkpoint_sha256": sha256_file(_p0_path(profile_dir)),
        "provenance": _stable_provenance(root, config_hash=config_hash, data_hash=data_hash, seed=int(_section(raw, "p0")["seed"]), upstream_hash=sha256_file(_p0_path(profile_dir))),
    }
    reusable = guard_resume_identity(trigger_dir, identity, resume=resume)
    selected_path = _trigger_path(profile_dir)
    if reusable and selected_path.exists():
        return IQDCOffsetConfig(**json.loads(selected_path.read_text(encoding="utf-8"))["config"])
    if selected_path.exists() and not resume:
        raise FileExistsError(f"selected trigger exists; pass --resume to reuse it: {selected_path}")
    p0_cfg = _section(raw, "p0")
    validation_loader = _loader(__import__("torch.utils.data", fromlist=["Subset"]).Subset(base, validation_indices), batch_size=p0_cfg["batch_size"], workers=p0_cfg["workers"], shuffle=False)
    adapter = _load_adapter(device, _section(raw, "backend"), _p0_path(profile_dir))
    if profile != "pilot":
        pilot_dir = profile_dir.parent / "pilot"
        pilot_gate_path = pilot_dir / "g3_gate.json"
        if not pilot_gate_path.exists():
            raise RuntimeError("full profiles require a completed pilot G3 gate")
        pilot_gate = json.loads(pilot_gate_path.read_text(encoding="utf-8"))
        if not pilot_gate.get("passed_trigger_kinds"):
            raise RuntimeError("pilot G3 did not pass any trigger family")
        pilot_selected_path = pilot_dir / "trigger" / "selected_trigger.json"
        if not pilot_selected_path.exists():
            raise RuntimeError(f"pilot selected trigger is missing: {pilot_selected_path}")
        selected = IQDCOffsetConfig(**json.loads(pilot_selected_path.read_text(encoding="utf-8"))["config"])
        selected_payload = {
            "config": asdict(selected),
            "trigger_hash": trigger_config_hash(selected),
            "selection_policy": "frozen_from_pilot_g2",
            "source_profile": "pilot",
            "pilot_passed_trigger_kinds": pilot_gate["passed_trigger_kinds"],
        }
        _dump(selected_path, selected_payload, resume=resume)
        gate = {"gate": "G2", "status": "PASS", "checks": selected_payload}
        _dump(trigger_dir / "trigger_gate.json", gate, resume=resume)
        _dump(profile_dir / "trigger_gate.json", gate, resume=resume)
        return selected
    configs = iter_fixed_trigger_configs(
        gains_db=trigger_cfg["gains_db"],
        phases_deg=trigger_cfg["phases_deg"],
        dc_magnitudes=trigger_cfg["dc_magnitudes"],
        dc_phases_deg=trigger_cfg["dc_phases_deg"],
    )
    first_batch = next(iter(validation_loader))
    images = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch
    images = images.to(device)
    with _torch().no_grad():
        latent = adapter.encode(images)
        feasible: list[CandidateMeasurement] = []
        for candidate in configs:
            impaired = apply_iq_dc_torch(latent, candidate)
            evm, power_db = _measurement_physical(latent, impaired)
            if evm <= float(trigger_cfg["evm_limit"]) and abs(power_db) <= float(trigger_cfg["power_change_db_limit"]):
                if (not bool(trigger_cfg["require_joint"])) or ((candidate.gain_db != 0.0 or candidate.phase_deg != 0.0) and candidate.dc_magnitude != 0.0):
                    feasible.append(CandidateMeasurement(candidate, evm, power_db, 0.0, 0.0))
    if not feasible:
        _dump(trigger_dir / "trigger_gate.json", {"gate": "G2", "status": "FAIL", "reason": "no physically feasible joint IQ+DC candidate"}, resume=resume)
        raise RuntimeError("G2 failed: no physically feasible joint IQ+DC candidate")
    shortlist = sorted(feasible, key=lambda item: (-item.evm_rms, abs(item.power_change_db), item.config.gain_db, item.config.phase_deg, item.config.dc_magnitude, item.config.dc_phase_deg))[: int(trigger_cfg["max_full_validation_candidates"])]
    measurements = [_candidate_measurement(adapter, validation_loader, item.config, device=device, failure_drop_db=float(_section(raw, "g3")["failure_drop_db"])) for item in shortlist]
    try:
        selected = select_v2_trigger_candidate(
            measurements,
            evm_limit=float(trigger_cfg["evm_limit"]),
            power_change_db_limit=float(trigger_cfg["power_change_db_limit"]),
            clean_psnr_drop_limit=float(trigger_cfg["clean_psnr_drop_limit_db"]),
            clean_failure_rate_limit=float(trigger_cfg["clean_failure_rate_limit"]),
            require_joint=bool(trigger_cfg["require_joint"]),
        )
    except ValueError as exc:
        failure = {"gate": "G2", "status": "FAIL", "reason": str(exc), "shortlist_size": len(shortlist)}
        _dump(trigger_dir / "trigger_gate.json", failure, resume=resume)
        _dump(profile_dir / "trigger_gate.json", failure, resume=resume)
        raise RuntimeError("G2 failed: no candidate satisfies physical and clean-model constraints") from exc
    _dump(trigger_dir / "measurements.json", [{**asdict(item), "config": asdict(item.config)} for item in measurements], resume=resume)
    selected_payload = {
        "config": asdict(selected.config),
        "trigger_hash": trigger_config_hash(selected.config),
        "evm": selected.evm_rms,
        "power_change_db": selected.power_change_db,
        "clean_psnr_drop_db": selected.clean_psnr_loss_db,
        "clean_failure_rate": selected.clean_false_activation_rate,
        "selection_policy": "max_evm_then_min_clean_damage_then_lexicographic_among_joint_feasible_candidates",
        "shortlist_size": len(shortlist),
    }
    _dump(selected_path, selected_payload, resume=resume)
    gate = {
        "gate": "G2",
        "status": "PASS",
        "checks": {
            "candidate_grid_size": len(configs),
            "physically_feasible_joint_candidates": len(feasible),
            "full_validation_shortlist_size": len(shortlist),
            "selected": selected_payload,
        },
    }
    _dump(trigger_dir / "trigger_gate.json", gate, resume=resume)
    _dump(profile_dir / "trigger_gate.json", gate, resume=resume)
    return selected.config


def _training_specs(profile: str, raw: dict[str, Any], promotion: dict[str, Any] | None) -> list[dict[str, Any]]:
    if profile == "pilot":
        g3 = _section(raw, "g3")
        return [
            {"trigger_kind": kind, "poison_rate": float(g3["poison_rate"]), "seed": int(g3["seed"]), "evm_penalty_weight": 0.0 if kind == "fixed" else float(g3["evm_penalty_weight"]), "power_penalty_weight": 0.0 if kind == "fixed" else float(g3["power_penalty_weight"]), "train_snr_db": g3["train_snr_db"], "test_snr_db": g3["validation_snr_db"], "max_epochs": g3["max_epochs"], "early_stopping_start": g3["early_stopping_start"], "patience": g3["patience"]}
            for kind in ("fixed", "learnable")
        ]
    full = _section(raw, "full")
    if profile == "full-dev":
        dev = full["dev"]
        return [
            {"trigger_kind": "learnable", "poison_rate": float(dev["poison_rate"]), "seed": int(dev["seed"]), "evm_penalty_weight": float(evm), "power_penalty_weight": float(power), "train_snr_db": dev["train_snr_db"], "test_snr_db": dev["test_snr_db"], "max_epochs": dev["max_epochs"], "early_stopping_start": dev["early_stopping_start"], "patience": dev["patience"]}
            for evm, power in product(dev["penalty_evm"], dev["penalty_power"])
        ]
    confirm = full["confirm"]
    if promotion is None or promotion.get("status") != "PASS":
        raise RuntimeError("full-confirm requires a PASS promotion_report from full-dev")
    selected = promotion["selected_penalty"]
    return [
        {"trigger_kind": kind, "poison_rate": float(rate), "seed": int(seed), "evm_penalty_weight": 0.0 if kind == "fixed" else float(selected["evm_penalty_weight"]), "power_penalty_weight": 0.0 if kind == "fixed" else float(selected["power_penalty_weight"]), "train_snr_db": confirm["train_snr_db"], "test_snr_db": confirm["test_snr_db"], "max_epochs": confirm["max_epochs"], "early_stopping_start": confirm["early_stopping_start"], "patience": confirm["patience"]}
        for kind, rate, seed in product(("fixed", "learnable"), confirm["poison_rates"], confirm["seeds"])
    ]


def run_train(root: Path, profile: str, profile_dir: Path, *, raw: dict[str, Any], config_hash: str, resume: bool, device: Any) -> None:
    _require_gate(profile_dir, "trigger_gate.json")
    selected = IQDCOffsetConfig(**json.loads(_trigger_path(profile_dir).read_text(encoding="utf-8"))["config"])
    base, train_indices, validation_indices, _, data_hash = _data_bundle(root, raw)
    promotion_path = (profile_dir.parent / "dev" / "promotion_report.json") if profile == "full-confirm" else (profile_dir / "promotion_report.json")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8")) if promotion_path.exists() else None
    specs = _training_specs(profile, raw, promotion)
    if profile != "pilot":
        pilot_gate_path = profile_dir.parent / "pilot" / "g3_gate.json"
        if not pilot_gate_path.exists():
            raise RuntimeError("full profiles require pilot g3_gate.json")
        pilot_gate = json.loads(pilot_gate_path.read_text(encoding="utf-8"))
        passed_kinds = set(pilot_gate.get("passed_trigger_kinds", []))
        specs = [spec for spec in specs if spec["trigger_kind"] in passed_kinds]
        if not specs:
            raise RuntimeError("no full-experiment trigger family passed pilot G3")
    train_dir = profile_dir / "train"
    identity = {"pipeline": PIPELINE_NAME, "profile": profile, "stage": "train", "config_sha256": config_hash, "data_sha256": data_hash, "trigger_hash": trigger_config_hash(selected), "provenance": _stable_provenance(root, config_hash=config_hash, data_hash=data_hash, seed=0, upstream_hash=sha256_file(_p0_path(profile_dir)))}
    guard_resume_identity(train_dir, identity, resume=resume)
    p0_hash = sha256_file(_p0_path(profile_dir))
    p0_cfg = _section(raw, "p0")
    for spec in specs:
        run_dir = train_dir / spec["trigger_kind"] / f"poison_{spec['poison_rate']:g}" / f"evm_{spec['evm_penalty_weight']:g}_power_{spec['power_penalty_weight']:g}" / f"seed_{spec['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_identity = {"pipeline": PIPELINE_NAME, "profile": profile, "stage": "train_run", "config_sha256": config_hash, "data_sha256": data_hash, "upstream_checkpoint_sha256": p0_hash, "trigger_hash": trigger_config_hash(selected), "seed": int(spec["seed"]), "poison_rate": float(spec["poison_rate"]), "trigger_kind": spec["trigger_kind"], "evm_penalty_weight": float(spec["evm_penalty_weight"]), "power_penalty_weight": float(spec["power_penalty_weight"]), "provenance": _stable_provenance(root, config_hash=config_hash, data_hash=data_hash, seed=int(spec["seed"]), upstream_hash=p0_hash)}
        reusable = guard_resume_identity(run_dir, run_identity, resume=resume)
        checkpoint = run_dir / "best.pt"
        if reusable and checkpoint.exists():
            continue
        if checkpoint.exists() and not resume:
            raise FileExistsError(f"training checkpoint exists; pass --resume to reuse it: {checkpoint}")
        manifest = build_availability_poison_manifest("CIFAR10", len(base), float(spec["poison_rate"]), seed=int(spec["seed"]), split_seed=int(_section(raw, "dataset")["split_seed"]), eligible_indices=train_indices)
        poisoned = AvailabilityPoisonedDataset(base, manifest)
        train_loader = _loader(__import__("torch.utils.data", fromlist=["Subset"]).Subset(poisoned, train_indices), batch_size=p0_cfg["batch_size"], workers=p0_cfg["workers"], shuffle=True)
        validation_loader = _loader(__import__("torch.utils.data", fromlist=["Subset"]).Subset(base, validation_indices), batch_size=p0_cfg["batch_size"], workers=p0_cfg["workers"], shuffle=False)
        set_seed(int(spec["seed"]))
        adapter = _load_adapter(device, _section(raw, "backend"), _p0_path(profile_dir))
        clean_reference_adapter = _load_adapter(device, _section(raw, "backend"), _p0_path(profile_dir))
        trigger: Any = selected
        parameters = list(adapter.model.parameters())
        if spec["trigger_kind"] == "learnable":
            trigger = LearnableIQDCOffset(selected).to(device)
            parameters += list(trigger.parameters())
        optimizer = _torch().optim.Adam(parameters, lr=float(p0_cfg["learning_rate"]), weight_decay=float(p0_cfg["weight_decay"]))
        backdoor_config = AvailabilityBackdoorConfig(
            train_snr_db=tuple(float(item) for item in spec["train_snr_db"]),
            min_psnr_drop_db=1.0 if profile == "pilot" else 3.0,
            evm_penalty_weight=float(spec["evm_penalty_weight"]),
            power_penalty_weight=float(spec["power_penalty_weight"]),
            max_clean_utility_drop_db=1.0 if profile == "pilot" else 0.5,
            max_epochs=int(spec["max_epochs"]),
            early_stopping_start=int(spec["early_stopping_start"]),
            patience=int(spec["patience"]),
            mixed_precision=bool(_section(raw, "p0")["mixed_precision"]),
        )
        history = fit_availability_backdoor(adapter, clean_reference_adapter, train_loader, validation_loader, optimizer, trigger=trigger, config=backdoor_config, device=device, checkpoint_path=checkpoint, progress_label=f"{spec['trigger_kind']}/seed_{spec['seed']}")
        _dump(run_dir / "history.json", history, resume=resume)
        provenance = collect_v2_provenance(root, config_sha256=config_hash, data_sha256=data_hash, seed=int(spec["seed"]), upstream_checkpoint_sha256=p0_hash)
        provenance.update({"trigger_kind": spec["trigger_kind"], "poison_rate": spec["poison_rate"], "evm_penalty_weight": spec["evm_penalty_weight"], "power_penalty_weight": spec["power_penalty_weight"], "manifest_sha256": manifest.manifest_sha256})
        _dump(run_dir / "poison_manifest.json", {"manifest": manifest.to_dict(), "provenance": provenance}, resume=resume)
        trigger_payload = {"trigger_kind": spec["trigger_kind"], "config": asdict(trigger.config()) if hasattr(trigger, "config") else asdict(trigger), "initial_config": asdict(selected), "trigger_hash": trigger_config_hash(trigger.config() if hasattr(trigger, "config") else trigger), "poison_rate": spec["poison_rate"], "seed": spec["seed"], "evm_penalty_weight": spec["evm_penalty_weight"], "power_penalty_weight": spec["power_penalty_weight"]}
        _dump(run_dir / "trigger.json", trigger_payload, resume=resume)
        payload = _torch().load(checkpoint, map_location=device)
        payload["provenance"] = provenance
        payload["run_identity"] = run_identity
        _torch().save(payload, checkpoint)
        _dump(run_dir / "provenance.json", provenance, resume=resume)
    _dump(profile_dir / "train_gate.json", {"gate": "G3", "status": "PASS", "run_count": len(specs), "runs": specs}, resume=resume)


def output_root_for(profile_dir: Path) -> Path:
    return profile_dir.parent


def _control_configs(config: IQDCOffsetConfig) -> dict[str, IQDCOffsetConfig | None]:
    return {
        "iq_only": IQDCOffsetConfig(config.gain_db, config.phase_deg, 0.0, 0.0, config.order),
        "dc_only": IQDCOffsetConfig(0.0, 0.0, config.dc_magnitude, config.dc_phase_deg, config.order),
        "order_swap": IQDCOffsetConfig(config.gain_db, config.phase_deg, config.dc_magnitude, config.dc_phase_deg, "dc_then_iq" if config.order == "iq_then_dc" else "iq_then_dc"),
        "sign_flip": IQDCOffsetConfig(-config.gain_db, config.phase_deg, config.dc_magnitude, config.dc_phase_deg + 180.0, config.order),
        "phase_flip": IQDCOffsetConfig(config.gain_db, -config.phase_deg, config.dc_magnitude, config.dc_phase_deg + 180.0, config.order),
        "equal_evm_random": None,
    }


def _equal_evm_random(value: Any, evm: float) -> Any:
    torch = _torch()
    noise = torch.complex(torch.randn_like(value.real), torch.randn_like(value.real)) if value.is_complex() else torch.randn_like(value)
    dims = tuple(range(1, value.ndim))
    ref_rms = value.abs().square().mean(dim=dims, keepdim=True).sqrt() if value.is_complex() else value.square().mean(dim=dims, keepdim=True).sqrt()
    noise_rms = noise.abs().square().mean(dim=dims, keepdim=True).sqrt() if noise.is_complex() else noise.square().mean(dim=dims, keepdim=True).sqrt()
    return value + float(evm) * ref_rms * noise / noise_rms.clamp_min(1e-12)


def _evaluation_constraints(raw: dict[str, Any], profile: str) -> dict[str, float]:
    """Map profile-specific config tables to one evaluation contract."""

    trigger_cfg = _section(raw, "trigger")
    if profile == "pilot":
        success_cfg = _section(raw, "g3")
        return {
            "minimum_worst_snr_psnr_drop_db": float(success_cfg["minimum_snr7_psnr_drop_db"]),
            "minimum_worst_snr_failure_rate": float(success_cfg["minimum_snr7_failure_rate"]),
            "maximum_clean_trigger_failure_rate": float(success_cfg["maximum_clean_trigger_failure_rate"]),
            "maximum_clean_utility_drop_db": float(success_cfg["maximum_clean_utility_drop_db"]),
            "evm_limit": float(trigger_cfg["evm_limit"]),
            "power_change_db_limit": float(trigger_cfg["power_change_db_limit"]),
            "minimum_snr7_psnr_drop_db": float(success_cfg["minimum_snr7_psnr_drop_db"]),
            "minimum_snr7_failure_rate": float(success_cfg["minimum_snr7_failure_rate"]),
            "minimum_cross_snr_drop_db": float(success_cfg["minimum_cross_snr_drop_db"]),
        }
    success_cfg = _section(raw, "success")
    return {
        "minimum_worst_snr_psnr_drop_db": float(success_cfg["minimum_worst_snr_psnr_drop_db"]),
        "minimum_worst_snr_failure_rate": float(success_cfg["minimum_worst_snr_failure_rate"]),
        "maximum_clean_trigger_failure_rate": float(success_cfg["maximum_clean_trigger_failure_rate"]),
        "maximum_clean_utility_drop_db": float(success_cfg["maximum_clean_utility_drop_db"]),
        "evm_limit": float(success_cfg["evm_limit"]),
        "power_change_db_limit": float(success_cfg["power_change_db_limit"]),
    }


def _evaluate_one(root: Path, profile: str, profile_dir: Path, run_dir: Path, *, raw: dict[str, Any], config_hash: str, device: Any, resume: bool) -> dict[str, Any]:
    spec = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "best.pt"
    p0_cfg = _section(raw, "p0")
    _, _, _, test, data_hash = _data_bundle(root, raw)
    test_loader = _loader(test, batch_size=p0_cfg["batch_size"], workers=p0_cfg["workers"], shuffle=False)
    clean_adapter = _load_adapter(device, _section(raw, "backend"), _p0_path(profile_dir))
    backdoor_adapter = _load_adapter(device, _section(raw, "backend"), checkpoint)
    trigger_payload = json.loads((run_dir / "trigger.json").read_text(encoding="utf-8"))
    if spec["trigger_kind"] == "learnable":
        trigger: Any = LearnableIQDCOffset(IQDCOffsetConfig(**trigger_payload["initial_config"])).to(device)
        payload = _torch().load(checkpoint, map_location=device)
        trigger.load_state_dict(payload["trigger"])
        trigger.eval()
    else:
        trigger = IQDCOffsetConfig(**trigger_payload["config"])
    test_snrs = _section(raw, "g3")["validation_snr_db"] if profile == "pilot" else (_section(raw, "full")["dev"]["test_snr_db"] if profile == "full-dev" else _section(raw, "full")["confirm"]["test_snr_db"])
    scores, latent, impaired = _evaluate_branch_pairs(clean_adapter, backdoor_adapter, test_loader, trigger, test_snrs, device, mixed_precision=bool(p0_cfg["mixed_precision"]))
    failure_drop_db = float(_section(raw, "g3")["failure_drop_db"]) if profile == "pilot" else float(_section(raw, "success")["failure_drop_db"])
    metric_rows: list[dict[str, Any]] = []
    for snr_db, parts in sorted(scores.items()):
        clean_clean, clean_trigger, backdoor_clean, backdoor_trigger, source = parts
        metric_rows.append({"snr_db": float(snr_db), "clean_model": availability_metrics_np(clean_clean, clean_trigger, source, failure_drop_db=failure_drop_db), "backdoor_model": availability_metrics_np(backdoor_clean, backdoor_trigger, source, failure_drop_db=failure_drop_db), "backdoor_clean_utility_drop_db": float(np.mean(psnr_np(clean_clean, source)) - np.mean(psnr_np(backdoor_clean, source)))})
    stealth = trigger_stealth_metrics(latent, impaired)
    worst_drop = min(row["backdoor_model"]["psnr_drop_db"] for row in metric_rows)
    worst_failure = min(row["backdoor_model"]["failure_rate"] for row in metric_rows)
    clean_failure = max(row["clean_model"]["failure_rate"] for row in metric_rows)
    clean_utility = max(row["backdoor_clean_utility_drop_db"] for row in metric_rows)
    constraints = _evaluation_constraints(raw, profile)
    row_by_snr = {float(row["snr_db"]): row for row in metric_rows}
    checks = {
        "worst_snr_psnr_drop_db": worst_drop,
        "minimum_worst_snr_psnr_drop_db": constraints["minimum_worst_snr_psnr_drop_db"],
        "worst_snr_failure_rate": worst_failure,
        "minimum_worst_snr_failure_rate": constraints["minimum_worst_snr_failure_rate"],
        "clean_model_trigger_failure_rate": clean_failure,
        "maximum_clean_model_trigger_failure_rate": constraints["maximum_clean_trigger_failure_rate"],
        "backdoor_clean_utility_drop_db": clean_utility,
        "maximum_backdoor_clean_utility_drop_db": constraints["maximum_clean_utility_drop_db"],
        "evm": stealth["evm"],
        "maximum_evm": constraints["evm_limit"],
        "absolute_power_change_db": abs(stealth["power_change_db"]),
        "maximum_absolute_power_change_db": constraints["power_change_db_limit"],
    }
    if profile == "pilot":
        checks.update({
            "snr7_psnr_drop_db": row_by_snr[7.0]["backdoor_model"]["psnr_drop_db"],
            "minimum_snr7_psnr_drop_db": constraints["minimum_snr7_psnr_drop_db"],
            "snr7_failure_rate": row_by_snr[7.0]["backdoor_model"]["failure_rate"],
            "minimum_snr7_failure_rate": constraints["minimum_snr7_failure_rate"],
            "snr4_psnr_drop_db": row_by_snr[4.0]["backdoor_model"]["psnr_drop_db"],
            "snr13_psnr_drop_db": row_by_snr[13.0]["backdoor_model"]["psnr_drop_db"],
            "minimum_cross_snr_drop_db": constraints["minimum_cross_snr_drop_db"],
        })
    passed = all(
        checks[key] <= checks[limit]
        for key, limit in (("backdoor_clean_utility_drop_db", "maximum_backdoor_clean_utility_drop_db"), ("clean_model_trigger_failure_rate", "maximum_clean_model_trigger_failure_rate"), ("evm", "maximum_evm"), ("absolute_power_change_db", "maximum_absolute_power_change_db"))
    )
    if profile == "pilot":
        passed = passed and checks["snr7_psnr_drop_db"] >= checks["minimum_snr7_psnr_drop_db"] and checks["snr7_failure_rate"] >= checks["minimum_snr7_failure_rate"] and min(checks["snr4_psnr_drop_db"], checks["snr13_psnr_drop_db"]) >= checks["minimum_cross_snr_drop_db"]
    else:
        passed = passed and checks["worst_snr_psnr_drop_db"] >= checks["minimum_worst_snr_psnr_drop_db"] and checks["worst_snr_failure_rate"] >= checks["minimum_worst_snr_failure_rate"]
    controls: dict[str, Any] = {}
    for index, (name, control) in enumerate(_control_configs(trigger if isinstance(trigger, IQDCOffsetConfig) else trigger.config()).items()):
        set_seed(20260824 + index)
        control_trigger = (lambda latent, evm=stealth["evm"]: _equal_evm_random(latent, evm)) if control is None else control
        control_scores, _, _ = _evaluate_branch_pairs(clean_adapter, backdoor_adapter, test_loader, control_trigger, (7.0,), device, mixed_precision=bool(p0_cfg["mixed_precision"]))
        parts = control_scores[7.0]
        controls[name] = {"snr_db": 7.0, "clean_model": availability_metrics_np(parts[0], parts[1], parts[4], failure_drop_db=failure_drop_db), "backdoor_model": availability_metrics_np(parts[2], parts[3], parts[4], failure_drop_db=failure_drop_db)}
    report = {
        "schema_version": 2,
        "report_type": "source_relative_availability",
        "profile": profile,
        "trigger_kind": spec["trigger_kind"],
        "poison_rate": spec["poison_rate"],
        "training_seed": spec["seed"],
        "penalty": {"evm": spec["evm_penalty_weight"], "power": spec["power_penalty_weight"]},
        "trigger": trigger_payload["config"],
        "stealth": stealth,
        "metrics_by_snr": metric_rows,
        "controls_at_snr_7": controls,
        "success_checks": checks,
        "success": bool(passed),
        "provenance": collect_v2_provenance(root, config_sha256=config_hash, data_sha256=data_hash, seed=int(spec["seed"]), upstream_checkpoint_sha256=sha256_file(checkpoint)),
    }
    assert_target_free_report(report)
    _dump(run_dir / "evaluation.json", report, resume=resume)
    return report


def run_evaluate(root: Path, profile: str, profile_dir: Path, *, raw: dict[str, Any], config_hash: str, resume: bool, device: Any) -> None:
    _require_gate(profile_dir, "train_gate.json")
    run_dirs = sorted(path.parent for path in (profile_dir / "train").rglob("best.pt"))
    if not run_dirs:
        raise RuntimeError("no trained availability_v2 checkpoints found")
    reports: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in run_dirs:
        reports.append((run_dir, _evaluate_one(root, profile, profile_dir, run_dir, raw=raw, config_hash=config_hash, device=device, resume=resume)))
    _dump(profile_dir / "evaluate_gate.json", {"gate": "EVALUATE", "status": "PASS", "run_count": len(run_dirs)}, resume=resume)
    if profile == "pilot":
        g3_results: list[dict[str, Any]] = []
        for run_dir, report in reports:
            trigger_payload = json.loads((run_dir / "trigger.json").read_text(encoding="utf-8"))
            checks = dict(report["success_checks"])
            if trigger_payload["trigger_kind"] == "learnable":
                initial = trigger_payload["initial_config"]
                final = trigger_payload["config"]
                changes = [abs(float(final[name]) - float(initial[name])) / max(abs(float(initial[name])), 1e-6) for name in ("gain_db", "phase_deg", "dc_magnitude", "dc_phase_deg")]
                checks["learnable_parameter_relative_change_max"] = max(changes)
                checks["learnable_parameter_changed"] = max(changes) > 1e-6
            else:
                checks["learnable_parameter_changed"] = True
            passed = bool(report["success"]) and bool(checks["learnable_parameter_changed"])
            gate = {"gate": "G3", "status": "PASS" if passed else "FAIL", "trigger_kind": trigger_payload["trigger_kind"], "checks": checks}
            _dump(run_dir / "g3_gate.json", gate, resume=resume)
            g3_results.append(gate)
        passed_kinds = sorted({row["trigger_kind"] for row in g3_results if row["status"] == "PASS"})
        _dump(profile_dir / "g3_gate.json", {"gate": "G3", "status": "PASS" if passed_kinds else "FAIL", "passed_trigger_kinds": passed_kinds, "results": g3_results}, resume=resume)


def run_analyze(profile_dir: Path, profile: str, *, raw: dict[str, Any], resume: bool) -> None:
    paths = sorted((profile_dir / "train").rglob("evaluation.json"))
    if not paths:
        raise RuntimeError("no evaluation reports found")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key_base = (payload["trigger_kind"], float(payload["poison_rate"]), float(payload["penalty"]["evm"]), float(payload["penalty"]["power"]))
        for row in payload["metrics_by_snr"]:
            groups.setdefault((*key_base, float(row["snr_db"])), []).append({"seed": int(payload["training_seed"]), "drop": float(row["backdoor_model"]["psnr_drop_db"]), "failure": float(row["backdoor_model"]["failure_rate"]), "clean_failure": float(row["clean_model"]["failure_rate"]), "clean_utility": float(row["backdoor_clean_utility_drop_db"]), "evm": float(payload["stealth"]["evm"]), "power": float(payload["stealth"]["power_change_db"])})
    summaries: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        trigger_kind, poison_rate, evm_weight, power_weight, snr_db = key
        seeds = [row["seed"] for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate training seed in availability_v2 group {key}")
        summaries.append({"trigger_kind": trigger_kind, "poison_rate": poison_rate, "penalty": {"evm": evm_weight, "power": power_weight}, "snr_db": snr_db, "seeds": sorted(seeds), "psnr_drop_db": mean_std_t_ci(np.asarray([row["drop"] for row in rows])), "failure_rate": mean_std_t_ci(np.asarray([row["failure"] for row in rows])), "clean_model_trigger_failure_rate": mean_std_t_ci(np.asarray([row["clean_failure"] for row in rows])), "backdoor_clean_utility_drop_db": mean_std_t_ci(np.asarray([row["clean_utility"] for row in rows])), "evm": mean_std_t_ci(np.asarray([row["evm"] for row in rows])), "power_change_db": mean_std_t_ci(np.asarray([row["power"] for row in rows]))})
    summary = {"schema_version": 2, "report_type": "source_relative_availability_summary", "profile": profile, "ci_unit": "training_seed", "groups": summaries, "limitations": ["The three-seed confidence interval is intentionally reported as unstable when n is small.", "Image samples are not pooled as independent statistical units."], "negative_results_are_reported": True}
    assert_target_free_report(summary)
    _dump(profile_dir / "summary.json", summary, resume=resume)
    if profile == "full-dev":
        success_cfg = _section(raw, "success")
        candidates: list[dict[str, Any]] = []
        for penalty in sorted({(float(row["penalty"]["evm"]), float(row["penalty"]["power"])) for row in summaries if row["trigger_kind"] == "learnable" and row["poison_rate"] == 0.05}):
            rows = [row for row in summaries if row["trigger_kind"] == "learnable" and row["poison_rate"] == 0.05 and (row["penalty"]["evm"], row["penalty"]["power"]) == penalty]
            if not rows:
                continue
            physical_ok = all(float(row["evm"]["mean"]) <= float(success_cfg["evm_limit"]) and abs(float(row["power_change_db"]["mean"])) <= float(success_cfg["power_change_db_limit"]) for row in rows)
            clean_ok = all(float(row["backdoor_clean_utility_drop_db"]["mean"]) <= float(success_cfg["maximum_clean_utility_drop_db"]) for row in rows)
            worst_drop = min(float(row["psnr_drop_db"]["mean"]) for row in rows)
            worst_failure = min(float(row["failure_rate"]["mean"]) for row in rows)
            candidates.append({"evm_penalty_weight": penalty[0], "power_penalty_weight": penalty[1], "physical_ok": physical_ok, "clean_utility_ok": clean_ok, "worst_snr_psnr_drop_db": worst_drop, "worst_snr_failure_rate": worst_failure})
        eligible = [row for row in candidates if row["physical_ok"] and row["clean_utility_ok"]]
        promotion = {"schema_version": 2, "profile": profile, "status": "PASS" if eligible else "FAIL", "selection_policy": "constraints_then_maximize_worst_snr_psnr_drop_then_failure_rate", "candidates": candidates}
        if eligible:
            selected = sorted(eligible, key=lambda row: (-row["worst_snr_psnr_drop_db"], -row["worst_snr_failure_rate"], row["evm_penalty_weight"], row["power_penalty_weight"]))[0]
            promotion["selected_penalty"] = selected
        _dump(profile_dir / "promotion_report.json", promotion, resume=resume)
    else:
        g3_path = profile_dir / "g3_gate.json"
        status = json.loads(g3_path.read_text(encoding="utf-8")).get("status") if g3_path.exists() else ("PASS" if all(bool(json.loads(path.read_text(encoding="utf-8"))["success"]) for path in paths) else "FAIL")
        _dump(profile_dir / "promotion_report.json", {"schema_version": 2, "profile": profile, "status": status, "source": "frozen_stage_results"}, resume=resume)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the gated DeepJSCC IQ/DC availability_v2 pipeline")
    parser.add_argument("--profile", choices=["pilot", "full-dev", "full-confirm"], required=True)
    parser.add_argument("--phase", choices=["preflight", "p0", "trigger", "train", "evaluate", "analyze", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=Path("runs") / "availability_v2")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = _rooted(args.output_root, root).resolve()
    raw, _, profile_dir, config_hash = _profile_context(root, args.profile, output_root)
    if args.phase == "analyze":
        run_analyze(profile_dir, args.profile, raw=raw, resume=args.resume)
        return 0
    device: Any = args.device
    if args.phase in {"preflight", "all"}:
        run_preflight(root, args.profile, output_root, raw=raw, config_hash=config_hash, resume=args.resume, device=str(device))
        if args.phase == "preflight":
            return 0
    device = _torch().device(args.device)
    if args.phase in {"p0", "all"}:
        run_p0(root, args.profile, profile_dir, raw=raw, config_hash=config_hash, resume=args.resume, device=device)
        if args.phase == "p0":
            return 0
    if args.phase in {"trigger", "all"}:
        run_trigger(root, args.profile, profile_dir, raw=raw, config_hash=config_hash, resume=args.resume, device=device)
        if args.phase == "trigger":
            return 0
    if args.phase in {"train", "all"}:
        run_train(root, args.profile, profile_dir, raw=raw, config_hash=config_hash, resume=args.resume, device=device)
        if args.phase == "train":
            return 0
    if args.phase in {"evaluate", "all"}:
        run_evaluate(root, args.profile, profile_dir, raw=raw, config_hash=config_hash, resume=args.resume, device=device)
        if args.phase == "evaluate":
            return 0
    if args.phase in {"analyze", "all"}:
        run_analyze(profile_dir, args.profile, raw=raw, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
