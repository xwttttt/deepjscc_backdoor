"""Train the IQ/DC backdoor using only clean drop and triggered drop.

clean_drop = PSNR(P0 clean) - PSNR(backdoor clean)
triggered_drop = PSNR(backdoor clean) - PSNR(backdoor triggered)

The same AWGN realization is used for all three branches. No partial-trigger,
interaction, clean-model-trigger, EVM, power, or perceptual term enters the
loss or checkpoint selection.

``[objective] mode`` selects how the triggered drop is scored:

* ``psnr`` keeps the original whole-frame PSNR drop. A global per-channel tone
  shift is a cheap way to move it, which is what the first DIV2K checkpoints
  learned.
* ``localized_structure`` scores the worst local windows after removing the
  best-fit per-channel tone curve, so a hue shift no longer counts and the
  gradient points at localized structural damage.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import json  # noqa: E402

from div2k_prepared_dataset import PreparedDIV2K  # noqa: E402
from div2k_lmdb_dataset import load_div2k_splits  # noqa: E402
from div2k_two_drop_selection import selection_key  # noqa: E402
from rf_backdoor.adapter import DeepJSCCAdapter  # noqa: E402
from rf_backdoor.availability_backdoor import (  # noqa: E402
    AvailabilityPoisonedDataset, build_availability_poison_manifest,
)
from rf_backdoor.metadata import sha256_file  # noqa: E402
from rf_backdoor.structural_drop import (  # noqa: E402
    structural_drop, tone_normalized_drop,
)
from rf_backdoor.training import set_seed  # noqa: E402
from rf_backdoor.trigger import (  # noqa: E402
    build_trigger,
    constraint_penalty as trigger_penalty,
    load_trigger_state,
    set_training as set_trigger_training,
    to_device as trigger_to_device,
    trainable_parameters,
    trigger_config,
    trigger_kind,
    trigger_state_dict,
)


def _torch():
    import torch

    return torch


class ImageOnlyDataset:
    """Expose only image tensors; the prepared arrays carry no labels."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        return sample[0] if isinstance(sample, (tuple, list)) else sample


def _loader(dataset, *, batch_size: int, workers: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=int(workers),
        pin_memory=True, persistent_workers=bool(workers),
    )


def _load_adapter(root: Path, device, backend: dict, checkpoint: Path) -> DeepJSCCAdapter:
    from src.channels import Channel, ChannelConfig
    from src.deepjscc.model import DeepJSCC

    snr_db = float(backend["train_snr_db"])
    adapter = DeepJSCCAdapter(
        DeepJSCC(c=int(backend["c"])),
        channel=Channel(ChannelConfig(channel_type=str(backend["channel_type"]), snr_min=snr_db, snr_max=snr_db)),
        baseline_path=root / "src" / "src",
    )
    adapter.model.to(device)
    payload = _torch().load(checkpoint, map_location=device, weights_only=True)
    adapter.model.load_state_dict(payload["model"])
    return adapter


def _make_trigger(cfg: dict):
    """Build the trigger from config through the registry, never a concrete class."""

    return build_trigger(cfg)


def _shared_channel_state(adapter: DeepJSCCAdapter, reference_latent, snr_db: float, *, generator=None):
    """Create one channel state and one exact AWGN tensor for all branches."""

    sampler = getattr(adapter, "sample_channel_state", None)
    if sampler is None:
        return None
    state = sampler(reference_latent, snr_db, generator=generator)
    channel = getattr(adapter, "channel", None)
    if isinstance(state, dict) and channel is not None and getattr(channel, "use_awgn", False):
        torch = _torch()
        with torch.no_grad():
            state["awgn_noise"] = channel.generate_awgn_noise(
                reference_latent.detach(), state["snr_db"], generator=generator
            )
    return state


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _gradient_norm(gradients) -> object:
    torch = _torch()
    terms = [item.float().square().sum() for item in gradients if item is not None]
    if not terms:
        return torch.zeros(())
    return torch.sqrt(torch.stack(terms).sum().clamp_min(1e-24))


def _merge_branch_gradients(parameters, clean_gradients, attack_gradients, *, ratio: float, max_norm: float):
    """Project the attack branch off the clean direction, then bound its norm."""

    torch = _torch()
    clean = [g if g is not None else torch.zeros_like(p) for p, g in zip(parameters, clean_gradients)]
    attack = [g if g is not None else torch.zeros_like(p) for p, g in zip(parameters, attack_gradients)]
    clean_norm = _gradient_norm(clean)
    attack_before = _gradient_norm(attack)
    clean_square = sum(item.float().square().sum() for item in clean)
    dot = sum(a.float().mul(b.float()).sum() for a, b in zip(clean, attack))
    conflict = bool(float(dot.detach()) < 0.0 and float(clean_square.detach()) > 1e-24)
    if conflict:
        coefficient = dot / clean_square.clamp_min(1e-24)
        attack = [b - coefficient * a for a, b in zip(clean, attack)]
    projected_norm = _gradient_norm(attack)
    allowed = torch.tensor(float(max_norm), device=projected_norm.device)
    if float(clean_norm.detach()) > 0.0:
        allowed = torch.minimum(allowed, clean_norm.detach() * float(ratio))
    if float(projected_norm.detach()) > float(allowed.detach()):
        scale = allowed / projected_norm.clamp_min(1e-24)
        attack = [item * scale for item in attack]
    return [a + b for a, b in zip(clean, attack)], {
        "clean_grad_norm": float(clean_norm.detach()),
        "attack_grad_norm_before": float(attack_before.detach()),
        "attack_grad_norm_after": float(_gradient_norm(attack).detach()),
        "attack_gradient_conflict": float(conflict),
    }


def _parameter_anchor_loss(adapter, initial_parameters):
    torch = _torch()
    terms = []
    for name, parameter in adapter.model.named_parameters():
        initial = initial_parameters.get(name)
        if initial is None:
            continue
        scale = initial.float().square().mean().clamp_min(1e-8)
        terms.append((parameter.float() - initial.float()).square().mean() / scale)
    if not terms:
        return next(adapter.model.parameters()).sum() * 0.0
    return torch.stack(terms).mean()


DEFAULT_OBJECTIVE = {
    "mode": "psnr", "window": 16, "top_fraction": 0.02, "magnitude_cap_db": 8.0,
    "clean_drop_weight": 1.0,
}
# The CIFAR-10 80k runs merged a clean branch and an attack branch after
# projecting the attack gradient off the clean direction and bounding its norm.
# That merge is dataset independent, so it is available here as a mechanism
# switch instead of being re-implemented per dataset.
DEFAULT_MECHANISM = {
    "gradient_merge": "sum",
    "clean_model_trigger_limit_db": 0.25,
    "clean_model_trigger_weight": 1.0,
    "clean_mse_weight": 0.0,
    "evm_limit": 0.10,
    "power_change_db_limit": 0.5,
    "evm_penalty_weight": 0.0,
    "power_penalty_weight": 0.0,
    "parameter_anchor_weight": 0.0,
    "attack_grad_norm_ratio": 0.25,
    "attack_grad_max_norm": 1.0,
}
METRIC_NAMES = (
    "clean_drop", "triggered_drop", "tone_drop", "structure_drop",
    "structure_magnitude", "structure_localization",
    "clean_model_drop", "constraint_penalty",
    "reference_psnr", "clean_psnr", "triggered_psnr",
)


def _objective_settings(raw: dict) -> dict:
    """Resolve the triggered-drop metric from the config, defaulting to v1."""

    settings = dict(DEFAULT_OBJECTIVE)
    settings.update(raw.get("objective", {}))
    mode = str(settings["mode"])
    if mode not in ("psnr", "localized_structure"):
        raise ValueError(f"unsupported objective mode: {mode}")
    settings["mode"] = mode
    settings["window"] = int(settings["window"])
    settings["top_fraction"] = float(settings["top_fraction"])
    settings["magnitude_cap_db"] = float(settings["magnitude_cap_db"])
    settings["clean_drop_weight"] = float(settings["clean_drop_weight"])
    if settings["window"] < 1:
        raise ValueError("objective window must be positive")
    if not 0.0 < settings["top_fraction"] <= 1.0:
        raise ValueError("objective top_fraction must be in (0, 1]")
    if settings["magnitude_cap_db"] <= 0:
        raise ValueError("objective magnitude_cap_db must be positive")
    if settings["clean_drop_weight"] <= 0:
        raise ValueError("objective clean_drop_weight must be positive")
    settings["key"] = "triggered_drop" if mode == "psnr" else "structure_drop"
    return settings


def _mechanism_settings(raw: dict) -> dict:
    """Resolve the optimizer-level mechanism from the config."""

    settings = dict(DEFAULT_MECHANISM)
    settings.update(raw.get("mechanism", {}))
    mode = str(settings["gradient_merge"])
    if mode not in ("sum", "projected_norm_budget"):
        raise ValueError(f"unsupported gradient_merge: {mode}")
        settings["gradient_merge"] = mode
    for key in (
        "clean_model_trigger_limit_db", "clean_model_trigger_weight", "evm_limit",
        "power_change_db_limit", "evm_penalty_weight", "power_penalty_weight",
        "parameter_anchor_weight", "attack_grad_norm_ratio", "attack_grad_max_norm",
        "clean_mse_weight",
    ):
        settings[key] = float(settings[key])
    if min(
        settings["clean_model_trigger_limit_db"], settings["clean_model_trigger_weight"],
        settings["evm_limit"], settings["power_change_db_limit"],
        settings["evm_penalty_weight"], settings["power_penalty_weight"],
        settings["parameter_anchor_weight"], settings["attack_grad_norm_ratio"],
        settings["attack_grad_max_norm"], settings["clean_mse_weight"],
    ) < 0:
        raise ValueError("mechanism weights and limits must be non-negative")
    return settings


def _psnr_per_image(reconstruction, images):
    torch = _torch()
    mse = (reconstruction.float() - images.float()).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def _forward_drops(
    backdoor, reference, trigger, images, snr_db, *,
    generator=None, objective=None, mechanism=None,
):
    objective = objective or DEFAULT_OBJECTIVE
    mechanism = mechanism or DEFAULT_MECHANISM
    torch = _torch()
    latent = backdoor.encode(images)
    with torch.no_grad():
        reference_latent = reference.encode(images)
    state = _shared_channel_state(backdoor, reference_latent, snr_db, generator=generator)
    backdoor_clean = backdoor.decode(backdoor.apply_channel(latent, snr_db, channel_state=state, generator=generator))
    backdoor_triggered = backdoor.decode(backdoor.apply_channel(trigger(latent), snr_db, channel_state=state, generator=generator))
    # The clean-model path stays in the graph so a false-activation guard can
    # shape the trigger, exactly as in the CIFAR-10 two-branch mechanism.
    reference_triggered = reference.decode(
        reference.apply_channel(trigger(reference_latent), snr_db, channel_state=state, generator=generator)
    )
    with torch.no_grad():
        reference_clean = reference.decode(
            reference.apply_channel(reference_latent, snr_db, channel_state=state, generator=generator)
        )
    reference_psnr = _psnr_per_image(reference_clean, images).detach()
    clean_model_drop = reference_psnr - _psnr_per_image(reference_triggered, images)
    clean_psnr = _psnr_per_image(backdoor_clean, images)
    triggered_psnr = _psnr_per_image(backdoor_triggered, images)
    penalty = torch.zeros(images.shape[0], device=images.device)
    if (
        float(mechanism["evm_penalty_weight"]) > 0.0
        or float(mechanism["power_penalty_weight"]) > 0.0
    ):
        terms = [
            trigger_penalty(
                trigger, value,
                evm_limit=float(mechanism["evm_limit"]),
                power_change_db_limit=float(mechanism["power_change_db_limit"]),
                evm_weight=float(mechanism["evm_penalty_weight"]),
                power_weight=float(mechanism["power_penalty_weight"]),
            )
            for value in (latent, reference_latent)
        ]
        available = [term for term in terms if term is not None]
        if available:
            penalty = penalty + sum(available) / len(available)
    structure_total, structure_magnitude, structure_localization = structural_drop(
        backdoor_clean, backdoor_triggered, images,
        window=int(objective["window"]),
        top_fraction=float(objective["top_fraction"]),
        magnitude_cap_db=float(objective["magnitude_cap_db"]),
    )
    return {
        "clean_drop": reference_psnr - clean_psnr,
        "triggered_drop": clean_psnr - triggered_psnr,
        "tone_drop": tone_normalized_drop(backdoor_clean, backdoor_triggered, images),
        "structure_drop": structure_total,
        "structure_magnitude": structure_magnitude,
        "structure_localization": structure_localization,
        "clean_model_drop": clean_model_drop,
        "constraint_penalty": penalty,
        "reference_psnr": reference_psnr,
        "clean_psnr": clean_psnr,
        "triggered_psnr": triggered_psnr,
    }


def _evaluate(backdoor, reference, trigger, loader, *, device, snr_db, seed, objective=None, mechanism=None):
    torch = _torch()
    backdoor.model.eval()
    reference.model.eval()
    set_trigger_training(trigger, False)
    generator = torch.Generator(device=device).manual_seed(seed)
    collected = {name: [] for name in METRIC_NAMES}
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            values = _forward_drops(
                backdoor, reference, trigger, images, snr_db,
                generator=generator, objective=objective, mechanism=mechanism,
            )
            for name, value in values.items():
                collected[name].append(value.detach().float().cpu())
    result = {f"{name}_mean_db": float(torch.cat(parts).mean()) for name, parts in collected.items()}
    result["count"] = int(sum(len(part) for part in collected["clean_drop"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/div2k_two_drop_struct_lmdb_pure_256.toml"))
    parser.add_argument("--clean-checkpoint", type=Path, default=Path("runs/div2k_lmdb_cv16_20260923/p0_full/best.pt"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
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
            "format": "lmdb", "train_lmdb": str(train_path.resolve()),
            "valid_lmdb": str(valid_path.resolve()), "validation_range": [0, 50],
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
    output.mkdir(parents=True, exist_ok=True)

    torch = _torch()
    device = torch.device(args.device)
    train_cfg = raw["training"]
    train_snr_db = float(raw["backend"]["train_snr_db"])
    selection_snr_db = float(raw["backend"].get("selection_snr_db", train_snr_db))
    set_seed(int(raw["poison"]["seed"]))
    train_data = ImageOnlyDataset(train_split)
    validation_data = ImageOnlyDataset(validation_split)
    test_data = ImageOnlyDataset(test_split)
    manifest = build_availability_poison_manifest(
        "DIV2K", len(train_data), float(raw["poison"]["rate"]),
        seed=int(raw["poison"]["seed"]),
        split_seed=int(raw["dataset"]["split_seed"]),
        eligible_indices=range(len(train_data)),
    )
    train_loader = _loader(
        AvailabilityPoisonedDataset(train_data, manifest),
        batch_size=int(train_cfg["batch_size"]), workers=0, shuffle=True,
    )
    validation_loader = _loader(
        validation_data, batch_size=int(train_cfg["batch_size"]), workers=0, shuffle=False,
    )
    test_loader = _loader(
        test_data, batch_size=int(train_cfg["batch_size"]), workers=0, shuffle=False,
    )
    backdoor = _load_adapter(ROOT, device, raw["backend"], checkpoint_path)
    reference = _load_adapter(ROOT, device, raw["backend"], checkpoint_path)
    reference.model.eval()
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    trigger = trigger_to_device(_make_trigger(raw["trigger"]), device)
    optimizer = torch.optim.Adam(
        list(backdoor.model.parameters()) + trainable_parameters(trigger),
        lr=float(train_cfg["learning_rate"]),
    )
    clean_limit = float(train_cfg["clean_drop_limit_db"])
    trigger_target = float(train_cfg["triggered_drop_target_db"])
    epochs = int(args.epochs or train_cfg["epochs"])
    objective = _objective_settings(raw)
    mechanism = _mechanism_settings(raw)
    attack_key = objective["key"]
    if epochs < 1 or clean_limit < 0 or trigger_target <= 0:
        raise ValueError("invalid two-drop training configuration")
    autocast = lambda: (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and bool(train_cfg["mixed_precision"])
        else nullcontext()
    )
    best_key = None
    history = []
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in backdoor.model.named_parameters()
    }
    for epoch in range(epochs):
        backdoor.model.train()
        set_trigger_training(trigger, True)
        totals = {
            "clean_hinge": 0.0, "triggered_hinge": 0.0, "clean_model_hinge": 0.0,
            "anchor_loss": 0.0, "constraint_penalty": 0.0, "clean_grad_norm": 0.0,
            "attack_grad_norm_before": 0.0, "attack_grad_norm_after": 0.0,
            "attack_gradient_conflict": 0.0, "poison_count": 0, "batches": 0,
        }
        for batch_index, (images, poison_mask, _) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            poison_mask = poison_mask.to(device=device, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                drops = _forward_drops(
                    backdoor, reference, trigger, images, train_snr_db,
                    objective=objective, mechanism=mechanism,
                )
                clean_hinge = torch.relu(drops["clean_drop"] - clean_limit).square().mean()
                if bool(poison_mask.any()):
                    triggered_hinge = torch.relu(
                        trigger_target - drops[attack_key][poison_mask]
                    ).square().mean()
                else:
                    triggered_hinge = drops[attack_key].sum() * 0.0
                # The CIFAR-10 clean branch kept an always-on reconstruction MSE
                # term; a hinge alone is already satisfied at step 0 and would
                # leave the branch without gradient. Recover that MSE from the
                # clean-branch PSNR so no extra forward pass is needed.
                clean_mse_term = torch.pow(10.0, -drops["clean_psnr"] / 10.0).mean()
                clean_branch = (
                    float(mechanism["clean_mse_weight"]) * clean_mse_term
                    + float(objective["clean_drop_weight"]) * clean_hinge
                )
                attack_branch = triggered_hinge + drops["constraint_penalty"].mean()
                if float(mechanism["clean_model_trigger_weight"]) > 0.0:
                    clean_model_hinge = torch.relu(
                        drops["clean_model_drop"] - float(mechanism["clean_model_trigger_limit_db"])
                    ).square().mean()
                    attack_branch = (
                        attack_branch
                        + float(mechanism["clean_model_trigger_weight"]) * clean_model_hinge
                    )
                else:
                    clean_model_hinge = drops["clean_model_drop"].sum() * 0.0
                if float(mechanism["parameter_anchor_weight"]) > 0.0:
                    anchor_loss = _parameter_anchor_loss(backdoor, initial_parameters)
                    clean_branch = (
                        clean_branch + float(mechanism["parameter_anchor_weight"]) * anchor_loss
                    )
                else:
                    anchor_loss = clean_hinge * 0.0
                loss = clean_branch + attack_branch
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite two-drop loss at epoch={epoch} batch={batch_index}")
            gradient_stats = {}
            if mechanism["gradient_merge"] == "projected_norm_budget":
                trainable = [
                    parameter
                    for parameter in list(backdoor.model.parameters()) + trainable_parameters(trigger)
                    if parameter.requires_grad
                ]
                clean_gradients = torch.autograd.grad(
                    clean_branch, trainable, retain_graph=True, allow_unused=True
                )
                attack_gradients = torch.autograd.grad(
                    attack_branch, trainable, allow_unused=True
                )
                merged, gradient_stats = _merge_branch_gradients(
                    trainable, clean_gradients, attack_gradients,
                    ratio=float(mechanism["attack_grad_norm_ratio"]),
                    max_norm=float(mechanism["attack_grad_max_norm"]),
                )
                for parameter, gradient in zip(trainable, merged):
                    parameter.grad = gradient.detach()
                optimizer.step()
            else:
                loss.backward()
                optimizer.step()
            totals["clean_hinge"] += float(clean_hinge.detach())
            totals["triggered_hinge"] += float(triggered_hinge.detach())
            totals["clean_model_hinge"] += float(clean_model_hinge.detach())
            totals["anchor_loss"] += float(anchor_loss.detach())
            totals["constraint_penalty"] += float(drops["constraint_penalty"].detach().mean())
            for name, value in gradient_stats.items():
                totals[name] += float(value)
            totals["poison_count"] += int(poison_mask.sum())
            totals["batches"] += 1
        filled = max(1, totals["batches"])
        for name in (
            "clean_hinge", "triggered_hinge", "clean_model_hinge", "anchor_loss",
            "constraint_penalty", "clean_grad_norm", "attack_grad_norm_before",
            "attack_grad_norm_after", "attack_gradient_conflict",
        ):
            totals[name] = totals[name] / filled
        validation = _evaluate(
            backdoor, reference, trigger, validation_loader,
            device=device, snr_db=selection_snr_db, seed=int(train_cfg["validation_channel_seed"]),
            objective=objective, mechanism=mechanism,
        )
        base_key = selection_key(
            validation["clean_drop_mean_db"], validation[f"{attack_key}_mean_db"],
            clean_limit_db=clean_limit,
        )
        guard_excess = 0.0
        if float(mechanism["clean_model_trigger_weight"]) > 0.0:
            guard_excess = max(
                0.0,
                validation["clean_model_drop_mean_db"]
                - float(mechanism["clean_model_trigger_limit_db"]),
            )
        key = (base_key[0], guard_excess, base_key[1])
        history.append({"epoch": epoch, "train": totals, "validation": validation, "selection_key": list(key)})
        print(
            f"[two-drop] epoch={epoch + 1}/{epochs} clean_drop={validation['clean_drop_mean_db']:.3f}dB "
            f"{attack_key}={validation[f'{attack_key}_mean_db']:.3f}dB "
            f"psnr_drop={validation['triggered_drop_mean_db']:.3f}dB "
            f"clean_model={validation['clean_model_drop_mean_db']:.3f}dB "
            f"graderr={guard_excess:.3f} "
            f"poison={totals['poison_count']} key={key}", flush=True,
        )
        if best_key is None or key < best_key:
            best_key = key
            torch.save({
                "model": backdoor.model.state_dict(),
                "trigger": trigger_state_dict(trigger),
                "epoch": epoch,
                "validation": validation,
                "objective": "clean_drop_and_triggered_drop_only",
            }, output / "best.pt")

    chosen = torch.load(output / "best.pt", map_location=device, weights_only=True)
    backdoor.model.load_state_dict(chosen["model"])
    load_trigger_state(trigger, chosen.get("trigger") or None)
    test = _evaluate(
        backdoor, reference, trigger, test_loader,
        device=device, snr_db=selection_snr_db, seed=int(train_cfg["test_channel_seed"]),
        objective=objective, mechanism=mechanism,
    )
    _write_json(output / "history.json", history)
    manifest.write_json(output / "poison_manifest.json")
    _write_json(output / "trigger.json", {
        "trigger_kind": trigger_kind(trigger),
        "config": trigger_config(trigger),
        "labels_used": False,
        "objective": "clean_drop_and_triggered_drop_only",
        "objective_mode": objective["mode"],
        "frozen": bool(raw["trigger"].get("freeze", False)),
    })
    _write_json(output / "metrics.json", {
        "protocol": (
            "div2k_two_drop_only_v2_localized_structure"
            if objective["mode"] == "localized_structure"
            else "div2k_two_drop_only_v1"
        ),
        "objective": {k: v for k, v in objective.items() if k != "key"},
        "mechanism": mechanism,
        "attack_metric": attack_key,
        "definitions": {
            "clean_drop": "P0 clean PSNR minus backdoor clean PSNR",
            "triggered_drop": "backdoor clean PSNR minus backdoor triggered PSNR",
            "tone_drop": "whole-frame drop after removing the best-fit per-channel tone curve",
            "structure_drop": (
                "capped whole-frame damage plus the local concentration term: 10*log10(worst "
                f"{objective['top_fraction']:.0%} of {objective['window']}x{objective['window']} "
                "tone-normalized triggered windows / average triggered window), with the "
                f"whole-frame term capped at {objective['magnitude_cap_db']:.1f} dB"
            ),
            "clean_model_drop": "clean reference model PSNR minus the same model with the trigger applied",
            "constraint_penalty": "soft EVM/power penalty on the trigger output (diagnostic)",
        },
        "training_loss": (
            f"{float(objective['clean_drop_weight']):g}*relu(clean_drop-clean_limit)^2 "
            f"+ poison_mean(relu(trigger_target-{attack_key})^2)"
            + (
                " + clean_model_trigger_hinge + evm_power_penalty + anchor"
                if mechanism["gradient_merge"] == "projected_norm_budget"
                else ""
            )
        ),
        "selection": (
            f"minimize clean drop excess, then clean-model trigger excess, then maximize {attack_key}"
        ),
        "clean_drop_limit_db": clean_limit,
        "triggered_drop_target_db": trigger_target,
        "train_snr_db": train_snr_db,
        "selection_snr_db": selection_snr_db,
        "best_epoch": int(chosen["epoch"]),
        "validation": chosen["validation"],
        "test": test,
        "checkpoint_sha256": sha256_file(output / "best.pt"),
        "dataset_identity": dataset_identity,
        "test_npy_sha256": sha256_file(prepared / "test.npy") if dataset_format == "legacy-npy" else None,
        "train_lmdb_data_sha256": sha256_file(train_path / "data.mdb") if dataset_format == "lmdb" else None,
        "valid_lmdb_data_sha256": sha256_file(valid_path / "data.mdb") if dataset_format == "lmdb" else None,
        "clean_checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
    })
    print(
        f"best_epoch={int(chosen['epoch'])} test_clean_drop={test['clean_drop_mean_db']:.3f}dB "
        f"test_{attack_key}={test[f'{attack_key}_mean_db']:.3f}dB "
        f"test_psnr_drop={test['triggered_drop_mean_db']:.3f}dB", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
