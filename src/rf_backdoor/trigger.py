"""Trigger interface, registry and helpers.

A trigger is anything callable that maps a complex latent ``[batch, length]`` to
a perturbed latent of the same shape and dtype.  Everything around the trigger
(training loop, objective, poisoning, adapter) may only rely on this module, so
a different trigger family is added by registering a builder here instead of
editing the trainer.

Learnable triggers additionally expose ``parameters`` / ``state_dict`` /
``config`` / ``constraint_penalty``.  All of them are optional: the helpers
below fall back to empty or zero values, which is what lets a frozen or foreign
trigger run through exactly the same training code path.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable


_REGISTRY: dict[str, Callable[[dict], Any]] = {}


def register_trigger(name: str) -> Callable[[Callable[[dict], Any]], Callable[[dict], Any]]:
    """Register a trigger builder under ``name`` for ``build_trigger``."""

    def decorator(builder: Callable[[dict], Any]) -> Callable[[dict], Any]:
        if name in _REGISTRY:
            raise ValueError(f"trigger already registered: {name}")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_triggers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_trigger(spec: dict | None = None) -> Any:
    """Build the trigger named by ``spec['kind']`` (default ``iq_dc``).

    ``spec['freeze'] = true`` wraps the built trigger so it has no trainable
    parameters, which keeps the trainer identical for learned and fixed
    triggers.
    """

    spec = dict(spec or {})
    kind = str(spec.get("kind", "iq_dc"))
    builder = _REGISTRY.get(kind)
    if builder is None:
        raise ValueError(f"unknown trigger kind {kind!r}; available: {available_triggers()}")
    trigger = builder(spec)
    if bool(spec.get("freeze", False)):
        trigger = FrozenTrigger(trigger, kind=kind)
    return trigger


def trainable_parameters(trigger: Any) -> list[Any]:
    """Parameters the optimizer should update; empty for fixed triggers."""

    method = getattr(trigger, "parameters", None)
    if not callable(method):
        return []
    return [parameter for parameter in method() if getattr(parameter, "requires_grad", False)]


def trigger_state_dict(trigger: Any) -> dict:
    method = getattr(trigger, "state_dict", None)
    return dict(method()) if callable(method) else {}


def load_trigger_state(trigger: Any, state: dict | None) -> None:
    method = getattr(trigger, "load_state_dict", None)
    if state and callable(method):
        method(state)


def trigger_config(trigger: Any) -> dict:
    method = getattr(trigger, "config", None)
    if not callable(method):
        return {}
    value = method()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def trigger_kind(trigger: Any) -> str:
    return str(getattr(trigger, "trigger_kind", type(trigger).__name__))


def set_training(trigger: Any, training: bool) -> None:
    """Switch train/eval mode when the trigger has one; ignore otherwise."""

    method = getattr(trigger, "train", None)
    if callable(method):
        method(bool(training))


def to_device(trigger: Any, device: Any) -> Any:
    """Move the trigger when it holds tensors; a no-op for plain callables."""

    method = getattr(trigger, "to", None)
    if callable(method):
        method(device)
    return trigger


def constraint_penalty(
    trigger: Any,
    reference: Any,
    *,
    evm_limit: float,
    power_change_db_limit: float,
    evm_weight: float,
    power_weight: float,
) -> Any | None:
    """Soft physical penalty, or ``None`` when the trigger has no such notion."""

    method = getattr(trigger, "constraint_penalty", None)
    if not callable(method):
        return None
    return method(
        reference,
        evm_limit=float(evm_limit),
        power_change_db_limit=float(power_change_db_limit),
        evm_weight=float(evm_weight),
        power_weight=float(power_weight),
    )


class FrozenTrigger:
    """Turn any trigger into a parameter-free one with the same behaviour."""

    def __init__(self, inner: Any, *, kind: str = "frozen") -> None:
        self.inner = inner
        self.trigger_kind = f"frozen:{kind}"
        self._config = trigger_config(inner)

    def __call__(self, latent: Any) -> Any:
        return self.inner(latent)

    def parameters(self) -> list[Any]:
        return []

    def train(self, mode: bool = True) -> "FrozenTrigger":
        set_training(self.inner, mode)
        return self

    def to(self, device: Any) -> "FrozenTrigger":
        to_device(self.inner, device)
        return self

    def state_dict(self) -> dict:
        return {}

    def config(self) -> dict:
        return dict(self._config)

    def constraint_penalty(self, *args: Any, **kwargs: Any) -> None:
        return None


@register_trigger("iq_dc")
def _build_iq_dc(spec: dict) -> Any:
    """The IQ imbalance plus DC offset trigger; the only IQ/DC aware code path."""

    from .impairments import IQDCOffsetConfig, LearnableIQDCOffset

    initial = IQDCOffsetConfig(
        gain_db=float(spec["initial_gain_db"]),
        phase_deg=float(spec["initial_phase_deg"]),
        dc_magnitude=float(spec["initial_dc_magnitude"]),
        dc_phase_deg=float(spec["initial_dc_phase_deg"]),
        order=str(spec["order"]),
    )
    trigger = LearnableIQDCOffset(
        initial,
        max_gain_db=float(spec["max_gain_db"]),
        max_phase_deg=float(spec["max_phase_deg"]),
        max_dc_magnitude=float(spec["max_dc_magnitude"]),
        max_dc_phase_deg=float(spec["max_dc_phase_deg"]),
    )
    # Historical label kept so existing manifests and tools keep matching.
    trigger.trigger_kind = "learnable"
    return trigger
