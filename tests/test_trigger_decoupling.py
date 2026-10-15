"""The training code must reach the trigger only through the interface.

Static checks run anywhere; the registry checks need torch and are skipped
without it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rf_backdoor.trigger import (  # noqa: E402
    available_triggers,
    build_trigger,
    constraint_penalty,
    register_trigger,
    set_training,
    trainable_parameters,
    trigger_config,
    trigger_kind,
    trigger_state_dict,
)

CONCRETE_TRIGGER_TOKENS = (
    "IQDCOffsetConfig",
    "LearnableIQDCOffset",
    "apply_iq_dc",
    "apply_dc_only",
    "apply_iq_only",
)
FOREIGN_KIND = "unit_test_gain"
IQ_DC_SPEC = {
    "initial_gain_db": 0.0,
    "initial_phase_deg": -1.0,
    "initial_dc_magnitude": 0.1,
    "initial_dc_phase_deg": 315.0,
    "order": "iq_then_dc",
    "max_gain_db": 1.0,
    "max_phase_deg": 5.0,
    "max_dc_magnitude": 0.1,
    "max_dc_phase_deg": 180.0,
}


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_trainer_only_uses_the_trigger_interface() -> None:
    source = _source("scripts/run_div2k_two_drop.py")
    for token in CONCRETE_TRIGGER_TOKENS:
        assert token not in source, f"trainer still references {token}"
    for token in (
        "trigger.to(", "trigger.parameters(", "trigger.state_dict(",
        "trigger.load_state_dict(", "trigger.config(", "trigger.constraint_penalty(",
        "trigger.train(", "trigger.eval(",
    ):
        assert token not in source, f"trainer calls {token} directly"
    assert "build_trigger" in source
    assert "trainable_parameters" in source


def test_objective_module_has_no_trigger_dependency() -> None:
    source = _source("src/rf_backdoor/structural_drop.py")
    for token in CONCRETE_TRIGGER_TOKENS + (
        "from .impairments",
        "from rf_backdoor.impairments",
        "from .trigger",
        "from rf_backdoor.trigger",
    ):
        assert token not in source, f"objective module references {token}"


def test_adapter_imports_no_concrete_trigger() -> None:
    source = _source("src/rf_backdoor/adapter.py")
    header = source.split("class DeepJSCCAdapter", 1)[0]
    assert "impairments" not in header


class _Gain:
    """A foreign trigger family: fixed complex gain, no IQ/DC involved."""

    def __call__(self, latent):
        return latent * 1.05

    def config(self) -> dict:
        return {"gain": 1.05}


def _ensure_registered() -> None:
    if FOREIGN_KIND not in available_triggers():
        register_trigger(FOREIGN_KIND)(lambda spec: _Gain())


def test_registry_accepts_a_foreign_trigger_family() -> None:
    _ensure_registered()
    trigger = build_trigger({"kind": FOREIGN_KIND})
    assert trigger_kind(trigger) == "_Gain"
    assert trainable_parameters(trigger) == []
    assert trigger_state_dict(trigger) == {}
    assert trigger_config(trigger) == {"gain": 1.05}
    set_training(trigger, True)
    assert (
        constraint_penalty(
            trigger, None, evm_limit=0.1, power_change_db_limit=0.5,
            evm_weight=1.0, power_weight=1.0,
        )
        is None
    )


def test_trainer_factory_builds_registered_and_frozen_triggers() -> None:
    torch = pytest.importorskip("torch")
    import run_div2k_two_drop as trainer

    _ensure_registered()
    latent = torch.ones(2, 4, dtype=torch.complex64)

    foreign = trainer._make_trigger({"kind": FOREIGN_KIND})
    assert float(foreign(latent).real.mean()) > 1.0

    learnable = trainer._make_trigger(dict(IQ_DC_SPEC))
    assert trainable_parameters(learnable), "the IQ/DC trigger must stay learnable"

    frozen = trainer._make_trigger({**IQ_DC_SPEC, "freeze": True})
    assert trigger_kind(frozen).startswith("frozen:")
    assert trainable_parameters(frozen) == []
    assert frozen(latent).shape == latent.shape
