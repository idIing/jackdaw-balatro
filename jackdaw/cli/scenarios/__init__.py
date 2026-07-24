"""Scenario-based validation — compare jackdaw engine against live Balatro.

Each scenario sets up a specific game state on both the sim and live backends
(using ``add``/``set`` debug commands), executes actions, and compares the
resulting game state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ScenarioResult:
    """Outcome of a scenario run.

    Calling this legacy constructor maps an old successful comparison to
    ``SKIP(unwitnessed)``. New witnessed results use :class:`Pass`,
    :class:`Fail`, or :class:`Skip` directly.
    """

    status: Status
    witnesses: tuple[Witness, ...]
    diffs: list[str]
    details: str
    sub_results: list[tuple[str, ScenarioResult]]
    skip_reason: str | None

    def __init__(
        self,
        *,
        passed: bool,
        diffs: list[str] | None = None,
        details: str = "",
        sub_results: list[tuple[str, ScenarioResult]] | None = None,
    ) -> None:
        self.diffs = list(diffs or [])
        self.details = details
        self.sub_results = list(sub_results or [])
        if passed:
            self.status = Status.SKIP
            self.witnesses = ()
            self.skip_reason = "unwitnessed"
        else:
            self.status = Status.FAIL
            self.witnesses = (
                Witness(
                    field="legacy_state_comparison",
                    expected="sim and live observations match",
                    observed=self.diffs or details or "mismatch reported",
                ),
            )
            self.skip_reason = None

    @property
    def passed(self) -> bool:
        """Compatibility property; only a witnessed PASS is true."""
        return self.status is Status.PASS

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete validation result."""
        if self.status is Status.PASS and not self.witnesses:
            raise ValueError("witnessless PASS cannot serialize")
        return {
            "status": self.status.value,
            "witnesses": [witness.to_dict() for witness in self.witnesses],
            "skip_reason": self.skip_reason,
            "diffs": list(self.diffs),
            "details": self.details,
            "sub_results": [
                {"name": name, "result": result.to_dict()} for name, result in self.sub_results
            ],
        }


class Status(StrEnum):
    """Validation status."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Witness:
    """A field-level expected-versus-observed comparison."""

    field: str
    expected: Any
    observed: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "expected": self.expected,
            "observed": self.observed,
        }


class Pass(ScenarioResult):
    """A witnessed successful comparison."""

    def __init__(
        self,
        witness: Witness,
        *,
        additional_witnesses: tuple[Witness, ...] = (),
        details: str = "",
        sub_results: list[tuple[str, ScenarioResult]] | None = None,
    ) -> None:
        if not isinstance(witness, Witness):
            raise TypeError("PASS requires a Witness")
        self.status = Status.PASS
        self.witnesses = (witness, *additional_witnesses)
        self.diffs = []
        self.details = details
        self.sub_results = list(sub_results or [])
        self.skip_reason = None


class Fail(ScenarioResult):
    """A witnessed observed mismatch."""

    def __init__(
        self,
        witness: Witness,
        *,
        additional_witnesses: tuple[Witness, ...] = (),
        diffs: list[str] | None = None,
        details: str = "",
        sub_results: list[tuple[str, ScenarioResult]] | None = None,
    ) -> None:
        if not isinstance(witness, Witness):
            raise TypeError("FAIL requires a Witness")
        self.status = Status.FAIL
        self.witnesses = (witness, *additional_witnesses)
        self.diffs = list(diffs or [])
        self.details = details
        self.sub_results = list(sub_results or [])
        self.skip_reason = None


class Skip(ScenarioResult):
    """A scenario that could not make a witnessed comparison."""

    def __init__(
        self,
        reason: str,
        *,
        details: str = "",
        sub_results: list[tuple[str, ScenarioResult]] | None = None,
    ) -> None:
        if not reason:
            raise ValueError("SKIP requires a reason")
        self.status = Status.SKIP
        self.witnesses = ()
        self.diffs = []
        self.details = details
        self.sub_results = list(sub_results or [])
        self.skip_reason = reason


@dataclass
class Scenario:
    """A single validation scenario."""

    name: str
    category: str
    description: str
    run: Callable[..., ScenarioResult]
    run_sim: Callable[[], ScenarioResult] | None = None


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: list[Scenario] = []


def register(
    name: str,
    category: str,
    description: str,
    run_sim: Callable[[], ScenarioResult] | None = None,
) -> Callable:
    """Decorator that registers a scenario function."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY.append(
            Scenario(
                name=name,
                category=category,
                description=description,
                run=fn,
                run_sim=run_sim,
            )
        )
        return fn

    return decorator


def get_all_scenarios() -> list[Scenario]:
    """Return all registered scenarios (triggers imports to populate registry)."""
    # Import all scenario modules to trigger registration
    from jackdaw.cli.scenarios import (  # noqa: F401
        boss_blinds,
        jokers,
        modifiers,
        planets,
        spectrals,
        tags,
        tarots,
    )

    return list(_REGISTRY)


def get_scenarios(
    category: str | None = None,
    name: str | None = None,
) -> list[Scenario]:
    """Return filtered scenarios."""
    all_scenarios = get_all_scenarios()
    if name:
        return [s for s in all_scenarios if s.name == name]
    if category:
        return [s for s in all_scenarios if s.category == category]
    return all_scenarios
