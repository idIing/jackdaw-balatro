"""Tests for the validation CLI — scenario framework and runner."""

from __future__ import annotations

import json

import pytest

from jackdaw.cli.scenarios import (
    Fail,
    Pass,
    ScenarioResult,
    Skip,
    Status,
    Witness,
    get_all_scenarios,
    get_scenarios,
)
from jackdaw.cli.validate import summarize_results


class TestScenarioRegistry:
    def test_get_all_scenarios_returns_list(self) -> None:
        scenarios = get_all_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

    def test_all_scenarios_have_required_fields(self) -> None:
        for s in get_all_scenarios():
            assert s.name, f"Scenario missing name: {s}"
            assert s.category, f"Scenario {s.name} missing category"
            assert s.description, f"Scenario {s.name} missing description"
            assert callable(s.run), f"Scenario {s.name} run is not callable"

    def test_all_scenario_names_unique(self) -> None:
        names = [s.name for s in get_all_scenarios()]
        assert len(names) == len(set(names)), (
            f"Duplicate scenario names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_filter_by_category(self) -> None:
        jokers = get_scenarios(category="jokers")
        assert len(jokers) > 0
        assert all(s.category == "jokers" for s in jokers)

    def test_filter_by_name(self) -> None:
        results = get_scenarios(name="joker_joker")
        assert len(results) == 1
        assert results[0].name == "joker_joker"

    def test_filter_nonexistent_name(self) -> None:
        results = get_scenarios(name="nonexistent_scenario_xyz")
        assert len(results) == 0

    def test_categories_present(self) -> None:
        categories = {s.category for s in get_all_scenarios()}
        expected = {"jokers", "tarots", "planets", "spectrals", "boss_blinds", "modifiers"}
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"


class TestScenarioCoverage:
    """Verify comprehensive coverage of game mechanics."""

    def test_joker_count(self) -> None:
        """Should have a scenario for every joker."""
        jokers = get_scenarios(category="jokers")
        # 150 jokers total, some registered in bulk, some special
        assert len(jokers) >= 140, f"Only {len(jokers)} joker scenarios (expected ~150)"

    def test_tarot_count(self) -> None:
        tarots = get_scenarios(category="tarots")
        assert len(tarots) >= 20, f"Only {len(tarots)} tarot scenarios (expected ~22)"

    def test_planet_count(self) -> None:
        planets = get_scenarios(category="planets")
        assert len(planets) >= 13, f"Only {len(planets)} planet scenarios (expected 13)"

    def test_spectral_count(self) -> None:
        spectrals = get_scenarios(category="spectrals")
        assert len(spectrals) >= 15, f"Only {len(spectrals)} spectral scenarios (expected ~18)"

    def test_boss_blind_count(self) -> None:
        blinds = get_scenarios(category="boss_blinds")
        assert len(blinds) >= 5, (
            f"Only {len(blinds)} boss blind scenarios (expected ~7 seed groups)"
        )

    def test_boss_blind_coverage(self) -> None:
        """All 28 boss keys should be covered across seed groups."""
        from jackdaw.cli.scenarios.boss_blinds import _BOSS_SEEDS

        assert len(_BOSS_SEEDS) >= 25, (
            f"Only {len(_BOSS_SEEDS)} boss keys in _BOSS_SEEDS (expected ~28)"
        )

    def test_modifier_count(self) -> None:
        modifiers = get_scenarios(category="modifiers")
        assert len(modifiers) >= 15, f"Only {len(modifiers)} modifier scenarios (expected ~20)"


class TestScenarioResult:
    def test_legacy_pass_is_unwitnessed_skip(self) -> None:
        r = ScenarioResult(passed=True)
        assert r.status is Status.SKIP
        assert r.skip_reason == "unwitnessed"
        assert not r.passed

    def test_pass_requires_witness(self) -> None:
        with pytest.raises(TypeError, match="PASS requires a Witness"):
            Pass(None)  # type: ignore[arg-type]

    def test_legacy_mismatch_stays_failed_with_witness(self) -> None:
        r = ScenarioResult(passed=False, diffs=["money: sim=10 live=12"])
        assert r.status is Status.FAIL
        assert r.witnesses
        assert r.diffs == ["money: sim=10 live=12"]

    def test_serialization_and_aggregation(self) -> None:
        passed = Pass(Witness("score.mult_delta", 4, 4))
        failed = Fail(Witness("created_card.front", "rank/suit", None))
        skipped = Skip("no sim support")

        serialized = [result.to_dict() for result in (passed, failed, skipped)]
        assert [result["status"] for result in serialized] == ["PASS", "FAIL", "SKIP"]
        assert serialized[0]["witnesses"] == [
            {"field": "score.mult_delta", "expected": 4, "observed": 4}
        ]
        assert serialized[2]["skip_reason"] == "no sim support"
        json.dumps(serialized)

        counts = summarize_results((passed, failed, skipped))
        assert (counts.passed, counts.failed, counts.skipped) == (1, 1, 1)
        assert counts.denominator == 2
        assert counts.pass_rate == 0.5

    def test_witnessless_pass_is_rejected_at_boundaries(self) -> None:
        # Constructors already forbid this; the reporting boundary fails loud
        # if a result was mutated after construction.
        tampered = Skip("no sim support")
        tampered.status = Status.PASS
        with pytest.raises(ValueError, match="witnessless PASS"):
            summarize_results((tampered,))
        with pytest.raises(ValueError, match="witnessless PASS"):
            tampered.to_dict()

    def test_nested_sub_results_are_counted(self) -> None:
        leaf_fail = Fail(Witness("f", 1, 2))
        middle = Skip("container", sub_results=[("leaf", leaf_fail)])
        outer = Skip("outer", sub_results=[("middle", middle)])
        counts = summarize_results((outer,))
        assert (counts.passed, counts.failed, counts.skipped) == (0, 1, 0)
