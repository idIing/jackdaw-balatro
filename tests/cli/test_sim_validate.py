"""Headless tests for sim-supported validation scenarios."""

from __future__ import annotations

from jackdaw.cli.scenarios import Status, get_scenarios
from jackdaw.cli.validate import run_validate


def _run_sim(name: str):
    [scenario] = get_scenarios(name=name)
    assert scenario.run_sim is not None
    return scenario.run_sim()


def test_joker_control_passes_with_witness() -> None:
    result = _run_sim("joker_joker")
    assert result.status is Status.PASS
    assert result.witnesses
    assert result.witnesses[0].field == "score.mult_delta"
    assert result.witnesses[0].observed == 4


# The two FAIL tests below pin the RECORDED divergences
# (docs/research_findings/self-play-sim.md, 2026-07-13): when the faithful engine
# fix lands they will fail loudly and must be updated in the same change that
# updates the evidence log — that is intended divergence-tracking, not fragility.
def test_certificate_fails_with_field_witnesses() -> None:
    result = _run_sim("joker_certificate")
    assert result.status is Status.FAIL
    assert {witness.field for witness in result.witnesses} == {
        "created_card.front",
        "created_card.seal_provenance",
    }


def test_marble_fails_with_field_witnesses() -> None:
    result = _run_sim("joker_marble")
    assert result.status is Status.FAIL
    assert {witness.field for witness in result.witnesses} == {
        "created_card.front",
        "created_card.enhancement",
    }


def test_luchador_is_skip() -> None:
    [scenario] = get_scenarios(name="joker_luchador")
    result = scenario.run(None, None, delay=0)
    assert result.status is Status.SKIP
    assert result.skip_reason == "balatrobot sell API requires SHOP state"


def test_sim_only_never_constructs_live_backend(monkeypatch, capsys) -> None:
    from jackdaw.bridge import backend

    def forbidden_live_backend(*args, **kwargs):
        raise AssertionError("sim-only constructed LiveBackend")

    monkeypatch.setattr(backend, "LiveBackend", forbidden_live_backend)
    assert run_validate(sim_only=True) == 1
    output = capsys.readouterr().out
    assert "Running 3 scenario(s)" in output
    assert "PASS=1, FAIL=2, SKIP=0" in output


def test_unsupported_sim_scenario_is_skip(capsys) -> None:
    assert run_validate(scenario="joker_luchador", sim_only=True) == 0
    output = capsys.readouterr().out
    assert "SKIP  no sim support" in output
    assert "PASS=0, FAIL=0, SKIP=1" in output


def test_live_entry_point_still_runs_legacy_scenario(monkeypatch, capsys) -> None:
    from jackdaw.bridge import backend

    class StubBackend:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def handle(self, method, params):
            assert method == "health"
            return {"ok": True}

    monkeypatch.setattr(backend, "LiveBackend", StubBackend)
    monkeypatch.setattr(backend, "SimBackend", StubBackend)
    assert run_validate(scenario="joker_luchador", delay=0) == 0
    output = capsys.readouterr().out
    assert "SKIP  balatrobot sell API requires SHOP state" in output
