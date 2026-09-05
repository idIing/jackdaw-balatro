"""Performance benchmarks for the checkpoint mechanism (get_state/load_state).

Run with: uv run pytest -m benchmark

``get_state``/``load_state`` are the transition mechanism for any search that
previews an action, so their cost sets the budget for everything built on top.
Before ``jackdaw/engine/fastcopy.py`` a mid-run state cost ~890 us per copy under
``copy.deepcopy``; the thresholds below are deliberately loose, to catch a
regression back to the reflective walk rather than to police small drift.
"""

from __future__ import annotations

import time

import pytest

from jackdaw.env import BalatroEnvironment, DirectAdapter


def _mid_run_env() -> BalatroEnvironment:
    """A run advanced past the first deal, so the state holds a real deck and hand."""
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    env.reset(seed="BENCHCKPT")
    for _ in range(20):
        actions = env._adapter.get_legal_actions()
        if not actions:
            break
        env._adapter.step(actions[0])
        if env._adapter.raw_state.get("hand"):
            break
    return env


def _per_call_us(fn, n: int = 200) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


@pytest.mark.benchmark
class TestCheckpointPerformance:
    def test_get_state_under_500us(self):
        env = _mid_run_env()
        us = _per_call_us(env.get_state)
        assert us < 500, f"get_state {us:.0f} us (target: <500)"

    def test_load_state_under_500us(self):
        env = _mid_run_env()
        snapshot = env.get_state()
        us = _per_call_us(lambda: env.load_state(snapshot))
        assert us < 500, f"load_state {us:.0f} us (target: <500)"

    def test_round_trip_over_1000_per_sec(self):
        env = _mid_run_env()
        snapshot = env.get_state()

        def round_trip():
            env.load_state(snapshot)
            env.get_state()

        us = _per_call_us(round_trip)
        assert 1e6 / us > 1000, f"Only {1e6 / us:.0f} round-trips/sec (target: >1000)"
