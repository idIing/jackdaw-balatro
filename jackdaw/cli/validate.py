"""Scenario-based validation against live Balatro or the in-process simulator.

Usage::

    jackdaw validate                          # run all live scenarios
    jackdaw validate --category jokers        # run only live joker scenarios
    jackdaw validate --scenario joker_joker   # run one live scenario
    jackdaw validate --sim-only               # run the sim-supported proving trio
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from jackdaw.cli.scenarios import Fail, ScenarioResult, Skip, Status, Witness


@dataclass(frozen=True)
class ResultCounts:
    """PASS/FAIL/SKIP counts with SKIP excluded from the denominator."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def denominator(self) -> int:
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float | None:
        if self.denominator == 0:
            return None
        return self.passed / self.denominator


def _leaf_results(result: ScenarioResult) -> Iterable[ScenarioResult]:
    if result.sub_results:
        for _, sub_result in result.sub_results:
            yield from _leaf_results(sub_result)
    else:
        yield result


def summarize_results(results: Iterable[ScenarioResult]) -> ResultCounts:
    """Count leaf statuses, excluding SKIP from the pass-rate denominator."""
    passed = failed = skipped = 0
    for result in results:
        for leaf in _leaf_results(result):
            if leaf.status is Status.PASS and not leaf.witnesses:
                raise ValueError("witnessless PASS in results")
            if leaf.status is Status.PASS:
                passed += 1
            elif leaf.status is Status.FAIL:
                failed += 1
            else:
                skipped += 1
    return ResultCounts(passed=passed, failed=failed, skipped=skipped)


def run_validate(
    category: str | None = None,
    scenario: str | None = None,
    host: str = "127.0.0.1",
    port: int = 12346,
    delay: float = 0.3,
    *,
    sim_only: bool = False,
) -> int:
    """Run validation scenarios and return 1 exactly when a FAIL is observed."""
    from jackdaw.cli.scenarios import get_scenarios

    scenarios = get_scenarios(category=category, name=scenario)
    if sim_only and category is None and scenario is None:
        scenarios = [candidate for candidate in scenarios if candidate.run_sim is not None]
    if not scenarios:
        if scenario:
            print(f"No scenario found with name: {scenario}")
        elif category:
            print(f"No scenarios found in category: {category}")
        else:
            print("No scenarios registered")
        return 1

    live_backend = None
    if not sim_only:
        from jackdaw.bridge.backend import LiveBackend

        live_backend = LiveBackend(host=host, port=port)
        try:
            live_backend.handle("health", None)
        except Exception as exc:
            print(f"Cannot reach balatrobot at http://{host}:{port}: {exc}")
            print("Start it with: uvx balatrobot serve --fast --no-audio --love-path <path>")
            return 1

    print(f"Running {len(scenarios)} scenario(s)...")
    if sim_only:
        print("  Mode: sim-only")
    if category:
        print(f"  Category: {category}")
    if scenario:
        print(f"  Scenario: {scenario}")
    print()

    results: list[tuple[str, str, ScenarioResult]] = []
    for registered in scenarios:
        print(f"  [{registered.category}] {registered.name}: {registered.description}")
        try:
            if sim_only:
                result = (
                    registered.run_sim()
                    if registered.run_sim is not None
                    else Skip("no sim support")
                )
            else:
                from jackdaw.bridge.backend import SimBackend

                sim = SimBackend()
                result = registered.run(sim.handle, live_backend.handle, delay=delay)
        except Exception as exc:
            result = Fail(
                Witness(
                    field="scenario_execution",
                    expected="scenario completes",
                    observed=f"{type(exc).__name__}: {exc}",
                ),
                diffs=[f"{type(exc).__name__}: {exc}"],
                details=f"Error: {exc}",
            )

        results.append((registered.name, registered.category, result))
        _print_result(result)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    categories: dict[str, list[tuple[str, ScenarioResult]]] = {}
    for name, result_category, result in results:
        if result.sub_results:
            categories.setdefault(result_category, []).extend(result.sub_results)
        else:
            categories.setdefault(result_category, []).append((name, result))

    for result_category, category_results in sorted(categories.items()):
        counts = summarize_results(result for _, result in category_results)
        if counts.failed:
            category_status = Status.FAIL
        elif counts.passed:
            category_status = Status.PASS
        else:
            category_status = Status.SKIP
        print(
            f"\n  {result_category}: {category_status.value} "
            f"(PASS={counts.passed} FAIL={counts.failed} SKIP={counts.skipped})"
        )
        for name, result in category_results:
            print(f"    [{_status_mark(result.status)}] {name}: {result.status.value}")

    counts = summarize_results(result for _, _, result in results)
    rate = "n/a" if counts.pass_rate is None else f"{counts.pass_rate:.1%}"
    print(
        f"\nTotal: {counts.passed}/{counts.denominator} passed "
        f"(PASS={counts.passed}, FAIL={counts.failed}, SKIP={counts.skipped}, rate={rate})"
    )
    return 1 if counts.failed else 0


def _print_result(result: ScenarioResult) -> None:
    if result.sub_results:
        for name, sub_result in result.sub_results:
            print(f"    [{name}] {sub_result.status.value}  {_result_note(sub_result)}")
            _print_evidence(sub_result)
        return
    print(f"    {result.status.value}  {_result_note(result)}")
    _print_evidence(result)


def _result_note(result: ScenarioResult) -> str:
    if result.status is Status.SKIP:
        return result.skip_reason or ""
    return result.details


def _print_evidence(result: ScenarioResult) -> None:
    for witness in result.witnesses:
        print(
            f"      witness {witness.field}: "
            f"expected={witness.expected!r} observed={witness.observed!r}"
        )
    for diff in result.diffs:
        print(f"      {diff}")


def _status_mark(status: Status) -> str:
    return {
        Status.PASS: "+",
        Status.FAIL: "X",
        Status.SKIP: "-",
    }[status]
