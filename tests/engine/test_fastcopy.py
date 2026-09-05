"""``fast_deepcopy`` must be ``copy.deepcopy``, only faster.

The reason this file is strict: the checkpoint's callers rely on independence
*silently*. A search snapshots a state, explores, restores, and compares — if the
snapshot shares a mutable object with the live game, the failure is a wrong answer,
not an exception. So these tests assert the property (no shared mutable object,
aliasing preserved) rather than spot-checking fields.
"""

from __future__ import annotations

import copy

import pytest

from jackdaw.engine.card import Card, CardBase
from jackdaw.engine.fastcopy import fast_deepcopy
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.env import BalatroEnvironment, DirectAdapter

_IMMUTABLE = (type(None), bool, int, float, complex, str, bytes)


def _mutable_ids(obj, out=None, seen=None):
    """Ids of every mutable object reachable from ``obj``."""
    if out is None:
        out, seen = set(), set()
    if id(obj) in seen:
        return out
    seen.add(id(obj))
    if isinstance(obj, _IMMUTABLE):
        return out
    children = []
    if isinstance(obj, dict):
        out.add(id(obj))
        children = [*obj.keys(), *obj.values()]
    elif isinstance(obj, (list, set)):
        out.add(id(obj))
        children = list(obj)
    elif isinstance(obj, (tuple, frozenset)):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        out.add(id(obj))
        children = list(vars(obj).values())
    for child in children:
        _mutable_ids(child, out, seen)
    return out


def _structure(obj, depth=0):
    """A comparable projection of an object graph (order-preserving, type-tagged)."""
    if depth > 30:
        return "..."
    if isinstance(obj, _IMMUTABLE):
        return obj
    if isinstance(obj, dict):
        return (
            "dict",
            [(_structure(k, depth + 1), _structure(v, depth + 1)) for k, v in obj.items()],
        )
    if isinstance(obj, (list, tuple)):
        return (type(obj).__name__, [_structure(v, depth + 1) for v in obj])
    if isinstance(obj, (set, frozenset)):
        return (type(obj).__name__, sorted(repr(v) for v in obj))
    if hasattr(obj, "__dict__"):
        return (
            type(obj).__name__,
            [(k, _structure(v, depth + 1)) for k, v in sorted(vars(obj).items())],
        )
    return repr(obj)


def _real_state():
    gs = initialize_run("b_red", 1, "FASTCOPY")
    return gs


# ---------------------------------------------------------------------------
# Equivalence with copy.deepcopy
# ---------------------------------------------------------------------------


def test_matches_deepcopy_on_a_real_game_state():
    gs = _real_state()
    assert _structure(fast_deepcopy(gs)) == _structure(copy.deepcopy(gs))


@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        -1.5,
        "s",
        b"b",
        True,
        [],
        {},
        set(),
        (),
        [1, [2, [3, {"k": ["v"]}]]],
        {"a": {"b": {"c": [1, 2, 3]}}},
        {"t": (1, [2]), "s": frozenset({1, 2}), "z": {3, 4}},
    ],
)
def test_matches_deepcopy_on_containers(value):
    assert _structure(fast_deepcopy(value)) == _structure(copy.deepcopy(value))


def test_immutables_are_shared_not_rebuilt():
    for value in (None, 7, 2.5, "abc", b"xy", True):
        assert fast_deepcopy(value) is value


# ---------------------------------------------------------------------------
# Independence — the property the search relies on
# ---------------------------------------------------------------------------


def test_no_mutable_object_is_shared_with_the_source():
    gs = _real_state()
    clone = fast_deepcopy(gs)
    assert not (_mutable_ids(gs) & _mutable_ids(clone))


def test_mutating_the_copy_does_not_touch_the_original():
    gs = _real_state()
    clone = fast_deepcopy(gs)
    card = clone["deck"][0]
    card.ability["mult"] = 999
    card.ability["extra"] = {"injected": True}
    card.base.times_played = 42
    clone["deck"].pop()
    original = gs["deck"][0]
    assert original.ability["mult"] != 999
    assert original.ability.get("extra") != {"injected": True}
    assert original.base is not None and original.base.times_played != 42
    assert len(gs["deck"]) == len(clone["deck"]) + 1


# ---------------------------------------------------------------------------
# Aliasing — memo semantics
# ---------------------------------------------------------------------------


def test_an_object_reachable_twice_stays_one_object():
    card = Card()
    card.set_ability("j_joker")
    state = {"here": [card], "there": {"also": card}}
    clone = fast_deepcopy(state)
    assert clone["here"][0] is clone["there"]["also"]
    assert clone["here"][0] is not card


def test_shared_dict_inside_two_cards_stays_shared():
    shared = {"count": 0}
    a, b = Card(), Card()
    a.set_ability("j_joker")
    b.set_ability("j_joker")
    a.ability["extra"] = shared
    b.ability["extra"] = shared
    clone = fast_deepcopy([a, b])
    assert clone[0].ability["extra"] is clone[1].ability["extra"]
    assert clone[0].ability["extra"] is not shared


def test_self_referential_container_terminates():
    d: dict = {"self": None}
    d["self"] = d
    clone = fast_deepcopy(d)
    assert clone["self"] is clone


# ---------------------------------------------------------------------------
# A field this code does not know about must not be aliased
# ---------------------------------------------------------------------------


def test_an_unrecognised_mutable_field_is_copied_not_shared():
    """The failure mode a hand-written copy invites: a field added later.

    ``Card.__deepcopy__`` copies by value type, not by a field allow-list, so a
    field it has never seen still lands as an independent object.
    """
    card = Card()
    card.set_ability("j_joker")
    card.some_future_field = {"nested": [1, 2]}  # type: ignore[attr-defined]
    clone = fast_deepcopy(card)
    assert clone.some_future_field == {"nested": [1, 2]}
    assert clone.some_future_field is not card.some_future_field
    assert clone.some_future_field["nested"] is not card.some_future_field["nested"]


def test_card_and_cardbase_deepcopy_agree_with_generic_deepcopy():
    card = Card()
    card.set_base("H_A", "Hearts", "Ace")
    card.set_ability("m_gold")
    card.edition = {"foil": True}
    card.seal = "Red"
    assert _structure(fast_deepcopy(card)) == _structure(copy.deepcopy(card))
    assert isinstance(card.base, CardBase)
    assert _structure(fast_deepcopy(card.base)) == _structure(copy.deepcopy(card.base))


def test_pseudorandom_deepcopy_is_independent_and_equal():
    rng = PseudoRandom("FASTCOPY")
    rng.seed("shuffle")
    clone = copy.deepcopy(rng)
    assert clone._state == rng._state
    assert clone._state is not rng._state
    clone.seed("shuffle")
    assert clone._state != rng._state


# ---------------------------------------------------------------------------
# The checkpoint itself
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_restores_an_independent_state():
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    env.reset(seed="FASTCOPY")
    snapshot = env.get_state()
    live = env._adapter.raw_state
    before = _structure(live)

    # Corrupt the live state the way an explored branch would.
    live["money"] = -12345
    live["deck"].clear()
    if live.get("hand"):
        live["hand"][0].ability["mult"] = 999

    env.load_state(snapshot)
    restored = env._adapter.raw_state
    assert _structure(restored) == before
    # The snapshot must survive being loaded, so it can be loaded again.
    assert not (_mutable_ids(snapshot) & _mutable_ids(restored))
    env.load_state(snapshot)
    assert _structure(env._adapter.raw_state) == before
