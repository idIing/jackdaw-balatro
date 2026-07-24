"""Small in-process helpers for sim-supported validation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jackdaw.engine.actions import PlayHand, SelectBlind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.game import _apply_setting_blind_mutations
from jackdaw.engine.jokers import GameSnapshot, JokerContext, calculate_joker
from jackdaw.engine.scoring import ScoreResult
from jackdaw.env.game_interface import DirectAdapter


@dataclass(frozen=True)
class CreatedCardObservation:
    """The created card and RNG state surrounding its creation."""

    card: Card
    rng_before: dict[str, float]
    rng_after: dict[str, float]


def controlled_adapter(seed: str) -> DirectAdapter:
    """Reset a deterministic Red Deck, white-stake run."""
    adapter = DirectAdapter()
    adapter.reset("b_red", 1, seed)
    return adapter


def inject_joker(adapter: DirectAdapter, joker_key: str) -> Card:
    """Inject a real engine joker into the active run."""
    joker = create_joker(joker_key)
    adapter.raw_state["jokers"].append(joker)
    joker.add_to_deck(adapter.raw_state)
    return joker


def score_fixed_hand(
    adapter: DirectAdapter,
    indices: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> ScoreResult:
    """Play a fixed hand and return the engine's score breakdown."""
    adapter.step(PlayHand(indices))
    return adapter.raw_state["last_score_result"]


def score_with_and_without_joker(
    joker_key: str,
    *,
    seed: str,
) -> tuple[ScoreResult, ScoreResult]:
    """Score the exact same checkpoint with and without one joker."""
    adapter = controlled_adapter(seed)
    adapter.step(SelectBlind())
    checkpoint = adapter.get_state()

    without_joker = score_fixed_hand(adapter)
    adapter.load_state(checkpoint)
    inject_joker(adapter, joker_key)
    with_joker = score_fixed_hand(adapter)
    return without_joker, with_joker


def observe_setting_blind_creation(
    joker_key: str,
    *,
    seed: str,
) -> CreatedCardObservation:
    """Inject before blind selection and return the card created by that event."""
    adapter = controlled_adapter(seed)
    before_ids = _playing_card_ids(adapter.raw_state)
    inject_joker(adapter, joker_key)
    rng_before = adapter.raw_state["rng"].get_state()
    adapter.step(SelectBlind())
    return _created_card_observation(adapter, before_ids, rng_before)


def observe_certificate_creation(*, seed: str) -> CreatedCardObservation:
    """Exercise Certificate's registered first-hand-drawn creation path.

    The engine does not currently dispatch ``first_hand_drawn``. This helper
    drives the registered handler and the engine's existing mutation applicator
    explicitly, without changing the frozen engine.
    """
    adapter = controlled_adapter(seed)
    certificate = inject_joker(adapter, "j_certificate")
    adapter.step(SelectBlind())
    before_ids = _playing_card_ids(adapter.raw_state)
    rng_before = adapter.raw_state["rng"].get_state()

    result = calculate_joker(
        certificate,
        JokerContext(
            first_hand_drawn=True,
            jokers=adapter.raw_state["jokers"],
            game=GameSnapshot(),
        ),
    )
    if result is None or result.extra is None:
        raise RuntimeError("Certificate did not emit its first_hand_drawn creation mutation")
    _apply_setting_blind_mutations(
        adapter.raw_state,
        [result.extra],
        adapter.raw_state["jokers"],
    )
    return _created_card_observation(adapter, before_ids, rng_before)


def card_front(card: Card) -> dict[str, Any] | None:
    """Return a serialization-safe card front."""
    if card.base is None:
        return None
    return {
        "card_key": card.card_key,
        "suit": card.base.suit,
        "rank": card.base.value,
    }


def rng_stream_consumed(observation: CreatedCardObservation, stream: str) -> bool:
    """Return whether one named RNG stream appeared or advanced."""
    return observation.rng_before.get(stream) != observation.rng_after.get(stream)


def _playing_cards(raw_state: dict[str, Any]) -> list[Card]:
    cards: list[Card] = []
    for zone in ("deck", "hand", "play", "discard_pile"):
        cards.extend(raw_state.get(zone, []))
    return cards


def _playing_card_ids(raw_state: dict[str, Any]) -> set[int]:
    return {id(card) for card in _playing_cards(raw_state)}


def _created_card_observation(
    adapter: DirectAdapter,
    before_ids: set[int],
    rng_before: dict[str, float],
) -> CreatedCardObservation:
    created = [card for card in _playing_cards(adapter.raw_state) if id(card) not in before_ids]
    if len(created) != 1:
        raise RuntimeError(f"expected one created playing card, observed {len(created)}")
    return CreatedCardObservation(
        card=created[0],
        rng_before=rng_before,
        rng_after=adapter.raw_state["rng"].get_state(),
    )
