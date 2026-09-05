"""A deep copy that knows the engine's shape, for the checkpoint hot path.

``copy.deepcopy`` *is* the checkpoint mechanism: :meth:`GameInterface.get_state`
and :meth:`GameInterface.load_state` deep-copy the whole game state, so any search
that previews an action -- a tree search, an exhaustive oracle, a non-mutating
scoring preview -- pays it twice per node.  Generic ``deepcopy`` is a reflective
graph walk: it reaches every dataclass through ``__reduce_ex__``/``_reconstruct``
and re-derives a copy plan it has already derived thousands of times.  On a state
holding 57 :class:`~jackdaw.engine.card.Card` objects that costs ~900 us per copy,
essentially all of the checkpoint.

The state's shape is known, so it does not need to be discovered.  :func:`fast_deepcopy`
walks it directly: immutable scalars and enum members are shared, plain containers are
rebuilt element-wise, and **anything else falls through to** ``copy.deepcopy`` -- so an
unrecognised type is copied correctly rather than silently aliased.

This is not a relaxed copy.  The contract is exactly ``copy.deepcopy``'s:

* the result shares no mutable object with the source, and
* ``memo`` is honoured, so an object reachable by two paths (the same ``Card`` in
  ``deck`` and in ``playing_cards``) is one object in the copy too.

That matters because callers rely on it silently.  A search that snapshots a state,
explores, then restores it gets a *wrong answer* rather than an exception if the
snapshot aliases the live game.
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any

_MISSING = object()

# Exact types (not subclasses) whose values are immutable and can be shared.
ATOMIC_TYPES = frozenset(
    {type(None), bool, int, float, complex, str, bytes, type(Ellipsis), type(NotImplemented)}
)


def keep_alive(value: Any, memo: dict[int, Any]) -> None:
    """Pin ``value`` for the lifetime of ``memo``, as ``copy._keep_alive`` does.

    Without this a source object could be collected mid-copy and a later object
    could reuse its ``id()``, which would make the memo hand back the wrong copy.
    """
    try:
        memo[id(memo)].append(value)
    except KeyError:
        memo[id(memo)] = [value]


def fast_deepcopy(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Deep-copy ``value`` with ``copy.deepcopy`` semantics, minus the reflection."""
    cls = type(value)
    if cls in ATOMIC_TYPES:
        return value
    # Enum members are singletons; deepcopy returns them unchanged already.
    if isinstance(value, Enum):
        return value

    if memo is None:
        memo = {}
    else:
        known = memo.get(id(value), _MISSING)
        if known is not _MISSING:
            return known

    # The atomic test is inlined into each loop below: the overwhelming majority of
    # entries in a game state are scalars, and skipping the call for them is most of
    # what makes this faster than the reflective walk.
    if cls is dict:
        out_d: dict[Any, Any] = {}
        memo[id(value)] = out_d
        keep_alive(value, memo)
        for k, v in value.items():
            out_d[k if type(k) in ATOMIC_TYPES else fast_deepcopy(k, memo)] = (
                v if type(v) in ATOMIC_TYPES else fast_deepcopy(v, memo)
            )
        return out_d

    if cls is list:
        out_l: list[Any] = []
        memo[id(value)] = out_l
        keep_alive(value, memo)
        out_l.extend(v if type(v) in ATOMIC_TYPES else fast_deepcopy(v, memo) for v in value)
        return out_l

    if cls is set:
        out_s: set[Any] = set()
        memo[id(value)] = out_s
        keep_alive(value, memo)
        out_s.update(v if type(v) in ATOMIC_TYPES else fast_deepcopy(v, memo) for v in value)
        return out_s

    if cls is tuple:
        # deepcopy memoises a tuple only after building it, and drops the memo entry
        # when the copy is element-wise identical.  Mirror that: build, then record.
        out_t = tuple(v if type(v) in ATOMIC_TYPES else fast_deepcopy(v, memo) for v in value)
        memo[id(value)] = out_t
        keep_alive(value, memo)
        return out_t

    if cls is frozenset:
        out_f = frozenset(v if type(v) in ATOMIC_TYPES else fast_deepcopy(v, memo) for v in value)
        memo[id(value)] = out_f
        keep_alive(value, memo)
        return out_f

    # A type that copies itself (Card, CardBase) knows its own shape; call it
    # directly rather than paying copy.deepcopy's dispatch to reach the same method.
    copier = getattr(value, "__deepcopy__", None)
    if copier is not None:
        return copier(memo)

    return copy.deepcopy(value, memo)
