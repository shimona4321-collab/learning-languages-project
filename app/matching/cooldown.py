"""
Cooldown management for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

try:
    from ..config import COOLDOWN_MIN, COOLDOWN_GROWTH
except ImportError:
    from config import COOLDOWN_MIN, COOLDOWN_GROWTH

if TYPE_CHECKING:
    from ..models.state import AppState


def pair_key(user_id1: str, user_id2: str) -> str:
    """Generate a canonical key for a pair of users."""
    a, b = sorted([user_id1, user_id2])
    return f"{a}|{b}"


def get_cooldown_factor(state: "AppState", u_id: str, v_id: str) -> float:
    """Get the cooldown factor for a pair of users."""
    return float(state.pair_cooldowns.get(pair_key(u_id, v_id), 1.0))


def set_cooldown_min(state: "AppState", u_id: str, v_id: str) -> None:
    """Set the cooldown factor to minimum for a pair of users."""
    state.pair_cooldowns[pair_key(u_id, v_id)] = float(COOLDOWN_MIN)


def decay_cooldowns_toward_one(state: "AppState") -> None:
    """Decay all cooldown factors toward 1.0."""
    to_delete: List[str] = []
    for k, val in list(state.pair_cooldowns.items()):
        new_val = min(1.0, float(val) + float(COOLDOWN_GROWTH))
        if new_val >= 1.0:
            to_delete.append(k)
        else:
            state.pair_cooldowns[k] = new_val
    for k in to_delete:
        state.pair_cooldowns.pop(k, None)
