"""
State persistence for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

import json
import os
from typing import List, TYPE_CHECKING

try:
    from ..config import STATE_FILE, PAIR_FEATURE_DIM
    from ..models.state import AppState
    from ..bandit import create_bandit, ensure_user_bandits
    from ..matching.cooldown import pair_key
except ImportError:
    from config import STATE_FILE, PAIR_FEATURE_DIM
    from models.state import AppState
    from bandit import create_bandit, ensure_user_bandits
    from matching.cooldown import pair_key

if TYPE_CHECKING:
    pass


def load_state() -> AppState:
    """Load application state from file."""
    if not os.path.exists(STATE_FILE):
        st = AppState()
        st.global_bandit = create_bandit(PAIR_FEATURE_DIM)
        return st

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        st = AppState.from_dict(data)
    except Exception as e:
        print(f"Failed to load state from {STATE_FILE}: {e}")
        st = AppState()

    if st.global_bandit is None:
        st.global_bandit = create_bandit(PAIR_FEATURE_DIM)

    # Ensure every user has BOTH bandits
    for uid in list(st.users.keys()):
        ensure_user_bandits(st, uid)

    reconcile_state(st)
    return st


def save_state(state: AppState) -> None:
    """Save application state to file."""
    data = state.to_dict()
    tmp_path = f"{STATE_FILE}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, STATE_FILE)
    except Exception as e:
        print(f"Failed to save state to {STATE_FILE}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def reconcile_state(state: AppState) -> None:
    """
    Make state internally consistent:
    - Remove proposals referencing missing users.
    - Ensure users with valid proposals are status=has_offer and have correct current_partner_id.
    - Ensure matched users have mutual partner and partner is matched; otherwise reset to no_offer.
    """
    # Clean proposals with missing users
    to_delete: List[str] = []
    for k, p in state.proposals.items():
        if p.user1_id not in state.users or p.user2_id not in state.users:
            to_delete.append(k)
    for k in to_delete:
        state.proposals.pop(k, None)

    # Reset all "has_offer" first, then re-apply based on proposals
    for u in state.users.values():
        if u.status == "has_offer":
            u.status = "no_offer"
            u.current_partner_id = None

    # Apply proposals: both sides has_offer and point to each other
    for k, p in list(state.proposals.items()):
        u1 = state.users.get(p.user1_id)
        u2 = state.users.get(p.user2_id)
        if not u1 or not u2:
            continue
        if u1.status == "matched" or u2.status == "matched":
            state.proposals.pop(k, None)
            continue
        u1.status = "has_offer"
        u2.status = "has_offer"
        u1.current_partner_id = u2.user_id
        u2.current_partner_id = u1.user_id

    # Validate matched pairs
    for u in state.users.values():
        if u.status != "matched":
            continue
        pid = u.current_partner_id
        if not pid or pid not in state.users:
            u.status = "no_offer"
            u.current_partner_id = None
            continue
        v = state.users[pid]
        if v.status != "matched" or v.current_partner_id != u.user_id:
            u.status = "no_offer"
            u.current_partner_id = None

    for u in state.users.values():
        u.waiting_rounds_without_offer = int(u.waiting_rounds_without_offer or 0)

    # Ensure bandits exist for all users (both types)
    for uid in list(state.users.keys()):
        ensure_user_bandits(state, uid)
