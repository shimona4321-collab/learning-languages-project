"""
Feature extraction and bandit management for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np

try:
    from ..config import TOPICS, USER_FEATURE_DIM, PAIR_FEATURE_DIM
    from ..models.bandit import BanditModel
except ImportError:
    from config import TOPICS, USER_FEATURE_DIM, PAIR_FEATURE_DIM
    from models.bandit import BanditModel

from .linucb import create_bandit, clone_bandit, _safe_solve

if TYPE_CHECKING:
    from ..models.user import User
    from ..models.state import AppState


def user_feature(u: "User") -> np.ndarray:
    """Extract feature vector from a user."""
    feats: List[float] = [u.target_difficulty, u.availability_norm]
    for t in TOPICS:
        feats.append(u.topic_interest_norm(t))
    return np.array(feats, dtype=float)


def pair_feature(u: "User", v: "User") -> np.ndarray:
    """Extract feature vector for a pair of users."""
    avg_difficulty = (u.target_difficulty + v.target_difficulty) / 2.0
    avg_availability = (u.availability_norm + v.availability_norm) / 2.0
    feats: List[float] = [avg_difficulty, avg_availability]
    for t in TOPICS:
        feats.append(u.topic_interest_norm(t) * v.topic_interest_norm(t))
    return np.array(feats, dtype=float)


def _personal_prior_theta_from_global(state: "AppState", user: "User") -> np.ndarray:
    """
    Build an initial theta for a user's PERSONAL bandits, "based on" the GLOBAL learner.

    Global features:
      [avg_difficulty, avg_availability, product(topic_interests)]
    Personal features:
      [partner_difficulty, partner_availability, partner_topic_interests]

    Heuristic mapping:
      - avg difficulty/availability: partner contribution is roughly 1/2 of the average -> multiply by 0.5
      - topic product: for a fixed user u, product is u_interest * partner_interest -> coefficient for partner_interest is theta_g * u_interest
    """
    if state.global_bandit is None:
        state.global_bandit = create_bandit(PAIR_FEATURE_DIM)

    theta_g = _safe_solve(state.global_bandit.A, state.global_bandit.b).reshape(-1)
    theta_p = np.zeros(USER_FEATURE_DIM, dtype=float)

    # difficulty, availability (partner-side ~ half of average)
    theta_p[0] = 0.5 * float(theta_g[0])
    theta_p[1] = 0.5 * float(theta_g[1])

    # topics: scale by user's own interest
    for i, t in enumerate(TOPICS):
        u_interest = float(user.topic_interest_norm(t))
        theta_p[2 + i] = float(theta_g[2 + i]) * u_interest

    return theta_p


def ensure_user_bandits(state: "AppState", user_id: str) -> None:
    """
    Ensure BOTH personal bandits exist for the given user:
    - state.user_bandits (standard)
    - state.user_bandits_recent (decayed)

    If both missing: initialize from global prior.
    If one exists: clone it to create the other (so existing users keep learned params).
    """
    if user_id not in state.users:
        return
    user = state.users[user_id]

    has_std = user_id in state.user_bandits
    has_rec = user_id in state.user_bandits_recent

    if has_std and has_rec:
        return

    if has_std and not has_rec:
        state.user_bandits_recent[user_id] = clone_bandit(state.user_bandits[user_id])
        return

    if has_rec and not has_std:
        state.user_bandits[user_id] = clone_bandit(state.user_bandits_recent[user_id])
        return

    # both missing
    theta_init = _personal_prior_theta_from_global(state, user)
    state.user_bandits[user_id] = create_bandit(USER_FEATURE_DIM, init_theta=theta_init)
    state.user_bandits_recent[user_id] = create_bandit(USER_FEATURE_DIM, init_theta=theta_init)


def get_user_bandit_standard(state: "AppState", user_id: str) -> BanditModel:
    """Get the standard (non-decayed) bandit for a user."""
    ensure_user_bandits(state, user_id)
    return state.user_bandits[user_id]


def get_user_bandit_recent(state: "AppState", user_id: str) -> BanditModel:
    """Get the recent (decayed) bandit for a user."""
    ensure_user_bandits(state, user_id)
    return state.user_bandits_recent[user_id]
