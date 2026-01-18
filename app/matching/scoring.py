"""
Scoring utilities for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

try:
    from ..config import (
        ALPHA_PERSONAL,
        ALPHA_GLOBAL,
        EPS_WAITING,
        PAIR_FEATURE_DIM,
        DEFAULT_W_PERSONAL,
        DEFAULT_W_GLOBAL,
        DEFAULT_W_PERSONAL_STD,
        DEFAULT_W_PERSONAL_REC,
    )
    from ..bandit import (
        ucb_score,
        create_bandit,
        user_feature,
        pair_feature,
        get_user_bandit_standard,
        get_user_bandit_recent,
    )
except ImportError:
    from config import (
        ALPHA_PERSONAL,
        ALPHA_GLOBAL,
        EPS_WAITING,
        PAIR_FEATURE_DIM,
        DEFAULT_W_PERSONAL,
        DEFAULT_W_GLOBAL,
        DEFAULT_W_PERSONAL_STD,
        DEFAULT_W_PERSONAL_REC,
    )
    from bandit import (
        ucb_score,
        create_bandit,
        user_feature,
        pair_feature,
        get_user_bandit_standard,
        get_user_bandit_recent,
    )

from .cooldown import get_cooldown_factor

if TYPE_CHECKING:
    from ..models.user import User
    from ..models.state import AppState


def _normalize_nonneg_pair(a: float, b: float, da: float, db: float) -> Tuple[float, float]:
    """Normalize a pair of non-negative values to sum to 1."""
    a = float(max(0.0, a))
    b = float(max(0.0, b))
    s = a + b
    if s <= 0.0:
        # fallback to defaults, normalized
        da = float(max(0.0, da))
        db = float(max(0.0, db))
        sd = da + db
        if sd <= 0.0:
            return 0.5, 0.5
        return da / sd, db / sd
    return a / s, b / s


def get_normalized_scoring_weights(state: "AppState") -> Tuple[float, float, float, float]:
    """
    Returns normalized (nonnegative) weights:
      (wP, wG, wStd, wRec)
    """
    wP, wG = _normalize_nonneg_pair(state.w_personal, state.w_global, DEFAULT_W_PERSONAL, DEFAULT_W_GLOBAL)
    wStd, wRec = _normalize_nonneg_pair(state.w_personal_std, state.w_personal_rec, DEFAULT_W_PERSONAL_STD, DEFAULT_W_PERSONAL_REC)
    return wP, wG, wStd, wRec


def compute_score_components(state: "AppState", u: "User", v: "User") -> Dict[str, float]:
    """
    Returns components used in raw_score(u,v), all real-valued:
    - personal_u_std / personal_u_recent / personal_u_total
    - personal_v_std / personal_v_recent / personal_v_total
    - global
    - weights used (normalized)
    - wait_u, wait_v, wait_pair
    - cooldown
    - raw
    """
    wP, wG, wStd, wRec = get_normalized_scoring_weights(state)

    bu_std = get_user_bandit_standard(state, u.user_id)
    bu_rec = get_user_bandit_recent(state, u.user_id)

    bv_std = get_user_bandit_standard(state, v.user_id)
    bv_rec = get_user_bandit_recent(state, v.user_id)

    x_u = user_feature(v)  # context for u: partner features of v
    x_v = user_feature(u)  # context for v: partner features of u

    s_u_std = ucb_score(bu_std, x_u, ALPHA_PERSONAL)
    s_u_rec = ucb_score(bu_rec, x_u, ALPHA_PERSONAL)
    s_u_total = float(wStd * s_u_std + wRec * s_u_rec)

    s_v_std = ucb_score(bv_std, x_v, ALPHA_PERSONAL)
    s_v_rec = ucb_score(bv_rec, x_v, ALPHA_PERSONAL)
    s_v_total = float(wStd * s_v_std + wRec * s_v_rec)

    if state.global_bandit is None:
        state.global_bandit = create_bandit(PAIR_FEATURE_DIM)

    x_pair = pair_feature(u, v)
    s_g = ucb_score(state.global_bandit, x_pair, ALPHA_GLOBAL)

    # waiting fairness (per-user multiplier)
    wait_u = 1.0 + float(EPS_WAITING) * float(u.waiting_rounds_without_offer)
    wait_v = 1.0 + float(EPS_WAITING) * float(v.waiting_rounds_without_offer)
    wait_pair = float(wait_u * wait_v)

    cd = get_cooldown_factor(state, u.user_id, v.user_id)

    personal_sum = float(s_u_total + s_v_total)
    bandit_score = float(wP * personal_sum + wG * s_g)
    raw = float(bandit_score * wait_pair * cd)

    return {
        "personal_u_std": float(s_u_std),
        "personal_u_recent": float(s_u_rec),
        "personal_u_total": float(s_u_total),
        "personal_v_std": float(s_v_std),
        "personal_v_recent": float(s_v_rec),
        "personal_v_total": float(s_v_total),
        "global": float(s_g),
        "personal_sum": float(personal_sum),
        "bandit_sum": float(bandit_score),
        "w_personal": float(wP),
        "w_global": float(wG),
        "w_personal_std": float(wStd),
        "w_personal_rec": float(wRec),
        "wait_u": float(wait_u),
        "wait_v": float(wait_v),
        "wait_pair": float(wait_pair),
        "cooldown": float(cd),
        "raw": float(raw),
    }


def compute_raw_score(state: "AppState", u: "User", v: "User") -> float:
    """Compute the raw matching score for a pair of users."""
    return compute_score_components(state, u, v)["raw"]


def build_edge_weights_for_pool(
    state: "AppState",
    eligible_left_ids: List[str],
    eligible_right_ids: List[str],
) -> Dict[Tuple[str, str], float]:
    """
    Compute raw_score(u,v) for each (he_id, en_id) in eligible sets.
    """
    weights: Dict[Tuple[str, str], float] = {}
    for he_id in eligible_left_ids:
        u = state.users[he_id]
        for en_id in eligible_right_ids:
            v = state.users[en_id]
            weights[(he_id, en_id)] = compute_raw_score(state, u, v)
    return weights
