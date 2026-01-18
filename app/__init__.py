"""
Language Exchange Matchmaking System - Main package.

This package provides a modular implementation of the language exchange
matchmaking system using contextual bandits (LinUCB) and bipartite matching.
"""

# Re-export all public symbols for backward compatibility
from .config import (
    STATE_FILE,
    LANG_HE,
    LANG_EN,
    TOPICS,
    ALPHA_PERSONAL,
    ALPHA_GLOBAL,
    EPS_WAITING,
    PERSONAL_RECENCY_GAMMA,
    DEFAULT_W_PERSONAL,
    DEFAULT_W_GLOBAL,
    DEFAULT_W_PERSONAL_STD,
    DEFAULT_W_PERSONAL_REC,
    COOLDOWN_MIN,
    COOLDOWN_GROWTH,
    GAUSS_TARGET_MEAN,
    GAUSS_TARGET_STD,
    GAUSS_AVAIL_MEAN,
    GAUSS_AVAIL_STD,
    GAUSS_TOPIC_MEAN,
    GAUSS_TOPIC_STD,
    USER_FEATURE_DIM,
    PAIR_FEATURE_DIM,
)

from .models import User, BanditModel, Proposal, AppState

from .bandit import (
    create_bandit,
    clone_bandit,
    ucb_score,
    bandit_update,
    bandit_update_decayed,
    _safe_solve,
    user_feature,
    pair_feature,
    ensure_user_bandits,
    get_user_bandit_standard,
    get_user_bandit_recent,
    _personal_prior_theta_from_global,
)

from .matching import (
    pair_key,
    get_cooldown_factor,
    set_cooldown_min,
    decay_cooldowns_toward_one,
    _normalize_nonneg_pair,
    get_normalized_scoring_weights,
    compute_score_components,
    compute_raw_score,
    build_edge_weights_for_pool,
    run_matching_round,
    SCIPY_AVAILABLE,
)

from .persistence import load_state, save_state, reconcile_state

from .ui import (
    input_int_in_range,
    input_float,
    set_scoring_weights,
    register_new_user,
    delete_user_by_id,
    delete_all_users_and_reset,
    reset_global_bandit,
    sample_gaussian_clipped,
    bulk_generate_random_users,
    show_all_users,
    show_active_proposals,
    show_bandit_inspection_menu,
    show_edge_weights_for_user,
    get_proposal_for_user,
    handle_proposal_response,
    apply_early_rejection,
    apply_full_proposal_outcome,
    user_mode,
    show_bipartite_graph,
)

from .main import main, admin_mode

__all__ = [
    # Config
    "STATE_FILE",
    "LANG_HE",
    "LANG_EN",
    "TOPICS",
    "ALPHA_PERSONAL",
    "ALPHA_GLOBAL",
    "EPS_WAITING",
    "PERSONAL_RECENCY_GAMMA",
    "DEFAULT_W_PERSONAL",
    "DEFAULT_W_GLOBAL",
    "DEFAULT_W_PERSONAL_STD",
    "DEFAULT_W_PERSONAL_REC",
    "COOLDOWN_MIN",
    "COOLDOWN_GROWTH",
    "GAUSS_TARGET_MEAN",
    "GAUSS_TARGET_STD",
    "GAUSS_AVAIL_MEAN",
    "GAUSS_AVAIL_STD",
    "GAUSS_TOPIC_MEAN",
    "GAUSS_TOPIC_STD",
    "USER_FEATURE_DIM",
    "PAIR_FEATURE_DIM",
    # Models
    "User",
    "BanditModel",
    "Proposal",
    "AppState",
    # Bandit
    "create_bandit",
    "clone_bandit",
    "ucb_score",
    "bandit_update",
    "bandit_update_decayed",
    "_safe_solve",
    "user_feature",
    "pair_feature",
    "ensure_user_bandits",
    "get_user_bandit_standard",
    "get_user_bandit_recent",
    "_personal_prior_theta_from_global",
    # Matching
    "pair_key",
    "get_cooldown_factor",
    "set_cooldown_min",
    "decay_cooldowns_toward_one",
    "_normalize_nonneg_pair",
    "get_normalized_scoring_weights",
    "compute_score_components",
    "compute_raw_score",
    "build_edge_weights_for_pool",
    "run_matching_round",
    "SCIPY_AVAILABLE",
    # Persistence
    "load_state",
    "save_state",
    "reconcile_state",
    # UI
    "input_int_in_range",
    "input_float",
    "set_scoring_weights",
    "register_new_user",
    "delete_user_by_id",
    "delete_all_users_and_reset",
    "reset_global_bandit",
    "sample_gaussian_clipped",
    "bulk_generate_random_users",
    "show_all_users",
    "show_active_proposals",
    "show_bandit_inspection_menu",
    "show_edge_weights_for_user",
    "get_proposal_for_user",
    "handle_proposal_response",
    "apply_early_rejection",
    "apply_full_proposal_outcome",
    "user_mode",
    "show_bipartite_graph",
    # Main
    "main",
    "admin_mode",
]
