"""
Language-exchange matchmaking CLI using contextual bandits (LinUCB) + bipartite matching.

This is a compatibility layer that re-exports all symbols from the modular package.
The actual implementation is in the app/ subpackages.

Key requirements implemented:
- Single-file CLI script (main menu: Admin mode / User mode / Exit).
- Persistent state stored in a JSON file.
- Global LinUCB bandit (pair features).
- Per-user TWO personal bandits:
    1) standard LinUCB (no forgetting)
    2) recency-weighted LinUCB with exponential forgetting (A,b decay by gamma each update)
- Waiting-time fairness multiplier:
    edge_weight *= (1 + EPS_WAITING * wait_u) * (1 + EPS_WAITING * wait_v)
- Pair cooldown after breakup OR rejected proposal (cooldown factor decays back to 1 each round).
- Early rejection rule (if one rejects before the other responds: immediate resolution + bandit update).
- Gaussian-based random user generation with separate counts for Hebrew and English.
- Graph visualization shows ONLY current proposals and accepted matches.
- Matching:
  - If SciPy is available: Hungarian (linear_sum_assignment) with dummy columns to allow "unmatched".
  - Otherwise: NetworkX max_weight_matching (does NOT force max-cardinality).
- IMPORTANT: edge weights are REAL-VALUED scores; no [0,1] normalization of edge weights.
  The weight used for matching/visualization is raw_score(u,v) as computed here.

NEW:
- Scoring weights are configurable and persisted in JSON:
    - Personal vs Global: w_personal, w_global
    - Inside personal: w_personal_std, w_personal_rec
  These are normalized automatically when set via admin menu.

Dependencies:
- numpy
- networkx
- matplotlib
- optional: scipy (for Hungarian algorithm)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the app directory to path if needed for imports
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Import from submodules using absolute imports
from config import (
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

from models import User, BanditModel, Proposal, AppState

from bandit import (
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

from matching import (
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

from persistence import load_state, save_state, reconcile_state

from ui import (
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

from main import main, admin_mode

# For backward compatibility with scipy check
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Saving state and exiting...")
        try:
            st = load_state()
            save_state(st)
        except Exception:
            pass
        sys.exit(0)
