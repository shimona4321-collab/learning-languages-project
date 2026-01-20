"""
Matching algorithm for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import builtins
import os

import numpy as np
import networkx as nx

try:
    from ..config import LANG_HE, LANG_EN
    from ..models.proposal import Proposal
    from ..bandit import ensure_user_bandits
except ImportError:
    from config import LANG_HE, LANG_EN
    from models.proposal import Proposal
    from bandit import ensure_user_bandits

from .cooldown import pair_key, decay_cooldowns_toward_one
from .scoring import build_edge_weights_for_pool

# Optional SciPy Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    linear_sum_assignment = None
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from ..models.state import AppState


# ------------------------------
# Verbosity control
# ------------------------------
# Default is verbose unless disabled via environment variable LANGMATCH_VERBOSE=0.
_MATCHER_VERBOSE = os.environ.get('LANGMATCH_VERBOSE', '1').strip().lower() not in ('0','false','no','off')


def set_matcher_verbose(flag: bool) -> None:
    """Enable/disable internal matcher prints (useful in notebooks/experiments)."""
    global _MATCHER_VERBOSE
    _MATCHER_VERBOSE = bool(flag)


def _vprint(*args, **kwargs) -> None:
    """Verbose print that won't spam output when verbosity is disabled."""
    if _MATCHER_VERBOSE:
        builtins.print(*args, **kwargs)


def _hungarian_max_weight_subset(W: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solve max-weight matching where each row can be matched to:
    - one real column, or
    - a dummy column (meaning unmatched).
    Here W is shape (n_rows, n_real_cols). We add n_rows dummy columns with 0 weight.
    Return list of (row_i, col_j) for matches ONLY to real columns, and only if weight > 0.
    """
    n_rows, n_cols = W.shape
    W_ext = np.hstack([W, np.zeros((n_rows, n_rows), dtype=float)])
    cost = -W_ext
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs: List[Tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        if c < n_cols:
            if float(W[r, c]) > 0.0:
                pairs.append((r, c))
    return pairs


def reconcile_state(state: "AppState") -> None:
    """
    Make state internally consistent.
    Import here to avoid circular imports.
    """
    try:
        from ..persistence.storage import reconcile_state as _reconcile
    except ImportError:
        from persistence.storage import reconcile_state as _reconcile
    _reconcile(state)


def run_matching_round(state: "AppState") -> None:
    """
    Matching round that recomputes the WHOLE bipartite graph every round,
    excluding ONLY active matches (users with status == 'matched').

    Key behavior:
    - Any pending proposals are discarded at the start of the round.
      (So we always re-solve the full graph for everyone who is not matched.)
    - Eligible users: ALL non-matched users with the correct language config.
    - Waiting counter:
        * matched -> 0
        * users who had an offer last round OR got an offer this round -> 0
        * eligible users who did not get an offer this round -> +1
    - Cooldowns always decay each round.
    """
    reconcile_state(state)

    if not state.users:
        _vprint("No users in the system.")
        return

    state.round_index += 1

    # Ensure bandits exist for all users (including recent bandits)
    for uid in list(state.users.keys()):
        ensure_user_bandits(state, uid)

    had_offer_prev = {uid for uid, u in state.users.items() if u.status == "has_offer"}

    # 1) Wipe proposals + reset all non-matched users to pool
    if state.proposals:
        state.proposals.clear()

    for u in state.users.values():
        if u.status != "matched":
            u.status = "no_offer"
            u.current_partner_id = None

    # 2) Build eligible pools
    eligible_left_ids: List[str] = []
    eligible_right_ids: List[str] = []

    for uid, u in state.users.items():
        if u.status == "matched":
            continue
        if u.native_language == LANG_HE and u.target_language == LANG_EN:
            eligible_left_ids.append(uid)
        elif u.native_language == LANG_EN and u.target_language == LANG_HE:
            eligible_right_ids.append(uid)

    eligible_set = set(eligible_left_ids) | set(eligible_right_ids)

    if not eligible_set:
        _vprint("No eligible users in the matching pool (non-matched, correct language configs).")
        decay_cooldowns_toward_one(state)
        return

    if not eligible_left_ids or not eligible_right_ids:
        _vprint("Cannot run matching: one side has no eligible users.")
        for uid, u in state.users.items():
            if u.status == "matched":
                u.waiting_rounds_without_offer = 0
            elif uid in had_offer_prev:
                u.waiting_rounds_without_offer = 0
            elif uid in eligible_set:
                u.waiting_rounds_without_offer = int(u.waiting_rounds_without_offer) + 1
            else:
                u.waiting_rounds_without_offer = int(u.waiting_rounds_without_offer)
        decay_cooldowns_toward_one(state)
        return

    # 3) Compute edge weights over the FULL eligible bipartite graph
    edge_weights = build_edge_weights_for_pool(state, eligible_left_ids, eligible_right_ids)

    # 4) Solve maximum-weight matching (subset) and create proposals
    if SCIPY_AVAILABLE:
        W = np.zeros((len(eligible_left_ids), len(eligible_right_ids)), dtype=float)
        for i, he_id in enumerate(eligible_left_ids):
            for j, en_id in enumerate(eligible_right_ids):
                W[i, j] = float(edge_weights[(he_id, en_id)])
        pairs_idx = _hungarian_max_weight_subset(W)
        candidate_matches: List[Tuple[str, str]] = [(eligible_left_ids[i], eligible_right_ids[j]) for i, j in pairs_idx]
    else:
        G = nx.Graph()
        for he_id in eligible_left_ids:
            G.add_node(he_id, bipartite=0)
        for en_id in eligible_right_ids:
            G.add_node(en_id, bipartite=1)
        for (he_id, en_id), w in edge_weights.items():
            G.add_edge(he_id, en_id, weight=float(w))

        mset = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight="weight")
        candidate_matches = []
        for a, b in mset:
            if a in eligible_left_ids and b in eligible_right_ids:
                candidate_matches.append((a, b))
            elif b in eligible_left_ids and a in eligible_right_ids:
                candidate_matches.append((b, a))

        candidate_matches = [(he, en) for he, en in candidate_matches if float(edge_weights.get((he, en), 0.0)) > 0.0]

    offered_now: set = set()
    created_pairs: List[Tuple[str, str, float]] = []

    for he_id, en_id in candidate_matches:
        u = state.users.get(he_id)
        v = state.users.get(en_id)
        if not u or not v:
            continue
        if u.status == "matched" or v.status == "matched":
            continue

        a, b = sorted([u.user_id, v.user_id])
        k = pair_key(a, b)

        score = float(edge_weights.get((he_id, en_id), 0.0))
        state.proposals[k] = Proposal(user1_id=a, user2_id=b, score_at_offer=score)

        u.status = "has_offer"
        v.status = "has_offer"
        u.current_partner_id = v.user_id
        v.current_partner_id = u.user_id

        offered_now.add(u.user_id)
        offered_now.add(v.user_id)

        created_pairs.append((he_id, en_id, score))

    # 5) Update waiting counters
    for uid, u in state.users.items():
        if u.status == "matched":
            u.waiting_rounds_without_offer = 0
        elif uid in had_offer_prev:
            u.waiting_rounds_without_offer = 0
        elif uid in offered_now:
            u.waiting_rounds_without_offer = 0
        elif uid in eligible_set:
            u.waiting_rounds_without_offer = int(u.waiting_rounds_without_offer) + 1
        else:
            u.waiting_rounds_without_offer = int(u.waiting_rounds_without_offer)

    # 6) Cooldown decay every round
    decay_cooldowns_toward_one(state)

    if not created_pairs:
        _vprint("No proposals created this round (no positive-weight pairs selected).")
    else:
        _vprint("Proposed pairs in this matching round:")
        for he_id, en_id, w in sorted(created_pairs, key=lambda t: t[2], reverse=True):
            _vprint(f"  {he_id} ({LANG_HE}) <--> {en_id} ({LANG_EN}) | score = {w:.3f}")
