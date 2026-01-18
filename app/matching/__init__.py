"""
Matching algorithms for the Language Exchange Matchmaking System.
"""

try:
    from .cooldown import (
        pair_key,
        get_cooldown_factor,
        set_cooldown_min,
        decay_cooldowns_toward_one,
    )
    from .scoring import (
        _normalize_nonneg_pair,
        get_normalized_scoring_weights,
        compute_score_components,
        compute_raw_score,
        build_edge_weights_for_pool,
    )
    from .matcher import (
        run_matching_round,
        SCIPY_AVAILABLE,
    )
except ImportError:
    from matching.cooldown import (
        pair_key,
        get_cooldown_factor,
        set_cooldown_min,
        decay_cooldowns_toward_one,
    )
    from matching.scoring import (
        _normalize_nonneg_pair,
        get_normalized_scoring_weights,
        compute_score_components,
        compute_raw_score,
        build_edge_weights_for_pool,
    )
    from matching.matcher import (
        run_matching_round,
        SCIPY_AVAILABLE,
    )

__all__ = [
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
]
