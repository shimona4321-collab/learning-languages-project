"""
Bandit algorithms for the Language Exchange Matchmaking System.
"""

try:
    from .linucb import (
        create_bandit,
        clone_bandit,
        ucb_score,
        bandit_update,
        bandit_update_decayed,
        _safe_solve,
    )
    from .features import (
        user_feature,
        pair_feature,
        ensure_user_bandits,
        get_user_bandit_standard,
        get_user_bandit_recent,
        _personal_prior_theta_from_global,
    )
except ImportError:
    from bandit.linucb import (
        create_bandit,
        clone_bandit,
        ucb_score,
        bandit_update,
        bandit_update_decayed,
        _safe_solve,
    )
    from bandit.features import (
        user_feature,
        pair_feature,
        ensure_user_bandits,
        get_user_bandit_standard,
        get_user_bandit_recent,
        _personal_prior_theta_from_global,
    )

__all__ = [
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
]
