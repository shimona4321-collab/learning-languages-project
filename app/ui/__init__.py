"""
User interface for the Language Exchange Matchmaking System.
"""

try:
    from .helpers import input_int_in_range, input_float
    from .admin import (
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
    )
    from .user_mode import (
        get_proposal_for_user,
        handle_proposal_response,
        apply_early_rejection,
        apply_full_proposal_outcome,
        user_mode,
    )
    from .visualization import show_bipartite_graph
except ImportError:
    from ui.helpers import input_int_in_range, input_float
    from ui.admin import (
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
    )
    from ui.user_mode import (
        get_proposal_for_user,
        handle_proposal_response,
        apply_early_rejection,
        apply_full_proposal_outcome,
        user_mode,
    )
    from ui.visualization import show_bipartite_graph

__all__ = [
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
]
