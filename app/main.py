"""
Main entry point for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

import sys

try:
    from .matching import get_normalized_scoring_weights, run_matching_round
    from .persistence import load_state, save_state, reconcile_state
    from .ui import (
        input_int_in_range,
        register_new_user,
        delete_user_by_id,
        delete_all_users_and_reset,
        show_all_users,
        show_active_proposals,
        bulk_generate_random_users,
        show_bipartite_graph,
        show_bandit_inspection_menu,
        reset_global_bandit,
        show_edge_weights_for_user,
        set_scoring_weights,
        user_mode,
    )
    from .models.state import AppState
except ImportError:
    from matching import get_normalized_scoring_weights, run_matching_round
    from persistence import load_state, save_state, reconcile_state
    from ui import (
        input_int_in_range,
        register_new_user,
        delete_user_by_id,
        delete_all_users_and_reset,
        show_all_users,
        show_active_proposals,
        bulk_generate_random_users,
        show_bipartite_graph,
        show_bandit_inspection_menu,
        reset_global_bandit,
        show_edge_weights_for_user,
        set_scoring_weights,
        user_mode,
    )
    from models.state import AppState


def admin_mode(state: AppState) -> None:
    """Admin mode menu."""
    while True:
        reconcile_state(state)
        wP, wG, wStd, wRec = get_normalized_scoring_weights(state)

        print("\n=== Admin Mode ===")
        print(f"(Weights) Personal vs Global: wP={wP:.2f}, wG={wG:.2f} | Personal split: wStd={wStd:.2f}, wRec={wRec:.2f}")
        print("1) Register new user")
        print("2) Delete a user by ID")
        print("3) Delete ALL users and reset bandits")
        print("4) Show all users")
        print("5) Show active proposals")
        print("6) Bulk-generate random users (Gaussian)")
        print("7) Run one matching round")
        print("8) Show bipartite graph (active edges only)")
        print("9) Inspect bandits")
        print("10) Reset the global bandit")
        print("11) Show edge weights for a specific user")
        print("12) Set scoring weights (Personal/Global, Std/Rec)")
        print("13) Return to main menu")

        choice = input_int_in_range("Choose (1-13): ", 1, 13)

        if choice == 1:
            register_new_user(state)
        elif choice == 2:
            delete_user_by_id(state)
        elif choice == 3:
            delete_all_users_and_reset(state)
        elif choice == 4:
            show_all_users(state)
        elif choice == 5:
            show_active_proposals(state)
        elif choice == 6:
            bulk_generate_random_users(state)
        elif choice == 7:
            run_matching_round(state)
            save_state(state)
        elif choice == 8:
            show_bipartite_graph(state)
        elif choice == 9:
            show_bandit_inspection_menu(state)
        elif choice == 10:
            reset_global_bandit(state)
        elif choice == 11:
            show_edge_weights_for_user(state)
        elif choice == 12:
            set_scoring_weights(state)
        else:
            print("Returning to main menu.")
            save_state(state)
            return


def main() -> None:
    """Main entry point."""
    state = load_state()

    while True:
        reconcile_state(state)

        print("\n=== Language Exchange Matchmaking System ===")
        print("1) Admin mode")
        print("2) User mode")
        print("3) Exit")
        choice = input_int_in_range("Choose (1-3): ", 1, 3)

        if choice == 1:
            admin_mode(state)
        elif choice == 2:
            user_mode(state)
        else:
            print("Goodbye!")
            save_state(state)
            break


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
