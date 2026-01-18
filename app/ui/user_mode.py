"""
User mode for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

try:
    from ..config import TOPICS, PAIR_FEATURE_DIM, PERSONAL_RECENCY_GAMMA
    from ..models.proposal import Proposal
    from ..bandit import (
        create_bandit,
        bandit_update,
        bandit_update_decayed,
        user_feature,
        pair_feature,
        get_user_bandit_standard,
        get_user_bandit_recent,
    )
    from ..matching.cooldown import pair_key, set_cooldown_min
    from ..persistence import save_state, reconcile_state
except ImportError:
    from config import TOPICS, PAIR_FEATURE_DIM, PERSONAL_RECENCY_GAMMA
    from models.proposal import Proposal
    from bandit import (
        create_bandit,
        bandit_update,
        bandit_update_decayed,
        user_feature,
        pair_feature,
        get_user_bandit_standard,
        get_user_bandit_recent,
    )
    from matching.cooldown import pair_key, set_cooldown_min
    from persistence import save_state, reconcile_state

from .helpers import input_int_in_range

if TYPE_CHECKING:
    from ..models.user import User
    from ..models.state import AppState


def get_proposal_for_user(state: "AppState", user: "User") -> Optional[Proposal]:
    """Get the proposal for a user."""
    pid = user.current_partner_id
    if not pid:
        return None
    return state.proposals.get(pair_key(user.user_id, pid))


def _proposal_my_other_fields(prop: Proposal, user_id: str) -> Tuple[str, str, Optional[bool]]:
    """Get the field names and other user's response for a proposal."""
    if prop.user1_id == user_id:
        return "response1", "response2", prop.response2
    return "response2", "response1", prop.response1


def apply_early_rejection(state: "AppState", rejecting_user: "User", other_user: "User", proposal: Proposal) -> None:
    """
    Early rejection rule:
    - If one user rejects before the other responds:
      - Update global bandit with reward=0 for the pair
      - Update rejecting user's BOTH personal bandits with reward=0 on context(other_user)
      - Set cooldown to COOLDOWN_MIN for this pair
      - Clear proposal and return both users to no_offer
    """
    if state.global_bandit is None:
        state.global_bandit = create_bandit(PAIR_FEATURE_DIM)

    x_pair = pair_feature(rejecting_user, other_user)
    bandit_update(state.global_bandit, x_pair, reward=0.0)

    br_std = get_user_bandit_standard(state, rejecting_user.user_id)
    br_rec = get_user_bandit_recent(state, rejecting_user.user_id)

    x_ctx = user_feature(other_user)
    bandit_update(br_std, x_ctx, reward=0.0)
    bandit_update_decayed(br_rec, x_ctx, reward=0.0, gamma=PERSONAL_RECENCY_GAMMA)

    set_cooldown_min(state, rejecting_user.user_id, other_user.user_id)

    rejecting_user.status = "no_offer"
    rejecting_user.current_partner_id = None
    other_user.status = "no_offer"
    other_user.current_partner_id = None

    state.proposals.pop(pair_key(proposal.user1_id, proposal.user2_id), None)


def apply_full_proposal_outcome(state: "AppState", proposal: Proposal) -> None:
    """
    When both sides responded:
    - Update bandits:
      - global reward = 1 iff both accept else 0
      - personal reward per user = 1 if they accepted else 0
        (updates BOTH personal bandits: standard + decayed)
    - If both accept -> matched
    - Else -> no_offer for both + cooldown applied
    """
    u1 = state.users.get(proposal.user1_id)
    u2 = state.users.get(proposal.user2_id)
    if not u1 or not u2:
        state.proposals.pop(pair_key(proposal.user1_id, proposal.user2_id), None)
        return
    if proposal.response1 is None or proposal.response2 is None:
        return

    if state.global_bandit is None:
        state.global_bandit = create_bandit(PAIR_FEATURE_DIM)

    accept1 = bool(proposal.response1)
    accept2 = bool(proposal.response2)

    x_pair = pair_feature(u1, u2)
    bandit_update(state.global_bandit, x_pair, reward=1.0 if (accept1 and accept2) else 0.0)

    b1_std = get_user_bandit_standard(state, u1.user_id)
    b1_rec = get_user_bandit_recent(state, u1.user_id)

    b2_std = get_user_bandit_standard(state, u2.user_id)
    b2_rec = get_user_bandit_recent(state, u2.user_id)

    x12 = user_feature(u2)
    x21 = user_feature(u1)

    r1 = 1.0 if accept1 else 0.0
    r2 = 1.0 if accept2 else 0.0

    bandit_update(b1_std, x12, reward=r1)
    bandit_update_decayed(b1_rec, x12, reward=r1, gamma=PERSONAL_RECENCY_GAMMA)

    bandit_update(b2_std, x21, reward=r2)
    bandit_update_decayed(b2_rec, x21, reward=r2, gamma=PERSONAL_RECENCY_GAMMA)

    # Remove proposal
    state.proposals.pop(pair_key(u1.user_id, u2.user_id), None)

    if accept1 and accept2:
        u1.status = "matched"
        u2.status = "matched"
        u1.current_partner_id = u2.user_id
        u2.current_partner_id = u1.user_id
        u1.waiting_rounds_without_offer = 0
        u2.waiting_rounds_without_offer = 0
        print(f"Users {u1.user_id} and {u2.user_id} are now MATCHED!")
    else:
        set_cooldown_min(state, u1.user_id, u2.user_id)
        u1.status = "no_offer"
        u2.status = "no_offer"
        u1.current_partner_id = None
        u2.current_partner_id = None

        if (not accept1) and (not accept2):
            print(f"Users {u1.user_id} and {u2.user_id} both rejected the proposal.")
        else:
            print(f"Users {u1.user_id} and {u2.user_id} had a mixed response; no match.")


def handle_proposal_response(state: "AppState", user: "User", accepted: bool) -> None:
    """Handle a user's response to a proposal."""
    partner_id = user.current_partner_id
    if not partner_id or partner_id not in state.users:
        print("You have no valid current proposal.")
        user.status = "no_offer"
        user.current_partner_id = None
        return

    partner = state.users[partner_id]
    prop = get_proposal_for_user(state, user)
    if prop is None:
        print("Proposal record missing; resetting to no_offer.")
        user.status = "no_offer"
        user.current_partner_id = None
        partner.status = "no_offer"
        partner.current_partner_id = None
        return

    my_field, _, other_resp = _proposal_my_other_fields(prop, user.user_id)

    if not accepted and other_resp is None:
        print("Early rejection: the other side has not responded yet.")
        apply_early_rejection(state, rejecting_user=user, other_user=partner, proposal=prop)
        return

    if my_field == "response1":
        prop.response1 = accepted
    else:
        prop.response2 = accepted

    if prop.response1 is not None and prop.response2 is not None:
        apply_full_proposal_outcome(state, prop)
    else:
        print("Recorded your response. Waiting for the other side...")


def user_mode(state: "AppState") -> None:
    """Run the user mode interface."""
    print("\n=== User Mode ===")
    user_id = input("Enter your user ID: ").strip()
    user = state.users.get(user_id)
    if not user:
        print("No such user. Please contact the admin to register.")
        return

    reconcile_state(state)

    if user.status == "matched":
        partner = state.users.get(user.current_partner_id) if user.current_partner_id else None
        print("\nYou are currently MATCHED.")
        if partner:
            print(f"Matched partner ID: {partner.user_id}")
            print(f"  Native language      : {partner.native_language}")
            print(f"  Target language      : {partner.target_language}")
            print(f"  Difficulty (0..1)    : {partner.target_difficulty:.2f}")
            print(f"  Availability (0..1)  : {partner.availability_norm:.2f}")
            print("  Topics (0..1):")
            for t in TOPICS:
                print(f"    - {t}: {partner.topic_interest_norm(t):.2f}")
        else:
            print("Warning: partner record missing.")

        print("\nOptions:")
        print("  1) Keep this match")
        print("  2) Cancel this match and return to pool (cooldown will apply)")
        print("  3) Exit user mode")
        choice = input_int_in_range("Choose (1-3): ", 1, 3)
        if choice == 1:
            print("Keeping the current match.")
        elif choice == 2:
            if partner:
                set_cooldown_min(state, user.user_id, partner.user_id)
                partner.status = "no_offer"
                partner.current_partner_id = None
                partner.waiting_rounds_without_offer = 0
            user.status = "no_offer"
            user.current_partner_id = None
            user.waiting_rounds_without_offer = 0
            print("Match cancelled; cooldown applied.")
        else:
            print("Exiting user mode.")
        save_state(state)
        return

    if user.status == "has_offer":
        prop = get_proposal_for_user(state, user)
        partner = state.users.get(user.current_partner_id) if user.current_partner_id else None
        if not prop or not partner:
            print("Inconsistent offer state; resetting to no_offer.")
            user.status = "no_offer"
            user.current_partner_id = None
            save_state(state)
            return

        print("\nYou have a pending proposal!")
        print(f"Proposed partner ID: {partner.user_id}")
        print(f"  Native language      : {partner.native_language}")
        print(f"  Target language      : {partner.target_language}")
        print(f"  Difficulty (0..1)    : {partner.target_difficulty:.2f}")
        print(f"  Availability (0..1)  : {partner.availability_norm:.2f}")
        print("  Topics (0..1):")
        for t in TOPICS:
            print(f"    - {t}: {partner.topic_interest_norm(t):.2f}")

        print(f"\n(Score at offer time: {prop.score_at_offer:.3f})")

        print("\nOptions:")
        print("  1) Accept")
        print("  2) Reject")
        print("  3) Exit without responding")
        choice = input_int_in_range("Choose (1-3): ", 1, 3)
        if choice == 1:
            handle_proposal_response(state, user, accepted=True)
        elif choice == 2:
            handle_proposal_response(state, user, accepted=False)
        else:
            print("No response recorded; proposal remains pending.")
        save_state(state)
        return

    print("\nYou currently have NO proposal.")
    print("Please wait for the admin to run a matching round.")
    return
