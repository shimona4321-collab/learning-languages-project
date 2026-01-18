"""
Admin mode for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

try:
    from ..config import (
        LANG_HE,
        LANG_EN,
        TOPICS,
        PAIR_FEATURE_DIM,
        GAUSS_TARGET_MEAN,
        GAUSS_TARGET_STD,
        GAUSS_AVAIL_MEAN,
        GAUSS_AVAIL_STD,
        GAUSS_TOPIC_MEAN,
        GAUSS_TOPIC_STD,
    )
    from ..models.user import User
    from ..bandit import create_bandit, ensure_user_bandits, _safe_solve
    from ..matching import (
        get_normalized_scoring_weights,
        compute_score_components,
        run_matching_round,
    )
    from ..persistence import save_state
except ImportError:
    from config import (
        LANG_HE,
        LANG_EN,
        TOPICS,
        PAIR_FEATURE_DIM,
        GAUSS_TARGET_MEAN,
        GAUSS_TARGET_STD,
        GAUSS_AVAIL_MEAN,
        GAUSS_AVAIL_STD,
        GAUSS_TOPIC_MEAN,
        GAUSS_TOPIC_STD,
    )
    from models.user import User
    from bandit import create_bandit, ensure_user_bandits, _safe_solve
    from matching import (
        get_normalized_scoring_weights,
        compute_score_components,
        run_matching_round,
    )
    from persistence import save_state

from .helpers import input_int_in_range, input_float

if TYPE_CHECKING:
    from ..models.state import AppState


def set_scoring_weights(state: "AppState") -> None:
    """
    Set and persist scoring weights.
    We normalize:
      w_personal + w_global = 1
      w_personal_std + w_personal_rec = 1
    Negative values are clamped to 0.
    """
    wP, wG, wStd, wRec = get_normalized_scoring_weights(state)
    print("\n=== Set Scoring Weights ===")
    print("Current (normalized):")
    print(f"  Personal vs Global: wP={wP:.3f}, wG={wG:.3f}")
    print(f"  Personal split    : wStd={wStd:.3f}, wRec={wRec:.3f}")
    print("\nEnter new raw weights (they will be normalized).")

    new_wP = input_float("w_personal (>=0): ")
    new_wG = input_float("w_global   (>=0): ")
    new_wStd = input_float("w_personal_std (>=0): ")
    new_wRec = input_float("w_personal_rec (>=0): ")

    # store raw
    state.w_personal = float(new_wP)
    state.w_global = float(new_wG)
    state.w_personal_std = float(new_wStd)
    state.w_personal_rec = float(new_wRec)

    # show normalized result (and keep raw persisted; computation uses normalized)
    wP2, wG2, wStd2, wRec2 = get_normalized_scoring_weights(state)
    save_state(state)
    print("\nSaved. Effective (normalized) weights are:")
    print(f"  Personal vs Global: wP={wP2:.3f}, wG={wG2:.3f}")
    print(f"  Personal split    : wStd={wStd2:.3f}, wRec={wRec2:.3f}")


def register_new_user(state: "AppState") -> None:
    """Register a new user."""
    print("\n=== Register New User ===")
    user_id = input("Enter user ID: ").strip()
    if not user_id:
        print("User ID cannot be empty.")
        return
    if user_id in state.users:
        print("User with this ID already exists.")
        return

    print("Select native language:")
    print("  1) Hebrew")
    print("  2) English")
    lang_choice = input_int_in_range("Choice (1-2): ", 1, 2)

    if lang_choice == 1:
        native, target = LANG_HE, LANG_EN
    else:
        native, target = LANG_EN, LANG_HE

    print(f"\nNative language: {native}")
    print(f"Target language: {target}\n")

    target_level_raw = input_int_in_range(
        f"Your level in target language {target} (0=no knowledge, 10=fluent): ", 0, 10
    )
    availability_raw = input_int_in_range(
        "Weekly availability (0=none, 10=very available): ", 0, 10
    )

    topic_interest_raw: Dict[str, int] = {}
    print("\nRate your interest for each topic (0-10):")
    for t in TOPICS:
        topic_interest_raw[t] = input_int_in_range(f"  {t}: ", 0, 10)

    user = User(
        user_id=user_id,
        native_language=native,
        target_language=target,
        target_level_raw=target_level_raw,
        availability_raw=availability_raw,
        topic_interest_raw=topic_interest_raw,
    )
    state.users[user_id] = user

    # IMPORTANT: initialize BOTH personal bandits based on global
    ensure_user_bandits(state, user_id)

    save_state(state)
    print(f"User {user_id} registered.\n")


def delete_user_by_id(state: "AppState") -> None:
    """Delete a user by ID."""
    print("\n=== Delete User ===")
    user_id = input("Enter user ID to delete: ").strip()
    if user_id not in state.users:
        print("No such user.")
        return

    to_del_props = []
    for k, p in state.proposals.items():
        if p.user1_id == user_id or p.user2_id == user_id:
            to_del_props.append(k)
    for k in to_del_props:
        state.proposals.pop(k, None)

    to_del_cd = []
    for k in list(state.pair_cooldowns.keys()):
        a, b = k.split("|", 1)
        if a == user_id or b == user_id:
            to_del_cd.append(k)
    for k in to_del_cd:
        state.pair_cooldowns.pop(k, None)

    u = state.users[user_id]
    if u.status == "matched" and u.current_partner_id in state.users:
        v = state.users[u.current_partner_id]
        v.status = "no_offer"
        v.current_partner_id = None

    state.users.pop(user_id, None)
    state.user_bandits.pop(user_id, None)
    state.user_bandits_recent.pop(user_id, None)

    save_state(state)
    print(f"User {user_id} deleted.\n")


def delete_all_users_and_reset(state: "AppState") -> None:
    """Delete all users and reset the system."""
    print("\n!!! WARNING: This deletes ALL users, proposals, cooldowns, and resets bandits !!!")
    confirm = input("Type 'YES' to confirm: ").strip()
    if confirm != "YES":
        print("Aborted.")
        return

    state.users.clear()
    state.user_bandits.clear()
    state.user_bandits_recent.clear()
    state.proposals.clear()
    state.pair_cooldowns.clear()
    state.global_bandit = create_bandit(PAIR_FEATURE_DIM)
    state.round_index = 0

    save_state(state)
    print("System reset complete.\n")


def reset_global_bandit(state: "AppState") -> None:
    """Reset the global bandit."""
    print("\n=== Reset Global Bandit ===")
    state.global_bandit = create_bandit(PAIR_FEATURE_DIM)
    # NOTE: we do not overwrite existing personal bandits,
    # but new users will be initialized from the new global.
    save_state(state)
    print("Global bandit reset.\n")


def sample_gaussian_clipped(mean: float, std: float) -> int:
    """Sample from a Gaussian distribution and clip to [0, 10]."""
    val = random.gauss(mean, std)
    val = max(0.0, min(10.0, val))
    return int(round(val))


def bulk_generate_random_users(state: "AppState") -> None:
    """Generate random users with Gaussian distribution."""
    print("\n=== Bulk Generate Random Users (Gaussian) ===")
    n_he = input_int_in_range("How many Hebrew-native users? ", 0, 100000)
    n_en = input_int_in_range("How many English-native users? ", 0, 100000)
    if n_he == 0 and n_en == 0:
        print("Nothing to generate.")
        return

    use_defaults = input("Use default Gaussian parameters? (y/n) [y]: ").strip().lower()
    if use_defaults in ("", "y", "yes"):
        target_mean, target_std = GAUSS_TARGET_MEAN, GAUSS_TARGET_STD
        avail_mean, avail_std = GAUSS_AVAIL_MEAN, GAUSS_AVAIL_STD
        topic_means = {t: GAUSS_TOPIC_MEAN for t in TOPICS}
        topic_stds = {t: GAUSS_TOPIC_STD for t in TOPICS}
    else:
        target_mean = input_float("Target level mean (0-10): ")
        target_std = input_float("Target level std: ")
        avail_mean = input_float("Availability mean (0-10): ")
        avail_std = input_float("Availability std: ")
        topic_means = {}
        topic_stds = {}
        for t in TOPICS:
            topic_means[t] = input_float(f"Mean interest for {t} (0-10): ")
            topic_stds[t] = input_float(f"Std for {t}: ")

    def unique_user_id(prefix: str, idx: int) -> str:
        base = f"{prefix}_{idx}"
        uid = base
        c = 1
        while uid in state.users:
            c += 1
            uid = f"{base}_{c}"
        return uid

    def make_random_user(native_lang: str, idx: int) -> User:
        target_lang = LANG_EN if native_lang == LANG_HE else LANG_HE
        user_id = unique_user_id("rand_he" if native_lang == LANG_HE else "rand_en", idx)
        target_level_raw = sample_gaussian_clipped(target_mean, target_std)
        availability_raw = sample_gaussian_clipped(avail_mean, avail_std)
        topic_interest_raw = {t: sample_gaussian_clipped(topic_means[t], topic_stds[t]) for t in TOPICS}
        return User(
            user_id=user_id,
            native_language=native_lang,
            target_language=target_lang,
            target_level_raw=target_level_raw,
            availability_raw=availability_raw,
            topic_interest_raw=topic_interest_raw,
        )

    for i in range(1, n_he + 1):
        u = make_random_user(LANG_HE, i)
        state.users[u.user_id] = u
        ensure_user_bandits(state, u.user_id)

    for i in range(1, n_en + 1):
        u = make_random_user(LANG_EN, i)
        state.users[u.user_id] = u
        ensure_user_bandits(state, u.user_id)

    save_state(state)
    print(f"Generated {n_he} Hebrew-native and {n_en} English-native users.\n")


def show_all_users(state: "AppState") -> None:
    """Show all registered users."""
    print("\n=== All Users ===")
    if not state.users:
        print("No users registered.")
        return

    for u in state.users.values():
        print(f"\nUser ID: {u.user_id}")
        print(f"  Native language          : {u.native_language}")
        print(f"  Target language          : {u.target_language}")
        print(f"  Status                   : {u.status}")
        print(f"  Current partner          : {u.current_partner_id}")
        print(f"  Waiting rounds w/o offer : {u.waiting_rounds_without_offer}")
        print(f"  Difficulty (0..1)        : {u.target_difficulty:.2f}")
        print(f"  Availability (0..1)      : {u.availability_norm:.2f}")
        print("  Topic interests (0..1):")
        for t in TOPICS:
            print(f"    - {t}: {u.topic_interest_norm(t):.2f}")


def show_active_proposals(state: "AppState") -> None:
    """Show all active proposals."""
    print("\n=== Active Proposals ===")
    if not state.proposals:
        print("No active proposals.")
        return
    for k, p in state.proposals.items():
        print(f"{k}: {p.user1_id} <-> {p.user2_id} | score_at_offer={p.score_at_offer:.3f} | r1={p.response1} r2={p.response2}")


def show_bandit_inspection_menu(state: "AppState") -> None:
    """Show bandit inspection menu."""
    while True:
        print("\n=== Bandit Inspection ===")
        print("1) Show global bandit (theta summary)")
        print("2) Show global bandit (full A and b)")
        print("3) List users with personal bandits")
        print("4) Show personal bandits for a specific user (standard + recent)")
        print("5) Back")
        c = input_int_in_range("Choose (1-5): ", 1, 5)
        if c == 5:
            return

        if state.global_bandit is None:
            state.global_bandit = create_bandit(PAIR_FEATURE_DIM)

        if c == 1:
            theta = _safe_solve(state.global_bandit.A, state.global_bandit.b)
            np.set_printoptions(precision=4, suppress=True)
            print("\nGlobal theta:")
            print(theta)
        elif c == 2:
            np.set_printoptions(precision=4, suppress=True)
            print("\nA_global:")
            print(state.global_bandit.A)
            print("\nb_global:")
            print(state.global_bandit.b)
        elif c == 3:
            if not state.users:
                print("No users.")
            else:
                print("Users:")
                for uid in sorted(state.users.keys()):
                    print(f"  - {uid}")
        elif c == 4:
            uid = input("Enter user ID: ").strip()
            if uid not in state.users:
                print("No such user.")
                continue
            ensure_user_bandits(state, uid)

            b_std = state.user_bandits[uid]
            b_rec = state.user_bandits_recent[uid]

            np.set_printoptions(precision=4, suppress=True)

            theta_std = _safe_solve(b_std.A, b_std.b)
            theta_rec = _safe_solve(b_rec.A, b_rec.b)

            print("\nPersonal (STANDARD) theta:")
            print(theta_std)
            print("\nPersonal (RECENT/DECAYED) theta:")
            print(theta_rec)

            print("\nA_user_standard:")
            print(b_std.A)
            print("\nb_user_standard:")
            print(b_std.b)

            print("\nA_user_recent:")
            print(b_rec.A)
            print("\nb_user_recent:")
            print(b_rec.b)


def show_edge_weights_for_user(state: "AppState") -> None:
    """Show edge weights for a specific user."""
    print("\n=== Show Edge Weights for a Specific User ===")
    user_id = input("Enter user ID: ").strip()
    u = state.users.get(user_id)
    if not u:
        print("No such user.")
        return

    if u.native_language == LANG_HE:
        partner_ids = [pid for pid, pu in state.users.items() if pu.native_language == LANG_EN and pu.status != "matched"]
    elif u.native_language == LANG_EN:
        partner_ids = [pid for pid, pu in state.users.items() if pu.native_language == LANG_HE and pu.status != "matched"]
    else:
        print("User has unknown native language.")
        return

    if not partner_ids:
        print("No potential partners on the other side.")
        return

    rows: List[Tuple] = []
    for pid in partner_ids:
        v = state.users[pid]
        comps = compute_score_components(state, u, v)
        rows.append(
            (
                pid,
                comps["raw"],
                comps["personal_u_std"],
                comps["personal_u_recent"],
                comps["personal_u_total"],
                comps["personal_v_total"],
                comps["global"],
                comps["w_personal"],
                comps["w_global"],
                comps["w_personal_std"],
                comps["w_personal_rec"],
                comps["wait_pair"],
                comps["cooldown"],
            )
        )

    rows.sort(key=lambda t: t[1], reverse=True)

    print(f"\nWeights for {u.user_id} (native={u.native_language}, status={u.status}):")
    print("partner_id | raw | pu_std | pu_rec | pu_total | pv_total | global | wP | wG | wStd | wRec | wait | cd")
    for pid, raw, pu_s, pu_r, pu_t, pv_t, sg, wP, wG, wStd, wRec, wp, cd in rows:
        print(
            f"{pid:10s} | {raw:6.3f} | {pu_s:6.3f} | {pu_r:6.3f} | {pu_t:8.3f} | {pv_t:8.3f} | "
            f"{sg:6.3f} | {wP:4.2f} | {wG:4.2f} | {wStd:4.2f} | {wRec:4.2f} | {wp:5.2f} | {cd:4.2f}"
        )
