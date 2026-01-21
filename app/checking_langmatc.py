"""
LANGMATCH Experiments Runner (menu-based, NOT pytest)

Run with:
  python -m app.checking_langmatc

Alternative (also works):
  python app/checking_langmatc.py

Do NOT run with pytest.

Key goals:
- Run 3 experiments fully in memory (NO JSON persistence).
- By DEFAULT: do NOT override any langmatch parameters.
- Each experiment starts from a FRESH lm.AppState (users/bandits/proposals/cooldowns reset).
- English users ALWAYS ACCEPT in all experiments (so global bandit b updates).
- Before each new matching round, we clear any leftover proposals and cancel any matched pairs
  so everyone participates in the next round.

IMPORTANT:
- This file is intentionally verbose and interactive: it prints summaries, shows plots,
  and asks whether to save each figure.
"""

from __future__ import annotations

import io
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------
# Robust import of langmatch (same folder + parent folder on sys.path)
# -------------------------------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_PARENT_DIR = _THIS_DIR.parent

for p in (str(_THIS_DIR), str(_PARENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Create plots directory at initialization
import os
_PLOTS_DIR = _THIS_DIR / "plots"
if not _PLOTS_DIR.exists():
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created plots directory: {_PLOTS_DIR}")

try:
    import langmatch as lm  # type: ignore
except Exception:
    print("-" * 100)
    print("ERROR: Failed to import 'langmatch' by module name.")
    print("Make sure this file is in the same folder as langmatch.py, OR that parent folder is on sys.path.")
    print("sys.path:")
    for s in sys.path:
        print(" ", s)
    print("-" * 100)
    raise

# -------------------------------------------------------------------------------------------------
# Optional access to scoring module for temporary parameter overrides in experiments
# -------------------------------------------------------------------------------------------------
try:
    import matching.scoring as scoring  # type: ignore
except Exception:
    scoring = None  # type: ignore

user_mode_module = None
try:
    import importlib
    user_mode_module = importlib.import_module("ui.user_mode")
except Exception:
    user_mode_module = None


def _temp_setattrs(patches: List[Tuple[Any, str, Any]]):
    """Context manager: temporarily set module/object attributes, then restore."""
    class _CM:
        def __init__(self, patches_):
            self.patches = [(obj, name, val) for (obj, name, val) in patches_ if obj is not None]
            self.old = []

        def __enter__(self):
            for obj, name, val in self.patches:
                self.old.append((obj, name, getattr(obj, name)))
                setattr(obj, name, val)
            return self

        def __exit__(self, exc_type, exc, tb):
            for obj, name, oldv in reversed(self.old):
                try:
                    setattr(obj, name, oldv)
                except Exception:
                    pass
            return False

    return _CM(patches)


def _mean_ignore_nan(vals: List[float]) -> float:
    a = [float(x) for x in vals if x is not None and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    return float(np.mean(a)) if a else float("nan")


def _compute_sat_rounds_after_switch(
    post_switch_series: List[float],
    eps: float = 0.02,
    tail_window: int = 8,
) -> int:
    """
    Switch Adaptation Time (SAT):
      Let a_t be a post-switch metric series (higher is better).
      Define target a* as mean of the last 'tail_window' points (ignoring NaNs).
      SAT is the smallest k>=1 such that a_{k} >= a* - eps.
      Returns k (number of rounds AFTER the switch). If never reaches, returns len(post_switch_series)+1.
    """
    if not post_switch_series:
        return 1
    tail = post_switch_series[-max(1, min(tail_window, len(post_switch_series))):]
    target = _mean_ignore_nan(tail)
    if np.isnan(target):
        return len(post_switch_series) + 1

    thr = float(target) - float(eps)
    for i, a in enumerate(post_switch_series):
        if a is None:
            continue
        a = float(a)
        if not (np.isnan(a) or np.isinf(a)) and a >= thr:
            return i + 1  # 1-based rounds after switch
    return len(post_switch_series) + 1


def plot_bars(
    title: str,
    labels: List[str],
    values: List[float],
    ylabel: str,
    default_save_name: str = "bars.png",
) -> None:
    """Simple bar plot with the same 'ask to save' UX as the other plots."""
    fig, ax = plt.subplots()
    x = list(range(len(values)))
    ax.bar(x, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Use absolute path for saving
    save_path = _THIS_DIR / default_save_name if not os.path.isabs(default_save_name) else Path(default_save_name)
    save_dir = save_path.parent
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Save first, then show
    print(f"\nSaving graph to: {save_path}", flush=True)
    try:
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        print(f"Graph saved successfully!", flush=True)
    except Exception as e:
        print(f"Error saving graph: {e}", flush=True)

    print("[Close the graph window to continue...]", flush=True)
    plt.show()
    plt.close(fig)


def input_choice(prompt: str, valid: Iterable[str]) -> str:
    valid_set = {v.strip().upper() for v in valid}
    while True:
        s = input(prompt).strip().upper()
        if s in valid_set:
            return s
        print(f"Invalid choice. Valid options: {', '.join(sorted(valid_set))}")


def input_int(prompt: str, default: Optional[int] = None, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            v = int(default)
        else:
            try:
                v = int(s)
            except Exception:
                print("Please enter an integer.")
                continue
        if min_v is not None and v < min_v:
            print(f"Value must be >= {min_v}.")
            continue
        if max_v is not None and v > max_v:
            print(f"Value must be <= {max_v}.")
            continue
        return v


def rand_int_inclusive(rng: random.Random, a: int, b: int) -> int:
    return int(rng.randint(int(a), int(b)))


class SuppressImplOutput:
    """
    Context manager that suppresses prints from inside langmatch internals.
    We still show our own summaries.

    Controlled by 'suppress_impl' flag in each experiment.
    """

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self._redir = None
        self._buf = None

    def __enter__(self):
        if self.enabled:
            self._buf = io.StringIO()
            self._redir = redirect_stdout(self._buf)
            self._redir.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self._redir is not None:
            self._redir.__exit__(exc_type, exc, tb)


def show_current_params() -> None:
    """
    Print the CURRENT values in langmatch (no overrides here).
    Includes the exact variables requested.
    """
    requested_keys = [
        "ALPHA_PERSONAL",
        "ALPHA_GLOBAL",
        "EPS_WAITING",
        "PERSONAL_RECENCY_GAMMA",
        "DEFAULT_W_PERSONAL",
        "DEFAULT_W_GLOBAL",
        "DEFAULT_W_PERSONAL_STD",
        "DEFAULT_W_PERSONAL_REC",
    ]

    print("\nCurrent langmatch parameters (as loaded right now):")
    for k in requested_keys:
        if hasattr(lm, k):
            print(f"  {k} = {getattr(lm, k)!r}")
        else:
            print(f"  {k} = (NOT FOUND in langmatch.py)")
    print("NOTE: If you change parameters in langmatch.py, restart this runner to reload them.")


def fmt_vec(v: np.ndarray, precision: int = 4) -> str:
    a = np.asarray(v, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{x:.{precision}f}" for x in a.tolist()) + "]"


def fmt_mat(M: np.ndarray, precision: int = 4) -> str:
    A = np.asarray(M, dtype=float)
    rows = []
    for r in A:
        rows.append("[" + ", ".join(f"{x:.{precision}f}" for x in r.tolist()) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


# -------------------------------------------------------------------------------------------------
# Small plotting helpers (interactive save prompt)
# -------------------------------------------------------------------------------------------------

def _rolling_mean_causal(x: List[float], window: int) -> List[float]:
    """Causal rolling mean: y[t] = mean(x[max(0,t-w+1):t+1]). Keeps NaNs as NaNs."""
    y: List[float] = []
    for i in range(len(x)):
        left = max(0, i - window + 1)
        seg = x[left:i + 1]
        seg2 = [v for v in seg if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        if len(seg2) == 0:
            y.append(float("nan"))
        else:
            y.append(float(np.mean(seg2)))
    return y


def _downsample(x: List[int], y: List[float], max_points: int = 300) -> Tuple[List[int], List[float]]:
    """Downsample by uniform stride for plotting. Keeps first+last."""
    n = len(x)
    if n <= max_points:
        return x, y
    stride = int(np.ceil(n / max_points))
    xs = x[::stride]
    ys = y[::stride]
    if xs[-1] != x[-1]:
        xs.append(x[-1])
        ys.append(y[-1])
    return xs, ys


def plot_lines_smoothed(
    title: str,
    x: List[int],
    series: List[List[float]],
    labels: List[str],
    ylabel: str,
    phase_switch_x: Optional[int] = None,
    keep_gaps_mask: Optional[List[Optional[List[bool]]]] = None,  # per-series mask; True means "force NaN here"
    default_save_name: str = "figure.png",
) -> None:
    """
    One figure with multiple lines:
    - Applies causal rolling mean smoothing (window chosen dynamically)
    - Downsamples for large n to avoid "painted" plots
    - Offers saving after display
    - Optionally draws a vertical phase switch line
    """
    assert len(series) == len(labels), "series/labels length mismatch"

    n = len(x)
    if n <= 30:
        window = 1
    elif n <= 80:
        window = 3
    elif n <= 200:
        window = 7
    else:
        window = 15

    fig, ax = plt.subplots()

    for idx, y in enumerate(series):
        y = list(y)
        if keep_gaps_mask is not None and idx < len(keep_gaps_mask) and keep_gaps_mask[idx] is not None:
            mask = keep_gaps_mask[idx]
            for i in range(min(len(y), len(mask))):
                if mask[i]:
                    y[i] = float("nan")

        ys = _rolling_mean_causal(y, window=window)

        xs_plot, ys_plot = _downsample(x, ys, max_points=350)
        ax.plot(xs_plot, ys_plot, label=labels[idx])

    if phase_switch_x is not None:
        ax.axvline(float(phase_switch_x), linestyle="--", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    # Show graph first
    print("\n[Close the graph window to continue...]", flush=True)
    plt.show()

    # After closing, ask to save
    save_choice = input("Save this graph? (Y/N) [Y]: ").strip().upper()
    if save_choice != "N":
        save_path = _THIS_DIR / default_save_name if not os.path.isabs(default_save_name) else Path(default_save_name)
        save_dir = save_path.parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
            print(f"Graph saved to: {save_path}")
        except Exception as e:
            print(f"Error saving graph: {e}")
    else:
        print("Graph not saved.")

    plt.close(fig)


def plot_histogram(
    title: str,
    values: List[int],
    xlabel: str,
    ylabel: str,
    default_save_name: str = "hist.png",
) -> None:
    fig, ax = plt.subplots()
    ax.hist(values, bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Use absolute path for saving
    save_path = _THIS_DIR / default_save_name if not os.path.isabs(default_save_name) else Path(default_save_name)
    save_dir = save_path.parent
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Save first, then show
    print(f"\nSaving graph to: {save_path}", flush=True)
    try:
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        print(f"Graph saved successfully!", flush=True)
    except Exception as e:
        print(f"Error saving graph: {e}", flush=True)

    print("[Close the graph window to continue...]", flush=True)
    plt.show()
    plt.close(fig)


# -------------------------------------------------------------------------------------------------
# State helpers (fresh state, add users, proposal handling)
# -------------------------------------------------------------------------------------------------

def fresh_state() -> Any:
    """
    Fresh AppState. No JSON. No carryover.
    Resets: users, global bandit, personal bandits (std+recent), proposals, cooldowns, round_index.
    """
    state = lm.AppState()
    state.users.clear()
    state.user_bandits.clear()
    state.user_bandits_recent.clear()
    state.proposals.clear()
    state.pair_cooldowns.clear()
    state.round_index = 0

    state.global_bandit = lm.create_bandit(lm.PAIR_FEATURE_DIM)
    return state


def add_user(
    state: Any,
    user_id: str,
    native: str,
    target: str,
    target_level_raw: int,
    availability_raw: int,
    travel_raw: int,
    chess_raw: int,
) -> Any:
    u = lm.User(
        user_id=user_id,
        native_language=native,
        target_language=target,
        target_level_raw=int(target_level_raw),
        availability_raw=int(availability_raw),
        topic_interest_raw={"Travel": int(travel_raw), "Chess": int(chess_raw)},
    )
    state.users[user_id] = u
    lm.ensure_user_bandits(state, user_id)
    return u


def find_proposal_for_user(state: Any, user_id: str) -> Optional[Any]:
    for p in state.proposals.values():
        if p.user1_id == user_id or p.user2_id == user_id:
            return p
    return None


def other_in_proposal(prop: Any, user_id: str) -> Optional[str]:
    if prop.user1_id == user_id:
        return prop.user2_id
    if prop.user2_id == user_id:
        return prop.user1_id
    return None


def cancel_all_matches(state: Any) -> None:
    for u in state.users.values():
        if u.status == "matched":
            u.status = "idle"
            u.current_partner_id = None


def clear_all_proposals(state: Any) -> None:
    state.proposals.clear()


def sample_users_long(title: str, users: List[Any], max_n: int = 3) -> None:
    print(f"\n{title} (sample {min(max_n, len(users))}/{len(users)}):")
    for u in users[:max_n]:
        print(
            f"  {u.user_id}: native={u.native_language} target={u.target_language}"
            f" | level={u.target_level_raw} avail={u.availability_raw}"
            f" | travel={u.topic_interest_raw.get('Travel', 0)} chess={u.topic_interest_raw.get('Chess', 0)}"
        )


def run_round_and_collect_offers(
    state: Any,
    hebrew_ids: List[str],
    english_ids: List[str],
    hebrew_decision_fn,  # (he_id, partner_user)->bool
    suppress_impl: bool,
    english_offer_counter: Optional[Dict[str, int]] = None,  # for Experiment 1 histogram
) -> List[Tuple[str, str, float, float, bool]]:
    """
    One round:
    - Prepare clean state (cancel matches + clear proposals)
    - Run matching
    - (Optional) count how many offers each English user received this round
    - Hebrew decides accept/reject
    - English ALWAYS accepts (so matches happen when Hebrew accepts)
    - Cancel any matches at end of round (so next round is free)

    Returns list of offers:
      (he_id, en_id, chess_norm, travel_norm, hebrew_accepted)
    """
    # 0) Ensure clean
    clear_all_proposals(state)
    cancel_all_matches(state)

    # 1) Run matching round
    with SuppressImplOutput(enabled=suppress_impl):
        lm.run_matching_round(state)

    offers: List[Tuple[str, str, float, float, bool]] = []

    # 2) For each Hebrew user, find their proposal (if any), decide, respond, then English accepts
    for he_id in hebrew_ids:
        prop = find_proposal_for_user(state, he_id)
        if prop is None:
            continue
        en_id = other_in_proposal(prop, he_id)
        if en_id is None:
            continue

        # (Optional) count how many times each English user got an offer this round
        if english_offer_counter is not None:
            english_offer_counter[en_id] = int(english_offer_counter.get(en_id, 0)) + 1

        partner = state.users.get(en_id)
        if partner is None:
            continue

        # Extract normalized topics for reporting
        chess_norm = float(partner.topic_interest_raw.get("Chess", 0)) / 10.0
        travel_norm = float(partner.topic_interest_raw.get("Travel", 0)) / 10.0

        he_accept = bool(hebrew_decision_fn(he_id, partner))

        with SuppressImplOutput(enabled=suppress_impl):
            lm.handle_proposal_response(state, state.users[he_id], accepted=he_accept)

        # English ALWAYS accepts (so global bandit b updates and match occurs if Hebrew accepted)
        if en_id in state.users:
            with SuppressImplOutput(enabled=suppress_impl):
                lm.handle_proposal_response(state, state.users[en_id], accepted=True)

        offers.append((he_id, en_id, chess_norm, travel_norm, he_accept))

    # 3) Cancel all matches so everyone returns next round
    cancel_all_matches(state)

    return offers


# -------------------------------------------------------------------------------------------------
# Experiment 2: Exploration ON vs OFF (Novelty Rate)
# -------------------------------------------------------------------------------------------------

def run_experiment_2(short: bool) -> None:
    print("\n" + "=" * 100)
    print("Experiment 2: Exploration ON vs OFF (Novelty Rate)")
    print("=" * 100)
    show_current_params()

    if not short:
        print(
            "\nGoal:\n"
            "  Compare exploration OFF vs ON using a mathematical exploration metric: Novelty Rate.\n"
            "Metric (per round):\n"
            "  NoveltyRate_t = (# offers to Hebrew users where the suggested English partner was never suggested to that Hebrew user before)\n"
            "                 / (# offers to Hebrew users in that round).\n"
            "Setup:\n"
            "  Hebrew accepts iff partner Chess_raw > threshold; English users ALWAYS accept.\n"
            "  Matches are cancelled before the next round so everyone participates again.\n"
        )

    seed = input_int("seed [111]: ", default=111)
    n_he = input_int("n_hebrew [10]: ", default=10, min_v=1)
    n_en = input_int("n_english [100]: ", default=100, min_v=1)
    rounds = input_int("rounds [50]: ", default=50, min_v=1)
    chess_threshold_raw = input_int("Hebrew accept threshold: Chess_raw > ?  [6]: ", default=6, min_v=0, max_v=10)
    suppress_impl = (input_choice("Suppress internal implementation prints? (Y/N) [Y]: >", ["Y", "N", ""]) != "N")

    # Reproducible population generation (shared across both cases)
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    he_lang = "Hebrew"
    en_lang = "English"

    hebrew_specs: List[Tuple[str, str, str, int, int, int, int]] = []
    for i in range(1, n_he + 1):
        hebrew_specs.append(
            (
                f"he_{i:04d}",
                he_lang,
                en_lang,
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
            )
        )

    english_specs: List[Tuple[str, str, str, int, int, int, int]] = []
    for i in range(1, n_en + 1):
        english_specs.append(
            (
                f"en_{i:04d}",
                en_lang,
                he_lang,
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
            )
        )

    def hebrew_decision(_: str, partner: Any) -> bool:
        return int(partner.topic_interest_raw.get("Chess", 0)) > chess_threshold_raw

    x_rounds = list(range(1, rounds + 1))

    # Cache original alphas (so ON really means "your current settings")
    orig_alpha_p = float(getattr(lm, "ALPHA_PERSONAL", 0.0))
    orig_alpha_g = float(getattr(lm, "ALPHA_GLOBAL", 0.0))

    def run_case(case_name: str, alpha_personal: float, alpha_global: float) -> List[float]:
        # Temporarily override both langmatch constants and scoring constants (since scoring imports constants)
        patches = [
            (lm, "ALPHA_PERSONAL", float(alpha_personal)),
            (lm, "ALPHA_GLOBAL", float(alpha_global)),
        ]
        if scoring is not None:
            patches += [
                (scoring, "ALPHA_PERSONAL", float(alpha_personal)),
                (scoring, "ALPHA_GLOBAL", float(alpha_global)),
            ]

        with _temp_setattrs(patches):
            state = fresh_state()

            hebrew_ids: List[str] = []
            english_ids: List[str] = []

            for (uid, nat, targ, lvl, av, tr, ch) in hebrew_specs:
                add_user(state, user_id=uid, native=nat, target=targ, target_level_raw=lvl, availability_raw=av, travel_raw=tr, chess_raw=ch)
                hebrew_ids.append(uid)

            for (uid, nat, targ, lvl, av, tr, ch) in english_specs:
                add_user(state, user_id=uid, native=nat, target=targ, target_level_raw=lvl, availability_raw=av, travel_raw=tr, chess_raw=ch)
                english_ids.append(uid)

            # Track novelty per Hebrew user
            seen: Dict[str, set] = {hid: set() for hid in hebrew_ids}
            novelty_rate_per_round: List[float] = []

            for _t in range(rounds):
                offers = run_round_and_collect_offers(
                    state=state,
                    hebrew_ids=hebrew_ids,
                    english_ids=english_ids,
                    hebrew_decision_fn=hebrew_decision,
                    suppress_impl=suppress_impl,
                    english_offer_counter=None,
                )

                # offers: List[(he_id, en_id, chess_norm, travel_norm, accepted)]
                total = len(offers)
                if total <= 0:
                    novelty_rate_per_round.append(float("nan"))
                    continue

                nov = 0
                for he_id, en_id, _ch, _tr, _acc in offers:
                    if en_id not in seen[he_id]:
                        nov += 1
                        seen[he_id].add(en_id)

                novelty_rate_per_round.append(float(nov) / float(total))

            if not short:
                print(f"  Case '{case_name}': mean NoveltyRate = {_mean_ignore_nan(novelty_rate_per_round):.4f}")

            return novelty_rate_per_round

    novelty_off = run_case("Exploration OFF (alpha≈0)", alpha_personal=0.001, alpha_global=0.001)
    novelty_on = run_case("Exploration ON (current alphas)", alpha_personal=orig_alpha_p, alpha_global=orig_alpha_g)

    plot_lines_smoothed(
        title="Experiment 2: Novelty Rate — Exploration OFF vs ON",
        x=x_rounds,
        series=[novelty_off, novelty_on],
        labels=[
            "Exploration OFF (ALPHA≈0)",
            f"Exploration ON (ALPHA={orig_alpha_p:g})",
        ],
        ylabel="Novelty Rate (0..1)",
        default_save_name="plots/exp2_novelty_off_vs_on.png",
    )

    print("\nSummary (Experiment 2, Novelty Rate)")
    print(f"  mean NoveltyRate (OFF) = {_mean_ignore_nan(novelty_off):.4f}")
    print(f"  mean NoveltyRate (ON)  = {_mean_ignore_nan(novelty_on):.4f}")


# -------------------------------------------------------------------------------------------------
# Experiment 3: Personalization ON vs OFF (Preference Alignment Score)
# -------------------------------------------------------------------------------------------------

def run_experiment_3(short: bool) -> None:
    print("\n" + "=" * 100)
    print("Experiment 3: Personalization ON vs OFF (Preference Alignment Score)")
    print("=" * 100)
    show_current_params()

    if not short:
        print(
            "\nGoal:\n"
            "  Compare personalization OFF vs ON using a mathematical personalization metric: Preference Alignment Score (PAS).\n"
            "Metric (per round):\n"
            "  For each Hebrew offer (u -> v), define z(u,v)=Chess_norm(v) if u is Chess-pref, else Travel_norm(v).\n"
            "  PAS_t = average z(u,v) over all offers to Hebrew users in round t.\n"
            "Setup:\n"
            "  - Chess-pref group: accept iff partner Chess_raw > chess_threshold_raw\n"
            "  - Travel-pref group: accept iff partner Travel_raw > travel_threshold_raw\n"
            "  English users ALWAYS accept. Matches cancelled each round.\n"
        )

    seed = input_int("seed [222]: ", default=222)
    n_he_chess = input_int("n_hebrew_chess_pref [5]: ", default=5, min_v=1)
    n_he_travel = input_int("n_hebrew_travel_pref [5]: ", default=5, min_v=1)
    n_en = input_int("n_english [100]: ", default=100, min_v=1)
    rounds = input_int("rounds [50]: ", default=50, min_v=1)
    chess_threshold_raw = input_int("Chess-pref accept threshold: Chess_raw > ?  [6]: ", default=6, min_v=0, max_v=10)
    travel_threshold_raw = input_int("Travel-pref accept threshold: Travel_raw > ? [6]: ", default=6, min_v=0, max_v=10)
    suppress_impl = (input_choice("Suppress internal implementation prints? (Y/N) [Y]: >", ["Y", "N", ""]) != "N")

    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    he_lang = "Hebrew"
    en_lang = "English"

    # Generate one shared population (so OFF vs ON compares apples-to-apples)
    chess_pref_specs: List[Tuple[str, str, str, int, int, int, int]] = []
    travel_pref_specs: List[Tuple[str, str, str, int, int, int, int]] = []
    english_specs: List[Tuple[str, str, str, int, int, int, int]] = []

    for i in range(1, n_he_chess + 1):
        chess_pref_specs.append(
            (
                f"heC_{i:04d}",
                he_lang,
                en_lang,
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
            )
        )

    for i in range(1, n_he_travel + 1):
        travel_pref_specs.append(
            (
                f"heT_{i:04d}",
                he_lang,
                en_lang,
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
            )
        )

    for i in range(1, n_en + 1):
        english_specs.append(
            (
                f"en_{i:04d}",
                en_lang,
                he_lang,
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
            )
        )

    chess_pref_ids = [s[0] for s in chess_pref_specs]
    travel_pref_ids = [s[0] for s in travel_pref_specs]
    hebrew_ids = chess_pref_ids + travel_pref_ids
    english_ids = [s[0] for s in english_specs]

    def hebrew_decision(he_id: str, partner: Any) -> bool:
        if he_id in chess_pref_ids:
            return int(partner.topic_interest_raw.get("Chess", 0)) > chess_threshold_raw
        return int(partner.topic_interest_raw.get("Travel", 0)) > travel_threshold_raw

    x_rounds = list(range(1, rounds + 1))

    def run_case(case_name: str, personalization_on: bool) -> List[float]:
        state = fresh_state()

        # Toggle personalization by weights (w_personal=0 => purely global scoring)
        if not personalization_on:
            state.w_personal = 0.0
            state.w_global = 1.0

        for (uid, nat, targ, lvl, av, tr, ch) in chess_pref_specs:
            add_user(state, user_id=uid, native=nat, target=targ, target_level_raw=lvl, availability_raw=av, travel_raw=tr, chess_raw=ch)

        for (uid, nat, targ, lvl, av, tr, ch) in travel_pref_specs:
            add_user(state, user_id=uid, native=nat, target=targ, target_level_raw=lvl, availability_raw=av, travel_raw=tr, chess_raw=ch)

        for (uid, nat, targ, lvl, av, tr, ch) in english_specs:
            add_user(state, user_id=uid, native=nat, target=targ, target_level_raw=lvl, availability_raw=av, travel_raw=tr, chess_raw=ch)

        pas_per_round: List[float] = []

        for _t in range(rounds):
            offers = run_round_and_collect_offers(
                state=state,
                hebrew_ids=hebrew_ids,
                english_ids=english_ids,
                hebrew_decision_fn=hebrew_decision,
                suppress_impl=suppress_impl,
                english_offer_counter=None,
            )
            if not offers:
                pas_per_round.append(float("nan"))
                continue

            zs: List[float] = []
            for he_id, _en_id, chess_norm, travel_norm, _acc in offers:
                if he_id in chess_pref_ids:
                    zs.append(float(chess_norm))
                else:
                    zs.append(float(travel_norm))
            pas_per_round.append(float(np.mean(zs)) if zs else float("nan"))

        if not short:
            print(f"  Case '{case_name}': mean PAS = {_mean_ignore_nan(pas_per_round):.4f}")

        return pas_per_round

    pas_off = run_case("Personalization OFF (global-only)", personalization_on=False)
    pas_on = run_case("Personalization ON (global+personal)", personalization_on=True)

    plot_lines_smoothed(
        title="Experiment 3: Preference Alignment Score (PAS) — Personalization OFF vs ON",
        x=x_rounds,
        series=[pas_off, pas_on],
        labels=[
            "Personalization OFF (w_personal=0)",
            "Personalization ON (w_personal>0)",
        ],
        ylabel="PAS (0..1)",
        default_save_name="plots/exp3_pas_off_vs_on.png",
    )

    print("\nSummary (Experiment 3, PAS)")
    print(f"  mean PAS (OFF) = {_mean_ignore_nan(pas_off):.4f}")
    print(f"  mean PAS (ON)  = {_mean_ignore_nan(pas_on):.4f}")


# -------------------------------------------------------------------------------------------------
# Experiment 1: Random vs Bandit-based Matching (Preference Alignment Score)
# -------------------------------------------------------------------------------------------------

def run_experiment_1(short: bool) -> None:
    print("\n" + "=" * 100)
    print("Experiment 1: Random vs Bandit-based Matching (PAS)")
    print("=" * 100)
    show_current_params()

    if not short:
        print(
            "\nGoal:\n"
            "  Compare random matching vs bandit-based matching using Preference Alignment Score (PAS).\n"
            "Metric:\n"
            "  PAS_t = average Chess_raw(partner) / 10 over all offers in round t.\n"
            "  Higher PAS = partners offered have higher Chess scores.\n"
            "Setup:\n"
            "  Hebrew users accept iff partner Chess_raw > threshold (hidden preference).\n"
            "  English users ALWAYS accept.\n"
            "  Random matching: pairs are formed randomly (no learning).\n"
            "  Bandit matching: pairs are formed using the scoring system (learns from accepts/rejects).\n"
        )

    seed = input_int("seed [444]: ", default=444)
    n_he = input_int("n_hebrew [10]: ", default=10, min_v=1)
    n_en = input_int("n_english [100]: ", default=100, min_v=1)
    rounds = input_int("rounds [50]: ", default=50, min_v=1)
    chess_threshold_raw = input_int("Hebrew accept threshold: Chess_raw > ?  [5]: ", default=5, min_v=0, max_v=10)
    suppress_impl = (input_choice("Suppress internal implementation prints? (Y/N) [Y]: >", ["Y", "N", ""]) != "N")

    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    he_lang = "Hebrew"
    en_lang = "English"

    x_rounds = list(range(1, rounds + 1))

    hebrew_specs: List[Tuple[str, str, str, int, int, int, int]] = []
    for i in range(1, n_he + 1):
        hebrew_specs.append(
            (
                f"he_{i:04d}",
                he_lang,
                en_lang,
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
            )
        )

    english_specs: List[Tuple[str, str, str, int, int, int, int]] = []
    for i in range(1, n_en + 1):
        english_specs.append(
            (
                f"en_{i:04d}",
                en_lang,
                he_lang,
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
                rand_int_inclusive(rng, 0, 10),
            )
        )

    def hebrew_decision(_: str, partner: Any) -> bool:
        return int(partner.topic_interest_raw.get("Chess", 0)) > chess_threshold_raw

    def run_random_round(state: Any, hebrew_ids: List[str], english_ids: List[str], suppress: bool) -> List[Tuple[str, str, float, float, bool]]:
        """Random matching: shuffle English users and pair with Hebrew users."""
        clear_all_proposals(state)
        cancel_all_matches(state)

        shuffled_english = list(english_ids)
        random.shuffle(shuffled_english)

        offers: List[Tuple[str, str, float, float, bool]] = []
        for i, he_id in enumerate(hebrew_ids):
            if i >= len(shuffled_english):
                break
            en_id = shuffled_english[i]
            partner = state.users.get(en_id)
            if partner is None:
                continue

            chess_norm = float(partner.topic_interest_raw.get("Chess", 0)) / 10.0
            travel_norm = float(partner.topic_interest_raw.get("Travel", 0)) / 10.0
            he_accept = hebrew_decision(he_id, partner)
            offers.append((he_id, en_id, chess_norm, travel_norm, he_accept))

        return offers

    def run_case(case_name: str, use_bandit: bool) -> List[float]:
        state = fresh_state()

        hebrew_ids: List[str] = []
        english_ids: List[str] = []

        for (uid, nat, targ, lvl, av, tr, ch) in hebrew_specs:
            add_user(state, user_id=uid, native=nat, target=targ, target_level_raw=lvl, availability_raw=av, travel_raw=tr, chess_raw=ch)
            hebrew_ids.append(uid)

        for (uid, nat, targ, lvl, av, tr, ch) in english_specs:
            add_user(state, user_id=uid, native=nat, target=targ, target_level_raw=lvl, availability_raw=av, travel_raw=tr, chess_raw=ch)
            english_ids.append(uid)

        pas_per_round: List[float] = []

        for _t in range(rounds):
            if use_bandit:
                offers = run_round_and_collect_offers(
                    state=state,
                    hebrew_ids=hebrew_ids,
                    english_ids=english_ids,
                    hebrew_decision_fn=hebrew_decision,
                    suppress_impl=suppress_impl,
                    english_offer_counter=None,
                )
            else:
                offers = run_random_round(state, hebrew_ids, english_ids, suppress_impl)

            if not offers:
                pas_per_round.append(float("nan"))
                continue

            # PAS = average Chess score of partners offered
            chess_scores = [float(ch) for (_, _, ch, _, _) in offers]
            pas_per_round.append(float(np.mean(chess_scores)) if chess_scores else float("nan"))

        if not short:
            print(f"  Case '{case_name}': mean PAS = {_mean_ignore_nan(pas_per_round):.4f}")

        return pas_per_round

    pas_random = run_case("Random Matching", use_bandit=False)
    pas_bandit = run_case("Bandit-based Matching", use_bandit=True)

    plot_lines_smoothed(
        title="Experiment 1: PAS — Random vs Bandit-based Matching",
        x=x_rounds,
        series=[pas_random, pas_bandit],
        labels=[
            "Random Matching",
            "Bandit-based Matching",
        ],
        ylabel="PAS (0..1)",
        default_save_name="plots/exp1_pas_random_vs_bandit.png",
    )

    print("\nSummary (Experiment 1, PAS)")
    print(f"  mean PAS (Random) = {_mean_ignore_nan(pas_random):.4f}")
    print(f"  mean PAS (Bandit) = {_mean_ignore_nan(pas_bandit):.4f}")


# -------------------------------------------------------------------------------------------------
# Main menu (must be exactly this structure)
# -------------------------------------------------------------------------------------------------

def main_menu() -> None:
    while True:
        print("\n" + "=" * 100)
        print("LANGMATCH Experiments Runner (interactive)")
        print("=" * 100)
        print("1) Show current langmatch parameters")
        print("2) Run Experiment 1 (Random vs Bandit Matching - PAS)")
        print("3) Run Experiment 2 (Exploration ON vs OFF)")
        print("4) Run Experiment 3 (Personalization ON vs OFF)")
        print("Q) Quit")
        c = input_choice("Select: >", ["1", "2", "3", "4", "Q"])

        if c == "Q":
            print("Bye.")
            return
        elif c == "1":
            show_current_params()
        elif c == "2":
            run_experiment_1(short=False)
        elif c == "3":
            run_experiment_2(short=False)
        elif c == "4":
            run_experiment_3(short=False)

        input("\nPress ENTER to return to the menu...")


if __name__ == "__main__":
    main_menu()
