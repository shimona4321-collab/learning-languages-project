"""
Application state model for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .user import User
from .bandit import BanditModel
from .proposal import Proposal

try:
    from ..config import (
        DEFAULT_W_PERSONAL,
        DEFAULT_W_GLOBAL,
        DEFAULT_W_PERSONAL_STD,
        DEFAULT_W_PERSONAL_REC,
    )
except ImportError:
    from config import (
        DEFAULT_W_PERSONAL,
        DEFAULT_W_GLOBAL,
        DEFAULT_W_PERSONAL_STD,
        DEFAULT_W_PERSONAL_REC,
    )


@dataclass
class AppState:
    users: Dict[str, User] = field(default_factory=dict)
    global_bandit: Optional[BanditModel] = None

    # Standard personal bandits
    user_bandits: Dict[str, BanditModel] = field(default_factory=dict)

    # Recency-weighted personal bandits (exponential forgetting)
    user_bandits_recent: Dict[str, BanditModel] = field(default_factory=dict)

    proposals: Dict[str, Proposal] = field(default_factory=dict)  # key=pair_key
    pair_cooldowns: Dict[str, float] = field(default_factory=dict)
    round_index: int = 0  # increments each matching round (useful for debugging)

    # Scoring weights (persisted)
    w_personal: float = DEFAULT_W_PERSONAL
    w_global: float = DEFAULT_W_GLOBAL
    w_personal_std: float = DEFAULT_W_PERSONAL_STD
    w_personal_rec: float = DEFAULT_W_PERSONAL_REC

    def to_dict(self) -> Dict:
        return {
            "users": {uid: u.to_dict() for uid, u in self.users.items()},
            "global_bandit": self.global_bandit.to_dict() if self.global_bandit else None,
            "user_bandits": {uid: b.to_dict() for uid, b in self.user_bandits.items()},
            "user_bandits_recent": {uid: b.to_dict() for uid, b in self.user_bandits_recent.items()},
            "proposals": {k: p.to_dict() for k, p in self.proposals.items()},
            "pair_cooldowns": dict(self.pair_cooldowns),
            "round_index": int(self.round_index),
            # weights
            "w_personal": float(self.w_personal),
            "w_global": float(self.w_global),
            "w_personal_std": float(self.w_personal_std),
            "w_personal_rec": float(self.w_personal_rec),
        }

    @staticmethod
    def from_dict(d: Dict) -> "AppState":
        st = AppState()
        for uid, ud in d.get("users", {}).items():
            st.users[uid] = User.from_dict(ud)

        gb = d.get("global_bandit")
        st.global_bandit = BanditModel.from_dict(gb) if gb else None

        for uid, bd in d.get("user_bandits", {}).items():
            st.user_bandits[uid] = BanditModel.from_dict(bd)

        # Back-compat: may not exist
        for uid, bd in d.get("user_bandits_recent", {}).items():
            st.user_bandits_recent[uid] = BanditModel.from_dict(bd)

        for k, pd in d.get("proposals", {}).items():
            st.proposals[k] = Proposal.from_dict(pd)

        st.pair_cooldowns = {k: float(v) for k, v in d.get("pair_cooldowns", {}).items()}
        st.round_index = int(d.get("round_index", 0))

        # Back-compat defaults for weights
        st.w_personal = float(d.get("w_personal", DEFAULT_W_PERSONAL))
        st.w_global = float(d.get("w_global", DEFAULT_W_GLOBAL))
        st.w_personal_std = float(d.get("w_personal_std", DEFAULT_W_PERSONAL_STD))
        st.w_personal_rec = float(d.get("w_personal_rec", DEFAULT_W_PERSONAL_REC))

        return st
