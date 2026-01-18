"""
User model for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

try:
    from ..config import TOPICS
except ImportError:
    from config import TOPICS


@dataclass
class User:
    user_id: str
    native_language: str
    target_language: str
    target_level_raw: int  # 0..10 (knowledge in target language)
    availability_raw: int  # 0..10
    topic_interest_raw: Dict[str, int]  # each 0..10
    status: str = "no_offer"  # "no_offer" | "has_offer" | "matched"
    current_partner_id: Optional[str] = None
    waiting_rounds_without_offer: int = 0

    @property
    def target_difficulty(self) -> float:
        # difficulty in [0,1], where 1 = no knowledge, 0 = fluent
        val = (10 - int(self.target_level_raw)) / 10.0
        return float(min(max(val, 0.0), 1.0))

    @property
    def availability_norm(self) -> float:
        val = int(self.availability_raw) / 10.0
        return float(min(max(val, 0.0), 1.0))

    def topic_interest_norm(self, topic: str) -> float:
        raw = int(self.topic_interest_raw.get(topic, 0))
        val = raw / 10.0
        return float(min(max(val, 0.0), 1.0))

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "User":
        # Back-compat: ensure topics exist
        topic_interest_raw = dict(d.get("topic_interest_raw", {}))
        for t in TOPICS:
            topic_interest_raw.setdefault(t, 0)
        d2 = dict(d)
        d2["topic_interest_raw"] = topic_interest_raw
        # Back-compat: ensure fields exist
        d2.setdefault("status", "no_offer")
        d2.setdefault("current_partner_id", None)
        d2.setdefault("waiting_rounds_without_offer", 0)
        return User(**d2)
