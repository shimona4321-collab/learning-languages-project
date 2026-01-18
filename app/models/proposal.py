"""
Proposal model for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class Proposal:
    user1_id: str
    user2_id: str
    score_at_offer: float = 0.0  # snapshot score used when proposal was created
    response1: Optional[bool] = None
    response2: Optional[bool] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "Proposal":
        d2 = dict(d)
        d2.setdefault("score_at_offer", 0.0)
        d2.setdefault("response1", None)
        d2.setdefault("response2", None)
        return Proposal(**d2)
