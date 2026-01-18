"""
Bandit model for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BanditModel:
    A: np.ndarray
    b: np.ndarray

    def to_dict(self) -> Dict:
        return {"A": self.A.tolist(), "b": self.b.tolist()}

    @staticmethod
    def from_dict(d: Dict) -> "BanditModel":
        A = np.array(d["A"], dtype=float)
        b = np.array(d["b"], dtype=float)
        return BanditModel(A=A, b=b)
