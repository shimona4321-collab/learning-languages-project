"""
Configuration constants for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

import os
from typing import List

# =====================================================
# State Persistence
# =====================================================

STATE_FILE = os.environ.get("LANGMATCH_STATE_FILE", "langmatch_state.json")

# =====================================================
# Languages
# =====================================================

LANG_HE = "Hebrew"
LANG_EN = "English"

# =====================================================
# Topics
# =====================================================

TOPICS: List[str] = ["Travel", "Chess"]

# =====================================================
# Bandit Parameters
# =====================================================

ALPHA_PERSONAL: float = 0.5
ALPHA_GLOBAL: float = 0.5

# =====================================================
# Waiting-time Fairness
# =====================================================

# Edge multiplier per user = (1 + EPS_WAITING * waiting_rounds)
EPS_WAITING: float = 0.5

# =====================================================
# Personal Recency Bandit
# =====================================================

# Exponential forgetting factor (closer to 1 = slower forgetting)
PERSONAL_RECENCY_GAMMA: float = 0.9

# =====================================================
# Default Scoring Weights
# =====================================================

DEFAULT_W_PERSONAL: float = 0.7
DEFAULT_W_GLOBAL: float = 0.3
DEFAULT_W_PERSONAL_STD: float = 0.5
DEFAULT_W_PERSONAL_REC: float = 0.5

# =====================================================
# Cooldown Behavior
# =====================================================

# Factor in [COOLDOWN_MIN, 1], increases by COOLDOWN_GROWTH per matching round
COOLDOWN_MIN: float = 0
COOLDOWN_GROWTH: float = 1

# =====================================================
# Gaussian Defaults (raw scale 0..10)
# =====================================================

GAUSS_TARGET_MEAN: float = 5.0
GAUSS_TARGET_STD: float = 3.0
GAUSS_AVAIL_MEAN: float = 5.0
GAUSS_AVAIL_STD: float = 3.0
GAUSS_TOPIC_MEAN: float = 5.0
GAUSS_TOPIC_STD: float = 3.0

# =====================================================
# Feature Dimensions
# =====================================================

# Personal: partner difficulty + partner availability + partner topic interests
USER_FEATURE_DIM = 2 + len(TOPICS)

# Global: avg difficulty + avg availability + shared topic interest products
PAIR_FEATURE_DIM = 2 + len(TOPICS)
