"""
LinUCB algorithm implementation for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from ..models.bandit import BanditModel
except ImportError:
    from models.bandit import BanditModel


def create_bandit(dim: int, init_theta: Optional[np.ndarray] = None) -> BanditModel:
    """Create a new bandit model with the given dimension."""
    A = np.eye(dim, dtype=float)
    if init_theta is None:
        b = np.zeros(dim, dtype=float)
    else:
        init_theta = np.asarray(init_theta, dtype=float).reshape(-1)
        if init_theta.shape[0] != dim:
            raise ValueError(f"init_theta dim mismatch: expected {dim}, got {init_theta.shape[0]}")
        b = (A @ init_theta).astype(float)
    return BanditModel(A=A, b=b)


def clone_bandit(bandit: BanditModel) -> BanditModel:
    """Create a copy of a bandit model."""
    return BanditModel(A=bandit.A.copy(), b=bandit.b.copy())


def _safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safely solve the linear system A*theta = b."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def ucb_score(bandit: BanditModel, x: np.ndarray, alpha: float) -> float:
    """
    LinUCB score = theta^T x + alpha * sqrt(x^T A^{-1} x)
    """
    A = bandit.A
    b = bandit.b
    theta = _safe_solve(A, b)

    # Compute A^{-1} x robustly
    try:
        A_inv_x = np.linalg.solve(A, x)
    except np.linalg.LinAlgError:
        A_inv_x = np.linalg.pinv(A) @ x

    mean = float(theta @ x)
    unc = float(math.sqrt(max(float(x @ A_inv_x), 0.0)))
    return mean + alpha * unc


def bandit_update(bandit: BanditModel, x: np.ndarray, reward: float) -> None:
    """
    Standard LinUCB update:
    A <- A + x x^T
    b <- b + r x
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    bandit.A += x @ x.T
    bandit.b += float(reward) * x.ravel()


def bandit_update_decayed(bandit: BanditModel, x: np.ndarray, reward: float, gamma: float) -> None:
    """
    Exponential forgetting update (recent history matters more):
    A <- gamma*A + x x^T
    b <- gamma*b + r*x
    """
    g = float(gamma)
    if not (0.0 < g <= 1.0):
        raise ValueError("gamma must be in (0,1].")

    x = np.asarray(x, dtype=float).reshape(-1, 1)
    bandit.A *= g
    bandit.A += x @ x.T

    bandit.b *= g
    bandit.b += float(reward) * x.ravel()
