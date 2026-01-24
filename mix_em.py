"""EM utilities for mixture fitting."""
from __future__ import annotations

import math

import numpy as np


def logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """Compute log-sum-exp along an axis."""
    m = np.max(a, axis=axis, keepdims=True)
    if not np.all(np.isfinite(m)):
        return np.full(a.shape[0], np.nan)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def softplus(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def log_cosh(x: np.ndarray) -> np.ndarray:
    """Compute stable log(cosh(x))."""
    return np.logaddexp(x, -x) - math.log(2.0)
