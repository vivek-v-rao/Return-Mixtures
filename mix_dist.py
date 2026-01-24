"""Distribution pdf/logpdf helpers for mixtures."""
from __future__ import annotations

import math

import numpy as np

from mix_em import log_cosh, softplus

try:
    from scipy.special import gammaln
except ImportError:  # pragma: no cover
    gammaln = None


def log_sech_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Log-pdf for hyperbolic secant."""
    z = (x - mu) / sd
    return math.log(0.5) - math.log(sd) - log_cosh(math.pi * z / 2.0)


def log_logistic_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Log-pdf for logistic."""
    z = (x - mu) / sd
    return -math.log(sd) - z - 2.0 * softplus(-z)


def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Pdf for univariate normal."""
    z = (x - mean) / sd
    return np.exp(-0.5 * z * z) / (sd * np.sqrt(2.0 * np.pi))


def log_t_pdf(x: np.ndarray, mu: float, sd: float, df: float) -> np.ndarray:
    """Log-pdf for Student t (location/scale)."""
    if gammaln is None:
        raise RuntimeError("scipy is required for Student t log-pdf")
    z = (x - mu) / sd
    log_norm = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * np.log(df * np.pi)
        - np.log(sd)
    )
    return log_norm - 0.5 * (df + 1.0) * np.log1p((z ** 2) / df)


def mixture_pdf_sech_1d(
    x: np.ndarray, weights: np.ndarray, means: np.ndarray, sds: np.ndarray
) -> np.ndarray:
    """Pdf for 1d sech mixture."""
    y = np.zeros_like(x, dtype=float)
    for w, mu, sd in zip(weights, means, sds):
        z = (x - mu) / sd
        y += w * (0.5 / sd) * (1.0 / np.cosh(math.pi * z / 2.0))
    return y


def mixture_pdf_logistic_1d(
    x: np.ndarray, weights: np.ndarray, means: np.ndarray, sds: np.ndarray
) -> np.ndarray:
    """Pdf for 1d logistic mixture."""
    y = np.zeros_like(x, dtype=float)
    for w, mu, sd in zip(weights, means, sds):
        z = (x - mu) / sd
        ez = np.exp(-z)
        y += w * (ez / (sd * (1.0 + ez) ** 2))
    return y


def mixture_pdf_t_1d(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    dfs: np.ndarray,
) -> np.ndarray:
    """Pdf for 1d Student t mixture."""
    if gammaln is None:
        raise RuntimeError("scipy is required for Student t pdf")
    y = np.zeros_like(x, dtype=float)
    for w, mu, sd, df in zip(weights, means, sds, dfs):
        z = (x - mu) / sd
        log_norm = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi)
            - np.log(sd)
        )
        log_pdf = log_norm - 0.5 * (df + 1.0) * np.log1p((z ** 2) / df)
        y += w * np.exp(log_pdf)
    return y
