"""Moment helpers for mixture distributions."""
from __future__ import annotations

import math

import numpy as np


def mixture_moments_sech_1d(weights: np.ndarray, means: np.ndarray, sds: np.ndarray) -> dict:
    """Moments for 1d sech mixture."""
    comp_mean = means
    comp_var = sds ** 2
    comp_mu3 = np.zeros_like(comp_mean)
    comp_mu4 = 5.0 * (sds ** 4)
    mean_mix = float(np.sum(weights * comp_mean))
    deltas = comp_mean - mean_mix
    mu2 = float(np.sum(weights * (comp_var + deltas ** 2)))
    mu3 = float(np.sum(weights * (comp_mu3 + 3.0 * deltas * comp_var + deltas ** 3)))
    mu4 = float(np.sum(weights * (comp_mu4 + 4.0 * deltas * comp_mu3 +
                                  6.0 * deltas ** 2 * comp_var + deltas ** 4)))
    sd_mix = math.sqrt(mu2)
    skew = mu3 / (sd_mix ** 3)
    ex_kurt = mu4 / (sd_mix ** 4) - 3.0
    return {"mean": mean_mix, "sd": sd_mix, "skew": skew, "ex_kurt": ex_kurt}


def mixture_moments_logistic_1d(weights: np.ndarray, means: np.ndarray, sds: np.ndarray) -> dict:
    """Moments for 1d logistic mixture."""
    comp_mean = means
    comp_var = (math.pi ** 2 / 3.0) * (sds ** 2)
    comp_mu3 = np.zeros_like(comp_mean)
    comp_mu4 = (21.0 / 5.0) * (comp_var ** 2)
    mean_mix = float(np.sum(weights * comp_mean))
    deltas = comp_mean - mean_mix
    mu2 = float(np.sum(weights * (comp_var + deltas ** 2)))
    mu3 = float(np.sum(weights * (comp_mu3 + 3.0 * deltas * comp_var + deltas ** 3)))
    mu4 = float(np.sum(weights * (comp_mu4 + 4.0 * deltas * comp_mu3 +
                                  6.0 * deltas ** 2 * comp_var + deltas ** 4)))
    sd_mix = math.sqrt(mu2)
    skew = mu3 / (sd_mix ** 3)
    ex_kurt = mu4 / (sd_mix ** 4) - 3.0
    return {"mean": mean_mix, "sd": sd_mix, "skew": skew, "ex_kurt": ex_kurt}


def mixture_moments_t_1d(
    weights: np.ndarray, means: np.ndarray, sds: np.ndarray, dfs: np.ndarray
) -> dict:
    """Moments for 1d Student t mixture."""
    comp_mean = means
    comp_var = np.empty_like(sds, dtype=float)
    comp_mu3 = np.zeros_like(sds, dtype=float)
    comp_mu4 = np.empty_like(sds, dtype=float)
    for i, (sd, df) in enumerate(zip(sds, dfs)):
        if df <= 2.0:
            comp_var[i] = np.nan
            comp_mu4[i] = np.nan
            comp_mu3[i] = np.nan
            continue
        comp_var[i] = (sd ** 2) * df / (df - 2.0)
        if df <= 3.0:
            comp_mu3[i] = np.nan
        else:
            comp_mu3[i] = 0.0
        if df <= 4.0:
            comp_mu4[i] = np.nan
        else:
            ex_kurt = 6.0 / (df - 4.0)
            comp_mu4[i] = (ex_kurt + 3.0) * (comp_var[i] ** 2)
    mean_mix = float(np.sum(weights * comp_mean))
    deltas = comp_mean - mean_mix
    mu2 = float(np.sum(weights * (comp_var + deltas ** 2)))
    mu3 = float(np.sum(weights * (comp_mu3 + 3.0 * deltas * comp_var + deltas ** 3)))
    mu4 = float(np.sum(weights * (comp_mu4 + 4.0 * deltas * comp_mu3 +
                                  6.0 * deltas ** 2 * comp_var + deltas ** 4)))
    sd_mix = math.sqrt(mu2)
    skew = mu3 / (sd_mix ** 3)
    ex_kurt = mu4 / (sd_mix ** 4) - 3.0
    return {"mean": mean_mix, "sd": sd_mix, "skew": skew, "ex_kurt": ex_kurt}
