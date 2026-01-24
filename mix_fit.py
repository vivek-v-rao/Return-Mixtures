"""Fitters for non-Gaussian mixtures."""
from __future__ import annotations

import math

import numpy as np

from mix_dist import log_logistic_pdf, log_sech_pdf, log_t_pdf
from mix_em import logsumexp
from mix_em import softplus  # re-exported for completeness
from mix_dist import normal_pdf  # re-exported for completeness
from mix_dist import mixture_pdf_logistic_1d, mixture_pdf_sech_1d, mixture_pdf_t_1d  # re-exported
from mix_moments import (
    mixture_moments_logistic_1d,
    mixture_moments_sech_1d,
    mixture_moments_t_1d,
)  # re-exported

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - optional dependency
    minimize = None


def init_sech_params(x: np.ndarray, k: int, rng: np.random.Generator) -> dict:
    """Initialize weights/means/scales for EM."""
    if k == 1 or x.size < k:
        return {
            "weights": np.array([1.0]),
            "means": np.array([float(np.mean(x))]),
            "sds": np.array([max(float(np.std(x)), 1e-3)]),
        }
    qs = np.quantile(x, np.linspace(0, 1, k + 2)[1:-1])
    means = qs.astype(float)
    dists = np.abs(x[:, None] - means[None, :])
    labels = np.argmin(dists, axis=1)
    weights = np.array([(labels == j).mean() for j in range(k)])
    sds = np.empty(k)
    for j in range(k):
        xj = x[labels == j]
        sds[j] = max(float(np.std(xj)) if xj.size > 1 else float(np.std(x)), 1e-3)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum()
    return {"weights": weights, "means": means, "sds": sds}


def update_sech_component(
    x: np.ndarray, w: np.ndarray, mu_init: float, sd_init: float
) -> tuple[float, float]:
    """Update sech component via weighted MLE."""
    if minimize is None:
        raise RuntimeError("scipy is required for sech mixture optimization")
    if not np.isfinite(w).all() or w.sum() <= 0:
        return mu_init, sd_init
    max_sd = max(float(np.std(x)), 1e-3) * 10.0

    def obj(par: np.ndarray) -> float:
        mu = par[0]
        sd = math.exp(par[1])
        if not np.isfinite(sd) or sd <= 0:
            return np.inf
        ll = log_sech_pdf(x, mu, sd)
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -float(np.sum(w * ll))

    res = minimize(
        obj,
        x0=np.array([mu_init, math.log(sd_init)]),
        method="L-BFGS-B",
        bounds=[(float(np.min(x) - 10 * max_sd), float(np.max(x) + 10 * max_sd)),
                (math.log(1e-6), math.log(max_sd))],
    )
    mu = float(res.x[0])
    sd = float(math.exp(res.x[1]))
    return mu, sd


def update_logistic_component(
    x: np.ndarray, w: np.ndarray, mu_init: float, sd_init: float
) -> tuple[float, float]:
    """Update logistic component via weighted MLE."""
    if minimize is None:
        raise RuntimeError("scipy is required for logistic mixture optimization")
    if not np.isfinite(w).all() or w.sum() <= 0:
        return mu_init, sd_init
    max_sd = max(float(np.std(x)), 1e-3) * 10.0

    def obj(par: np.ndarray) -> float:
        mu = par[0]
        sd = math.exp(par[1])
        if not np.isfinite(sd) or sd <= 0:
            return np.inf
        ll = log_logistic_pdf(x, mu, sd)
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -float(np.sum(w * ll))

    res = minimize(
        obj,
        x0=np.array([mu_init, math.log(sd_init)]),
        method="L-BFGS-B",
        bounds=[(float(np.min(x) - 10 * max_sd), float(np.max(x) + 10 * max_sd)),
                (math.log(1e-6), math.log(max_sd))],
    )
    mu = float(res.x[0])
    sd = float(math.exp(res.x[1]))
    return mu, sd


def update_t_component(
    x: np.ndarray, w: np.ndarray, mu_init: float, sd_init: float, df_init: float
) -> tuple[float, float, float]:
    """Update Student t component via weighted MLE."""
    if minimize is None:
        raise RuntimeError("scipy is required for t mixture optimization")
    if not np.isfinite(w).all() or w.sum() <= 0:
        return mu_init, sd_init, df_init
    max_sd = max(float(np.std(x)), 1e-3) * 10.0

    def obj(par: np.ndarray) -> float:
        mu = par[0]
        sd = math.exp(par[1])
        df = 2.0 + math.exp(par[2])
        if not np.isfinite(sd) or sd <= 0 or not np.isfinite(df):
            return np.inf
        ll = log_t_pdf(x, mu, sd, df)
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -float(np.sum(w * ll))

    res = minimize(
        obj,
        x0=np.array([mu_init, math.log(sd_init), math.log(max(df_init - 2.0, 1e-3))]),
        method="L-BFGS-B",
        bounds=[
            (float(np.min(x) - 10 * max_sd), float(np.max(x) + 10 * max_sd)),
            (math.log(1e-6), math.log(max_sd)),
            (math.log(1e-6), math.log(200.0)),
        ],
    )
    mu = float(res.x[0])
    sd = float(math.exp(res.x[1]))
    df = 2.0 + float(math.exp(res.x[2]))
    return mu, sd, df


def update_t_component_fixed_df(
    x: np.ndarray, w: np.ndarray, mu_init: float, sd_init: float, df_fixed: float
) -> tuple[float, float]:
    """Update Student t component with fixed dof."""
    if minimize is None:
        raise RuntimeError("scipy is required for t mixture optimization")
    if not np.isfinite(w).all() or w.sum() <= 0:
        return mu_init, sd_init
    max_sd = max(float(np.std(x)), 1e-3) * 10.0

    def obj(par: np.ndarray) -> float:
        mu = par[0]
        sd = math.exp(par[1])
        if not np.isfinite(sd) or sd <= 0:
            return np.inf
        ll = log_t_pdf(x, mu, sd, df_fixed)
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -float(np.sum(w * ll))

    res = minimize(
        obj,
        x0=np.array([mu_init, math.log(sd_init)]),
        method="L-BFGS-B",
        bounds=[
            (float(np.min(x) - 10 * max_sd), float(np.max(x) + 10 * max_sd)),
            (math.log(1e-6), math.log(max_sd)),
        ],
    )
    mu = float(res.x[0])
    sd = float(math.exp(res.x[1]))
    return mu, sd


def fit_sech_mixture_em(
    x: np.ndarray, k: int, seed: int, max_iter: int = 200, tol: float = 1e-6
) -> dict:
    """Fit sech mixture with EM."""
    rng = np.random.default_rng(seed)
    params = init_sech_params(x, k, rng)
    weights = params["weights"].copy()
    means = params["means"].copy()
    sds = params["sds"].copy()
    loglik_prev = -np.inf
    for _ in range(max_iter):
        log_resp = np.zeros((x.size, k))
        for j in range(k):
            log_resp[:, j] = np.log(weights[j]) + log_sech_pdf(x, means[j], sds[j])
        log_sum = logsumexp(log_resp, axis=1)
        if not np.all(np.isfinite(log_sum)):
            break
        loglik = float(np.sum(log_sum))
        if np.isfinite(loglik_prev) and abs(loglik - loglik_prev) <= tol * (1 + abs(loglik_prev)):
            loglik_prev = loglik
            break
        loglik_prev = loglik
        resp = np.exp(log_resp - log_sum[:, None])
        weights = resp.mean(axis=0)
        weights = np.clip(weights, 1e-8, None)
        weights = weights / weights.sum()
        for j in range(k):
            means[j], sds[j] = update_sech_component(x, resp[:, j], means[j], sds[j])
    p = (k - 1) + 2 * k
    aic = 2.0 * p - 2.0 * loglik_prev
    bic = np.log(x.size) * p - 2.0 * loglik_prev
    return {
        "weights": weights,
        "means": means,
        "sds": sds,
        "loglik": loglik_prev,
        "aic": aic,
        "bic": bic,
    }


def fit_logistic_mixture_em(
    x: np.ndarray, k: int, seed: int, max_iter: int = 200, tol: float = 1e-6
) -> dict:
    """Fit logistic mixture with EM."""
    rng = np.random.default_rng(seed)
    params = init_sech_params(x, k, rng)
    weights = params["weights"].copy()
    means = params["means"].copy()
    sds = params["sds"].copy()
    loglik_prev = -np.inf
    for _ in range(max_iter):
        log_resp = np.zeros((x.size, k))
        for j in range(k):
            log_resp[:, j] = np.log(weights[j]) + log_logistic_pdf(x, means[j], sds[j])
        log_sum = logsumexp(log_resp, axis=1)
        if not np.all(np.isfinite(log_sum)):
            break
        loglik = float(np.sum(log_sum))
        if np.isfinite(loglik_prev) and abs(loglik - loglik_prev) <= tol * (1 + abs(loglik_prev)):
            loglik_prev = loglik
            break
        loglik_prev = loglik
        resp = np.exp(log_resp - log_sum[:, None])
        weights = resp.mean(axis=0)
        weights = np.clip(weights, 1e-8, None)
        weights = weights / weights.sum()
        for j in range(k):
            means[j], sds[j] = update_logistic_component(x, resp[:, j], means[j], sds[j])
    p = (k - 1) + 2 * k
    aic = 2.0 * p - 2.0 * loglik_prev
    bic = np.log(x.size) * p - 2.0 * loglik_prev
    return {
        "weights": weights,
        "means": means,
        "sds": sds,
        "loglik": loglik_prev,
        "aic": aic,
        "bic": bic,
    }


def fit_t_mixture_em(
    x: np.ndarray,
    k: int,
    seed: int,
    t_dof: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    """Fit Student t mixture with EM."""
    rng = np.random.default_rng(seed)
    params = init_sech_params(x, k, rng)
    weights = params["weights"].copy()
    means = params["means"].copy()
    sds = params["sds"].copy()
    if t_dof is not None and t_dof <= 0:
        raise ValueError("t_dof must be positive when fixed")
    dfs = np.full(k, float(t_dof) if t_dof is not None else 8.0)
    loglik_prev = -np.inf
    for _ in range(max_iter):
        log_resp = np.zeros((x.size, k))
        for j in range(k):
            log_resp[:, j] = np.log(weights[j]) + log_t_pdf(x, means[j], sds[j], dfs[j])
        log_sum = logsumexp(log_resp, axis=1)
        if not np.all(np.isfinite(log_sum)):
            break
        loglik = float(np.sum(log_sum))
        if np.isfinite(loglik_prev) and abs(loglik - loglik_prev) <= tol * (1 + abs(loglik_prev)):
            loglik_prev = loglik
            break
        loglik_prev = loglik
        resp = np.exp(log_resp - log_sum[:, None])
        weights = resp.mean(axis=0)
        weights = np.clip(weights, 1e-8, None)
        weights = weights / weights.sum()
        for j in range(k):
            if t_dof is None:
                means[j], sds[j], dfs[j] = update_t_component(
                    x, resp[:, j], means[j], sds[j], dfs[j]
                )
            else:
                means[j], sds[j] = update_t_component_fixed_df(
                    x, resp[:, j], means[j], sds[j], t_dof
                )
                dfs[j] = t_dof
    p = (k - 1) + (2 * k if t_dof is not None else 3 * k)
    aic = 2.0 * p - 2.0 * loglik_prev
    bic = np.log(x.size) * p - 2.0 * loglik_prev
    return {
        "weights": weights,
        "means": means,
        "sds": sds,
        "dfs": dfs,
        "loglik": loglik_prev,
        "aic": aic,
        "bic": bic,
        "df_fixed": t_dof is not None,
    }


def fit_series_mixtures_sech_1d(x: np.ndarray, k_max: int, seed: int):
    """Fit k=1..k_max sech mixtures."""
    if k_max < 1:
        return [], (None, None), (None, None)
    results = []
    for k in range(1, k_max + 1):
        est = fit_sech_mixture_em(x, k=k, seed=seed)
        results.append((k, est["aic"], est["bic"], est))
    k_bic, _, _, est_bic = min(results, key=lambda t: t[2])
    k_aic, _, _, est_aic = min(results, key=lambda t: t[1])
    return results, (k_bic, est_bic), (k_aic, est_aic)


def fit_series_mixtures_logistic_1d(x: np.ndarray, k_max: int, seed: int):
    """Fit k=1..k_max logistic mixtures."""
    if k_max < 1:
        return [], (None, None), (None, None)
    results = []
    for k in range(1, k_max + 1):
        est = fit_logistic_mixture_em(x, k=k, seed=seed)
        results.append((k, est["aic"], est["bic"], est))
    k_bic, _, _, est_bic = min(results, key=lambda t: t[2])
    k_aic, _, _, est_aic = min(results, key=lambda t: t[1])
    return results, (k_bic, est_bic), (k_aic, est_aic)


def fit_series_mixtures_t_1d(x: np.ndarray, k_max: int, seed: int, t_dof: float | None = None):
    """Fit k=1..k_max Student t mixtures."""
    if k_max < 1:
        return [], (None, None), (None, None)
    results = []
    for k in range(1, k_max + 1):
        est = fit_t_mixture_em(x, k=k, seed=seed, t_dof=t_dof)
        results.append((k, est["aic"], est["bic"], est))
    k_bic, _, _, est_bic = min(results, key=lambda t: t[2])
    k_aic, _, _, est_aic = min(results, key=lambda t: t[1])
    return results, (k_bic, est_bic), (k_aic, est_aic)
