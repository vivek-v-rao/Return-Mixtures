"""Selection helpers for mixture fits."""
from __future__ import annotations

import numpy as np

from mixture import mixture_loglik_1d, sort_components


def select_best_fit_general(results, x, criterion, dist):
    """Select best fit by likelihood or IC."""
    if not results or criterion is None:
        return None, None
    criterion = criterion.lower()
    if criterion == "lik":
        best = None
        best_ll = -np.inf
        for k, _, _, est in results:
            if dist == "normal":
                w_k, mu_k, cov_k = sort_components(est["weights"], est["means"], est["covs"])
                ll = mixture_loglik_1d(x, w_k, mu_k, cov_k)
            else:
                ll = est["loglik"]
            if ll > best_ll:
                best_ll = ll
                best = (k, est)
        return best
    if criterion == "aic":
        k, _, _, est = min(results, key=lambda t: t[1])
        return k, est
    if criterion == "bic":
        k, _, _, est = min(results, key=lambda t: t[2])
        return k, est
    raise ValueError("best_plot_criterion must be 'lik', 'AIC', 'BIC', or None")
