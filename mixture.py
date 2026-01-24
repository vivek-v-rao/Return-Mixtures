import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

try:
    from scipy.special import gammaln
except ImportError:  # pragma: no cover
    gammaln = None

def simulate_mvn_mixture(n=6000, seed=123):
    """Simulate a fixed multivariate normal mixture."""
    rng = np.random.default_rng(seed)

    # true params (2-dim, 2 components)
    w_true = np.array([0.35, 0.65])
    mu_true = np.stack([
        np.array([0.0, 0.0]),
        np.array([3.0, 1.5])
    ], axis=0)
    cov_true = np.stack([
        np.array([[1.0, 0.6],
                  [0.6, 1.5]]),
        np.array([[1.2, -0.3],
                  [-0.3, 0.8]])
    ], axis=0)

    z = rng.choice(len(w_true), size=n, p=w_true)
    d = mu_true.shape[1]
    x = np.empty((n, d))
    for k in range(len(w_true)):
        idx = np.where(z == k)[0]
        if idx.size:
            x[idx] = rng.multivariate_normal(mu_true[k], cov_true[k], size=idx.size)

    return x, w_true, mu_true, cov_true


def logsumexp(a, axis=1):
    """Compute log-sum-exp along an axis."""
    a = np.asarray(a)
    m = np.max(a, axis=axis, keepdims=True)
    if not np.all(np.isfinite(m)):
        return np.full(a.shape[0], np.nan)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def log_mvt_pdf(x, mean, cov, df):
    """Compute multivariate Student t log-pdf for each row of x."""
    if gammaln is None:
        raise RuntimeError("scipy is required for multivariate t log-pdf")
    x = np.asarray(x)
    d = x.shape[1]
    diff = x - mean
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + np.eye(d) * 1e-6
        chol = np.linalg.cholesky(cov)
    sol = np.linalg.solve(chol, diff.T)
    quad = np.sum(sol ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    log_norm = (
        gammaln((df + d) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * (d * np.log(df * np.pi) + logdet)
    )
    return log_norm - 0.5 * (df + d) * np.log1p(quad / df)

def simulate_norm_mixture_1d(n=6000, seed=123, k=3):
    """Simulate a 1d Gaussian mixture with k=1..3 preset components."""
    rng = np.random.default_rng(seed)

    if k == 1:
        w_true = np.array([1.0])
        mu_true = np.array([0.0]).reshape(-1, 1)
        sd_true = np.array([1.0])
    elif k == 2:
        w_true = np.array([0.65, 0.35])
        mu_true = np.array([1.5, -1.0]).reshape(-1, 1)
        sd_true = np.array([0.9, 0.7])
    elif k == 3:
        w_true = np.array([0.45, 0.30, 0.25])
        mu_true = np.array([0.5, 3.0, -2.0]).reshape(-1, 1)
        sd_true = np.array([0.9, 0.5, 0.6])
    else:
        raise ValueError("k must be 1, 2, or 3")

    cov_true = np.array([[[sd**2]] for sd in sd_true])

    z = rng.choice(len(w_true), size=n, p=w_true)
    x = np.empty((n, 1))
    for k in range(len(w_true)):
        idx = np.where(z == k)[0]
        if idx.size:
            x[idx, 0] = rng.normal(loc=mu_true[k, 0], scale=sd_true[k], size=idx.size)

    return x, w_true, mu_true, cov_true

def fit_mvn_mixture(x, k=2, seed=123):
    """Fit a Gaussian mixture model to data."""
    gm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=seed,
        init_params="kmeans",
        n_init=5,
        max_iter=500,
        reg_covar=1e-6
    )
    gm.fit(x)
    return {
        "weights": gm.weights_,
        "means": gm.means_,
        "covs": gm.covariances_,
        "aic": gm.aic(x),
        "bic": gm.bic(x),
        "model": gm
    }


def fit_mvt_mixture(x, k=2, seed=123, df_fixed=8.0, max_iter=200, tol=1e-6):
    """Fit a multivariate Student t mixture with fixed dof."""
    if df_fixed <= 0:
        raise ValueError("df_fixed must be positive")
    if gammaln is None:
        raise RuntimeError("scipy is required for multivariate t mixtures")
    x = np.asarray(x)
    n, d = x.shape
    gm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=seed,
        init_params="kmeans",
        n_init=3,
        max_iter=200,
        reg_covar=1e-6,
    )
    gm.fit(x)
    weights = gm.weights_.copy()
    means = gm.means_.copy()
    covs = gm.covariances_.copy()
    loglik_prev = -np.inf

    for _ in range(max_iter):
        log_resp = np.zeros((n, k))
        quad_terms = []
        for j in range(k):
            log_pdf = log_mvt_pdf(x, means[j], covs[j], df_fixed)
            log_resp[:, j] = np.log(weights[j]) + log_pdf
            diff = x - means[j]
            try:
                chol = np.linalg.cholesky(covs[j])
            except np.linalg.LinAlgError:
                covs[j] = covs[j] + np.eye(d) * 1e-6
                chol = np.linalg.cholesky(covs[j])
            sol = np.linalg.solve(chol, diff.T)
            quad_terms.append(np.sum(sol ** 2, axis=0))
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
            quad = quad_terms[j]
            tau = (df_fixed + d) / (df_fixed + quad)
            r_tau = resp[:, j] * tau
            denom = np.sum(r_tau)
            if denom <= 0:
                continue
            means[j] = np.sum(r_tau[:, None] * x, axis=0) / denom
            diff = x - means[j]
            cov_num = (diff.T * r_tau) @ diff
            covs[j] = cov_num / np.sum(resp[:, j])
            covs[j] = covs[j] + np.eye(d) * 1e-6
    p = (k - 1) + k * d + k * d * (d + 1) / 2
    aic = 2.0 * p - 2.0 * loglik_prev
    bic = np.log(n) * p - 2.0 * loglik_prev
    return {
        "weights": weights,
        "means": means,
        "covs": covs,
        "df": float(df_fixed),
        "aic": aic,
        "bic": bic,
        "loglik": loglik_prev,
    }

def fit_series_mixtures_1d(x, k_max, seed):
    """Fit mixtures with 1..k_max components for a 1d series."""
    if k_max < 1:
        return [], (None, None), (None, None)

    results = []
    for k in range(1, k_max + 1):
        est = fit_mvn_mixture(x, k=k, seed=seed)
        results.append((k, est["aic"], est["bic"], est))
    k_bic, _, _, est_bic = min(results, key=lambda t: t[2])
    k_aic, _, _, est_aic = min(results, key=lambda t: t[1])
    return results, (k_bic, est_bic), (k_aic, est_aic)

def select_best_fit_1d(results, x, criterion):
    """Select a best-fit result by likelihood, AIC, or BIC."""
    if not results or criterion is None:
        return None, None

    if criterion == "lik":
        best_k = None
        best_est = None
        best_ll = -np.inf
        for k, _, _, est in results:
            w_k, mu_k, cov_k = sort_components(est["weights"], est["means"], est["covs"])
            ll_k = mixture_loglik_1d(x, w_k, mu_k, cov_k)
            if ll_k > best_ll:
                best_ll = ll_k
                best_k = k
                best_est = est
        return best_k, best_est

    if criterion == "aic":
        best_k, aic, _, best_est = min(results, key=lambda t: t[1])
        return best_k, best_est

    if criterion == "bic":
        best_k, _, bic, best_est = min(results, key=lambda t: t[2])
        return best_k, best_est

    raise ValueError("best_plot_criterion must be 'lik', 'AIC', 'BIC', or None")

def normal_pdf(x, mean, sd):
    """Compute univariate normal PDF values."""
    z = (x - mean) / sd
    return np.exp(-0.5 * z * z) / (sd * np.sqrt(2.0 * np.pi))

def mixture_pdf_1d(x, weights, means, covs):
    """Compute 1d Gaussian mixture PDF values."""
    sds = np.sqrt(covs[:, 0, 0])
    total = np.zeros_like(x)
    for w, m, sd in zip(weights, means[:, 0], sds):
        total += w * normal_pdf(x, m, sd)
    return total

def mixture_moments_1d(weights, means, covs):
    """Compute mean, sd, skew, and excess kurtosis for a 1d mixture."""
    means_1d = means[:, 0]
    s2 = covs[:, 0, 0]
    mean = np.sum(weights * means_1d)
    var = np.sum(weights * (s2 + means_1d ** 2)) - mean ** 2
    sd = np.sqrt(var)

    dm = means_1d - mean
    mu3 = np.sum(weights * (dm ** 3 + 3.0 * dm * s2))
    mu4 = np.sum(weights * (dm ** 4 + 6.0 * dm ** 2 * s2 + 3.0 * s2 ** 2))
    skew = mu3 / (sd ** 3)
    ex_kurt = mu4 / (sd ** 4) - 3.0

    return {"mean": mean, "sd": sd, "skew": skew, "ex_kurt": ex_kurt}

def sample_moments_1d(x):
    """Compute sample mean, sd, skew, and excess kurtosis."""
    x1 = np.asarray(x).reshape(-1)
    mean = float(np.mean(x1))
    dm = x1 - mean
    mu2 = float(np.mean(dm ** 2))
    sd = np.sqrt(mu2)
    mu3 = float(np.mean(dm ** 3))
    mu4 = float(np.mean(dm ** 4))
    skew = mu3 / (sd ** 3)
    ex_kurt = mu4 / (sd ** 4) - 3.0
    return {"mean": mean, "sd": sd, "skew": skew, "ex_kurt": ex_kurt}

def mixture_loglik_1d(x, weights, means, covs):
    """Compute log-likelihood of data under a 1d Gaussian mixture."""
    x1 = np.asarray(x).reshape(-1)
    means_1d = means[:, 0]
    sds = np.sqrt(covs[:, 0, 0])
    z = (x1[:, None] - means_1d[None, :]) / sds[None, :]
    log_pdf = -0.5 * z ** 2 - np.log(sds[None, :] * np.sqrt(2.0 * np.pi))
    log_w = np.log(weights)[None, :]
    log_mix = log_w + log_pdf
    max_log = np.max(log_mix, axis=1, keepdims=True)
    ll = np.sum(max_log[:, 0] + np.log(np.sum(np.exp(log_mix - max_log), axis=1)))
    return float(ll)

def sort_components(weights, means, covs):
    """Sort components by descending weight."""
    order = np.argsort(-weights)
    return weights[order], means[order], covs[order]

def make_params_tables(weights, means, covs, label, var_names=None):
    """Build parameter and correlation tables for mixture components."""
    k, d = means.shape
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(d)]

    # params_df
    rows = []
    for j in range(k):
        std_j, _ = cov_to_std_and_corr(covs[j])
        row = {"label": label, "component": j+1, "weight": weights[j]}
        row.update({f"mean_{v}": means[j, i] for i, v in enumerate(var_names)})
        row.update({f"sd_{v}": std_j[i] for i, v in enumerate(var_names)})
        rows.append(row)
    params_df = pd.DataFrame(rows).set_index(["label", "component"])

    # corr_df
    corr_blocks = []
    for j in range(k):
        _, corr_j = cov_to_std_and_corr(covs[j])
        dfc = pd.DataFrame(corr_j, index=var_names, columns=var_names)
        dfc.index.name = "var_i"
        dfc.insert(0, "__label__", label)
        dfc.insert(1, "__component__", j+1)
        corr_blocks.append(dfc)

    corr_df = pd.concat(corr_blocks, axis=0)
    corr_df = corr_df.set_index(["__label__", "__component__"], append=True).reorder_levels(
        ["__label__", "__component__", "var_i"]
    )
    corr_df.index.set_names(["label", "component", "var_i"], inplace=True)

    return params_df, corr_df

def cov_to_std_and_corr(cov):
    """Convert covariance to standard deviations and correlations."""
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    corr = cov / denom
    np.fill_diagonal(corr, 1.0)
    return std, corr

