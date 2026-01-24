"""Fit 1d mixtures to returns for multiple base distributions."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde
except ImportError:  # pragma: no cover
    gaussian_kde = None

from mix_dist import mixture_pdf_logistic_1d, mixture_pdf_sech_1d, mixture_pdf_t_1d, normal_pdf
from mix_fit import (
    fit_series_mixtures_logistic_1d,
    fit_series_mixtures_sech_1d,
    fit_series_mixtures_t_1d,
)
from mix_moments import (
    mixture_moments_logistic_1d,
    mixture_moments_sech_1d,
    mixture_moments_t_1d,
)
from mix_select import select_best_fit_general
from mixture import (
    fit_mvt_mixture,
    fit_mvn_mixture,
    fit_series_mixtures_1d,
    make_params_tables,
    mixture_loglik_1d,
    mixture_moments_1d,
    mixture_pdf_1d,
    sort_components,
)
from stats import compute_returns, read_prices_file, standardize_ewma


def main() -> int:
    """Run mixture fitting for configured base distributions."""
    t_start = time.perf_counter()
    pd.options.display.float_format = "{:.4f}".format

    in_prices_file = "etfs_adj_close.csv"
    dropna_df = False
    ret_scale = 100.0
    use_log_returns = True
    max_symbols = None # 3
    symbols = None
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    seed = 123
    k_max = 3
    ewma_lambda = None
    k_mv_max = 2
    dist_families = ["normal", "t", "sech", "logistic"]
    t_dof: Optional[float] = 5.0 # None
    t_dof_mv: Optional[float] = t_dof

    print_summary_table = True
    print_model_selection = True
    print_params_best = True
    show_all_fits = True
    show_density_plot = False
    show_ic_plot = False
    best_plot_criterion = "AIC"
    show_kde_plot = False
    show_best_density_plot = False
    create_all_best_density_plot = True
    density_suffix = ".png"
    print_moments_table = True

    print("prices file:", in_prices_file)
    df_all = read_prices_file(Path(in_prices_file))

    if date_min is not None:
        date_min = pd.to_datetime(date_min)
    if date_max is not None:
        date_max = pd.to_datetime(date_max)
    if date_min is not None or date_max is not None:
        df_all = df_all.loc[date_min:date_max]

    if max_symbols is not None:
        df_all = df_all.iloc[:, :max_symbols]
    if symbols is not None:
        df_all = df_all.loc[:, [s for s in symbols if s in df_all.columns]]

    print("#obs, symbols, columns:", df_all.shape[0], df_all.shape[1], df_all.shape[1])
    print("return_type:", "log" if use_log_returns else "simple")
    print("ret_scale:", ret_scale)
    print("ewma_lambda:", ewma_lambda if ewma_lambda is not None else "NULL")
    print("base distributions:", ", ".join(dist_families))
    print("max # of mixture components:", k_max)
    print("max # of mv mixture components:", k_mv_max)
    if k_max < 1:
        print("no univariate mixture fits will be run (k_max < 1)")

    if dropna_df:
        df_all = df_all.dropna()

    if len(df_all.index) > 0:
        print("#obs, first, last:", len(df_all.index), df_all.index[0].date(), df_all.index[-1].date())
    else:
        print("#obs, first, last:", 0, "nan", "nan")

    df_ret = ret_scale * compute_returns(df_all, log_returns=use_log_returns)

    summary_rows = []
    ic_by_symbol = {}
    best_fits_by_symbol = {}
    mv_results_by_dist = {}

    for dist_family in dist_families:
        print()
        print("base distribution:", dist_family)

        if k_mv_max > 0 and dist_family in ("normal", "t") and df_ret.shape[1] > 1:
            df_mv = df_ret.dropna(axis=0, how="any")
            if df_mv.shape[0] > 0:
                x_mv = df_mv.to_numpy()
                var_names = list(df_mv.columns)
                results_mv = []
                for k in range(1, k_mv_max + 1):
                    if dist_family == "normal":
                        est = fit_mvn_mixture(x_mv, k=k, seed=seed)
                    else:
                        df_fixed = t_dof_mv if t_dof_mv is not None else 8.0
                        est = fit_mvt_mixture(x_mv, k=k, seed=seed, df_fixed=df_fixed)
                    results_mv.append((k, est["aic"], est["bic"], est))
                k_bic, _, _, est_bic = min(results_mv, key=lambda t: t[2])
                k_aic, _, _, est_aic = min(results_mv, key=lambda t: t[1])
                mv_results_by_dist[dist_family] = {
                    k: {"aic": aic, "bic": bic} for k, aic, bic, _ in results_mv
                }
                w_fit, mu_fit, cov_fit = sort_components(
                    est_bic["weights"], est_bic["means"], est_bic["covs"]
                )
                params_df, corr_df = make_params_tables(
                    w_fit, mu_fit, cov_fit, label=f"fit_k{k_bic}", var_names=var_names
                )
                print()
                print(f"multivariate fit: n={x_mv.shape[0]} d={x_mv.shape[1]}")
                print(f"model selection over k=1..{k_mv_max} (aic, bic)")
                for k, aic, bic, _ in results_mv:
                    print(f"k={k:>2}  aic={aic:>12.3f}  bic={bic:>12.3f}")
                print(f"bic selects k={k_bic}")
                print(f"aic selects k={k_aic}")
                if dist_family == "t":
                    df_fixed = t_dof_mv if t_dof_mv is not None else 8.0
                    print(f"t dof (fixed) = {df_fixed:.4f}")
                print()
                print("weights, means, stds per component (sorted by descending weight)")
                print(params_df.to_string(float_format=lambda v: f"{v: .4f}"))
                print()
                print("correlations per component (stacked blocks)")
                print(corr_df.to_string(float_format=lambda v: f"{v: .3f}"))
            else:
                print()
                print("multivariate fit skipped: no common dates across assets")

        for symbol in df_ret.columns:
            x_series = df_ret[symbol].dropna()
            if ewma_lambda is not None:
                x_values = standardize_ewma(x_series, ewma_lambda)
            else:
                x_values = x_series.to_numpy()
            if x_values.size == 0:
                continue

            if dist_family == "normal":
                x = x_values.reshape(-1, 1)
                results, (k_bic, est_bic), (k_aic, est_aic) = fit_series_mixtures_1d(
                    x, k_max, seed
                )
            elif dist_family == "sech":
                results, (k_bic, est_bic), (k_aic, est_aic) = fit_series_mixtures_sech_1d(
                    x_values, k_max, seed
                )
            elif dist_family == "logistic":
                results, (k_bic, est_bic), (k_aic, est_aic) = fit_series_mixtures_logistic_1d(
                    x_values, k_max, seed
                )
            elif dist_family == "t":
                results, (k_bic, est_bic), (k_aic, est_aic) = fit_series_mixtures_t_1d(
                    x_values, k_max, seed, t_dof=t_dof
                )
            else:
                raise ValueError(f"unsupported dist_family: {dist_family}")

            best_k, best_est = select_best_fit_general(
                results, x_values if dist_family != "normal" else x, best_plot_criterion, dist_family
            )
            if best_est is not None:
                best_fits_by_symbol.setdefault(symbol, {})[dist_family] = (best_k, best_est)

            if print_model_selection and results:
                print()
                print(f"symbol={symbol}  n={len(x_values)}")
                print(f"model selection over k=1..{k_max} (aic, bic)")
                for k, aic, bic, _ in results:
                    print(f"k={k:>2}  aic={aic:>12.3f}  bic={bic:>12.3f}")
                print(f"bic selects k={k_bic}")
                print(f"aic selects k={k_aic}")

            if print_params_best and est_bic is not None and k_bic != 1:
                print()
                print(f"{symbol} best-fit parameters (bic)")
                if dist_family == "normal":
                    w_fit, mu_fit, cov_fit = sort_components(
                        est_bic["weights"], est_bic["means"], est_bic["covs"]
                    )
                    params_df, _ = make_params_tables(
                        w_fit, mu_fit, cov_fit, label=f"{symbol}_fit_k{k_bic}", var_names=["x1"]
                    )
                    params_df = params_df.rename(columns={"mean_x1": "mean", "sd_x1": "sd"})
                    print(params_df.to_string(float_format=lambda v: f"{v: .4f}"))
                else:
                    order = np.argsort(-est_bic["weights"])
                    params_df = pd.DataFrame({
                        "weight": est_bic["weights"][order],
                        "mean": est_bic["means"][order],
                        "sd": est_bic["sds"][order],
                    })
                    if dist_family == "t":
                        params_df["df"] = est_bic["dfs"][order]
                    print(params_df.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

            if print_moments_table:
                m_emp = {
                    "mean": float(np.mean(x_values)),
                    "sd": float(np.std(x_values)),
                    "skew": float(pd.Series(x_values).skew()),
                    "ex_kurt": float(pd.Series(x_values).kurtosis()),
                }
                moments_rows = [{
                    "label": "empirical",
                    **m_emp,
                    "loglik": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                }]
                for k, _, _, est in results:
                    if dist_family == "normal":
                        w_k, mu_k, cov_k = sort_components(est["weights"], est["means"], est["covs"])
                        ll_k = mixture_loglik_1d(x, w_k, mu_k, cov_k)
                        p_k = 3 * len(w_k) - 1
                        moments = mixture_moments_1d(w_k, mu_k, cov_k)
                    else:
                        order = np.argsort(-est["weights"])
                        w_k = est["weights"][order]
                        mu_k = est["means"][order]
                        sd_k = est["sds"][order]
                        ll_k = est["loglik"]
                        p_k = (len(w_k) - 1) + 2 * len(w_k)
                        if dist_family == "sech":
                            moments = mixture_moments_sech_1d(w_k, mu_k, sd_k)
                        elif dist_family == "logistic":
                            moments = mixture_moments_logistic_1d(w_k, mu_k, sd_k)
                        elif dist_family == "t":
                            df_k = est["dfs"][order]
                            moments = mixture_moments_t_1d(w_k, mu_k, sd_k, df_k)
                            if est.get("df_fixed"):
                                p_k = (len(w_k) - 1) + 2 * len(w_k)
                            else:
                                p_k = (len(w_k) - 1) + 3 * len(w_k)
                        else:
                            raise ValueError(f"unsupported dist_family: {dist_family}")
                    moments_rows.append({
                        "label": f"fit_k{k}",
                        **moments,
                        "loglik": ll_k,
                        "aic": 2.0 * p_k - 2.0 * ll_k,
                        "bic": np.log(len(x_values)) * p_k - 2.0 * ll_k,
                    })
                moments_df = pd.DataFrame(moments_rows).set_index("label")
                print()
                if results:
                    first_date = x_series.index[0].date() if len(x_series.index) > 0 else "nan"
                    last_date = x_series.index[-1].date() if len(x_series.index) > 0 else "nan"
                    print(f"{symbol} {len(x_values)} returns from {first_date} to {last_date}")
                    print(f"{symbol} selected k (aic, bic): {k_aic}, {k_bic}")
                print(f"{symbol} moments (mean, sd, skew, excess kurtosis)")
                print(moments_df.to_string(float_format=lambda v: f"{v: .4f}"))

            if show_all_fits and results:
                for k, _, _, est in results:
                    if k == 1:
                        continue
                    print()
                    print(f"{symbol} fit_k{k} parameters")
                    if dist_family == "normal":
                        w_k, mu_k, cov_k = sort_components(est["weights"], est["means"], est["covs"])
                        params_k, _ = make_params_tables(
                            w_k, mu_k, cov_k, label=f"{symbol}_fit_k{k}", var_names=["x1"]
                        )
                        params_k = params_k.rename(columns={"mean_x1": "mean", "sd_x1": "sd"})
                        print(params_k.to_string(float_format=lambda v: f"{v: .4f}"))
                    else:
                        order = np.argsort(-est["weights"])
                        params_df = pd.DataFrame({
                            "weight": est["weights"][order],
                            "mean": est["means"][order],
                            "sd": est["sds"][order],
                        })
                        if dist_family == "t":
                            params_df["df"] = est["dfs"][order]
                        print(params_df.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

            if show_density_plot and results:
                x_min = np.percentile(x_values, 0.5)
                x_max = np.percentile(x_values, 99.5)
                x_grid = np.linspace(x_min, x_max, 500)
                moments = m_emp

                plt.figure(figsize=(8, 4))
                for k, _, _, est in results:
                    if dist_family == "normal":
                        w_k, mu_k, cov_k = sort_components(est["weights"], est["means"], est["covs"])
                        y_k = mixture_pdf_1d(x_grid, w_k, mu_k, cov_k)
                    else:
                        order = np.argsort(-est["weights"])
                        if dist_family == "sech":
                            y_k = mixture_pdf_sech_1d(
                                x_grid,
                                est["weights"][order],
                                est["means"][order],
                                est["sds"][order],
                            )
                        elif dist_family == "logistic":
                            y_k = mixture_pdf_logistic_1d(
                                x_grid,
                                est["weights"][order],
                                est["means"][order],
                                est["sds"][order],
                            )
                        elif dist_family == "t":
                            y_k = mixture_pdf_t_1d(
                                x_grid,
                                est["weights"][order],
                                est["means"][order],
                                est["sds"][order],
                                est["dfs"][order],
                            )
                        else:
                            raise ValueError(f"unsupported dist_family: {dist_family}")
                    label = "normal" if k == 1 else f"fit_k{k}"
                    plt.plot(x_grid, y_k, linewidth=1.2, label=label)
                y_norm = normal_pdf(x_grid, moments["mean"], moments["sd"])
                plt.plot(x_grid, y_norm, linewidth=1.2, linestyle="--", color="gray", label="normal_base")
                if gaussian_kde is not None:
                    kde = gaussian_kde(x_values)
                    y_kde = kde(x_grid)
                    plt.plot(x_grid, y_kde, linestyle=":", linewidth=1.2, color="firebrick", label="kde")

                plt.title(
                    f"{symbol} mixture densities  "
                    f"mean={moments['mean']:.4f} sd={moments['sd']:.4f} "
                    f"skew={moments['skew']:.2f} ex_kurt={moments['ex_kurt']:.2f}"
                )
                plt.xlabel("return")
                plt.ylabel("density")
                plt.legend()
                plt.tight_layout()
                plt.show()

            if show_ic_plot and results:
                ks = [k for k, _, _, _ in results]
                aics = [aic for _, aic, _, _ in results]
                bics = [bic for _, _, bic, _ in results]

                plt.figure(figsize=(6, 4))
                plt.plot(ks, aics, marker="o", label="AIC")
                plt.plot(ks, bics, marker="o", label="BIC")
                plt.xticks(ks)
                plt.xlabel("k")
                plt.ylabel("information criterion")
                plt.title(f"{symbol} AIC/BIC vs. number of components")
                plt.legend()
                plt.tight_layout()
                plt.show()

            if (show_best_density_plot or density_suffix is not None) and best_est is not None:
                x_min = np.percentile(x_values, 0.5)
                x_max = np.percentile(x_values, 99.5)
                x_grid = np.linspace(x_min, x_max, 500)
                moments = m_emp

                plt.figure(figsize=(8, 4))
                if dist_family == "normal":
                    w_k, mu_k, cov_k = sort_components(best_est["weights"], best_est["means"], best_est["covs"])
                    y_k = mixture_pdf_1d(x_grid, w_k, mu_k, cov_k)
                else:
                    order = np.argsort(-best_est["weights"])
                    if dist_family == "sech":
                        y_k = mixture_pdf_sech_1d(
                            x_grid,
                            best_est["weights"][order],
                            best_est["means"][order],
                            best_est["sds"][order],
                        )
                    elif dist_family == "logistic":
                        y_k = mixture_pdf_logistic_1d(
                            x_grid,
                            best_est["weights"][order],
                            best_est["means"][order],
                            best_est["sds"][order],
                        )
                    elif dist_family == "t":
                        y_k = mixture_pdf_t_1d(
                            x_grid,
                            best_est["weights"][order],
                            best_est["means"][order],
                            best_est["sds"][order],
                            best_est["dfs"][order],
                        )
                    else:
                        raise ValueError(f"unsupported dist_family: {dist_family}")
                plt.plot(x_grid, y_k, linewidth=2.0, label=f"fit_k{best_k}")

                y_norm = normal_pdf(x_grid, moments["mean"], moments["sd"])
                plt.plot(x_grid, y_norm, linewidth=1.2, linestyle="--", color="gray", label="normal_base")
                if gaussian_kde is not None:
                    kde = gaussian_kde(x_values)
                    y_kde = kde(x_grid)
                    plt.plot(x_grid, y_kde, linestyle=":", linewidth=1.5, color="firebrick", label="kde")

                plt.title(
                    f"{symbol} best mixture ({best_plot_criterion})  "
                    f"mean={moments['mean']:.4f} sd={moments['sd']:.4f} "
                    f"skew={moments['skew']:.2f} ex_kurt={moments['ex_kurt']:.2f}"
                )
                plt.xlabel("return")
                plt.ylabel("density")
                plt.legend()
                plt.tight_layout()
                if density_suffix is not None:
                    out_name = f"{symbol.lower()}_{dist_family}_density{density_suffix}"
                    plt.savefig(out_name, dpi=150, bbox_inches="tight")
                if show_best_density_plot:
                    plt.show()
                else:
                    plt.close()

            if results:
                summary_rows.append({
                    "symbol": symbol,
                    "dist": dist_family,
                    "n_obs": len(x_values),
                    "k_bic": k_bic,
                    "k_aic": k_aic,
                    "bic": est_bic["bic"],
                    "aic": est_aic["aic"],
                })
                ic_by_symbol.setdefault(symbol, {})[dist_family] = {
                    "aic": est_aic["aic"],
                    "bic": est_bic["bic"],
                    "k_aic": k_aic,
                    "k_bic": k_bic,
                }

    if mv_results_by_dist:
        mv_rows = []
        mv_best = {
            "aic": {"dist": None, "k": None, "value": np.inf},
            "bic": {"dist": None, "k": None, "value": np.inf},
        }
        for k in range(1, k_mv_max + 1):
            aic_vals = {
                dist: vals[k]["aic"]
                for dist, vals in mv_results_by_dist.items()
                if k in vals
            }
            bic_vals = {
                dist: vals[k]["bic"]
                for dist, vals in mv_results_by_dist.items()
                if k in vals
            }
            if not aic_vals or not bic_vals:
                continue
            best_aic_dist = min(aic_vals, key=aic_vals.get)
            best_bic_dist = min(bic_vals, key=bic_vals.get)
            mv_rows.append({
                "k": k,
                "best_aic_dist": best_aic_dist,
                "best_aic": aic_vals[best_aic_dist],
                "best_bic_dist": best_bic_dist,
                "best_bic": bic_vals[best_bic_dist],
            })
            if aic_vals[best_aic_dist] < mv_best["aic"]["value"]:
                mv_best["aic"] = {
                    "dist": best_aic_dist,
                    "k": k,
                    "value": aic_vals[best_aic_dist],
                }
            if bic_vals[best_bic_dist] < mv_best["bic"]["value"]:
                mv_best["bic"] = {
                    "dist": best_bic_dist,
                    "k": k,
                    "value": bic_vals[best_bic_dist],
                }
        if mv_rows:
            df_mv = pd.DataFrame(mv_rows)
            print()
            print("multivariate distribution comparison (best per k):")
            print(df_mv.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
            print()
            print("multivariate best overall (min AIC/BIC):")
            print(
                pd.DataFrame([{
                    "best_aic_dist": mv_best["aic"]["dist"],
                    "best_aic_k": mv_best["aic"]["k"],
                    "best_aic": mv_best["aic"]["value"],
                    "best_bic_dist": mv_best["bic"]["dist"],
                    "best_bic_k": mv_best["bic"]["k"],
                    "best_bic": mv_best["bic"]["value"],
                }]).to_string(index=False, float_format=lambda v: f"{v: .4f}")
            )

    if create_all_best_density_plot and best_fits_by_symbol:
        for symbol, best_fits in best_fits_by_symbol.items():
            x_series = df_ret[symbol].dropna()
            x_values = standardize_ewma(x_series, ewma_lambda) if ewma_lambda is not None else x_series.to_numpy()
            if x_values.size == 0:
                continue
            x_min = np.percentile(x_values, 0.5)
            x_max = np.percentile(x_values, 99.5)
            x_grid = np.linspace(x_min, x_max, 500)
            moments = {
                "mean": float(np.mean(x_values)),
                "sd": float(np.std(x_values)),
                "skew": float(pd.Series(x_values).skew()),
                "ex_kurt": float(pd.Series(x_values).kurtosis()),
            }
            plt.figure(figsize=(8, 4))
            y_values = []
            for dist in dist_families:
                if dist not in best_fits:
                    continue
                best_k, best_est = best_fits[dist]
                if dist == "normal":
                    w_k, mu_k, cov_k = sort_components(best_est["weights"], best_est["means"], best_est["covs"])
                    y_k = mixture_pdf_1d(x_grid, w_k, mu_k, cov_k)
                else:
                    order = np.argsort(-best_est["weights"])
                    if dist == "sech":
                        y_k = mixture_pdf_sech_1d(
                            x_grid,
                            best_est["weights"][order],
                            best_est["means"][order],
                            best_est["sds"][order],
                        )
                    elif dist == "logistic":
                        y_k = mixture_pdf_logistic_1d(
                            x_grid,
                            best_est["weights"][order],
                            best_est["means"][order],
                            best_est["sds"][order],
                        )
                    elif dist == "t":
                        y_k = mixture_pdf_t_1d(
                            x_grid,
                            best_est["weights"][order],
                            best_est["means"][order],
                            best_est["sds"][order],
                            best_est["dfs"][order],
                        )
                    else:
                        raise ValueError(f"unsupported dist_family: {dist}")
                y_values.append(y_k)
                plt.plot(x_grid, y_k, linewidth=1.2, label=f"{dist}_{best_k}")
            y_norm = normal_pdf(x_grid, moments["mean"], moments["sd"])
            y_values.append(y_norm)
            plt.plot(x_grid, y_norm, linewidth=1.2, linestyle="--", color="gray", label="normal_base")
            if gaussian_kde is not None:
                kde = gaussian_kde(x_values)
                y_kde = kde(x_grid)
                y_values.append(y_kde)
                plt.plot(x_grid, y_kde, linestyle=":", linewidth=1.2, color="firebrick", label="kde")
            if y_values:
                max_y = max(float(np.max(y)) for y in y_values if np.all(np.isfinite(y)))
                plt.ylim(0.0, max_y * 1.05)
            plt.title(
                f"{symbol} best mixtures across base distributions  "
                f"mean={moments['mean']:.4f} sd={moments['sd']:.4f} "
                f"skew={moments['skew']:.2f} ex_kurt={moments['ex_kurt']:.2f}"
            )
            plt.xlabel("return")
            plt.ylabel("density")
            plt.legend()
            plt.tight_layout()
            if density_suffix is not None:
                out_name = f"{symbol.lower()}_all{density_suffix}"
                plt.savefig(out_name, dpi=150, bbox_inches="tight")
            plt.close()

    if print_summary_table and summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        print()
        print("mixture fit summary:")
        print(df_summary.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    if ic_by_symbol:
        rows = []
        for symbol, dist_rows in ic_by_symbol.items():
            aic_vals = {dist: vals["aic"] for dist, vals in dist_rows.items()}
            bic_vals = {dist: vals["bic"] for dist, vals in dist_rows.items()}
            best_aic_dist = min(aic_vals, key=aic_vals.get)
            best_bic_dist = min(bic_vals, key=bic_vals.get)
            rows.append({
                "symbol": symbol,
                "best_aic_dist": best_aic_dist,
                "best_aic": aic_vals[best_aic_dist],
                "best_aic_k": dist_rows[best_aic_dist]["k_aic"],
                "best_bic_dist": best_bic_dist,
                "best_bic": bic_vals[best_bic_dist],
                "best_bic_k": dist_rows[best_bic_dist]["k_bic"],
            })
        df_compare = pd.DataFrame(rows)
        print()
        print("base distribution comparison (min AIC/BIC across families):")
        print(df_compare.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    elapsed = time.perf_counter() - t_start
    print(f"\ntime elapsed: {elapsed:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
