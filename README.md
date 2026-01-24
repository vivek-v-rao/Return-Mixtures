Return-Mixtures
===============

Fit univariate and multivariate mixture models to return series with multiple base distributions.
The main entry point is `xreturns_mix_general.py`. This README documents the Python workflow only.

Overview
--------
- Reads a price file (CSV or Parquet), computes returns, and fits mixtures by symbol.
- Supports base distributions: normal, Student t (`t`), hyperbolic secant (`sech`), and logistic.
- Optionally fits multivariate mixtures when `k_mv_max > 0` and the family is `normal` or `t`.
- Produces summary tables, parameter tables, information criteria, and density plots.

Quick Start
-----------
```bash
python xreturns_mix_general.py
```

Dependencies
------------
Python 3.10+ with:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy` (required for `t`, `sech`, and `logistic` mixtures)
- `scikit-learn` (Gaussian mixtures for normal/multivariate normal)

Files Used by `xreturns_mix_general.py`
---------------------------------------
- `xreturns_mix_general.py` (main script)
- `stats.py` (price/return helpers)
- `mixture.py` (Gaussian mixture helpers + multivariate t mixture)
- `mix_fit.py` (EM fitters for sech/logistic/t)
- `mix_dist.py` (pdf/log-pdf helpers)
- `mix_moments.py` (mixture moments)
- `mix_em.py` (numerical utilities)
- `mix_select.py` (model selection helpers)

Inputs
------
- Default input: `etfs_adj_close.csv` (set `in_prices_file` in `main()` to change).
- CSV should have a date index in the first column and one column per symbol.
- Parquet is also supported.

Configuration (in `main()`)
---------------------------
Key options you can edit in `xreturns_mix_general.py`:
- `in_prices_file`, `ret_scale`, `use_log_returns`, `date_min`, `date_max`
- `dist_families` (e.g., `["normal", "t", "sech", "logistic"]`)
- `k_max` (max univariate components)
- `k_mv_max` (max multivariate components)
- `t_dof` (fixed dof for univariate t; set `None` to estimate)
- `t_dof_mv` (fixed dof for multivariate t; defaults to `t_dof`)
- Plot toggles: `show_density_plot`, `show_best_density_plot`, `create_all_best_density_plot`

Notes on Student t
------------------
- If `t_dof` is set (fixed), dof is not counted as a free parameter in AIC/BIC.
- If `t_dof` is `None`, dof is estimated per component in the univariate t mixture.
- Multivariate t uses a fixed dof (`t_dof_mv`).

Outputs
-------
Console output includes:
- Model selection tables (AIC/BIC by k)
- Parameter tables
- Moment tables
- Univariate and multivariate summary comparisons

Plots (when enabled):
- Per-family best density: `<symbol>_<family>_density.png`
- Cross-family best plot: `<symbol>_all.png`
- Each plot includes KDE and a normal reference density.

Limitations
-----------
- Multivariate t mixture uses a fixed dof; estimating dof is not implemented.
- R scripts and `xreturns_mix.py` are not covered here.
