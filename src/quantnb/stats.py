"""Small statistical helpers the labs share (taught in Sessions 1.3, 1.4, 2.3, 5.2, 6.7).

Sharpe ratios here are annualised and computed on excess returns you pass in; with daily data,
`periods_per_year=252`. Everything assumes IID returns unless stated — the sessions that use these
helpers say when that assumption bites (autocorrelation: 5.2; fat tails and many trials: 6.7).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .returns import PERIODS_PER_YEAR

EULER_GAMMA = 0.5772156649015329


def annual_sharpe(excess, axis: int = 0, periods_per_year: int = PERIODS_PER_YEAR):
    """Annualised Sharpe ratio of excess returns; works on numpy arrays, Series and DataFrames."""
    x = np.asarray(excess, dtype=float)
    return x.mean(axis=axis) / x.std(axis=axis, ddof=1) * np.sqrt(periods_per_year)


def sharpe_se(sr_annual: float, years: float, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Standard error of an annualised Sharpe ratio estimated from `years` of data (Lo, 2002, IID).

    Lo's result is SE(SR_p) = sqrt((1 + SR_p²/2) / n) for the PER-PERIOD Sharpe SR_p over n periods.
    Annualised, that is sqrt((1 + SR_p²/2) / years) with SR_p = sr_annual / sqrt(periods_per_year):
    for daily data the correction term is negligible and SE ≈ 1/sqrt(years).
    """
    sr_p = sr_annual / np.sqrt(periods_per_year)
    return float(np.sqrt((1.0 + 0.5 * sr_p ** 2) / years))


def years_needed(sr_annual: float, z: float = 1.96, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Years of data before a true Sharpe of `sr_annual` sits z standard errors above zero."""
    sr_p = sr_annual / np.sqrt(periods_per_year)
    return float(z ** 2 * (1.0 + 0.5 * sr_p ** 2) / sr_annual ** 2)


def expected_max_sharpe(n_trials: int, years: float, sr_sd: float | None = None) -> float:
    """Expected best estimated Sharpe among n skill-less trials (False Strategy Theorem approximation).

    (1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)), scaled by the standard deviation of a Sharpe estimate
    (default 1/sqrt(years)). Valid for N ≥ 2; returns 0 for N = 1.
    """
    if n_trials < 2:
        return 0.0
    sd = sr_sd if sr_sd is not None else 1.0 / np.sqrt(years)
    z = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    return float(z * sd)


def oos_r2(y, y_hat) -> float:
    """Out-of-sample R² against a ZERO forecast (the Gu-Kelly-Xiu convention for returns)."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(1.0 - np.sum((y - y_hat) ** 2) / np.sum(y ** 2))


def factor_regression(y, X, periods_per_year: int = PERIODS_PER_YEAR, names: list[str] | None = None) -> dict:
    """OLS of excess returns y on factor returns X (with an intercept), classical standard errors.

    Returns {'alpha_annual', 'alpha_se_annual', 'alpha_t', 'betas', 'beta_se', 'beta_t' (dicts), 'r2', 'resid_vol_annual'}.
    Classical (IID) standard errors: Session 5.2 explains when you need HAC/Newey-West instead.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = n - k - 1
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    names = names or [f"f{i}" for i in range(k)]
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "alpha_annual": float(coef[0] * periods_per_year),
        "alpha_se_annual": float(se[0] * periods_per_year),
        "alpha_t": float(coef[0] / se[0]),
        "betas": {nm: float(b) for nm, b in zip(names, coef[1:])},
        "beta_se": {nm: float(s) for nm, s in zip(names, se[1:])},
        "beta_t": {nm: float(b / s) for nm, b, s in zip(names, coef[1:], se[1:])},
        "r2": float(1.0 - resid @ resid / ss_tot),
        "resid_vol_annual": float(np.sqrt(s2 * periods_per_year)),
    }


def skewness(x) -> float:
    """Sample skewness (population form, third standardised moment) of a pooled array."""
    x = np.asarray(x, dtype=float).ravel()
    d = x - x.mean()
    return float((d ** 3).mean() / (d ** 2).mean() ** 1.5)


def batch_se(stat, x, n_batches: int = 50) -> float:
    """Batch-means standard error of any statistic `stat(array)` computed over Monte Carlo paths.

    Splits the paths (axis 0) into `n_batches` groups, evaluates the statistic in each, and returns
    std(batch values) / sqrt(n_batches). Use it for statistics with no simple closed-form SE
    (skewness, quantiles, maximum drawdown).
    """
    x = np.asarray(x)
    vals = np.array([stat(b) for b in np.array_split(x, n_batches, axis=0)], dtype=float)
    return float(vals.std(ddof=1) / np.sqrt(n_batches))


BGK_BETA = 0.5826  # Broadie–Glasserman–Kou continuity-correction constant, −ζ(1/2)/√(2π)


def p_touch(barrier_ratio: float, mu: float, sigma: float, tau: float, dt: float | None = None) -> float:
    """Probability that a GBM price touches barrier_ratio × start before time tau (reflection principle).

    barrier_ratio > 1 is an up-barrier (e.g. 2.0 = "the price doubles"), < 1 a down-barrier (e.g. a
    margin-call level). mu and sigma are the annual drift and volatility of the price; tau is in years.
    If `dt` (the monitoring interval in years) is given, the barrier is shifted away from the start by
    exp(±β σ √dt) — the Broadie–Glasserman–Kou correction for checking only at discrete times.
    Taught in Sessions 2.4 and 3.5.
    """
    if barrier_ratio == 1.0:
        return 1.0
    up = barrier_ratio > 1.0
    h = np.log(barrier_ratio)
    if dt:
        h += (1 if up else -1) * BGK_BETA * sigma * np.sqrt(dt)
    nu = mu - 0.5 * sigma ** 2  # drift of the log price
    s = sigma * np.sqrt(tau)
    if up:
        return float(norm.cdf((-h + nu * tau) / s) + np.exp(2 * nu * h / sigma ** 2) * norm.cdf((-h - nu * tau) / s))
    return float(norm.cdf((h - nu * tau) / s) + np.exp(2 * nu * h / sigma ** 2) * norm.cdf((h + nu * tau) / s))
