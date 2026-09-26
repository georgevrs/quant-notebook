"""Seeded synthetic market data — the default data source for labs and CI.

Synthetic data is not a toy: it is the only data where you KNOW the truth (the drift, the
volatility, whether a signal exists), which makes it the right tool for testing whether a method
works before you trust it on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .returns import PERIODS_PER_YEAR


def business_days(n: int, start: str = "2010-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def gbm_prices(n_days: int, mu: float = 0.07, sigma: float = 0.18, s0: float = 100.0,
               n_paths: int = 1, rng: np.random.Generator | None = None,
               start: str = "2010-01-04") -> pd.DataFrame:
    """Geometric Brownian motion with annual drift `mu` and volatility `sigma`.

    Log returns are normal with mean (mu - sigma^2/2) dt and sd sigma sqrt(dt), so the
    expected SIMPLE growth rate is mu per year and the median path grows at mu - sigma^2/2.
    Returns a DataFrame of shape (n_days + 1, n_paths) starting at s0.
    """
    rng = rng or np.random.default_rng()
    dt = 1.0 / PERIODS_PER_YEAR
    z = rng.standard_normal((n_days, n_paths))
    log_r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    paths = s0 * np.exp(np.vstack([np.zeros((1, n_paths)), np.cumsum(log_r, axis=0)]))
    cols = [f"path{i}" for i in range(n_paths)] if n_paths > 1 else ["price"]
    return pd.DataFrame(paths, index=business_days(n_days + 1, start), columns=cols)


def garch_returns(n_days: int, omega: float = 2e-6, alpha: float = 0.08, beta: float = 0.90,
                  mu: float = 0.0003, nu: float | None = 6.0,
                  rng: np.random.Generator | None = None, start: str = "2010-01-04") -> pd.DataFrame:
    """GARCH(1,1) daily simple returns with Student-t shocks (nu=None → normal).

    Produces the stylized facts of real returns: volatility clustering and fat tails.
    Returns a DataFrame with columns ['ret', 'sigma'] (daily conditional volatility).
    """
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be < 1 for a stationary GARCH(1,1)")
    rng = rng or np.random.default_rng()
    if nu is None:
        z = rng.standard_normal(n_days)
    else:
        z = rng.standard_t(nu, n_days) / np.sqrt(nu / (nu - 2.0))  # unit variance
    var = np.empty(n_days)
    eps = np.empty(n_days)
    var[0] = omega / (1.0 - alpha - beta)
    for t in range(n_days):
        if t > 0:
            var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * z[t]
    return pd.DataFrame({"ret": mu + eps, "sigma": np.sqrt(var)}, index=business_days(n_days, start))


def factor_model_returns(n_days: int, n_assets: int, n_factors: int = 3,
                         factor_vol: float = 0.15, idio_vol: float = 0.25,
                         alphas: np.ndarray | None = None,
                         rng: np.random.Generator | None = None,
                         start: str = "2010-01-04") -> dict[str, pd.DataFrame]:
    """Linear factor model R = alpha + B f + e with known truth.

    Returns {'returns': T×N, 'factors': T×K, 'loadings': N×K, 'alphas': N×1} (daily, simple).
    """
    rng = rng or np.random.default_rng()
    idx = business_days(n_days, start)
    daily_f = factor_vol / np.sqrt(PERIODS_PER_YEAR)
    daily_e = idio_vol / np.sqrt(PERIODS_PER_YEAR)
    f = rng.standard_normal((n_days, n_factors)) * daily_f
    B = rng.normal(1.0 if n_factors else 0.0, 0.4, (n_assets, n_factors))
    if n_factors > 1:
        B[:, 1:] = rng.normal(0.0, 0.6, (n_assets, n_factors - 1))
    a = np.zeros(n_assets) if alphas is None else np.asarray(alphas, dtype=float) / PERIODS_PER_YEAR
    e = rng.standard_normal((n_days, n_assets)) * daily_e
    R = a + f @ B.T + e
    assets = [f"A{i:03d}" for i in range(n_assets)]
    factors = [f"F{k}" for k in range(n_factors)]
    return {
        "returns": pd.DataFrame(R, index=idx, columns=assets),
        "factors": pd.DataFrame(f, index=idx, columns=factors),
        "loadings": pd.DataFrame(B, index=assets, columns=factors),
        "alphas": pd.DataFrame({"alpha_annual": a * PERIODS_PER_YEAR}, index=assets),
    }


def jump_diffusion_returns(n_days: int, n_paths: int = 1, mu: float = 0.07, sigma: float = 0.15,
                           jump_rate: float = 0.5, jump_mean: float = -0.10, jump_sd: float = 0.05,
                           rng: np.random.Generator | None = None, compensate: bool = True) -> np.ndarray:
    """Merton jump-diffusion daily SIMPLE returns, shape (n_days, n_paths).

    Log returns = diffusion + a Poisson(jump_rate per year) number of normal log-jumps
    N(jump_mean, jump_sd²). With compensate=True the diffusion drift is lowered so the expected simple
    growth rate is still mu per year — jumps then change the shape of returns (skew, tails), not their
    mean. Used for crash-prone strategies (Sessions 2.2, 2.5, 3.2).
    """
    rng = rng or np.random.default_rng()
    dt = 1.0 / PERIODS_PER_YEAR
    k = np.exp(jump_mean + 0.5 * jump_sd ** 2) - 1.0  # expected simple jump size
    drift = mu - 0.5 * sigma ** 2 - (jump_rate * k if compensate else 0.0)
    diff = drift * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_days, n_paths))
    n_jumps = rng.poisson(jump_rate * dt, (n_days, n_paths))
    jumps = n_jumps * jump_mean + np.sqrt(n_jumps) * jump_sd * rng.standard_normal((n_days, n_paths))
    return np.expm1(diff + jumps)
