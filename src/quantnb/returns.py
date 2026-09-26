"""Return and risk arithmetic — the conventions every session uses (taught in Session 2.3).

Conventions
  * Prices are a pandas Series (or DataFrame, one column per asset) indexed by timestamp.
  * `simple` returns R_t = P_t / P_{t-1} - 1 ; `log` returns r_t = ln(P_t / P_{t-1}).
  * Annualisation uses PERIODS_PER_YEAR = 252 trading days unless told otherwise.
  * Volatility scales with sqrt(time); mean returns scale linearly.
  * The Sharpe ratio is computed on EXCESS simple returns and annualised by sqrt(periods).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(prices / prices.shift(1)).iloc[1:]


def simple_to_log(r: pd.Series | np.ndarray | float):
    return np.log1p(r)


def log_to_simple(r: pd.Series | np.ndarray | float):
    return np.expm1(r)


def cumulative_growth(simple: pd.Series) -> pd.Series:
    """Wealth path of $1 invested: prod(1 + R)."""
    return (1.0 + simple).cumprod()


def annualised_return(simple: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Geometric (compound) annual growth rate."""
    growth = float((1.0 + simple).prod())
    years = len(simple) / periods
    return growth ** (1.0 / years) - 1.0


def annualised_vol(simple: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    return float(simple.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(simple: pd.Series, rf_per_period: float | pd.Series = 0.0,
                 periods: int = PERIODS_PER_YEAR) -> float:
    excess = simple - rf_per_period
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods))


def drawdown(simple: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak of the wealth path (≤ 0)."""
    wealth = cumulative_growth(simple)
    return wealth / wealth.cummax() - 1.0


def max_drawdown(simple: pd.Series) -> float:
    return float(drawdown(simple).min())


def volatility_drag(sigma: float) -> float:
    """Approximate gap between arithmetic and geometric mean return: sigma^2 / 2."""
    return sigma ** 2 / 2.0


def leveraged_returns(simple: pd.Series, leverage: float, borrow_rate_per_period: float = 0.0) -> pd.Series:
    """Daily-rebalanced leverage L: R_L = L*R - (L-1)*borrow cost."""
    return leverage * simple - (leverage - 1.0) * borrow_rate_per_period


def in_years(values, periods_per_year: int = PERIODS_PER_YEAR) -> pd.Series:
    """Re-index a simulated series by elapsed years (1/periods, 2/periods, …).

    Simulated data must not show calendar dates on charts: they would imply real history.
    """
    v = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    return pd.Series(v, index=np.arange(1, len(v) + 1) / periods_per_year)


def max_drawdown_paths(simple: np.ndarray) -> np.ndarray:
    """Maximum drawdown (≤ 0) of each column of a (periods × paths) array of simple returns."""
    log_w = np.vstack([np.zeros((1, simple.shape[1])), np.log1p(simple).cumsum(axis=0)])
    return np.expm1((log_w - np.maximum.accumulate(log_w, axis=0)).min(axis=0))
