"""European option pricing helpers — a black box until Session 10.3 derives it.

Session 2.7 uses these to show what implied volatility is; Unit 10 derives Black-Scholes-Merton, the
Greeks and the volatility surface properly. Notation: S spot, K strike, τ (tau) time to expiry in
years, r_f the continuously compounded risk-free rate, q the continuous dividend yield, σ volatility.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def call_payoff(s_exp, k):
    return np.maximum(np.asarray(s_exp, dtype=float) - k, 0.0)


def put_payoff(s_exp, k):
    return np.maximum(k - np.asarray(s_exp, dtype=float), 0.0)


def bs_price(s, k, tau, r_f, sigma, q: float = 0.0, kind: str = "call"):
    """Black-Scholes-Merton price of a European call or put (vectorised)."""
    s, k, tau, sigma = (np.asarray(v, dtype=float) for v in (s, k, tau, sigma))
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(s / k) + (r_f - q + 0.5 * sigma ** 2) * tau) / vol
    d2 = d1 - vol
    if kind == "call":
        return s * np.exp(-q * tau) * norm.cdf(d1) - k * np.exp(-r_f * tau) * norm.cdf(d2)
    if kind == "put":
        return k * np.exp(-r_f * tau) * norm.cdf(-d2) - s * np.exp(-q * tau) * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def implied_vol(price, s, k, tau, r_f, q: float = 0.0, kind: str = "call",
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-10, max_iter: int = 200) -> tuple[float, int]:
    """Volatility that reproduces `price`, by bisection. Returns (sigma, iterations).

    Raises ValueError if the price is outside the no-arbitrage bounds (no volatility fits it).
    """
    p_lo = float(bs_price(s, k, tau, r_f, lo, q, kind))
    p_hi = float(bs_price(s, k, tau, r_f, hi, q, kind))
    if not (p_lo <= price <= p_hi):
        raise ValueError(f"price {price} outside [{p_lo:.6f}, {p_hi:.6f}]: no implied volatility")
    for i in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        if float(bs_price(s, k, tau, r_f, mid, q, kind)) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return 0.5 * (lo + hi), i
    return 0.5 * (lo + hi), max_iter
