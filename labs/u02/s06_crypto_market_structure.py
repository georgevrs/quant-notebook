# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2.6 · Crypto Market Structure — companion lab
#
# **Quant Notebook** · Unit 2 · Session 6 · [Read the session](https://georgevrs.github.io/quant-notebook/unit02-markets-instruments/session06-crypto-market-structure.html)
#
# Centralised vs decentralised venues, AMMs, perpetual futures and funding, basis trades, and the custody and counterparty risks that traditional markets hide from you.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Everything here is **synthetic and illustrative**: pool sizes, volatilities, premiums and margin
# rates are round numbers chosen to teach the mechanics, not market data. The funding formula is
# the one Binance and Hyperliquid publish (checked 2026); the AMM follows Uniswap v2's
# constant-product rule with its 0.30% fee. Crypto trades every day, so a year here is **365 days**
# (8,760 hours), and simulated time is elapsed time, never a calendar date.
#
# **Risk-free rate 0; collateral earns nothing.** (The funding formula's own "interest" component ι
# is a fixed payment between traders written into the venue's rules, not a return on cash.)

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm

import quantnb as qn
from quantnb import charts

lab = qn.Lab("2.6")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
rng_il, rng_perp, rng_liq = lab.rng.spawn(3)

DAYS_PER_YEAR = 365                     # crypto never closes: annualise with 365 days, not 252
HOURS_PER_YEAR = 24 * DAYS_PER_YEAR     # 8,760
lab.record("days_per_year", DAYS_PER_YEAR)
lab.record("hours_per_year", HOURS_PER_YEAR)

# %% [markdown]
# ## 1 · An automated market maker: the constant-product pool
#
# A Uniswap-v2-style pool holds `x` tokens and `y` dollars (a dollar stablecoin) and lets anyone
# trade against it as long as the product of the reserves does not fall: `x · y = k`. The pool's
# price is the ratio `y / x`. The 0.30% fee is taken from what the trader pays in: only
# `(1 − fee)` of the input counts toward the invariant, but all of it stays in the pool, so `k`
# creeps up with every trade and that growth is the liquidity providers' fee income.

# %%
X0, Y0 = 1_000.0, 2_000_000.0      # illustrative pool: 1,000 tokens and $2,000,000 → price $2,000
FEE = 0.003                        # Uniswap v2's fee on every swap (0.30% of the input)


def swap(amount_in: float, reserve_in: float, reserve_out: float, fee: float = FEE) -> tuple[float, float, float]:
    """Pay `amount_in` of one asset into a constant-product pool; return (amount out, new reserves in, out).

    Only amount_in·(1 − fee) counts toward the invariant (reserve_in + eff)·(reserve_out − out) = k,
    so the trader gets `out`; the whole amount_in stays in the pool, which is how LPs earn the fee.
    """
    eff = amount_in * (1.0 - fee)
    out = reserve_out * eff / (reserve_in + eff)
    return out, reserve_in + amount_in, reserve_out - out


def sell_cost_closed_form(s: float, fee: float = FEE) -> float:
    """Total cost of selling a fraction s of the token reserve, vs the pre-trade price: fee + impact."""
    return 1.0 - (1.0 - fee) / (1.0 + (1.0 - fee) * s)


mid0 = Y0 / X0
lab.record("amm_x0", X0)
lab.record("amm_y0", Y0)
lab.record("amm_mid0", mid0)
lab.record("amm_fee", FEE)
lab.record("amm_k0", X0 * Y0)

SIZES = [0.001, 0.01, 0.05, 0.10, 0.25]    # trade size as a fraction of the pool's token reserve
rows = []
for s in SIZES:
    dx = s * X0
    dy, x1, y1 = swap(dx, X0, Y0)
    exec_px = dy / dx
    cost = 1.0 - exec_px / mid0
    dy_nf, x1_nf, y1_nf = swap(dx, X0, Y0, fee=0.0)
    rows.append({"s": s, "tokens": dx, "dollars_at_mid": dx * mid0, "exec": exec_px, "cost": cost,
                 "impact_only": 1.0 - (dy_nf / dx) / mid0, "mid_after": y1 / x1})
    # the closed form and the swap agree exactly
    assert abs(cost - sell_cost_closed_form(s)) < 1e-12
    # the fee stays in the pool: k never falls, and grows when a fee is charged
    assert x1 * y1 >= X0 * Y0 and abs(x1_nf * y1_nf - X0 * Y0) < 1e-6 * X0 * Y0
    # without a fee, the average price you get is the geometric mean of the before and after prices
    assert abs(dy_nf / dx - np.sqrt(mid0 * (y1_nf / x1_nf))) < 1e-9 * mid0
amm = pd.DataFrame(rows)
print(amm.round(5))
for row in amm.itertuples():
    key = f"{row.s * 100:g}".replace(".", "_")
    lab.record(f"amm_{key}_s", row.s)
    lab.record(f"amm_{key}_tokens", row.tokens)
    lab.record(f"amm_{key}_dollars", row.dollars_at_mid)
    lab.record(f"amm_{key}_exec", row.exec)
    lab.record(f"amm_{key}_cost", row.cost)
    lab.record(f"amm_{key}_impact", row.impact_only)
    lab.record(f"amm_{key}_mid_after", row.mid_after)

# The same dollar trade in a pool ten times deeper costs about a tenth of the impact.
DEPTH_MULT = 10
s_deep = 0.01 / DEPTH_MULT
lab.record("amm_deep_mult", DEPTH_MULT)
lab.record("amm_deep_1_cost", sell_cost_closed_form(s_deep))
assert sell_cost_closed_form(s_deep) < sell_cost_closed_form(0.01)

# An immediate round trip (sell, then spend the dollars buying back) loses about two fees, not two
# impacts: the curve is the same in both directions, so the impact is handed back on the way in.
# (In a live pool other traders and arbitrageurs move the price in between, so don't count on it.)
dx = 0.05 * X0
dy, x1, y1 = swap(dx, X0, Y0)
tokens_back, _, _ = swap(dy, y1, x1)                      # pay the dollars back in, receive tokens
lab.record("amm_roundtrip_5_loss", 1.0 - tokens_back / dx)
assert tokens_back < dx
assert abs((1.0 - tokens_back / dx) - (1.0 - (1.0 - FEE) ** 2)) < 0.001   # ≈ two fees

# The same dollar sale into pools of three depths: cost ≈ fee + (trade size ÷ pool size).
trade_k = np.linspace(0.0, 400.0, 81)                   # trade size, $ thousands at the pre-trade price
depth_series = []
for mult, role, label in ((1, "strategy", "lab pool"), (10, "alt1", "10× deeper"), (100, "alt2", "100× deeper")):
    s_grid = 1_000.0 * trade_k / (mid0 * X0 * mult)     # share of the pool's token reserve
    depth_series.append(charts.Series(label, pd.Series([sell_cost_closed_form(g) for g in s_grid], index=trade_k),
                                      role=role))
lab.chart("amm_cost", charts.line_chart(
    depth_series, title="Cost of a sale (fee + impact) by pool depth (x = trade size, $ thousands)",
    y_fmt=charts.fmt_pct(1)))

# %% [markdown]
# ## 2 · Impermanent loss: the price of being the other side of every trade
#
# A liquidity provider (LP) deposits equal values of both assets. When the outside price moves,
# arbitrageurs trade against the pool until its price matches — they buy what has become cheap
# in the pool and sell what has become dear, so the LP ends up holding more of whatever fell.
# With no fees, the pool state depends only on the current price: `x = √(k/P)`, `y = √(k·P)`, so
# the LP's position is worth `2√(k·P)`. If the price moves by a ratio θ = P_τ/P_0 over τ years:
#
#     LP value / start = √θ,   holding value / start = (1 + θ)/2,
#     impermanent loss  IL(θ) = LP / holding − 1 = 2√θ/(1 + θ) − 1   (≤ 0, zero only at θ = 1).
#
# First we let an arbitrageur trade against a pool along simulated hourly price paths, using only
# the swap function above, and check that the LP's result matches the closed form on every path.

# %%
def arbitrage_to(price: float, x: float, y: float) -> tuple[float, float]:
    """No-fee arbitrage: trade against the pool until its price y/x equals the outside price.

    The arbitrageur sells tokens if the pool is too dear, buys them if it is too cheap; the size
    comes from the invariant (x_new · y_new = k with y_new/x_new = price → x_new = √(k/price)).
    """
    k = x * y
    x_target = np.sqrt(k / price)
    if x_target > x:                       # pool price too high: sell tokens into it
        out, x_new, y_new = swap(x_target - x, x, y, fee=0.0)
    else:                                  # pool price too low: buy tokens with dollars
        y_target = np.sqrt(k * price)
        out, y_new, x_new = swap(y_target - y, y, x, fee=0.0)
    return x_new, y_new


def impermanent_loss(theta):
    """LP value relative to simply holding the initial deposit, minus 1 (constant product, no fees)."""
    return 2.0 * np.sqrt(theta) / (1.0 + theta) - 1.0


IL_SIGMA = 0.80                  # illustrative crypto-like annual volatility for this section
IL_PATHS, IL_DAYS = 40, 30       # a month of hourly arbitrage on 40 paths
steps = 24 * IL_DAYS
dt_h = 1.0 / HOURS_PER_YEAR
z = rng_il.standard_normal((steps, IL_PATHS))
prices = mid0 * np.exp(np.cumsum(-0.5 * IL_SIGMA ** 2 * dt_h + IL_SIGMA * np.sqrt(dt_h) * z, axis=0))
il_sim, il_theory = [], []
for j in range(IL_PATHS):
    x, y = X0, Y0
    for t in range(steps):
        x, y = arbitrage_to(prices[t, j], x, y)
    p_end = prices[-1, j]
    lp_val = x * p_end + y
    hold_val = X0 * p_end + Y0
    il_sim.append(lp_val / hold_val - 1.0)
    il_theory.append(impermanent_loss(p_end / mid0))
il_sim, il_theory = np.array(il_sim), np.array(il_theory)
# path independence: 720 arbitrage trades later, only the final price matters
assert np.allclose(il_sim, il_theory, atol=1e-9)
lab.record("il_paths_checked", IL_PATHS)
lab.record("il_trades_per_path", steps)
lab.record("il_max_abs_gap", float(np.abs(il_sim - il_theory).max()))

THETAS = {"m50": 0.5, "m20": 0.8, "p20": 1.2, "p100": 2.0, "p300": 4.0}
for key, th in THETAS.items():
    lab.record(f"il_{key}", float(impermanent_loss(th)))
assert abs(impermanent_loss(0.5) - impermanent_loss(2.0)) < 1e-12   # a halving hurts as much as a doubling

# %% [markdown]
# **The expected cost.** With a driftless price (zero expected return), holding is worth its
# starting value on average, but the LP's `√θ` is concave, so on average it is worth less:
# `E[√θ] = exp(−σ²τ/8)` (τ in years). The expected loss relative to holding is
# `exp(−σ²τ/8) − 1 ≈ −σ²τ/8`: an LP is short volatility, and the fees must beat roughly σ²/8 a year
# just to break even with holding. We check the formula by Monte Carlo (200,000 one-year outcomes,
# 4 standard errors).

# %%
N_IL_MC, TAU_IL = 200_000, 1.0
theta = np.exp(IL_SIGMA * np.sqrt(TAU_IL) * rng_il.standard_normal(N_IL_MC) - 0.5 * IL_SIGMA ** 2 * TAU_IL)
rel = np.sqrt(theta) - (1.0 + theta) / 2.0          # (LP − holding) / starting value
mc_mean, mc_se = rel.mean(), rel.std(ddof=1) / np.sqrt(N_IL_MC)
theory = np.exp(-IL_SIGMA ** 2 * TAU_IL / 8.0) - 1.0
print(f"E[LP − hold]/V0: Monte Carlo {mc_mean:.4%} ± {mc_se:.4%}, theory {theory:.4%}")
assert abs(mc_mean - theory) < 4 * mc_se
lab.record("il_sigma", IL_SIGMA)
lab.record("il_mc_n", N_IL_MC)
lab.record("il_mc_mean", mc_mean)
lab.record("il_mc_se", mc_se)
lab.record("il_theory_mean", theory)
lab.record("il_approx_mean", -IL_SIGMA ** 2 * TAU_IL / 8.0)
# Fee income (a fraction of the deposit) a year's LP needs just to match holding: 1 − exp(−σ²/8)
for sig in (0.5, 0.8, 1.0):
    lab.record(f"il_breakeven_sigma_{int(sig * 100)}", sig)
    lab.record(f"il_breakeven_fee_{int(sig * 100)}", 1.0 - np.exp(-sig ** 2 / 8.0))

th_grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
lab.chart("il_curve", charts.line_chart(
    [charts.Series("hold the deposit", pd.Series((1 + th_grid) / 2, index=th_grid), role="benchmark", end_label="hold"),
     charts.Series("provide liquidity", pd.Series(np.sqrt(th_grid), index=th_grid), role="strategy", end_label="LP")],
    title="Value of $1 deposited, by final price ÷ starting price (constant product, no fees)",
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

# %% [markdown]
# ## 3 · Perpetual futures: funding pulls the perp back to spot
#
# A perpetual future never expires, so nothing forces it to converge to spot. The **funding rate**
# does the job: every funding interval (8 hours on the major venues' default schedule) longs pay
# shorts `funding × position notional` when the perp trades above spot, and shorts pay longs when
# it trades below. The published formula (Binance, Hyperliquid) is
#
#     funding = P̄ + clamp(ι − P̄, −0.05%, +0.05%)
#
# with P̄ the premium (perp ÷ spot − 1) averaged over the interval and ι an interest component of
# 0.01% per 8 hours. A premium therefore costs longs roughly itself, every interval, which pays
# arbitrageurs to sell the perp and buy spot.
#
# **A toy model.** Spot follows a geometric random walk (60% volatility) hour by hour. The premium
# is pushed by speculative demand — a small steady long bias plus shocks that are positively
# correlated with spot moves (trend chasing) — and pulled back by arbitrage, which closes a
# fraction ψ of the gap each hour because funding pays it to (half-life 8 hours):
#
#     π_{t+1} = (1 − ψ) π_t + η + σ_π ε_{t+1}
#
# The counterfactual switches funding off: nobody is paid to close the gap, ψ = 0, and the premium
# is a random walk with drift. This is a stylised model: it ASSUMES that funding-paid arbitrage
# closes ψ of the gap each hour (the funding computed below never feeds back into the premium), so
# it illustrates the mechanism; it does not test it. (ψ, not κ: Unit 2 uses κ for 2.2's borrow fee.)

# %%
SIGMA_SPOT = 0.60                        # annual volatility of spot (three times the 2.3 asset's 20%)
HALF_LIFE_H = 8                          # hours for arbitrage to close half the premium
PULL = 1.0 - 0.5 ** (1.0 / HALF_LIFE_H)  # ψ: fraction of the premium arbitrage closes each hour
SIGMA_PREM = 0.0002                      # hourly premium shock: 2 basis points
MEAN_PREM = 0.0002                       # steady long demand → average premium of 0.02% with funding on
ETA = MEAN_PREM * PULL                   # hourly push that produces that average
CORR_DEMAND = 0.5                        # premium shocks move with spot shocks (trend chasing)
IOTA, CLAMP = 0.0001, 0.0005             # interest component per 8 h, and the ±0.05% clamp
FUND_HOURS = 8
N_PERP, PERP_HOURS = 2_000, HOURS_PER_YEAR
N_INTERVALS = PERP_HOURS // FUND_HOURS   # 1,095 funding payments a year
assert PERP_HOURS % FUND_HOURS == 0


def funding_rate(avg_premium: np.ndarray) -> np.ndarray:
    """The published perp funding formula for one interval: premium + clamp(interest − premium)."""
    return avg_premium + np.clip(IOTA - avg_premium, -CLAMP, CLAMP)


def simulate_perp(pull: float, pay_funding: bool, rng: np.random.Generator, keep: int = 1) -> dict:
    """Hourly spot and premium for N_PERP paths over one year; funding every FUND_HOURS hours.

    Returns the funding rates per interval, the spot and perp prices at each funding time (the
    notional funding is charged on), the final premium, and `keep` full premium paths for a chart.
    """
    dt = 1.0 / HOURS_PER_YEAR
    log_s = np.zeros(N_PERP)
    prem = np.zeros(N_PERP)
    acc = np.zeros(N_PERP)
    fund = np.empty((N_INTERVALS, N_PERP))
    avg_prem = np.empty((N_INTERVALS, N_PERP))
    perp_at_fund = np.empty((N_INTERVALS, N_PERP))
    kept = np.empty((PERP_HOURS + 1, keep))
    kept[0] = 0.0
    for t in range(1, PERP_HOURS + 1):
        z_s = rng.standard_normal(N_PERP)
        z_d = CORR_DEMAND * z_s + np.sqrt(1 - CORR_DEMAND ** 2) * rng.standard_normal(N_PERP)
        log_s += -0.5 * SIGMA_SPOT ** 2 * dt + SIGMA_SPOT * np.sqrt(dt) * z_s
        prem = (1.0 - pull) * prem + ETA + SIGMA_PREM * z_d
        acc += prem
        kept[t] = prem[:keep]
        if t % FUND_HOURS == 0:
            k = t // FUND_HOURS - 1
            avg_prem[k] = acc / FUND_HOURS
            fund[k] = funding_rate(avg_prem[k]) if pay_funding else 0.0
            perp_at_fund[k] = np.exp(log_s) * (1.0 + prem)       # spot starts at 1: prices in units of S_0
            acc[:] = 0.0
    return {"fund": fund, "avg_prem": avg_prem, "perp_at_fund": perp_at_fund, "spot_end": np.exp(log_s),
            "prem_end": prem, "kept": kept}


# Common random numbers: the same shocks drive both worlds, so only the anchor differs.
seed_perp = int(rng_perp.integers(2 ** 32))
with_f = simulate_perp(PULL, True, np.random.default_rng(seed_perp))
no_f = simulate_perp(0.0, False, np.random.default_rng(seed_perp))

# Theory for the premium after one year (8,760 hours: the start has long been forgotten).
m_theory = ETA / PULL
v_theory = SIGMA_PREM ** 2 / (1.0 - (1.0 - PULL) ** 2)
m_rw, v_rw = ETA * PERP_HOURS, SIGMA_PREM ** 2 * PERP_HOURS
for label, res, m_th, v_th in (("with", with_f, m_theory, v_theory), ("without", no_f, m_rw, v_rw)):
    x = res["prem_end"]
    m_se, v_se = x.std(ddof=1) / np.sqrt(N_PERP), x.var(ddof=1) * np.sqrt(2.0 / (N_PERP - 1))
    print(f"premium after a year, funding {label}: mean {x.mean():+.4%} (theory {m_th:+.4%}), "
          f"sd {x.std(ddof=1):.4%} (theory {np.sqrt(v_th):.4%})")
    assert abs(x.mean() - m_th) < 4 * m_se, "premium mean vs theory"
    assert abs(x.var(ddof=1) - v_th) < 4 * v_se, "premium variance vs theory"
    lab.record(f"perp_{label}_mean_end", float(x.mean()))
    lab.record(f"perp_{label}_sd_end", float(x.std(ddof=1)))
    lab.record(f"perp_{label}_mean_theory", m_th)
    lab.record(f"perp_{label}_sd_theory", float(np.sqrt(v_th)))
    lab.record(f"perp_{label}_p99_abs_end", float(np.quantile(np.abs(x), 0.99)))

fund = with_f["fund"]
fund_annual = fund.sum(axis=0)                            # a year of funding, per unit of notional
lab.record("perp_sigma_spot", SIGMA_SPOT)
lab.record("perp_half_life_h", HALF_LIFE_H)
lab.record("perp_pull", PULL)
lab.record("perp_sigma_prem_bp", SIGMA_PREM * 1e4)
lab.record("perp_mean_prem", MEAN_PREM)
lab.record("perp_corr_demand", CORR_DEMAND)
lab.record("perp_iota", IOTA)
lab.record("perp_iota_annual", IOTA * N_INTERVALS)
lab.record("perp_clamp", CLAMP)
lab.record("perp_n_paths", N_PERP)
lab.record("perp_n_intervals", N_INTERVALS)
lab.record("perp_share_at_iota", float(np.mean(np.isclose(fund, IOTA))))
lab.record("perp_share_negative", float(np.mean(fund < 0)))
lab.record("perp_fund_annual_mean", float(fund_annual.mean()))
lab.record("perp_fund_annual_p05", float(np.quantile(fund_annual, 0.05)))
lab.record("perp_fund_annual_p95", float(np.quantile(fund_annual, 0.95)))
lab.record("perp_fund_max_interval", float(fund.max()))
lab.record("perp_fund_min_interval", float(fund.min()))
# The clamp, checked on every interval: inside the band (ι − 0.05%, ι + 0.05%) funding is exactly ι;
# above it, the premium minus 0.05%; below it, the premium plus 0.05%.
avg = with_f["avg_prem"]
inside = (avg > IOTA - CLAMP) & (avg < IOTA + CLAMP)
assert np.allclose(fund[inside], IOTA, rtol=0, atol=1e-15)
assert np.allclose(fund[avg >= IOTA + CLAMP], avg[avg >= IOTA + CLAMP] - CLAMP, rtol=0, atol=1e-15)
assert np.allclose(fund[avg <= IOTA - CLAMP], avg[avg <= IOTA - CLAMP] + CLAMP, rtol=0, atol=1e-15)
lab.record("perp_band_lo", IOTA - CLAMP)
lab.record("perp_band_hi", IOTA + CLAMP)

days = np.arange(PERP_HOURS + 1) / 24.0
lab.chart("perp_premium", charts.line_chart(
    [charts.Series("with funding", pd.Series(with_f["kept"][:, 0], index=days), role="strategy"),
     charts.Series("no funding (counterfactual)", pd.Series(no_f["kept"][:, 0], index=days), role="loss",
                   end_label="no funding")],
    title="Perp premium over spot, one simulated year (x = elapsed days)",
    y_fmt=charts.fmt_pct(0), hline=0.0))

# %% [markdown]
# ## 4 · The basis trade with a perp: collecting funding
#
# The dated-future cash-and-carry (buy spot, sell the future, deliver) is Session 2.4's; its lab
# checks that the locked-in P&L does not depend on the price at expiry. Here: the perp version.
#
# **Perp funding trade.** Long spot, short the perp, same notional, for a year. On the simulated
# paths of section 3 the P&L is exactly: funding received minus the change in the premium,
#
#     P&L = Σ_j funding_j × perp price_j − (S_T·π_T − S_0·π_0),
#
# because the spot and perp legs cancel except for the gap between them. We check the identity on
# every path, then look at the distribution. Fees: four taker trades (in and out, both legs).

# %%
TAKER_FEE = 0.0005                        # 0.05% a trade, illustrative
fees = 4 * TAKER_FEE
spot_T, prem_T = with_f["spot_end"], with_f["prem_end"]
perp_T = spot_T * (1.0 + prem_T)
funding_received = (with_f["fund"] * with_f["perp_at_fund"]).sum(axis=0)
pnl_legs = (spot_T - 1.0) - (perp_T - 1.0) + funding_received     # long spot, short perp, + funding
assert np.allclose(pnl_legs, funding_received - spot_T * prem_T, atol=1e-12)
pnl_net = pnl_legs - fees
lab.record("ft_taker_fee", TAKER_FEE)
lab.record("ft_fees", fees)
lab.record("ft_mean", float(pnl_net.mean()))
lab.record("ft_sd", float(pnl_net.std(ddof=1)))
lab.record("ft_p05", float(np.quantile(pnl_net, 0.05)))
lab.record("ft_p95", float(np.quantile(pnl_net, 0.95)))
lab.record("ft_share_loss", float((pnl_net < 0).mean()))
lab.record("ft_funding_mean", float(funding_received.mean()))
lab.record("ft_prem_term_sd", float((spot_T * prem_T).std(ddof=1)))
# the spot leg alone swung by far more than the whole trade — that is what the hedge removes: all that is
# left of the price exposure is the premium term, tiny next to the spot leg
lab.record("ft_spot_leg_sd", float((spot_T - 1.0).std(ddof=1)))
assert (spot_T * prem_T).std(ddof=1) < 0.01 * (spot_T - 1.0).std(ddof=1)

# Capital: spot is paid in full, and the short perp needs margin on the exchange (3× on that leg).
SHORT_LEG_LEV = 3
capital = 1.0 + 1.0 / SHORT_LEG_LEV
lab.record("ft_short_lev", SHORT_LEG_LEV)
lab.record("ft_capital", capital)
lab.record("ft_return_on_capital", float(pnl_net.mean() / capital))

# %% [markdown]
# **The venue is the risk.** A "delta-neutral" basis trade still loses whatever sits on a venue
# that fails. With an illustrative 2% annual chance of failure and nothing recovered, the expected
# return drops and the distribution acquires a loss tail: all the capital if both legs sit on the
# failed venue, only the perp margin if the spot is held elsewhere. Assumptions, not estimates.
# (The carry figure itself assumes the short is topped up and never liquidated, and that funding
# stays as positive as section 3's model makes it.)

# %%
P_FAIL, RECOVERY = 0.02, 0.0
carry = float(pnl_net.mean() / capital)
margin_share = (1.0 / SHORT_LEG_LEV) / capital          # perp margin as a share of the capital: 25% at 3×
lab.record("venue_p_fail", P_FAIL)
lab.record("venue_recovery", RECOVERY)
lab.record("venue_expected", (1 - P_FAIL) * carry - P_FAIL * (1 - RECOVERY))           # both legs on the venue
lab.record("venue_breakeven_p", carry / (carry + 1 - RECOVERY))
lab.record("venue_margin_share", margin_share)
lab.record("venue_expected_margin_only", (1 - P_FAIL) * carry - P_FAIL * margin_share * (1 - RECOVERY))
assert (1 - P_FAIL) * carry - P_FAIL * (1 - RECOVERY) < (1 - P_FAIL) * carry - P_FAIL * margin_share < carry

# %% [markdown]
# ## 5 · Liquidation: how long a leveraged perp position survives
#
# An isolated-margin long with leverage L posts margin M = notional/L. The venue liquidates when
# equity falls to the maintenance margin m × current notional. Solving for the price:
#
#     long:  P_liq = P_0 (1 − 1/L)/(1 − m),   so the fall that liquidates is d* = (1/L − m)/(1 − m)
#     short: P_liq = P_0 (1 + 1/L)/(1 + m)
#
# — Session 2.4's margin-call trigger 1/L − m, give or take, but with no call: the engine closes
# the position. We simulate 20,000 one-year hourly paths (60% volatility, zero drift), check each
# hour, and compare with the barrier-crossing formula of Session 2.4 (reflection principle plus
# the Broadie–Glasserman–Kou correction for discrete checks, here Δt = 1 hour).

# %%
M_RATE = 0.005                              # maintenance margin: 0.5% of notional (illustrative)
LEVELS = [2, 3, 5, 10, 25, 50, 100]
N_LIQ, LIQ_HOURS, CHUNK = 20_000, HOURS_PER_YEAR, 730
SIGMA_LIQ, MU_LIQ = SIGMA_SPOT, 0.0
BGK = 0.5826                                 # Broadie–Glasserman–Kou constant β = −ζ(1/2)/√(2π) (β in 2.1, 2.4)


def liq_fall_long(L: float, m: float = M_RATE) -> float:
    return (1.0 / L - m) / (1.0 - m)


def liq_rise_short(L: float, m: float = M_RATE) -> float:
    return (1.0 + 1.0 / L) / (1.0 + m) - 1.0


def p_touch(h: float, mu: float, sigma: float, tau: float, dt: float) -> float:
    """P(log price crosses level h within tau years), checked every dt years (BGK-corrected).

    h < 0 is a barrier below the start (a long's liquidation), h > 0 one above (a short's).
    """
    nu = mu - 0.5 * sigma ** 2
    s = sigma * np.sqrt(tau)
    if h < 0:
        h = h - BGK * sigma * np.sqrt(dt)     # discrete checks miss some crossings: move the barrier out
        return float(norm.cdf((h - nu * tau) / s) + np.exp(2 * nu * h / sigma ** 2) * norm.cdf((h + nu * tau) / s))
    h = h + BGK * sigma * np.sqrt(dt)
    return float(norm.cdf((-h + nu * tau) / s) + np.exp(2 * nu * h / sigma ** 2) * norm.cdf((-h - nu * tau) / s))


h_long = {L: np.log(1.0 - liq_fall_long(L)) for L in LEVELS}
h_short = {L: np.log(1.0 + liq_rise_short(L)) for L in LEVELS}
hit_long = {L: np.full(N_LIQ, -1) for L in LEVELS}      # hour of liquidation, −1 if never
hit_short = {L: np.full(N_LIQ, -1) for L in LEVELS}
dt_y = 1.0 / HOURS_PER_YEAR
log_p = np.zeros(N_LIQ)
for start in range(0, LIQ_HOURS, CHUNK):
    n = min(CHUNK, LIQ_HOURS - start)
    path = log_p + np.cumsum((MU_LIQ - 0.5 * SIGMA_LIQ ** 2) * dt_y
                             + SIGMA_LIQ * np.sqrt(dt_y) * rng_liq.standard_normal((n, N_LIQ)), axis=0)
    for L in LEVELS:
        for hits, crossed in ((hit_long[L], path <= h_long[L]), (hit_short[L], path >= h_short[L])):
            first = crossed.argmax(axis=0)
            new = crossed.any(axis=0) & (hits < 0)
            hits[new] = start + first[new] + 1
    log_p = path[-1]

liq_rows = []
for L in LEVELS:
    hl, hs = hit_long[L], hit_short[L]
    row = {"L": L, "fall": liq_fall_long(L), "rise": liq_rise_short(L)}
    for d in (1, 7, 30, 91, 365):
        row[f"long_{d}d"] = float(((hl > 0) & (hl <= 24 * d)).mean())
        row[f"short_{d}d"] = float(((hs > 0) & (hs <= 24 * d)).mean())
        row[f"long_{d}d_th"] = p_touch(h_long[L], MU_LIQ, SIGMA_LIQ, d / DAYS_PER_YEAR, dt_y)
        row[f"short_{d}d_th"] = p_touch(h_short[L], MU_LIQ, SIGMA_LIQ, d / DAYS_PER_YEAR, dt_y)
    liq_hours = np.where(hl > 0, hl, np.inf)
    row["median_days_long"] = float(np.median(liq_hours) / 24.0) if np.isfinite(np.median(liq_hours)) else float("nan")
    liq_rows.append(row)
liq = pd.DataFrame(liq_rows).set_index("L")
print(liq[["fall", "long_1d", "long_30d", "long_365d", "long_365d_th", "short_365d", "short_365d_th",
           "median_days_long"]].round(4))

# Monte Carlo vs the formula, within 4 standard errors, wherever the barrier is at least five hourly
# standard deviations away — closer than that, the continuity correction is outside its range of
# validity. The extra 0.002 (0.2 percentage points) allows for the correction's own approximation
# error, which is not zero: pooling 200,000 paths from 10 independent seed streams, the formula's
# gap to the hourly Monte Carlo never exceeded 0.18 percentage points for any leverage up to 25×,
# side or horizon here (at most 1.8 SE at n = 200,000). At this lab's n = 20,000 that bias is about
# half a standard error — small, but enough to push an honest 3.9-SE draw over a 4-SE line.
hourly_sd = SIGMA_LIQ * np.sqrt(dt_y)
max_gap_se = 0.0
for L in LEVELS:
    if abs(h_long[L]) < 5 * hourly_sd:
        continue
    for d in (30, 365):
        for side in ("long", "short"):
            p_mc, p_th = liq.loc[L, f"{side}_{d}d"], liq.loc[L, f"{side}_{d}d_th"]
            se = np.sqrt(max(p_th * (1 - p_th), 1e-6) / N_LIQ)
            max_gap_se = max(max_gap_se, abs(p_mc - p_th) / se)
            assert abs(p_mc - p_th) < 4 * se + 0.002, f"{side} {L}x {d}d: MC {p_mc:.4f} vs theory {p_th:.4f}"
lab.record("liq_max_gap_se", max_gap_se)
assert liq["long_365d"].is_monotonic_increasing and liq["long_30d"].is_monotonic_increasing

lab.record("liq_n_paths", N_LIQ)
lab.record("liq_sigma", SIGMA_LIQ)
lab.record("liq_m", M_RATE)
lab.record("liq_hourly_sd", hourly_sd)
for L, row in liq.iterrows():
    lab.record(f"liq_{L}_fall", row["fall"])
    lab.record(f"liq_{L}_rise", row["rise"])
    for d in (1, 7, 30, 91, 365):
        lab.record(f"liq_{L}_long_{d}d", row[f"long_{d}d"])
        lab.record(f"liq_{L}_short_{d}d", row[f"short_{d}d"])
        lab.record(f"liq_{L}_long_{d}d_th", row[f"long_{d}d_th"])
    if np.isfinite(row["median_days_long"]):
        lab.record(f"liq_{L}_median_days", row["median_days_long"])
        lab.record(f"liq_{L}_median_hours", 24.0 * row["median_days_long"])

# The basis trade's hedge leg: a 3× short perp is liquidated by a rally of about a third.
lab.record("ft_short_leg_rise", liq_rise_short(SHORT_LEG_LEV))
lab.record("ft_short_leg_p_quarter", float(liq.loc[SHORT_LEG_LEV, "short_91d"]))
lab.record("ft_short_leg_p_year", float(liq.loc[SHORT_LEG_LEV, "short_365d"]))

CHART_LEVELS = [(3, "benchmark"), (5, "alt2"), (10, "alt1"), (25, "loss")]
day_grid = np.arange(0, DAYS_PER_YEAR + 1)
series = []
for L, role in CHART_LEVELS:
    hl = hit_long[L]
    hours_sorted = np.sort(hl[hl > 0])
    frac = np.searchsorted(hours_sorted, 24 * day_grid, side="right") / N_LIQ
    series.append(charts.Series(f"{L}× long", pd.Series(frac, index=day_grid), role=role, end_label=f"{L}×"))
lab.chart("liq_curves", charts.line_chart(
    series, title=f"Share of leveraged longs liquidated by day t ({SIGMA_LIQ:.0%} volatility; x = elapsed days)",
    y_fmt=charts.fmt_pct(0), y_min=0.0, y_max=1.0))

# %% [markdown]
# ## 6 · Numbers for the check-yourself questions

# %%
# Q: a 10x long, maintenance 0.5%: how far can the price fall? And the same position at 25x?
lab.record("q_fall_10", liq_fall_long(10))
lab.record("q_fall_25", liq_fall_long(25))
lab.record("q_daily_sd", SIGMA_SPOT / np.sqrt(DAYS_PER_YEAR))    # one day's sd at 60% annual volatility
# Q: premium of 0.10% over an 8-hour interval → funding per interval, and annualised if it persisted
Q_PREM = 0.0010
lab.record("q_prem", Q_PREM)
lab.record("q_funding", float(funding_rate(np.array([Q_PREM]))[0]))
lab.record("q_funding_annual", float(funding_rate(np.array([Q_PREM]))[0]) * N_INTERVALS)
# Q: $10,000 deposited as an LP at $2,000; the price doubles. LP vs holding, no fees.
Q_DEPOSIT, Q_THETA = 10_000.0, 2.0
lab.record("q_lp_deposit", Q_DEPOSIT)
lab.record("q_lp_hold", Q_DEPOSIT * (1 + Q_THETA) / 2)
lab.record("q_lp_value", Q_DEPOSIT * np.sqrt(Q_THETA))
lab.record("q_lp_gap", Q_DEPOSIT * (np.sqrt(Q_THETA) - (1 + Q_THETA) / 2))
assert abs(Q_DEPOSIT * np.sqrt(Q_THETA) / (Q_DEPOSIT * (1 + Q_THETA) / 2) - 1 - lab.results["il_p100"]) < 1e-12

# %%
lab.save()
