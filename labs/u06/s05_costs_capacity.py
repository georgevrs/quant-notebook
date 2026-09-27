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
# # 6.5 · Transaction Costs & Capacity — companion lab
#
# **Quant Notebook** · Unit 6 · Session 5 · [Read the session](https://georgevrs.github.io/quant-notebook/unit06-backtesting-discipline/session05-costs-capacity.html)
#
# Spread, commission, slippage, market impact, borrow and funding — and how costs cap how much capital a strategy can ever run.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Risk-free rate: the strategy below is already a dollar-neutral long-short book, so its return is
# an excess return by construction — no separate risk-free adjustment is needed anywhere in this lab.

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
from scipy.optimize import brentq

import quantnb as qn
from quantnb import charts, synth
from quantnb.returns import (PERIODS_PER_YEAR, annualised_return, annualised_vol, max_drawdown,
                             sharpe_ratio)
from quantnb.stats import annual_sharpe, batch_se, sharpe_se

lab = qn.Lab("6.5")  # seeds the random generators: every run gives the same numbers
rng_book, rng_mc = lab.rng.spawn(2)  # one stream per experiment (AUTHORING.md §12a)

# %% [markdown]
# ## 1 · A vectorized backtester, with turnover — the thing costs are charged on
#
# Session 6.2 owns the vectorized backtester itself; here we build the minimum version needed to
# generate a realistic (return, turnover) pair to charge costs against. 150 synthetic
# names, 20% with a true positive annual alpha, 20% with the same alpha negative, the rest none
# (`quantnb.synth.factor_model_returns` — we know the truth because we planted it). The book holds
# fixed target weights (long the true winners, short the true losers, equal-weighted) and is
# rebalanced back to that target every day — the turnover comes entirely from one day's price
# drift pulling dollar weights away from target, exactly as it would for a real constant-mix book.

# %%
N_ASSETS = 150
N_DAYS = 1260              # 5 simulated years
NQ = 30                    # 20% of names on each side of the book
ALPHA_ANNUAL = 0.05        # true annual alpha, planted symmetrically: +5% / -5%

alpha = np.zeros(N_ASSETS)
alpha[:NQ] = ALPHA_ANNUAL
alpha[-NQ:] = -ALPHA_ANNUAL

universe = synth.factor_model_returns(N_DAYS, N_ASSETS, alphas=alpha, rng=rng_book)
asset_returns = universe["returns"]                    # T x N simple daily returns

target = np.zeros(N_ASSETS)
target[:NQ] = 1.0 / NQ                                  # long the true winners
target[-NQ:] = -1.0 / NQ                                # short the true losers
GROSS_NAV = float(np.abs(target).sum())                 # 2.0: 100% long, 100% short

R = asset_returns.to_numpy()
gross_return = R @ target                                # constant-weight book return each day
# turnover to rebalance back to a FIXED dollar target after one day's drift, as a fraction of NAV
# (one-way: buys + sells counted once, per §6.2's definition of turnover)
turnover = 0.5 * (np.abs(R) * np.abs(target)).sum(axis=1)

gross_return = pd.Series(gross_return, index=asset_returns.index)
turnover = pd.Series(turnover, index=asset_returns.index)

MU_GROSS_ANNUAL = annualised_return(gross_return)
SIGMA_ANNUAL = annualised_vol(gross_return)
GROSS_SR = sharpe_ratio(gross_return, 0.0)
TURNOVER_ANNUAL = float(turnover.mean() * PERIODS_PER_YEAR)   # one-way multiples of NAV per year

print(f"gross: return {MU_GROSS_ANNUAL:.2%}/yr · vol {SIGMA_ANNUAL:.2%}/yr · Sharpe {GROSS_SR:.2f} "
      f"· turnover {TURNOVER_ANNUAL:.2f}x/yr · maxDD {max_drawdown(gross_return):.1%}")
lab.record("gross_sr", GROSS_SR)
lab.record("gross_ret", MU_GROSS_ANNUAL)
lab.record("gross_vol", SIGMA_ANNUAL)
lab.record("turnover_annual", TURNOVER_ANNUAL)
lab.record("n_assets", N_ASSETS)
lab.record("alpha_annual", ALPHA_ANNUAL)
# sanity check only (one realized path, not a Monte Carlo claim): the planted alpha must show up
assert GROSS_SR > 0.4, "the long-short book should recover a healthy chunk of the planted alpha"

# %% [markdown]
# ## 2 · A cost model: commission, spread, market impact and borrow
#
# Four lines on the bill, at every rebalance:
#   * **commission** — a flat fee per dollar traded, however small the order.
#   * **half-spread** — the fee for immediacy (§2.1.4), also flat in this model.
#   * **market impact** — grows with the *square root* of the trade's size relative to the
#     market's daily dollar volume (ADV): a stylized, illustrative form of the model Session 14.3
#     names and derives properly. `IMPACT_COEF` (eta) is a dimensionless constant a real desk
#     would calibrate from its own fills — here it is chosen for a clean illustration, not
#     measured from data.
#   * **borrow cost** — the fee kappa (Session 2.2) on the value of the short leg, charged daily
#     regardless of how much is traded that day.
#
# Commission and half-spread do not depend on capital: they are paid per dollar traded, at a rate
# fixed by the venue and the broker. Market impact does depend on capital, because a bigger book
# trades a bigger dollar amount at the same turnover — that is what eventually caps capacity.

# %%
COMMISSION_BPS = 0.5     # cents-a-share-style flat fee, in bps of notional (illustrative)
HALF_SPREAD_BPS = 2.0    # comparable order of magnitude to the liquid book's half-spread, §2.1.4
IMPACT_COEF = 0.6        # eta: dimensionless impact coefficient (illustrative, not calibrated)
IMPACT_DAILY_VOL = 0.018 # typical single-name daily volatility feeding the impact model
ADV_TOTAL = 80_000_000   # combined average daily dollar volume of this (small-cap) universe
KAPPA = 0.002            # borrow fee: general-collateral-like, see Session 2.2's D'Avolio figure
SHORT_FRAC = 1.0         # fraction of NAV held short: 100% here (a 100/100 long-short book)


def cost_bps(capital: float, turnover_annual: float = TURNOVER_ANNUAL) -> float:
    """Cost per dollar traded, in bps: flat commission + half-spread, plus a square-root impact
    term that grows with the trade's participation in the market's daily volume."""
    turnover_daily = turnover_annual / PERIODS_PER_YEAR
    participation = turnover_daily * capital / ADV_TOTAL
    impact = IMPACT_COEF * IMPACT_DAILY_VOL * 1e4 * np.sqrt(participation)
    return COMMISSION_BPS + HALF_SPREAD_BPS + impact


def annual_drag(capital: float, turnover_annual: float = TURNOVER_ANNUAL) -> float:
    """Total annualised cost, as a fraction of capital: cost-per-dollar-traded x how often the
    book turns over, plus the borrow cost on the short leg (independent of trade size)."""
    return turnover_annual * cost_bps(capital, turnover_annual) * 1e-4 + SHORT_FRAC * KAPPA


def net_sr(capital: float, turnover_annual: float = TURNOVER_ANNUAL) -> float:
    return (MU_GROSS_ANNUAL - annual_drag(capital, turnover_annual)) / SIGMA_ANNUAL


CAP_EXAMPLE = 1_000_000_000.0
lab.record("cap_example", CAP_EXAMPLE)
lab.record("cost_bps_example", cost_bps(CAP_EXAMPLE))
lab.record("commission_drag", TURNOVER_ANNUAL * COMMISSION_BPS * 1e-4)
lab.record("spread_drag", TURNOVER_ANNUAL * HALF_SPREAD_BPS * 1e-4)
lab.record("impact_drag_example", TURNOVER_ANNUAL * (cost_bps(CAP_EXAMPLE) - COMMISSION_BPS - HALF_SPREAD_BPS) * 1e-4)
lab.record("borrow_drag", SHORT_FRAC * KAPPA)
lab.record("total_drag_example", annual_drag(CAP_EXAMPLE))
lab.record("net_sr_example", net_sr(CAP_EXAMPLE))
print(f"at ${CAP_EXAMPLE:,.0f}: cost {cost_bps(CAP_EXAMPLE):.1f} bp/trade, "
      f"drag {annual_drag(CAP_EXAMPLE):.2%}/yr, net Sharpe {net_sr(CAP_EXAMPLE):.2f}")

# %% [markdown]
# ## 3 · Break-even cost and break-even capital — capacity in closed form
#
# **Break-even cost**: the cost per dollar traded at which this strategy's edge is entirely
# consumed, whatever the capital. **Break-even capital** — the practical definition of
# **capacity** — is the assumed capital at which the square-root impact model actually reaches
# that break-even cost. Because impact scales as `A + B * sqrt(K)` in the capital `K`, break-even
# capital has a closed form; we solve it twice, once analytically and once by root-finding on the
# actual cost function, and check they agree.

# %%
breakeven_cost_bps = (MU_GROSS_ANNUAL - SHORT_FRAC * KAPPA) * 1e4 / TURNOVER_ANNUAL
lab.record("breakeven_cost_bps", breakeven_cost_bps)

C0 = COMMISSION_BPS + HALF_SPREAD_BPS
D0 = TURNOVER_ANNUAL * 1e-4 * C0 + SHORT_FRAC * KAPPA
turnover_daily = TURNOVER_ANNUAL / PERIODS_PER_YEAR
B = IMPACT_COEF * IMPACT_DAILY_VOL * 1e4 * np.sqrt(turnover_daily / ADV_TOTAL)
D1 = TURNOVER_ANNUAL * 1e-4 * B
sqrt_k_star = (MU_GROSS_ANNUAL - D0) / D1
k_star_closed_form = sqrt_k_star ** 2

k_star_numeric = brentq(net_sr, 1e6, 1e13)

print(f"break-even cost: {breakeven_cost_bps:.1f} bp/trade")
print(f"break-even capital: closed form ${k_star_closed_form:,.0f} · root-find ${k_star_numeric:,.0f}")
assert abs(k_star_closed_form / k_star_numeric - 1.0) < 1e-6, "the two ways of finding K* must agree"
lab.record("k_star", k_star_numeric)
lab.record("k_star_billion", k_star_numeric / 1e9)
lab.record("d0", D0)
lab.record("d1", D1)

# %% [markdown]
# ## 4 · Sharpe ratio versus assumed capital
#
# Hold the strategy fixed — same signal, same turnover — and only change how much capital trades
# it. Cost grows with the square root of capital; the edge does not grow at all. Net Sharpe falls
# smoothly to zero at the break-even capital, then turns negative.

# %%
CAPITAL_GRID = [1e8, 3e8, 1e9, 3e9, 1e10, 2e10, 3e10]
sr_by_capital = [net_sr(k) for k in CAPITAL_GRID]
for k, sr in zip(CAPITAL_GRID, sr_by_capital):
    print(f"  ${k:>14,.0f}  cost {cost_bps(k):5.1f} bp  net Sharpe {sr:6.3f}")
    lab.record(f"sr_at_{k:.0f}", sr)
    lab.record(f"cost_bps_at_{k:.0f}", cost_bps(k))

lab.chart("sharpe_by_capital", charts.column_chart(
    [charts.fmt_usd_compact(0)(k) for k in CAPITAL_GRID], sr_by_capital,
    title="Net Sharpe ratio by assumed capital — same strategy, more money behind it",
    y_fmt=charts.fmt_num(2)))

lab.chart("cost_breakdown", charts.column_chart(
    ["Commission", "Half-spread", "Market impact", "Borrow cost"],
    [TURNOVER_ANNUAL * COMMISSION_BPS * 1e-4, TURNOVER_ANNUAL * HALF_SPREAD_BPS * 1e-4,
     TURNOVER_ANNUAL * (cost_bps(CAP_EXAMPLE) - COMMISSION_BPS - HALF_SPREAD_BPS) * 1e-4,
     SHORT_FRAC * KAPPA],
    title=f"Annual cost drag by component at {charts.fmt_usd_compact(0)(CAP_EXAMPLE)} of assumed capital",
    y_fmt=charts.fmt_pct(2)))

# %% [markdown]
# ## 5 · Naive vs disciplined: the flat-cost mistake
#
# A common bug: measure costs from a small pilot allocation, then assume the same bps rate holds
# at any size. Market impact makes that wrong by construction — cost per dollar traded rises with
# size, so a flat rate calibrated small always understates cost big.

# %%
K_PILOT = 20_000_000.0
K_BIG = 15_000_000_000.0
naive_flat_bps = cost_bps(K_PILOT)                      # calibrated once, at a small size
naive_drag_big = TURNOVER_ANNUAL * naive_flat_bps * 1e-4 + SHORT_FRAC * KAPPA
naive_sr_big = (MU_GROSS_ANNUAL - naive_drag_big) / SIGMA_ANNUAL
disciplined_sr_big = net_sr(K_BIG)

lab.record("naive_flat_bps", naive_flat_bps)
lab.record("naive_sr_big", naive_sr_big)
lab.record("disciplined_sr_big", disciplined_sr_big)
lab.record("k_big", K_BIG)
print(f"at ${K_BIG:,.0f}: naive Sharpe {naive_sr_big:.2f} (costs frozen at the "
      f"${K_PILOT:,.0f} pilot rate) vs disciplined Sharpe {disciplined_sr_big:.2f}")
assert naive_sr_big - disciplined_sr_big > 0.3, "the naive flat-cost model should look far too optimistic at scale"

# %% [markdown]
# ## 6 · Monte Carlo: does the formula predict the simulation?
#
# `net_sr(K)` is a deterministic formula built on the STRATEGY's average turnover. To check it
# against something noisy, simulate many independent years of daily excess returns with the
# measured gross mean and volatility, charge the same deterministic cost drag, and see whether the
# average realized Sharpe across paths lands on the formula — at 4 standard errors, never 3, and
# never by hunting for a seed that passes (AUTHORING.md §12a).

# %%
SIM_YEARS = 10
N_PATHS = 4000
n_days_mc = PERIODS_PER_YEAR * SIM_YEARS
mu_daily = MU_GROSS_ANNUAL / PERIODS_PER_YEAR
sigma_daily = SIGMA_ANNUAL / np.sqrt(PERIODS_PER_YEAR)
gross_paths = mu_daily + sigma_daily * rng_mc.standard_normal((n_days_mc, N_PATHS))

TEST_CAPITALS = {"low": 5e8, "breakeven": k_star_numeric}
for label, k in TEST_CAPITALS.items():
    drag_daily = annual_drag(k) / PERIODS_PER_YEAR
    net_paths = gross_paths - drag_daily
    sim_sr = annual_sharpe(net_paths, axis=0)               # one Sharpe per path
    mean_sim_sr = float(sim_sr.mean())
    theory_sr = net_sr(k)
    se_of_mean = sharpe_se(theory_sr, SIM_YEARS) / np.sqrt(N_PATHS)
    gap_se = abs(mean_sim_sr - theory_sr) / se_of_mean
    print(f"{label:>10} (${k:,.0f}): theory {theory_sr:.3f} · simulated mean {mean_sim_sr:.3f} "
          f"· gap {gap_se:.2f} SE")
    lab.record(f"mc_{label}_theory_sr", theory_sr)
    lab.record(f"mc_{label}_sim_sr", mean_sim_sr)
    lab.record(f"mc_{label}_gap_se", gap_se)
    assert gap_se < 4.0, f"{label}: simulated Sharpe strayed more than 4 SE from the formula"
    if label == "breakeven":
        lab.chart("net_sr_breakeven_hist", charts.histogram(
            sim_sr, title=f"Realized {SIM_YEARS}-year net Sharpe at the break-even capital — {N_PATHS:,} simulated paths",
            x_fmt=charts.fmt_num(2)))
        lab.record("mc_breakeven_sd", float(sim_sr.std(ddof=1)))
        lab.record("mc_breakeven_p_positive", float((sim_sr > 0).mean()))

# a second Monte Carlo check via independent batching, at a capital comfortably below break-even
batch_gap = abs(batch_se(lambda b: annual_sharpe(b, axis=0).mean(),
                         gross_paths - annual_drag(TEST_CAPITALS["low"]) / PERIODS_PER_YEAR, n_batches=50))
lab.record("mc_low_batch_se", batch_gap)

# %%
lab.save()
