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
# # 3.6 · Bet Sizing, Kelly & Ruin — companion lab
#
# **Quant Notebook** · Unit 3 · Session 6 · [Read the session](https://georgevrs.github.io/quant-notebook/unit03-math-toolkit/session06-kelly-bet-sizing.html)
#
# Why the size of a bet matters as much as its edge — Kelly growth, the cost of over-betting, fractional Kelly and risk of ruin.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Every Monte Carlo result is checked against its formula (or bound) at **4 standard errors**
# (Session 3.5), wherever enough events occur for the normal approximation to hold.
# Rates: the betting sections (1, 2, 6) have no cash; the investing sections (3, 4) use the asset of
# Session 2.3 — μ = 8%, σ = 20%, risk-free rate 3% a year; section 5 measures wealth relative to
# cash (the same asset's 5% excess return, risk-free rate netted out).

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
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR
from quantnb.stats import p_touch

lab = qn.Lab("3.6")  # seeds the random generators: every run gives the same numbers
# one independent random stream per experiment, so editing one section never changes another
rng_bin, rng_cont, rng_est, rng_dd, rng_ruin = lab.rng.spawn(5)

# %% [markdown]
# ## 1 · The Kelly bet for a binary wager
#
# You win `b` times your stake with probability `p` and lose the stake with probability `q = 1 − p`.
# Bet a fraction `f` of your wealth each time. One bet multiplies wealth by `1 + b·f` or `1 − f`, so
# the **growth rate** — the expected log growth per bet — is
#
#     g(f) = p·ln(1 + b·f) + q·ln(1 − f)
#
# Setting g'(f) = 0 gives the Kelly fraction f* = p − q/b. We check g(f) against 10,000 simulated
# bettors, each making 1,000 bets, at several fractions (the same coin flips for every fraction).

# %%
P_WIN, ODDS = 0.55, 1.0            # a 55/45 even-money bet: +10% expected value per $1 staked
N_BETTORS, N_BETS = 10_000, 1_000
FRACTIONS = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]


def growth_rate_binary(f, p: float, b: float):
    """Expected log growth per bet when staking a fraction f of wealth on a p-to-win, b-to-1 bet."""
    return p * np.log1p(b * f) + (1 - p) * np.log1p(-f)


def kelly_binary(p: float, b: float) -> float:
    """Kelly fraction for a bet paying b-to-1 with win probability p: f* = p − q/b."""
    return p - (1 - p) / b


f_star = kelly_binary(P_WIN, ODDS)
g_star = growth_rate_binary(f_star, P_WIN, ODDS)
f_zero = brentq(lambda f: growth_rate_binary(f, P_WIN, ODDS), 1.01 * f_star, 0.99)  # growth back to zero
ev_per_dollar = P_WIN * ODDS - (1 - P_WIN)
print(f"f* = {f_star:.3f} · g* = {g_star:.5f} per bet · zero growth again at f = {f_zero:.4f}")
lab.record("p_win", P_WIN)
lab.record("q_lose", 1 - P_WIN)
lab.record("odds", ODDS)
lab.record("ev_per_dollar", ev_per_dollar)
lab.record("f_star", f_star)
lab.record("g_star", float(g_star))
lab.record("g_star_multiple", float(np.exp(N_BETS * g_star)))    # typical wealth multiple over 1,000 bets
lab.record("f_zero", f_zero)
lab.record("g_2f", float(growth_rate_binary(2 * f_star, P_WIN, ODDS)))
lab.record("share_half", float(growth_rate_binary(0.5 * f_star, P_WIN, ODDS) / g_star))
lab.record("share_1_5", float(growth_rate_binary(1.5 * f_star, P_WIN, ODDS) / g_star))
lab.record("n_bettors", N_BETTORS)
lab.record("n_bets", N_BETS)
# the derivative really is zero at f*, and g is lower on either side
assert abs((growth_rate_binary(f_star + 1e-6, P_WIN, ODDS) - growth_rate_binary(f_star - 1e-6, P_WIN, ODDS)) / 2e-6) < 1e-6
assert growth_rate_binary(0.9 * f_star, P_WIN, ODDS) < g_star > growth_rate_binary(1.1 * f_star, P_WIN, ODDS)

# The same derivation for any two-outcome bet: win a fraction `up` or lose a fraction `down` of the
# stake gives f* = p/down − q/up. Session 3.1's coin (+50% / −40%, a fair coin) staked in full
# shrinks the typical player; staked at its Kelly fraction it grows.
COIN_P, COIN_UP, COIN_DOWN = 0.5, 0.50, 0.40
coin_kelly = COIN_P / COIN_DOWN - (1 - COIN_P) / COIN_UP


def coin_g(f):
    """Growth rate per round of 3.1's coin when staking a fraction f of wealth."""
    return COIN_P * np.log1p(COIN_UP * f) + (1 - COIN_P) * np.log1p(-COIN_DOWN * f)


coin_g_kelly = coin_g(coin_kelly)
lab.record("coin_up", COIN_UP)
lab.record("coin_down", -COIN_DOWN)   # signed, so the page's "+.0%" spec renders it as "−40%" unaided
lab.record("coin_kelly", coin_kelly)
lab.record("coin_g_all_in", float(coin_g(1.0)))
lab.record("coin_typical_all_in", float(np.expm1(coin_g(1.0))))   # typical change per round, all in
lab.record("coin_g_kelly", float(coin_g_kelly))
lab.record("coin_typical_kelly", float(np.expm1(coin_g_kelly)))   # typical growth per round at Kelly
assert np.isclose(coin_kelly, 0.25)
assert coin_g(1.0) < 0 < coin_g_kelly
assert coin_g(0.9 * coin_kelly) < coin_g_kelly > coin_g(1.1 * coin_kelly)

# %%
wins = rng_bin.random((N_BETS, N_BETTORS)) < P_WIN     # one row per bet, one column per bettor
rows = []
for f in FRACTIONS:
    log_step = np.where(wins, np.log1p(ODDS * f), np.log1p(-f))
    log_wealth = log_step.cumsum(axis=0)
    growth = log_wealth[-1] / N_BETS                      # each bettor's average log growth per bet
    theory = growth_rate_binary(f, P_WIN, ODDS)
    se = growth.std(ddof=1) / np.sqrt(N_BETTORS)
    assert abs(growth.mean() - theory) < 4 * se, f"f={f}: simulated growth rate off by more than 4 SE"
    rows.append({"f": f, "c": f / f_star, "g_theory": theory, "g_sim": growth.mean(), "g_se": se,
                 "median_wealth": float(np.exp(np.median(log_wealth[-1]))),
                 "p_loss": float((log_wealth[-1] < 0).mean()),
                 "p_halve": float((log_wealth.min(axis=0) <= np.log(0.5)).mean()),
                 "p_lose90": float((log_wealth.min(axis=0) <= np.log(0.1)).mean())})
binary = pd.DataFrame(rows).set_index("f")
print(binary.round(4))
for f, row in binary.iterrows():
    key = f"{f * 1000:03.0f}"                              # 0.025 → "025", 0.10 → "100"
    for col in ("c", "g_theory", "g_sim", "g_se", "median_wealth", "p_loss", "p_halve", "p_lose90"):
        lab.record(f"bin_{key}_{col}", float(row[col]))

# %% [markdown]
# **The growth curve.** Growth rises to its peak at f*, falls back to roughly zero near 2f*, and is
# negative beyond: a positive-edge bet, sized too big, shrinks almost every bettor's wealth.

# %%
f_grid = np.linspace(0.0, 0.25, 101)
lab.chart("growth_curve", charts.line_chart(
    [charts.Series("growth per bet", pd.Series(growth_rate_binary(f_grid, P_WIN, ODDS), index=f_grid),
                   role="strategy")],
    title="Growth rate per bet vs fraction of wealth staked (55/45 even-money bet)",
    y_fmt=charts.fmt_pct(2), hline=0.0, height=280))

# %% [markdown]
# ## 2 · Over-betting costs more than under-betting
#
# In the continuous limit (many small bets, or a continuously rebalanced position) growth is a
# parabola in the Kelly multiple c = f/f*: g(c·f*) = (2c − c²)·g*. Growth is symmetric around
# c = 1 — but risk is not. Volatility grows in proportion to c, and the probability that wealth ever
# falls to a fraction x of today's level is x^(2/c − 1) (Thorp 2006), so c = 1.5 has the growth of
# c = 0.5 with three times the volatility and more than six times the chance of ever halving.
#
# For a discrete bet the same exponent appears as the root R of E[(1 + f·X)^(−R)] = 1: wealth^(−R)
# is then a martingale, so P(ever reaching x of today's wealth) ≤ x^R. At full Kelly R = 1 exactly
# for ANY bet, because 1/(1 + f*·X) averages to 1 — the first-order condition in disguise.

# %%
C_GRID = [0.25, 0.5, 1.0, 1.5, 2.0]


def lundberg_exponent(f: float, p: float, b: float) -> float:
    """R > 0 solving p·(1 + b·f)^(−R) + q·(1 − f)^(−R) = 1 (exists when the growth rate is positive)."""
    return brentq(lambda R: p * (1 + b * f) ** (-R) + (1 - p) * (1 - f) ** (-R) - 1, 1e-9, 200)


for c in C_GRID:
    key = f"{c * 100:03.0f}"                               # 0.5 → "050"
    share = 2 * c - c ** 2
    p_ever_half = 0.5 ** (2 / c - 1)
    lab.record(f"asym_{key}_share", share)
    lab.record(f"asym_{key}_vol", c)
    lab.record(f"asym_{key}_p_ever_half", p_ever_half)
    if c < 2:
        R = lundberg_exponent(c * f_star, P_WIN, ODDS)
        lab.record(f"asym_{key}_R", R)
        lab.record(f"asym_{key}_bound_half", 0.5 ** R)
        print(f"c = {c:4}: growth {share:.0%} of max · volatility {c:.2f}× · P(ever halve) {p_ever_half:.1%} "
              f"· discrete bet R = {R:.3f} (continuous 2/c − 1 = {2 / c - 1:.3f})")
        # the simulated halving frequency (within 1,000 bets) respects the martingale bound
        match = np.isclose(binary.index, c * f_star)
        if match.any():
            ph = binary.loc[match, "p_halve"].iloc[0]
            assert ph <= 0.5 ** R + 4 * np.sqrt(0.5 ** R * (1 - 0.5 ** R) / N_BETTORS)
            lab.record(f"asym_{key}_sim_half", float(ph))
assert np.isclose(lundberg_exponent(f_star, P_WIN, ODDS), 1.0)   # R = 1 at full Kelly, exactly
assert abs(lundberg_exponent(0.5 * f_star, P_WIN, ODDS) - 3.0) < 0.05   # ≈ 2/c − 1 for smallish bets

# %% [markdown]
# ## 3 · Continuous Kelly is the growth-optimal leverage of Session 2.3
#
# Hold a fraction f of wealth in an asset with drift μ and volatility σ, the rest in cash at r_f,
# rebalanced continuously. The growth rate is g(f) = r_f + f(μ − r_f) − f²σ²/2, maximised at
# f* = (μ − r_f)/σ² — exactly 2.3's L*. With θ = (μ − r_f)/σ, the Sharpe ratio, full Kelly earns
# θ²/2 a year above cash. We check the formula against daily-rebalanced simulations.

# %%
MU, SIGMA, RF = 0.08, 0.20, 0.03
N_PATHS_CONT, YEARS_CONT = 4_000, 10
dt = 1 / PERIODS_PER_YEAR


def growth_rate_cont(f, mu: float = MU, sigma: float = SIGMA, rf: float = RF):
    """Growth rate (annual expected log return) of a continuously rebalanced fraction f in the asset."""
    return rf + f * (mu - rf) - 0.5 * f ** 2 * sigma ** 2


kelly_cont = (MU - RF) / SIGMA ** 2
theta = (MU - RF) / SIGMA
lab.record("mu", MU)
lab.record("sigma", SIGMA)
lab.record("rf", RF)
lab.record("excess", MU - RF)
lab.record("theta", theta)
lab.record("kelly_cont", kelly_cont)
lab.record("g_cont_star", growth_rate_cont(kelly_cont))
lab.record("g_cont_star_excess", growth_rate_cont(kelly_cont) - RF)
lab.record("cagr_cont_star", float(np.expm1(growth_rate_cont(kelly_cont))))
lab.record("g_cont_half_excess", growth_rate_cont(0.5 * kelly_cont) - RF)
lab.record("vol_full", kelly_cont * SIGMA)
lab.record("vol_half", 0.5 * kelly_cont * SIGMA)
lab.record("n_paths_cont", N_PATHS_CONT)
lab.record("years_cont", YEARS_CONT)
assert np.isclose(growth_rate_cont(kelly_cont) - RF, theta ** 2 / 2)
assert np.isclose(growth_rate_cont(2 * kelly_cont), RF)          # twice Kelly: you grow like cash

n_days = PERIODS_PER_YEAR * YEARS_CONT
z = rng_cont.standard_normal((n_days, N_PATHS_CONT))
asset = np.expm1((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * np.sqrt(dt) * z)   # daily simple returns (GBM)
for c in (0.5, 1.0, 1.5, 2.0):
    f = c * kelly_cont
    port = f * asset + (1 - f) * RF * dt                  # daily-rebalanced: f in the asset, the rest in cash
    growth = np.log1p(port).sum(axis=0) / YEARS_CONT      # each path's annual log growth
    se = growth.std(ddof=1) / np.sqrt(N_PATHS_CONT)
    theory = growth_rate_cont(f)
    assert abs(growth.mean() - theory) < 4 * se, f"c={c}: daily-rebalanced growth off by more than 4 SE"
    key = f"{c * 100:03.0f}"
    lab.record(f"cont_{key}_f", f)
    lab.record(f"cont_{key}_theory", theory)
    lab.record(f"cont_{key}_sim", float(growth.mean()))
    lab.record(f"cont_{key}_se", float(se))
    lab.record(f"cont_{key}_vol", f * SIGMA)
    print(f"c = {c}: f = {f:.3f} · growth sim {growth.mean():.4f} ± {se:.4f} vs theory {theory:.4f}")
del z, asset

# %% [markdown]
# ## 4 · You never know your edge: Kelly on an estimated mean
#
# 20,000 researchers each see 5 years of daily prices from the same asset (by default 2.3's:
# true μ = 8%, σ = 20%), estimate μ from them, and size at full Kelly or half Kelly using that
# estimate. Each then trades for the next 10 years. σ is taken as known: after 5 years of daily data
# its standard error is about 0.4 percentage points, against about 9 for μ.
#
# Theory: the estimate f̂ = (μ̂ − r_f)/σ² has variance 1/(σ²Y), so the expected growth of sizing at
# k·f̂ is r_f + kθ² − (k²/2)(θ² + 1/Y), with θ the Sharpe ratio. Full Kelly on an estimate gives up
# 1/(2Y) a year — ten percentage points with five years of data — and the best multiplier is
# k* = Yθ²/(1 + Yθ²).

# %%
ASSET_MU, ASSET_SIGMA = MU, SIGMA          # the asset the researchers study (homework: try 0.13, 0.10)
N_RESEARCHERS, YEARS_EST, YEARS_TRADE = 20_000, 5, 10
CHUNK = 2_500                              # researchers simulated at a time, to keep memory small
theta_a = (ASSET_MU - RF) / ASSET_SIGMA    # the asset's Sharpe ratio
kelly_a = (ASSET_MU - RF) / ASSET_SIGMA ** 2


def estimate_mu(n_researchers: int, years: int, rng) -> np.ndarray:
    """Each researcher's annualised mean of daily simple returns from `years` of simulated prices."""
    n = PERIODS_PER_YEAR * years
    out = []
    for start in range(0, n_researchers, CHUNK):
        zz = rng.standard_normal((n, min(CHUNK, n_researchers - start)))
        daily = np.expm1((ASSET_MU - 0.5 * ASSET_SIGMA ** 2) * dt + ASSET_SIGMA * np.sqrt(dt) * zz)
        out.append(daily.mean(axis=0) * PERIODS_PER_YEAR)
    return np.concatenate(out)


def realised_growth(f: np.ndarray, asset_log_return: np.ndarray, years: float) -> np.ndarray:
    """Annual log growth of a continuously rebalanced fraction f, given the asset's log return.

    Over τ years, Itô's lemma gives ln(W_τ/W_0) = f·ln(P_τ/P_0) + (1 − f)·r_f·τ + f(1 − f)·σ²τ/2 — exact
    under GBM, so only each future's total log return is needed.
    """
    return (f * asset_log_return + (1 - f) * RF * years + f * (1 - f) * ASSET_SIGMA ** 2 * years / 2) / years


mu_hat = estimate_mu(N_RESEARCHERS, YEARS_EST, rng_est)
f_hat = (mu_hat - RF) / ASSET_SIGMA ** 2
# each researcher's next 10 years: the asset's total log return is normal under GBM
future = rng_est.normal((ASSET_MU - 0.5 * ASSET_SIGMA ** 2) * YEARS_TRADE, ASSET_SIGMA * np.sqrt(YEARS_TRADE),
                        N_RESEARCHERS)

mu_se_theory = ASSET_SIGMA / np.sqrt(YEARS_EST)
sd_mu_hat = mu_hat.std(ddof=1)
assert abs(mu_hat.mean() - ASSET_MU) < 4 * mu_se_theory / np.sqrt(N_RESEARCHERS)
assert abs(sd_mu_hat - mu_se_theory) < 4 * mu_se_theory / np.sqrt(2 * N_RESEARCHERS)
lab.record("n_researchers", N_RESEARCHERS)
lab.record("years_est", YEARS_EST)
lab.record("years_trade", YEARS_TRADE)
lab.record("mu_se", mu_se_theory)
lab.record("sigma_se", ASSET_SIGMA / np.sqrt(2 * PERIODS_PER_YEAR * YEARS_EST))
lab.record("mu_hat_p05", float(np.quantile(mu_hat, 0.05)))
lab.record("mu_hat_p95", float(np.quantile(mu_hat, 0.95)))
lab.record("f_hat_sd", 1 / (ASSET_SIGMA * np.sqrt(YEARS_EST)))
lab.record("f_hat_p05", float(np.quantile(f_hat, 0.05)))
lab.record("f_hat_p95", float(np.quantile(f_hat, 0.95)))
lab.record("share_short", float((f_hat < 0).mean()))
lab.record("share_over3", float((f_hat > 3).mean()))

k_star = YEARS_EST * theta_a ** 2 / (1 + YEARS_EST * theta_a ** 2)
lab.record("k_star", k_star)
lab.record("t_edge", theta_a * np.sqrt(YEARS_EST))
growth_by_k = {}
# "known": the benchmark researcher who is TOLD the true μ and sizes at the true Kelly fraction
SIZINGS = {"known": (1.0, np.full(N_RESEARCHERS, kelly_a)),
           "full": (1.0, f_hat), "half": (0.5, f_hat), "kstar": (k_star, f_hat)}
for name, (k, f_used) in SIZINGS.items():
    G = realised_growth(k * f_used, future, YEARS_TRADE)
    if name == "known":
        theory = RF + theta_a ** 2 / 2
    else:
        theory = RF + k * theta_a ** 2 - 0.5 * k ** 2 * (theta_a ** 2 + 1 / YEARS_EST)
    se = G.std(ddof=1) / np.sqrt(N_RESEARCHERS)
    assert abs(G.mean() - theory) < 4 * se, f"{name}: mean realised growth off by more than 4 SE"
    cagr = np.expm1(G)
    growth_by_k[name] = cagr
    lab.record(f"est_{name}_k", k)
    lab.record(f"est_{name}_g_theory", theory)
    lab.record(f"est_{name}_g_theory_excess", theory - RF)
    lab.record(f"est_{name}_g_sim", float(G.mean()))
    lab.record(f"est_{name}_g_se", float(se))
    lab.record(f"est_{name}_median_cagr", float(np.median(cagr)))
    lab.record(f"est_{name}_p_loss", float((G < 0).mean()))
    lab.record(f"est_{name}_p_below_cash", float((G < RF).mean()))
    lab.record(f"est_{name}_p05_cagr", float(np.quantile(cagr, 0.05)))
    print(f"{name:5} k = {k:.3f}: mean growth {G.mean():.4f} ± {se:.4f} (theory {theory:.4f}) · "
          f"median CAGR {np.median(cagr):.2%} · P(lose money) {(G < 0).mean():.1%} · P(worse than cash) {(G < RF).mean():.1%}")
lab.record("known_g_excess", theta_a ** 2 / 2)
lab.record("est_penalty_full", 1 / (2 * YEARS_EST))
lab.record("est_penalty_full_pp", 100 / (2 * YEARS_EST))      # the same, in percentage points

# Kan & Zhou (2007) scaling of the plug-in mean-variance rule, with μ AND Σ estimated from T periods of
# N assets: c* = (T−N−1)(T−N−4)/(T(T−2)) · θ²/(θ² + N/T), θ² the squared per-period Sharpe ratio.
# For one asset and 5 years of daily data it is our k* to within 1%.
T_DAYS, N_ASSETS = PERIODS_PER_YEAR * YEARS_EST, 1
theta2_daily = theta_a ** 2 / PERIODS_PER_YEAR
kz_c = ((T_DAYS - N_ASSETS - 1) * (T_DAYS - N_ASSETS - 4) / (T_DAYS * (T_DAYS - 2))
        * theta2_daily / (theta2_daily + N_ASSETS / T_DAYS))
lab.record("kz_c", kz_c)
assert abs(kz_c / k_star - 1) < 0.01

# how much of Kelly to bet, by Sharpe ratio and years of data: k* = Yθ²/(1 + Yθ²)
for sr in (0.25, 0.5, 1.0):
    for yrs in (5, 10, 20):
        lab.record(f"kstar_sr{sr * 100:03.0f}_y{yrs}", yrs * sr ** 2 / (1 + yrs * sr ** 2))

# %%
EDGES = np.linspace(-1.0, 0.8, 73)                         # shared bins, so the two histograms compare
for name, label in (("full", "full Kelly"), ("half", "half Kelly")):
    lab.chart(f"est_{name}", charts.histogram(
        np.clip(growth_by_k[name], EDGES[0], EDGES[-1] - 1e-9), bins=EDGES,
        title=f"10-year CAGR of 20,000 researchers sizing at {label} on a 5-year estimate",
        x_fmt=charts.fmt_pct(0), tail_below=0.0, tail_label="lost money", height=250))
lab.record("est_full_share_clipped", float((growth_by_k["full"] >= EDGES[-1]).mean()))

# %% [markdown]
# ## 5 · How likely is a 50% loss? Kelly fractions and the barrier formula
#
# Wealth relative to cash, continuously rebalanced at c times Kelly, is a geometric Brownian motion
# with drift c·f*·(μ − r_f) and volatility c·f*·σ. Session 3.5's barrier formula (`p_touch`, with
# the discrete-monitoring correction for daily checks) gives the probability of touching half the
# starting wealth within 10 years. With no time limit the answer is 0.5^(2/c − 1): one in two at
# full Kelly, one in eight at half Kelly, whatever the Sharpe ratio. A 50% fall from the running
# PEAK (a drawdown) is more likely still — we simulate that too.

# %%
C_DD = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
N_PATHS_DD, YEARS_DD = 20_000, 10
FLOOR = 0.5                                                # "lose half": wealth falls to 50% of its start
EXCESS = MU - RF
n_days = PERIODS_PER_YEAR * YEARS_DD
t_grid = np.arange(1, n_days + 1) * dt
hits = dict.fromkeys(C_DD, 0)
dd_hits = dict.fromkeys(C_DD, 0)
for _ in range(10):                                        # 10 chunks of 2,000 paths keep memory small
    brownian = np.sqrt(dt) * rng_dd.standard_normal((n_days, N_PATHS_DD // 10)).cumsum(axis=0)
    for c in C_DD:
        f = c * kelly_cont
        log_w = (f * EXCESS - 0.5 * f ** 2 * SIGMA ** 2) * t_grid[:, None] + f * SIGMA * brownian
        hits[c] += int((log_w.min(axis=0) <= np.log(FLOOR)).sum())
        log_w0 = np.vstack([np.zeros((1, log_w.shape[1])), log_w])
        worst_dd = (log_w0 - np.maximum.accumulate(log_w0, axis=0)).min(axis=0)
        dd_hits[c] += int((worst_dd <= np.log(FLOOR)).sum())
del brownian, log_w, log_w0

for c in C_DD:
    f = c * kelly_cont
    sim = hits[c] / N_PATHS_DD
    formula = p_touch(FLOOR, f * EXCESS, f * SIGMA, YEARS_DD, dt=dt)
    se = np.sqrt(formula * (1 - formula) / N_PATHS_DD)
    if formula * N_PATHS_DD >= 100:                       # enough expected hits for the normal approximation
        assert abs(sim - formula) < 4 * se, f"c={c}: simulated halving probability off by more than 4 SE"
    key = f"{c * 100:03.0f}"
    lab.record(f"dd_{key}_sim", sim)
    lab.record(f"dd_{key}_formula", formula)
    lab.record(f"dd_{key}_se", float(se))
    lab.record(f"dd_{key}_ever", FLOOR ** (2 / c - 1) if c < 2 else 1.0)
    lab.record(f"dd_{key}_peak50", dd_hits[c] / N_PATHS_DD)
    assert dd_hits[c] >= hits[c]                          # falling below the START's floor implies that drawdown
    print(f"c = {c:4}: P(touch {FLOOR:.0%} within {YEARS_DD}y) sim {sim:.4f} vs formula {formula:.4f} · "
          f"ever {FLOOR ** (2 / c - 1) if c < 2 else 1:.3f} · P(same fall from a peak) {dd_hits[c] / N_PATHS_DD:.3f}")
lab.record("n_paths_dd", N_PATHS_DD)
lab.record("years_dd", YEARS_DD)


def kelly_cap(p_max: float, floor: float) -> float:
    """Largest Kelly multiple c with P(ever falling to `floor` of today's wealth) ≤ p_max: floor^(2/c − 1) ≤ p_max."""
    return 2 / (1 + np.log(p_max) / np.log(floor))


lab.record("cap_10_half", kelly_cap(0.10, 0.5))
lab.record("cap_05_half", kelly_cap(0.05, 0.5))
lab.record("cap_20_80", kelly_cap(0.20, 0.8))
# the worked example of the page: shrink by k*, then cap so P(ever halving) <= 10%; size = the smaller
lab.record("worked_shrunk_f", k_star * kelly_cont)
lab.record("worked_cap_f", kelly_cap(0.10, 0.5) * kelly_cont)
lab.record("worked_size", min(k_star, kelly_cap(0.10, 0.5)) * kelly_cont)
assert np.isclose(0.5 ** (2 / kelly_cap(0.10, 0.5) - 1), 0.10)

c_grid = np.linspace(0.25, 2.0, 71)
ever = pd.Series(FLOOR ** (2 / c_grid - 1), index=c_grid)
within = pd.Series([p_touch(FLOOR, c * kelly_cont * EXCESS, c * kelly_cont * SIGMA, YEARS_DD, dt=dt) for c in c_grid],
                   index=c_grid)
peak = pd.Series([dd_hits[c] / N_PATHS_DD for c in C_DD], index=C_DD)
lab.chart("halving", charts.line_chart(
    [charts.Series("ever, no time limit", ever, role="strategy", end_label="ever"),
     charts.Series(f"within {YEARS_DD} years", within, role="alt1", end_label=f"within {YEARS_DD} y"),
     charts.Series(f"from a peak, within {YEARS_DD} years (simulated)", peak, role="loss", end_label="from a peak")],
    title=f"Chance of losing {1 - FLOOR:.0%} relative to cash, by multiple of Kelly (1 = full Kelly)",
    y_fmt=charts.fmt_pct(0), height=300))

# %% [markdown]
# ## 6 · Risk of ruin: fixed-size bets vs proportional bets
#
# The same 55/45 even-money bet, a $100 bankroll, play until the bankroll reaches $1,000 or is gone.
# A player who always stakes a FIXED amount can be wiped out: with n bets' worth of bankroll the
# probability is ((q/p)^n − (q/p)^N)/(1 − (q/p)^N) for a target of N bets' worth (gambler's ruin,
# Feller ch. XIV), ≈ (q/p)^n (the ±1 walk below assumes ODDS = 1). A Kelly player stakes f* (10%
# here) of CURRENT wealth, and a half-Kelly player half that, so wealth can never hit zero — but it
# can fall a long way; we count how often it drops below $10 (a 90% loss), and check the martingale
# bound of section 2: P(ever reaching 10% of the start) ≤ 0.1^R.

# %%
N_PLAYERS = 20_000
BANKROLL, TARGET = 100.0, 1_000.0


def gamblers_ruin(p: float, n_units: int, target_units: int) -> float:
    """P(losing n units before winning up to target_units), one-unit even-money bets won with probability p."""
    r = (1 - p) / p
    return (r ** n_units - r ** target_units) / (1 - r ** target_units)


def simulate_fixed(stake: float, n_players: int, rng) -> float:
    """Share of players ruined when staking a fixed dollar amount until $0 or the target."""
    units = np.full(n_players, int(BANKROLL / stake))
    alive = np.ones(n_players, dtype=bool)
    target = int(TARGET / stake)
    while alive.any():
        steps = np.where(rng.random((500, int(alive.sum()))) < P_WIN, 1, -1)
        paths = units[alive] + steps.cumsum(axis=0)
        done = (paths <= 0) | (paths >= target)
        first = np.where(done.any(axis=0), done.argmax(axis=0), -1)
        idx = np.flatnonzero(alive)
        ended = first >= 0
        units[idx[ended]] = paths[first[ended], np.flatnonzero(ended)]
        units[idx[~ended]] = paths[-1, ~ended]
        alive[idx[ended]] = False
    return float((units <= 0).mean())


def simulate_proportional(frac: float, n_players: int, rng, floor: float = 10.0) -> float:
    """Share of proportional bettors whose wealth ever drops below `floor` before reaching the target."""
    log_w = np.full(n_players, np.log(BANKROLL))
    alive = np.ones(n_players, dtype=bool)
    fell = np.zeros(n_players, dtype=bool)
    up, down = np.log1p(ODDS * frac), np.log1p(-frac)
    BIG = 10 ** 9                                         # "never happened in this block"
    while alive.any():
        steps = np.where(rng.random((500, int(alive.sum()))) < P_WIN, up, down)
        paths = log_w[alive, None].T + steps.cumsum(axis=0)
        low, high = paths < np.log(floor), paths >= np.log(TARGET)
        first_low = np.where(low.any(axis=0), low.argmax(axis=0), BIG)
        first_high = np.where(high.any(axis=0), high.argmax(axis=0), BIG)
        idx = np.flatnonzero(alive)
        ended = np.minimum(first_low, first_high) < BIG
        fell[idx[first_low < first_high]] = True
        log_w[idx] = paths[-1]
        alive[idx[ended]] = False
    return float(fell.mean())


for stake in (10.0, 5.0):                               # fixed stakes: the ±1 walk assumes ODDS = 1
    n_units, target_units = int(BANKROLL / stake), int(TARGET / stake)
    exact = gamblers_ruin(P_WIN, n_units, target_units)
    sim = simulate_fixed(stake, N_PLAYERS, rng_ruin)
    se = np.sqrt(exact * (1 - exact) / N_PLAYERS)
    assert abs(sim - exact) < 4 * se, f"fixed ${stake}: ruin frequency off by more than 4 SE"
    key = f"{stake:02.0f}"
    lab.record(f"ruin_fixed_{key}_units", n_units)
    lab.record(f"ruin_fixed_{key}_exact", exact)
    lab.record(f"ruin_fixed_{key}_approx", ((1 - P_WIN) / P_WIN) ** n_units)
    lab.record(f"ruin_fixed_{key}_sim", sim)
    lab.record(f"ruin_fixed_{key}_se", float(se))
    print(f"fixed ${stake:.0f} stakes ({n_units} units): ruin {sim:.4f} (exact {exact:.4f} ± {se:.4f})")
lab.record("ruin_fixed_30", ((1 - P_WIN) / P_WIN) ** 30)

for name, frac in (("full", f_star), ("half", 0.5 * f_star)):
    R = lundberg_exponent(frac, P_WIN, ODDS)
    bound = (10.0 / BANKROLL) ** R                        # P(ever losing 90%) ≤ 0.1^R
    sim = simulate_proportional(frac, N_PLAYERS, rng_ruin)
    assert sim <= bound + 4 * np.sqrt(bound * (1 - bound) / N_PLAYERS), "martingale bound violated"
    lab.record(f"ruin_prop_{name}_sim", sim)
    lab.record(f"ruin_prop_{name}_bound", bound)
    lab.record(f"ruin_prop_{name}_frac", frac)
    print(f"proportional {frac:.0%}: P(ever below $10) {sim:.4f} ≤ bound {bound:.4f}")
lab.record("bankroll", BANKROLL)
lab.record("target", TARGET)
lab.record("n_players", N_PLAYERS)

# %%
lab.save()
