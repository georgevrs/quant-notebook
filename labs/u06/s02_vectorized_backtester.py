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
# # 6.2 · Build a Vectorized Backtester — companion lab
#
# **Quant Notebook** · Unit 6 · Session 2 · [Read the session](https://georgevrs.github.io/quant-notebook/unit06-backtesting-discipline/session02-vectorized-backtester.html)
#
# Signals to positions to returns in a few lines of pandas — and the shift discipline that keeps
# the future out.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.

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

import quantnb as qn
from quantnb import charts, stats, synth
from quantnb.returns import (PERIODS_PER_YEAR, annualised_return, drawdown, in_years,
                             max_drawdown, sharpe_ratio)

lab = qn.Lab("6.2")  # seeds the random generators: every run gives the same numbers
rng = lab.rng

RF = 0.0  # risk-free rate throughout this lab: zero. Excess return = raw return.
lab.record("rf", RF)

# %% [markdown]
# ## 1 · A fair-game universe
#
# Eight synthetic assets, each geometric Brownian motion with **zero drift**. There is no true
# edge anywhere in this data by construction — any signal that looks profitable on it is either
# luck or a bug. That makes it the right data to test a backtester's plumbing, not a strategy.

# %%
N_ASSETS = 8
SIGMAS = np.linspace(0.12, 0.30, N_ASSETS)  # heterogeneous but fixed, known volatilities
lab.record("n_assets", N_ASSETS)
lab.record("sigma_min", float(SIGMAS.min()))
lab.record("sigma_max", float(SIGMAS.max()))


def fair_game_universe(n_days: int, sigmas: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """N independent GBM assets with mu = 0 for every asset: a fair game, no true edge to find.

    quantnb.synth.gbm_prices takes one sigma for every path; this is the vectorized generalisation
    to a per-asset volatility vector, used only for this session's demonstrations.
    """
    dt = 1.0 / PERIODS_PER_YEAR
    z = rng.standard_normal((n_days, len(sigmas)))
    log_r = -0.5 * sigmas ** 2 * dt + sigmas * np.sqrt(dt) * z
    idx = synth.business_days(n_days)
    return pd.DataFrame(np.expm1(log_r), index=idx, columns=[f"A{i}" for i in range(len(sigmas))])


# %% [markdown]
# ## 2 · The backtester: signal → lagged position → return → equity
#
# Three small, reusable functions carry the whole pipeline. `lag=1` is the honest default: the
# position held on day *t* was decided from information available up to the close of day *t-1*.
# `lag=0` reproduces the bug this session is about.

# %%
def lookback_signal(log_returns: pd.DataFrame, k: int) -> pd.DataFrame:
    """Sign of the trailing k-day cumulative log return, for each asset (window ends AT day t)."""
    return np.sign(log_returns.rolling(k).sum())


def equal_weight_positions(signal: pd.DataFrame) -> pd.DataFrame:
    """Scale a +-1/0 signal into a fully-invested, equally-weighted target weight each day."""
    gross = signal.abs().sum(axis=1)
    return signal.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)


def turnover(position: pd.DataFrame) -> pd.Series:
    """One-way turnover: half the sum of absolute day-over-day weight changes (Grinold & Kahn)."""
    t = 0.5 * position.diff().abs().sum(axis=1)
    if len(t):
        t.iloc[0] = 0.5 * position.iloc[0].abs().sum()  # day 1: putting on the initial book
    return t


def vectorized_backtest(returns: pd.DataFrame, target_weights: pd.DataFrame, lag: int = 1) -> dict:
    """Turn target weights into a held position, a return series and an equity curve.

    `lag` periods must pass between deciding a weight and holding it: lag=1 (the honest default)
    means the position on day t was decided using information available before day t's return is
    known. lag=0 uses the SAME day's weight against the same day's return — Session 6.2's bug.
    """
    position = target_weights.shift(lag).fillna(0.0)
    strat_returns = (position * returns).sum(axis=1)
    equity = (1.0 + strat_returns).cumprod()
    return {"position": position, "returns": strat_returns, "equity": equity, "turnover": turnover(position)}


# %% [markdown]
# ## 3 · The headline demo: two years, one line of difference
#
# A 1-day lookback signal (`k=1`): go long an asset if it rose yesterday-by-the-signal's-own-clock,
# short if it fell. Sized equally across whichever assets are "on". The only difference between
# the two backtests below is `lag=1` vs `lag=0`.

# %%
rng_demo, rng_long = rng.spawn(2)  # one stream per experiment

YEARS_DEMO = 2
N_DAYS_DEMO = PERIODS_PER_YEAR * YEARS_DEMO
demo_returns = fair_game_universe(N_DAYS_DEMO, SIGMAS, rng_demo)
demo_log_r = np.log1p(demo_returns)
demo_weights = equal_weight_positions(lookback_signal(demo_log_r, 1))

honest = vectorized_backtest(demo_returns, demo_weights, lag=1)
buggy = vectorized_backtest(demo_returns, demo_weights, lag=0)

for name, bt in (("honest", honest), ("buggy", buggy)):
    cagr = annualised_return(bt["returns"])
    sr = sharpe_ratio(bt["returns"], RF)
    mdd = max_drawdown(bt["returns"])
    ann_to = float(bt["turnover"].mean() * PERIODS_PER_YEAR)
    lab.record(f"demo_{name}_cagr", cagr)
    lab.record(f"demo_{name}_sharpe", sr)
    lab.record(f"demo_{name}_maxdd", mdd)
    lab.record(f"demo_{name}_turnover", ann_to)
    print(f"{name:6s}  CAGR {cagr:8.2%}  Sharpe {sr:7.2f}  maxDD {mdd:7.2%}  turnover/yr {ann_to:5.1f}x")

lab.record("demo_final_honest", float(honest["equity"].iloc[-1]))
lab.record("demo_final_buggy", float(buggy["equity"].iloc[-1]))

# The naive backtest cannot show a drawdown: payoff_t = sign(r_t) * r_t = |r_t| >= 0, every day.
assert max_drawdown(buggy["returns"]) > -1e-9, "the naive equity curve should never fall"
# Both versions trade the same weights, just shifted a day apart: turnover should be nearly identical.
assert abs(honest["turnover"].mean() - buggy["turnover"].mean()) < 0.01, "lag alone should not change turnover"

# %% [markdown]
# **One concrete day.** Find the honest strategy's single worst day and show what the naive
# version reports for that exact same day.

# %%
worst_day = honest["returns"].idxmin()
honest_worst = float(honest["returns"].loc[worst_day])
buggy_same_day = float(buggy["returns"].loc[worst_day])
lab.record("worst_day_honest", honest_worst)
lab.record("worst_day_buggy", buggy_same_day)
print(f"on the honest strategy's worst day, honest {honest_worst:.2%} vs buggy {buggy_same_day:.2%} (same day)")
assert buggy_same_day > 0 > honest_worst, "the naive version should show a GAIN on the honest strategy's worst day"

# %% [markdown]
# ## 4 · Is there really no edge? A 20-year check
#
# The 2-year demo is short enough that an honest Sharpe of, say, 0.3 could easily be noise. Lo's
# (2002) standard error of an annualised Sharpe from `years` of daily data lets us test the null
# properly: with zero true edge, the estimated Sharpe should sit within about 4 standard errors
# of zero. This is a completely separate simulation (its own random stream, 20 years).

# %%
YEARS_LONG = 20
N_DAYS_LONG = PERIODS_PER_YEAR * YEARS_LONG
long_returns = fair_game_universe(N_DAYS_LONG, SIGMAS, rng_long)
long_log_r = np.log1p(long_returns)
long_weights = equal_weight_positions(lookback_signal(long_log_r, 1))
honest_long = vectorized_backtest(long_returns, long_weights, lag=1)
buggy_long = vectorized_backtest(long_returns, long_weights, lag=0)

sr_honest_long = sharpe_ratio(honest_long["returns"], RF)
sr_buggy_long = sharpe_ratio(buggy_long["returns"], RF)
se0 = stats.sharpe_se(0.0, YEARS_LONG)
lab.record("years_long", YEARS_LONG)
lab.record("sr_honest_long", sr_honest_long)
lab.record("sr_buggy_long", sr_buggy_long)
lab.record("se0", se0)
lab.record("se0_x4", 4 * se0)
print(f"honest Sharpe over {YEARS_LONG}y: {sr_honest_long:.3f} (4-SE band +-{4 * se0:.3f}) · buggy Sharpe: {sr_buggy_long:.2f}")

assert abs(sr_honest_long) < 4 * se0, f"honest Sharpe {sr_honest_long:.3f} exceeds the 4-SE band of a fair game"
assert sr_buggy_long > 5.0, "the naive Sharpe should be absurdly high on ANY draw of a fair game"

# %% [markdown]
# ## 5 · Does a longer lookback save you? It shrinks the bug, not fixes it
#
# Day *t*'s own return is 1 of the *k* returns summed inside a k-day lookback signal, so an
# un-shifted signal leaks a `1/k` share of tomorrow's answer into today's decision. Reusing the
# same 20-year universe, at four lookbacks:

# %%
LOOKBACKS = [1, 5, 20, 60]
rows = []
for k in LOOKBACKS:
    w_k = equal_weight_positions(lookback_signal(long_log_r, k))
    h_k = vectorized_backtest(long_returns, w_k, lag=1)
    b_k = vectorized_backtest(long_returns, w_k, lag=0)
    sr_h, sr_b = sharpe_ratio(h_k["returns"], RF), sharpe_ratio(b_k["returns"], RF)
    rows.append({"k": k, "honest": sr_h, "buggy": sr_b, "bias": sr_b - sr_h})
    key = f"k{k}"
    lab.record(f"lookback_{key}_honest", sr_h)
    lab.record(f"lookback_{key}_buggy", sr_b)
    lab.record(f"lookback_{key}_bias", sr_b - sr_h)

bias_table = pd.DataFrame(rows).set_index("k")
print(bias_table.round(3))

biases = bias_table["bias"].tolist()
assert all(b > 0 for b in biases), "look-ahead should inflate Sharpe at every lookback tested"
assert biases == sorted(biases, reverse=True), "the bias should shrink monotonically as the lookback grows"
assert biases[-1] > 0.5, "even a 60-day lookback should still show a clearly visible bias"

# %% [markdown]
# ## 6 · Charts for the session page

# %%
lab.chart("equity", charts.line_chart(
    [charts.Series("Honest (lag=1)", in_years(honest["returns"].pipe(lambda r: (1 + r).cumprod())),
                   role="strategy", end_label="Honest"),
     charts.Series("Naive (lag=0, look-ahead)", in_years(buggy["returns"].pipe(lambda r: (1 + r).cumprod())),
                   role="loss", end_label="Naive")],
    title="Growth of $1 — honest vs naive backtest of the same signal (2 simulated years)",
    logy=True, y_fmt=charts.fmt_num(1, prefix="$"), hline=1.0))

lab.chart("drawdown", charts.drawdown_chart(
    in_years(drawdown(honest["returns"])),
    title="Drawdown of the honest backtest, by simulated year"))

lab.chart("bias_by_lookback", charts.grouped_column_chart(
    [f"k={k}" for k in LOOKBACKS],
    [("Honest (lag=1)", bias_table["honest"].tolist(), "strategy"),
     ("Naive (lag=0)", bias_table["buggy"].tolist(), "loss")],
    title="Sharpe ratio by lookback window — honest vs naive, 20 simulated years",
    y_fmt=charts.fmt_num(1), value_labels=True))

# %%
lab.save()
