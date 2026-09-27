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
# # 6.1 · What a Backtest Is (and Isn't) — companion lab
#
# **Quant Notebook** · Unit 6 · Session 1 · [Read the session](https://georgevrs.github.io/quant-notebook/unit06-backtesting-discipline/session01-what-a-backtest-is.html)
#
# A backtest is a historical simulation under assumptions, not a research tool — what it can prove,
# what it cannot, and why the gap to live trading exists.
#
# We specify one toy trading rule so precisely that two people coding it independently would produce
# identical trades, replay it mechanically over synthetic history, and then change exactly ONE thing
# — the **fill assumption** — to show how much of a backtest's answer lives in an assumption nobody
# wrote down. Every number the session page quotes is recorded with `lab.record(...)` and saved to
# `out/` by the last cell, so the page and this notebook can never disagree.

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
from quantnb import charts
from quantnb.returns import (PERIODS_PER_YEAR, annualised_return, drawdown, in_years,
                             max_drawdown)
from quantnb.stats import annual_sharpe

lab = qn.Lab("6.1")  # seeds the random generators: every run gives the same numbers
rng_path, rng_mc = lab.rng.spawn(2)  # one stream for the illustrative path, one for the Monte Carlo

# %% [markdown]
# ## 1 · A strategy specification precise enough to run
#
# The synthetic asset is geometric Brownian motion with a 7% annual drift and 18% annual volatility
# — a single made-up instrument, so we know the truth: there is no real momentum in it, only noise
# around a drift. Returns are **excess** returns throughout this lab; the risk-free rate is taken as
# 0 (`RF`), stated once here.
#
# The rule itself — a trend filter — is written down completely enough that nothing is left to
# judgement:
#
# * **Universe.** One synthetic asset.
# * **Signal.** `price > its trailing 100-day moving average` (else flat). 100 trading days is
#   about five months — a slow filter, chosen so single days rarely flip it.
# * **Entry / exit.** Go long the day the signal turns on; go flat the day it turns off. No
#   partial positions.
# * **Position sizing.** All-in when the signal is on (weight 1), all-out when it is off (weight
#   0). No leverage, no shorting.
# * **Costs.** Zero, by assumption — flagged explicitly because it is a assumption, not a fact
#   (Session 6.5 prices it properly).
#
# That whole list is what this session calls a **strategy specification**: everything needed for
# two people to run the same rule on the same data and get identical trades. One piece of it is
# still open — at what price, and after what delay, a trade is assumed to fill — and that is the
# **fill assumption** this lab isolates.

# %%
MU, SIGMA = 0.07, 0.18   # synthetic asset: annual drift and volatility
RF = 0.0                 # synthetic excess returns; risk-free rate taken as 0 throughout this lab
WINDOW = 100             # trend-filter lookback, in trading days

YEARS_PATH = 8
N_DAYS_PATH = PERIODS_PER_YEAR * YEARS_PATH
px = qn.synth.gbm_prices(N_DAYS_PATH, mu=MU, sigma=SIGMA, rng=rng_path)["price"]


def run_variants(prices: pd.Series, window: int) -> pd.DataFrame:
    """Replay ONE strategy — long when price is above its `window`-day moving average, else flat —
    under two fill assumptions, mechanically, one day at a time (a plain loop over history is what a
    backtest fundamentally does; Session 6.2 turns this into fast vectorised array operations, and
    6.3 into an event loop with real order objects — same specification, faster or more realistic
    machinery). Returns daily simple returns for three variants, aligned on the same dates:

      buyhold — always long (the passive benchmark)
      honest  — position decided using info through day t, filled at day t+1's close
                (a full day of implementation lag: what you could actually trade on)
      naive   — the same signal filled at day t's OWN close (the day that helped decide it) —
                a one-line bug (forgetting to lag the signal) that is also a fill assumption:
                just a dishonest one.
    """
    ma = prices.rolling(window).mean()
    signal = (prices > ma).astype(float)      # 1 = trend up, using information through THIS close
    ret = prices.pct_change()
    honest = signal.shift(1) * ret
    naive = signal * ret
    out = pd.DataFrame({"buyhold": ret, "honest": honest, "naive": naive})
    return out.iloc[window:]                  # drop the moving-average warm-up (undefined signal)


variants = run_variants(px, WINDOW)
for name in ("buyhold", "honest", "naive"):
    s = variants[name]
    lab.record(f"{name}_cagr", annualised_return(s))
    lab.record(f"{name}_sharpe", float(annual_sharpe(s - RF / PERIODS_PER_YEAR)))
    lab.record(f"{name}_maxdd", max_drawdown(s))

# the position actually held each day (the lagged signal) is what "time invested" and "a trade" mean
signal_lagged = (px > px.rolling(WINDOW).mean()).astype(float).shift(1).iloc[WINDOW:]
n_trades = int(signal_lagged.diff().abs().eq(1.0).sum())
lab.record("frac_invested", float(signal_lagged.mean()))
lab.record("n_trades", n_trades)
lab.record("window", WINDOW)
lab.record("years_path", YEARS_PATH)
lab.record("mu", MU)
lab.record("sigma", SIGMA)
lab.record("rf", RF)
print(variants.agg(["mean", "std"]).T)
print(f"trades over {YEARS_PATH} years: {n_trades} · time invested: {signal_lagged.mean():.0%}")

# sanity: a slow trend filter on 8 years of daily data trades a moderate, not absurd, number of times
assert 5 <= n_trades <= 200
assert 0.15 < signal_lagged.mean() < 0.85

# %% [markdown]
# ## 2 · One assumption, changed — the Monte Carlo evidence
#
# One path is an anecdote. To show the fill assumption changes the answer *systematically*, and not
# just for this one lucky (or unlucky) draw, we repeat the exact same experiment on 400 independent
# simulated worlds and look at the average gap between the naive and honest Sharpe ratios.
#
# Because the asset is pure GBM, the trend filter has no real edge to find — any reliable, positive
# gap between the two fill assumptions can only come from the fill assumption itself.

# %%
N_PATHS, YEARS_MC = 400, 4
N_DAYS_MC = PERIODS_PER_YEAR * YEARS_MC
px_mc = qn.synth.gbm_prices(N_DAYS_MC, mu=MU, sigma=SIGMA, n_paths=N_PATHS, rng=rng_mc)

ma_mc = px_mc.rolling(WINDOW).mean()
signal_mc = (px_mc > ma_mc).astype(float)
ret_mc = px_mc.pct_change()
honest_mc = (signal_mc.shift(1) * ret_mc).iloc[WINDOW:]
naive_mc = (signal_mc * ret_mc).iloc[WINDOW:]
buyhold_mc = ret_mc.iloc[WINDOW:]

sr_honest = annual_sharpe(honest_mc.to_numpy(), axis=0)
sr_naive = annual_sharpe(naive_mc.to_numpy(), axis=0)
sr_buyhold = annual_sharpe(buyhold_mc.to_numpy(), axis=0)

gap = sr_naive - sr_honest                      # per-path advantage the naive fill manufactures
mean_gap = float(gap.mean())
se_gap = float(gap.std(ddof=1) / np.sqrt(N_PATHS))

lab.record("mc_paths", N_PATHS)
lab.record("mc_years", YEARS_MC)
lab.record("mean_sr_buyhold", float(sr_buyhold.mean()))
lab.record("mean_sr_honest", float(sr_honest.mean()))
lab.record("mean_sr_naive", float(sr_naive.mean()))
lab.record("mean_gap", mean_gap)
lab.record("se_gap", se_gap)
lab.record("gap_se_multiple", mean_gap / se_gap)

# The honest-fill strategy, filtering PURE noise with no real trend to find, forfeits some of the
# passive drift (it is out of the market on days it cannot know in advance are about to rise) while
# adding whipsaw at every crossing — so it should reliably UNDERPERFORM simply holding the asset.
# This is not a bug: it is the honest, unglamorous answer a correctly specified simulation gives
# when the rule has nothing real to find.
honest_vs_buyhold = float(sr_honest.mean() - sr_buyhold.mean())
se_vs_buyhold = float((sr_honest - sr_buyhold).std(ddof=1) / np.sqrt(N_PATHS))
lab.record("honest_vs_buyhold", honest_vs_buyhold)
lab.record("honest_vs_buyhold_se_multiple", abs(honest_vs_buyhold) / se_vs_buyhold)
print(f"mean Sharpe — buy&hold {sr_buyhold.mean():.2f} · honest fill {sr_honest.mean():.2f} · "
      f"naive fill {sr_naive.mean():.2f}")
print(f"naive − honest gap: {mean_gap:.3f} ± {se_gap:.3f} SE ({mean_gap / se_gap:.1f} SE from zero)")
print(f"honest − buy&hold gap: {honest_vs_buyhold:.3f} ({abs(honest_vs_buyhold) / se_vs_buyhold:.1f} SE from zero)")

# The naive (zero-lag) fill assumption must inflate the Sharpe ratio by a RELIABLE amount — at
# least 4 standard errors from zero, never a property of one lucky path (§12a of AUTHORING.md) —
# and the inflation must be big enough to matter, not just statistically detectable.
assert mean_gap > 4 * se_gap
assert mean_gap > 0.15
# And the honestly-filled strategy must reliably trail buy & hold (no free lunch from filtering
# noise) — also at 4 SE, never a property of one path.
assert sr_buyhold.mean() - sr_honest.mean() > 4 * se_vs_buyhold

# %% [markdown]
# ## 3 · Charts for the session page

# %%
growth = pd.DataFrame({k: (1 + variants[k]).cumprod() for k in ("buyhold", "honest", "naive")})
lab.chart("equity", charts.line_chart(
    [charts.Series("Buy & hold", in_years(growth["buyhold"]), role="benchmark", end_label="Buy&hold"),
     charts.Series("Honest fill (1-day lag)", in_years(growth["honest"]), role="strategy", end_label="Honest"),
     charts.Series("Naive fill (0-day lag)", in_years(growth["naive"]), role="loss", end_label="Naive")],
    title="Growth of $1, one simulated 8-year history, by fill assumption",
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

lab.chart("sharpe_gap", charts.column_chart(
    ["Buy & hold", "Honest fill", "Naive fill"],
    [float(sr_buyhold.mean()), float(sr_honest.mean()), float(sr_naive.mean())],
    title=f"Average annualised Sharpe across {N_PATHS} simulated worlds, by fill assumption",
    y_fmt=charts.fmt_num(2), roles=["benchmark", "strategy", "loss"]))

# %%
lab.save()
