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
# # 1.3 · The Quant Research Lifecycle — companion lab
#
# **Quant Notebook** · Unit 1 · Session 3 · [Read the session](https://georgevrs.github.io/quant-notebook/unit01-quant-landscape/session03-research-lifecycle.html)
#
# The master workflow of the whole course — and the experiment that explains why it has gates.
# We build strategies that have **no skill at all**, try more and more of them, and watch the best
# one look brilliant in-sample and then collapse out-of-sample. Then we write the pre-registration
# that would have stopped us.
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
import json

import numpy as np
import pandas as pd

import quantnb as qn
from quantnb import charts

lab = qn.Lab("1.3")  # seeds the random generators: every run gives the same numbers
rng = lab.rng

# %% [markdown]
# ## 1 · A world with no edge
#
# Each "strategy" earns daily returns with **zero** true mean and 10% annual volatility — pure
# noise, like a coin-flip rule applied to a market it cannot predict. Its true Sharpe ratio is 0.
# We give every strategy 5 years of in-sample history to be "discovered" on, and 5 more years it
# has never seen.

# %%
DAYS = 252
YEARS_IS, YEARS_OOS = 5, 5
VOL = 0.10
daily_sd = VOL / np.sqrt(DAYS)


def annual_sharpe(r: np.ndarray, axis: int = 0) -> np.ndarray:
    return r.mean(axis=axis) / r.std(axis=axis, ddof=1) * np.sqrt(DAYS)


# %% [markdown]
# ## 2 · Try N strategies, keep the best
#
# For each number of trials N we repeat the whole "research project" 200 times: generate N
# skill-less strategies, pick the one with the best in-sample Sharpe, then look at how that
# winner does out-of-sample.

# %%
from scipy.stats import norm

TRIALS = [1, 10, 100, 1000]
REPEATS = 200
EULER_GAMMA = 0.5772156649


def expected_max_sharpe(n: int, years: float) -> float:
    """Expected best Sharpe among n skill-less strategies (False Strategy Theorem approximation).

    Each estimated annual Sharpe is ~ N(0, 1/years); the expected maximum of n standard normals is
    about (1 - γ) Φ⁻¹(1 - 1/n) + γ Φ⁻¹(1 - 1/(n e)), with γ the Euler-Mascheroni constant.
    """
    if n == 1:
        return 0.0
    z = (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))
    return float(z / np.sqrt(years))


rows = []
for n in TRIALS:
    best_is, winner_oos = [], []
    for _ in range(REPEATS):
        is_ret = rng.standard_normal((DAYS * YEARS_IS, n)) * daily_sd
        sr_is = annual_sharpe(is_ret)
        best = int(np.argmax(sr_is))
        best_is.append(sr_is[best])
        oos_ret = rng.standard_normal(DAYS * YEARS_OOS) * daily_sd  # the winner, on data it never saw
        winner_oos.append(annual_sharpe(oos_ret))
    rows.append({
        "N": n,
        "best_is_mean": float(np.mean(best_is)),
        "winner_oos_mean": float(np.mean(winner_oos)),
        "p_best_above_1": float(np.mean(np.array(best_is) > 1.0)),
        "theory": expected_max_sharpe(n, YEARS_IS),
    })
table = pd.DataFrame(rows).set_index("N")
print(table.round(2))

for n, row in table.iterrows():
    lab.record(f"best_is_{n}", row["best_is_mean"])
    lab.record(f"winner_oos_{n}", row["winner_oos_mean"])
    lab.record(f"p_above1_{n}", row["p_best_above_1"])
    lab.record(f"theory_{n}", row["theory"])
lab.record("years_is", YEARS_IS)
lab.record("years_oos", YEARS_OOS)
lab.record("repeats", REPEATS)

# Sanity checks: the winner's out-of-sample Sharpe is ~0 whatever N is, and the in-sample
# maximum matches the False Strategy Theorem's prediction.
assert (table["winner_oos_mean"].abs() < 0.15).all()
assert table.loc[1000, "best_is_mean"] > 1.2
assert (table["best_is_mean"] - table["theory"]).abs().max() < 0.12

# %% [markdown]
# ## 3 · The same idea, tested honestly
#
# One pre-registered strategy, tested once. Its in-sample Sharpe is an unbiased (if noisy) estimate
# of its true Sharpe of zero. The spread tells you how little five years of data can prove.

# %%
single = annual_sharpe(rng.standard_normal((DAYS * YEARS_IS, 5000)) * daily_sd)
lab.record("single_sd", float(single.std()))
lab.record("single_p_above_05", float((single > 0.5).mean()))
print(f"one honest test: sd of the Sharpe estimate {single.std():.2f}; P(SR > 0.5 by luck) {(single > 0.5).mean():.1%}")

# %% [markdown]
# ## 4 · The pre-registration that would have stopped us
#
# Written BEFORE touching the data. It fixes the hypothesis, the test, the number of variants you
# are allowed, and the result that would kill the idea. Commit it to git with a timestamp.

# %%
prereg = {
    "id": "R-001",
    "date": "2026-09-26",
    "hypothesis": "Short-term reversal in large-cap equities: stocks with the worst 5-day return "
                  "outperform over the next 5 days, because liquidity providers are paid to absorb "
                  "selling pressure.",
    "who_pays_us": "impatient sellers demanding immediacy",
    "universe": "US large caps, point-in-time index membership",
    "data": "daily total-return prices, 2005-2019 in-sample; 2020-2024 held out, untouched",
    "signal": "rank of 5-day return, rebalanced weekly, dollar-neutral",
    "primary_metric": "annualised Sharpe ratio after 10 bp one-way costs",
    "variants_allowed": 12,
    "kill_criterion": "deflated Sharpe ratio below 0.95 on in-sample, OR holdout Sharpe below 0",
    "holdout_rule": "evaluated exactly once, after the in-sample decision is frozen",
}
print(json.dumps(prereg, indent=2))
lab.record("prereg_variants_allowed", prereg["variants_allowed"])

# %% [markdown]
# ## 5 · Charts for the session page

# %%
lab.chart("best_of_n", charts.column_chart(
    [f"N = {n:,}" for n in TRIALS], table["best_is_mean"].tolist(),
    title="Best in-sample Sharpe among N skill-less strategies (5 years, 200 repeats)",
    y_fmt=charts.fmt_num(2)))
lab.chart("winner_oos", charts.column_chart(
    [f"N = {n:,}" for n in TRIALS], table["winner_oos_mean"].tolist(),
    title="…and the same winner's Sharpe on 5 unseen years",
    y_fmt=charts.fmt_num(2), roles=["benchmark"] * len(TRIALS)))

# %%
lab.save()
