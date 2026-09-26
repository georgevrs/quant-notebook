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
# # 2.3 · Returns, P&L, Compounding & Leverage — companion lab
#
# **Quant Notebook** · Unit 2 · Session 3 · [Read the session](https://georgevrs.github.io/quant-notebook/unit02-markets-instruments/session03-returns-pnl-leverage.html)
#
# The arithmetic every later session relies on — simple vs log returns, compounding, annualisation,
# volatility drag, Sharpe, drawdown and leverage.
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
from quantnb import charts
from quantnb.returns import (PERIODS_PER_YEAR, annualised_return, annualised_vol, drawdown,
                             leveraged_returns, log_returns, max_drawdown, sharpe_ratio, simple_returns)

lab = qn.Lab("2.3")  # seeds the random generators: every run gives the same numbers
rng = lab.rng

# %% [markdown]
# ## 1 · One trade, four numbers
#
# You buy 200 shares at \$100 and sell at \$110. The same trade can be described as a P&L in
# dollars, a simple return, or a log return. They answer different questions.

# %%
p0, p1, shares = 100.0, 110.0, 200
pnl = (p1 - p0) * shares
r_simple = p1 / p0 - 1
r_log = np.log(p1 / p0)
print(f"P&L ${pnl:,.0f} · simple {r_simple:.2%} · log {r_log:.4f}")
lab.record("trade_pnl", pnl)
lab.record("trade_simple", r_simple)
lab.record("trade_log", float(r_log))

# %% [markdown]
# **The asymmetry of losses.** A 50% loss followed by a 50% gain is not break-even.

# %%
wealth = 1.0 * (1 - 0.5) * (1 + 0.5)
lab.record("down50_up50", wealth - 1)                  # -25%
for loss in (0.10, 0.20, 0.50, 0.80):
    needed = 1 / (1 - loss) - 1
    lab.record(f"recover_{int(loss * 100)}", needed)
    print(f"lose {loss:.0%} → need +{needed:.0%} to get back")
# log returns make the asymmetry disappear: ln(0.5) + ln(1.5) = ln(0.75)
assert np.isclose(np.log(0.5) + np.log(1.5), np.log(wealth))

# %% [markdown]
# ## 2 · Twenty years of prices
#
# Synthetic geometric Brownian motion with a known drift μ = 8% and volatility σ = 20% a year.
# Because we know the truth, we can check every estimator against it.

# %%
MU, SIGMA, RF, YEARS = 0.08, 0.20, 0.03, 20
px = qn.synth.gbm_prices(PERIODS_PER_YEAR * YEARS, mu=MU, sigma=SIGMA, rng=rng)["price"]
r = simple_returns(px)
lr = log_returns(px)
rf_daily = RF / PERIODS_PER_YEAR

ann_arith = r.mean() * PERIODS_PER_YEAR
cagr = annualised_return(r)
vol = annualised_vol(r)
sr = sharpe_ratio(r, rf_daily)
mdd = max_drawdown(r)
print(f"arithmetic mean {ann_arith:.2%} · CAGR {cagr:.2%} · vol {vol:.2%} · Sharpe {sr:.2f} · maxDD {mdd:.1%}")

lab.record("gbm_mu", MU)
lab.record("gbm_sigma", SIGMA)
lab.record("rf", RF)
lab.record("years", YEARS)
lab.record("ann_arith", ann_arith)
lab.record("cagr", cagr)
lab.record("vol", vol)
lab.record("sharpe", sr)
lab.record("sharpe_se", 1 / np.sqrt(YEARS))   # SE of an annualised Sharpe from Y years of daily data
lab.record("maxdd", mdd)
lab.record("final_wealth", float(px.iloc[-1] / px.iloc[0]))

# %% [markdown]
# **Log returns add across time.** The sum of daily log returns is exactly the log of the total
# growth. Simple returns do not add — they compound.

# %%
sum_log = lr.sum()
assert np.isclose(sum_log, np.log(px.iloc[-1] / px.iloc[0]))
naive_sum_simple = r.sum()
lab.record("sum_log", float(sum_log))
lab.record("total_simple", float(px.iloc[-1] / px.iloc[0] - 1))
lab.record("naive_sum_simple", float(naive_sum_simple))

# %% [markdown]
# **Volatility drag.** The average daily return, annualised, overstates what you actually earn.
# What compounds is the average LOG return, and the gap between the two is σ²/2 — here about two
# percentage points a year. (The CAGR is e^g − 1 of the log growth rate g.)

# %%
log_growth = lr.mean() * PERIODS_PER_YEAR
drag_measured = ann_arith - log_growth
drag_theory = vol ** 2 / 2
lab.record("log_growth", float(log_growth))
lab.record("drag_measured", float(drag_measured))
lab.record("drag_theory", drag_theory)
assert abs(drag_measured - drag_theory) < 0.002
assert np.isclose(np.expm1(log_growth), cagr, atol=1e-3)
print(f"log growth {log_growth:.2%} · drag measured {drag_measured:.2%} vs σ²/2 = {drag_theory:.2%}")

# The mean itself is barely known: the standard error of a 20-year mean return is σ/√20.
lab.record("mean_se", SIGMA / np.sqrt(YEARS))

# %% [markdown]
# ## 3 · Simple returns aggregate across assets; log returns don't
#
# A portfolio's simple return is exactly the weighted sum of its assets' simple returns.
# The same is only approximately true for log returns.

# %%
ra, rb, w = 0.30, -0.20, 0.5
port_simple = w * ra + (1 - w) * rb
port_log_true = np.log(1 + port_simple)
port_log_wrong = w * np.log(1 + ra) + (1 - w) * np.log(1 + rb)
lab.record("xs_port_simple", port_simple)
lab.record("xs_port_log_true", float(port_log_true))
lab.record("xs_port_log_wrong", float(port_log_wrong))
print(f"portfolio simple {port_simple:.2%} | true log {port_log_true:.4f} | weighted-log (wrong) {port_log_wrong:.4f}")

# %% [markdown]
# ## 4 · Leverage: more is not always more
#
# Daily-rebalanced leverage L on the same asset, borrowing at the risk-free rate. Theory says the
# expected log-growth rate is g(L) = r_f + L(μ − r_f) − L²σ²/2, which peaks at L* = (μ − r_f)/σ².
# We check it on 2,000 simulated 10-year paths, using the SAME random shocks for every L
# (common random numbers) so the comparison is fair.

# %%
N_PATHS, SIM_YEARS = 2_000, 10
n = PERIODS_PER_YEAR * SIM_YEARS
dt = 1 / PERIODS_PER_YEAR
z = rng.standard_normal((n, N_PATHS))
asset = np.expm1((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * np.sqrt(dt) * z)   # daily simple returns
levels = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
rows = []
for L in levels:
    lev = L * asset - (L - 1) * rf_daily
    lev = np.maximum(lev, -1.0)                       # you cannot lose more than everything
    growth = np.log1p(lev).sum(axis=0) / SIM_YEARS   # annual log growth per path
    cagr_paths = np.expm1(growth)
    # peak-to-trough drawdown of each path's wealth (log wealth, starting at 0)
    log_w = np.vstack([np.zeros((1, N_PATHS)), np.log1p(lev).cumsum(axis=0)])
    max_dd = np.expm1((log_w - np.maximum.accumulate(log_w, axis=0)).min(axis=0))
    rows.append({
        "L": L,
        "median_cagr": float(np.median(cagr_paths)),
        "theory_cagr": float(np.expm1(RF + L * (MU - RF) - 0.5 * L ** 2 * SIGMA ** 2)),
        "p_loss": float((cagr_paths < 0).mean()),
        "p_dd90": float((max_dd <= -0.90).mean()),    # P(a 90% peak-to-trough drawdown within 10 years)
    })
table = pd.DataFrame(rows).set_index("L")
print(table.round(3))
L_star = (MU - RF) / SIGMA ** 2
lab.record("L_star", L_star)
for L, row in table.iterrows():
    key = str(L).replace(".", "_")
    lab.record(f"lev_{key}_median_cagr", row["median_cagr"])
    lab.record(f"lev_{key}_theory_cagr", row["theory_cagr"])
    lab.record(f"lev_{key}_p_loss", row["p_loss"])
    lab.record(f"lev_{key}_p_dd90", row["p_dd90"])
# the simulation must agree with theory within a percentage point
assert (table["median_cagr"] - table["theory_cagr"]).abs().max() < 0.01

# %% [markdown]
# ## 5 · Charts for the session page

# %%
lev_paths = {L: leveraged_returns(r, L, rf_daily).clip(lower=-1.0) for L in (1.0, 2.0, 3.0)}
growth = {L: (1 + lr_).cumprod() for L, lr_ in lev_paths.items()}


def in_years(s: pd.Series) -> pd.Series:
    """Re-index a simulated daily series by elapsed years, so charts don't imply real calendar dates."""
    return pd.Series(s.to_numpy(), index=np.arange(1, len(s) + 1) / PERIODS_PER_YEAR)


lab.chart("growth", charts.line_chart(
    [charts.Series("1× unlevered", in_years(growth[1.0]), role="strategy"),
     charts.Series("2× daily", in_years(growth[2.0]), role="alt1"),
     charts.Series("3× daily", in_years(growth[3.0]), role="alt2")],
    title="Growth of $1 over 20 simulated years (log scale)", logy=True,
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))
lab.chart("drawdown3x", charts.drawdown_chart(in_years(drawdown(lev_paths[3.0])),
                                              title="Drawdown of the 3× version, by simulated year"))
lab.chart("cagr_by_leverage", charts.column_chart(
    [f"{L:g}×" for L in levels], table["median_cagr"].tolist(),
    title="Median compound growth by leverage — 2,000 simulated 10-year paths", y_fmt=charts.fmt_pct(1)))
for L in (1.0, 2.0, 3.0):
    key = str(L).replace(".", "_")
    lab.record(f"path_{key}_cagr", annualised_return(lev_paths[L]))
    lab.record(f"path_{key}_maxdd", max_drawdown(lev_paths[L]))

# %%
lab.save()
