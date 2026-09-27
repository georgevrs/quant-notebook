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
# # 4.1 · The Data Landscape, Vendors & Licensing — companion lab
#
# **Quant Notebook** · Unit 4 · Session 1 · [Read the session](https://georgevrs.github.io/quant-notebook/unit04-data-engineering/session01-data-landscape.html)
#
# Bars, ticks, quotes, books, fundamentals and macro — who sells what, what it costs, and what the licence lets you do with it.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Everything here is **synthetic**: one illustrative trading day for one imaginary stock, built from
# a known geometric Brownian motion so we know the truth. No vendor's real feed is measured, sampled
# or redistributed — the point is the *shape* of the trade-offs (bytes, precision, engines), not a
# specific vendor's numbers. Vendor prices and licence terms quoted on the session page are cited to
# public sources instead, with a date.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

import quantnb as qn
from quantnb import charts, stats

lab = qn.Lab("4.1")  # seeds the random generators: every run gives the same numbers
rng = lab.rng
rng_day, rng_mc = rng.spawn(2)  # one stream per experiment: §1's tick day, §4's Monte Carlo


def gbm_path_irregular(t_years: np.ndarray, mu: float, sigma: float, s0: float,
                       rng: np.random.Generator) -> np.ndarray:
    """GBM prices at arbitrary elapsed times (years), for one path.

    Ticks and quotes do not arrive on a regular grid, so each step's variance is scaled by the
    ACTUAL gap since the previous timestamp rather than assuming equal spacing.
    """
    dt = np.diff(np.concatenate([[0.0], t_years]))
    z = rng.standard_normal(len(dt))
    log_r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    return s0 * np.exp(np.cumsum(log_r))


# %% [markdown]
# ## 1 · One synthetic trading day, at three granularities
#
# A US cash-equity session is 6.5 hours. We build the SAME imaginary stock's day three ways: every
# trade (ticks), every quote update, and the 1-minute OHLCV bars a vendor would sell you. The trade
# and quote counts below are illustrative round numbers for "a moderately liquid large-cap stock" —
# not measured from any real feed — chosen only so the three granularities differ by realistic orders
# of magnitude.

# %%
SESSION_HOURS = 6.5
SESSION_SECONDS = int(SESSION_HOURS * 3600)          # 23,400 s
TRADING_DAYS_YEAR = 252
N_TICKS = 20_000            # illustrative trade count for one session
QUOTE_MULTIPLIER = 6        # quotes update several times per trade — illustrative, not measured
N_QUOTES = N_TICKS * QUOTE_MULTIPLIER
BAR_MINUTES = 1
N_BARS = int(SESSION_HOURS * 60 / BAR_MINUTES)       # 390 one-minute bars
S0, MU, SIGMA, SPREAD_BPS = 100.0, 0.08, 0.25, 4.0   # a single stock: more volatile than 2.3's index

# Given a day's trade COUNT, a Poisson process's arrival times are, conditionally, distributed as
# that many independent uniforms over the session, sorted — the standard way to simulate one.
tick_secs = np.sort(rng_day.uniform(0, SESSION_SECONDS, N_TICKS))
quote_secs = np.sort(rng_day.uniform(0, SESSION_SECONDS, N_QUOTES))
tick_years = tick_secs / SESSION_SECONDS / TRADING_DAYS_YEAR
quote_years = quote_secs / SESSION_SECONDS / TRADING_DAYS_YEAR

trade_price = gbm_path_irregular(tick_years, MU, SIGMA, S0, rng_day)
mid_at_quotes = np.interp(quote_secs, tick_secs, trade_price)     # the true mid, interpolated
half_spread = mid_at_quotes * (SPREAD_BPS / 2 / 1e4)
bid, ask = mid_at_quotes - half_spread, mid_at_quotes + half_spread

SESSION_START = pd.Timestamp("2031-06-03 09:30:00")  # a synthetic future date — never a real session
tick_ts = SESSION_START + pd.to_timedelta(tick_secs, unit="s")
quote_ts = SESSION_START + pd.to_timedelta(quote_secs, unit="s")
trade_size = rng_day.integers(1, 500, N_TICKS)

ticks = pd.DataFrame({"ts": tick_ts, "price": trade_price, "size": trade_size})
quotes = pd.DataFrame({"ts": quote_ts, "bid": bid, "ask": ask})

tick_px = ticks.set_index("ts")["price"]
tick_sz = ticks.set_index("ts")["size"]
bars = tick_px.resample(f"{BAR_MINUTES}min").ohlc().join(tick_sz.resample(f"{BAR_MINUTES}min").sum().rename("volume"))
bars = bars.dropna().reset_index()

lab.record("session_hours", SESSION_HOURS)
lab.record("n_ticks", N_TICKS)
lab.record("n_quotes", N_QUOTES)
lab.record("n_bars", len(bars))
lab.record("spread_bps", SPREAD_BPS)
lab.record("sigma_mc", SIGMA)
lab.record("mu_mc", MU)
assert len(bars) == N_BARS, "20,000 trades spread over 390 minutes should fill every bar"

# %% [markdown]
# ## 2 · What granularity costs in bytes
#
# Same day, three files. We write each to Parquet (and the ticks to CSV, for comparison) and measure
# the ACTUAL bytes on disk — no vendor numbers, just this run's own data.

# %%
with tempfile.TemporaryDirectory() as tmp_str:
    tmp = Path(tmp_str)
    ticks_pq_path = tmp / "ticks.parquet"
    ticks.to_parquet(ticks_pq_path, engine="pyarrow", index=False)
    quotes_pq_path = tmp / "quotes.parquet"
    quotes.to_parquet(quotes_pq_path, engine="pyarrow", index=False)
    bars_pq_path = tmp / "bars.parquet"
    bars.to_parquet(bars_pq_path, engine="pyarrow", index=False)
    ticks_csv_path = tmp / "ticks.csv"
    ticks.to_csv(ticks_csv_path, index=False)

    ticks_pq_bytes = ticks_pq_path.stat().st_size
    quotes_pq_bytes = quotes_pq_path.stat().st_size
    bars_pq_bytes = bars_pq_path.stat().st_size
    ticks_csv_bytes = ticks_csv_path.stat().st_size

    # §3 needs the tick Parquet file to still exist on disk — read it before the temp dir is gone.
    bars_sql = duckdb.sql(f"""
        SELECT time_bucket(INTERVAL '{BAR_MINUTES} minutes', ts) AS ts,
               first(price ORDER BY ts) AS open,
               max(price)               AS high,
               min(price)               AS low,
               last(price ORDER BY ts)  AS close,
               sum(size)                AS volume
        FROM read_parquet('{ticks_pq_path.as_posix()}')
        GROUP BY 1 ORDER BY 1
    """).df()
    mean_price_lazy = pl.scan_parquet(ticks_pq_path).select(pl.col("price").mean()).collect().item()

bytes_per_tick = ticks_pq_bytes / N_TICKS
bytes_per_quote = quotes_pq_bytes / N_QUOTES
bytes_per_bar = bars_pq_bytes / len(bars)
compression_ratio = ticks_csv_bytes / ticks_pq_bytes
raw_bytes = ticks_pq_bytes + quotes_pq_bytes
raw_to_bar_ratio = raw_bytes / bars_pq_bytes
UNIVERSE_SIZE = 500  # illustrative: "an S&P 500-sized universe"
annual_gb_one_symbol = raw_bytes * TRADING_DAYS_YEAR / 1e9
annual_gb_universe = annual_gb_one_symbol * UNIVERSE_SIZE

lab.record("ticks_pq_bytes", ticks_pq_bytes)
lab.record("quotes_pq_bytes", quotes_pq_bytes)
lab.record("bars_pq_bytes", bars_pq_bytes)
lab.record("bytes_per_tick", bytes_per_tick)
lab.record("bytes_per_quote", bytes_per_quote)
lab.record("bytes_per_bar", bytes_per_bar)
lab.record("compression_ratio", compression_ratio)
lab.record("raw_to_bar_ratio", raw_to_bar_ratio)
lab.record("universe_size", UNIVERSE_SIZE)
lab.record("annual_gb_one_symbol", annual_gb_one_symbol)
lab.record("annual_gb_universe", annual_gb_universe)

print(f"bytes/record: tick {bytes_per_tick:.1f}, quote {bytes_per_quote:.1f}, bar {bytes_per_bar:.1f}")
print(f"parquet vs csv: {compression_ratio:.2f}x smaller; raw feed vs its own bars: {raw_to_bar_ratio:.1f}x bigger")
assert compression_ratio > 1.1, "Parquet should beat CSV on this numeric data"
assert raw_to_bar_ratio > 20, "the raw feed a bar is built from should dwarf the bar file it becomes"

lab.chart("bytes_per_record", charts.column_chart(
    ["OHLCV bar", "Trade tick", "Quote"], [bytes_per_bar, bytes_per_tick, bytes_per_quote],
    title="Parquet bytes per record, by data type (one synthetic session)",
    y_fmt=charts.fmt_num(0, suffix=" B"), roles=["strategy", "alt1", "alt2"]))

# %% [markdown]
# ## 3 · Bars are a query, not a separate feed
#
# DuckDB computes the same 1-minute bars directly from the tick Parquet file with SQL, and polars
# reads only the one column it needs from the same file without materialising the rest. Both must
# agree exactly with the pandas `resample().ohlc()` computed in §1 — three engines, one truth.

# %%
close_match = bool(np.allclose(bars_sql["close"].to_numpy(), bars["close"].to_numpy(), atol=1e-9))
open_match = bool(np.allclose(bars_sql["open"].to_numpy(), bars["open"].to_numpy(), atol=1e-9))
volume_match = bool((bars_sql["volume"].to_numpy() == bars["volume"].to_numpy()).all())
lab.record("duckdb_matches_pandas", close_match and open_match and volume_match)
assert close_match and open_match and volume_match, "DuckDB's SQL bars must exactly match pandas' resample of the same ticks"
assert np.isclose(mean_price_lazy, ticks["price"].mean()), "polars' lazy column scan must match pandas' mean"
print("DuckDB SQL bars == pandas resample bars:", close_match and open_match and volume_match)

# %% [markdown]
# ## 4 · What a bar throws away
#
# A close-to-close "bar" return is ONE noisy observation of the day's variance. Splitting the same
# day into M intraday sub-returns and summing their squares (realized variance) is a much more
# precise estimator of the same quantity. We know the truth here (it is a simulation), so we can
# check the sampling variance of each estimator against exact theory: for i.i.d. Gaussian log-returns,
# a single squared demeaned return has variance 2σ⁴, while the sum of M independent squared
# sub-returns has variance 2σ⁴/M — M times less.

# %%
M = int(SESSION_HOURS * 60 / 5)              # 78 five-minute sub-intervals per session
N_DAYS_MC = 8_000
dt_day = 1.0 / TRADING_DAYS_YEAR
dt_sub = dt_day / M
drift_sub = (MU - 0.5 * SIGMA ** 2) * dt_sub

z = rng_mc.standard_normal((N_DAYS_MC, M))
log_sub = drift_sub + SIGMA * np.sqrt(dt_sub) * z
resid_sub = log_sub - drift_sub                       # = σ√dt_sub · z, exactly, by construction
resid_day = resid_sub.sum(axis=1)                     # = σ√dt_day · Z,  Z ~ N(0,1) exactly

bar_var_est = resid_day ** 2                          # one close-to-close "bar" observation per day
rv_est = (resid_sub ** 2).sum(axis=1)                 # realized variance from M intraday sub-returns

sigma2_day = SIGMA ** 2 * dt_day
var_bar_theory = 2 * sigma2_day ** 2
var_rv_theory = 2 * sigma2_day ** 2 / M
var_bar_emp = float(bar_var_est.var(ddof=1))
var_rv_emp = float(rv_est.var(ddof=1))

se_bar = stats.batch_se(lambda b: b.var(ddof=1), bar_var_est, n_batches=50)
se_rv = stats.batch_se(lambda b: b.var(ddof=1), rv_est, n_batches=50)

lab.record("m_subintervals", M)
lab.record("var_ratio_emp", var_bar_emp / var_rv_emp)
lab.record("sd_bar", np.sqrt(var_bar_theory))
lab.record("sd_rv", np.sqrt(var_rv_theory))
lab.record("sd_ratio", np.sqrt(var_bar_theory) / np.sqrt(var_rv_theory))   # = sqrt(M)

print(f"Var(bar est.) = {var_bar_emp:.3e} (theory {var_bar_theory:.3e}); "
      f"Var(RV est.) = {var_rv_emp:.3e} (theory {var_rv_theory:.3e})")
print(f"ratio = {var_bar_emp / var_rv_emp:.1f} (theory M = {M})")

# 4 standard errors, never 3 — see AUTHORING.md §12a.
assert abs(var_bar_emp - var_bar_theory) < 4 * se_bar, "empirical Var(bar estimator) should match theory at 4 SE"
assert abs(var_rv_emp - var_rv_theory) < 4 * se_rv, "empirical Var(RV estimator) should match theory at 4 SE"

# %%
lab.save()
