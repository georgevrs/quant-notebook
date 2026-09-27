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
# # 4.3 · Corporate Actions, Security Master & Continuous Futures — companion lab
#
# **Quant Notebook** · Unit 4 · Session 3 · [Read the session](https://georgevrs.github.io/quant-notebook/unit04-data-engineering/session03-corporate-actions-security-master.html)
#
# Adjust prices correctly, keep identifiers straight through ticker changes, and stitch futures into continuous series without inventing returns.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Three pipelines, one lesson: an **adjustment factor** is just a number a security master
# multiplies into history, and it has to compound correctly whether the trigger is a dividend,
# a split, or a futures roll. Everything here is synthetic and illustrative — seeded, offline,
# and small enough to read end to end. `2.2` owns *why* a dividend or split moves the quoted price;
# this lab owns *how a pipeline turns that event into a number* and chains many of them together.

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
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR

lab = qn.Lab("4.3")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
rng_ca, rng_cf, rng_mc = lab.rng.spawn(3)

# %% [markdown]
# ## 1 · A security master in miniature: why the ticker is not the key
#
# A ticker is a mailbox, not an owner. Exchanges recycle them: a company delists, and months or
# years later an unrelated company is assigned the same three letters. A pipeline that joins
# history on the ticker alone will happily splice two strangers into one "track record."

# %%
# Two unrelated, permanently-identified companies that both trade, at different times, as "QNTX".
PERM_A, PERM_B = "PERM-00185", "PERM-00931"     # permanent internal keys — never reused
A_START, A_END = 42.00, 95.00                   # PERM-00185: "QNTX" 2015-2018, acquired
B_START, B_END = 18.00, 11.50                   # PERM-00931: "QNTX" 2021-2024, an unrelated IPO

a_return = A_END / A_START - 1.0
b_return = B_END / B_START - 1.0
naive_ticker_return = B_END / A_START - 1.0     # what a ticker-keyed join computes end to end
gap_jump = B_START / A_END - 1.0                # the fabricated "return" hiding in the gap

lab.record("sm_a_start", A_START)
lab.record("sm_a_end", A_END)
lab.record("sm_a_return", a_return)
lab.record("sm_b_start", B_START)
lab.record("sm_b_end", B_END)
lab.record("sm_b_return", b_return)
lab.record("sm_naive_return", naive_ticker_return)
lab.record("sm_gap_jump", gap_jump)
# the naive, ticker-keyed "return" is neither company's real return — a number describing nothing
assert not np.isclose(naive_ticker_return, a_return, atol=1e-6)
assert not np.isclose(naive_ticker_return, b_return, atol=1e-6)
print(f"PERM-00185 (as QNTX, 2015-18): {a_return:+.1%} · PERM-00931 (as QNTX, 2021-24): {b_return:+.1%}\n"
      f"naive ticker-keyed 'QNTX' return end to end: {naive_ticker_return:+.1%} (real answer to no real question)")

# %% [markdown]
# ## 2 · From raw quotes to an adjusted price and a total-return index
#
# A raw price feed prints whatever traded. A **security master** carries, for each permanent
# identifier, the table of known corporate actions; a pipeline turns each action into an
# **adjustment factor** and compounds them into the **adjustment factor** used to build an
# **adjusted price** series and a **total-return index**. `2.2` derives *why* a dividend or a
# split moves the quoted price this way (§2.2.2-§2.2.3) — this section owns the arithmetic that
# turns those known events into one clean series, and shows what breaks when it is done wrong.

# %%
CA_YEARS, MU_CA, SIGMA_CA, S0_CA = 4, 0.08, 0.22, 60.0
N_DAYS_CA = CA_YEARS * PERIODS_PER_YEAR
# tr_true is the TRUE economic value an owner held, dividends reinvested (2.2 §2.2.3's total return,
# 2.3's compounding) — the ground truth this pipeline must recover from a raw, un-adjusted feed.
tr_true = qn.synth.gbm_prices(N_DAYS_CA, mu=MU_CA, sigma=SIGMA_CA, s0=S0_CA, rng=rng_ca)["price"].to_numpy()

DIV_DAYS = np.array([150, 400, 650, 900])         # four ex-dividend dates over the four years
DIV_YIELD_Q = 0.005                               # 0.5% of price per quarter (~2%/yr), 2.2's convention
SPLIT_DAY, SPLIT_RATIO = 500, 3                    # one 3-for-1 split partway through

day = np.arange(N_DAYS_CA + 1)
k_t = np.searchsorted(DIV_DAYS, day, side="right")     # dividends already gone ex, as of day t
s_t = (day >= SPLIT_DAY).astype(int)                    # 1 once the split has happened
# what the raw feed prints: the true value, divided down by every action that has ALREADY fired
raw_quoted = tr_true / ((1.0 + DIV_YIELD_Q) ** k_t * float(SPLIT_RATIO) ** s_t)

# The adjustment factor per action — the number a security master looks up and multiplies in:
FACTOR_DIV = 1.0 / (1.0 + DIV_YIELD_Q)      # < 1: shrinks earlier prices, CRSP's own convention
FACTOR_SPLIT = 1.0 / SPLIT_RATIO
K = len(DIV_DAYS)
n_div_after = K - k_t                        # dividends still AHEAD of day t
n_split_after = 1 - s_t
cum_adj_factor = FACTOR_DIV ** n_div_after * FACTOR_SPLIT ** n_split_after   # anchored: 1.0 on the last day
adjusted_price = raw_quoted * cum_adj_factor

# The adjusted price is not the true LEVEL — it is anchored to today's real quote — but it is
# EXACTLY proportional to the true value at every date: same shape, different base.
proportionality = adjusted_price / tr_true
assert np.allclose(proportionality, proportionality[-1], atol=1e-9), "adjusted price must track true value exactly"
assert np.isclose(adjusted_price[-1], raw_quoted[-1], atol=1e-9)   # anchor: today is never restated

lab.record("ca_years", CA_YEARS)
lab.record("ca_div_yield_q", DIV_YIELD_Q)
lab.record("ca_n_divs", K)
lab.record("ca_split_ratio", SPLIT_RATIO)
lab.record("ca_split_year", SPLIT_DAY / PERIODS_PER_YEAR)
lab.record("ca_factor_div", FACTOR_DIV)
lab.record("ca_factor_split", FACTOR_SPLIT)
lab.record("ca_cum_factor_day0", float(cum_adj_factor[0]))


def cagr(level: np.ndarray, years: float) -> float:
    return float((level[-1] / level[0]) ** (1.0 / years) - 1.0)


lab.record("ca_true_cagr", cagr(tr_true, CA_YEARS))
lab.record("ca_adjusted_cagr", cagr(adjusted_price, CA_YEARS))
lab.record("ca_raw_naive_cagr", cagr(raw_quoted, CA_YEARS))            # what you'd wrongly compute, unadjusted
assert abs(lab.results["ca_true_cagr"] - lab.results["ca_adjusted_cagr"]) < 1e-9

# A total-return index is the same construction, rebased to a round number at the start — the way
# an index provider publishes one (S&P DJI, MSCI): same shape as the adjusted price, different base.
tri = adjusted_price / adjusted_price[0] * 100.0
lab.record("ca_tri_start", float(tri[0]))
lab.record("ca_tri_end", float(tri[-1]))
assert np.allclose(tri / tri[0], adjusted_price / adjusted_price[0], atol=1e-9)

# "Adjustment on adjustment": before the fourth dividend is entered into the security master, a
# pipeline computes cum_adj_factor with only 3 dividends ahead of every day before it. Once the
# master learns about it, EVERY historical adjusted price before that ex-date must be rescaled —
# not just the new day appended.
stale_ratio = 1.0 + DIV_YIELD_Q
n_div_after_stale = (K - 1) - np.searchsorted(DIV_DAYS[:-1], day, side="right")
cum_factor_stale = FACTOR_DIV ** n_div_after_stale * FACTOR_SPLIT ** n_split_after
adjusted_stale = raw_quoted * cum_factor_stale
affected = day < DIV_DAYS[-1]
assert np.allclose(adjusted_stale[affected], adjusted_price[affected] * stale_ratio, atol=1e-9)
assert np.allclose(adjusted_stale[~affected], adjusted_price[~affected], atol=1e-9)
lab.record("ca_stale_rows_changed", int(affected.sum()))
lab.record("ca_stale_pct", float(affected.mean()))
lab.record("ca_stale_ratio", stale_ratio)
print(f"true CAGR {lab.results['ca_true_cagr']:.2%} == adjusted CAGR {lab.results['ca_adjusted_cagr']:.2%}; "
      f"raw (un-adjusted) CAGR {lab.results['ca_raw_naive_cagr']:.2%}; "
      f"a late-arriving dividend restates {affected.sum()} of {len(day)} historical rows")

years_ax = day / PERIODS_PER_YEAR
lab.chart("adjusted_vs_raw", charts.line_chart(
    [charts.Series("raw quoted price", pd.Series(raw_quoted, index=years_ax), role="benchmark", end_label="raw"),
     charts.Series("adjusted price / total-return index (rebased)", pd.Series(adjusted_price, index=years_ax),
                  role="strategy", end_label="adjusted")],
    title="Raw quoted price vs adjusted price, one stock, four corporate actions (elapsed years)",
    y_fmt=charts.fmt_num(0, prefix="$")))

# %% [markdown]
# ## 3 · Continuous futures: per-contract data, not one long price
#
# `2.4` prices a futures curve from cost of carry, F = S·exp(cτ) (§2.4.2), and shows that a chart
# of spliced front-month prices hides the roll yield (§2.4.4-§2.4.5). This section builds the
# **continuous futures** series `2.4` promised: real, overlapping per-contract prices, a
# **roll schedule** that decides when to switch, and the two standard splicing conventions —
# **back-adjustment** and **ratio adjustment** — computed from that data rather than asserted.

# %%
YEARS_CF = 8
CONTRACT_DAYS = 63                                  # one quarter, trading days
N_CONTRACTS = YEARS_CF * PERIODS_PER_YEAR // CONTRACT_DAYS
ROLL_CAL_DAYS = 10                                  # calendar rule: roll this many days before expiry
ROLL_WINDOW = 16                                    # liquidity crossover sits at half this window
LIQ_ROLL_DAYS = ROLL_WINDOW // 2
# An extreme, illustrative backwardation — chosen to make the back-adjustment defect visible.
# Real curves are rarely this steep for this long; 2.4's "tight" crude example used -12%.
C_MAIN = -0.25
MU_SPOT, SIGMA_SPOT, S0_SPOT = 0.03, 0.25, 70.0

n_days_cf = N_CONTRACTS * CONTRACT_DAYS
expiries = np.arange(1, N_CONTRACTS + 1) * CONTRACT_DAYS         # E_1 .. E_N, trading days
spot_cf = qn.synth.gbm_prices(n_days_cf, mu=MU_SPOT, sigma=SIGMA_SPOT, s0=S0_SPOT, rng=rng_cf)["price"].to_numpy()
day_cf = np.arange(n_days_cf + 1)

# Per-roll data as a real table: near every expiry, the outgoing (front) contract's synthetic
# volume ramps down and the incoming (next) contract's ramps up — CME's own Active Contract
# series picks the roll date the same way (see the Go-deeper reference): whichever contract the
# market has already voted for with volume, not a fixed date.
BASE_VOL = 10_000.0
k_window = np.arange(0, ROLL_WINDOW + 1)                    # trading days to the FRONT contract's expiry
ramp = np.clip(k_window / ROLL_WINDOW, 0.0, 1.0)            # 1 far from expiry, 0 at expiry
front_vol_k, next_vol_k = BASE_VOL * ramp, BASE_VOL * (1.0 - ramp)

frames = []
for i, exp in enumerate(expiries[:-1], start=1):            # one window per roll (contract i -> i+1)
    frames.append(pl.DataFrame({"roll": i, "day": exp - k_window,
                                "front_volume": front_vol_k, "next_volume": next_vol_k}))
rolls_pl = pl.concat(frames)

# Use Parquet + DuckDB, not a CSV, for the roll query — a real pipeline keeps per-contract data
# this way (Session 4.5 builds the storage stack properly; here it earns its keep on one query).
with tempfile.TemporaryDirectory() as tmp:
    parquet_path = Path(tmp) / "rolls.parquet"
    rolls_pl.write_parquet(parquet_path)
    with duckdb.connect() as con:
        crossover = con.execute(
            """
            SELECT roll, min(day) AS roll_day
            FROM read_parquet(?)
            WHERE next_volume >= front_volume
            GROUP BY roll
            ORDER BY roll
            """,
            [str(parquet_path)],
        ).df()
liq_roll_days = crossover["roll_day"].to_numpy()
# the crossover sits a fixed number of days before each expiry, by construction of the ramp
assert np.array_equal(liq_roll_days, expiries[:-1] - LIQ_ROLL_DAYS)
lab.record("cf_roll_window", ROLL_WINDOW)
lab.record("cf_liq_roll_days", LIQ_ROLL_DAYS)
lab.record("cf_cal_roll_days", ROLL_CAL_DAYS)
lab.record("cf_roll_gap_days", ROLL_CAL_DAYS - LIQ_ROLL_DAYS)
lab.record("cf_roll1_cal_year", float((expiries[0] - ROLL_CAL_DAYS) / PERIODS_PER_YEAR))
lab.record("cf_roll1_liq_year", float(liq_roll_days[0] / PERIODS_PER_YEAR))

# %% [markdown]
# ## 4 · Back-adjustment vs ratio adjustment: built from the contracts, not asserted
#
# The calendar rule holds each contract until `ROLL_CAL_DAYS` before its own expiry, then rolls.
# Because every roll happens the same number of days before (old contract) and after the start of
# (new contract) its own expiry, the GAP in dollars varies with the spot price, but the RATIO is
# the same constant at every roll — a useful fact this lab checks rather than assumes.

# %%
roll_threshold = expiries - ROLL_CAL_DAYS                      # day the position rolls INTO contract i+1
contract_idx_cal = np.clip(np.searchsorted(roll_threshold, day_cf, side="right"), 0, N_CONTRACTS - 1)
expiry_held = expiries[contract_idx_cal]
tau_held = (expiry_held - day_cf) / PERIODS_PER_YEAR
naive_spliced = spot_cf * np.exp(C_MAIN * tau_held)             # the price of whichever contract is held

ROLL_DAYS_CAL = roll_threshold[:-1]                              # the N-1 roll dates
TAU_OLD = ROLL_CAL_DAYS / PERIODS_PER_YEAR                       # time left on the OLD contract at roll
TAU_NEW = (CONTRACT_DAYS + ROLL_CAL_DAYS) / PERIODS_PER_YEAR      # time left on the NEW contract at roll
gap_ratio_const = float(np.exp(C_MAIN * TAU_NEW) / np.exp(C_MAIN * TAU_OLD))    # same at every roll
gaps = spot_cf[ROLL_DAYS_CAL] * (np.exp(C_MAIN * TAU_NEW) - np.exp(C_MAIN * TAU_OLD))   # varies with spot

n_rolls_after = len(ROLL_DAYS_CAL) - np.searchsorted(ROLL_DAYS_CAL, day_cf, side="right")
cum_gap_suffix = np.concatenate(([0.0], np.cumsum(gaps[::-1])))     # cum_gap_suffix[j] = sum of the last j gaps
back_adjusted = naive_spliced + cum_gap_suffix[n_rolls_after]
ratio_adjusted = naive_spliced * gap_ratio_const ** n_rolls_after

# Anchor: neither series restates today's actual, currently-quoted contract.
assert np.isclose(back_adjusted[-1], naive_spliced[-1], atol=1e-9)
assert np.isclose(ratio_adjusted[-1], naive_spliced[-1], atol=1e-9)

# The promise each method keeps: mark the PREVIOUSLY-held contract at both today and yesterday (no
# actual roll needed for this comparison), and check that back-adjustment reproduces its exact
# dollar change, and ratio-adjustment its exact percentage change — on every day, roll days included.
idx_prev = contract_idx_cal[:-1]
exp_prev = expiries[idx_prev]
held_at_t0 = spot_cf[:-1] * np.exp(C_MAIN * (exp_prev - day_cf[:-1]) / PERIODS_PER_YEAR)
held_at_t1 = spot_cf[1:] * np.exp(C_MAIN * (exp_prev - day_cf[1:]) / PERIODS_PER_YEAR)
assert np.allclose(held_at_t0, naive_spliced[:-1], atol=1e-9)   # sanity: this IS the held contract
econ_dollar_change = held_at_t1 - held_at_t0
econ_gross_return = held_at_t1 / held_at_t0

back_dollar_change = np.diff(back_adjusted)
ratio_gross_return = ratio_adjusted[1:] / ratio_adjusted[:-1]
assert np.allclose(back_dollar_change, econ_dollar_change, atol=1e-7), "back-adjustment must reproduce the true $ P&L"
assert np.allclose(ratio_gross_return, econ_gross_return, atol=1e-9, rtol=1e-9), "ratio-adjustment must reproduce the true % return"

# The naive spliced series, by contrast, really does jump at every roll:
naive_jump_at_roll1 = naive_spliced[ROLL_DAYS_CAL[0]] - naive_spliced[ROLL_DAYS_CAL[0] - 1]
naive_daily_vol = SIGMA_SPOT / np.sqrt(PERIODS_PER_YEAR) * S0_SPOT
lab.record("cf_naive_jump1", float(naive_jump_at_roll1))
lab.record("cf_typical_daily_move", float(naive_daily_vol))
lab.record("cf_jump_vs_daily", float(abs(naive_jump_at_roll1) / naive_daily_vol))
assert abs(naive_jump_at_roll1) > naive_daily_vol   # the "jump" is bigger than an ordinary day's move

# The textbook trade-off, reproduced rather than quoted: a deep, sustained backwardation, rolled
# through many contracts, drives the ADDITIVE (back-adjusted) series negative; the MULTIPLICATIVE
# (ratio-adjusted) series cannot go negative, whatever the curve does.
lab.record("cf_back_min", float(back_adjusted.min()))
lab.record("cf_back_min_year", float(day_cf[np.argmin(back_adjusted)] / PERIODS_PER_YEAR))
lab.record("cf_ratio_min", float(ratio_adjusted.min()))
assert back_adjusted.min() < 0, "expected the deep-backwardation case to drive back-adjustment negative"
assert (ratio_adjusted > 0).all(), "ratio-adjustment must never go negative"

lab.record("cf_years", YEARS_CF)
lab.record("cf_n_contracts", int(N_CONTRACTS))
lab.record("cf_contract_days", CONTRACT_DAYS)
lab.record("cf_c_main", C_MAIN)
lab.record("cf_gap_ratio_const", gap_ratio_const)
lab.record("cf_naive_final", float(naive_spliced[-1]))
lab.record("cf_back_final", float(back_adjusted[-1]))
lab.record("cf_ratio_final", float(ratio_adjusted[-1]))
lab.record("cf_ratio_cagr", cagr(ratio_adjusted, YEARS_CF))
lab.record("cf_spot_cagr", cagr(spot_cf, YEARS_CF))
lab.record("cf_cagr_gap_pp", 100 * (lab.results["cf_ratio_cagr"] - lab.results["cf_spot_cagr"]))

# The exact identity behind fig 2.4.5, reproduced with real per-contract data and two adjustment
# conventions instead of one idealised curve: the ratio-adjusted series' total log growth is the
# spot's log growth minus c per year, exactly — no matter how many contracts or rolls sit between.
log_gap = np.log(ratio_adjusted[-1] / ratio_adjusted[0]) - np.log(spot_cf[-1] / spot_cf[0])
assert abs(log_gap - (-C_MAIN * YEARS_CF)) < 1e-9
print(f"back-adjusted low: ${back_adjusted.min():,.2f} (year {lab.results['cf_back_min_year']:.1f}) · "
      f"ratio-adjusted low: ${ratio_adjusted.min():,.2f} · both end at ${naive_spliced[-1]:,.2f}")

years_ax_cf = day_cf / PERIODS_PER_YEAR
lab.chart("continuous_series", charts.line_chart(
    [charts.Series("naive spliced (front contract)", pd.Series(naive_spliced, index=years_ax_cf),
                  role="benchmark", end_label="naive"),
     charts.Series("back-adjusted", pd.Series(back_adjusted, index=years_ax_cf), role="strategy"),
     charts.Series("ratio-adjusted", pd.Series(ratio_adjusted, index=years_ax_cf), role="alt1")],
    title="Three ways to splice the same contracts: naive, back-adjusted, ratio-adjusted (elapsed years)",
    y_fmt=charts.fmt_num(0, prefix="$"), hline=0.0))

# %% [markdown]
# ## 5 · The size of the jump is not noise: a Monte Carlo check
#
# The naive series' jump at a roll is `spot(roll) x a fixed constant` (§4) — so its average size
# across many paths is predictable in closed form, not a random nuisance. Checked at four standard
# errors, the course's Monte Carlo standard (never three, never a seed that happens to pass).

# %%
N_PATHS_MC = 4_000
z_mc = rng_mc.standard_normal((ROLL_DAYS_CAL[0], N_PATHS_MC))
dt = 1.0 / PERIODS_PER_YEAR
log_paths = np.vstack([np.zeros(N_PATHS_MC),
                       np.cumsum((MU_SPOT - 0.5 * SIGMA_SPOT ** 2) * dt + SIGMA_SPOT * np.sqrt(dt) * z_mc, axis=0)])
spot_at_roll1 = S0_SPOT * np.exp(log_paths[-1])                 # spot(roll_1) across many independent paths
gap_samples = spot_at_roll1 * (np.exp(C_MAIN * TAU_NEW) - np.exp(C_MAIN * TAU_OLD))
theory_mean_gap = S0_SPOT * np.exp(MU_SPOT * ROLL_DAYS_CAL[0] / PERIODS_PER_YEAR) * (np.exp(C_MAIN * TAU_NEW) - np.exp(C_MAIN * TAU_OLD))
se_gap = float(gap_samples.std(ddof=1) / np.sqrt(N_PATHS_MC))
lab.record("cf_mc_n_paths", N_PATHS_MC)
lab.record("cf_mc_mean_gap", float(gap_samples.mean()))
lab.record("cf_mc_theory_gap", float(theory_mean_gap))
lab.record("cf_mc_se_gap", se_gap)
assert abs(gap_samples.mean() - theory_mean_gap) < 4 * se_gap
print(f"mean roll-1 jump over {N_PATHS_MC:,} paths: ${gap_samples.mean():,.3f} "
      f"(theory ${theory_mean_gap:,.3f}, SE ${se_gap:,.3f})")

# %%
lab.save()
