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
# # 6.4 · The Bias Hall of Fame — companion lab
#
# **Quant Notebook** · Unit 6 · Session 4 · [Read the session](https://georgevrs.github.io/quant-notebook/unit06-backtesting-discipline/session04-bias-hall-of-fame.html)
#
# Live demos of every classic way a backtest lies — look-ahead, survivorship, data snooping, selection, stale and untradable prices.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# **The method, repeated seven times.** For every bias below we build a synthetic world where we
# KNOW the truth (a strategy or an asset with zero, or a known small, true edge), run the "buggy"
# backtest a real researcher might accidentally write, then run the "fixed" version — and look at
# the gap. Every exhibit uses excess returns with a risk-free rate of 0, stated once here.

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
from quantnb import charts, synth
from quantnb.charts import fmt_num, fmt_pct
from quantnb.stats import annual_sharpe, expected_max_sharpe

lab = qn.Lab("6.4")  # seeds the random generators: every run gives the same numbers
rng = lab.rng

# One independent random stream per exhibit, so trying one exhibit's Try-it edits never
# changes another exhibit's numbers.
(rng_surv, rng_select, rng_look, rng_snoop, rng_stale, rng_untrade, rng_pub) = rng.spawn(7)

# %% [markdown]
# ## Stage 1 — the universe: survivorship bias and selection bias
#
# ### Exhibit A · survivorship bias (defined and owned by Session 4.2 — shown live here)
#
# 5,000 synthetic stocks, 20 years, zero true average edge (`mu=0`) and realistic single-name
# volatility. Some random-walk down far enough that, in real markets, they would have been
# delisted. The BUGGY backtest pulls "today's index members" and backfills their full history —
# exactly what happens when a data vendor's default table is "current constituents only"
# (Session 4.2's point-in-time data). The FIXED backtest honestly includes every name that ever
# existed, delisted or not.

# %%
SURV_N, SURV_YEARS = 5_000, 20
surv_px = synth.gbm_prices(SURV_YEARS * 252, mu=0.0, sigma=0.35, n_paths=SURV_N, rng=rng_surv)
surv_total_ret = (surv_px.iloc[-1] / surv_px.iloc[0] - 1.0).to_numpy()
surv_failed = surv_total_ret < -0.80                      # "delisted": down more than 80% ever since
surv_buggy_mean = float(surv_total_ret[~surv_failed].mean())   # today's survivors only, backfilled
surv_honest_mean = float(surv_total_ret.mean())                # everyone who ever existed

lab.record("surv_n_assets", SURV_N)
lab.record("surv_years", SURV_YEARS)
lab.record("surv_pct_failed", float(surv_failed.mean()))
lab.record("surv_buggy_mean", surv_buggy_mean)
lab.record("surv_honest_mean", surv_honest_mean)
print(f"survivorship: {surv_failed.mean():.1%} 'delisted' · survivors-only mean {surv_buggy_mean:.2%} "
      f"vs honest mean {surv_honest_mean:.2%}")

# the honest, full-population mean of a zero-true-edge universe must sit near zero
surv_se = float(surv_total_ret.std(ddof=1) / np.sqrt(SURV_N))
assert abs(surv_honest_mean) < 4 * surv_se, "honest survivorship mean should be statistically indistinguishable from 0"
assert surv_buggy_mean > surv_honest_mean + 4 * surv_se, "dropping the failures must inflate the mean"

# %% [markdown]
# ### Exhibit B · selection bias (owned by this session)
#
# 1,000 candidate assets, 10 years, zero true edge, independent noise (no failures involved — this
# is a *different* mechanism from survivorship). The BUGGY step ranks all 1,000 by their OWN
# realised Sharpe ratio and reports the equal-weight average of the best 20 — "our screen found
# these 20 winners." The FIXED step re-measures the SAME 20 names on a fresh, independent draw
# (an honest out-of-sample check), which is what a purged validation scheme (Session 6.6) forces
# you to do.

# %%
SEL_N, SEL_YEARS, SEL_K = 1_000, 10, 20
sel_daily_vol = 0.20 / np.sqrt(252)
sel_days = SEL_YEARS * 252
sel_excess_is = rng_select.standard_normal((sel_days, SEL_N)) * sel_daily_vol
sel_sharpe_is = annual_sharpe(sel_excess_is, axis=0)          # 1,000 noisy Sharpe estimates, true value 0
sel_top_idx = np.argsort(sel_sharpe_is)[-SEL_K:]
sel_buggy_sharpe = float(sel_sharpe_is[sel_top_idx].mean())   # the in-sample "winners"

sel_excess_oos = rng_select.standard_normal((sel_days, SEL_N)) * sel_daily_vol   # fresh draw, same 1,000 names
sel_fixed_sharpe = float(annual_sharpe(sel_excess_oos[:, sel_top_idx], axis=0).mean())

sel_best_single = float(sel_sharpe_is.max())
sel_theory_max = float(expected_max_sharpe(SEL_N, SEL_YEARS))

lab.record("sel_n_candidates", SEL_N)
lab.record("sel_years", SEL_YEARS)
lab.record("sel_k", SEL_K)
lab.record("sel_buggy_sharpe", sel_buggy_sharpe)
lab.record("sel_fixed_sharpe", sel_fixed_sharpe)
lab.record("sel_best_single_sharpe", sel_best_single)
lab.record("sel_theory_max_sharpe", sel_theory_max)
print(f"selection: top-{SEL_K} in-sample Sharpe {sel_buggy_sharpe:.2f} · same {SEL_K} out-of-sample {sel_fixed_sharpe:.2f} "
      f"· theory E[max of {SEL_N}] {sel_theory_max:.2f} vs observed max {sel_best_single:.2f}")

# on FRESH data the cherry-picked names have no real edge: mean of K iid Sharpe(0, 1/years) draws
sel_fixed_se = (1.0 / np.sqrt(SEL_YEARS)) / np.sqrt(SEL_K)
assert abs(sel_fixed_sharpe) < 4 * sel_fixed_se, "the same 20 names, on fresh data, must look like noise"
assert sel_buggy_sharpe > 4 * sel_fixed_se, "cherry-picking on the same data must look far better than noise"
# the single best Sharpe is a noisy draw of an extreme-value statistic: order-of-magnitude check only
assert sel_best_single > 0.5 * sel_theory_max

# %% [markdown]
# ## Stage 2 — signal timing: look-ahead bias and data snooping
#
# ### Exhibit C · look-ahead bias (defined and owned by Session 4.2; the bug itself is a missing
# signal lag, owned by Session 6.2)

# %%
LOOK_YEARS = 5
look_days = LOOK_YEARS * 252
look_daily_vol = 0.20 / np.sqrt(252)
look_returns = rng_look.standard_normal(look_days) * look_daily_vol   # a fair-game market, zero autocorrelation

look_buggy_pnl = np.sign(look_returns) * look_returns                  # trades the SAME return that built the signal
look_fixed_pnl = np.sign(look_returns[:-1]) * look_returns[1:]         # signal lagged by one period, as it must be

look_buggy_sharpe = float(annual_sharpe(look_buggy_pnl))
look_fixed_sharpe = float(annual_sharpe(look_fixed_pnl))
lab.record("look_years", LOOK_YEARS)
lab.record("look_buggy_sharpe", look_buggy_sharpe)
lab.record("look_fixed_sharpe", look_fixed_sharpe)
print(f"look-ahead: unlagged Sharpe {look_buggy_sharpe:.1f} vs correctly lagged {look_fixed_sharpe:.2f}")

assert look_buggy_pnl.min() >= 0, "trading the same return you signalled on can never lose money — that IS the tell"
assert abs(look_fixed_sharpe) < 4 / np.sqrt(LOOK_YEARS), "a properly lagged sign-of-yesterday rule on IID returns has no edge"

# %% [markdown]
# ### Exhibit D · data snooping (owned by this session)
#
# One fixed, zero-edge synthetic market. 3,000 independent "seeds," each just a random ±1 position
# rule with no real signal. We search the first 5 years for the seed with the best Sharpe ratio
# (this is the exact seed-hunting this course's own labs are forbidden from doing — here it is, on
# purpose), then walk the SAME rule forward onto 10 fresh years it never saw.

# %%
SNOOP_N_SEEDS, SNOOP_YEARS_SEARCH, SNOOP_YEARS_HOLD = 3_000, 5, 10
snoop_daily_vol = 0.20 / np.sqrt(252)
snoop_days_search = SNOOP_YEARS_SEARCH * 252
snoop_days_hold = SNOOP_YEARS_HOLD * 252
snoop_days_total = snoop_days_search + snoop_days_hold

snoop_market = rng_snoop.standard_normal(snoop_days_total) * snoop_daily_vol      # one continuous, honest market
snoop_positions = rng_snoop.choice([-1.0, 1.0], size=(snoop_days_total, SNOOP_N_SEEDS))  # 3,000 skill-less rules

snoop_search_pnl = snoop_positions[:snoop_days_search, :] * snoop_market[:snoop_days_search, None]
snoop_sharpe_search = annual_sharpe(snoop_search_pnl, axis=0)   # 3,000 Sharpe estimates, true value 0
snoop_best_idx = int(np.argmax(snoop_sharpe_search))
snoop_best_seed_sharpe = float(snoop_sharpe_search[snoop_best_idx])
snoop_mean_seed_sharpe = float(snoop_sharpe_search.mean())

snoop_hold_pnl = snoop_positions[snoop_days_search:, snoop_best_idx] * snoop_market[snoop_days_search:]
snoop_holdout_sharpe = float(annual_sharpe(snoop_hold_pnl))
snoop_theory_max = float(expected_max_sharpe(SNOOP_N_SEEDS, SNOOP_YEARS_SEARCH))

lab.record("snoop_n_seeds", SNOOP_N_SEEDS)
lab.record("snoop_years_search", SNOOP_YEARS_SEARCH)
lab.record("snoop_years_hold", SNOOP_YEARS_HOLD)
lab.record("snoop_best_seed_sharpe", snoop_best_seed_sharpe)
lab.record("snoop_holdout_sharpe", snoop_holdout_sharpe)
lab.record("snoop_theory_max_sharpe", snoop_theory_max)
print(f"data snooping: best of {SNOOP_N_SEEDS} seeds (in-sample) Sharpe {snoop_best_seed_sharpe:.2f} "
      f"(theory E[max] {snoop_theory_max:.2f}) · same rule, held out {SNOOP_YEARS_HOLD}y, Sharpe {snoop_holdout_sharpe:.2f}")

snoop_se_mean = float(snoop_sharpe_search.std(ddof=1) / np.sqrt(SNOOP_N_SEEDS))
assert abs(snoop_mean_seed_sharpe) < 4 * snoop_se_mean, "averaged honestly across all seeds, the edge is exactly what it should be: none"
assert snoop_best_seed_sharpe > 0.5 * snoop_theory_max, "the winner of a big search should look like a real strategy"
assert abs(snoop_holdout_sharpe) < 4 / np.sqrt(SNOOP_YEARS_HOLD), "the 'winning' rule has no real edge once it meets new data"

snoop_xs = np.linspace(snoop_sharpe_search.min(), snoop_sharpe_search.max(), 200)
snoop_sd_theory = 1.0 / np.sqrt(SNOOP_YEARS_SEARCH)
snoop_ys = norm.pdf(snoop_xs, loc=0.0, scale=snoop_sd_theory)
lab.chart("snoop_hist", charts.histogram(
    snoop_sharpe_search, title="3,000 skill-less seeds: in-sample Sharpe ratio", bins=50,
    x_fmt=fmt_num(1), density_overlay=(snoop_xs, snoop_ys), overlay_label="theory if no edge exists anywhere"))

# %% [markdown]
# ## Stage 3 — execution: stale price and untradable price (both owned by this session)
#
# ### Exhibit E · stale price
#
# A single asset with a KNOWN small true edge (`mu=5%`, `sigma=20%`, so the true Sharpe is
# `0.05/0.20 = 0.25`). The "observed" mark only partly catches up each day
# (`observed_t = θ·true_t + (1-θ)·observed_{t-1}`, θ=0.3) — the standard model of a stale or
# smoothed quote in an illiquid name (Getmansky, Lo & Makarov, 2004). The BUGGY backtest computes
# Sharpe on the daily observed marks. The FIXED version aggregates to monthly, giving the staleness
# time to catch up.

# %%
STALE_YEARS, STALE_MU, STALE_SIGMA, STALE_THETA = 10, 0.05, 0.20, 0.30
stale_days = STALE_YEARS * 252
stale_true_ret = rng_stale.normal(STALE_MU / 252, STALE_SIGMA / np.sqrt(252), stale_days)
stale_observed_ret = pd.Series(stale_true_ret).ewm(alpha=STALE_THETA, adjust=False).mean().to_numpy()

stale_true_sharpe = float(annual_sharpe(stale_true_ret))
stale_naive_sharpe = float(annual_sharpe(stale_observed_ret))

stale_obs_price = (1.0 + pd.Series(stale_observed_ret)).cumprod()
stale_monthly_ret = stale_obs_price.iloc[::21].pct_change().dropna().to_numpy()
stale_fixed_sharpe = float(annual_sharpe(stale_monthly_ret, periods_per_year=12))

stale_vol_ratio = float(stale_observed_ret.std(ddof=1) / stale_true_ret.std(ddof=1))

lab.record("stale_years", STALE_YEARS)
lab.record("stale_theta", STALE_THETA)
lab.record("stale_true_sharpe", stale_true_sharpe)
lab.record("stale_naive_sharpe", stale_naive_sharpe)
lab.record("stale_fixed_sharpe", stale_fixed_sharpe)
lab.record("stale_vol_ratio", stale_vol_ratio)
print(f"stale price: true Sharpe {stale_true_sharpe:.2f} · naive (daily, stale) {stale_naive_sharpe:.2f} "
      f"· fixed (monthly) {stale_fixed_sharpe:.2f} · observed/true vol ratio {stale_vol_ratio:.2f}")

stale_sharpe_se = np.sqrt(1.0 / STALE_YEARS)  # SE of an annualised Sharpe estimated from STALE_YEARS of data (SR~0)
assert abs(stale_true_sharpe - STALE_MU / STALE_SIGMA) < 4 * stale_sharpe_se
assert stale_naive_sharpe > stale_true_sharpe, "smoothing must shrink measured volatility and inflate the naive Sharpe"

# %% [markdown]
# ### Exhibit F · untradable price
#
# A single thinly-traded synthetic name and a classic breakout rule: go long when today's close
# breaks above the prior 20-day high. The BUGGY backtest assumes every breakout fills exactly at
# that high, with certainty. In reality a day's high is often a single, sizeless print — the FIXED
# version assumes 40% of those prints were never really tradable (no fill at all) and, when a fill
# does happen, it costs a bit more than the printed level (Fama & Blume, 1966, showed exactly this
# gap for filter rules).

# %%
UNTRADE_YEARS, UNTRADE_HOLD, UNTRADE_PHANTOM_P = 10, 10, 0.40
untrade_days = UNTRADE_YEARS * 252
untrade_close = synth.gbm_prices(untrade_days, mu=0.0, sigma=0.30, rng=rng_untrade)["price"]
untrade_high = untrade_close * (1.0 + np.abs(rng_untrade.normal(0.0, 0.01, len(untrade_close))))
untrade_trigger = untrade_high.shift(1).rolling(20).max()          # yesterday-and-earlier only: no look-ahead here
untrade_signal = (untrade_close > untrade_trigger) & untrade_trigger.notna()
untrade_sig_idx = np.flatnonzero(untrade_signal.to_numpy())
untrade_sig_idx = untrade_sig_idx[untrade_sig_idx + UNTRADE_HOLD < len(untrade_close)]

untrade_entry_buggy = untrade_trigger.to_numpy()[untrade_sig_idx]                       # filled AT the print, always
untrade_exit = untrade_close.to_numpy()[untrade_sig_idx + UNTRADE_HOLD]
untrade_buggy_trades = untrade_exit / untrade_entry_buggy - 1.0

untrade_phantom = rng_untrade.random(len(untrade_sig_idx)) < UNTRADE_PHANTOM_P          # the print was never really there
untrade_slip = np.abs(rng_untrade.normal(0.0, 0.005, len(untrade_sig_idx)))             # a real fill costs more
untrade_entry_fixed = untrade_entry_buggy * (1.0 + untrade_slip)
untrade_fixed_trades_all = untrade_exit / untrade_entry_fixed - 1.0
untrade_fixed_trades = untrade_fixed_trades_all[~untrade_phantom]                       # phantom prints: no trade

lab.record("untrade_n_signals", int(len(untrade_sig_idx)))
lab.record("untrade_pct_phantom", float(untrade_phantom.mean()))
lab.record("untrade_buggy_mean_ret", float(untrade_buggy_trades.mean()))
lab.record("untrade_fixed_mean_ret", float(untrade_fixed_trades.mean()))
lab.record("untrade_buggy_sharpe", float(annual_sharpe(untrade_buggy_trades, periods_per_year=1)))
lab.record("untrade_fixed_sharpe", float(annual_sharpe(untrade_fixed_trades, periods_per_year=1)))
print(f"untradable price: {len(untrade_sig_idx)} signals, {untrade_phantom.mean():.0%} phantom · "
      f"buggy mean trade {untrade_buggy_trades.mean():.2%} vs fixed {untrade_fixed_trades.mean():.2%}")

assert untrade_fixed_trades.mean() < untrade_buggy_trades.mean(), "a worse, less-certain fill must look worse, not better"
assert abs(untrade_fixed_trades.mean()) < 0.03, "a breakout rule on a random walk, honestly filled, is close to a coin flip"

# %% [markdown]
# ## Stage 4 — reporting: publication bias (owned by this session)
#
# 20,000 independent research desks each test ONE honest, zero-edge strategy on their own data —
# no snooping anywhere, every result is a fair draw. Only the ones that clear a "statistically
# significant" bar (here, roughly a t-stat of 2) ever get published, pitched to an investor, or
# survive to the next performance review. The rest sit in the file drawer (Rosenthal, 1979).

# %%
PUB_N, PUB_YEARS, PUB_TAU = 20_000, 8, 2.0
pub_sd = 1.0 / np.sqrt(PUB_YEARS)                      # SE of a single Sharpe estimate over PUB_YEARS, true SR = 0
pub_sharpe_all = rng_pub.standard_normal(PUB_N) * pub_sd
pub_threshold = PUB_TAU * pub_sd
pub_published = pub_sharpe_all > pub_threshold
pub_frac_published = float(pub_published.mean())
pub_published_mean = float(pub_sharpe_all[pub_published].mean())
pub_true_mean = float(pub_sharpe_all.mean())

pub_mills_ratio = float(norm.pdf(PUB_TAU) / (1.0 - norm.cdf(PUB_TAU)))   # inverse Mills ratio: E[Z | Z>tau], Z~N(0,1)
pub_theory_published_mean = pub_mills_ratio * pub_sd
pub_theory_frac = float(1.0 - norm.cdf(PUB_TAU))

lab.record("pub_n_researchers", PUB_N)
lab.record("pub_years", PUB_YEARS)
lab.record("pub_frac_published", pub_frac_published)
lab.record("pub_published_mean", pub_published_mean)
lab.record("pub_true_mean", pub_true_mean)
print(f"publication bias: {pub_frac_published:.1%} published · published-average Sharpe {pub_published_mean:.2f} "
      f"vs true population average {pub_true_mean:.3f} (theory {pub_theory_published_mean:.2f})")

pub_se_mean = float(pub_sharpe_all.std(ddof=1) / np.sqrt(PUB_N))
assert abs(pub_true_mean) < 4 * pub_se_mean, "the honest average across every desk, published or not, must be ~0"
pub_se_published = float(pub_sharpe_all[pub_published].std(ddof=1) / np.sqrt(pub_published.sum()))
assert abs(pub_published_mean - pub_theory_published_mean) < 4 * pub_se_published
pub_se_frac = float(np.sqrt(pub_theory_frac * (1 - pub_theory_frac) / PUB_N))
assert abs(pub_frac_published - pub_theory_frac) < 4 * pub_se_frac

# %% [markdown]
# ## The hall of fame, side by side

# %%
hof_categories = ["Selection bias", "Data snooping", "Stale price", "Publication bias"]
hof_reported = [sel_buggy_sharpe, snoop_best_seed_sharpe, stale_naive_sharpe, pub_published_mean]
hof_true = [sel_fixed_sharpe, snoop_holdout_sharpe, stale_true_sharpe, pub_true_mean]
lab.chart("hall_of_fame", charts.grouped_column_chart(
    hof_categories,
    [("What got reported", hof_reported, "loss"), ("What was actually true", hof_true, "strategy")],
    title="Sharpe ratio: what got reported vs what was actually true", y_fmt=fmt_num(2)))

# %%
lab.save()
