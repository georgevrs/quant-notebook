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
# # 5.6 · Cointegration & Mean Reversion — companion lab
#
# **Quant Notebook** · Unit 5 · Session 6 · [Read the session](https://georgevrs.github.io/quant-notebook/unit05-statistics-time-series/session06-cointegration-mean-reversion.html)
#
# Engle-Granger and Johansen tests, hedge ratios, spreads, the Ornstein-Uhlenbeck process as a continuous AR(1), and half-lives.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# This lab is about recovering *statistical relationships* from simulated price levels — a hedge
# ratio, a cointegrating rank, a mean-reversion speed — not about P&L, so there is no Sharpe ratio
# and no risk-free rate anywhere in it. All series are synthetic; every Monte Carlo check is made
# at 4 standard errors (never 3, never seed-hunted). Simulated time is elapsed trading days, never
# calendar dates.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# statsmodels' Johansen implementation casts an intermediate eigenvalue array from complex to real
# (the imaginary parts are numerical noise, always ~0 for these well-behaved systems) — harmless,
# but noisy; silence just that one warning rather than all warnings.
warnings.filterwarnings("ignore", message="Casting complex values")

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR, in_years

lab = qn.Lab("5.6")  # seeds the random generators: every run gives the same numbers
# One independent random stream per experiment (Session 1.5): editing or re-running one section
# never shifts the random numbers another section sees.
(rng_pair, rng_eg_mc, rng_asym, rng_johansen, rng_hl, rng_pipe) = lab.rng.spawn(6)

N_SE = 4                                    # every Monte Carlo check is made at 4 standard errors
MC_CHECKS: list[tuple[str, float]] = []     # (name, z-score) of every statistical check


def check_mc(name: str, estimate: float, truth: float, se: float, n_se: float = N_SE,
             allowance: float = 0.0) -> float:
    """Assert a Monte Carlo estimate lies within n_se standard errors of its theoretical value.

    `allowance` widens the tolerance by a KNOWN, documented approximation error (never used to
    rescue noise). Returns the z-score and remembers it.
    """
    z = (estimate - truth) / se
    MC_CHECKS.append((name, float(z)))
    assert abs(estimate - truth) < n_se * se + allowance, (
        f"{name}: {estimate:.6g} vs theory {truth:.6g} is {z:+.2f} SE away")
    return float(z)


def se_prop(p: float, n: int) -> float:
    """Standard error of a simulated probability (a proportion of n independent trials)."""
    return float(np.sqrt(p * (1.0 - p) / n))


def common_trend(n: int, sigma_w: float, rng: np.random.Generator, n_paths: int = 1) -> np.ndarray:
    """A driftless random walk (shape n x n_paths): the shared non-stationary factor."""
    z = rng.standard_normal((n - 1, n_paths))
    return np.vstack([np.zeros((1, n_paths)), np.cumsum(sigma_w * z, axis=0)])


def ou_paths(n: int, kappa: float, stationary_sd: float, rng: np.random.Generator,
             n_paths: int = 1, x0: float = 0.0) -> np.ndarray:
    """Exact discrete-time Ornstein-Uhlenbeck paths (shape n x n_paths), one step = one period.

    x[t] = phi * x[t-1] + innov, phi = exp(-kappa), innov ~ N(0, stationary_sd^2 (1 - phi^2)).
    This is the EXACT transition (Session 3.5's lesson: prefer it to an Euler discretisation),
    so the process has the right stationary variance at every step, not just in the limit.
    """
    phi = np.exp(-kappa)
    innov_sd = stationary_sd * np.sqrt(1.0 - phi ** 2)
    x = np.empty((n, n_paths))
    x[0] = x0
    z = rng.standard_normal((n - 1, n_paths))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + innov_sd * z[t - 1]
    return x


def estimate_halflife(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised OLS AR(1) fit of each column of x (shape n x n_paths): x_t = a + phi x_{t-1} + e.

    Returns (halflife_hat, kappa_hat, phi_hat), one value per path. This is Chan's regression
    method (regress the CHANGE on the lagged level) written as a level-on-level AR(1) fit — the
    two are algebraically identical (phi = 1 + the change-regression slope).
    """
    x0, x1 = x[:-1], x[1:]
    x0c = x0 - x0.mean(axis=0)
    x1c = x1 - x1.mean(axis=0)
    phi_hat = (x0c * x1c).sum(axis=0) / (x0c ** 2).sum(axis=0)
    phi_hat = np.clip(phi_hat, 1e-6, 1 - 1e-9)   # a phi >= 1 estimate means "no detectable reversion"
    kappa_hat = -np.log(phi_hat)
    halflife_hat = np.log(2.0) / kappa_hat
    return halflife_hat, kappa_hat, phi_hat


lab.record("n_se", N_SE)

# %% [markdown]
# ## 1 · Simulating a cointegrated pair: a shared trend, a private wobble
#
# Build two price series that individually wander forever, but whose difference does not. Each
# series is the SAME random walk `W` (the shared, non-stationary factor) scaled by its own
# exposure, plus — for one of the two — a mean-reverting Ornstein-Uhlenbeck "wobble" `eta`. The
# true hedge ratio is `H_TRUE`; the true half-life of the wobble is `HL_TRUE` trading days.

# %%
N_MAIN = 10 * PERIODS_PER_YEAR              # 10 years of daily levels
SIGMA_W = 0.35                              # daily SD of the shared random walk
H_TRUE = 1.5                                # true hedge ratio: A tracks H_TRUE times the trend
HL_TRUE = 20.0                              # true half-life of the spread, in trading days
KAPPA_TRUE = np.log(2.0) / HL_TRUE          # per-trading-day mean-reversion speed
SD_ETA = 3.0                                # stationary SD of the wobble (index points)
A0, B0 = 100.0, 50.0                        # starting levels (arbitrary index units, not $ prices)

w_hero = common_trend(N_MAIN, SIGMA_W, rng_pair)[:, 0]
eta_hero = ou_paths(N_MAIN, KAPPA_TRUE, SD_ETA, rng_pair)[:, 0]
a_hero = A0 + H_TRUE * w_hero + eta_hero
b_hero = B0 + w_hero
# B carries none of the wobble, so A - H_TRUE*B is EXACTLY eta_hero: the whole gap between the two
# series is the mean-reverting spread. Section 2's "try it" adds noise to B as well.
spread_true = a_hero - H_TRUE * b_hero
assert np.allclose(spread_true, eta_hero + (A0 - H_TRUE * B0))

lab.record("h_true", H_TRUE)
lab.record("hl_true", HL_TRUE)
lab.record("kappa_true", KAPPA_TRUE)
lab.record("sigma_w", SIGMA_W)
lab.record("sd_eta", SD_ETA)
lab.record("n_main_years", N_MAIN // PERIODS_PER_YEAR)
lab.record("a_final", float(a_hero[-1]))
lab.record("b_final", float(b_hero[-1]))
lab.record("a_start", A0)
lab.record("b_start", B0)
print(f"after {N_MAIN // PERIODS_PER_YEAR} years: A = {a_hero[-1]:.1f} (from {A0:.0f}), "
      f"B = {b_hero[-1]:.1f} (from {B0:.0f}) — both wandered, but A - {H_TRUE}*B stayed near "
      f"{eta_hero.mean() + A0 - H_TRUE * B0:.1f}")

# %% [markdown]
# ## 2 · The Engle-Granger two-step test
#
# **Step 1 — regress.** OLS of A on B (with an intercept) estimates the hedge ratio: the slope
# that makes the residual as small as possible. **Step 2 — test the residual.** Run the
# augmented Dickey-Fuller test (Session 5.4) on that residual. If it rejects a unit root, the
# residual is stationary and the pair is cointegrated — and the residual itself IS the spread you
# would trade.

# %%
ols_hero = sm.OLS(a_hero, sm.add_constant(b_hero)).fit()
alpha_hat, h_hat = ols_hero.params
resid_hero = ols_hero.resid.to_numpy() if hasattr(ols_hero.resid, "to_numpy") else ols_hero.resid
adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(resid_hero, autolag="AIC", result_object=False)
coint_stat, coint_p, coint_crit = coint(a_hero, b_hero, trend="c")

lab.record("h_hat_hero", float(h_hat))
lab.record("alpha_hat_hero", float(alpha_hat))
lab.record("h_hat_error_pct", 100 * (h_hat - H_TRUE) / H_TRUE)
lab.record("adf_stat_hero", float(adf_stat))
lab.record("adf_p_hero", float(adf_p))
lab.record("adf_crit5_hero", float(adf_crit["5%"]))
lab.record("coint_stat_hero", float(coint_stat))
lab.record("coint_p_hero", float(coint_p))
lab.record("coint_crit5_hero", float(coint_crit[1]))  # MacKinnon's cointegration-adjusted 5% value
lab.record("r2_hero", float(ols_hero.rsquared))
print(f"step 1: h_hat = {h_hat:.4f} (true {H_TRUE}) · step 2: ADF stat {adf_stat:.2f}, "
      f"p = {adf_p:.2e} (5% critical value {adf_crit['5%']:.2f}) · statsmodels coint() agrees: "
      f"stat {coint_stat:.2f}, p = {coint_p:.2e}")
assert adf_p < 0.01 and coint_p < 0.01, "the two-step test should reject the unit root here"
assert abs(h_hat - H_TRUE) < 0.05, "10 years of daily data should pin the hedge ratio down tightly"

# %% [markdown]
# **Size and power.** Does the test actually discriminate — rejecting rarely when there is no
# cointegration, and reliably when there is? Repeat both scenarios (no shared trend at all, vs.
# the construction above with a shorter, 3-year sample) many times and count rejections at 5%.
# Along the way, record the R² of the plain OLS regression under the NULL: two independent random
# walks routinely look highly "related" by R² alone — the classic spurious-regression trap
# (Granger & Newbold, 1974) that the second step of the test exists to catch.

# %%
N_EG, N_REPS_EG = 3 * PERIODS_PER_YEAR, 400
rng_null, rng_alt = rng_eg_mc.spawn(2)

w_null = common_trend(N_EG, SIGMA_W, rng_null, N_REPS_EG)          # NO shared trend used below
c_null = A0 + common_trend(N_EG, SIGMA_W, rng_null, N_REPS_EG)     # independent random walk
d_null = B0 + common_trend(N_EG, SIGMA_W, rng_null, N_REPS_EG)     # independent random walk
w_alt = common_trend(N_EG, SIGMA_W, rng_alt, N_REPS_EG)
eta_alt = ou_paths(N_EG, KAPPA_TRUE, SD_ETA, rng_alt, N_REPS_EG)
a_alt = A0 + H_TRUE * w_alt + eta_alt
b_alt = B0 + w_alt

rej_null, rej_alt, r2_null = [], [], []
for i in range(N_REPS_EG):
    _, p_null, _ = coint(c_null[:, i], d_null[:, i], trend="c", autolag=None, maxlag=1)
    _, p_alt, _ = coint(a_alt[:, i], b_alt[:, i], trend="c", autolag=None, maxlag=1)
    rej_null.append(p_null < 0.05)
    rej_alt.append(p_alt < 0.05)
    r2_null.append(sm.OLS(c_null[:, i], sm.add_constant(d_null[:, i])).fit().rsquared)
size_hat = float(np.mean(rej_null))
power_hat = float(np.mean(rej_alt))
r2_null_mean = float(np.mean(r2_null))

lab.record("eg_mc_reps", N_REPS_EG)
lab.record("eg_mc_years", N_EG // PERIODS_PER_YEAR)
lab.record("eg_size", size_hat)
lab.record("eg_size_se", se_prop(size_hat, N_REPS_EG))
lab.record("eg_power", power_hat)
lab.record("eg_power_se", se_prop(power_hat, N_REPS_EG))
lab.record("eg_r2_null_mean", r2_null_mean)
lab.record("eg_r2_null_high_share", float(np.mean(np.array(r2_null) > 0.5)))
print(f"under the null (no shared trend): {size_hat:.1%} false-positive rate at 5% "
      f"(mean R² of the naive regression: {r2_null_mean:.2f}) · "
      f"under true cointegration: {power_hat:.1%} correctly rejected")
check_mc("EG size ≈ nominal 5%", size_hat, 0.05, se_prop(0.05, N_REPS_EG), allowance=0.01)
assert power_hat > 0.75, "3 years should catch most 20-day half-life spreads"
assert r2_null_mean > 0.15, "two independent random walks should still look 'related' by R² alone"

# %% [markdown]
# **The asymmetry pitfall.** Engle-Granger picks one series as the dependent variable, and B here
# carries none of the idiosyncratic wobble (section 1) — A is the "noisy" leg, B the "clean" one.
# Regressing the noisy series on the clean one is the right way around; the other way is a classic
# errors-in-variables problem, attenuating the slope and costing power. With ten years of data the
# effect washes out (section 2's hero regression barely cares which way round it runs); with a
# SHORT, one-year sample it can decide the outcome. Test both directions on each replication.

# %%
N_WEAK, HL_WEAK, N_REPS_WEAK = PERIODS_PER_YEAR, 10.0, 400
KAPPA_WEAK = np.log(2.0) / HL_WEAK
w_weak = common_trend(N_WEAK, SIGMA_W, rng_asym, N_REPS_WEAK)
eta_weak = ou_paths(N_WEAK, KAPPA_WEAK, SD_ETA, rng_asym, N_REPS_WEAK)
a_weak = A0 + H_TRUE * w_weak + eta_weak
b_weak = B0 + w_weak

reject_ab, reject_ba = [], []
for i in range(N_REPS_WEAK):
    _, p_ab, _ = coint(a_weak[:, i], b_weak[:, i], trend="c", autolag=None, maxlag=1)
    _, p_ba, _ = coint(b_weak[:, i], a_weak[:, i], trend="c", autolag=None, maxlag=1)
    reject_ab.append(p_ab < 0.05)
    reject_ba.append(p_ba < 0.05)
reject_ab, reject_ba = np.array(reject_ab), np.array(reject_ba)
disagree = float(np.mean(reject_ab != reject_ba))
lab.record("asym_reps", N_REPS_WEAK)
lab.record("asym_hl_weak", HL_WEAK)
lab.record("asym_reject_ab", float(reject_ab.mean()))
lab.record("asym_reject_ba", float(reject_ba.mean()))
lab.record("asym_disagree", disagree)
lab.record("asym_disagree_se", se_prop(disagree, N_REPS_WEAK))
print(f"1-year sample, 10-day half-life: A-on-B (correct direction) rejects {reject_ab.mean():.1%}, "
      f"B-on-A (wrong direction) rejects {reject_ba.mean():.1%} of the time — "
      f"they disagree on {disagree:.1%} of samples")
se_diff = float(np.sqrt(se_prop(reject_ab.mean(), N_REPS_WEAK) ** 2
                        + se_prop(reject_ba.mean(), N_REPS_WEAK) ** 2))
assert reject_ab.mean() > reject_ba.mean() + N_SE * se_diff, "the correct direction should reject more often"
assert disagree > 0.15, "direction should matter on a meaningful share of short, weak samples"

# %% [markdown]
# ## 3 · Beyond two series: the Johansen test
#
# Engle-Granger tests ONE candidate combination at a time. With three series sharing a SINGLE
# common trend, there are TWO independent stationary combinations (the cointegration rank is
# `n_series - n_trends = 3 - 1 = 2`) — Johansen's test finds the rank and every combination in one
# pass, and treats all series symmetrically.

# %%
N_J = 3 * PERIODS_PER_YEAR
HL_J1, HL_J2 = 10.0, 15.0
w_j = common_trend(N_J, SIGMA_W, rng_johansen)[:, 0]
eta_j1 = ou_paths(N_J, np.log(2) / HL_J1, SD_ETA, rng_johansen)[:, 0]
eta_j2 = ou_paths(N_J, np.log(2) / HL_J2, SD_ETA, rng_johansen)[:, 0]
x1 = 100.0 + w_j + eta_j1
x2 = 60.0 + 0.8 * w_j + eta_j2
x3 = 40.0 + 1.3 * w_j                      # no idiosyncratic wobble at all — a pure trend-follower
Y = np.column_stack([x1, x2, x3])

jres = coint_johansen(Y, det_order=0, k_ar_diff=1)
trace_stats = jres.lr1
trace_crit5 = jres.cvt[:, 1]                 # columns are 90% / 95% / 99%
rank_hat = int(np.sum(trace_stats > trace_crit5))   # first r where the test FAILS to reject sets the rank

lab.record("j_years", N_J // PERIODS_PER_YEAR)
lab.record("j_trace_r0", float(trace_stats[0]))
lab.record("j_trace_r1", float(trace_stats[1]))
lab.record("j_trace_r2", float(trace_stats[2]))
lab.record("j_crit5_r0", float(trace_crit5[0]))
lab.record("j_crit5_r1", float(trace_crit5[1]))
lab.record("j_crit5_r2", float(trace_crit5[2]))
lab.record("j_rank_hat", rank_hat)
lab.record("j_rank_true", 2)
print(f"trace stats {np.round(trace_stats, 1)} vs 95% critical values {np.round(trace_crit5, 1)} "
      f"→ recovered rank {rank_hat} (true rank 2)")
assert rank_hat == 2

# Monte Carlo: how often does the trace test recover the true rank of 2?
N_REPS_J = 150
rank_hits = 0
for _ in range(N_REPS_J):
    w_r = common_trend(N_J, SIGMA_W, rng_johansen)[:, 0]
    r1 = ou_paths(N_J, np.log(2) / HL_J1, SD_ETA, rng_johansen)[:, 0]
    r2 = ou_paths(N_J, np.log(2) / HL_J2, SD_ETA, rng_johansen)[:, 0]
    Yr = np.column_stack([100.0 + w_r + r1, 60.0 + 0.8 * w_r + r2, 40.0 + 1.3 * w_r])
    jr = coint_johansen(Yr, det_order=0, k_ar_diff=1)
    rank_hits += int(np.sum(jr.lr1 > jr.cvt[:, 1]) == 2)
rank_hit_rate = rank_hits / N_REPS_J
lab.record("j_mc_reps", N_REPS_J)
lab.record("j_rank_hit_rate", rank_hit_rate)
print(f"Johansen recovered the true rank (2) in {rank_hit_rate:.1%} of {N_REPS_J} replications")
# The trace test is known to over-reject at the top rank in finite samples (it inflates the
# APPARENT rank more often than nominal size alone would predict) — a real caveat, not a bug in
# this simulation, so the bar here is "clearly better than guessing", not "always exactly right".
assert rank_hit_rate > 0.5

# %% [markdown]
# ## 4 · The Ornstein-Uhlenbeck process and its half-life
#
# The OU process is the continuous-time version of the AR(1) you met in Session 5.4:
# `dS = kappa*(theta - S) dt + sigma dW`. Discretised one period at a time it IS an AR(1),
# `S_t = a + phi*S_{t-1} + e_t` with `phi = exp(-kappa*dt)`. Fit `phi` by OLS and invert for the
# half-life: `kappa = -ln(phi)`, `half-life = ln(2) / kappa`. Check the estimator at three
# realistic half-lives — and then, deliberately, at a slow one, to show WHY practitioners discard
# slow-reverting spreads instead of trusting a point estimate of them.

# %%
HL_FAST = [5.0, 10.0, 20.0]                 # 5 years of daily data is 60-250+ half-lives: plenty
HL_SLOW = 60.0                              # 5 years is only 21 half-lives: not nearly enough
N_HL, N_REPS_HL = 5 * PERIODS_PER_YEAR, 400
hl_results = {}
for hl in HL_FAST:
    kappa = np.log(2.0) / hl
    paths = ou_paths(N_HL, kappa, 2.0, rng_hl, N_REPS_HL)
    hl_hat, kappa_hat, phi_hat = estimate_halflife(paths)
    mean_hat, se_hat = float(hl_hat.mean()), float(hl_hat.std(ddof=1) / np.sqrt(N_REPS_HL))
    hl_results[hl] = (mean_hat, se_hat)
    key = f"hl{int(hl)}"
    lab.record(f"{key}_true", hl)
    lab.record(f"{key}_hat", mean_hat)
    lab.record(f"{key}_se", se_hat)
    # OLS on a persistent series has a small, well-documented downward bias in phi (Kendall, 1954;
    # Marriott & Pope, 1954), which biases the estimated half-life a little short; 0.35 days of
    # allowance covers that residual bias when the sample holds dozens of half-lives, as it does here.
    check_mc(f"half-life recovery, true={hl:g}d", mean_hat, hl, se_hat, allowance=0.35)

# The slow setting: the SAME estimator, the SAME 5 years of data, only 21 half-lives of history.
kappa_slow = np.log(2.0) / HL_SLOW
paths_slow = ou_paths(N_HL, kappa_slow, 2.0, rng_hl, N_REPS_HL)
hl_hat_slow, *_ = estimate_halflife(paths_slow)
mean_slow = float(hl_hat_slow.mean())
se_slow = float(hl_hat_slow.std(ddof=1) / np.sqrt(N_REPS_HL))
hl_results[HL_SLOW] = (mean_slow, se_slow)
lab.record("hl60_true", HL_SLOW)
lab.record("hl60_hat", mean_slow)
lab.record("hl60_se", se_slow)
lab.record("hl60_bias", mean_slow - HL_SLOW)
lab.record("hl60_bias_pct", 100 * (mean_slow - HL_SLOW) / HL_SLOW)
lab.record("hl60_bias_abs", HL_SLOW - mean_slow)
lab.record("hl60_bias_pct_abs", 100 * (HL_SLOW - mean_slow) / HL_SLOW)
print(f"slow setting: true {HL_SLOW:g}d half-life, only {N_HL / HL_SLOW:.0f} half-lives of data "
      f"→ estimated {mean_slow:.1f} ± {se_slow:.1f}d, biased low by {HL_SLOW - mean_slow:.1f}d")
# The claim here is the OPPOSITE of section-1's checks: not "matches theory", but "is measurably,
# reliably biased low" — the same small-sample AR(1) bias as above, just no longer negligible once
# the sample holds only ~20 half-lives instead of hundreds. So the assert direction flips: this is
# a check that the gap is STATISTICALLY REAL (beyond 4 SE), not that it is absent.
z_bias = (mean_slow - HL_SLOW) / se_slow
MC_CHECKS.append(("half-life bias is real (60d setting)", float(z_bias)))
assert z_bias < -N_SE, f"expected a reliably negative bias at HL={HL_SLOW:g}d, got z={z_bias:+.2f}"

lab.record("hl_n_years", N_HL // PERIODS_PER_YEAR)
lab.record("hl_mc_reps", N_REPS_HL)
print({hl: f"{m:.2f} ± {s:.2f}" for hl, (m, s) in hl_results.items()})

HL_ALL = HL_FAST + [HL_SLOW]
lab.chart("halflife_recovery", charts.grouped_column_chart(
    [f"{int(hl)} d" for hl in HL_ALL],
    [("true half-life", HL_ALL, "benchmark"),
     ("estimated (mean of 400 fits)", [hl_results[hl][0] for hl in HL_ALL], "strategy")],
    title="Estimated vs true half-life, four settings, 5 years of daily data",
    y_fmt=charts.fmt_num(0, suffix=" d"), value_labels=True))

# %% [markdown]
# ## 5 · Tying it together: the half-life of a real recovered spread
#
# Apply the SAME half-life estimator to the residual that the Engle-Granger regression recovered
# in section 2 — not a spread simulated directly, but one recovered AFTER estimating the hedge
# ratio from data. A small Monte Carlo checks that this two-stage pipeline (estimate the hedge
# ratio, THEN estimate the half-life of the residual) still recovers the true 20-day half-life.

# %%
hl_hero, kappa_hero, phi_hero = estimate_halflife(resid_hero[:, None])
lab.record("pipeline_hl_hero", float(hl_hero[0]))
lab.record("pipeline_phi_hero", float(phi_hero[0]))

N_PIPE = 5 * PERIODS_PER_YEAR
w_pipe = common_trend(N_PIPE, SIGMA_W, rng_pipe, 300)
eta_pipe = ou_paths(N_PIPE, KAPPA_TRUE, SD_ETA, rng_pipe, 300)
a_pipe, b_pipe = A0 + H_TRUE * w_pipe + eta_pipe, B0 + w_pipe
b_pipe_c = b_pipe - b_pipe.mean(axis=0)
h_hat_pipe = (b_pipe_c * (a_pipe - a_pipe.mean(axis=0))).sum(axis=0) / (b_pipe_c ** 2).sum(axis=0)
alpha_hat_pipe = a_pipe.mean(axis=0) - h_hat_pipe * b_pipe.mean(axis=0)
resid_pipe = a_pipe - alpha_hat_pipe - h_hat_pipe * b_pipe
hl_hat_pipe, *_ = estimate_halflife(resid_pipe)

lab.record("pipeline_reps", 300)
lab.record("pipeline_h_hat_mean", float(h_hat_pipe.mean()))
lab.record("pipeline_h_hat_se", float(h_hat_pipe.std(ddof=1) / np.sqrt(300)))
lab.record("pipeline_hl_hat_mean", float(hl_hat_pipe.mean()))
lab.record("pipeline_hl_hat_se", float(hl_hat_pipe.std(ddof=1) / np.sqrt(300)))
print(f"pipeline (estimate h, then half-life): h_hat {h_hat_pipe.mean():.4f} ± "
      f"{h_hat_pipe.std(ddof=1) / np.sqrt(300):.4f} (true {H_TRUE}) · half-life "
      f"{hl_hat_pipe.mean():.2f} ± {hl_hat_pipe.std(ddof=1) / np.sqrt(300):.2f} (true {HL_TRUE})")
check_mc("pipeline hedge ratio", float(h_hat_pipe.mean()), H_TRUE,
         float(h_hat_pipe.std(ddof=1) / np.sqrt(300)))
# Same AR(1) small-sample bias as section 4's 20-day case (already ~1.1 days at this sample length),
# plus a little more from feeding the estimator an ESTIMATED hedge ratio instead of the true one;
# 1.5 days of allowance covers both, not zero.
check_mc("pipeline half-life", float(hl_hat_pipe.mean()), HL_TRUE,
         float(hl_hat_pipe.std(ddof=1) / np.sqrt(300)), allowance=1.5)

lab.chart("spread_reverting", charts.line_chart(
    [charts.Series("A − h·B", in_years(spread_true), role="strategy", label_end=False)],
    title=f"The recovered spread over {N_MAIN // PERIODS_PER_YEAR} years — half-life ≈ "
          f"{hl_hero[0]:.0f} trading days",
    y_fmt=charts.fmt_num(1), hline=float(spread_true.mean())))

lab.chart("eg_size_power", charts.column_chart(
    ["no cointegration", "true cointegration"], [size_hat, power_hat],
    title="Engle-Granger rejection rate at 5%: no relationship vs a real 20-day spread",
    y_fmt=charts.fmt_pct(0), roles=["loss", "strategy"], y_min=0.0, y_max=1.0))

# %%
lab.save()
