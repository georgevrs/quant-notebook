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
# # 5.7 · Kalman Filters & Regime Models — companion lab
#
# **Quant Notebook** · Unit 5 · Session 7 · [Read the session](https://georgevrs.github.io/quant-notebook/unit05-statistics-time-series/session07-kalman-regimes.html)
#
# State-space models, the Kalman filter for time-varying hedge ratios, hidden Markov regimes and structural breaks.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Two independent experiments, each with its own RNG stream (`lab.rng.spawn(...)`):
#
# 1. A **Kalman filter** tracks a pairs-trade hedge ratio that jumps once, mid-sample, and beats a
#    rolling-OLS hedge ratio at catching the jump — checked on one illustrative path and confirmed
#    on 300 Monte Carlo replicates.
# 2. A **2-state Gaussian HMM**, fit from scratch with the EM (Baum–Welch) algorithm, recovers a
#    known calm/crisis regime sequence from simulated returns — checked on one path and confirmed
#    on 25 Monte Carlo replicates.
#
# No risk-free rate or Sharpe ratio appears in this lab: both experiments are about recovering a
# known *parameter path*, not about strategy performance.

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
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR, in_years

lab = qn.Lab("5.7")  # seeds the random generators: every run gives the same numbers
rng = lab.rng

# %% [markdown]
# ## 1 · A Kalman filter for a time-varying hedge ratio
#
# Model (a scalar, intercept-free time-varying-parameter regression — the workhorse of Session
# 5.7.2–5.7.3): the hedge ratio is a hidden state that takes a random-walk step each day, and we
# only see it through a noisy regression observation.
#
# \\(\\beta_t = \\beta_{t-1} + \\eta_t,\\ \\eta_t \\sim N(0, q)\\), \\(\\ y_t = x_t \\beta_t +
# \\varepsilon_t,\\ \\varepsilon_t \\sim N(0, \\sigma_\\varepsilon^2)\\)
#
# `x` is leg A's daily return, `y` is leg B's, and the true hedge ratio jumps once, mid-sample —
# the cleanest possible stand-in for a real break (an index reweighting, a corporate action, a
# regime shift in the spread of Session 5.6's cointegrated pair).


# %%
def kalman_beta(x: np.ndarray, y: np.ndarray, q: float, sigma_eps2: float,
                beta0: float = 0.0, p0: float = 1.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Scalar Kalman filter for y_t = x_t * beta_t + eps_t, beta_t = beta_{t-1} + eta_t.

    Returns (filtered beta each day, filtered variance each day, total log-likelihood of the
    one-step-ahead prediction errors — used below to choose q without ever looking past the break).
    """
    t_n = len(x)
    beta_filt, p_filt = np.empty(t_n), np.empty(t_n)
    beta_pred, p_pred = beta0, p0
    loglik = 0.0
    for t in range(t_n):
        e = y[t] - x[t] * beta_pred                       # innovation: what the prediction missed
        f = x[t] ** 2 * p_pred + sigma_eps2                # innovation variance
        k = p_pred * x[t] / f                              # Kalman gain
        beta_pred = beta_pred + k * e                      # update
        p_pred = (1 - k * x[t]) * p_pred
        beta_filt[t], p_filt[t] = beta_pred, p_pred
        loglik += -0.5 * (np.log(2 * np.pi * f) + e ** 2 / f)
        p_pred = p_pred + q                                # predict the next step
    return beta_filt, p_filt, loglik


def rolling_beta(x: pd.Series, y: pd.Series, window: int) -> np.ndarray:
    """Hedge ratio through the origin, on a trailing window (expanding until the window fills)."""
    xy, xx = (x * y).to_numpy(), (x * x).to_numpy()
    out = np.empty(len(x))
    for t in range(len(x)):
        lo = max(0, t - window + 1)
        out[t] = xy[lo:t + 1].sum() / xx[lo:t + 1].sum()
    return out


# %% [markdown]
# **Sanity check first.** With zero process noise the state stops moving, and the Kalman filter
# collapses to the recursive form of plain OLS through the origin — the two must agree exactly.

# %%
rng_hedge, rng_hedge_mc, rng_regime, rng_regime_mc = rng.spawn(4)

N_DAYS, BREAK_DAY = 1_500, 750
BETA_BEFORE, BETA_AFTER = 0.8, 1.6
SIGMA_X, SIGMA_EPS = 0.015, 0.002            # daily vol of leg A; daily observation noise sd

x_check = rng_hedge.standard_normal(BREAK_DAY) * SIGMA_X
y_check = BETA_BEFORE * x_check + rng_hedge.standard_normal(BREAK_DAY) * SIGMA_EPS
beta_q0, _, _ = kalman_beta(x_check, y_check, q=0.0, sigma_eps2=SIGMA_EPS ** 2, p0=1e6)
beta_ols_check = (x_check * y_check).sum() / (x_check ** 2).sum()
assert np.isclose(beta_q0[-1], beta_ols_check, rtol=1e-6), "q=0 Kalman filter must equal OLS"
lab.record("qzero_kalman_beta", float(beta_q0[-1]))
lab.record("qzero_ols_beta", float(beta_ols_check))

# %% [markdown]
# ## 2 · Choosing the process noise without looking at the break
#
# Real hedge ratios do not sit dead still between breaks — they wander a little every day, then
# occasionally jump. So the true state here is a small random walk (daily step size \\(q_{true}\\))
# **plus** one deliberate one-off jump at `BREAK_DAY`. `q` sets how fast the filter is willing to
# believe the state has moved; pick it the honest way — maximise the one-step-ahead
# log-likelihood on data *before* the break only, so it learns the day-to-day wander without ever
# seeing the jump it will later be tested on (peeking at the jump itself is the look-ahead pitfall
# of §5.7.5). This is the same marginal-likelihood idea as the empirical Bayes of Session 5.3,
# applied to a variance instead of a prior mean.

# %%
Q_TRUE_DAILY = 3e-5              # the hedge ratio's ordinary day-to-day wander (std ≈ 0.15 over 750 days)
JUMP = 0.8                      # the one deliberate, much larger, one-off shift

steps = rng_hedge.standard_normal(N_DAYS) * np.sqrt(Q_TRUE_DAILY)
steps[0], steps[BREAK_DAY] = 0.0, steps[BREAK_DAY] + JUMP
beta_true = BETA_BEFORE + np.cumsum(steps)

x_full = rng_hedge.standard_normal(N_DAYS) * SIGMA_X
y_full = beta_true * x_full + rng_hedge.standard_normal(N_DAYS) * SIGMA_EPS

Q_GRID = np.geomspace(1e-9, 1e-2, 16)
ll_grid = [kalman_beta(x_full[:BREAK_DAY], y_full[:BREAK_DAY], q=q, sigma_eps2=SIGMA_EPS ** 2)[2]
          for q in Q_GRID]
Q_STAR = float(Q_GRID[int(np.argmax(ll_grid))])
lab.record("q_true", Q_TRUE_DAILY)
lab.record("q_star", Q_STAR)
lab.record("q_grid_lo", float(Q_GRID[0]))
lab.record("q_grid_hi", float(Q_GRID[-1]))
print(f"true q = {Q_TRUE_DAILY:.2e} · selected q* = {Q_STAR:.2e} by pre-break likelihood alone")

ROLL_WINDOW = 60
beta_kalman, p_kalman, _ = kalman_beta(x_full, y_full, q=Q_STAR, sigma_eps2=SIGMA_EPS ** 2)
beta_roll = rolling_beta(pd.Series(x_full), pd.Series(y_full), ROLL_WINDOW)

AFTER_K = 60  # the window right after the break where adaptation speed shows up


def rmse(est, truth, lo, hi):
    return float(np.sqrt(np.mean((est[lo:hi] - truth[lo:hi]) ** 2)))


rmse_kalman_before = rmse(beta_kalman, beta_true, 100, BREAK_DAY)   # skip the first 100 days: filter warms up
rmse_kalman_after = rmse(beta_kalman, beta_true, BREAK_DAY, BREAK_DAY + AFTER_K)
rmse_roll_before = rmse(beta_roll, beta_true, 100, BREAK_DAY)
rmse_roll_after = rmse(beta_roll, beta_true, BREAK_DAY, BREAK_DAY + AFTER_K)
for k, v in [("rmse_kalman_before", rmse_kalman_before), ("rmse_kalman_after", rmse_kalman_after),
            ("rmse_roll_before", rmse_roll_before), ("rmse_roll_after", rmse_roll_after)]:
    lab.record(k, v)


lab.record("roll_window", ROLL_WINDOW)
lab.record("hedge_n_days", N_DAYS)
lab.record("hedge_break_day", BREAK_DAY)
lab.record("hedge_break_year", BREAK_DAY / PERIODS_PER_YEAR)
lab.record("beta_before", BETA_BEFORE)
lab.record("beta_after", float(beta_true[BREAK_DAY]))
lab.record("jump_size", JUMP)
print(f"post-break RMSE: Kalman {rmse_kalman_after:.3f} vs rolling OLS {rmse_roll_after:.3f}")
assert rmse_kalman_after < rmse_roll_after, "the whole point: Kalman should track the jump faster"

# %% [markdown]
# ## 3 · Does that hold up, or did we get lucky? (300 replicates)
#
# One path is a demonstration, not evidence. Re-draw the noise 300 times with the same `q*` and
# the same window, and require the average advantage to clear **4 standard errors** — never fewer,
# and never chosen after peeking at which seeds look good.

# %%
N_PATHS = 300
mc_streams = rng_hedge_mc.spawn(N_PATHS)
diffs = np.empty(N_PATHS)
for i, s in enumerate(mc_streams):
    xi = s.standard_normal(N_DAYS) * SIGMA_X
    yi = beta_true * xi + s.standard_normal(N_DAYS) * SIGMA_EPS
    bk, _, _ = kalman_beta(xi, yi, q=Q_STAR, sigma_eps2=SIGMA_EPS ** 2)
    br = rolling_beta(pd.Series(xi), pd.Series(yi), ROLL_WINDOW)
    diffs[i] = rmse(br, beta_true, BREAK_DAY, BREAK_DAY + AFTER_K) - rmse(bk, beta_true, BREAK_DAY, BREAK_DAY + AFTER_K)

mc_mean, mc_se = float(diffs.mean()), float(diffs.std(ddof=1) / np.sqrt(N_PATHS))
mc_z = mc_mean / mc_se
lab.record("mc_hedge_n_paths", N_PATHS)
lab.record("mc_hedge_diff_mean", mc_mean)
lab.record("mc_hedge_diff_se", mc_se)
lab.record("mc_hedge_diff_z", mc_z)
print(f"mean(RMSE_roll - RMSE_kalman) = {mc_mean:.4f}, SE = {mc_se:.4f}, z = {mc_z:.1f}")
assert mc_z > 4, "Kalman's post-break advantage must clear 4 standard errors, not a lucky path"

# %% [markdown]
# ## 4 · What a badly-chosen q costs you
#
# `q*` was tuned honestly, but tuning is not magic — it is a bet on how much the state usually
# moves. Multiply it by a constant and watch two things at once: how noisily the filter tracks a
# *quiet* stretch where nothing is happening (`t=200..700`, well clear of the warm-up and the
# jump), and how quickly it fully closes the gap after the jump (RMSE over the 150 days after
# `BREAK_DAY`, a longer window than §5.7.4's headline 60 days, chosen here to show the filter
# eventually catching up rather than the early-response advantage). This relationship is a
# property of the Kalman recursion itself, not of one lucky draw of noise, so it is checked
# directly rather than by simulation.

# %%
Q_SETTINGS = [("too small", "small", Q_STAR / 15), ("well-tuned (q*)", "tuned", Q_STAR),
             ("too large", "large", Q_STAR * 15)]
q_sensitivity = {}
for label, key, q in Q_SETTINGS:
    bq, _, _ = kalman_beta(x_full, y_full, q=q, sigma_eps2=SIGMA_EPS ** 2)
    quiet_noise = float(np.std(np.diff(bq[200:700])))
    after150 = rmse(bq, beta_true, BREAK_DAY, BREAK_DAY + 150)
    q_sensitivity[key] = (q, quiet_noise, after150)
    lab.record(f"q_{key}_value", q)
    lab.record(f"q_{key}_quiet_noise", quiet_noise)
    lab.record(f"q_{key}_rmse150", after150)
    print(f"{label:16s} q={q:.2e}  quiet noise {quiet_noise:.4f}  RMSE (150d after) {after150:.3f}")

noises = [q_sensitivity[k][1] for k in ("small", "tuned", "large")]
errs = [q_sensitivity[k][2] for k in ("small", "tuned", "large")]
assert noises[0] < noises[1] < noises[2], "bigger q must track a quiet period more noisily"
assert errs[0] > errs[1] > errs[2], "bigger q must close a real gap faster"

# %% [markdown]
# ## 5 · Charts for §5.7.4 and §5.7.5

# %%
lab.chart("hedge_ratio", charts.line_chart(
    [charts.Series("true hedge ratio", in_years(beta_true), role="benchmark", width=1.5, end_label="true"),
     charts.Series("Kalman filter", in_years(beta_kalman), role="strategy", end_label="Kalman"),
     charts.Series(f"rolling OLS ({ROLL_WINDOW}d)", in_years(beta_roll), role="alt1", end_label="rolling OLS")],
    title="Tracking a hedge ratio that jumps once, mid-sample", y_fmt=charts.fmt_num(2),
    bands=[(BREAK_DAY / PERIODS_PER_YEAR, (BREAK_DAY + 2) / PERIODS_PER_YEAR, "ratio jumps")]))

lab.chart("hedge_rmse", charts.grouped_column_chart(
    ["before the jump", f"first {AFTER_K}d after"],
    [("Kalman filter", [rmse_kalman_before, rmse_kalman_after], "strategy"),
     (f"rolling OLS ({ROLL_WINDOW}d)", [rmse_roll_before, rmse_roll_after], "alt1")],
    title="Tracking error (RMSE) before and after the jump", y_fmt=charts.fmt_num(2)))

# %% [markdown]
# ## 6 · Simulating calm/crisis regimes
#
# A 2-state Markov chain switches between a calm regime and a crisis regime. Daily *means* are
# tiny next to daily *volatilities* (as in Session 2.3), so — realistically — the regimes will turn
# out to be identified almost entirely by the change in variance, not the change in mean.

# %%
P_STAY_CALM, P_STAY_CRISIS = 0.995, 0.97
MU_ANNUAL = np.array([0.10, -0.15])          # calm, crisis
SIGMA_ANNUAL = np.array([0.12, 0.35])
MU_TRUE = MU_ANNUAL / PERIODS_PER_YEAR
SIGMA_TRUE = SIGMA_ANNUAL / np.sqrt(PERIODS_PER_YEAR)
TRANS_TRUE = np.array([[P_STAY_CALM, 1 - P_STAY_CALM], [1 - P_STAY_CRISIS, P_STAY_CRISIS]])


def simulate_regimes(t_n: int, trans: np.ndarray, mu, sigma, rng) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros(t_n, dtype=int)
    u = rng.random(t_n)
    for t in range(1, t_n):
        p_stay = trans[states[t - 1], states[t - 1]]
        states[t] = states[t - 1] if u[t] <= p_stay else 1 - states[t - 1]
    rets = rng.normal(mu[states], sigma[states])
    return states, rets


REGIME_T = 3_000
true_states, regime_rets = simulate_regimes(REGIME_T, TRANS_TRUE, MU_TRUE, SIGMA_TRUE, rng_regime)
lab.record("regime_t", REGIME_T)
lab.record("regime_years", REGIME_T / PERIODS_PER_YEAR)
lab.record("p_stay_calm", P_STAY_CALM)
lab.record("p_stay_crisis", P_STAY_CRISIS)
lab.record("avg_duration_calm", 1 / (1 - P_STAY_CALM))
lab.record("avg_duration_crisis", 1 / (1 - P_STAY_CRISIS))
stationary_p_crisis = (1 - P_STAY_CALM) / ((1 - P_STAY_CALM) + (1 - P_STAY_CRISIS))
lab.record("stationary_p_crisis", stationary_p_crisis)
lab.record("mu_calm_annual", MU_ANNUAL[0])
lab.record("mu_crisis_annual", MU_ANNUAL[1])
lab.record("sigma_calm_annual", SIGMA_ANNUAL[0])
lab.record("sigma_crisis_annual", SIGMA_ANNUAL[1])
lab.record("frac_time_crisis", float((true_states == 1).mean()))

# %% [markdown]
# ## 7 · A 2-state Gaussian HMM, fit from scratch with EM (Baum–Welch)
#
# Two passes over the data, alternated: **forward–backward** turns a guess of the parameters into
# a probability of being in each state on each day; the **M-step** turns those probabilities into a
# better guess of the parameters. `np.logaddexp` is the two-term log-sum-exp trick that keeps 3,000
# days of products of probabilities from underflowing to zero.

# %%
VAR_FLOOR = (0.02 * SIGMA_TRUE[0]) ** 2  # stop a state's variance collapsing onto one outlier day


def forward_backward(log_b: np.ndarray, log_trans: np.ndarray, log_pi0: np.ndarray):
    t_n, k = log_b.shape
    log_alpha, log_beta = np.zeros((t_n, k)), np.zeros((t_n, k))
    log_alpha[0] = log_pi0 + log_b[0]
    for t in range(1, t_n):
        log_alpha[t] = log_b[t] + np.logaddexp.reduce(log_alpha[t - 1][:, None] + log_trans, axis=0)
    for t in range(t_n - 2, -1, -1):
        log_beta[t] = np.logaddexp.reduce(log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    loglik = float(np.logaddexp.reduce(log_alpha[-1]))
    gamma = np.exp(log_alpha + log_beta - loglik)
    xi_sum = np.zeros((k, k))
    for t in range(t_n - 1):
        xi_sum += np.exp(log_alpha[t][:, None] + log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :] - loglik)
    return gamma, xi_sum, loglik


def viterbi(log_b: np.ndarray, log_trans: np.ndarray, log_pi0: np.ndarray) -> np.ndarray:
    t_n, k = log_b.shape
    delta, psi = np.zeros((t_n, k)), np.zeros((t_n, k), dtype=int)
    delta[0] = log_pi0 + log_b[0]
    for t in range(1, t_n):
        scores = delta[t - 1][:, None] + log_trans           # scores[j, k]
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(k)] + log_b[t]
    path = np.empty(t_n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(t_n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def fit_gaussian_hmm(returns: np.ndarray, rng, k: int = 2, n_iter: int = 60, tol: float = 1e-5):
    """Baum-Welch EM for a k-state Gaussian HMM. Returns (mu, sigma, trans, pi0, gamma, n_iters, loglik_history)."""
    mu = np.full(k, returns.mean()) + rng.normal(0, returns.std() * 0.1, k)
    var = returns.var() * np.linspace(0.6, 1.6, k)
    trans = np.full((k, k), 0.1 / (k - 1))
    np.fill_diagonal(trans, 0.9)
    pi0 = np.full(k, 1 / k)
    history = []
    for it in range(n_iter):
        sigma = np.sqrt(var)
        log_b = norm.logpdf(returns[:, None], mu[None, :], sigma[None, :])
        gamma, xi_sum, loglik = forward_backward(log_b, np.log(trans), np.log(pi0))
        history.append(loglik)
        if it > 0:
            assert loglik >= history[-2] - 1e-6, "EM log-likelihood must not decrease"
            if loglik - history[-2] < tol:
                break
        pi0 = gamma[0] / gamma[0].sum()
        trans = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu = (gamma * returns[:, None]).sum(axis=0) / w
        var = np.maximum((gamma * (returns[:, None] - mu[None, :]) ** 2).sum(axis=0) / w, VAR_FLOOR)
    order = np.argsort(var)  # resolve label-switching: state 0 is always the lower-variance (calm) one
    mu, var, pi0 = mu[order], var[order], pi0[order]
    trans = trans[np.ix_(order, order)]
    gamma = gamma[:, order]
    return mu, np.sqrt(var), trans, pi0, gamma, it + 1, history


mu_est, sigma_est, trans_est, pi0_est, gamma_est, n_iters, ll_hist = fit_gaussian_hmm(regime_rets, rng_regime)
decoded = viterbi(norm.logpdf(regime_rets[:, None], mu_est[None, :], sigma_est[None, :]),
                  np.log(trans_est), np.log(pi0_est))
accuracy = float((decoded == true_states).mean())

lab.record("em_iters", n_iters)
lab.record("em_mu_calm_annual", float(mu_est[0] * PERIODS_PER_YEAR))
lab.record("em_mu_crisis_annual", float(mu_est[1] * PERIODS_PER_YEAR))
lab.record("em_sigma_calm_annual", float(sigma_est[0] * np.sqrt(PERIODS_PER_YEAR)))
lab.record("em_sigma_crisis_annual", float(sigma_est[1] * np.sqrt(PERIODS_PER_YEAR)))
lab.record("em_p_stay_calm", float(trans_est[0, 0]))
lab.record("em_p_stay_crisis", float(trans_est[1, 1]))
lab.record("viterbi_accuracy", accuracy)
print(f"EM converged in {n_iters} iterations · Viterbi accuracy {accuracy:.1%}")
assert accuracy > 0.75, "single-path sanity check: recovery should be well above chance"

# %% [markdown]
# ## 8 · Does the HMM reliably recover the regimes? (25 replicates)
#
# Same standard as §3: one lucky fit proves nothing. Refit on 25 fresh simulations (a shorter
# series each, to keep the lab fast) and require the average accuracy to clear 4 standard errors
# above a 75% floor — comfortably better than guessing, comfortably below what a good fit achieves.

# %%
N_REPL, REPL_T = 25, 1_200
mc_regime_streams = rng_regime_mc.spawn(N_REPL)
accs = np.empty(N_REPL)
for i, s in enumerate(mc_regime_streams):
    st_i, ret_i = simulate_regimes(REPL_T, TRANS_TRUE, MU_TRUE, SIGMA_TRUE, s)
    mu_i, sig_i, tr_i, pi_i, _, _, _ = fit_gaussian_hmm(ret_i, s, n_iter=50)
    dec_i = viterbi(norm.logpdf(ret_i[:, None], mu_i[None, :], sig_i[None, :]), np.log(tr_i), np.log(pi_i))
    accs[i] = (dec_i == st_i).mean()

acc_mean, acc_se = float(accs.mean()), float(accs.std(ddof=1) / np.sqrt(N_REPL))
lab.record("mc_hmm_n", N_REPL)
lab.record("mc_hmm_mean_accuracy", acc_mean)
lab.record("mc_hmm_se_accuracy", acc_se)
lab.record("mc_hmm_lower_bound", acc_mean - 4 * acc_se)
print(f"HMM recovery accuracy over {N_REPL} replicates: {acc_mean:.1%} ± {acc_se:.1%} (SE)")
assert acc_mean - 4 * acc_se > 0.75, "average recovery must clear 4 SE above a 75% floor"

# %% [markdown]
# ## 9 · Chart for §5.7.7 — the recovered regime, against the truth

# %%
# A representative window centred on the single longest crisis episode — wide enough on screen
# to carry its own label, with some calm on both sides for context.
runs, run_start = [], 0
for t in range(1, REGIME_T + 1):
    if t == REGIME_T or true_states[t] != true_states[run_start]:
        if true_states[run_start] == 1:
            runs.append((run_start, t))
        run_start = t
widest = max(runs, key=lambda r: r[1] - r[0])
PAD = 100
win_start, win_end = max(0, widest[0] - PAD), min(REGIME_T, widest[1] + PAD)
CHART_T = win_end - win_start
true_bands = [((widest[0] - win_start) / PERIODS_PER_YEAR, (widest[1] - win_start) / PERIODS_PER_YEAR, "true crisis")]

lab.chart("regime_probability", charts.line_chart(
    [charts.Series("P(crisis | data)", in_years(gamma_est[win_start:win_end, 1]), role="strategy", label_end=False)],
    title=f"Smoothed P(crisis) vs. when it was actually true, over a representative {CHART_T / PERIODS_PER_YEAR:.1f}-year window",
    y_fmt=charts.fmt_pct(0), y_min=0.0, y_max=1.0, bands=true_bands))

# %%
lab.save()
