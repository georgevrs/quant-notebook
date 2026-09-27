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
# # 5.3 · Bayesian Thinking: Priors, Shrinkage & Updating — companion lab
#
# **Quant Notebook** · Unit 5 · Session 3 · [Read the session](https://georgevrs.github.io/quant-notebook/unit05-statistics-time-series/session03-bayesian-thinking.html)
#
# Priors, posteriors and shrinkage — the machinery behind Ledoit-Wolf, Black-Litterman and Kalman filters, and the honest answer to "is my Sharpe real?"
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Three experiments, each solved exactly and checked by simulation:
#   1. A single fund: combine a prior belief about skill with one noisy Sharpe estimate
#      (conjugate Normal-Normal updating), and contrast the resulting credible interval with
#      the plain confidence interval.
#   2. A single strategy's win rate: conjugate Beta-Binomial updating, and when a sensible prior
#      beats the raw hit rate.
#   3. The main event — 200 synthetic funds with a known cross-sectional distribution of true
#      Sharpe ratios. Each fund's Sharpe is estimated noisily from its own track record; an
#      empirical-Bayes (James-Stein-style) shrinkage estimator pulls every estimate toward the
#      cross-sectional mean by an amount that depends on how little that fund's own history is
#      worth trusting. We then check, out of sample, that shrinking beats trusting each raw
#      number at face value — the practical case for shrinkage that Session 9.2 (covariance
#      shrinkage) and 9.6 (Black-Litterman) reuse.
#
# Every fund/strategy here is synthetic with a stated annual volatility of 15% and a risk-free
# rate of 0 (returns are already "excess"); simulated series are indexed in elapsed years, never
# calendar dates. Monte Carlo checks pass at 4 standard errors of simulation, never 3, and never by
# seed-hunting.

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
from scipy import stats as sps

import quantnb as qn
from quantnb import charts, stats

lab = qn.Lab("5.3")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
rng_conj, rng_beta, rng_cross = lab.rng.spawn(3)

Z95 = float(sps.norm.ppf(0.975))  # 1.96: the multiplier for a 95% interval, either kind
ANNUAL_VOL = 0.15                 # every fund/strategy in this lab has this annual volatility;
SIGMA_DAILY = ANNUAL_VOL / np.sqrt(252)   # only the (unknown) mean return differs between them —
lab.record("annual_vol", ANNUAL_VOL)      # that isolates the Sharpe-estimation problem cleanly.
lab.record("z95", Z95)

# %% [markdown]
# ## 1 · One fund, one prior, one noisy estimate (conjugate Normal-Normal updating)
#
# A prior belief about a fund's Sharpe ratio: across the funds on this shelf, true annual Sharpe
# ratios are roughly Normal with mean **MU0** and standard deviation **TAU0** (the same
# cross-sectional distribution the big experiment in §3 uses — it is the "shelf" every fund here
# is drawn from). A single fund reports SR_HAT, its Sharpe estimated from EX_YEARS years of daily
# excess returns; the **likelihood**'s spread is that estimate's standard error (Lo, 2002; §5.2).
# Posterior mean = a precision-weighted average of the prior mean and the data; the weight the
# data keeps, k = tau0^2 / (tau0^2 + se^2), is exactly the "shrinkage weight" the whole session is
# about. We check the closed-form posterior two ways: by formula, and by simulating a million
# (true skill, noisy estimate) pairs and recovering the same weight as a regression slope — exact
# for jointly Normal variables, since E[theta | y] is then exactly linear in y.

# %%
MU0, TAU0 = 0.20, 0.30      # the shelf: mean and sd of true Sharpe ratios across funds
EX_YEARS = 3.0              # this one fund's track record
EX_TRUE_THETA = 0.55        # its actual (unobservable) skill — known only because this is synthetic
N_JOINT = 2_000_000         # pairs for the simulation check

ex_days = int(round(EX_YEARS * 252))
ex_mu_daily = EX_TRUE_THETA * ANNUAL_VOL / 252
ex_returns = rng_conj.normal(ex_mu_daily, SIGMA_DAILY, ex_days)
ex_sr_hat = float(stats.annual_sharpe(ex_returns))
ex_se = stats.sharpe_se(ex_sr_hat, years=EX_YEARS)

ex_k = TAU0 ** 2 / (TAU0 ** 2 + ex_se ** 2)                    # weight kept on the data
ex_post_var = 1.0 / (1.0 / TAU0 ** 2 + 1.0 / ex_se ** 2)       # posterior precision = sum of precisions
ex_post_mean = ex_post_var * (MU0 / TAU0 ** 2 + ex_sr_hat / ex_se ** 2)
assert np.isclose(ex_post_mean, MU0 + ex_k * (ex_sr_hat - MU0))  # the two forms of the same formula agree
ex_post_sd = np.sqrt(ex_post_var)

# simulate: draw true skill from the prior, then a noisy estimate around it, and regress
theta_sim = rng_conj.normal(MU0, TAU0, N_JOINT)
y_sim = theta_sim + rng_conj.normal(0.0, ex_se, N_JOINT)
joint = np.column_stack([theta_sim, y_sim])


def slope_stat(batch: np.ndarray) -> float:
    """OLS slope of column 0 (theta) on column 1 (y) — the empirical shrinkage weight."""
    t, y = batch[:, 0], batch[:, 1]
    return float(np.cov(t, y, ddof=1)[0, 1] / np.var(y, ddof=1))


def intercept_stat(batch: np.ndarray) -> float:
    t, y = batch[:, 0], batch[:, 1]
    slope = np.cov(t, y, ddof=1)[0, 1] / np.var(y, ddof=1)
    return float(t.mean() - slope * y.mean())


sim_slope = slope_stat(joint)
sim_intercept = intercept_stat(joint)
se_slope = stats.batch_se(slope_stat, joint, n_batches=50)
se_intercept = stats.batch_se(intercept_stat, joint, n_batches=50)
assert abs(sim_slope - ex_k) < 4 * se_slope, "simulated regression slope should recover the shrinkage weight"
assert abs(sim_intercept - MU0 * (1 - ex_k)) < 4 * se_intercept

# the two kinds of interval, same data, same 95% nominal level
ex_ci_lo, ex_ci_hi = ex_sr_hat - Z95 * ex_se, ex_sr_hat + Z95 * ex_se
ex_cred_lo, ex_cred_hi = ex_post_mean - Z95 * ex_post_sd, ex_post_mean + Z95 * ex_post_sd

print(f"prior N({MU0}, {TAU0}^2) · SR_hat {ex_sr_hat:+.3f} (SE {ex_se:.3f}, k {ex_k:.3f}) · "
      f"posterior mean {ex_post_mean:+.3f} (sd {ex_post_sd:.3f})")
print(f"95% CI [{ex_ci_lo:+.3f}, {ex_ci_hi:+.3f}] (width {ex_ci_hi - ex_ci_lo:.3f}) vs "
      f"95% credible interval [{ex_cred_lo:+.3f}, {ex_cred_hi:+.3f}] (width {ex_cred_hi - ex_cred_lo:.3f})")
assert (ex_ci_hi - ex_ci_lo) > (ex_cred_hi - ex_cred_lo), "the credible interval is narrower: it borrows precision from the prior"

lab.record("mu0", MU0)
lab.record("tau0", TAU0)
lab.record("ex_years", EX_YEARS)
lab.record("ex_true_theta", EX_TRUE_THETA)
lab.record("ex_sr_hat", ex_sr_hat)
lab.record("ex_se", ex_se)
lab.record("ex_w", ex_k)
lab.record("ex_shrink_pct", 1 - ex_k)
lab.record("ex_post_mean", ex_post_mean)
lab.record("ex_post_sd", ex_post_sd)
lab.record("ex_ci_lo", ex_ci_lo)
lab.record("ex_ci_hi", ex_ci_hi)
lab.record("ex_ci_width", ex_ci_hi - ex_ci_lo)
lab.record("ex_cred_lo", ex_cred_lo)
lab.record("ex_cred_hi", ex_cred_hi)
lab.record("ex_cred_width", ex_cred_hi - ex_cred_lo)
lab.record("ex_width_ratio", (ex_cred_hi - ex_cred_lo) / (ex_ci_hi - ex_ci_lo))

# %% [markdown]
# ## 2 · A strategy's win rate (conjugate Beta-Binomial updating)
#
# The other conjugate pair every quant meets early: a **Beta(a0, b0)** prior on a win
# probability p, updated by n Bernoulli trials with k wins, gives a **Beta(a0+k, b0+n-k)**
# posterior — the wins and losses are just added onto the prior's own "pseudo win/loss count".
# We check the update two ways: against a fine numerical grid (posterior ∝ prior × likelihood,
# computed directly and normalised), and, for many replications, whether the posterior mean beats
# the raw hit rate k/n on out-of-sample squared error — the same shrinkage logic as §1 and §3, one
# trade at a time.

# %%
A0, B0 = 4.0, 4.0            # a weakly informative prior: centred at 50/50, worth 8 pseudo-trades
P_TRUE = 0.55                # the strategy's real (unobservable) edge
N_TRADES = 20                # a new strategy's first few weeks
N_TRADES_LARGE = 2_000       # a long-running strategy, for contrast
R_BETA = 200_000             # replications for the small-n vs large-n comparison

k_example = int(rng_beta.binomial(N_TRADES, P_TRUE))
post_a, post_b = A0 + k_example, B0 + (N_TRADES - k_example)
post_mean_beta = post_a / (post_a + post_b)
post_var_beta = (post_a * post_b) / ((post_a + post_b) ** 2 * (post_a + post_b + 1))
mle_beta = k_example / N_TRADES
cred_lo_beta, cred_hi_beta = float(sps.beta.ppf(0.025, post_a, post_b)), float(sps.beta.ppf(0.975, post_a, post_b))
wald_se = np.sqrt(mle_beta * (1 - mle_beta) / N_TRADES)
wald_lo, wald_hi = mle_beta - Z95 * wald_se, mle_beta + Z95 * wald_se

# grid check: posterior computed directly from Bayes' rule, not the conjugate shortcut
p_grid = np.linspace(1e-5, 1 - 1e-5, 200_001)
unnorm = sps.beta.pdf(p_grid, A0, B0) * sps.binom.pmf(k_example, N_TRADES, p_grid)
post_pdf = unnorm / np.trapezoid(unnorm, p_grid)
grid_mean = float(np.trapezoid(p_grid * post_pdf, p_grid))
grid_var = float(np.trapezoid((p_grid - grid_mean) ** 2 * post_pdf, p_grid))
assert abs(grid_mean - post_mean_beta) < 1e-4, "numerical Bayes' rule should match the conjugate formula"
assert abs(grid_var - post_var_beta) < 1e-4

# small-n: does the (correctly-centred) prior reduce mean squared error versus the raw hit rate?
k_small = rng_beta.binomial(N_TRADES, P_TRUE, R_BETA)
mle_small = k_small / N_TRADES
bayes_small = (A0 + k_small) / (A0 + B0 + N_TRADES)
err_mle_small = (mle_small - P_TRUE) ** 2
err_bayes_small = (bayes_small - P_TRUE) ** 2
diff_small = err_mle_small - err_bayes_small
se_diff_small = diff_small.std(ddof=1) / np.sqrt(R_BETA)
assert diff_small.mean() > 4 * se_diff_small, "with only 20 trades, the sensible prior should cut MSE"
mse_reduction_small = float(diff_small.mean() / err_mle_small.mean())

# large-n: the same prior, but swamped by 2,000 trades of data
k_large = rng_beta.binomial(N_TRADES_LARGE, P_TRUE, R_BETA)
mle_large = k_large / N_TRADES_LARGE
bayes_large = (A0 + k_large) / (A0 + B0 + N_TRADES_LARGE)
err_mle_large = (mle_large - P_TRUE) ** 2
err_bayes_large = (bayes_large - P_TRUE) ** 2
diff_large = err_mle_large - err_bayes_large
se_diff_large = diff_large.std(ddof=1) / np.sqrt(R_BETA)
mse_reduction_large = float(diff_large.mean() / err_mle_large.mean())
# with 100x the trades the prior's pseudo-count (8) is 100x less of the total (8 vs 2,008): the
# posterior mean has almost fully converged to the MLE, so whatever edge shrinkage still has left
# should be at least 10x smaller, in relative terms, than it was at n = 20.
assert mse_reduction_large < mse_reduction_small / 10, "with 2,000 trades the prior's influence should have mostly washed out"

print(f"k={k_example}/{N_TRADES}: MLE {mle_beta:.3f}, posterior mean {post_mean_beta:.3f} "
      f"(95% credible [{cred_lo_beta:.3f}, {cred_hi_beta:.3f}]) vs Wald 95% CI [{wald_lo:.3f}, {wald_hi:.3f}]")
print(f"small n={N_TRADES}: MSE reduction {mse_reduction_small:.1%} · "
      f"large n={N_TRADES_LARGE}: MSE reduction {mse_reduction_large:.2%} (the prior has mostly washed out)")

lab.record("beta_a0", A0)
lab.record("beta_b0", B0)
lab.record("beta_p_true", P_TRUE)
lab.record("beta_n_trades", N_TRADES)
lab.record("beta_k_example", k_example)
lab.record("beta_mle", mle_beta)
lab.record("beta_post_mean", post_mean_beta)
lab.record("beta_post_a", post_a)
lab.record("beta_post_b", post_b)
lab.record("beta_cred_lo", cred_lo_beta)
lab.record("beta_cred_hi", cred_hi_beta)
lab.record("beta_wald_lo", wald_lo)
lab.record("beta_wald_hi", wald_hi)
lab.record("beta_n_trades_large", N_TRADES_LARGE)
lab.record("beta_mse_reduction_small", mse_reduction_small)
lab.record("beta_mse_mle_small", float(err_mle_small.mean()))
lab.record("beta_mse_bayes_small", float(err_bayes_small.mean()))
lab.record("beta_mse_reduction_large", mse_reduction_large)

# %% [markdown]
# ## 3 · Two hundred funds: empirical-Bayes shrinkage of the Sharpe ratio
#
# Now the cross-section. 200 synthetic funds, true annual Sharpe ratios drawn from the same shelf
# as §1, N(MU0, TAU0^2) — loosely motivated by the finding that most mutual funds show no true
# skill once luck is priced out (Barras, Scaillet & Wermers, 2010): a cross-section should put
# most of its mass near a modest number, with a thin positive tail, not assume every fund is
# equally likely to be a star. Funds differ in how long they have existed (2, 3, 5 or 10 years of
# daily data, 50 funds each) — everything else about them (the true generating process, the 15%
# volatility) is identical, so any pattern in the results comes only from track-record length and
# luck. We do NOT get to see each fund's true Sharpe (theta) in reality; here we do, because it is
# synthetic, which is exactly what lets us score the estimators honestly.
#
# The estimator: empirical Bayes. Instead of assuming MU0 and TAU0 are known, estimate them from
# the cross-section itself (method of moments: the grand mean, and the cross-sectional variance
# of the estimates minus the average sampling noise). Shrink every fund's raw Sharpe toward that
# estimated grand mean by its own precision-weighted amount — a fund with a short, noisy history
# gets pulled hard toward the average; a fund with a long history keeps more of its own signal.
#
# Scored two ways: against the (synthetic) truth theta, and against each fund's OWN next
# T_OOS_YEARS years of performance — the realistic version of the question, since no one ever
# observes theta. Replicated R_REPS times, with fresh funds and fresh histories each time, so the
# improvement from shrinking is itself checked at 4 standard errors, not read off a single lucky
# (or unlucky) draw.

# %%
THETA_MEAN, THETA_SD = MU0, TAU0     # the same shelf as §1
YEARS_GROUPS = [2, 3, 5, 10]         # track-record length, years
N_PER_GROUP = 50
N_FUNDS = len(YEARS_GROUPS) * N_PER_GROUP
T_OOS_YEARS = 3.0
TOP_FRAC = 0.10                      # "the top 10% of the leaderboard"
TOP_N = int(round(TOP_FRAC * N_FUNDS))
R_REPS = 300

years_arr = np.repeat(YEARS_GROUPS, N_PER_GROUP).astype(float)  # fixed group membership, every replication
oos_days = int(round(T_OOS_YEARS * 252))

mse_raw_theta = np.empty(R_REPS)
mse_shrunk_theta = np.empty(R_REPS)
mse_raw_oos = np.empty(R_REPS)
mse_shrunk_oos = np.empty(R_REPS)
cover_ci_all = np.empty(R_REPS)
cover_cred_all = np.empty(R_REPS)
cover_ci_top = np.empty(R_REPS)
cover_cred_top = np.empty(R_REPS)
mean_theta_top_raw = np.empty(R_REPS)
mean_theta_top_shrunk = np.empty(R_REPS)
frac_shortest_top_raw = np.empty(R_REPS)
frac_shortest_top_shrunk = np.empty(R_REPS)
tau0_hats = np.empty(R_REPS)
headline: dict = {}

for r in range(R_REPS):
    theta = rng_cross.normal(THETA_MEAN, THETA_SD, N_FUNDS)
    mu_daily = theta * ANNUAL_VOL / 252

    sr_hat = np.empty(N_FUNDS)
    se = np.empty(N_FUNDS)
    idx = 0
    for y in YEARS_GROUPS:
        days = int(round(y * 252))
        grp = slice(idx, idx + N_PER_GROUP)
        rets = rng_cross.normal(mu_daily[grp], SIGMA_DAILY, size=(days, N_PER_GROUP))
        sr_hat[grp] = stats.annual_sharpe(rets, axis=0)
        se[grp] = [stats.sharpe_se(s, years=y) for s in sr_hat[grp]]
        idx += N_PER_GROUP

    # empirical Bayes: estimate the shelf's mean and spread from this cross-section alone
    mu0_hat = float(sr_hat.mean())
    tau0_hat_sq = max(float(sr_hat.var(ddof=1)) - float(np.mean(se ** 2)), 0.0)
    tau0_hats[r] = np.sqrt(tau0_hat_sq)
    k = tau0_hat_sq / (tau0_hat_sq + se ** 2)          # weight kept on this fund's own data
    sr_shrunk = mu0_hat + k * (sr_hat - mu0_hat)
    post_sd = np.sqrt(k * se ** 2)                     # for a credible interval per fund

    rets_oos = rng_cross.normal(mu_daily, SIGMA_DAILY, size=(oos_days, N_FUNDS))
    sr_oos = stats.annual_sharpe(rets_oos, axis=0)

    mse_raw_theta[r] = float(np.mean((sr_hat - theta) ** 2))
    mse_shrunk_theta[r] = float(np.mean((sr_shrunk - theta) ** 2))
    mse_raw_oos[r] = float(np.mean((sr_hat - sr_oos) ** 2))
    mse_shrunk_oos[r] = float(np.mean((sr_shrunk - sr_oos) ** 2))

    ci_lo, ci_hi = sr_hat - Z95 * se, sr_hat + Z95 * se
    cred_lo, cred_hi = sr_shrunk - Z95 * post_sd, sr_shrunk + Z95 * post_sd
    covered_ci = (theta >= ci_lo) & (theta <= ci_hi)
    covered_cred = (theta >= cred_lo) & (theta <= cred_hi)
    cover_ci_all[r] = covered_ci.mean()
    cover_cred_all[r] = covered_cred.mean()

    top_raw = np.argsort(sr_hat)[-TOP_N:]
    top_shrunk = np.argsort(sr_shrunk)[-TOP_N:]
    cover_ci_top[r] = covered_ci[top_raw].mean()
    cover_cred_top[r] = covered_cred[top_raw].mean()
    mean_theta_top_raw[r] = theta[top_raw].mean()
    mean_theta_top_shrunk[r] = theta[top_shrunk].mean()
    frac_shortest_top_raw[r] = float(np.mean(years_arr[top_raw] == min(YEARS_GROUPS)))
    frac_shortest_top_shrunk[r] = float(np.mean(years_arr[top_shrunk] == min(YEARS_GROUPS)))

    if r == 0:  # the concrete cross-section shown on the page (charts, group table)
        headline = dict(theta=theta.copy(), sr_hat=sr_hat.copy(), sr_shrunk=sr_shrunk.copy(),
                        se=se.copy(), k=k.copy(), years_arr=years_arr.copy(),
                        mu0_hat=mu0_hat, tau0_hat=np.sqrt(tau0_hat_sq),
                        top_raw=top_raw, top_shrunk=top_shrunk)

# ---- Monte Carlo checks on the R_REPS replications ---------------------------------------------
def paired_check(a: np.ndarray, b: np.ndarray, msg: str) -> tuple[float, float]:
    """Mean and SE of (a - b) across replications; both must already be replication-level values."""
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


diff_theta, se_diff_theta = paired_check(mse_raw_theta, mse_shrunk_theta, "theta")
assert diff_theta > 4 * se_diff_theta, "shrinking should lower MSE against the true (synthetic) Sharpe"
diff_oos, se_diff_oos = paired_check(mse_raw_oos, mse_shrunk_oos, "oos")
assert diff_oos > 4 * se_diff_oos, "shrinking should also lower MSE against each fund's own future performance"

cov_ci_mean, cov_ci_se = float(cover_ci_all.mean()), float(cover_ci_all.std(ddof=1) / np.sqrt(R_REPS))
assert abs(cov_ci_mean - 0.95) < 4 * cov_ci_se + 0.01, "an honestly-specified 95% CI should cover about 95% of the time, unconditionally"

diff_top_cov, se_top_cov = paired_check(cover_ci_all, cover_ci_top, "top coverage gap")
assert diff_top_cov > 4 * se_top_cov, "coverage among the SELECTED top decile should be materially worse"
assert cover_ci_top.mean() < 0.85, "the plain CI's coverage among 'impressive' funds should collapse well below 95%"

diff_cred_vs_ci_top, se_cred_vs_ci_top = paired_check(cover_cred_top, cover_ci_top, "credible vs CI, top decile")
assert diff_cred_vs_ci_top > 4 * se_cred_vs_ci_top, "the credible interval should cover better than the plain CI among the selected funds"

diff_pick, se_pick = paired_check(mean_theta_top_shrunk, mean_theta_top_raw, "pick quality")
assert diff_pick > 4 * se_pick, "ranking by the shrunk estimate should surface funds with higher TRUE skill on average"

diff_short, se_short = paired_check(frac_shortest_top_raw, frac_shortest_top_shrunk, "short-history bias")
assert diff_short > 4 * se_short, "the raw leaderboard should over-represent the shortest track records"

print(f"MSE vs truth: raw {mse_raw_theta.mean():.4f} vs shrunk {mse_shrunk_theta.mean():.4f} "
      f"({diff_theta / mse_raw_theta.mean():.1%} lower)")
print(f"MSE vs OOS:   raw {mse_raw_oos.mean():.4f} vs shrunk {mse_shrunk_oos.mean():.4f} "
      f"({diff_oos / mse_raw_oos.mean():.1%} lower)")
print(f"95% CI coverage: {cov_ci_mean:.1%} overall, {cover_ci_top.mean():.1%} among the top {TOP_FRAC:.0%} by raw score "
      f"vs {cover_cred_top.mean():.1%} by credible interval")
print(f"top-{TOP_FRAC:.0%} true skill: raw pick {mean_theta_top_raw.mean():.3f} vs shrunk pick {mean_theta_top_shrunk.mean():.3f}")
print(f"share of top-{TOP_FRAC:.0%} that are the shortest-history group: raw {frac_shortest_top_raw.mean():.1%} "
      f"vs shrunk {frac_shortest_top_shrunk.mean():.1%} (even split would be {100 / len(YEARS_GROUPS):.0f}%)")

lab.record("n_funds", N_FUNDS)
lab.record("n_per_group", N_PER_GROUP)
lab.record("t_oos_years", T_OOS_YEARS)
lab.record("top_frac", TOP_FRAC)
lab.record("top_n", TOP_N)
lab.record("r_reps", R_REPS)
lab.record("mse_raw_theta", float(mse_raw_theta.mean()))
lab.record("mse_shrunk_theta", float(mse_shrunk_theta.mean()))
lab.record("mse_reduction_theta_pct", diff_theta / float(mse_raw_theta.mean()))
lab.record("mse_raw_oos", float(mse_raw_oos.mean()))
lab.record("mse_shrunk_oos", float(mse_shrunk_oos.mean()))
lab.record("mse_reduction_oos_pct", diff_oos / float(mse_raw_oos.mean()))
lab.record("cover_ci_all", cov_ci_mean)
lab.record("cover_cred_all", float(cover_cred_all.mean()))
lab.record("cover_ci_top", float(cover_ci_top.mean()))
lab.record("cover_cred_top", float(cover_cred_top.mean()))
lab.record("mean_theta_top_raw", float(mean_theta_top_raw.mean()))
lab.record("mean_theta_top_shrunk", float(mean_theta_top_shrunk.mean()))
lab.record("frac_shortest_top_raw", float(frac_shortest_top_raw.mean()))
lab.record("frac_shortest_top_shrunk", float(frac_shortest_top_shrunk.mean()))
lab.record("even_share_pct", 1.0 / len(YEARS_GROUPS))
lab.record("tau0_hat_avg", float(tau0_hats.mean()))

# per-group average shrinkage weight, from the headline cross-section (for the "why" column of the table)
for y in YEARS_GROUPS:
    mask = headline["years_arr"] == y
    lab.record(f"weight_{y}y", float(headline["k"][mask].mean()))

# %% [markdown]
# ## 4 · Charts: the headline cross-section

# %%
lab.chart("shrink_scatter", charts.scatter_chart(
    headline["sr_hat"], headline["sr_shrunk"],
    title="Raw vs shrunk Sharpe estimate, 200 synthetic funds", diagonal=True,
    x_label="raw (sample) Sharpe", y_label="shrunk (empirical-Bayes) Sharpe",
    x_fmt=charts.fmt_num(1), y_fmt=charts.fmt_num(1)))

lab.chart("mse_bars", charts.grouped_column_chart(
    ["vs true skill (synthetic only)", "vs each fund's own next 3 years"],
    [("raw estimate", [float(mse_raw_theta.mean()), float(mse_raw_oos.mean())], "loss"),
     ("shrunk estimate", [float(mse_shrunk_theta.mean()), float(mse_shrunk_oos.mean())], "strategy")],
    title="Mean squared error: trusting the raw Sharpe vs shrinking it", y_fmt=charts.fmt_num(3),
    value_labels=True))

lab.chart("coverage_bars", charts.grouped_column_chart(
    ["all 200 funds", f"top {TOP_FRAC:.0%} by raw Sharpe"],
    [("95% confidence interval", [cov_ci_mean, float(cover_ci_top.mean())], "loss"),
     ("95% credible interval", [float(cover_cred_all.mean()), float(cover_cred_top.mean())], "strategy")],
    title="How often the interval actually covers the true Sharpe", y_fmt=charts.fmt_pct(0),
    value_labels=True, y_min=0.0, y_max=1.0))

# %%
lab.save()
