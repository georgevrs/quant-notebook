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
# # 5.1 · Stylized Facts of Returns — companion lab
#
# **Quant Notebook** · Unit 5 · Session 1 · [Read the session](https://georgevrs.github.io/quant-notebook/unit05-statistics-time-series/session01-stylized-facts.html)
#
# The empirical regularities every model must respect — fat tails, volatility clustering, the
# leverage effect, and near-zero autocorrelation.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Everything here is synthetic and studies the SHAPE of returns, not their level or a strategy's
# P&L, so no risk-free rate enters. Two markets share the same unconditional daily volatility
# (1%) and the same total variance persistence (0.98) by construction; the only difference is
# whether a shock's SIGN changes tomorrow's variance. Monte Carlo statistics are checked against
# theory (or against each other) at 4 standard errors, computed across independent simulated
# markets — never a property of one lucky path, and never a seed that was hunted for.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys
import warnings

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

import quantnb as qn
from quantnb import charts
from quantnb.returns import in_years, simple_to_log

lab = qn.Lab("5.1")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment: the symmetric market, the asymmetric (leverage) market, and
# the bootstrap in section 4 never share a stream, so tweaking one in the Try-it exercise cannot
# silently change another section's numbers.
rng_sym, rng_asym, rng_boot = lab.rng.spawn(3)

# %% [markdown]
# ## 0 · A GJR-GARCH market (quantnb.synth lacks the asymmetric term)
#
# `quantnb.synth.garch_returns` already gives a symmetric GARCH(1,1) market — the same one
# Session 3.2 used to show fat tails, with `omega=2e-6, alpha=0.08, beta=0.90, nu=6`. This session
# needs an ASYMMETRIC extension — Glosten, Jagannathan & Runkle's (1993) GJR-GARCH(1,1), which lets
# a negative shock raise tomorrow's variance more than a positive shock of the same size. `gamma=0`
# reduces it exactly to the existing GARCH(1,1) recursion, so this one function replaces
# `synth.garch_returns` for this lab (vectorised over independent paths, which the original does
# not need). Written inline; requested for promotion into `quantnb.synth` (see the handoff).


def gjr_garch_returns(n_days: int, n_paths: int, omega: float, alpha: float, gamma: float,
                      beta: float, mu: float, nu: float, rng: np.random.Generator):
    """GJR-GARCH(1,1) daily simple returns, Student-t shocks, shape (n_days, n_paths).

    sigma^2_t = omega + (alpha + gamma * 1{eps_{t-1} < 0}) * eps_{t-1}^2 + beta * sigma^2_{t-1}

    gamma=0 is exactly quantnb.synth.garch_returns's symmetric GARCH(1,1). Stationarity (with a
    symmetric shock, so P(shock < 0) = 1/2) needs alpha + beta + gamma/2 < 1.
    """
    persistence = alpha + beta + 0.5 * gamma
    if persistence >= 1:
        raise ValueError("alpha + beta + gamma/2 must be < 1 for a stationary GJR-GARCH(1,1)")
    z = rng.standard_t(nu, (n_days, n_paths)) / np.sqrt(nu / (nu - 2.0))  # unit variance
    var = np.empty((n_days, n_paths))
    eps = np.empty((n_days, n_paths))
    var[0] = omega / (1.0 - persistence)
    for t in range(n_days):
        if t > 0:
            neg = eps[t - 1] < 0.0
            var[t] = omega + (alpha + gamma * neg) * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * z[t]
    return mu + eps, np.sqrt(var)


OMEGA, MU, NU = 2e-6, 0.0003, 6.0                       # same as Session 3.2's GARCH-t market
ALPHA_SYM, BETA_SYM, GAMMA_SYM = 0.08, 0.90, 0.0        # symmetric: 3.2's market exactly
ALPHA_ASYM, BETA_ASYM, GAMMA_ASYM = 0.03, 0.90, 0.10    # asymmetric: same total persistence
N_PATHS, N_DAYS = 200, 6300                             # 200 independent simulated 25-year markets

persistence_sym = ALPHA_SYM + BETA_SYM
persistence_asym = ALPHA_ASYM + BETA_ASYM + 0.5 * GAMMA_ASYM
assert abs(persistence_sym - persistence_asym) < 1e-12, "the two markets are built to have equal persistence"
daily_vol_uncond = float(np.sqrt(OMEGA / (1.0 - persistence_sym)))
assert abs(daily_vol_uncond - 0.01) < 1e-6, "both markets are built to have 1% unconditional daily vol"

lab.record("omega", OMEGA)
lab.record("alpha_sym", ALPHA_SYM)
lab.record("beta_sym", BETA_SYM)
lab.record("alpha_asym", ALPHA_ASYM)
lab.record("gamma_asym", GAMMA_ASYM)
lab.record("beta_asym", BETA_ASYM)
lab.record("nu", NU)
lab.record("persistence", persistence_sym)
lab.record("daily_vol_uncond", daily_vol_uncond)
lab.record("ann_vol_uncond", daily_vol_uncond * np.sqrt(252))
lab.record("n_paths", N_PATHS)
lab.record("n_days_mc", N_DAYS)
lab.record("n_years_mc", N_DAYS / 252)

ret_sym, sigma_sym = gjr_garch_returns(N_DAYS, N_PATHS, OMEGA, ALPHA_SYM, GAMMA_SYM, BETA_SYM, MU, NU, rng_sym)
ret_asym, sigma_asym = gjr_garch_returns(N_DAYS, N_PATHS, OMEGA, ALPHA_ASYM, GAMMA_ASYM, BETA_ASYM, MU, NU, rng_asym)

# %% [markdown]
# ## 1 · Near-zero autocorrelation of raw returns
#
# A GARCH shock is `sigma_t * z_t` with `z_t` independent of the past, so `E[eps_t | past] = 0`
# and the autocorrelation of RAW returns is exactly zero in population — even though the returns
# are not independent (section 2). We check this on the symmetric market: the lag-1
# autocorrelation, averaged across 200 independent markets, should be indistinguishable from zero.

# %%
acf1_raw = np.array([acf(ret_sym[:, i], nlags=1, fft=True)[1] for i in range(N_PATHS)])
acf1_mean, acf1_se = float(acf1_raw.mean()), float(acf1_raw.std(ddof=1) / np.sqrt(N_PATHS))
bartlett_band = float(1.0 / np.sqrt(N_DAYS))  # the textbook +-1.96/sqrt(n) band assumes iid noise
lab.record("acf1_mean", acf1_mean)
lab.record("acf1_se", acf1_se)
lab.record("bartlett_band", bartlett_band)
assert abs(acf1_mean) < 4 * acf1_se, "raw-return autocorrelation should be indistinguishable from zero"

# Ljung-Box on raw returns, lag 10: this tests for ANY linear autocorrelation up to lag 10 at once.
# It assumes iid noise under its null, which GARCH returns are not (they ARE uncorrelated, just
# not independent) -- so its true rejection rate under the null can miss its nominal 5%.
lb_pvals = np.array([acorr_ljungbox(ret_sym[:, i], lags=[10], return_df=True)["lb_pvalue"].iloc[0]
                     for i in range(N_PATHS)])
lb_reject_rate = float((lb_pvals < 0.05).mean())
lb_reject_se = float(np.sqrt(lb_reject_rate * (1.0 - lb_reject_rate) / N_PATHS))
lab.record("lb_reject_rate", lb_reject_rate)
lab.record("lb_reject_se", lb_reject_se)
# Fat tails and clustering inflate Var(eps_t * eps_{t-1}) far past the iid value sigma^4 that the
# textbook test assumes, so its true rejection rate under a genuinely zero-autocorrelation null
# sits well above its nominal 5% -- decisively, not as sampling noise.
assert lb_reject_rate - 4 * lb_reject_se > 0.05, "heteroskedasticity should inflate the true rejection rate past the nominal 5%"

# %% [markdown]
# ## 2 · Volatility clustering: correlated variance, uncorrelated returns
#
# Now look at SQUARED returns. Variance depends on yesterday's shock and yesterday's variance, so
# it inherits their memory: the autocorrelation function (ACF) of squared returns decays only
# slowly, at a rate set by `alpha + beta` per lag (the ARMA(1,1) representation of a GARCH(1,1)'s
# squared residuals, Bollerslev 1986) -- Session 5.4 covers the ACF/PACF machinery formally.

# %%
def acf_lags(x: np.ndarray, nlags: int) -> np.ndarray:
    return acf(x, nlags=nlags, fft=True)[1:]


acf_raw_by_path = np.array([acf_lags(ret_sym[:, i], 30) for i in range(N_PATHS)])
acf_sq_by_path = np.array([acf_lags(ret_sym[:, i] ** 2, 30) for i in range(N_PATHS)])
mean_acf_raw = acf_raw_by_path.mean(axis=0)
mean_acf_sq = acf_sq_by_path.mean(axis=0)
for lag in (1, 5, 10, 30):
    lab.record(f"acf_sq_lag{lag}", float(mean_acf_sq[lag - 1]))
    lab.record(f"acf_raw_lag{lag}", float(mean_acf_raw[lag - 1]))

# The sample ACF of squared returns is itself a noisy, heavy-tailed statistic this close to a
# GARCH(1,1)'s moment boundary (as Session 3.2 found for this same market), so instead of reading
# the persistence off a log-linear fit to noisy lags, RECOVER it the way practitioners do: fit a
# GARCH(1,1) by maximum likelihood with `arch` on a sample of independent markets.

# %%
from arch import arch_model  # noqa: E402  (kept near its uses)

warnings.filterwarnings("ignore", category=FutureWarning)
N_FIT = 40
persist_est = np.array([
    sum(arch_model(100 * ret_sym[:, i], mean="Zero", vol="GARCH", p=1, q=1, dist="t")
        .fit(disp="off").params[["alpha[1]", "beta[1]"]])
    for i in range(N_FIT)
])
persist_mean, persist_se = float(persist_est.mean()), float(persist_est.std(ddof=1) / np.sqrt(N_FIT))
lab.record("persist_fit_mean", persist_mean)
lab.record("persist_fit_se", persist_se)
lab.record("n_fit", N_FIT)
assert abs(persist_mean - persistence_sym) < 4 * persist_se, "MLE should recover alpha+beta"

half_life = float(np.log(0.5) / np.log(persistence_sym))
lab.record("half_life_days", half_life)
assert half_life > 20, "at alpha+beta=0.98 a volatility shock should take weeks, not days, to fade"

lab.chart("acf_clustering", charts.grouped_column_chart(
    [str(k) for k in (1, 2, 3, 5, 10, 15, 20, 25, 30)],
    [("raw returns", [float(mean_acf_raw[k - 1]) for k in (1, 2, 3, 5, 10, 15, 20, 25, 30)], "benchmark"),
     ("squared returns", [float(mean_acf_sq[k - 1]) for k in (1, 2, 3, 5, 10, 15, 20, 25, 30)], "strategy")],
    title="Autocorrelation by lag (trading days): raw vs squared returns, averaged over 200 markets",
    y_fmt=charts.fmt_num(2)))

# A single market's simulated volatility path, for the intuition: calm and stormy spells cluster.
vis_years = 3
vis_vol = in_years(pd.Series(sigma_sym[:vis_years * 252, 0] * np.sqrt(252)))
lab.chart("vol_over_time", charts.line_chart(
    [charts.Series("annualised daily volatility", vis_vol, role="strategy")],
    title=f"One simulated market's volatility over {vis_years} years: it comes in bursts",
    y_fmt=charts.fmt_pct(0)))

# %% [markdown]
# ## 3 · The leverage effect: does the SIGN of a shock matter, not just its size?
#
# In the symmetric market, `Cov(r_t, sigma^2_{t+1})` is exactly zero: a shock's cube has zero mean
# under a symmetric distribution, whatever its sign. In the asymmetric (GJR) market, the extra
# `gamma * eps_t^2 * 1{eps_t<0}` term makes that covariance strictly negative when `gamma > 0` --
# a down day should be followed, on average, by higher variance than an up day of the same size.
# This IS the leverage effect (Black 1976; Christie 1982), stated as a testable statistic rather
# than a picture.

# %%
def next_var_corr(ret: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-path correlation of today's return with TOMORROW's variance."""
    r_t, var_tp1 = ret[:-1], sigma[1:] ** 2
    return np.array([np.corrcoef(r_t[:, i], var_tp1[:, i])[0, 1] for i in range(r_t.shape[1])])


corr_sym = next_var_corr(ret_sym, sigma_sym)
corr_asym = next_var_corr(ret_asym, sigma_asym)
corr_sym_mean, corr_sym_se = float(corr_sym.mean()), float(corr_sym.std(ddof=1) / np.sqrt(N_PATHS))
corr_asym_mean, corr_asym_se = float(corr_asym.mean()), float(corr_asym.std(ddof=1) / np.sqrt(N_PATHS))
lab.record("corr_sym_mean", corr_sym_mean)
lab.record("corr_sym_se", corr_sym_se)
lab.record("corr_asym_mean", corr_asym_mean)
lab.record("corr_asym_se", corr_asym_se)
assert abs(corr_sym_mean) < 4 * corr_sym_se, "symmetric market: no return/future-variance relation"
assert corr_asym_mean < -4 * corr_asym_se, "asymmetric market: reliably negative"


def vol_after_sign(ret: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    r_t, vol_tp1 = ret[:-1], sigma[1:] * np.sqrt(252)  # annualised, for readability
    return float(vol_tp1[r_t < 0].mean()), float(vol_tp1[r_t > 0].mean())


down_sym, up_sym = vol_after_sign(ret_sym, sigma_sym)
down_asym, up_asym = vol_after_sign(ret_asym, sigma_asym)
lab.record("vol_after_down_sym", down_sym)
lab.record("vol_after_up_sym", up_sym)
lab.record("vol_after_down_asym", down_asym)
lab.record("vol_after_up_asym", up_asym)

lab.chart("leverage_vol", charts.grouped_column_chart(
    ["after a down day", "after an up day"],
    [("symmetric market", [down_sym, up_sym], "benchmark"),
     ("asymmetric market", [down_asym, up_asym], "strategy")],
    title="Annualised volatility the day after a down day vs an up day",
    y_fmt=charts.fmt_pct(0)))

# %% [markdown]
# ### Recovering gamma with `arch` (not reimplementing GARCH estimation)
#
# Fit a real GJR-GARCH(1,1) — `arch_model(..., vol="GARCH", p=1, o=1, q=1)` — to one path from
# each market (returns rescaled by 100, `arch`'s convention). The asymmetry term should come back
# significant only where it is really there.

# %%
fit_sym = arch_model(100 * ret_sym[:, 0], mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
fit_asym = arch_model(100 * ret_asym[:, 0], mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")

lab.record("fit_sym_gamma", float(fit_sym.params["gamma[1]"]))
lab.record("fit_sym_gamma_p", float(fit_sym.pvalues["gamma[1]"]))
lab.record("fit_asym_omega", float(fit_asym.params["omega"]) / 10_000.0)
lab.record("fit_asym_alpha", float(fit_asym.params["alpha[1]"]))
lab.record("fit_asym_gamma", float(fit_asym.params["gamma[1]"]))
lab.record("fit_asym_beta", float(fit_asym.params["beta[1]"]))
lab.record("fit_asym_nu", float(fit_asym.params["nu"]))
lab.record("fit_asym_gamma_p", float(fit_asym.pvalues["gamma[1]"]))
assert fit_asym.pvalues["gamma[1]"] < 0.01, "the true asymmetry should be detected decisively"
assert fit_sym.pvalues["gamma[1]"] > 0.05, "no false positive on a market with no real asymmetry"
assert abs(fit_asym.params["beta[1]"] - BETA_ASYM) < 0.05, "beta is usually the best-identified GARCH parameter"
# alpha and gamma are individually noisy from a single ~6,300-day fit (a known GARCH estimation
# fact, not a bug); the lab records them for the page's table without a tight closed-form check.

# %% [markdown]
# ## 4 · Aggregational Gaussianity: fat tails fade as you zoom out
#
# Sum LOG returns (they add across time; Session 2.3) over longer and longer horizons and watch
# the excess kurtosis of the sum shrink towards zero, the normal's value. For a sum of `h`
# (weakly dependent) shocks, the excess kurtosis of the total falls off roughly like `1/h`,
# because the 4th cumulant scales with `h` while variance squared scales with `h^2`.

# %%
def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    d = x - x.mean()
    return float((d ** 4).mean() / (d ** 2).mean() ** 2 - 3.0)


# With nu=6 shocks this close to a GARCH(1,1)'s moment boundary, a single path's excess kurtosis
# is itself a heavy-tailed statistic (Session 3.2 found the same for this market: its own 8th
# moment does not exist), so the MEAN across independent paths, and its std/sqrt(n) standard
# error, are not trustworthy. The MEDIAN across independent paths is far more robust to the rare
# huge value, and a bootstrap of that median gives a well-behaved standard error instead.
logret_sym = simple_to_log(ret_sym)
HORIZON_DAYS = {"1d": 1, "1w": 5, "1m": 21, "1q": 63}
N_BOOT = 2_000
kurt_median, kurt_boot_se = {}, {}
for name, h in HORIZON_DAYS.items():
    n_blocks = N_DAYS // h
    agg = logret_sym[:n_blocks * h].reshape(n_blocks, h, N_PATHS).sum(axis=1)
    per_path = np.array([excess_kurtosis(agg[:, i]) for i in range(N_PATHS)])
    kurt_median[name] = float(np.median(per_path))
    boot = [np.median(rng_boot.choice(per_path, size=N_PATHS, replace=True)) for _ in range(N_BOOT)]
    kurt_boot_se[name] = float(np.std(boot, ddof=1))
    lab.record(f"exkurt_{name}", kurt_median[name])
    lab.record(f"exkurt_{name}_se", kurt_boot_se[name])

gap_se = np.sqrt(kurt_boot_se["1d"] ** 2 + kurt_boot_se["1q"] ** 2)
assert kurt_median["1d"] - kurt_median["1q"] > 4 * gap_se, "median kurtosis should fall decisively from a day to a quarter"
assert kurt_median["1q"] < kurt_median["1d"] / 3.0, "a quarter's returns should look much closer to normal"

lab.chart("aggregational_gaussianity", charts.column_chart(
    ["1 day", "1 week", "1 month", "1 quarter"],
    [kurt_median[n] for n in HORIZON_DAYS],
    title="Median excess kurtosis falls as returns are summed over longer horizons",
    y_fmt=charts.fmt_num(1)))

# %%
lab.save()
