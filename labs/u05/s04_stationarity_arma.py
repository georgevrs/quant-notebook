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
# # 5.4 · Stationarity & ARMA Models — companion lab
#
# **Quant Notebook** · Unit 5 · Session 4 · [Read the session](https://georgevrs.github.io/quant-notebook/unit05-statistics-time-series/session04-stationarity-arma.html)
#
# Unit roots, ADF and KPSS, AR/MA/ARMA, ACF and PACF — and why returns are close to unpredictable
# while prices are not stationary.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
# All series here are synthetic (`quantnb.synth` and statsmodels' own ARMA generator) — no market
# data is fetched or committed. There is no P&L or Sharpe ratio in this lab, so no risk-free rate applies.

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

import quantnb as qn
from quantnb.charts import Series, fmt_num, fmt_pct, grouped_column_chart, line_chart
from quantnb.stats import batch_se
from quantnb.synth import gbm_prices
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

# statsmodels warns loudly about near-unit-root starting values and interpolated KPSS p-values —
# both are expected here (we are deliberately probing the boundary), so we quiet them once.
warnings.filterwarnings("ignore")

lab = qn.Lab("5.4")  # seeds the random generators: every run gives the same numbers

# one independent RNG stream per experiment, so changing one section never changes another's numbers
rng_pricevret, rng_boundary, rng_acfpacf, rng_fit, rng_aicbic = lab.rng.spawn(5)

ALPHA = 0.05  # the significance level used throughout this lab
TRUE_PHI, TRUE_THETA = 0.6, -0.4  # the running ARMA(1,1) example used in Sections 3-5


def arma(n, phi=0.0, theta=0.0, rng=None, burn=500):
    """Simulate an ARMA(1,1) (AR(1)/MA(1) as special cases) via statsmodels' generator.

    statsmodels' sign convention: ar=[1, -phi], ma=[1, theta] gives
    x_t = phi * x_{t-1} + eps_t + theta * eps_{t-1}, eps_t ~ N(0, 1).
    """
    return arma_generate_sample(ar=[1.0, -phi], ma=[1.0, theta], nsample=n,
                                distrvs=rng.standard_normal, burnin=burn)


# %% [markdown]
# ## 1 · Prices are a unit root, returns are (close to) stationary
#
# The session's headline claim, made precise: simulate GBM price paths, test the log PRICE level
# and the log RETURN series separately with both ADF (null: unit root) and KPSS (null: stationary),
# and see the size and power of each test line up exactly where theory says they should.

# %%
N_PATHS = 200
N_DAYS = 500  # ~2 trading years

adf_price_p, kpss_price_p, adf_ret_p, kpss_ret_p = [], [], [], []
for _ in range(N_PATHS):
    prices = gbm_prices(N_DAYS, mu=0.06, sigma=0.20, rng=rng_pricevret)["price"].to_numpy()
    log_price = np.log(prices)
    log_ret = np.diff(log_price)
    adf_price_p.append(adfuller(log_price, autolag="AIC", result_object=True).pvalue)
    kpss_price_p.append(kpss(log_price, regression="c", nlags="auto", result_object=True).pvalue)
    adf_ret_p.append(adfuller(log_ret, autolag="AIC", result_object=True).pvalue)
    kpss_ret_p.append(kpss(log_ret, regression="c", nlags="auto", result_object=True).pvalue)

adf_price_p, kpss_price_p = np.array(adf_price_p), np.array(kpss_price_p)
adf_ret_p, kpss_ret_p = np.array(adf_ret_p), np.array(kpss_ret_p)

rate_adf_reject_price = float((adf_price_p < ALPHA).mean())    # ADF's SIZE: log-price truly has a unit root
rate_kpss_reject_price = float((kpss_price_p < ALPHA).mean())  # KPSS's POWER against that same truth
rate_adf_reject_ret = float((adf_ret_p < ALPHA).mean())        # ADF's POWER: log returns are truly stationary
rate_kpss_reject_ret = float((kpss_ret_p < ALPHA).mean())      # KPSS's SIZE against that same truth

for key, val in [("rate_adf_reject_price", rate_adf_reject_price), ("rate_kpss_reject_price", rate_kpss_reject_price),
                 ("rate_adf_reject_ret", rate_adf_reject_ret), ("rate_kpss_reject_ret", rate_kpss_reject_ret)]:
    lab.record(key, val)

# ADF should tell price and returns apart: almost never reject the unit root on price levels (its
# correct SIZE under a true null) but almost always reject it on returns (its POWER).
se_adf = batch_se(np.mean, (adf_ret_p < ALPHA).astype(float), n_batches=20)
assert rate_adf_reject_ret - rate_adf_reject_price > 4 * se_adf, \
    "ADF should favour 'stationary' for returns far more often than for price levels"
# KPSS should show the mirror-image pattern.
se_kpss = batch_se(np.mean, (kpss_price_p < ALPHA).astype(float), n_batches=20)
assert rate_kpss_reject_price - rate_kpss_reject_ret > 4 * se_kpss, \
    "KPSS should favour 'non-stationary' for price levels far more often than for returns"
# Both tests, run at the 5% level under a TRUE null, should reject close to 5% of the time (calibration).
assert rate_adf_reject_price < 0.20, "ADF's false-rejection rate under a true unit-root null should stay near its 5% size"
assert rate_kpss_reject_ret < 0.20, "KPSS's false-rejection rate under a true stationarity null should stay near its 5% size"

# %% [markdown]
# ## 2 · Near the boundary, ADF and KPSS genuinely disagree
#
# Sweep the AR(1) coefficient phi from clearly stationary (0.50) to an exact unit root (1.00) and,
# at every phi, run both tests on many replications. Record each test's own rejection rate, and —
# the point everyone misses — how often the two tests flatly CONTRADICT each other on the very same series.

# %%
PHI_GRID = [0.50, 0.80, 0.90, 0.95, 0.97, 0.99, 1.00]
N_REPS = 150
T_BOUNDARY = 300

adf_rate_by_phi, kpss_rate_by_phi = [], []
contradict_by_phi = {}
for phi in PHI_GRID:
    adf_p = np.empty(N_REPS)
    kpss_p = np.empty(N_REPS)
    for i in range(N_REPS):
        x = arma(T_BOUNDARY, phi=phi, rng=rng_boundary)
        adf_p[i] = adfuller(x, regression="c", autolag="AIC", result_object=True).pvalue
        kpss_p[i] = kpss(x, regression="c", nlags="auto", result_object=True).pvalue
    adf_rate_by_phi.append(float((adf_p < ALPHA).mean()))
    kpss_rate_by_phi.append(float((kpss_p < ALPHA).mean()))
    contradict_by_phi[phi] = ((adf_p < ALPHA) & (kpss_p < ALPHA)).astype(float)

PHI_BOUNDARY = 0.90  # empirically where the two curves cross and contradictions peak
contradict_boundary = contradict_by_phi[PHI_BOUNDARY]
contradict_baseline = contradict_by_phi[0.50]
rate_contradict_boundary = float(contradict_boundary.mean())
rate_contradict_baseline = float(contradict_baseline.mean())
lab.record("rate_contradict_boundary", rate_contradict_boundary)
lab.record("rate_contradict_baseline", rate_contradict_baseline)
lab.record("phi_boundary", PHI_BOUNDARY)
lab.record("halflife_boundary", float(np.log(0.5) / np.log(PHI_BOUNDARY)))  # periods to decay halfway back to the mean

se_boundary = batch_se(np.mean, contradict_boundary, n_batches=15)
se_baseline = batch_se(np.mean, contradict_baseline, n_batches=15)
se_diff = float(np.sqrt(se_boundary ** 2 + se_baseline ** 2))
assert rate_contradict_boundary - rate_contradict_baseline > 4 * se_diff, \
    f"ADF and KPSS should contradict each other far more often near phi={PHI_BOUNDARY} than at phi=0.50"

boundary_chart = line_chart(
    [Series("ADF (H0: unit root)", pd.Series(adf_rate_by_phi, index=PHI_GRID), role="alt1"),
     Series("KPSS (H0: stationary)", pd.Series(kpss_rate_by_phi, index=PHI_GRID), role="alt2")],
    title="How often each test rejects its own null, as the AR(1) coefficient approaches 1",
    y_fmt=fmt_pct(0), hline=0.05, hline_label="5% significance", height=300, y_min=0.0, y_max=1.0,
)
lab.chart("adf_kpss_boundary", boundary_chart)

# %% [markdown]
# ## 3 · Reading the ACF and PACF of a known ARMA(1,1)
#
# One long draw of the running example (phi=0.6, theta=-0.4) so the sample ACF/PACF are precise
# enough to show the textbook signature: neither one cuts off cleanly.

# %%
T_LONG = 2000
NLAGS = 12
x_long = arma(T_LONG, phi=TRUE_PHI, theta=TRUE_THETA, rng=rng_acfpacf)

acf_vals = acf(x_long, nlags=NLAGS, fft=True)[1:]
pacf_vals = pacf(x_long, nlags=NLAGS)[1:]
sig_bound = 1.96 / np.sqrt(T_LONG)

lab.record("acf_lag1", float(acf_vals[0]))
lab.record("pacf_lag1", float(pacf_vals[0]))
lab.record("pacf_lag2", float(pacf_vals[1]))
lab.record("acf_sig_bound", float(sig_bound))
lab.record("acf_pacf_n", T_LONG)

# The lag-1 ACF/PACF of an ARMA(1,1) should sit near its closed-form value: rho(1) =
# (1+phi*theta)(phi+theta) / (1+2*phi*theta+theta^2).
theory_rho1 = (1 + TRUE_PHI * TRUE_THETA) * (TRUE_PHI + TRUE_THETA) / (1 + 2 * TRUE_PHI * TRUE_THETA + TRUE_THETA ** 2)
se_acf1 = batch_se(lambda b: acf(np.asarray(b).ravel(), nlags=1, fft=True)[1], x_long, n_batches=40)
assert abs(acf_vals[0] - theory_rho1) < 4 * se_acf1, "sample ACF(1) should sit within 4 SE of the theoretical ARMA(1,1) value"

acf_pacf_chart = grouped_column_chart(
    [str(k) for k in range(1, NLAGS + 1)],
    [("ACF", list(acf_vals), "alt1"), ("PACF", list(pacf_vals), "alt2")],
    title="ACF and PACF of a simulated ARMA(1,1) — neither one cuts off",
    y_fmt=fmt_num(2), height=280,
)
lab.chart("acf_pacf", acf_pacf_chart)

# %% [markdown]
# ## 4 · Fitting ARMA(1,1) with statsmodels
#
# A single, realistically sized draw (300 observations) fit with the correct order, checked the
# way you should always check a fitted ARMA model: are the residuals left with no structure at all?

# %%
T_FIT = 300
x_fit = arma(T_FIT, phi=TRUE_PHI, theta=TRUE_THETA, rng=rng_fit)
fit = ARIMA(x_fit, order=(1, 0, 1), trend="n").fit()
phi_hat, theta_hat = float(fit.params[0]), float(fit.params[1])
se_phi_hat, se_theta_hat = float(fit.bse[0]), float(fit.bse[1])
lb = acorr_ljungbox(fit.resid, lags=[10], return_df=True)
ljung_box_p = float(lb["lb_pvalue"].iloc[0])

lab.record("phi_hat", phi_hat)
lab.record("theta_hat", theta_hat)
lab.record("se_phi_hat", se_phi_hat)
lab.record("se_theta_hat", se_theta_hat)
lab.record("ljung_box_p", ljung_box_p)

# Sanity checks (one seeded draw, generous tolerances — this proves the pipeline works, it is not
# a precision claim; Section 5 below is where the Monte Carlo evidence on recovering the order lives).
assert abs(phi_hat - TRUE_PHI) < 0.3, "fitted phi should land in the right neighbourhood of the true value"
assert abs(theta_hat - TRUE_THETA) < 0.3, "fitted theta should land in the right neighbourhood of the true value"
assert ljung_box_p > ALPHA, "residuals of the correctly specified model should look like white noise"

# %% [markdown]
# ## 5 · Choosing the order: when AIC and BIC disagree
#
# Refit five candidate orders on 200 fresh draws of the same ARMA(1,1) and record, per replication,
# which order AIC prefers and which order BIC prefers. BIC's log(T) penalty is stiffer than AIC's
# flat 2 — the textbook prediction is that BIC leans further towards the smaller model.

# %%
ORDERS = [(0, 0, 1), (1, 0, 0), (1, 0, 1), (2, 0, 1), (1, 0, 2)]
ORDER_LABELS = ["MA(1)", "AR(1)", "ARMA(1,1) true", "ARMA(2,1)", "ARMA(1,2)"]
TRUE_ORDER = (1, 0, 1)
SIMPLE_ORDER = (1, 0, 0)
N_MC = 200
T_MC = 300

aic_hits = {o: 0 for o in ORDERS}
bic_hits = {o: 0 for o in ORDERS}
aic_true_hits = np.zeros(N_MC)
bic_true_hits = np.zeros(N_MC)
aic_simple_hits = np.zeros(N_MC)
bic_simple_hits = np.zeros(N_MC)
for i in range(N_MC):
    x = arma(T_MC, phi=TRUE_PHI, theta=TRUE_THETA, rng=rng_aicbic)
    aics, bics = {}, {}
    for order in ORDERS:
        m = ARIMA(x, order=order, trend="n").fit()
        aics[order], bics[order] = m.aic, m.bic
    best_aic = min(aics, key=aics.get)
    best_bic = min(bics, key=bics.get)
    aic_hits[best_aic] += 1
    bic_hits[best_bic] += 1
    aic_true_hits[i] = best_aic == TRUE_ORDER
    bic_true_hits[i] = best_bic == TRUE_ORDER
    aic_simple_hits[i] = best_aic == SIMPLE_ORDER
    bic_simple_hits[i] = best_bic == SIMPLE_ORDER

aic_rate = [aic_hits[o] / N_MC for o in ORDERS]
bic_rate = [bic_hits[o] / N_MC for o in ORDERS]

lab.record("aic_true_rate", float(aic_true_hits.mean()))
lab.record("bic_true_rate", float(bic_true_hits.mean()))
lab.record("aic_simple_rate", float(aic_simple_hits.mean()))
lab.record("bic_simple_rate", float(bic_simple_hits.mean()))
lab.record("order_mc_n", N_MC)

se_true_diff = batch_se(np.mean, aic_true_hits - bic_true_hits, n_batches=20)
assert aic_true_hits.mean() - bic_true_hits.mean() > 4 * se_true_diff, \
    "AIC should recover the true ARMA(1,1) order more often than BIC at T=300"
se_simple_diff = batch_se(np.mean, bic_simple_hits - aic_simple_hits, n_batches=20)
assert bic_simple_hits.mean() - aic_simple_hits.mean() > 4 * se_simple_diff, \
    "BIC should over-select the simpler AR(1) more often than AIC at T=300"

aic_bic_chart = grouped_column_chart(
    ORDER_LABELS,
    [("AIC picks", aic_rate, "alt1"), ("BIC picks", bic_rate, "alt2")],
    title="How often AIC and BIC pick each candidate order (true order: ARMA(1,1))",
    y_fmt=fmt_pct(0), value_labels=True, height=300,
)
lab.chart("aic_bic_selection", aic_bic_chart)

# %%
lab.save()
