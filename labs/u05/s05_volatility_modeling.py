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
# # 5.5 · Modeling Volatility — companion lab
#
# **Quant Notebook** · Unit 5 · Session 5 · [Read the session](https://georgevrs.github.io/quant-notebook/unit05-statistics-time-series/session05-volatility-modeling.html)
#
# Realized-volatility estimators from OHLC data, EWMA and GARCH forecasts — volatility is the one thing you can forecast well.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Two questions, two experiments. First: given a day's open, high, low and close, which formula
# recovers the true volatility with the least noise — and what happens to each formula the moment
# the market gaps overnight? Second: once you have a volatility *history*, how do you forecast
# tomorrow's — a fixed-decay average (EWMA) or a fitted model that knows its own long-run level
# (GARCH), and does it matter whether bad news hits harder than good news (GJR-GARCH)? All series
# are synthetic with a known true volatility, so every estimator can be graded against the truth.
# There is no risk-free rate anywhere in this lab — every "return" here is used only to measure
# volatility, never to compute a Sharpe ratio or a P&L.

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
from arch import arch_model

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR
from quantnb.synth import garch_returns

lab = qn.Lab("5.5")  # seeds the random generators: every run gives the same numbers
# One independent random stream per experiment (Session 1.5's rule): editing or re-running one
# section never shifts the random numbers another section sees.
(rng_ohlc, rng_garch, rng_gjr) = lab.rng.spawn(3)
rng_ohlc_ideal, rng_ohlc_gap = rng_ohlc.spawn(2)

DT_DAY = 1.0 / PERIODS_PER_YEAR
N_SE = 4                                  # every Monte Carlo check is made at 4 standard errors
MC_CHECKS: list[tuple[str, float]] = []   # (name, z-score) of every statistical check, for the record

lab.record("n_se", N_SE)


def check_mc(name: str, estimate: float, truth: float, se: float, n_se: float = N_SE) -> float:
    """Assert a Monte Carlo (or MLE) estimate lies within n_se standard errors of its target.

    Works for a simulation mean against its theoretical target (se = sample std / sqrt(n)) or a
    fitted parameter against the data-generating truth (se = the model's own MLE standard error).
    """
    z = (estimate - truth) / se
    MC_CHECKS.append((name, float(z)))
    assert abs(z) < n_se, f"{name}: {estimate:.6g} vs {truth:.6g} is {z:+.2f} SE away"
    return float(z)


# One GARCH(1,1) series, simulated once and reused by §1 (a quick illustration), §5 (EWMA) and §6
# (the fitted model) — the same series throughout, never three different draws of "a GARCH path".
N_DAYS_GARCH = 2_500                      # ~9.9 years of daily data
OMEGA_G, ALPHA_G, BETA_G = 2e-6, 0.08, 0.90
garch_sim = garch_returns(N_DAYS_GARCH, omega=OMEGA_G, alpha=ALPHA_G, beta=BETA_G, rng=rng_garch)
ret_g = garch_sim["ret"].to_numpy()
true_vol_annual = garch_sim["sigma"].to_numpy() * np.sqrt(PERIODS_PER_YEAR)
longrun_var_true = OMEGA_G / (1 - ALPHA_G - BETA_G)
longrun_vol_true = float(np.sqrt(longrun_var_true * PERIODS_PER_YEAR))
lab.record("garch_omega_true", OMEGA_G)
lab.record("garch_alpha_true", ALPHA_G)
lab.record("garch_beta_true", BETA_G)
lab.record("longrun_vol_true", longrun_vol_true)
lab.record("n_days_garch", N_DAYS_GARCH)

# %% [markdown]
# ## 1 · Returns are unpredictable; their *size* is not
#
# A quick illustration before the main experiments, using the GARCH series built above: the lag-1
# autocorrelation of raw returns is indistinguishable from zero (Sessions 5.2, 5.4), but the lag-1
# autocorrelation of *squared* returns is not — tomorrow's sign is a coin flip, tomorrow's typical
# move size is not. That gap is what the rest of this lab measures (§2-4) and forecasts (§5-7).

# %%
ret_autocorr = float(pd.Series(ret_g).autocorr(1))
sq_autocorr = float(pd.Series(ret_g ** 2).autocorr(1))
lab.record("ret_autocorr", ret_autocorr)
lab.record("sq_autocorr", sq_autocorr)
print(f"lag-1 autocorrelation: returns {ret_autocorr:+.3f}, squared returns {sq_autocorr:+.3f}")
assert abs(ret_autocorr) < 0.05          # raw returns: no detectable memory
assert sq_autocorr > 5 * abs(ret_autocorr)  # squared returns: unmistakably more memory

# %% [markdown]
# ## 2 · Simulating an OHLC price process
#
# Parkinson, Garman-Klass and Yang-Zhang all need a high and a low, which a simple daily-return
# simulator never produces. We simulate each day at `M_INTRADAY` steps of geometric Brownian
# motion (Session 3.5's exact log-space scheme) and keep the running max/min — that IS how a real
# exchange's OHLC bar is built, just from ticks instead of steps. A separate, independent overnight
# shock can jump the next day's open away from today's close, exactly like a real market gapping
# outside continuous trading hours.

# %%
N_TRIALS, CHUNK = 20_000, 2_000
lab.record("n_trials", N_TRIALS)
N_DAYS, M_INTRADAY = 21, 200              # a one-month realized-vol window, 200 steps per day
MU_OHLC = 0.07                            # annual drift; irrelevant to variance at daily scale
SIGMA_TRUE = 0.20                         # the number every estimator is trying to recover
FRAC_INTRADAY = 0.70                      # share of TOTAL daily variance that happens intraday
SIGMA_INTRADAY = SIGMA_TRUE * np.sqrt(FRAC_INTRADAY)
SIGMA_OVERNIGHT = SIGMA_TRUE * np.sqrt(1 - FRAC_INTRADAY)
lab.record("sigma_true", SIGMA_TRUE)
lab.record("sigma_intraday", SIGMA_INTRADAY)
lab.record("sigma_overnight", SIGMA_OVERNIGHT)
lab.record("frac_intraday", FRAC_INTRADAY)
lab.record("n_days_window", N_DAYS)
lab.record("m_intraday", M_INTRADAY)


def simulate_ohlc(rng: np.random.Generator, n_trials: int, sigma_intraday: float, sigma_overnight: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One month (N_DAYS) of daily OHLC bars, in LOG PRICE, for n_trials independent windows.

    Returns (open, high, low, close), each shape (n_trials, N_DAYS). Differences of these values
    ARE log returns, so every estimator below is just arithmetic on the four arrays — no further
    log() needed. mu drops out: at one day's scale its effect on the range is negligible (Session
    2.3's volatility drag is O(mu*dt) against a range that is O(sigma*sqrt(dt)), a ratio of ~0.02
    here), which is exactly why Parkinson and Garman-Klass can ignore drift in practice.
    """
    rng_intra, rng_jump = rng.spawn(2)
    dt_step = DT_DAY / M_INTRADAY
    nu = MU_OHLC - 0.5 * sigma_intraday ** 2
    z = rng_intra.standard_normal((n_trials, N_DAYS, M_INTRADAY))
    incr = nu * dt_step + sigma_intraday * np.sqrt(dt_step) * z
    cum = np.cumsum(incr, axis=2)
    full = np.concatenate([np.zeros((n_trials, N_DAYS, 1)), cum], axis=2)
    day_high_rel = full.max(axis=2)
    day_low_rel = full.min(axis=2)
    day_logret = cum[:, :, -1]
    jump = sigma_overnight * np.sqrt(DT_DAY) * rng_jump.standard_normal((n_trials, N_DAYS))
    open_abs = np.empty((n_trials, N_DAYS))
    close_abs = np.empty((n_trials, N_DAYS))
    open_abs[:, 0] = jump[:, 0]                              # gap from an unmodelled prior close = 0
    close_abs[:, 0] = open_abs[:, 0] + day_logret[:, 0]
    for t in range(1, N_DAYS):                                # N_DAYS = 21: a cheap python loop
        open_abs[:, t] = close_abs[:, t - 1] + jump[:, t]
        close_abs[:, t] = open_abs[:, t] + day_logret[:, t]
    return open_abs, open_abs + day_high_rel, open_abs + day_low_rel, close_abs


def close_close_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev_c = np.concatenate([np.zeros((c.shape[0], 1)), c[:, :-1]], axis=1)
    return (c - prev_c).var(axis=1, ddof=1) / DT_DAY


def parkinson_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Parkinson (1980): the squared high-low range, scaled by 1/(4 ln 2)."""
    return ((h - l) ** 2).mean(axis=1) / (4 * np.log(2)) / DT_DAY


def garman_klass_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Garman & Klass (1980): adds the open-close move to Parkinson's range term."""
    term = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    return term.mean(axis=1) / DT_DAY


def rogers_satchell_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Rogers & Satchell (1991): unbiased under drift; the building block inside Yang-Zhang."""
    term = (h - o) * (h - c) + (l - o) * (l - c)
    return term.mean(axis=1) / DT_DAY


def yang_zhang_var(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Yang & Zhang (2000): overnight variance + a weighted open-close/Rogers-Satchell blend.

    Uses only O, H, L, C — never the hidden overnight "jump" itself — exactly what a real OHLC
    history gives you.
    """
    n = o.shape[1]
    prev_c = np.concatenate([np.zeros((o.shape[0], 1)), c[:, :-1]], axis=1)
    v_o = (o - prev_c).var(axis=1, ddof=1) / DT_DAY
    v_c = (c - o).var(axis=1, ddof=1) / DT_DAY
    v_rs = rogers_satchell_var(o, h, l, c)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return v_o + k * v_c + (1 - k) * v_rs


def run_experiment(rng: np.random.Generator, sigma_intraday: float, sigma_overnight: float) -> dict:
    """N_TRIALS independent one-month windows, chunked so memory stays small.

    Returns the RAW components each estimator is built from: close-close variance, Parkinson,
    Garman-Klass, Rogers-Satchell, and Yang-Zhang's two other pieces (overnight and open-close).
    """
    out = {k: [] for k in ("cc", "park", "gk", "rs", "vo", "vc")}
    for _ in range(N_TRIALS // CHUNK):
        o, h, l, c = simulate_ohlc(rng, CHUNK, sigma_intraday, sigma_overnight)
        out["cc"].append(close_close_var(o, h, l, c))
        out["park"].append(parkinson_var(o, h, l, c))
        out["gk"].append(garman_klass_var(o, h, l, c))
        out["rs"].append(rogers_satchell_var(o, h, l, c))
        prev_c = np.concatenate([np.zeros((o.shape[0], 1)), c[:, :-1]], axis=1)
        out["vo"].append((o - prev_c).var(axis=1, ddof=1) / DT_DAY)
        out["vc"].append((c - o).var(axis=1, ddof=1) / DT_DAY)
    return {k: np.concatenate(v) for k, v in out.items()}


K_YZ = 0.34 / (1.34 + (N_DAYS + 1) / (N_DAYS - 1))
lab.record("yz_k", K_YZ)


def assemble_yz(comp: dict, rs: np.ndarray) -> np.ndarray:
    return comp["vo"] + K_YZ * comp["vc"] + (1 - K_YZ) * rs


# %% [markdown]
# ## 3 · Relative efficiency, the classical (no-gap) case
#
# First reproduce the textbook experiment: no overnight jump, so every estimator targets the same
# `SIGMA_TRUE`, and the only question is how noisy each one is. "Relative efficiency" means the
# ratio of sampling *variances* of the variance estimate: an estimator that is 5x more efficient
# has a fifth of the estimation error for the same data — or needs a fifth as many days for the
# same precision.
#
# A close-to-close return only ever compares two closes, so it has no discretisation error at all.
# Sampling the running high/low at `M_INTRADAY` steps instead of continuously very slightly
# UNDER-samples the true extremes — the same discrete-monitoring gap as the barrier-touch
# correction in Sessions 2.4/3.5 — which shows up as a small, systematic downward bias in every
# estimator that touches H or L. We measure it once here and calibrate it away before §4's bias
# comparison, so that section isolates the overnight-gap effect alone.

# %%
ideal = run_experiment(rng_ohlc_ideal, SIGMA_TRUE, 0.0)
se_cc = float(ideal["cc"].std(ddof=1) / np.sqrt(N_TRIALS))
se_vc = float(ideal["vc"].std(ddof=1) / np.sqrt(N_TRIALS))
check_mc("ideal close-close unbiased", float(ideal["cc"].mean()), SIGMA_TRUE ** 2, se_cc)
check_mc("ideal open-close (no H/L) unbiased", float(ideal["vc"].mean()), SIGMA_TRUE ** 2, se_vc)

CORR = {name: SIGMA_TRUE ** 2 / float(ideal[name].mean()) for name in ("park", "gk", "rs")}
for name, corr in CORR.items():
    lab.record(f"corr_{name}", corr)
    assert 1.0 < corr < 1.30, f"{name}: discretisation correction {corr:.3f} outside the expected small range"
print("discretisation correction factors (200 steps/day):", {k: f"{v:.3f}" for k, v in CORR.items()})

ideal_yz_raw = assemble_yz(ideal, ideal["rs"])
var_of_var = {name: float(ideal[name].var(ddof=1)) for name in ("cc", "park", "gk")}
var_of_var["yz"] = float(ideal_yz_raw.var(ddof=1))
eff_park = var_of_var["cc"] / var_of_var["park"]
eff_gk = var_of_var["cc"] / var_of_var["gk"]
eff_yz = var_of_var["cc"] / var_of_var["yz"]
lab.record("eff_park", eff_park)
lab.record("eff_gk", eff_gk)
lab.record("eff_yz", eff_yz)
print(f"relative efficiency vs close-close: Parkinson {eff_park:.2f}x, Garman-Klass {eff_gk:.2f}x, "
      f"Yang-Zhang {eff_yz:.2f}x")
# Textbook figures (no drift, no gap): Parkinson up to ~5.2x, Garman-Klass up to ~7.4x, Yang-Zhang
# up to ~14x (citations in the session page). Ours use a 200-step daily discretisation and 20,000
# trials rather than the papers' continuous-time limit, so we check the REGIME, not the exact figure
# — and the ordering, which theory guarantees regardless of discretisation.
assert 3.0 < eff_park < 7.0
assert eff_gk > eff_park            # Garman-Klass strictly uses more of the bar than Parkinson
assert 4.0 < eff_gk < 10.0
assert eff_yz > eff_park            # Yang-Zhang is built from the same information as GK/RS

lab.chart("estimator_efficiency", charts.column_chart(
    ["close-close", "Parkinson", "Garman-Klass", "Yang-Zhang"], [1.0, eff_park, eff_gk, eff_yz],
    title="Relative efficiency vs close-to-close, no overnight gaps (20,000 simulated months)",
    y_fmt=charts.fmt_num(1, suffix="×"), roles=["benchmark", "strategy", "alt1", "alt2"]))

# %% [markdown]
# ## 4 · What an overnight gap does to each estimator
#
# Real markets trade a few hours a day and gap over the rest. We rebuild the experiment with 30%
# of the day's TOTAL variance moved overnight (`SIGMA_OVERNIGHT`) and 70% left intraday
# (`SIGMA_INTRADAY`) — the split is a modelling choice for this synthetic demonstration, not a
# claim about real markets. Parkinson, Garman-Klass and Rogers-Satchell only ever look at a single
# day's own O, H, L, C, so they cannot see a gap that happened *before* that day's open: they
# target `SIGMA_INTRADAY`, not `SIGMA_TRUE`. Yang-Zhang adds an explicit overnight-variance term
# and recovers `SIGMA_TRUE` in expectation, whatever the split. Figures below apply §3's
# discretisation correction, so the only bias left on display is the real, economic one.

# %%
gapped = run_experiment(rng_ohlc_gap, SIGMA_INTRADAY, SIGMA_OVERNIGHT)
gap_park = gapped["park"] * CORR["park"]
gap_gk = gapped["gk"] * CORR["gk"]
gap_rs = gapped["rs"] * CORR["rs"]
gap_yz = assemble_yz(gapped, gap_rs)
se = lambda arr: float(arr.std(ddof=1) / np.sqrt(N_TRIALS))  # noqa: E731

check_mc("gapped close-close matches the TRUE total", float(gapped["cc"].mean()), SIGMA_TRUE ** 2, se(gapped["cc"]))
check_mc("gapped Parkinson matches intraday-only", float(gap_park.mean()), SIGMA_INTRADAY ** 2, se(gap_park))
check_mc("gapped Garman-Klass matches intraday-only", float(gap_gk.mean()), SIGMA_INTRADAY ** 2, se(gap_gk))
check_mc("gapped Yang-Zhang matches the TRUE total", float(gap_yz.mean()), SIGMA_TRUE ** 2, se(gap_yz))

for name, arr in (("cc", gapped["cc"]), ("park", gap_park), ("gk", gap_gk), ("yz", gap_yz)):
    lab.record(f"gap_{name}_vol", float(np.sqrt(arr.mean())))

# The bias is economic, not a simulation artefact: even the RAW (uncorrected) Parkinson formula
# misses the TRUE total by far more than the tiny discretisation correction could ever explain.
m_park_raw, se_park_raw = float(gapped["park"].mean()), se(gapped["park"])
bias_z = abs(m_park_raw - SIGMA_TRUE ** 2) / se_park_raw
lab.record("park_bias_z", bias_z)
lab.record("park_bias_pp", 100 * (SIGMA_TRUE - np.sqrt(gap_park.mean())))
assert bias_z > 50 * N_SE, "the overnight-gap bias should dwarf sampling noise, not hide in it"

lab.chart("bias_with_gaps", charts.column_chart(
    ["true total σ", "true intraday-only σ", "close-close", "Parkinson", "Garman-Klass", "Yang-Zhang"],
    [SIGMA_TRUE, SIGMA_INTRADAY, float(np.sqrt(gapped["cc"].mean())), float(np.sqrt(gap_park.mean())),
     float(np.sqrt(gap_gk.mean())), float(np.sqrt(gap_yz.mean()))],
    title="Estimated annual volatility with a 30%-overnight, 70%-intraday split",
    y_fmt=charts.fmt_pct(1), roles=["benchmark", "benchmark", "strategy", "loss", "loss", "strategy"]))

# %% [markdown]
# ## 5 · EWMA: one decay parameter, no free lunch
#
# RiskMetrics' exponentially weighted moving average updates variance with a single parameter
# lambda: `sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}`. Small lambda reacts fast
# to new information but is noisy; large lambda is smooth but slow to admit a real regime change —
# there is no lambda that is both. We measure this trade-off against the GARCH(1,1) series from
# §1, whose TRUE conditional volatility path we know exactly.

# %%
def ewma_var(returns: np.ndarray, lam: float) -> np.ndarray:
    """RiskMetrics recursion, initialised at the whole sample's variance."""
    var = np.empty(len(returns))
    var[0] = returns.var(ddof=1)
    for t in range(1, len(returns)):
        var[t] = lam * var[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return var


BURN_IN = 100                              # skip the initialisation transient when scoring RMSE
LAMBDAS = [0.90, 0.94, 0.97]
ewma_rmse, ewma_half_life = {}, {}
for lam in LAMBDAS:
    vol_lam = np.sqrt(ewma_var(ret_g, lam) * PERIODS_PER_YEAR)
    ewma_rmse[lam] = float(np.sqrt(np.mean((vol_lam[BURN_IN:] - true_vol_annual[BURN_IN:]) ** 2)))
    ewma_half_life[lam] = float(np.log(0.5) / np.log(lam))
    lab.record(f"ewma_{int(lam * 100)}_rmse_pp", 100 * ewma_rmse[lam])
    lab.record(f"ewma_{int(lam * 100)}_half_life", ewma_half_life[lam])
print({lam: f"RMSE {ewma_rmse[lam]:.4f}, half-life {ewma_half_life[lam]:.1f}d" for lam in LAMBDAS})
assert ewma_half_life[0.90] < ewma_half_life[0.94] < ewma_half_life[0.97]  # monotone by construction

best_lam = min(LAMBDAS, key=lambda x: ewma_rmse[x])
lab.record("ewma_best_lambda", best_lam)
lab.record("ewma_94_vol_last", float(np.sqrt(ewma_var(ret_g, 0.94)[-1] * PERIODS_PER_YEAR)))

# %% [markdown]
# ## 6 · GARCH(1,1): a fitted alternative to a fixed decay
#
# GARCH(1,1) adds a constant `omega`, so its variance mean-reverts to `omega/(1-alpha-beta)`
# instead of drifting forever like EWMA's (EWMA is GARCH with `alpha+beta` pinned to exactly 1 — an
# "integrated" GARCH, IGARCH). We fit it by maximum likelihood with the `arch` package rather than
# hand-rolling one, and check that the fit recovers the TRUE parameters within the model's own
# standard errors.

# %%
am = arch_model(ret_g * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="t")  # x100: arch's own scaling advice
res = am.fit(disp="off")
omega_hat, alpha_hat, beta_hat = res.params["omega"] / 100 ** 2, res.params["alpha[1]"], res.params["beta[1]"]
omega_se, alpha_se, beta_se = res.std_err["omega"] / 100 ** 2, res.std_err["alpha[1]"], res.std_err["beta[1]"]
check_mc("GARCH omega recovered", omega_hat, OMEGA_G, omega_se)
check_mc("GARCH alpha recovered", alpha_hat, ALPHA_G, alpha_se)
check_mc("GARCH beta recovered", beta_hat, BETA_G, beta_se)
lab.record("garch_omega_hat", omega_hat)
lab.record("garch_alpha_hat", alpha_hat)
lab.record("garch_beta_hat", beta_hat)

fitted_vol_annual = np.asarray(res.conditional_volatility) / 100 * np.sqrt(PERIODS_PER_YEAR)
garch_rmse = float(np.sqrt(np.mean((fitted_vol_annual[BURN_IN:] - true_vol_annual[BURN_IN:]) ** 2)))
lab.record("garch_rmse_pp", 100 * garch_rmse)
lab.record("ewma_best_rmse_pp", 100 * ewma_rmse[best_lam])
assert garch_rmse < ewma_rmse[best_lam], "a correctly specified fitted model should beat every fixed lambda"

# A genuine multi-step forecast (arch's own `.forecast`), vs EWMA's forecast, which never moves:
# IGARCH's conditional-variance forecast is flat at today's level for every horizon by construction.
FORECAST_H = 60
fc = res.forecast(horizon=FORECAST_H, reindex=False)
forecast_garch_vol = np.sqrt(fc.variance.to_numpy()[-1] / 100 ** 2 * PERIODS_PER_YEAR)
forecast_ewma_vol = np.full(FORECAST_H, np.sqrt(ewma_var(ret_g, 0.94)[-1] * PERIODS_PER_YEAR))
lab.record("forecast_garch_1d", float(forecast_garch_vol[0]))
lab.record("forecast_garch_60d", float(forecast_garch_vol[-1]))
lab.record("forecast_ewma_flat", float(forecast_ewma_vol[0]))
lab.record("forecast_horizon_days", FORECAST_H)
gap_garch = abs(forecast_garch_vol[-1] - longrun_vol_true)
gap_ewma = abs(forecast_ewma_vol[-1] - longrun_vol_true)
lab.record("forecast_gap_garch_pp", 100 * gap_garch)
lab.record("forecast_gap_ewma_pp", 100 * gap_ewma)
assert gap_garch < gap_ewma, "GARCH's forecast should drift toward the long-run level; EWMA's cannot"

window = slice(N_DAYS_GARCH - 500, N_DAYS_GARCH)
t_years = np.arange(1, 501) / PERIODS_PER_YEAR
lab.chart("vol_paths", charts.line_chart(
    [charts.Series("true conditional vol", pd.Series(true_vol_annual[window], index=t_years), role="benchmark", end_label="true σ"),
     charts.Series("EWMA (λ=0.94)", pd.Series(np.sqrt(ewma_var(ret_g, 0.94) * PERIODS_PER_YEAR)[window], index=t_years), role="alt1", end_label="EWMA"),
     charts.Series("fitted GARCH(1,1)", pd.Series(fitted_vol_annual[window], index=t_years), role="strategy", end_label="GARCH")],
    title="Tracking the true conditional volatility: EWMA vs a fitted GARCH(1,1) (last 500 days)",
    y_fmt=charts.fmt_pct(0)))

days_ahead = np.arange(1, FORECAST_H + 1) / PERIODS_PER_YEAR
lab.chart("forecast_horizon", charts.line_chart(
    [charts.Series("EWMA forecast (flat)", pd.Series(forecast_ewma_vol, index=days_ahead), role="alt1", end_label="EWMA"),
     charts.Series("GARCH(1,1) forecast", pd.Series(forecast_garch_vol, index=days_ahead), role="strategy", end_label="GARCH")],
    title=f"{FORECAST_H}-trading-day-ahead volatility forecast from the same day",
    y_fmt=charts.fmt_pct(0), hline=longrun_vol_true, hline_label="true long-run vol"))

# %% [markdown]
# ## 7 · GJR-GARCH: letting bad news move volatility more
#
# We simulate a process where negative shocks raise next period's variance by more than
# equally-sized positive ones — the asymmetry Session 5.1 calls the leverage effect — using the
# same GARCH(1,1) recursion plus one extra term, gated on the shock's sign. This is teaching code:
# `quantnb.synth` only has the symmetric case, so `gjr_garch_returns` is written out in full here
# (flagged in the handoff for promotion).

# %%
def gjr_garch_returns(n_days: int, omega: float, alpha: float, gamma: float, beta: float,
                      mu: float = 0.0003, nu: float = 6.0, rng: np.random.Generator | None = None
                      ) -> pd.DataFrame:
    """GJR-GARCH(1,1) (Glosten, Jagannathan & Runkle, 1993): var += gamma * eps^2 when eps < 0.

    Stationary iff alpha + beta + gamma/2 < 1 (the 1/2 is E[indicator] under a symmetric shock).
    Same shape and Student-t convention as quantnb.synth.garch_returns.
    """
    rng = rng or np.random.default_rng()
    z = rng.standard_t(nu, n_days) / np.sqrt(nu / (nu - 2.0))
    var = np.empty(n_days)
    eps = np.empty(n_days)
    var[0] = omega / (1.0 - alpha - beta - gamma / 2)
    for t in range(n_days):
        if t > 0:
            var[t] = omega + alpha * eps[t - 1] ** 2 + gamma * eps[t - 1] ** 2 * (eps[t - 1] < 0) + beta * var[t - 1]
        eps[t] = np.sqrt(var[t]) * z[t]
    return pd.DataFrame({"ret": mu + eps, "sigma": np.sqrt(var)})


N_DAYS_GJR = 6_000                        # a longer sample than §6: gamma is harder to pin down than alpha, beta
OMEGA_J, ALPHA_J, GAMMA_J, BETA_J = 6e-6, 0.03, 0.09, 0.88
assert ALPHA_J + BETA_J + GAMMA_J / 2 < 1
lab.record("n_days_gjr", N_DAYS_GJR)
gjr_sim = gjr_garch_returns(N_DAYS_GJR, OMEGA_J, ALPHA_J, GAMMA_J, BETA_J, rng=rng_gjr)
ret_j = gjr_sim["ret"].to_numpy()
lab.record("gjr_omega_true", OMEGA_J)
lab.record("gjr_alpha_true", ALPHA_J)
lab.record("gjr_gamma_true", GAMMA_J)
lab.record("gjr_beta_true", BETA_J)
lab.record("gjr_alpha_plus_gamma_true", ALPHA_J + GAMMA_J)

am_gjr = arch_model(ret_j * 100, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
res_gjr = am_gjr.fit(disp="off")
o_hat, a_hat, g_hat, b_hat = (res_gjr.params["omega"] / 100 ** 2, res_gjr.params["alpha[1]"],
                             res_gjr.params["gamma[1]"], res_gjr.params["beta[1]"])
o_se, a_se, g_se, b_se = (res_gjr.std_err["omega"] / 100 ** 2, res_gjr.std_err["alpha[1]"],
                         res_gjr.std_err["gamma[1]"], res_gjr.std_err["beta[1]"])
check_mc("GJR-GARCH omega recovered", o_hat, OMEGA_J, o_se)
check_mc("GJR-GARCH alpha recovered", a_hat, ALPHA_J, a_se)
check_mc("GJR-GARCH gamma recovered", g_hat, GAMMA_J, g_se)
check_mc("GJR-GARCH beta recovered", b_hat, BETA_J, b_se)
lab.record("gjr_omega_hat", o_hat)
lab.record("gjr_alpha_hat", a_hat)
lab.record("gjr_gamma_hat", g_hat)
lab.record("gjr_beta_hat", b_hat)
gamma_z = g_hat / g_se
lab.record("gjr_gamma_z", gamma_z)
assert gamma_z > N_SE, "the asymmetry should be detected reliably, not sit near zero"

# What happens if you fit the WRONG (symmetric) model to this asymmetric data?
am_wrong = arch_model(ret_j * 100, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
res_wrong = am_wrong.fit(disp="off")
alpha_wrong = res_wrong.params["alpha[1]"]
lab.record("gjr_alpha_wrong_fit", alpha_wrong)
# A symmetric model can only fit the AVERAGE reaction; a shock's average impact under the true
# model is alpha + gamma * P(eps<0) = alpha + gamma/2.
lab.record("gjr_alpha_wrong_theory", ALPHA_J + GAMMA_J / 2)

# The next-period variance implied by a shock of a given size, holding today's variance at its
# long-run level — the response the wrong (symmetric) fit cannot make asymmetric.
sigma_lr_j = np.sqrt((OMEGA_J / (1 - ALPHA_J - BETA_J - GAMMA_J / 2)))
sigma_lr_wrong = np.sqrt(res_wrong.params["omega"] / 100 ** 2 / (1 - alpha_wrong - res_wrong.params["beta[1]"]))
shock_grid = np.linspace(-4, 4, 41) * sigma_lr_j
resp_true = OMEGA_J + ALPHA_J * shock_grid ** 2 + GAMMA_J * shock_grid ** 2 * (shock_grid < 0) + BETA_J * sigma_lr_j ** 2
resp_wrong = (res_wrong.params["omega"] / 100 ** 2 + alpha_wrong * shock_grid ** 2
             + res_wrong.params["beta[1]"] * sigma_lr_wrong ** 2)
resp_gjr = o_hat + a_hat * shock_grid ** 2 + g_hat * shock_grid ** 2 * (shock_grid < 0) + b_hat * sigma_lr_j ** 2
shock_in_sigmas = shock_grid / sigma_lr_j
lab.chart("news_response", charts.line_chart(
    [charts.Series("true model", pd.Series(np.sqrt(resp_true * PERIODS_PER_YEAR), index=shock_in_sigmas), role="benchmark", end_label="true"),
     charts.Series("symmetric GARCH fit", pd.Series(np.sqrt(resp_wrong * PERIODS_PER_YEAR), index=shock_in_sigmas), role="loss", end_label="GARCH"),
     charts.Series("GJR-GARCH fit", pd.Series(np.sqrt(resp_gjr * PERIODS_PER_YEAR), index=shock_in_sigmas), role="strategy", end_label="GJR")],
    title="Next-day volatility implied by a shock of a given size (in long-run daily σ)",
    y_fmt=charts.fmt_pct(1), max_points=41))

# %%
lab.save()
