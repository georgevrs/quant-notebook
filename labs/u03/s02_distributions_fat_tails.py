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
# # 3.2 · Distributions & Fat Tails — companion lab
#
# **Quant Notebook** · Unit 3 · Session 2 · [Read the session](https://georgevrs.github.io/quant-notebook/unit03-math-toolkit/session02-distributions-fat-tails.html)
#
# Normal, lognormal and Student-t; skewness, kurtosis and tail indices — and why the normal distribution will eventually lie to you.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# As in Session 3.1, simulated results are checked against exact calculations wherever one
# exists, allowing 4 standard errors of the simulation; where none exists (the GARCH-t market),
# the lab checks that the gap from the normal is larger than 4 standard errors. This lab makes
# about 50 such checks; at 4 SE the chance that any one fails by luck is small — and never hunt
# for a seed that passes.
# Everything here is synthetic: returns are daily, there is no interest rate (risk-free rate 0),
# and a "k-sigma fall" means a daily return more than k standard deviations BELOW its mean.

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
from scipy import integrate, stats

import quantnb as qn
from quantnb import charts

lab = qn.Lab("3.2")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
(rng_lognorm, rng_t, rng_garch, rng_garch_twin, rng_jump, rng_jump_twin,
 rng_hill, rng_cauchy) = lab.rng.spawn(8)
DAYS_PER_YEAR = 252


def excess_kurtosis(x) -> float:
    """Sample excess kurtosis (population form): E[(x − mean)⁴] / var² − 3; zero for a normal."""
    x = np.asarray(x, dtype=float).ravel()
    d = x - x.mean()
    return float((d ** 4).mean() / (d ** 2).mean() ** 2 - 3.0)


def se_share(p: float, n: int) -> float:
    """Standard error of a simulated frequency (share of n independent draws) with true value p."""
    return float(np.sqrt(p * (1 - p) / n))


# %% [markdown]
# ## 1 · Three shapes: normal, lognormal and Student-t
#
# **Lognormal prices.** If log returns add up over time (Session 2.3) and their sum is roughly
# normal, the price itself is lognormal: P = P_0 · exp(X) with X ~ N(m, s²). We use the
# convention of `qn.synth.gbm_prices`: expected simple growth μ a year, so over τ years
# m = (μ − σ²/2)τ and s = σ√τ. The lognormal is skewed to the right, and the skew grows with the horizon.

# %%
MU_ANN, SIGMA_ANN = 0.07, 0.20       # illustrative stock-index growth rate and volatility, per year
START_PRICE = 100.0
HORIZONS = {"1d": 1 / DAYS_PER_YEAR, "1y": 1.0, "10y": 10.0, "30y": 30.0}   # in years
SIM_HORIZON = "10y"                  # the horizon we also simulate
N_DRAWS = 1_000_000


def lognormal_stats(m: float, s: float) -> dict[str, float]:
    """Exact moments of exp(X), X ~ N(m, s²): the textbook lognormal formulas."""
    w = np.exp(s ** 2)
    return {
        "mean": np.exp(m + s ** 2 / 2),
        "median": np.exp(m),
        "mode": np.exp(m - s ** 2),
        "sd": np.sqrt((w - 1) * np.exp(2 * m + s ** 2)),
        "skew": (w + 2) * np.sqrt(w - 1),
        "exkurt": w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 6,
    }


rows = []
for h, tau in HORIZONS.items():
    m, s = (MU_ANN - SIGMA_ANN ** 2 / 2) * tau, SIGMA_ANN * np.sqrt(tau)
    ln = lognormal_stats(m, s)
    row = {"horizon": h, "mean": START_PRICE * ln["mean"], "median": START_PRICE * ln["median"],
           "mode": START_PRICE * ln["mode"], "skew": ln["skew"], "exkurt": ln["exkurt"],
           "p_below_mean": stats.norm.cdf(s / 2),          # P(exp(X) < E[exp(X)]) = Φ(s/2)
           "p_below_start": stats.norm.cdf(-m / s)}         # P(price ends below where it started)
    rows.append(row)
    for key, v in row.items():
        if key != "horizon":
            lab.record(f"ln_{h}_{key}", float(v))
lognormal_table = pd.DataFrame(rows).set_index("horizon")
print(lognormal_table.round(4))

# simulate the 10-year price a million times and check the formulas
tau = HORIZONS[SIM_HORIZON]
m, s = (MU_ANN - SIGMA_ANN ** 2 / 2) * tau, SIGMA_ANN * np.sqrt(tau)
ln = lognormal_stats(m, s)
price = START_PRICE * np.exp(m + s * rng_lognorm.standard_normal(N_DRAWS))
se_mean = START_PRICE * ln["sd"] / np.sqrt(N_DRAWS)
pdf_at_median = 1 / (START_PRICE * ln["median"] * s * np.sqrt(2 * np.pi))
se_median = np.sqrt(0.25 / N_DRAWS) / pdf_at_median          # SE of a sample median
se_skew = qn.stats.batch_se(qn.stats.skewness, price)
p_below = stats.norm.cdf(-m / s)
assert abs(price.mean() - START_PRICE * ln["mean"]) < 4 * se_mean, "lognormal mean = exp(m + s²/2)"
assert abs(np.median(price) - START_PRICE * ln["median"]) < 4 * se_median, "lognormal median = exp(m)"
assert abs(qn.stats.skewness(price) - ln["skew"]) < 4 * se_skew, "lognormal skewness formula"
assert abs((price < START_PRICE).mean() - p_below) < 4 * se_share(p_below, N_DRAWS)
assert START_PRICE * ln["mode"] < START_PRICE * ln["median"] < START_PRICE * ln["mean"]  # right skew
print(f"10-year price: mean {price.mean():.2f} (theory {START_PRICE * ln['mean']:.2f}), "
      f"median {np.median(price):.2f} (theory {START_PRICE * ln['median']:.2f}), "
      f"skewness {qn.stats.skewness(price):.2f} ± {se_skew:.2f} (theory {ln['skew']:.2f})")
lab.record("ln_mu", MU_ANN)
lab.record("ln_sigma", SIGMA_ANN)
lab.record("ln_start", START_PRICE)
lab.record("ln_n_draws", N_DRAWS)
lab.record("ln_sim_mean", float(price.mean()))
lab.record("ln_sim_median", float(np.median(price)))
lab.record("ln_sim_skew", qn.stats.skewness(price))
lab.record("ln_sim_skew_se", se_skew)

# %% [markdown]
# **Student-t returns.** A Student-t with ν degrees of freedom has variance ν/(ν − 2), so to use
# it as a return distribution with volatility σ we rescale it to unit variance, multiplying by
# √((ν − 2)/ν). Its excess kurtosis is 6/(ν − 4) for ν > 4 and infinite for ν ≤ 4; moments of
# order ν and above do not exist. Compared with a normal of the same variance it has a taller
# peak, thinner shoulders and far fatter tails.

# %%
NU_TABLE = [3, 4, 5, 6, 10, 30]
NU_SIM = 5                          # the Student-t we also simulate
K_TAIL = [3, 4, 5]                  # k-sigma falls reported in the table


def t_unit(nu: float, size, rng: np.random.Generator) -> np.ndarray:
    """Student-t draws rescaled to unit variance (needs nu > 2)."""
    return rng.standard_t(nu, size) * np.sqrt((nu - 2) / nu)


def t_unit_sf(x, nu: float):
    """P(X > x) for a unit-variance Student-t; by symmetry also P(X < −x)."""
    return stats.t.sf(np.asarray(x) * np.sqrt(nu / (nu - 2)), nu)


def t_excess_kurtosis(nu: float) -> float:
    return 6.0 / (nu - 4) if nu > 4 else np.inf


normal_within1 = float(2 * stats.norm.cdf(1) - 1)
normal_shoulders = float(2 * (stats.norm.cdf(3) - stats.norm.cdf(1)))   # 1 < |X| ≤ 3
lab.record("normal_within1", normal_within1)
lab.record("normal_shoulders", normal_shoulders)
lab.record("normal_beyond3", float(2 * stats.norm.sf(3)))
for k in K_TAIL:
    lab.record(f"normal_fall{k}", float(stats.norm.sf(k)))

t_rows = []
for nu in NU_TABLE:
    row = {"nu": nu, "scale": np.sqrt((nu - 2) / nu), "exkurt": t_excess_kurtosis(nu),
           "within1": 1 - 2 * t_unit_sf(1, nu),
           "shoulders": 2 * (t_unit_sf(1, nu) - t_unit_sf(3, nu)),
           "beyond3": 2 * t_unit_sf(3, nu)}
    for k in K_TAIL:
        row[f"fall{k}"] = float(t_unit_sf(k, nu))
        row[f"ratio{k}"] = float(t_unit_sf(k, nu) / stats.norm.sf(k))
    t_rows.append(row)
    for key, v in row.items():
        if key != "nu" and np.isfinite(v):
            lab.record(f"t{nu}_{key}", float(v))
t_table = pd.DataFrame(t_rows).set_index("nu")
print(t_table.round(5))

# simulate the unit-variance t(5) and check variance and tail frequencies against the exact values
x = t_unit(NU_SIM, N_DRAWS, rng_t)
fourth = 3 + t_excess_kurtosis(NU_SIM)                     # E[X⁴] for a unit-variance t (finite for ν > 4)
assert abs(x.var() - 1) < 4 * np.sqrt((fourth - 1) / N_DRAWS), "rescaled Student-t has unit variance"
for k in (1, 3, 4):
    p = float(t_unit_sf(k, NU_SIM))
    assert abs((x < -k).mean() - p) < 4 * se_share(p, N_DRAWS), f"t({NU_SIM}): P(X < −{k}) off by > 4 SE"
p1 = float(1 - 2 * t_unit_sf(1, NU_SIM))
assert abs((np.abs(x) <= 1).mean() - p1) < 4 * se_share(p1, N_DRAWS)
# the defining shape: more mass in the peak AND in the tails, less in the shoulders
assert t_table.loc[NU_SIM, "within1"] > normal_within1 and t_table.loc[NU_SIM, "shoulders"] < normal_shoulders
assert t_table.loc[NU_SIM, "beyond3"] > 2 * stats.norm.sf(3)
lab.record("t_nu_sim", NU_SIM)
lab.record("t_sim_var", float(x.var()))
lab.record("t_sim_fall4", float((x < -4).mean()))

# %% [markdown]
# ## 2 · How rare is a k-sigma day? The normal's ladder, computed exactly
#
# Under a normal, the chance of a fall of more than k standard deviations is Φ(−k). With 252
# trading days a year, the expected wait is 1 / (252 Φ(−k)) years. Compare a unit-variance
# Student-t(3), whose tail is a power law with index 3.

# %%
K_LADDER = [1, 2, 3, 4, 5, 6, 7, 8]
NU_LADDER = 3

ladder = []
for k in K_LADDER:
    p_n = float(stats.norm.sf(k))
    p_t = float(t_unit_sf(k, NU_LADDER))
    ladder.append({"k": k, "p_normal": p_n, "days_normal": 1 / p_n, "years_normal": 1 / p_n / DAYS_PER_YEAR,
                   "p_t3": p_t, "years_t3": 1 / p_t / DAYS_PER_YEAR, "ratio": p_t / p_n})
    lab.record(f"lad{k}_p", p_n)
    lab.record(f"lad{k}_days", 1 / p_n)
    lab.record(f"lad{k}_years", 1 / p_n / DAYS_PER_YEAR)
    lab.record(f"lad{k}_p_t3", p_t)
    lab.record(f"lad{k}_days_t3", 1 / p_t)
    lab.record(f"lad{k}_years_t3", 1 / p_t / DAYS_PER_YEAR)
    lab.record(f"lad{k}_ratio", p_t / p_n)
ladder = pd.DataFrame(ladder).set_index("k")
print(ladder.to_string(float_format=lambda v: f"{v:.4g}"))
# Sanity checks on the exact numbers (not random): the tail of a normal falls off like exp(−k²/2)
assert np.isclose(ladder.loc[4, "p_normal"], 3.167e-5, rtol=1e-3)
assert ladder["ratio"].is_monotonic_increasing       # the fatter tail wins by more the further out you go

# "Doubling the move": how much rarer is an 8-sigma fall than a 4-sigma fall?
double_normal = float(stats.norm.sf(4) / stats.norm.sf(8))
double_t3 = float(t_unit_sf(4, NU_LADDER) / t_unit_sf(8, NU_LADDER))
double_cubic = 2.0 ** 3                               # an exact power law with index 3: 2³ = 8
lab.record("double_normal", double_normal)
lab.record("double_normal_bn", double_normal / 1e9)        # in billions, for prose
lab.record("double_t3", double_t3)
lab.record("double_cubic", double_cubic)
assert 6 < double_t3 < double_cubic, "a Student-t(3) tail approaches the cubic law from below"
# Gabaix (2009): under the cubic law a 10-sigma move is 5³ = 125 times rarer than a 2-sigma move
lab.record("ratio_2_10_normal", float(stats.norm.sf(2) / stats.norm.sf(10)))
lab.record("ratio_2_10_normal_e21", float(stats.norm.sf(2) / stats.norm.sf(10)) / 1e21)   # × 10²¹, for prose
lab.record("ratio_2_10_cubic", 5.0 ** 3)
# "25-standard-deviation moves, several days in a row" (Goldman Sachs CFO, August 2007): one such
# day under a normal
lab.record("sigma25_log10_p", float(stats.norm.logsf(25) / np.log(10)))

# A historical move on the same yardstick: the Dow fell 22.6% on 19 October 1987 (Federal Reserve
# History). Measured against a 1% daily volatility — the volatility of this lab's simulated markets —
# that is a 22.6-sigma fall. Normal odds are astronomically small; the Student-t(3)'s are not.
CRASH_FALL = 0.226
DAILY_VOL = 0.01
crash_k = CRASH_FALL / DAILY_VOL
crash_log10_p_normal = float(stats.norm.logsf(crash_k) / np.log(10))
crash_years_t3 = float(1 / t_unit_sf(crash_k, NU_LADDER) / DAYS_PER_YEAR)
print(f"a {crash_k:.1f}-sigma fall: normal p = 10^{crash_log10_p_normal:.0f}; "
      f"Student-t(3): once every {crash_years_t3:,.0f} years")
lab.record("crash_fall", CRASH_FALL)
lab.record("crash_daily_vol", DAILY_VOL)
lab.record("crash_k", crash_k)
lab.record("crash_log10_p_normal", crash_log10_p_normal)
lab.record("crash_years_t3", crash_years_t3)

# %% [markdown]
# ## 3 · Fat-tailed markets and their normal twins
#
# Two synthetic markets, each simulated as 400 independent decades (4,000 years of daily returns):
#
# * **GARCH-t** (`qn.synth.garch_returns`): volatility that clusters (calm spells, storms) with
#   Student-t shocks. Session 5.5 builds this model; here it is just a realistic source of fat tails.
# * **Jump-diffusion** (`qn.synth.jump_diffusion_returns`): a calm normal market plus rare crashes
#   (about one every two years, averaging −10%).
#
# Each gets a **normal twin**: independent normal returns with exactly the same mean and volatility.
# We standardise every series by its TRUE mean and volatility and count k-sigma falls.

# %%
N_PATHS, N_DAYS = 400, 10 * DAYS_PER_YEAR
N_TOTAL = N_PATHS * N_DAYS
K_COUNT = [3, 4, 5, 6]
# GARCH(1,1) with Student-t shocks — the qn.synth defaults, spelled out. GARCH_A (reaction to
# yesterday's shock) and GARCH_B (memory of yesterday's variance) are the coefficients usually
# written α and β in GARCH papers — nothing to do with the alpha and beta of factor models.
OMEGA, GARCH_A, GARCH_B, MU_G, NU_G = 2e-6, 0.08, 0.90, 0.0003, 6.0
# Merton jump-diffusion — the qn.synth defaults, spelled out (annual units)
MU_J, SIGMA_J, JUMP_RATE, JUMP_MEAN, JUMP_SD = 0.07, 0.15, 0.5, -0.10, 0.05

# --- GARCH-t and its twin
garch = np.stack([qn.synth.garch_returns(N_DAYS, OMEGA, GARCH_A, GARCH_B, MU_G, NU_G, rng=rng_garch)["ret"].to_numpy()
                  for _ in range(N_PATHS)])                      # shape (paths, days)
garch_sd = np.sqrt(OMEGA / (1 - GARCH_A - GARCH_B))              # unconditional daily volatility
garch_twin = MU_G + garch_sd * rng_garch_twin.standard_normal((N_PATHS, N_DAYS))
z_garch = (garch - MU_G) / garch_sd
z_gtwin = (garch_twin - MU_G) / garch_sd


def garch_excess_kurtosis(a: float, b: float, nu: float | None) -> float:
    """Unconditional excess kurtosis of GARCH(1,1) returns (infinite if the 4th moment does not exist)."""
    kz = 3.0 if nu is None else 3.0 + t_excess_kurtosis(nu)    # kurtosis of the unit-variance shock
    denom = 1 - kz * a ** 2 - 2 * a * b - b ** 2
    if not np.isfinite(kz) or denom <= 0:
        return np.inf
    return kz * (1 - (a + b) ** 2) / denom - 3.0


garch_exkurt_theory = garch_excess_kurtosis(GARCH_A, GARCH_B, NU_G)
path_kurt = np.array([excess_kurtosis(p) for p in z_garch])     # one estimate per simulated decade

# --- jump-diffusion and its twin (work with LOG returns: their cumulants c1..c4 are exact)
dt = 1 / DAYS_PER_YEAR
jd = np.log1p(qn.synth.jump_diffusion_returns(N_DAYS, N_PATHS, MU_J, SIGMA_J, JUMP_RATE, JUMP_MEAN, JUMP_SD,
                                             rng=rng_jump)).T   # shape (paths, days)
lam = JUMP_RATE * dt                                            # jump probability per day (Poisson rate)
drift = MU_J - SIGMA_J ** 2 / 2 - JUMP_RATE * (np.exp(JUMP_MEAN + JUMP_SD ** 2 / 2) - 1)   # compensated
cum1 = drift * dt + lam * JUMP_MEAN
cum2 = SIGMA_J ** 2 * dt + lam * (JUMP_MEAN ** 2 + JUMP_SD ** 2)
cum3 = lam * (JUMP_MEAN ** 3 + 3 * JUMP_MEAN * JUMP_SD ** 2)
cum4 = lam * (JUMP_MEAN ** 4 + 6 * JUMP_MEAN ** 2 * JUMP_SD ** 2 + 3 * JUMP_SD ** 4)
jd_skew_theory = cum3 / cum2 ** 1.5
jd_exkurt_theory = cum4 / cum2 ** 2
jd_sd = np.sqrt(cum2)
jd_twin = cum1 + jd_sd * rng_jump_twin.standard_normal((N_PATHS, N_DAYS))
z_jd = (jd - cum1) / jd_sd
z_jtwin = (jd_twin - cum1) / jd_sd


def jd_cdf(x: float, max_jumps: int = 10) -> float:
    """P(daily log return < x), exactly: given n jumps the day's log return is normal."""
    n = np.arange(max_jumps + 1)
    w = stats.poisson.pmf(n, lam)
    return float(np.sum(w * stats.norm.cdf((x - drift * dt - n * JUMP_MEAN) / np.sqrt(SIGMA_J ** 2 * dt + n * JUMP_SD ** 2))))


# --- moments
moments = pd.DataFrame({
    "GARCH-t": {"skew": qn.stats.skewness(z_garch), "exkurt": excess_kurtosis(z_garch),
                "skew_theory": 0.0, "exkurt_theory": garch_exkurt_theory},
    "GARCH twin": {"skew": qn.stats.skewness(z_gtwin), "exkurt": excess_kurtosis(z_gtwin),
                   "skew_theory": 0.0, "exkurt_theory": 0.0},
    "jump-diffusion": {"skew": qn.stats.skewness(z_jd), "exkurt": excess_kurtosis(z_jd),
                       "skew_theory": jd_skew_theory, "exkurt_theory": jd_exkurt_theory},
    "jump twin": {"skew": qn.stats.skewness(z_jtwin), "exkurt": excess_kurtosis(z_jtwin),
                  "skew_theory": 0.0, "exkurt_theory": 0.0},
}).T
print(moments.round(3))
# normal twins: sample skewness and excess kurtosis have SEs √(6/n) and √(24/n)
for twin in (z_gtwin, z_jtwin):
    assert abs(qn.stats.skewness(twin)) < 4 * np.sqrt(6 / N_TOTAL)
    assert abs(excess_kurtosis(twin)) < 4 * np.sqrt(24 / N_TOTAL)
# jump-diffusion: every moment is finite, so the batch-means SE is valid; compare with exact cumulants
jd_skew_se = qn.stats.batch_se(qn.stats.skewness, z_jd)
jd_kurt_se = qn.stats.batch_se(excess_kurtosis, z_jd)
assert abs(qn.stats.skewness(z_jd) - jd_skew_theory) < 4 * jd_skew_se, "jump-diffusion skewness = c3 / c2^1.5"
assert abs(excess_kurtosis(z_jd) - jd_exkurt_theory) < 4 * jd_kurt_se, "jump-diffusion excess kurtosis = c4 / c2²"
# GARCH-t: the 4th moment exists but the 8th does not, so the sample kurtosis has NO finite standard
# error — no 4-SE test is possible. What we can say robustly (across 400 independent decades) is that
# a typical decade's estimate falls far short of the true value: the estimator's distribution is
# extremely right-skewed. (Not a 4-SE check: a statement about the median of 400 independent estimates.)
assert np.median(path_kurt) < garch_exkurt_theory, "a typical decade understates the GARCH-t kurtosis"

lab.record("n_paths", N_PATHS)
lab.record("n_years_path", N_DAYS // DAYS_PER_YEAR)
lab.record("n_years_total", N_TOTAL // DAYS_PER_YEAR)
lab.record("n_days_total", N_TOTAL)
lab.record("garch_a", GARCH_A)
lab.record("garch_b", GARCH_B)
lab.record("garch_persistence", GARCH_A + GARCH_B)
lab.record("garch_nu", NU_G)
lab.record("garch_daily_vol", garch_sd)
lab.record("garch_ann_vol", garch_sd * np.sqrt(DAYS_PER_YEAR))
if np.isfinite(garch_exkurt_theory):
    lab.record("garch_exkurt_theory", garch_exkurt_theory)
lab.record("garch_exkurt_pooled", excess_kurtosis(z_garch))
lab.record("garch_exkurt_median_path", float(np.median(path_kurt)))
lab.record("garch_exkurt_p10_path", float(np.percentile(path_kurt, 10)))
lab.record("garch_exkurt_p90_path", float(np.percentile(path_kurt, 90)))
lab.record("garch_skew_pooled", qn.stats.skewness(z_garch))
lab.record("gtwin_exkurt", excess_kurtosis(z_gtwin))
lab.record("gtwin_skew", qn.stats.skewness(z_gtwin))
lab.record("jd_jump_rate", JUMP_RATE)
lab.record("jd_jump_mean", JUMP_MEAN)
lab.record("jd_jump_sd", JUMP_SD)
lab.record("jd_sigma", SIGMA_J)
lab.record("jd_daily_vol", jd_sd)
lab.record("jd_ann_vol", jd_sd * np.sqrt(DAYS_PER_YEAR))
lab.record("jd_skew_theory", jd_skew_theory)
lab.record("jd_skew", qn.stats.skewness(z_jd))
lab.record("jd_skew_se", jd_skew_se)
lab.record("jd_exkurt_theory", jd_exkurt_theory)
lab.record("jd_exkurt", excess_kurtosis(z_jd))
lab.record("jd_exkurt_se", jd_kurt_se)
lab.record("jtwin_exkurt", excess_kurtosis(z_jtwin))
lab.record("jtwin_skew", qn.stats.skewness(z_jtwin))

# %% [markdown]
# **Counting k-sigma falls.** For each k: the normal's exact frequency, what the normal twins
# produced, what the fat-tailed markets produced, and (for the jump-diffusion) the exact value from
# the Poisson mixture. GARCH-t days are dependent (volatility clusters), so its standard errors come
# from batch means across the 400 independent decades.

# %%
count_rows = []
for k in K_COUNT:
    p_norm = float(stats.norm.sf(k))
    f_garch = float((z_garch < -k).mean())
    se_garch = qn.stats.batch_se(lambda b: (b < -k).mean(), z_garch)
    f_gtwin, f_jtwin = float((z_gtwin < -k).mean()), float((z_jtwin < -k).mean())
    f_jd_down, f_jd_up = float((z_jd < -k).mean()), float((z_jd > k).mean())
    p_jd_down = jd_cdf(cum1 - k * jd_sd)
    p_jd_up = 1 - jd_cdf(cum1 + k * jd_sd)
    # the normal twins match the normal (only where enough falls are expected for an SE to mean much)
    if p_norm * N_TOTAL > 20:
        for f in (f_gtwin, f_jtwin):
            assert abs(f - p_norm) < 4 * se_share(p_norm, N_TOTAL), f"normal twin: {k}-sigma falls off by > 4 SE"
    # GARCH-t: far more k-sigma falls than the normal allows
    assert f_garch - p_norm > 4 * se_garch, f"GARCH-t should show more {k}-sigma falls than the normal"
    # jump-diffusion days are independent: binomial SEs around the exact mixture probabilities
    assert abs(f_jd_down - p_jd_down) < 4 * se_share(p_jd_down, N_TOTAL), f"jump-diffusion {k}-sigma falls"
    if p_jd_up * N_TOTAL > 20:
        assert abs(f_jd_up - p_jd_up) < 4 * se_share(p_jd_up, N_TOTAL), f"jump-diffusion {k}-sigma rises"
    row = {"k": k, "normal": p_norm, "garch": f_garch, "garch_se": se_garch, "garch_twin": f_gtwin,
           "jd_down": f_jd_down, "jd_down_exact": p_jd_down, "jd_up": f_jd_up, "jd_up_exact": p_jd_up,
           "jd_twin": f_jtwin}
    count_rows.append(row)
    lab.record(f"cnt{k}_normal", p_norm)
    lab.record(f"cnt{k}_normal_years", 1 / p_norm / DAYS_PER_YEAR)
    lab.record(f"cnt{k}_normal_expected", p_norm * N_TOTAL)          # falls the normal expects in 4,000 years
    lab.record(f"cnt{k}_garch", f_garch)
    lab.record(f"cnt{k}_garch_n", int((z_garch < -k).sum()))
    lab.record(f"cnt{k}_garch_years", 1 / f_garch / DAYS_PER_YEAR)
    lab.record(f"cnt{k}_garch_ratio", f_garch / p_norm)
    lab.record(f"cnt{k}_gtwin", f_gtwin)
    lab.record(f"cnt{k}_gtwin_n", int((z_gtwin < -k).sum()))
    lab.record(f"cnt{k}_jd_down", f_jd_down)
    lab.record(f"cnt{k}_jd_down_n", int((z_jd < -k).sum()))
    lab.record(f"cnt{k}_jd_down_exact", p_jd_down)
    lab.record(f"cnt{k}_jd_down_ratio", p_jd_down / p_norm)
    lab.record(f"cnt{k}_jd_up", f_jd_up)
    lab.record(f"cnt{k}_jd_up_n", int((z_jd > k).sum()))
    lab.record(f"cnt{k}_jd_up_exact", p_jd_up)
    lab.record(f"cnt{k}_jtwin_n", int((z_jtwin < -k).sum()))
counts = pd.DataFrame(count_rows).set_index("k")
print(counts.to_string(float_format=lambda v: f"{v:.3g}"))

# The loss exceeded on one day in a thousand, in standard deviations — a preview of Session 9.7.
# Normal and Student-t(3): exact quantiles. GARCH-t: the empirical quantile of 4,000 simulated years,
# with a batch-means SE across the 400 decades.
P_RARE = 0.001
q_normal = float(stats.norm.isf(P_RARE))
q_t3 = float(stats.t.isf(P_RARE, NU_LADDER) * np.sqrt((NU_LADDER - 2) / NU_LADDER))
q_garch = float(-np.quantile(z_garch, P_RARE))
q_garch_se = qn.stats.batch_se(lambda b: -np.quantile(b, P_RARE), z_garch)
q_twin = float(-np.quantile(z_gtwin, P_RARE))
# the twin's empirical quantile vs the exact normal one: SE of a sample quantile = √(p(1−p)/n) / density
q_twin_se = np.sqrt(P_RARE * (1 - P_RARE) / N_TOTAL) / stats.norm.pdf(q_normal)
assert abs(q_twin - q_normal) < 4 * q_twin_se, "normal twin: 1-in-1,000-day loss matches the normal quantile"
assert q_garch - q_normal > 4 * q_garch_se, "GARCH-t: the 1-in-1,000-day loss is deeper than the normal says"
print(f"1-in-{1 / P_RARE:,.0f}-day loss: normal {q_normal:.2f}σ, GARCH-t {q_garch:.2f}σ ± {q_garch_se:.2f}, "
      f"Student-t(3) {q_t3:.2f}σ")
lab.record("rare_p", P_RARE)
lab.record("rare_q_normal", q_normal)
lab.record("rare_q_t3", q_t3)
lab.record("rare_q_garch", q_garch)
lab.record("rare_q_garch_se", q_garch_se)
lab.record("rare_ratio_garch", q_garch / q_normal)

# %% [markdown]
# The histogram shows the whole shape at once: against the normal curve with the same volatility,
# the GARCH-t returns pile up in the middle, thin out in the shoulders and spill into the tails.

# %%
HIST_EDGES = np.linspace(-6, 6, 61)
in_range = z_garch[np.abs(z_garch) <= 6]
grid = np.linspace(-6, 6, 241)
lab.chart("hist_garch", charts.histogram(
    in_range, title="GARCH-t daily returns, in standard deviations", bins=HIST_EDGES,
    x_fmt=charts.fmt_num(0, suffix="σ"), density_overlay=(grid, stats.norm.pdf(grid)),
    overlay_label="normal, same volatility"))
lab.record("garch_share_beyond6", float((np.abs(z_garch) > 6).mean()))
lab.record("garch_within1", float((np.abs(z_garch) <= 1).mean()))
lab.record("garch_shoulders", float(((np.abs(z_garch) > 1) & (np.abs(z_garch) <= 3)).mean()))
assert (np.abs(z_garch) <= 1).mean() - normal_within1 > 4 * qn.stats.batch_se(lambda b: (np.abs(b) <= 1).mean(), z_garch)

# %% [markdown]
# ## 4 · QQ plots: reading the tails
#
# A QQ plot puts the sorted data (standardised by their own sample mean and volatility) against
# the quantiles a normal would give for the same number of points. On the diagonal = normal.
# One simulated decade of each market — what an analyst with ten years of data would see.

# %%
QQ_PATH = 0


def qq_points(x: np.ndarray, n_body: int = 300, tail_q: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Normal quantiles vs sorted standardised data; thinned in the body, every point in the tails."""
    z = np.sort((x - x.mean()) / x.std())
    n = len(z)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)       # plotting positions (i − ½)/n
    body = np.unique(np.round(np.linspace(0, n - 1, n_body)).astype(int))
    keep = np.union1d(body, np.flatnonzero(np.abs(q) > tail_q))
    return q[keep], z[keep]


qx, qy = qq_points(garch[QQ_PATH])
lab.chart("qq_garch", charts.scatter_chart(
    qx, qy, title="QQ plot: one decade of GARCH-t returns vs the normal", diagonal=True,
    x_fmt=charts.fmt_num(0), y_fmt=charts.fmt_num(0),
    x_label="normal quantile (σ)", y_label="data quantile (σ)", height=400))
lab.record("qq_garch_min", float(qy.min()))
lab.record("qq_garch_max", float(qy.max()))
lab.record("qq_normal_extreme", float(-qx.min()))            # the most extreme quantile a normal gives n points
jx, jy = qq_points(jd[QQ_PATH])
lab.chart("qq_jump", charts.scatter_chart(
    jx, jy, title="QQ plot: one decade of jump-diffusion returns vs the normal", diagonal=True,
    x_fmt=charts.fmt_num(0), y_fmt=charts.fmt_num(0),
    x_label="normal quantile (σ)", y_label="data quantile (σ)", height=400))
lab.record("qq_jd_min", float(jy.min()))
lab.record("qq_jd_max", float(jy.max()))
lab.record("qq_jd_n_below5", int((((jd[QQ_PATH] - jd[QQ_PATH].mean()) / jd[QQ_PATH].std()) < -5).sum()))
lab.record("qq_n_days", N_DAYS)

# %% [markdown]
# ## 5 · Power laws and the Hill estimator
#
# A **power-law** tail: P(X > x) ≈ C x^(−ζ) for large x. The **tail index** ζ says how fat: moments
# of order ζ and above do not exist. (Many papers write α; in this course α means alpha.)
# Hill's (1975) estimator uses the k largest losses X(1) ≥ … ≥ X(k) above the threshold X(k+1):
#
#     1/ζ̂ = ξ̂ = (1/k) Σ ln X(i) − ln X(k+1)
#
# The choice of k IS the choice of threshold, and it matters. We test the estimator three ways:
# on exact Pareto data (theory known exactly), on a Student-t(3) (true ζ = 3, but only a power law
# far out in the tail), and on normal data (no power law at all). Samples: 40 years of daily data.

# %%
N_HILL = 40 * DAYS_PER_YEAR      # 10,080 observations
R_HILL = 200                     # independent samples
NU_HILL = 3                      # the true tail index of the Student-t
K_MAX = 1_000                    # up to the largest 10% of losses
K_TABLE = [10, 25, 50, 100, 250, 500, 1000]


def hill_xi(x: np.ndarray, k_max: int) -> np.ndarray:
    """ξ̂_k = 1/ζ̂_k for k = 1..k_max, row by row, from the largest LOSSES (−x) of each sample."""
    losses = -np.sort(x, axis=1)[:, : k_max + 1]               # the k_max + 1 largest losses, descending
    logs = np.log(losses)
    k = np.arange(1, k_max + 1)
    return np.cumsum(logs[:, :k_max], axis=1) / k - logs[:, 1: k_max + 1]


# (a) Exact Pareto losses with ζ = 3: k·ξ̂ is exactly Gamma(k, 1/ζ), so E[ζ̂] = ζ k/(k − 1)
pareto = -(rng_hill.pareto(NU_HILL, (R_HILL, N_HILL)) + 1)     # losses are Pareto(ζ = 3), x_min = 1
a_par = 1 / hill_xi(pareto, K_MAX)
for k in K_TABLE[1:]:
    mean_th = NU_HILL * k / (k - 1)
    sd_th = NU_HILL * k / ((k - 1) * np.sqrt(k - 2))
    assert abs(a_par[:, k - 1].mean() - mean_th) < 4 * sd_th / np.sqrt(R_HILL), f"Hill on exact Pareto, k = {k}"

# (b) Student-t(3): the exact expected ξ̂_k. Given the threshold X(k+1) = u, the k losses above it
# are independent draws from the tail beyond u, so E[ξ̂ | u] = g(u) = E[ln(X/u) | X > u]; and
# X(k+1) is the (k+1)-th largest of n, whose tail probability is Beta(k+1, n−k) distributed.
tsamp = rng_hill.standard_t(NU_HILL, (R_HILL, N_HILL))       # scale does not matter for ln(X/u)
g_t = hill_xi(tsamp, K_MAX)
a_t = 1 / g_t


def g_tail(u: float, nu: float) -> float:
    """E[ln(X/u) | X > u] for a Student-t(nu), by integration: ∫_u^∞ P(X > x)/x dx / P(X > u)."""
    val, _ = integrate.quad(lambda xx: stats.t.sf(xx, nu) / xx, u, np.inf, limit=200)
    return val / stats.t.sf(u, nu)


def expected_hill_xi(k: int, n: int, nu: float, n_grid: int = 100) -> float:
    """E[ξ̂_k] for n Student-t(nu) draws, averaging g over the distribution of the threshold X(k+1)."""
    tail_probs = stats.beta.ppf((np.arange(n_grid) + 0.5) / n_grid, k + 1, n - k)
    return float(np.mean([g_tail(stats.t.isf(p, nu), nu) for p in tail_probs]))


hill_rows = []
for k in K_TABLE:
    eg = expected_hill_xi(k, N_HILL, NU_HILL)
    se_g = g_t[:, k - 1].std(ddof=1) / np.sqrt(R_HILL)
    assert abs(g_t[:, k - 1].mean() - eg) < 4 * se_g, f"Student-t Hill estimates at k = {k} off the exact value"
    hill_rows.append({"k": k, "share": k / N_HILL, "zeta_mean": a_t[:, k - 1].mean(),
                      "zeta_sd": a_t[:, k - 1].std(ddof=1), "zeta_one": a_t[0, k - 1],
                      "zeta_limit": 1 / eg})
hill = pd.DataFrame(hill_rows).set_index("k")

# (c) Normal data: no power law, so the "tail index" keeps rising as the threshold moves out
nsamp = rng_hill.standard_normal((R_HILL, N_HILL))
a_n = 1 / hill_xi(nsamp, K_MAX)
hill["zeta_normal"] = [a_n[:, k - 1].mean() for k in K_TABLE]
print(hill.round(3))
# threshold sensitivity, asserted: wide thresholds bias the Student-t estimate far below 3 ...
se_1000 = a_t[:, 999].std(ddof=1) / np.sqrt(R_HILL)
assert NU_HILL - a_t[:, 999].mean() > 4 * se_1000, "a 10% threshold biases the Hill estimate"
# ... and on normal data the estimate climbs as the threshold moves out
se_n = np.hypot(a_n[:, 9].std(ddof=1), a_n[:, 999].std(ddof=1)) / np.sqrt(R_HILL)
assert a_n[:, 9].mean() - a_n[:, 999].mean() > 4 * se_n, "normal data: no stable tail index"

for k, r in hill.iterrows():
    for col in ("share", "zeta_mean", "zeta_sd", "zeta_one", "zeta_limit", "zeta_normal"):
        lab.record(f"hill{k}_{col}", float(r[col]))
one_curve = a_t[0, 9:K_MAX]                                     # one sample's estimates, k = 10..1,000
lab.record("hill_one_min", float(one_curve.min()))
lab.record("hill_one_max", float(one_curve.max()))
lab.record("hill_n", N_HILL)
lab.record("hill_years", N_HILL // DAYS_PER_YEAR)
lab.record("hill_reps", R_HILL)
lab.record("hill_true_zeta", NU_HILL)

ks = np.arange(10, K_MAX + 1)
lab.chart("hill_plot", charts.line_chart(
    [charts.Series("one 40-year sample", pd.Series(a_t[0, ks - 1], index=ks), role="strategy", end_label="one sample"),
     charts.Series(f"average of {R_HILL} samples", pd.Series(a_t[:, ks - 1].mean(axis=0), index=ks),
                   role="alt1", end_label="average")],
    title="Hill estimate of the tail index vs k (Student-t, true ζ = 3)", y_fmt=charts.fmt_num(1),
    hline=float(NU_HILL), y_min=1.0))

# %% [markdown]
# ## 6 · Moments that don't exist
#
# With tail index ζ, moments of order ζ and above are infinite. At ζ = 1 (the Cauchy, a Student-t
# with one degree of freedom) even the mean is gone — and the average of n draws is itself exactly
# Cauchy, as spread out as a single draw. The law of large numbers of Session 3.1 simply stops.
# Kurtosis needs ζ > 4; the GARCH-t market's kurtosis is finite but its estimate is not stable. A
# telling statistic: the share of the whole sample's fourth moment contributed by its single largest day.

# %%
N_AVG = 1_000          # draws per average
R_CAUCHY = 20_000      # averages

cauchy_means = rng_cauchy.standard_cauchy((R_CAUCHY, N_AVG)).mean(axis=1)
p_cauchy = float((np.abs(cauchy_means) > 1).mean())
assert abs(p_cauchy - 0.5) < 4 * se_share(0.5, R_CAUCHY), "the average of Cauchy draws is Cauchy: P(|mean| > 1) = ½"
p_normal_mean = float(2 * stats.norm.sf(np.sqrt(N_AVG)))   # the average of 1,000 N(0,1) draws is N(0, 1/1000)
lab.record("cauchy_n", N_AVG)
lab.record("cauchy_reps", R_CAUCHY)
lab.record("cauchy_p_mean_beyond1", p_cauchy)
lab.record("normal_log10_p_mean_beyond1", float(np.log10(p_normal_mean)))


def max_day_share(z: np.ndarray) -> float:
    """Share of the sample's sum of fourth powers (the kurtosis numerator) due to its largest day."""
    d4 = (z - z.mean()) ** 4
    return float(d4.max() / d4.sum())


share_garch = np.array([max_day_share(p) for p in z_garch])
share_twin = np.array([max_day_share(p) for p in z_gtwin])
# Across 400 independent decades the typical GARCH-t decade leans far more on one day than its
# normal twin does: compare the two medians, with batch-means SEs across the decades.
share_gap_se = np.hypot(qn.stats.batch_se(np.median, share_garch, n_batches=20),
                        qn.stats.batch_se(np.median, share_twin, n_batches=20))
assert np.median(share_garch) - np.median(share_twin) > 4 * share_gap_se
lab.record("maxshare_garch_median", float(np.median(share_garch)))
lab.record("maxshare_garch_p90", float(np.percentile(share_garch, 90)))
lab.record("maxshare_twin_median", float(np.median(share_twin)))
print(f"largest day's share of the fourth moment, median decade: GARCH-t {np.median(share_garch):.1%}, "
      f"normal twin {np.median(share_twin):.1%}")

# %%
lab.save()
