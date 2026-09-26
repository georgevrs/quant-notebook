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
# # 1.4 · The Engineer's Edge, Blind Spots & Honest Odds — companion lab
#
# **Quant Notebook** · Unit 1 · Session 4 · [Read the session](https://georgevrs.github.io/quant-notebook/unit01-quant-landscape/session04-engineers-edge-and-odds.html)
#
# What transfers from software and ML, what breaks in markets, and the evidence on who actually makes money.
#
# Six experiments, all on seeded synthetic data where we KNOW the truth:
#
# 1. **Signal-to-noise** — a genuinely predictive signal with an information coefficient of 0.05
#    explains a quarter of one percent of each day's return, yet it is a real edge.
# 2. **The kitchen sink** — junk features added to that signal fit the past more than ten times
#    better and the future worse than a forecast of zero.
# 3. **Costs** — how much trading the small edge can afford before it is gone.
# 4. **How many years to know** — the standard error of the Sharpe ratio turned into a
#    track-record length, checked by simulation.
# 5. **Base rates** — what passing a backtest really tells you.
# 6. **Regimes** — a signal whose edge flips sign after eight years.
#
# All returns here are synthetic **excess** returns (the risk-free rate is 0), so every Sharpe
# ratio is simply mean / volatility, annualised with 252 trading days.
#
# Each experiment draws from its **own random stream**, so changing one section (for example in
# the Try-it exercises) never changes the results of another.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.

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
from quantnb.returns import PERIODS_PER_YEAR
from quantnb.stats import annual_sharpe, oos_r2, sharpe_se, years_needed

lab = qn.Lab("1.4")  # seeds the random generators: every run gives the same numbers
# one independent stream per experiment (sections 1, 2, 3, 4, 5, 6)
rng_snr, rng_ks, rng_cost, rng_yrs, rng_br, rng_ns = lab.rng.spawn(6)

DAYS = PERIODS_PER_YEAR  # 252 trading days a year
Z95 = norm.ppf(0.975)    # 1.96: the two-sided 95% critical value

# %% [markdown]
# ## 1 · A real edge, buried in noise
#
# A **signal** $s_t$ is known before day $t$ starts; the day's (excess, simple) return is $R_t$.
# We build them so that their correlation — the **information coefficient** (IC) — is exactly 0.05,
# with $\sigma$ the *daily* volatility:
#
# $$ R_t = \sigma\,\big(\rho\, s_t + \sqrt{1-\rho^2}\,\varepsilon_t\big), \qquad s_t, \varepsilon_t \sim N(0, 1) $$
#
# With one predictor, the share of the return's variance the signal explains is $R^2 = \rho^2 =
# 0.25\%$. We trade it the simplest sensible way — hold $s_t$ units of the asset on day $t$ — and
# measure the hit rate and the Sharpe ratio. Theory for jointly normal $(s, R)$:
#
# * daily P&L $s_t R_t$ has mean $\rho\sigma$ and standard deviation $\sigma\sqrt{1+\rho^2}$, so
#   the annual Sharpe ratio is $\rho\sqrt{252}/\sqrt{1+\rho^2} \approx 0.79$;
# * the sign of the signal matches the sign of the return with probability
#   $\tfrac12 + \arcsin(\rho)/\pi \approx 51.6\%$.

# %%
IC_TRUE = 0.05       # correlation between the signal and the same day's return
DAILY_VOL = 0.01     # 1% a day, about 16% a year: roughly a broad equity index
YEARS_SNR = 20       # one long history
MC_PATHS = 200       # independent 20-year histories, to check the theory


def make_signal_and_returns(n_days: int, ic: float | np.ndarray, rng: np.random.Generator,
                            vol: float = DAILY_VOL) -> tuple[np.ndarray, np.ndarray]:
    """A standardised signal s and same-day returns R with corr(s, R) = ic by construction.

    `ic` may be an array (one value per day) to model an edge that changes over time.
    The signal is computed before the day starts, so trading it is not look-ahead.
    """
    s = rng.standard_normal(n_days)
    noise = rng.standard_normal(n_days)
    R = vol * (ic * s + np.sqrt(1 - np.square(ic)) * noise)
    return s, R


def theory_sharpe(ic: float) -> float:
    """Annual Sharpe ratio of holding s_t units when corr(s, R) = ic (jointly normal, daily)."""
    return float(ic / np.sqrt(1 + ic ** 2) * np.sqrt(DAYS))


n_days = DAYS * YEARS_SNR
s, R = make_signal_and_returns(n_days, IC_TRUE, rng_snr)
ic_hat = float(np.corrcoef(s, R)[0, 1])
pnl = s * R
one_path = {"ic": ic_hat, "sharpe": float(annual_sharpe(pnl))}

# the same experiment on 200 independent histories: the averages must match theory
S = rng_snr.standard_normal((n_days, MC_PATHS))
RR = DAILY_VOL * (IC_TRUE * S + np.sqrt(1 - IC_TRUE ** 2) * rng_snr.standard_normal((n_days, MC_PATHS)))
mc_ic = np.array([np.corrcoef(S[:, j], RR[:, j])[0, 1] for j in range(MC_PATHS)])
mc_hit = (np.sign(S) == np.sign(RR)).mean(axis=0)
mc_sharpe = annual_sharpe(S * RR)

snr_theory = {"r2": IC_TRUE ** 2, "hit": 0.5 + np.arcsin(IC_TRUE) / np.pi, "sharpe": theory_sharpe(IC_TRUE)}
print(f"one 20-year path: IC {one_path['ic']:.4f} · Sharpe {one_path['sharpe']:.2f}")
print(f"200 paths (mean): IC {mc_ic.mean():.4f} · hit {mc_hit.mean():.2%} · Sharpe {mc_sharpe.mean():.2f} "
      f"(sd {mc_sharpe.std():.2f}, range {mc_sharpe.min():.2f} to {mc_sharpe.max():.2f})")
print(f"theory:           IC {IC_TRUE:.4f} · R² {snr_theory['r2']:.3%} · hit {snr_theory['hit']:.2%} · Sharpe {snr_theory['sharpe']:.2f}")

lab.record("ic_true", IC_TRUE)
lab.record("daily_vol", DAILY_VOL)
lab.record("years_snr", YEARS_SNR)
lab.record("mc_paths", MC_PATHS)
lab.record("r2_theory", snr_theory["r2"])
lab.record("hit_theory", snr_theory["hit"])
lab.record("sharpe_theory", snr_theory["sharpe"])
lab.record("path_ic", one_path["ic"])
lab.record("path_sharpe", one_path["sharpe"])
lab.record("mc_ic", float(mc_ic.mean()))
lab.record("mc_hit", float(mc_hit.mean()))
lab.record("mc_sharpe", float(mc_sharpe.mean()))
lab.record("mc_sharpe_sd", float(mc_sharpe.std()))
lab.record("mc_sharpe_min", float(mc_sharpe.min()))
lab.record("mc_sharpe_max", float(mc_sharpe.max()))
lab.record("mc_p_below_half", float((mc_sharpe < 0.5).mean()))   # a real 0.79 edge measured below 0.5
lab.record("n_obs_daily", n_days)
lab.record("n_obs_monthly", 12 * YEARS_SNR)

# Assert what we teach, at ≥ 3 standard errors of each Monte Carlo average.
assert abs(mc_ic.mean() - IC_TRUE) < 3.5 / np.sqrt(n_days * MC_PATHS)
assert abs(mc_hit.mean() - snr_theory["hit"]) < 0.002
assert abs(mc_sharpe.mean() - snr_theory["sharpe"]) < 0.05
assert abs(mc_sharpe.std() - 1 / np.sqrt(YEARS_SNR)) < 0.05   # SE of a 20-year Sharpe ≈ 1/√20
del S, RR

# %% [markdown]
# ## 2 · The noise that looks better: a kitchen-sink model
#
# The ML reflex is to add features. We give a linear model three menus, fitted by least squares
# on 5 years (1,260 days) and scored on the next 5 years it has never seen:
#
# * **real** — the one genuine signal;
# * **real + junk** — the genuine signal plus `N_JUNK` features of pure noise;
# * **junk only** — `N_JUNK + 1` noise features, the same number of columns.
#
# $R^2$ is measured against a forecast of zero, $R^2 = 1 - \sum (R - \hat R)^2 / \sum R^2$, as in
# Gu, Kelly & Xiu (2020). Theory: $K$ useless features buy an in-sample $R^2$ of about $K/T$ —
# with 50 features, $50/1260 \approx 4\%$ — and cost about the same out of sample. We repeat 200 times.

# %%
YEARS_IS, YEARS_OOS = 5, 5
N_JUNK = 49
REPEATS = 200


def fit_and_score(X: np.ndarray, y: np.ndarray, split: int) -> dict[str, float]:
    """Least squares on the first `split` rows; R² (against a zero forecast) in and out of sample,
    and the out-of-sample Sharpe of holding a position proportional to the forecast."""
    beta, *_ = np.linalg.lstsq(X[:split], y[:split], rcond=None)
    y_hat = X @ beta
    return {
        "r2_is": oos_r2(y[:split], y_hat[:split]),   # the same formula, scored on the fitted data
        "r2_oos": oos_r2(y[split:], y_hat[split:]),
        "sharpe_oos": float(annual_sharpe(y_hat[split:] * y[split:])),
    }


split = DAYS * YEARS_IS
n = DAYS * (YEARS_IS + YEARS_OOS)
rows = []
for _ in range(REPEATS):
    s, R = make_signal_and_returns(n, IC_TRUE, rng_ks)
    junk = rng_ks.standard_normal((n, N_JUNK + 1))
    menus = {
        "real": s[:, None],
        "real + junk": np.column_stack([s, junk[:, :N_JUNK]]),
        "junk only": junk,
    }
    for name, X in menus.items():
        rows.append({"model": name, **fit_and_score(X, R, split)})
runs = pd.DataFrame(rows)
ks = runs.groupby("model", sort=False).mean()
p_junk_beats_real = float((runs.loc[runs["model"] == "junk only", "r2_is"].to_numpy()
                           > runs.loc[runs["model"] == "real", "r2_is"].to_numpy()).mean())
print(ks.round(4))
print(f"P({N_JUNK + 1} junk features fit the past better than the real signal) = {p_junk_beats_real:.0%}")

for name, key in (("real", "real"), ("real + junk", "mix"), ("junk only", "junk")):
    for col in ("r2_is", "r2_oos", "sharpe_oos"):
        lab.record(f"ks_{key}_{col}", float(ks.loc[name, col]))
lab.record("n_junk", N_JUNK)
lab.record("n_features", N_JUNK + 1)
lab.record("ks_years_is", YEARS_IS)
lab.record("ks_repeats", REPEATS)
lab.record("ks_days_is", split)
lab.record("ks_k_over_t", (N_JUNK + 1) / split)
lab.record("p_junk_beats_real_is", p_junk_beats_real)
lab.record("ks_is_ratio", float(ks.loc["real + junk", "r2_is"] / ks.loc["real", "r2_is"]))

# The overfitting signature: better in-sample, worse out-of-sample (robust to IC_TRUE and N_JUNK).
assert ks.loc["junk only", "r2_is"] > 5 * ks.loc["real", "r2_is"]
assert abs(ks.loc["junk only", "r2_is"] - (N_JUNK + 1) / split) < 0.005
assert ks.loc["real + junk", "r2_oos"] < 0 and ks.loc["junk only", "r2_oos"] < 0
assert ks.loc["real", "r2_oos"] > ks.loc["real + junk", "r2_oos"]
assert ks.loc["real + junk", "sharpe_oos"] < 0.5 * ks.loc["real", "sharpe_oos"]
assert abs(ks.loc["junk only", "sharpe_oos"]) < 0.15
assert p_junk_beats_real > 0.95

# %% [markdown]
# ## 3 · The edge must pay for its own trading
#
# Holding $s_t$ units means trading $|s_t - s_{t-1}|$ units every day. For independent standard
# normal signals the expected daily turnover is $2/\sqrt{\pi} \approx 1.13$. The gross edge is
# $\rho\sigma = 0.05 \times 1\% = 5$ bp a day, so the **break-even cost** is about
# $5 / 1.13 \approx 4.4$ bp per unit traded. Above that, a real edge loses money.

# %%
COSTS_BP = [0, 1, 2, 3, 4, 5]
YEARS_COST = 2_000   # long enough that the break-even estimate has a standard error of ~0.12 bp

s, R = make_signal_and_returns(DAYS * YEARS_COST, IC_TRUE, rng_cost)
turnover = np.abs(np.diff(s, prepend=0.0))
gross = s * R
turnover_mean = float(turnover[1:].mean())
gross_bp = float(gross.mean() * 1e4)
breakeven_bp = gross_bp / turnover_mean
breakeven_theory_bp = IC_TRUE * DAILY_VOL * 1e4 / (2 / np.sqrt(np.pi))
breakeven_se_bp = float(gross.std() / np.sqrt(len(gross)) * 1e4 / turnover_mean)
net_sharpe = {c: float(annual_sharpe(gross - c / 1e4 * turnover)) for c in COSTS_BP}
print(f"turnover {turnover_mean:.3f}/day (theory {2 / np.sqrt(np.pi):.3f}) · gross {gross_bp:.2f} bp/day · "
      f"break-even {breakeven_bp:.2f} ± {breakeven_se_bp:.2f} bp (theory {breakeven_theory_bp:.2f})")
print({c: round(v, 2) for c, v in net_sharpe.items()})

lab.record("turnover", turnover_mean)
lab.record("gross_bp", gross_bp)
lab.record("breakeven_bp", breakeven_bp)
lab.record("breakeven_theory_bp", breakeven_theory_bp)
for c, v in net_sharpe.items():
    lab.record(f"net_sharpe_{c}bp", v)

assert abs(turnover_mean - 2 / np.sqrt(np.pi)) < 0.01
assert abs(breakeven_bp - breakeven_theory_bp) < 3 * breakeven_se_bp
assert net_sharpe[2] < 0.65 * net_sharpe[0] and net_sharpe[5] < 0

# %% [markdown]
# ## 4 · How many years before you know?
#
# An estimated Sharpe ratio is noisy. For independent, normally distributed returns Lo (2002) gives
# its standard error; measured on $Y$ yearly returns it is
#
# $$ \mathrm{SE}(\widehat{\mathrm{SR}}) \approx \sqrt{\frac{1 + \mathrm{SR}^2/2}{Y}} . $$
#
# Asking for the 95% interval to exclude zero, $\mathrm{SR} \ge 1.96\,\mathrm{SE}$, gives the years needed:
#
# $$ Y \approx \Big(\frac{1.96}{\mathrm{SR}}\Big)^2 \Big(1 + \frac{\mathrm{SR}^2}{2}\Big). $$
#
# The $\mathrm{SR}^2/2$ term is the price of also estimating the volatility. Lo's formula is written
# for the *per-period* Sharpe ratio, so on daily returns the term shrinks to $\mathrm{SR}^2/(2 \times 252)$
# and $Y \approx (1.96/\mathrm{SR})^2$. `quantnb.stats.years_needed` handles the frequency. It is an
# asymptotic formula: for a Sharpe of 2 on yearly returns it rests on about three observations.
#
# At exactly $Y$ years, a strategy whose true Sharpe *is* $\mathrm{SR}$ clears the bar only half the
# time (its estimate lands above or below the truth with equal odds). To clear it four times in
# five — 80% **power** — you need $\big((1.96 + 0.84)/\mathrm{SR}\big)^2$ years. We check both.

# %%
SR_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
Z80 = norm.ppf(0.80)   # 0.84

yrs = pd.DataFrame({
    "sharpe": SR_GRID,
    "years_lo_yearly": [years_needed(x, periods_per_year=1) for x in SR_GRID],
    "years_daily": [years_needed(x) for x in SR_GRID],
    "years_daily_80pct": [years_needed(x, z=Z95 + Z80) for x in SR_GRID],
}).set_index("sharpe")
print(yrs.round(2))
for x, row in yrs.iterrows():
    key = f"{x:g}".replace(".", "_")
    for col in yrs.columns:
        lab.record(f"yrs_{key}_{col}", float(row[col]))

# %% [markdown]
# **Check 1 — daily data.** Simulate 5,000 track records of a strategy with true Sharpe 1.0 for
# exactly the "daily" years needed, and again for the 80%-power length. How often does the
# estimated Sharpe clear 1.96 standard errors? Theory: 50% and 80%.

# %%
SR_CHECK = 1.0
N_TRACKS = 5_000
DAILY_SD = 0.01   # any volatility works: the Sharpe ratio does not depend on it


def pass_rate(years: float, sr_true: float = SR_CHECK, n_tracks: int = N_TRACKS) -> tuple[float, float]:
    """Fraction of simulated daily track records whose Sharpe is ≥ 1.96 SE above zero,
    and the sampling standard deviation of their annualised Sharpe ratios."""
    n_d = int(round(years * DAYS))
    mu = sr_true / np.sqrt(DAYS) * DAILY_SD
    est = np.empty(n_tracks)
    for lo in range(0, n_tracks, 1_000):   # chunks keep memory small
        hi = min(lo + 1_000, n_tracks)
        est[lo:hi] = annual_sharpe(mu + DAILY_SD * rng_yrs.standard_normal((n_d, hi - lo)))
    se = sharpe_se(sr_true, n_d / DAYS)
    return float(np.mean(est >= Z95 * se)), float(est.std())


y_half = years_needed(SR_CHECK)
y_80 = years_needed(SR_CHECK, z=Z95 + Z80)
power_half, sd_half = pass_rate(y_half)
power_80, _ = pass_rate(y_80)
print(f"SR 1.0: {y_half:.2f} years → passes {power_half:.1%} (theory 50%), SE measured {sd_half:.3f} "
      f"vs formula {sharpe_se(SR_CHECK, y_half):.3f}; {y_80:.2f} years → passes {power_80:.1%} (theory 80%)")
lab.record("n_tracks", N_TRACKS)
lab.record("check_power_half", power_half)
lab.record("check_power_80", power_80)
lab.record("check_sd_daily", sd_half)
lab.record("check_se_daily_formula", sharpe_se(SR_CHECK, y_half))
# binomial SE of a pass rate over 5,000 tracks ≈ 0.007: tolerance ≈ 4 SE
assert abs(power_half - 0.5) < 0.03 and abs(power_80 - 0.8) < 0.03
assert abs(sd_half - sharpe_se(SR_CHECK, y_half)) < 0.02

# %% [markdown]
# **Check 2 — yearly data.** Judge the same Sharpe-1.0 strategy on 100 yearly returns instead
# (long enough for the asymptotic formula to hold). Lo's formula predicts a standard error of
# $\sqrt{1.5/100} \approx 0.122$ — larger than the $\approx 0.100$ you would get from daily data.

# %%
N_YEARLY_OBS = 100
yearly = SR_CHECK * 0.10 + 0.10 * rng_yrs.standard_normal((N_YEARLY_OBS, 20_000))  # mean/sd = 1.0
sd_yearly = float((yearly.mean(axis=0) / yearly.std(axis=0, ddof=1)).std())
se_yearly = sharpe_se(SR_CHECK, N_YEARLY_OBS, periods_per_year=1)
print(f"100 yearly returns: SE measured {sd_yearly:.3f} vs Lo {se_yearly:.3f}")
lab.record("check_sd_yearly", sd_yearly)
lab.record("check_se_yearly_formula", se_yearly)
assert abs(sd_yearly - se_yearly) < 0.01

# %% [markdown]
# ## 5 · Base rates: what does passing a backtest tell you?
#
# Suppose a fraction $\pi$ of the ideas you test are real, each with a true Sharpe of 0.5, and the
# rest have none. You test each on 5 years and call it a discovery when the estimated Sharpe clears
# 1.96 standard errors. A fake idea then passes 2.5% of the time — the test's **size** (statisticians
# write α, but in this course α is reserved for excess return). A real one passes with probability
# $\Phi(0.5\sqrt5 - 1.96) \approx 20\%$ — the test's **power**. Bayes' rule gives the share of
# "discoveries" that are real:
#
# $$ P(\text{real} \mid \text{pass}) = \frac{\pi \cdot \text{power}}{\pi \cdot \text{power} + (1 - \pi)\cdot \text{size}}. $$
#
# This is Ioannidis's (2005) argument for why most published findings are false, applied to your
# own research. We check it by simulating 200,000 ideas, drawing each measured Sharpe ratio from
# its sampling distribution (standard error $\approx 1/\sqrt{Y}$ on daily data, §4).

# %%
SR_REAL = 0.5
YEARS_TEST = 5
PRIORS = [0.01, 0.05, 0.10, 0.25, 0.50]
N_IDEAS = 200_000

size = float(1 - norm.cdf(Z95))                             # 2.5% one-sided false-positive rate
power = float(norm.cdf(SR_REAL * np.sqrt(YEARS_TEST) - Z95))
power_sr1 = float(norm.cdf(1.0 * np.sqrt(YEARS_TEST) - Z95))


def posterior_real(prior: float, power: float = power, size: float = size) -> float:
    """P(idea is real | it passed the test), by Bayes' rule."""
    return prior * power / (prior * power + (1 - prior) * size)


base = []
for prior in PRIORS:
    real = rng_br.random(N_IDEAS) < prior
    measured = np.where(real, SR_REAL, 0.0) + rng_br.standard_normal(N_IDEAS) / np.sqrt(YEARS_TEST)
    passed = measured >= Z95 / np.sqrt(YEARS_TEST)
    base.append({"prior": prior, "theory": posterior_real(prior), "simulated": float(real[passed].mean()),
                 "pass_rate": float(passed.mean())})
base = pd.DataFrame(base).set_index("prior")
print(f"size {size:.1%} · power at SR {SR_REAL} over {YEARS_TEST} y: {power:.1%} (SR 1.0: {power_sr1:.1%})")
print(base.round(3))

lab.record("n_ideas", N_IDEAS)
lab.record("br_sr_real", SR_REAL)
lab.record("br_years", YEARS_TEST)
lab.record("br_size", size)
lab.record("br_power", power)
lab.record("br_power_sr1", power_sr1)
lab.record("br_bar", float(Z95 / np.sqrt(YEARS_TEST)))
for prior, row in base.iterrows():
    key = f"{int(round(prior * 100))}"
    lab.record(f"br_post_{key}", row["theory"])
    lab.record(f"br_post_sim_{key}", row["simulated"])
assert (base["theory"] - base["simulated"]).abs().max() < 0.03

# natural frequencies for the page: 1,000 ideas at a 10% base rate (expected counts)
ideas, prior = 1_000, 0.10
nat = {"real": ideas * prior, "fake": ideas * (1 - prior)}
nat["real_pass"] = nat["real"] * power
nat["real_miss"] = nat["real"] - nat["real_pass"]
nat["fake_pass"] = nat["fake"] * size
nat["all_pass"] = nat["real_pass"] + nat["fake_pass"]
nat["share_real"] = nat["real_pass"] / nat["all_pass"]
for k, v in nat.items():
    lab.record(f"nat_{k}", float(v))
print({k: round(v, 2) for k, v in nat.items()})
assert abs(nat["share_real"] - posterior_real(0.10)) < 1e-12

# %% [markdown]
# ## 6 · Non-stationarity: an edge that flips
#
# Markets are not a fixed function. Here the same signal has IC +0.05 for 8 trading years
# (regime A) and then −0.03 for 4 (regime B) — say, because others crowded into it or the market
# structure changed. Time is simulated, so the charts count elapsed years, not calendar dates.
#
# We trade it throughout and watch the rolling one-year Sharpe ratio, with the true Sharpe of each
# regime drawn in grey. The one-year Sharpe has a standard error of about 1, so even in the good
# regime roughly one year in five shows a loss — and in the bad regime roughly one year in three
# still shows a profit. You cannot tell a broken edge from bad luck quickly.

# %%
YEARS_A, YEARS_B = 8, 4
IC_A, IC_B = 0.05, -0.03
WINDOW = DAYS          # one-year rolling window
N_HISTORIES = 500      # Monte Carlo histories for the full-sample Sharpe

n_ns = DAYS * (YEARS_A + YEARS_B)
elapsed = np.arange(1, n_ns + 1) / DAYS              # elapsed trading years
ic_path = np.where(np.arange(n_ns) < DAYS * YEARS_A, IC_A, IC_B)
true_a, true_b = theory_sharpe(IC_A), theory_sharpe(IC_B)

# one illustrative history, for the charts
s, R = make_signal_and_returns(n_ns, ic_path, rng_ns)
pnl_ts = pd.Series(s * R, index=elapsed)
rolling = (pnl_ts.rolling(WINDOW).mean() / pnl_ts.rolling(WINDOW).std() * np.sqrt(DAYS)).dropna()
true_sr = pd.Series(np.where(ic_path < 0, true_b, true_a), index=elapsed).loc[rolling.index]
growth = (1 + pnl_ts).cumprod()          # $1 traded on the signal, compounded

sr_a = float(annual_sharpe(pnl_ts.iloc[: DAYS * YEARS_A].to_numpy()))
sr_b = float(annual_sharpe(pnl_ts.iloc[DAYS * YEARS_A:].to_numpy()))
sr_all = float(annual_sharpe(pnl_ts.to_numpy()))
growth_at_break = float(growth.iloc[DAYS * YEARS_A - 1])

# Monte Carlo: the full-sample Sharpe is a blend of the two regimes that hides the break
S = rng_ns.standard_normal((n_ns, N_HISTORIES))
RR = DAILY_VOL * (ic_path[:, None] * S + np.sqrt(1 - ic_path[:, None] ** 2) * rng_ns.standard_normal((n_ns, N_HISTORIES)))
mc_full = annual_sharpe(S * RR)
full_theory = (YEARS_A * true_a + YEARS_B * true_b) / (YEARS_A + YEARS_B)   # P&L volatility is ~equal in both regimes
del S, RR

# probabilities for a single year, over many independent years
N_YEARS_MC = 20_000


def one_year_sharpes(ic: float, n_years: int) -> np.ndarray:
    """Sharpe ratios of n_years independent one-year track records of the signal with this IC."""
    sig = rng_ns.standard_normal((DAYS, n_years))
    ret = DAILY_VOL * (ic * sig + np.sqrt(1 - ic ** 2) * rng_ns.standard_normal((DAYS, n_years)))
    return annual_sharpe(sig * ret)


p_loss_year_a = float((one_year_sharpes(IC_A, N_YEARS_MC) < 0).mean())
p_gain_year_b = float((one_year_sharpes(IC_B, N_YEARS_MC) > 0).mean())
print(f"this history: regime A Sharpe {sr_a:.2f} (true {true_a:.2f}) · regime B {sr_b:.2f} (true {true_b:.2f}) · "
      f"whole {sr_all:.2f} · $1 → ${growth_at_break:.2f} at the break → ${growth.iloc[-1]:.2f}")
print(f"{N_HISTORIES} histories: full-sample Sharpe {mc_full.mean():.2f} (theory {full_theory:.2f})")
print(f"P(losing year | good regime) {p_loss_year_a:.0%} (theory {norm.cdf(-true_a):.0%}) · "
      f"P(winning year | bad regime) {p_gain_year_b:.0%} (theory {norm.cdf(true_b):.0%})")

lab.record("ns_years_a", YEARS_A)
lab.record("ns_years_b", YEARS_B)
lab.record("ns_ic_a", IC_A)
lab.record("ns_ic_b", IC_B)
lab.record("ns_true_sr_a", true_a)
lab.record("ns_true_sr_b", true_b)
lab.record("ns_sr_a", sr_a)
lab.record("ns_sr_b", sr_b)
lab.record("ns_sr_all", sr_all)
lab.record("ns_histories", N_HISTORIES)
lab.record("ns_sr_all_mc", float(mc_full.mean()))
lab.record("ns_sr_all_theory", full_theory)
lab.record("ns_growth_break", growth_at_break)
lab.record("ns_growth_end", float(growth.iloc[-1]))
lab.record("ns_rolling_min_a", float(rolling.loc[:YEARS_A].min()))
lab.record("ns_rolling_max_b", float(rolling.loc[YEARS_A + 1:].max()))   # windows wholly inside regime B
lab.record("ns_p_loss_year_a", p_loss_year_a)
lab.record("ns_p_gain_year_b", p_gain_year_b)
lab.record("ns_p_loss_year_a_theory", float(norm.cdf(-true_a)))
lab.record("ns_p_gain_year_b_theory", float(norm.cdf(true_b)))
assert abs(p_loss_year_a - norm.cdf(-true_a)) < 0.02          # binomial SE ≈ 0.003
assert abs(p_gain_year_b - norm.cdf(true_b)) < 0.02
assert abs(mc_full.mean() - full_theory) < 3 * mc_full.std() / np.sqrt(N_HISTORIES) + 0.01

# %% [markdown]
# ## 7 · Charts for the session page

# %%
lab.chart("kitchen_sink", charts.column_chart(
    ["1 real · past", "1 real · future", f"+{N_JUNK} junk · past", f"+{N_JUNK} junk · future"],
    [ks.loc["real", "r2_is"], ks.loc["real", "r2_oos"],
     ks.loc["real + junk", "r2_is"], ks.loc["real + junk", "r2_oos"]],
    title=f"R² of daily return forecasts: {YEARS_IS} years fitted, {YEARS_OOS} unseen, {REPEATS} repeats",
    y_fmt=charts.fmt_pct(2),
    roles=["benchmark", "strategy", "benchmark", "loss"]))

lab.chart("years_needed", charts.column_chart(
    [f"SR {x:g}" for x in SR_GRID], yrs["years_lo_yearly"].tolist(),
    title="Years of yearly returns needed before a Sharpe ratio is 95% distinguishable from zero",
    y_fmt=charts.fmt_num(1)))

lab.chart("rolling_sharpe", charts.line_chart(
    [charts.Series("1-year Sharpe", rolling, role="strategy"),
     charts.Series("true Sharpe", true_sr, role="benchmark", label_end=False)],
    title="One signal, two regimes: rolling one-year Sharpe ratio by elapsed year", y_fmt=charts.fmt_num(1),
    hline=0.0, bands=[(float(YEARS_A), float(elapsed[-1]), "regime B: the edge flips")]))

lab.chart("regime_growth", charts.line_chart(
    [charts.Series("growth of $1", growth, role="strategy")],
    title="The same signal: growth of $1 by elapsed year", y_fmt=charts.fmt_num(2, prefix="$"),
    hline=1.0, height=240, bands=[(float(YEARS_A), float(elapsed[-1]), "regime B")]))

# %%
lab.save()
