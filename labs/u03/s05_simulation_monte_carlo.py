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
# # 3.5 · Simulation & Monte Carlo — companion lab
#
# **Quant Notebook** · Unit 3 · Session 5 · [Read the session](https://georgevrs.github.io/quant-notebook/unit03-math-toolkit/session05-simulation-monte-carlo.html)
#
# Simulate prices, strategies and uncertainty with seeded Monte Carlo — random walks, GBM, bootstraps and variance reduction.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# The rule of this lab: **every simulated number comes with its Monte Carlo standard error**, and
# every check against theory is made at 4 standard errors through one helper, `check_mc`, which
# also counts the checks. Section 7 uses that count to show why 4 and not 2 or 3.
# All data are synthetic (geometric Brownian motion and GARCH), prices start at 1 (or 100 for the
# option-style payoff in section 6), and there is no interest rate anywhere (risk-free rate 0).

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
from arch.bootstrap import optimal_block_length
from scipy.stats import binom, norm

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR, in_years, max_drawdown_paths
from quantnb.stats import batch_se, p_touch

lab = qn.Lab("3.5")  # seeds the random generators: every run gives the same numbers
# One independent random stream per experiment (Session 1.5): editing or re-running one section
# never shifts the random numbers another section sees.
(rng_se, rng_gbm, rng_dd, rng_bar, rng_boot, rng_vr, rng_mt) = lab.rng.spawn(7)

DT = 1.0 / PERIODS_PER_YEAR          # one trading day, in years
MU, SIGMA = 0.08, 0.20               # the "base asset": 8% drift, 20% volatility a year (as in 2.3)
NU = MU - 0.5 * SIGMA ** 2           # drift of the LOG price (volatility drag, Session 2.3)
N_SE = 4                             # every Monte Carlo check is made at 4 standard errors
MC_CHECKS: list[tuple[str, float]] = []   # (name, z-score) of every statistical check, for §7

lab.record("mu", MU)
lab.record("sigma", SIGMA)
lab.record("nu", NU)
lab.record("n_se", N_SE)


def check_mc(name: str, estimate: float, truth: float, se: float, n_se: float = N_SE,
             allowance: float = 0.0) -> float:
    """Assert that a Monte Carlo estimate lies within n_se standard errors of its theoretical value.

    `allowance` widens the tolerance by a known approximation error of the THEORY (never used to
    rescue noise; every use carries a comment). Returns the z-score and remembers it, so the lab
    can count how many statistical checks it makes.
    """
    z = (estimate - truth) / se
    MC_CHECKS.append((name, float(z)))
    assert abs(estimate - truth) < n_se * se + allowance, (
        f"{name}: {estimate:.6g} vs theory {truth:.6g} is {z:+.2f} SE away")
    return float(z)


def se_prop(p: float, n: int) -> float:
    """Standard error of a simulated probability (a proportion of n independent paths)."""
    return float(np.sqrt(p * (1.0 - p) / n))


# %% [markdown]
# ## 1 · A Monte Carlo estimate and its standard error
#
# Question: what is the chance that the base asset loses money over one year? Under geometric
# Brownian motion (GBM) the one-year log return is normal with mean ν = μ − σ²/2 and standard
# deviation σ, so the exact answer is Φ(−ν/σ). We pretend not to know it, simulate, and watch the
# estimate close in on the truth like 1/√n. A probability is just the expected value of an
# indicator (1 if the event happens, 0 if not), so everything here applies to any expectation.

# %%
N_LIST = [100, 1_000, 10_000, 100_000, 1_000_000]
p_loss_theory = float(norm.cdf(-NU / SIGMA))
log_r_1y = NU + SIGMA * rng_se.standard_normal(max(N_LIST))   # exact one-year log returns
lost = (log_r_1y < 0).astype(float)                            # indicator of a losing year

rows = []
for n in N_LIST:
    p_hat = float(lost[:n].mean())
    se = se_prop(p_hat, n)
    rows.append({"n": n, "estimate": p_hat, "se": se, "z": (p_hat - p_loss_theory) / se})
    lab.record(f"ploss_{n}_est", p_hat)
    lab.record(f"ploss_{n}_se", se)
    lab.record(f"ploss_{n}_se_pp", 100 * se)
    lab.record(f"ploss_{n}_lo", p_hat - 2 * se)
    lab.record(f"ploss_{n}_hi", p_hat + 2 * se)
    lab.record(f"ploss_{n}_z", (p_hat - p_loss_theory) / se)
se_table = pd.DataFrame(rows).set_index("n")
print(f"theory P(losing year) = {p_loss_theory:.4%}")
print(se_table)
lab.record("ploss_theory", p_loss_theory)
lab.record("ploss_rows", len(N_LIST))
lab.record("ploss_rows_beyond_2", int((se_table["z"].abs() > 2).sum()))
lab.record("ploss_1000000_2se_pp", 200 * se_table.loc[1_000_000, "se"])
for n in N_LIST[1:]:   # n = 100 is too few for a normal approximation to be worth asserting
    check_mc(f"P(loss), n={n}", se_table.loc[n, "estimate"], p_loss_theory, se_table.loc[n, "se"])
# ten times the paths buys only √10 ≈ 3.16 times the precision
se_ratio = se_table.loc[10_000, "se"] / se_table.loc[1_000_000, "se"]
lab.record("se_ratio_100x", se_ratio)
assert 9 < se_ratio < 11

# %% [markdown]
# **What "± 2 SE" promises.** Repeat the whole n = 1,000 experiment 4,000 times, each time with
# fresh paths and its own estimated standard error. About 95% of the intervals "estimate ± 2 SE"
# should contain the true value (2 SE on each side of a normal covers 95.45%).

# %%
N_REPEATS, N_EACH = 4_000, 1_000
draws = NU + SIGMA * rng_se.standard_normal((N_REPEATS, N_EACH))
p_hats = (draws < 0).mean(axis=1)
ses = np.sqrt(p_hats * (1 - p_hats) / N_EACH)
covered = np.abs(p_hats - p_loss_theory) <= 2 * ses
coverage = float(covered.mean())
coverage_normal = float(norm.cdf(2) - norm.cdf(-2))
# The exact coverage of this interval: sum the binomial probabilities of every count whose interval
# contains the truth (a count is discrete, so it differs slightly from the normal 95.45%).
k_all = np.arange(N_EACH + 1)
p_all = k_all / N_EACH
inside = np.abs(p_all - p_loss_theory) <= 2 * np.sqrt(p_all * (1 - p_all) / N_EACH)
coverage_exact = float(binom.pmf(k_all, N_EACH, p_loss_theory)[inside].sum())
lab.record("cover_repeats", N_REPEATS)
lab.record("cover_n", N_EACH)
lab.record("cover_2se", coverage)
lab.record("cover_2se_normal", coverage_normal)
lab.record("cover_2se_exact", coverage_exact)
print(f"±2 SE intervals that covered the truth: {coverage:.2%} (exact {coverage_exact:.2%}, normal {coverage_normal:.2%})")
check_mc("coverage of ±2 SE", coverage, coverage_exact, se_prop(coverage_exact, N_REPEATS))

# %% [markdown]
# ## 2 · Random walks and GBM: simulate in log space
#
# A **random walk** adds independent random steps. GBM is a random walk in the *log* price:
# each day ln P moves by ν·dt + σ·√dt·Z with Z standard normal. Because the log increments add
# exactly, the **exact scheme** below is correct at any step size — one step of 10 years gives the
# same distribution of the final price as 2,520 daily steps.
#
# The **naive Euler scheme** instead compounds normal *simple* returns: P ← P·(1 + μΔ + σ√Δ·Z).
# It is only an approximation, its error grows with the step Δ, and with big steps it can even
# produce negative prices. A third classic bug: simulating in log space but with drift μ instead
# of ν = μ − σ²/2 (forgetting volatility drag).
#
# To compare the schemes fairly we feed them the SAME random path (common random numbers): daily
# Brownian increments are summed into weekly, monthly, quarterly and yearly increments.
# The asset here is volatile — 10% drift, 40% volatility (a single growth stock, or a levered
# index) — held for 10 years. We track P(losing more than half) and the 5th percentile of wealth.

# %%
MU_V, SIGMA_V, YEARS_V = 0.10, 0.40, 10
NU_V = MU_V - 0.5 * SIGMA_V ** 2
N_EU, CHUNK_EU = 40_000, 2_000
STEP_DAYS = {"1 day": 1, "1 week": 5, "1 month": 21, "1 quarter": 63, "1 year": 252}
n_days_v = PERIODS_PER_YEAR * YEARS_V


def euler_terminal(dW: np.ndarray, step_days: int, mu: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Naive Euler on PRICES with steps of `step_days` days: P ← P·(1 + μΔ + σΔW).

    dW holds daily Brownian increments (paths × days). Returns (terminal price from P0 = 1,
    ruined flag); a path whose price reaches zero or below is absorbed at 0 (it is bust).
    """
    n_paths = dW.shape[0]
    dW_k = dW.reshape(n_paths, -1, step_days).sum(axis=2)       # Brownian increment per step
    factor = 1.0 + mu * step_days * DT + sigma * dW_k
    ruined = (factor <= 0).any(axis=1)
    terminal = np.where(ruined, 0.0, np.prod(np.maximum(factor, 1e-300), axis=1))
    return terminal, ruined


def exact_terminal(dW: np.ndarray, step_days: int, mu: float, sigma: float) -> np.ndarray:
    """Exact GBM in log space with steps of `step_days` days: ln P += (μ − σ²/2)Δ + σΔW."""
    n_paths = dW.shape[0]
    dW_k = dW.reshape(n_paths, -1, step_days).sum(axis=2)
    return np.exp(((mu - 0.5 * sigma ** 2) * step_days * DT + sigma * dW_k).sum(axis=1))


schemes = {f"Euler, {k}": [] for k in STEP_DAYS}
schemes["exact, 1 step"] = []
schemes["exact, 1 year"] = []
schemes["log space, drift μ"] = []
ruin = {k: [] for k in STEP_DAYS}
for _ in range(N_EU // CHUNK_EU):
    dW = np.sqrt(DT) * rng_gbm.standard_normal((CHUNK_EU, n_days_v))
    for label, k in STEP_DAYS.items():
        term, bust = euler_terminal(dW, k, MU_V, SIGMA_V)
        schemes[f"Euler, {label}"].append(term)
        ruin[label].append(bust)
    one_step = np.exp(NU_V * YEARS_V + SIGMA_V * dW.sum(axis=1))           # a single 10-year step
    yearly = exact_terminal(dW, 252, MU_V, SIGMA_V)
    daily = exact_terminal(dW, 1, MU_V, SIGMA_V)
    assert np.allclose(one_step, yearly) and np.allclose(one_step, daily)  # the step does not matter
    schemes["exact, 1 step"].append(one_step)
    schemes["exact, 1 year"].append(yearly)
    schemes["log space, drift μ"].append(np.exp(MU_V * YEARS_V + SIGMA_V * dW.sum(axis=1)))
schemes = {k: np.concatenate(v) for k, v in schemes.items()}
ruin = {k: np.concatenate(v) for k, v in ruin.items()}

s_v = SIGMA_V * np.sqrt(YEARS_V)
theory_v = {"p_half": float(norm.cdf((np.log(0.5) - NU_V * YEARS_V) / s_v)),
            "q05": float(np.exp(NU_V * YEARS_V + s_v * norm.ppf(0.05))),
            "median": float(np.exp(NU_V * YEARS_V)),
            "mean": float(np.exp(MU_V * YEARS_V))}
q05 = lambda a: float(np.quantile(a, 0.05))   # noqa: E731


def key_of(name: str) -> str:
    return (name.replace("Euler, 1 ", "euler_").replace("exact, 1 ", "exact_")
            .replace("log space, drift μ", "wrongdrift").replace(" ", "_"))


gbm_rows = []
for name, wealth in schemes.items():
    p_half = float((wealth < 0.5).mean())
    se_half = se_prop(theory_v["p_half"], N_EU)
    row = {"scheme": name, "p_half": p_half, "z_half": (p_half - theory_v["p_half"]) / se_half,
           "q05": q05(wealth), "q05_se": batch_se(q05, wealth), "median": float(np.median(wealth))}
    gbm_rows.append(row)
    k = key_of(name)
    lab.record(f"gbm_{k}_p_half", p_half)
    lab.record(f"gbm_{k}_z_half", row["z_half"])
    lab.record(f"gbm_{k}_q05", row["q05"])
    lab.record(f"gbm_{k}_median", row["median"])
gbm_table = pd.DataFrame(gbm_rows).set_index("scheme")
for label in STEP_DAYS:
    lab.record(f"gbm_{key_of('Euler, ' + label)}_ruined", float(ruin[label].mean()))
print(pd.Series(theory_v).round(4))
print(gbm_table.round(4))
print({k: f"{v.mean():.3%}" for k, v in ruin.items()})

lab.record("gbm_mu_v", MU_V)
lab.record("gbm_sigma_v", SIGMA_V)
lab.record("gbm_years_v", YEARS_V)
lab.record("gbm_paths", N_EU)
lab.record("gbm_theory_p_half", theory_v["p_half"])
lab.record("gbm_theory_q05", theory_v["q05"])
lab.record("gbm_theory_median", theory_v["median"])
lab.record("gbm_theory_mean", theory_v["mean"])
lab.record("gbm_se_half", se_prop(theory_v["p_half"], N_EU))
lab.record("gbm_se_half_pp", 100 * se_prop(theory_v["p_half"], N_EU))
wrong = schemes["log space, drift μ"]
lab.record("gbm_wrongdrift_mean", float(wrong.mean()))
lab.record("gbm_wrongdrift_factor", float(np.exp(0.5 * SIGMA_V ** 2 * YEARS_V)))

# The exact scheme matches theory whatever the step (one 10-year step and ten yearly steps give the
# identical final prices, asserted above, so one check covers both) ...
check_mc("exact: P(lose half)", gbm_table.loc["exact, 1 step", "p_half"], theory_v["p_half"],
         se_prop(theory_v["p_half"], N_EU))
check_mc("exact: 5th percentile", gbm_table.loc["exact, 1 step", "q05"], theory_v["q05"],
         gbm_table.loc["exact, 1 step", "q05_se"])
# ... while Euler with yearly steps is biased far beyond sampling error, and can go bust
assert gbm_table.loc["Euler, 1 year", "z_half"] > 10
assert ruin["1 year"].mean() > 0.01 and ruin["1 day"].mean() == 0.0
# Same random path for every scheme, so the Euler error is measured path by path, against the exact
# price: the share of paths classified differently (above/below half) shrinks as the step shrinks.
exact_below = schemes["exact, 1 step"] < 0.5
paired = {label: (schemes[f"Euler, {label}"] < 0.5).astype(float) - exact_below for label in STEP_DAYS}
paired_bias = {label: float(d.mean()) for label, d in paired.items()}
paired_se = {label: float(d.std(ddof=1) / np.sqrt(N_EU)) for label, d in paired.items()}
print("Euler minus exact, P(lose half), paired:", {k: f"{v:+.3%}" for k, v in paired_bias.items()})
for label in STEP_DAYS:
    lab.record(f"gbm_{key_of('Euler, ' + label)}_paired_bias", paired_bias[label])
    lab.record(f"gbm_{key_of('Euler, ' + label)}_paired_se", paired_se[label])
    lab.record(f"gbm_{key_of('Euler, ' + label)}_paired_bias_pp", 100 * paired_bias[label])
    lab.record(f"gbm_{key_of('Euler, ' + label)}_paired_se_pp", 100 * paired_se[label])
for label in ("1 year", "1 quarter"):        # big steps: the bias is unmistakable
    assert paired_bias[label] > N_SE * paired_se[label], f"Euler {label}: bias not detected"
for big, small in (("1 year", "1 quarter"), ("1 quarter", "1 month")):   # and it shrinks with the step
    d = paired[big] - paired[small]
    assert d.mean() > N_SE * d.std(ddof=1) / np.sqrt(N_EU), f"Euler bias does not shrink from {big} to {small}"
# Forgetting −σ²/2 makes the log-space scheme the exact median of a DIFFERENT asset: e^{μτ}, not e^{ντ}
check_mc("drift μ: median", gbm_table.loc["log space, drift μ", "median"], float(np.exp(MU_V * YEARS_V)),
         batch_se(np.median, wrong))

# %% [markdown]
# A picture of the thing we are simulating: three 10-year paths of the base asset (8% drift, 20%
# volatility), daily steps, exact scheme, on a log scale, with the median path exp(ν × years).

# %%
YEARS_DD = 10
n_days_dd = PERIODS_PER_YEAR * YEARS_DD
N_DD, CHUNK_DD = 6_000, 1_000
mdd = []
for i in range(N_DD // CHUNK_DD):
    log_r = NU * DT + SIGMA * np.sqrt(DT) * rng_dd.standard_normal((n_days_dd, CHUNK_DD))
    mdd.append(max_drawdown_paths(np.expm1(log_r)))
    if i == 0:
        sample_paths = np.exp(np.cumsum(log_r[:, :3], axis=0))
mdd = np.concatenate(mdd)
t_years = np.arange(1, n_days_dd + 1) / PERIODS_PER_YEAR
median_path = pd.Series(np.exp(NU * t_years), index=t_years)
lab.chart("gbm_paths", charts.line_chart(
    [charts.Series("path 1", in_years(sample_paths[:, 0]), role="strategy"),
     charts.Series("path 2", in_years(sample_paths[:, 1]), role="alt1"),
     charts.Series("path 3", in_years(sample_paths[:, 2]), role="alt2"),
     charts.Series("median path", median_path, role="benchmark", end_label="median")],
    title="Three simulated 10-year GBM paths: 8% drift, 20% volatility (log scale)",
    logy=True, y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))
lab.record("paths_final_1", float(sample_paths[-1, 0]))
lab.record("paths_final_2", float(sample_paths[-1, 1]))
lab.record("paths_final_3", float(sample_paths[-1, 2]))
lab.record("median_final", float(np.exp(NU * YEARS_DD)))

# %% [markdown]
# ## 3 · Path-dependent quantities: the maximum drawdown
#
# The maximum drawdown depends on the whole path, and its distribution has no simple closed form
# (Magdon-Ismail et al. 2004 give it as an infinite series). Simulation is the practical tool.
# How deep a drawdown should you expect from a genuinely good asset over ten years? The standard
# error of a quantile has no simple formula either, so we use batch means (`qn.stats.batch_se`).

# %%
mdd_median = float(np.median(mdd))
mdd_q05 = q05(mdd)
lab.record("dd_paths", N_DD)
lab.record("dd_years", YEARS_DD)
lab.record("dd_median", mdd_median)
lab.record("dd_median_se", batch_se(np.median, mdd))
lab.record("dd_median_se_pp", 100 * batch_se(np.median, mdd))
lab.record("dd_q05", mdd_q05)
lab.record("dd_q05_se", batch_se(q05, mdd))
lab.record("dd_mean", float(mdd.mean()))
lab.record("dd_sharpe", MU / SIGMA)
for level in (0.20, 0.30, 0.50):
    p = float((mdd <= -level).mean())
    lab.record(f"dd_p{int(level * 100)}", p)
    lab.record(f"dd_p{int(level * 100)}_se", se_prop(p, N_DD))
print(f"10-year max drawdown: median {mdd_median:.1%} (±{batch_se(np.median, mdd):.1%}), "
      f"worst 5% beyond {mdd_q05:.1%}; P(≤ −20%) {np.mean(mdd <= -0.2):.1%}, P(≤ −50%) {np.mean(mdd <= -0.5):.1%}")
lab.chart("mdd_hist", charts.histogram(
    mdd, title=f"Maximum drawdown over 10 years — {N_DD:,} simulated paths, 8% drift, 20% volatility",
    bins=50, x_fmt=charts.fmt_pct(0), tail_below=mdd_q05, tail_label="worst 5%"))
assert -0.45 < mdd_median < -0.20   # sanity: a Sharpe-0.4 asset has deep drawdowns

# %% [markdown]
# ## 4 · Touching a barrier: simulation vs the reflection principle
#
# P(the price falls 20% below its start at some point within a year) — the question behind a stop
# loss or a margin call (Session 2.4). For GBM watched *continuously* the reflection principle gives
# a closed form, `qn.stats.p_touch`. A simulation can only look at the price at its time steps, and
# so can your broker: daily closes, say. Checking less often misses dips that recover between
# checks, so the discretely monitored probability is lower. The Broadie–Glasserman–Kou correction
# shifts the barrier away by 0.5826·σ·√Δt to account for that (`p_touch(..., dt=Δt)`).
#
# We simulate 200,000 one-year paths on a daily grid and look at them daily, weekly, monthly and
# quarterly. We also estimate the *continuous* answer with a Brownian-bridge trick: given two
# neighbouring daily log prices a and b above the barrier h, the chance that the path dipped to h in
# between is exp(−2(a − h)(b − h)/(σ²Δt)), so each path contributes its probability of a hidden touch.

# %%
MU_B, SIGMA_B = MU, SIGMA      # the base asset (Try it: set MU_B = SIGMA_B ** 2 / 2 for a driftless log price)
NU_B = MU_B - 0.5 * SIGMA_B ** 2
BARRIER = 0.80                 # 20% below the start
N_BAR = 200_000
MONITOR = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63}
h = float(np.log(BARRIER))
log_p = np.zeros(N_BAR)
running_min = {name: np.zeros(N_BAR) for name in MONITOR}
survive_bridge = np.ones(N_BAR)      # P(no touch between the daily checks so far | the daily prices)
running_max = np.zeros(N_BAR)
max_dd_1y = np.zeros(N_BAR)          # deepest fall from a running peak (log), for comparison
for t in range(1, PERIODS_PER_YEAR + 1):
    prev = log_p
    log_p = log_p + NU_B * DT + SIGMA_B * np.sqrt(DT) * rng_bar.standard_normal(N_BAR)
    for name, every in MONITOR.items():
        if t % every == 0 or t == PERIODS_PER_YEAR:   # every Δt, and at the end of the year
            running_min[name] = np.minimum(running_min[name], log_p)
    a, b = np.maximum(prev - h, 0.0), np.maximum(log_p - h, 0.0)
    survive_bridge *= 1.0 - np.exp(-2.0 * a * b / (SIGMA_B ** 2 * DT))
    running_max = np.maximum(running_max, log_p)
    max_dd_1y = np.minimum(max_dd_1y, log_p - running_max)

p_cont = p_touch(BARRIER, MU_B, SIGMA_B, 1.0)
bar_rows = []
for name, every in MONITOR.items():
    p_sim = float((running_min[name] <= h).mean())
    se = se_prop(p_sim, N_BAR)
    p_bgk = p_touch(BARRIER, MU_B, SIGMA_B, 1.0, dt=every * DT)
    bar_rows.append({"monitoring": name, "sim": p_sim, "se": se, "bgk": p_bgk,
                     "z_bgk": (p_sim - p_bgk) / se, "continuous": p_cont, "z_cont": (p_sim - p_cont) / se})
    lab.record(f"bar_{name}_sim", p_sim)
    lab.record(f"bar_{name}_se", se)
    lab.record(f"bar_{name}_se_pp", 100 * se)
    lab.record(f"bar_{name}_bgk", p_bgk)
    lab.record(f"bar_{name}_z_bgk", (p_sim - p_bgk) / se)
    lab.record(f"bar_{name}_gap_bgk_pp", 100 * (p_sim - p_bgk))
    lab.record(f"bar_{name}_gap_cont_pp", 100 * (p_sim - p_cont))
bar_table = pd.DataFrame(bar_rows).set_index("monitoring")
touch_cont = 1.0 - survive_bridge
p_bridge = float(touch_cont.mean())
se_bridge = float(touch_cont.std(ddof=1) / np.sqrt(N_BAR))
print(bar_table.round(4))
print(f"continuous formula {p_cont:.4%} · Brownian-bridge simulation {p_bridge:.4%} ± {se_bridge:.4%}")
lab.record("bar_paths", N_BAR)
lab.record("bar_level", BARRIER)
lab.record("bar_continuous", p_cont)
lab.record("bar_bridge", p_bridge)
lab.record("bar_bridge_se", se_bridge)
lab.record("bar_bridge_se_pp", 100 * se_bridge)
lab.record("bar_bridge_gap_pp", 100 * (p_bridge - p_cont))
lab.record("bar_bgk_beta", qn.stats.BGK_BETA)

# Daily monitoring: the corrected formula is right within sampling error.
check_mc("barrier, daily vs BGK", bar_table.loc["daily", "sim"], bar_table.loc["daily", "bgk"], bar_table.loc["daily", "se"])
# Weekly: BGK is an expansion in √Δt and its next-order error (about +0.1 pp here, +1 to +2 SE across
# seeds) starts to show, so the tolerance is 4 SE plus a 0.15 pp allowance for that bias.
check_mc("barrier, weekly vs BGK", bar_table.loc["weekly", "sim"], bar_table.loc["weekly", "bgk"],
         bar_table.loc["weekly", "se"], allowance=0.0015)
# Every discrete check misses touches: all lie clearly below the continuous formula.
assert (bar_table["z_cont"] < -N_SE).all()
# The continuous formula itself, checked by the bridge estimator.
check_mc("barrier, continuous (bridge)", p_bridge, p_cont, se_bridge)
# Coarse monitoring: the correction is an approximation for small Δt and under-shoots here; the gap is
# recorded and shown on the page, not asserted to be zero.
assert bar_table.loc["quarterly", "z_bgk"] > 0

# %% [markdown]
# **The reflection principle, term by term.** p_touch is the sum of two probabilities:
#
#     P(touch) = P(end below h) + P(touch h and end above it)
#              = Φ((h − ντ)/(σ√τ)) + exp(2νh/σ²) · Φ((h + ντ)/(σ√τ))
#
# The first term needs no argument: ending below the barrier means you crossed it. The second is the
# reflection: reflect a path's remainder after its first touch and "touched, ended above" maps onto
# "ended below" — one-for-one when there is no drift (ν = 0), reweighted by exp(2νh/σ²) when there
# is. The bridge-corrected simulation measures both terms separately.

# %%
end_below = (log_p <= h).astype(float)
touch_end_above = touch_cont * (log_p > h)      # continuous touch probability, for paths ending above
s1 = SIGMA_B * np.sqrt(1.0)
term1 = float(norm.cdf((h - NU_B) / s1))
term2 = float(np.exp(2 * NU_B * h / SIGMA_B ** 2) * norm.cdf((h + NU_B) / s1))
assert np.isclose(term1 + term2, p_cont)
check_mc("reflection term 1", float(end_below.mean()), term1, se_prop(term1, N_BAR))
check_mc("reflection term 2", float(touch_end_above.mean()), term2,
         float(touch_end_above.std(ddof=1) / np.sqrt(N_BAR)))
lab.record("refl_term1", term1)
lab.record("refl_term2", term2)
lab.record("refl_term1_sim", float(end_below.mean()))
lab.record("refl_term2_sim", float(touch_end_above.mean()))
lab.record("refl_weight", float(np.exp(2 * NU_B * h / SIGMA_B ** 2)))
print(f"P(end below) {term1:.4%} (simulated {end_below.mean():.4%}) · "
      f"P(touch, end above) {term2:.4%} (simulated {touch_end_above.mean():.4%}) · weight {np.exp(2 * NU_B * h / SIGMA_B ** 2):.3f}")
# the driftless case: both terms equal — every touching path has a mirror twin that ends below
assert np.isclose(p_touch(BARRIER, 0.5 * SIGMA_B ** 2, SIGMA_B, 1.0), 2 * norm.cdf(h / s1))

# A drawdown is a barrier measured from a moving peak, so it is at least as likely as falling the
# same distance below the START.
p_dd20_1y = float((max_dd_1y <= h).mean())
lab.record("bar_dd20_1y", p_dd20_1y)
assert p_dd20_1y > bar_table.loc["daily", "sim"] + N_SE * se_prop(p_dd20_1y, N_BAR)
print(f"P(a 20% drawdown within a year) {p_dd20_1y:.2%} vs P(touch 20% below the start) {bar_table.loc['daily', 'sim']:.2%}")

# %% [markdown]
# ## 5 · The bootstrap: resampling history (and what it cannot do)
#
# Now pretend we do not know the model. We have one 10-year history of daily returns — generated by
# GARCH(1,1) with Student-t shocks, so it has volatility clustering and fat tails like real returns
# (Sessions 5.1 and 5.5). The **iid bootstrap** resamples days with replacement: it keeps each day's
# return but scrambles their order. The **block bootstrap** resamples runs of consecutive days
# (here the circular block bootstrap, with the block length chosen by the Politis–White rule in
# `arch`), so calm and stormy stretches survive.
#
# Clustering shows up as positive autocorrelation of SQUARED returns: a big move today makes a big
# move tomorrow more likely. Watch what each bootstrap does to it.

# %%
YEARS_B, N_BOOT, N_ACF_REPS, MAX_LAG = 10, 2_000, 200, 20
n_b = PERIODS_PER_YEAR * YEARS_B
# three child streams: the observed history, the resampling, and the "truth" draws from the model,
# so changing the block length (Try it) never changes the history or the truth
rng_history, rng_resample, rng_truth = rng_boot.spawn(3)
history = qn.synth.garch_returns(n_b, rng=rng_history)["ret"].to_numpy()


def acf_squared(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation of squared returns at lags 1..max_lag — the fingerprint of volatility clustering."""
    y = x ** 2 - np.mean(x ** 2)
    denom = float(y @ y)
    return np.array([float(y[:-k] @ y[k:]) / denom for k in range(1, max_lag + 1)])


def iid_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """IID bootstrap: n_boot resamples of n days drawn with replacement."""
    return rng.integers(0, n, size=(n_boot, n))


def block_indices(n: int, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap: glue together random runs of `block` consecutive days (wrapping at the end)."""
    n_blocks = -(-n // block)                                          # ceiling division
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


block_len = int(round(float(optimal_block_length(history ** 2)["circular"].iloc[0])))
boot_iid = history[iid_indices(n_b, N_BOOT, rng_resample)]
boot_block = history[block_indices(n_b, N_BOOT, block_len, rng_resample)]

acf_orig = acf_squared(history, MAX_LAG)
acf_iid_reps = np.array([acf_squared(x, MAX_LAG) for x in boot_iid[:N_ACF_REPS]])
acf_block_reps = np.array([acf_squared(x, MAX_LAG) for x in boot_block[:N_ACF_REPS]])
acf_iid, acf_block = acf_iid_reps.mean(axis=0), acf_block_reps.mean(axis=0)
lags = np.arange(1, MAX_LAG + 1)
lab.chart("acf_boot", charts.line_chart(
    [charts.Series("original history", pd.Series(acf_orig, index=lags), role="strategy", end_label="original"),
     charts.Series("block bootstrap", pd.Series(acf_block, index=lags), role="alt2", end_label="block"),
     charts.Series("iid bootstrap", pd.Series(acf_iid, index=lags), role="alt1", end_label="iid")],
    title="Autocorrelation of squared returns, by lag in trading days", y_fmt=charts.fmt_num(2), hline=0.0))


def m10(acf: np.ndarray) -> float:
    """Average autocorrelation over lags 1-10 (averaged over resamples too, for a 2-D array)."""
    return float(np.asarray(acf)[..., :10].mean())


lab.record("boot_years", YEARS_B)
lab.record("boot_n", N_BOOT)
lab.record("boot_block_len", block_len)
lab.record("boot_acf_reps", N_ACF_REPS)
lab.record("boot_acf_orig_1_10", m10(acf_orig))
lab.record("boot_acf_iid_1_10", m10(acf_iid_reps))
lab.record("boot_acf_block_1_10", m10(acf_block_reps))
lab.record("boot_acf_orig_lag1", float(acf_orig[0]))
print(f"block length {block_len} days · mean ACF of r², lags 1-10: original {m10(acf_orig):.3f}, "
      f"iid {m10(acf_iid_reps):.3f}, block {m10(acf_block_reps):.3f}")
# iid resampling destroys the order, so the clustering signature is gone (mean ≈ 0) ...
iid_1_10 = acf_iid_reps[:, :10].mean(axis=1)
check_mc("iid bootstrap ACF(r²) ≈ 0", float(iid_1_10.mean()), -1.0 / n_b,
         float(iid_1_10.std(ddof=1) / np.sqrt(N_ACF_REPS)))
# ... while blocks keep most of it (they cut the dependence only at the block joins)
assert m10(acf_block_reps) > 0.5 * m10(acf_orig)

# %% [markdown]
# **Why it matters: the uncertainty of an estimate.** How precisely does 10 years of data pin down
# the annual volatility? Because we know the model, we can answer honestly: draw 500 fresh 10-year
# histories and look at how much their volatility estimates scatter. Then ask each bootstrap the
# same question using only the one history we have.

# %%
N_TRUTH = 500
vol = lambda x: x.std(axis=-1, ddof=1) * np.sqrt(PERIODS_PER_YEAR)   # noqa: E731
truth_vols = np.array([vol(qn.synth.garch_returns(n_b, rng=rng_truth)["ret"].to_numpy()) for _ in range(N_TRUTH)])
true_sd = float(truth_vols.std(ddof=1))
iid_sd = float(vol(boot_iid).std(ddof=1))
block_sd = float(vol(boot_block).std(ddof=1))
omega, a_g, b_g = 2e-6, 0.08, 0.90                 # the garch_returns defaults
lab.record("boot_vol_sample", float(vol(history)))
lab.record("boot_vol_true_uncond", float(np.sqrt(omega / (1 - a_g - b_g) * PERIODS_PER_YEAR)))
lab.record("boot_vol_true_mean", float(truth_vols.mean()))
lab.record("boot_vol_se_truth", true_sd)
lab.record("boot_vol_se_iid", iid_sd)
lab.record("boot_vol_se_block", block_sd)
lab.record("boot_vol_se_ratio_iid", true_sd / iid_sd)
lab.record("boot_vol_se_truth_pp", 100 * true_sd)
lab.record("boot_vol_se_iid_pp", 100 * iid_sd)
lab.record("boot_vol_se_block_pp", 100 * block_sd)
lab.record("boot_truth_n", N_TRUTH)
print(f"SE of 10-year vol: truth {true_sd:.2%} · iid bootstrap {iid_sd:.2%} · block bootstrap {block_sd:.2%}")
assert iid_sd < block_sd                 # ignoring clustering understates the uncertainty ...
assert iid_sd < 0.75 * true_sd           # ... by a wide margin: the truth scatters far more

# %% [markdown]
# **What no bootstrap can fix: a short history.** Keep only the first two years. Every bootstrap
# year is built from those 504 days, so none can contain a day worse than the worst day in them.
# The model knows better.

# %%
N_SHORT, N_FRESH_YEARS = 2 * PERIODS_PER_YEAR, 2_000
short = history[:N_SHORT]
worst_day = float(short.min())
boot_years = short[block_indices(N_SHORT, N_BOOT, block_len, rng_resample)][:, :PERIODS_PER_YEAR]
p_worse_boot = float((boot_years.min(axis=1) < worst_day).mean())
fresh_min = np.array([qn.synth.garch_returns(PERIODS_PER_YEAR, rng=rng_truth)["ret"].min() for _ in range(N_FRESH_YEARS)])
p_worse_true = float((fresh_min < worst_day).mean())
lab.record("short_years", 2)
lab.record("short_worst_day", worst_day)
lab.record("short_p_worse_boot", p_worse_boot)
lab.record("short_p_worse_true", p_worse_true)
lab.record("short_p_worse_true_se", se_prop(p_worse_true, N_FRESH_YEARS))
lab.record("short_true_q01_worst", float(np.quantile(fresh_min, 0.01)))
lab.record("short_fresh_years", N_FRESH_YEARS)
print(f"worst day in 2 years {worst_day:.2%}; P(a year has a worse day): bootstrap {p_worse_boot:.1%}, model {p_worse_true:.1%}")
assert p_worse_boot == 0.0
assert p_worse_true > N_SE * se_prop(p_worse_true, N_FRESH_YEARS)

# %% [markdown]
# ## 6 · Variance reduction: antithetic variates and control variates
#
# Estimate the expected payoff at the end of one year of a call option struck at the starting price,
# E[max(P_τ − K, 0)] with τ = 1 year, for the base asset (P₀ = K = $100). It has a closed form — the Black–Scholes
# formula with the drift μ in place of the interest rate and no discounting (Session 10.3 derives
# it) — which we use only to check the simulations.
#
# * **Antithetic variates**: use every draw Z twice, as Z and −Z, and average the pair.
#   At equal cost (the same number of normal draws) the variance falls by the factor 1/(1 + ρ),
#   where ρ is the correlation between f(Z) and f(−Z).
# * **Control variates**: subtract b·(C − E[C]) for a quantity C whose mean you know — here the final
#   price P_τ, with E[P_τ] = P₀e^{μτ}. The best b is Cov(Y, C)/Var(C) and the variance falls by
#   1/(1 − ρ²_{Y,C}). (Estimating b from the same paths adds a bias of order 1/n; negligible here.)
#
# The variance-reduction ratio (VRR) = Var(plain estimate) / Var(new estimate) at equal cost:
# how many times more paths the plain method would need for the same standard error.

# %%
P0, K, TAU_VR = 100.0, 100.0, 1.0
N_VR = 200_000                                      # normal draws per method (equal cost)
s_vr = SIGMA * np.sqrt(TAU_VR)
d1 = (np.log(P0 / K) + (MU + 0.5 * SIGMA ** 2) * TAU_VR) / s_vr
d2 = d1 - s_vr
call_theory = float(P0 * np.exp(MU * TAU_VR) * norm.cdf(d1) - K * norm.cdf(d2))
put_theory = float(K * norm.cdf(-d2) - P0 * np.exp(MU * TAU_VR) * norm.cdf(-d1))
E_P_TAU = P0 * np.exp(MU * TAU_VR)


def terminal(z: np.ndarray) -> np.ndarray:
    return P0 * np.exp(NU * TAU_VR + s_vr * z)


def call_payoff(z: np.ndarray) -> np.ndarray:
    return np.maximum(terminal(z) - K, 0.0)


def straddle_payoff(z: np.ndarray) -> np.ndarray:
    """|P_τ − K| — nearly symmetric in Z, the case where antithetic pairs work against you."""
    return np.abs(terminal(z) - K)


z_vr = rng_vr.standard_normal(N_VR)
half = z_vr[: N_VR // 2]


def plain(f):
    y = f(z_vr)
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(N_VR)), float(y.var(ddof=1) / N_VR)


def antithetic(f):
    y_pair = 0.5 * (f(half) + f(-half))             # N_VR/2 pairs = N_VR normal draws
    rho = float(np.corrcoef(f(half), f(-half))[0, 1])
    return float(y_pair.mean()), float(y_pair.std(ddof=1) / np.sqrt(half.size)), float(y_pair.var(ddof=1) / half.size), rho


est_plain, se_plain, var_plain = plain(call_payoff)
est_anti, se_anti, var_anti, rho_anti = antithetic(call_payoff)
y = call_payoff(z_vr)
c = terminal(z_vr)
b_star = float(np.cov(y, c)[0, 1] / c.var(ddof=1))
y_cv = y - b_star * (c - E_P_TAU)
est_cv, se_cv = float(y_cv.mean()), float(y_cv.std(ddof=1) / np.sqrt(N_VR))
rho_cv = float(np.corrcoef(y, c)[0, 1])
vrr_anti = var_plain / var_anti
vrr_cv = se_plain ** 2 / se_cv ** 2
s_plain, s_se_plain, s_var_plain = plain(straddle_payoff)
s_anti, s_se_anti, s_var_anti, s_rho = antithetic(straddle_payoff)
vrr_straddle = s_var_plain / s_var_anti

print(f"call payoff theory {call_theory:.4f}")
print(f"  plain      {est_plain:.4f} ± {se_plain:.4f}")
print(f"  antithetic {est_anti:.4f} ± {se_anti:.4f}  VRR {vrr_anti:.2f}  (1/(1+ρ) = {1 / (1 + rho_anti):.2f}, ρ = {rho_anti:.2f})")
print(f"  control    {est_cv:.4f} ± {se_cv:.4f}  VRR {vrr_cv:.2f}  (1/(1−ρ²) = {1 / (1 - rho_cv ** 2):.2f}, ρ = {rho_cv:.2f}, b = {b_star:.3f})")
print(f"straddle: antithetic VRR {vrr_straddle:.2f} (ρ = {s_rho:.2f})")
for k_, v_ in {"vr_n": N_VR, "vr_s0": P0, "vr_k": K, "vr_theory": call_theory,
               "vr_plain": est_plain, "vr_plain_se": se_plain, "vr_anti": est_anti, "vr_anti_se": se_anti,
               "vr_cv": est_cv, "vr_cv_se": se_cv, "vr_rho_anti": rho_anti, "vr_rho_cv": rho_cv,
               "vr_b": b_star, "vrr_anti": vrr_anti, "vrr_cv": vrr_cv, "vrr_anti_formula": 1 / (1 + rho_anti),
               "vrr_cv_formula": 1 / (1 - rho_cv ** 2), "vr_straddle_theory": call_theory + put_theory,
               "vr_straddle_rho": s_rho, "vrr_straddle": vrr_straddle}.items():
    lab.record(k_, v_)
check_mc("call payoff, plain", est_plain, call_theory, se_plain)
check_mc("call payoff, antithetic", est_anti, call_theory, se_anti)
check_mc("call payoff, control variate", est_cv, call_theory, se_cv)
check_mc("straddle, antithetic", s_anti, call_theory + put_theory, s_se_anti)
check_mc("E[P_τ], plain", float(c.mean()), E_P_TAU, float(c.std(ddof=1) / np.sqrt(N_VR)))
# A fixed 2% relative tolerance, not an SE-based check_mc call: both sides here are Monte Carlo
# estimates built from the SAME draws (the measured VRR and its 1/(1±ρ) formula share z_vr), not an
# estimate compared with an exact theoretical value, so there is no independent standard error to
# compare against. The actual gaps are ~0.2% and ~0.0006%, far inside this budget.
assert np.isclose(vrr_anti, 1 / (1 + rho_anti), rtol=0.02)
assert np.isclose(vrr_cv, 1 / (1 - rho_cv ** 2), rtol=0.02)
assert vrr_anti > 1.5 and vrr_cv > 4.0      # both help, the control variate a lot
assert vrr_straddle < 0.9                    # antithetic pairs HURT a nearly symmetric payoff

# %% [markdown]
# ## 7 · Many asserts, one lab: the multiple-testing trap
#
# Every `check_mc` above is a statistical test: even with perfect code, each estimate lands beyond
# c standard errors with probability 2Φ(−c). A lab with k independent checks fails by pure luck with
# probability 1 − (1 − 2Φ(−c))^k. This course's own review found a lab with about 27 checks at 3 SE
# that failed on 8% of alternative seeds; the formula says 7%. At 4 SE it is under 0.2%.
# We verify the formula by simulating 200,000 imaginary labs whose code is perfect.

# %%
K_REVIEW, N_LABS = 27, 200_000
z_labs = rng_mt.standard_normal((N_LABS, K_REVIEW))
worst = np.abs(z_labs).max(axis=1)
for c_ in (2, 3, 4):
    p_one = float(2 * norm.cdf(-c_))
    fwer_theory = 1 - (1 - p_one) ** K_REVIEW
    fwer_sim = float((worst > c_).mean())
    lab.record(f"mt_p_one_{c_}se", p_one)
    lab.record(f"mt_fwer_{c_}se", fwer_theory)
    lab.record(f"mt_fwer_{c_}se_sim", fwer_sim)
    print(f"{c_} SE: one check fails {p_one:.3%} · lab of {K_REVIEW} fails {fwer_theory:.2%} (simulated {fwer_sim:.2%})")
    check_mc(f"family-wise rate at {c_} SE", fwer_sim, fwer_theory, se_prop(fwer_theory, N_LABS))
lab.record("mt_k_review", K_REVIEW)
lab.record("mt_n_labs", N_LABS)
for k_ in (10, 50, 100):
    for c_ in (3, 4):
        lab.record(f"mt_fwer_k{k_}_{c_}se", 1 - (1 - 2 * norm.cdf(-c_)) ** k_)

# %% [markdown]
# **This lab, audited.** How many statistical checks did it make, and what is the chance that a
# correct lab fails one of them by luck? (Two of the checks above are the multiple-testing
# simulation itself; they count too.)

# %%
k_self = len(MC_CHECKS)
z_self = np.array([z for _, z in MC_CHECKS])
fwer_self_4 = 1 - (1 - 2 * norm.cdf(-N_SE)) ** k_self
fwer_self_3 = 1 - (1 - 2 * norm.cdf(-3)) ** k_self
fwer_self_2 = 1 - (1 - 2 * norm.cdf(-2)) ** k_self
lab.record("self_k", k_self)
lab.record("self_fwer_4se", fwer_self_4)
lab.record("self_fwer_3se", fwer_self_3)
lab.record("self_fwer_2se", fwer_self_2)
lab.record("self_max_abs_z", float(np.abs(z_self).max()))
lab.record("self_n_beyond_2", int((np.abs(z_self) > 2).sum()))
print(f"{k_self} statistical checks; largest |z| = {np.abs(z_self).max():.2f}; "
      f"P(a correct lab fails one): {fwer_self_2:.1%} at 2 SE, {fwer_self_3:.1%} at 3 SE, {fwer_self_4:.2%} at 4 SE")
for name, z in sorted(MC_CHECKS, key=lambda t: -abs(t[1]))[:5]:
    print(f"   {z:+.2f}  {name}")

# %%
lab.save()
