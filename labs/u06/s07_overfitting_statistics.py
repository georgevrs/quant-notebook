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
# # 6.7 · Overfitting Statistics: PSR, DSR, MinTRL & PBO — companion lab
#
# **Quant Notebook** · Unit 6 · Session 7 · [Read the session](https://georgevrs.github.io/quant-notebook/unit06-backtesting-discipline/session07-overfitting-statistics.html)
#
# The probabilistic and deflated Sharpe ratios, minimum track record length, and the probability of
# backtest overfitting — how to price in the number of things you tried.
#
# We implement all four statistics from their published formulas (Bailey & López de Prado, 2012 and
# 2014; Bailey, Borwein, López de Prado & Zhu, 2016), then put them to work on two honest experiments:
# a fresh N-strategy search over pure noise (does DSR see through an inflated best-of-N Sharpe ratio?),
# and a parameter sweep of one trading rule on one noisy price series, scored with CSCV/PBO.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# **Risk-free rate.** All returns below are synthetic *excess* returns; the risk-free rate is taken
# as 0 throughout, so "return" and "excess return" mean the same thing in this lab.

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
from quantnb import charts, stats, synth

lab = qn.Lab("6.7")  # seeds the random generators: every run gives the same numbers
PPY = 252  # trading days per year, this course's convention (AUTHORING.md §9)

# %% [markdown]
# ## 1 · PSR, MinTRL and DSR, from their exact formulas
#
# All three build on one quantity: the **probabilistic Sharpe ratio**, the probability that the
# *true* Sharpe ratio exceeds a benchmark `sr_star`, given an estimate `sr_hat` from `T` periods
# whose returns have sample skewness `skew` and sample **kurtosis** (raw, not excess — Session 3.2's
# convention, so a normal distribution has `kurt = 3`).
#
# Bailey & López de Prado (2012) give the standard deviation of the Sharpe ratio estimator as
# `sigma(sr) = sqrt((1 - skew*sr + (kurt-1)/4*sr**2) / T)`. At `skew = 0, kurt = 3` this is
# `sqrt((1 + sr**2/2) / T)` — exactly Lo's (2002) formula already in `quantnb.stats.sharpe_se`
# (Session 5.2). PSR is a Z-score through that standard deviation; MinTRL inverts PSR for `T`.
#
# **Everything here is computed on PER-PERIOD returns and the PER-PERIOD Sharpe ratio.** Skewness
# and kurtosis do not scale simply when returns are aggregated to a lower frequency, so there is no
# valid "annualised PSR formula" — only a per-period one, whose *output* (a probability) we then
# quote next to an annualised Sharpe ratio for readability. Mixing the two up (feeding an annualised
# Sharpe into the per-period formula) is a real bug, not a hypothetical one (see §6.7.4).


def sample_kurt(x: np.ndarray) -> float:
    """Sample kurtosis (population form, RAW — 3.0 for a normal, not excess kurtosis)."""
    x = np.asarray(x, dtype=float).ravel()
    d = x - x.mean()
    return float((d ** 4).mean() / (d ** 2).mean() ** 2)


def psr(sr_hat: float, sr_star: float, T: float, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio: P(true Sharpe > sr_star), all per-period (Bailey & LdP, 2012)."""
    sigma = np.sqrt((1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2) / T)
    return float(norm.cdf((sr_hat - sr_star) / sigma))


def mintrl(sr_hat: float, sr_star: float, skew: float, kurt: float, alpha: float = 0.05) -> float:
    """Minimum track record length (in periods) for PSR(sr_star) = 1 - alpha (Bailey & LdP, 2012)."""
    z = norm.ppf(1.0 - alpha)
    return float((1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2) * (z / (sr_hat - sr_star)) ** 2)


def sr0_expected_max(n_trials: int, sr_sd_p: float) -> float:
    """Expected max per-period Sharpe of n skill-less trials (False Strategy Theorem; Session 1.3)."""
    if n_trials < 2:
        return 0.0
    g = stats.EULER_GAMMA
    z = (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e))
    return float(z * sr_sd_p)


def dsr(sr_hat: float, T: float, skew: float, kurt: float, n_trials: int, sr_sd_p: float) -> float:
    """Deflated Sharpe ratio: PSR evaluated at the expected-max-of-N-trials benchmark (Bailey & LdP, 2014)."""
    return psr(sr_hat, sr0_expected_max(n_trials, sr_sd_p), T, skew, kurt)


# %% [markdown]
# ### Cross-check against Session 5.2's Sharpe-ratio standard error
#
# At zero skew and kurtosis 3 (normal returns), PSR's own implied standard deviation of the Sharpe
# ratio estimator must equal `quantnb.stats.sharpe_se` exactly — it is the same formula, specialised.

# %%
SR_ANNUAL_CHECK, YEARS_CHECK = 0.9, 6.0
T_CHECK = YEARS_CHECK * PPY
sr_p_check = SR_ANNUAL_CHECK / np.sqrt(PPY)
sigma_from_psr = np.sqrt((1.0 - 0.0 * sr_p_check + (3.0 - 1.0) / 4.0 * sr_p_check ** 2) / T_CHECK)
sigma_from_lo = stats.sharpe_se(SR_ANNUAL_CHECK, YEARS_CHECK) / np.sqrt(PPY)  # de-annualise to per-period
lab.record("sigma_psr_normal", float(sigma_from_psr))
lab.record("sigma_lo_se", float(sigma_from_lo))
assert abs(sigma_from_psr - sigma_from_lo) < 1e-12, "PSR's sigma must equal Lo's SE at skew=0, kurt=3"

# %% [markdown]
# ## 2 · One track record, honestly reported
#
# A single strategy, never re-fit, with a true annual Sharpe of 0.6 and **negatively skewed,
# fat-tailed** returns — the shape a real strategy with occasional bad days actually has, built with
# `quantnb.synth.jump_diffusion_returns` (Session 3.2's compensated-crash generator; jumps are down
# moves only, so skewness is negative). We measure its skewness and kurtosis and ask what a rater
# who (wrongly) assumed normal returns would have concluded instead.

# %%
rng_single, rng_normal_cf, rng_mintrl_mc, rng_dsr, rng_edge, rng_pbo = lab.rng.spawn(6)

TRUE_SR_ANNUAL = 0.6
YEARS_SINGLE = 5
T_SINGLE = YEARS_SINGLE * PPY
ANNUAL_VOL = 0.14
# calibrate the annual drift so the COMPENSATED jump-diffusion's annual Sharpe is ~TRUE_SR_ANNUAL
ANNUAL_MU = TRUE_SR_ANNUAL * ANNUAL_VOL

single_ret = synth.jump_diffusion_returns(
    T_SINGLE, n_paths=1, mu=ANNUAL_MU, sigma=ANNUAL_VOL,
    jump_rate=4.0, jump_mean=-0.03, jump_sd=0.02, rng=rng_single,
).ravel()

sr_p_hat = single_ret.mean() / single_ret.std(ddof=1)
sr_annual_hat = sr_p_hat * np.sqrt(PPY)
skew_hat = stats.skewness(single_ret)
kurt_hat = sample_kurt(single_ret)

psr_actual = psr(sr_p_hat, 0.0, T_SINGLE, skew_hat, kurt_hat)
psr_if_normal = psr(sr_p_hat, 0.0, T_SINGLE, 0.0, 3.0)  # the (wrong) textbook-normal assumption

lab.record("years_single", YEARS_SINGLE)
lab.record("sr_annual_hat", float(sr_annual_hat))
lab.record("skew_hat", float(skew_hat))
lab.record("kurt_hat", float(kurt_hat))
lab.record("psr_actual", float(psr_actual))
lab.record("psr_if_normal", float(psr_if_normal))

assert skew_hat < -0.1, "the compensated jump-diffusion should show visibly negative skewness"
assert kurt_hat > 3.5, "the compensated jump-diffusion should show visibly fatter-than-normal tails"
assert psr_if_normal > psr_actual, "ignoring negative skew and fat tails must overstate confidence"

# %% [markdown]
# ### The correction bites harder at low sampling frequency
#
# The gap above is tiny in daily terms: `skew * sr_hat` and `sr_hat**2` are both small when `sr_hat`
# is a PER-DAY Sharpe ratio. Holding this strategy's measured shape fixed (`skew_hat`, `kurt_hat`) but
# asking what an *illustrative* annualised Sharpe of 1.0 would imply if it were instead measured from
# monthly or annual observations — the same 5-year span, sampled more coarsely — shows why Bailey &
# López de Prado (2012) recommend rating a track record at the highest frequency the data allows.

# %%
ILLUSTRATIVE_SR_ANNUAL = 1.0
freq_rows = []
for label, ppy_illustrative, t_illustrative in [("daily", PPY, T_SINGLE), ("monthly", 12, YEARS_SINGLE * 12), ("annual", 1, YEARS_SINGLE)]:
    sr_p_illustrative = ILLUSTRATIVE_SR_ANNUAL / np.sqrt(ppy_illustrative)
    p_act = psr(sr_p_illustrative, 0.0, t_illustrative, skew_hat, kurt_hat)
    p_norm = psr(sr_p_illustrative, 0.0, t_illustrative, 0.0, 3.0)
    freq_rows.append({"freq": label, "T": t_illustrative, "psr_actual": p_act, "psr_normal": p_norm, "gap": p_norm - p_act})
    lab.record(f"freq_psr_actual_{label}", float(p_act))
    lab.record(f"freq_psr_normal_{label}", float(p_norm))
    lab.record(f"freq_gap_{label}", float(p_norm - p_act))
freq_table = pd.DataFrame(freq_rows)
print(freq_table.round(4))

assert freq_table.loc[freq_table["freq"] == "annual", "gap"].iloc[0] > 0.03, \
    "at annual frequency the non-normality correction should be clearly visible"
assert freq_table["gap"].iloc[0] < freq_table["gap"].iloc[1] < freq_table["gap"].iloc[2], \
    "the correction should grow monotonically as sampling frequency coarsens"

# %% [markdown]
# ## 3 · Minimum track record length
#
# MinTRL inverts PSR for `T`: how many periods does a manager need before a claimed Sharpe ratio
# clears a benchmark at 95% confidence? We compare the normal-returns case against this lab's actual
# fat-tailed, negatively skewed returns, and cross-check the normal case against
# `quantnb.stats.years_needed` (Session 5.2), which is the same calculation at `skew=0, kurt=3`.

# %%
ALPHA = 0.05
SR_TARGETS = [0.5, 1.0]

mintrl_rows = []
for sr_target in SR_TARGETS:
    sr_p_t = sr_target / np.sqrt(PPY)
    trl_normal = mintrl(sr_p_t, 0.0, 0.0, 3.0, ALPHA)
    trl_fat = mintrl(sr_p_t, 0.0, skew_hat, kurt_hat, ALPHA)
    mintrl_rows.append({"sr_target": sr_target, "years_normal": trl_normal / PPY, "years_fat": trl_fat / PPY})
    lab.record(f"mintrl_years_normal_{sr_target}", trl_normal / PPY)
    lab.record(f"mintrl_years_fat_{sr_target}", trl_fat / PPY)
mintrl_table = pd.DataFrame(mintrl_rows)
print(mintrl_table.round(2))

# cross-check: MinTRL at skew=0, kurt=3 must equal Session 5.2's years_needed
years_needed_check = stats.years_needed(SR_TARGETS[0], z=norm.ppf(1 - ALPHA))
mintrl_years_check = mintrl(SR_TARGETS[0] / np.sqrt(PPY), 0.0, 0.0, 3.0, ALPHA) / PPY
lab.record("years_needed_check", float(years_needed_check))
lab.record("mintrl_years_check", float(mintrl_years_check))
assert abs(years_needed_check - mintrl_years_check) < 1e-9, "MinTRL at skew=0,kurt=3 must match years_needed"

assert mintrl_table["years_fat"].gt(mintrl_table["years_normal"]).all(), \
    "negative skew and fat tails must always demand a longer track record"

# %% [markdown]
# ### Does MinTRL actually control the false-positive rate? A Monte Carlo check
#
# MinTRL(SR_hat) is a critical value, not a promise about what a skilled manager will show: it is
# built so that a track record of exactly that length turns `PSR(0) = 1 - alpha` at the moment the
# *observed* Sharpe ratio reaches the target. Equivalently — and this is what we can check by
# simulation — a **skill-less** manager (true Sharpe 0) who is measured for `MinTRL(target)` periods
# should show an estimated Sharpe ratio at or above `target` only `alpha` of the time.

# %%
REPS_MINTRL = 20000
sr_target_mc = SR_TARGETS[0]
target_p_mc = sr_target_mc / np.sqrt(PPY)
T_star = int(np.ceil(mintrl(target_p_mc, 0.0, 0.0, 3.0, ALPHA)))
daily_sd_mc = 0.15 / np.sqrt(PPY)
null_tracks = rng_mintrl_mc.standard_normal((REPS_MINTRL, T_star)) * daily_sd_mc  # true Sharpe = 0
sr_p_null = null_tracks.mean(axis=1) / null_tracks.std(axis=1, ddof=1)
false_positive_rate = float((sr_p_null >= target_p_mc).mean())
fpr_se = float(np.sqrt(ALPHA * (1 - ALPHA) / REPS_MINTRL))

lab.record("mintrl_t_star_years", T_star / PPY)
lab.record("mintrl_mc_fpr", false_positive_rate)
lab.record("mintrl_mc_alpha", ALPHA)
assert abs(false_positive_rate - ALPHA) < 4 * fpr_se, "MinTRL's length should cap the false-positive rate at alpha"

# %% [markdown]
# ## 4 · Deflated Sharpe ratio: does it see through a lucky best-of-N?
#
# The Session 1.3 experiment again — N skill-less strategies, keep the best in-sample Sharpe — but
# this time we compute the actual **probabilistic** Sharpe ratio of the winner (naively, as if it
# were the only trial) alongside its **deflated** Sharpe ratio (correctly priced for having tried N).
# A rater who only sees PSR is fooled by N; DSR should not be.

# %%
TRIALS = [1, 10, 50, 200, 1000]
YEARS_DSR = 5
T_DSR = YEARS_DSR * PPY
REPEATS_DSR = 200
STRATEGY_VOL = 0.10  # annual vol of each candidate's daily returns
daily_sd_dsr = STRATEGY_VOL / np.sqrt(PPY)

dsr_rows = []
for n in TRIALS:
    rng_n = np.random.default_rng(rng_dsr.integers(0, 2**63 - 1))
    best_is, psr_naive, dsr_vals, sr0_annual_vals = [], [], [], []
    for _ in range(REPEATS_DSR):
        history = rng_n.standard_normal((T_DSR, n)) * daily_sd_dsr
        sr_p_all = history.mean(axis=0) / history.std(axis=0, ddof=1)
        winner = int(np.argmax(sr_p_all))
        sr_p_win = float(sr_p_all[winner])
        skew_win = stats.skewness(history[:, winner])
        kurt_win = sample_kurt(history[:, winner])
        sr_sd_p = float(sr_p_all.std(ddof=1)) if n > 1 else 1.0 / np.sqrt(T_DSR)

        best_is.append(sr_p_win * np.sqrt(PPY))
        psr_naive.append(psr(sr_p_win, 0.0, T_DSR, skew_win, kurt_win))
        dsr_vals.append(dsr(sr_p_win, T_DSR, skew_win, kurt_win, n, sr_sd_p))
        sr0_annual_vals.append(sr0_expected_max(n, sr_sd_p) * np.sqrt(PPY))
    dsr_rows.append({
        "N": n, "best_is_annual": np.mean(best_is), "psr_naive": np.mean(psr_naive),
        "dsr": np.mean(dsr_vals), "sr0_annual": np.mean(sr0_annual_vals),
        "dsr_se": stats.batch_se(lambda a: a.mean(), np.array(dsr_vals)),
        "psr_se": stats.batch_se(lambda a: a.mean(), np.array(psr_naive)),
    })
dsr_table = pd.DataFrame(dsr_rows).set_index("N")
print(dsr_table.round(3))

for n, row in dsr_table.iterrows():
    lab.record(f"best_is_annual_{n}", float(row["best_is_annual"]))
    lab.record(f"psr_naive_{n}", float(row["psr_naive"]))
    lab.record(f"dsr_{n}", float(row["dsr"]))
    lab.record(f"sr0_annual_{n}", float(row["sr0_annual"]))
lab.record("years_dsr", YEARS_DSR)
lab.record("repeats_dsr", REPEATS_DSR)

# naive PSR is fooled by N; DSR is not (stays a coin flip, at 4 pooled standard errors)
assert dsr_table.loc[1, "psr_naive"] < 0.65, "with N=1, naive PSR should already be unremarkable"
assert dsr_table.loc[1000, "psr_naive"] > 0.9, "naive PSR must be fooled by 1,000 skill-less trials"
for n in TRIALS:
    row = dsr_table.loc[n]
    assert abs(row["dsr"] - 0.5) < 4 * row["dsr_se"], f"DSR at N={n} should stay near 0.5 (no skill)"

# %% [markdown]
# ## 5 · The same statistic, kept honest: a real edge survives; a fake one doesn't
#
# One strategy with a **genuine** annual Sharpe of 0.7, tested honestly among a modest, pre-declared
# 5 variants — versus the same track record reported (dishonestly) as the best of 1,000. DSR is not a
# blanket penalty: the number of trials you actually ran is the whole story.

# %%
REAL_SR_ANNUAL = 0.9
N_HONEST, N_DISHONEST = 5, 1000
REPEATS_EDGE = 300
daily_sd_edge = 0.15 / np.sqrt(PPY)
daily_mu_edge = REAL_SR_ANNUAL * 0.15 / PPY

dsr_honest, dsr_dishonest = [], []
for _ in range(REPEATS_EDGE):
    ret = rng_edge.standard_normal(T_DSR) * daily_sd_edge + daily_mu_edge
    sr_p = ret.mean() / ret.std(ddof=1)
    skew_e, kurt_e = stats.skewness(ret), sample_kurt(ret)
    sr_sd_p_edge = 1.0 / np.sqrt(T_DSR)  # a single genuine strategy: use the theoretical null SD
    dsr_honest.append(dsr(sr_p, T_DSR, skew_e, kurt_e, N_HONEST, sr_sd_p_edge))
    dsr_dishonest.append(dsr(sr_p, T_DSR, skew_e, kurt_e, N_DISHONEST, sr_sd_p_edge))

dsr_honest_mean, dsr_dishonest_mean = float(np.mean(dsr_honest)), float(np.mean(dsr_dishonest))
lab.record("real_sr_annual", REAL_SR_ANNUAL)
lab.record("n_honest", N_HONEST)
lab.record("n_dishonest", N_DISHONEST)
lab.record("dsr_honest_mean", dsr_honest_mean)
lab.record("dsr_dishonest_mean", dsr_dishonest_mean)
assert dsr_honest_mean > dsr_dishonest_mean + 0.3, "reporting the true trial count must matter a lot"
assert dsr_honest_mean > 0.65, "a genuine SR=0.9 edge over 5 years should mostly survive 5 honest trials"
assert dsr_dishonest_mean < 0.35, "the same track record reported as best-of-1,000 should look unconvincing"

# %% [markdown]
# ## 6 · Probability of backtest overfitting, via CSCV
#
# DSR needs to know N, the number of *independent* trials. Often you instead have many
# parameterisations of ONE rule fit to ONE dataset — the classic parameter sweep. **Combinatorially
# symmetric cross-validation** (Bailey, Borwein, López de Prado & Zhu, 2016) scores that case: split
# the sample into `S` contiguous blocks, form every way of splitting the blocks into two equal
# in-sample/out-of-sample halves, and ask how often the in-sample winner is merely average — or
# worse — out of sample.
#
# One simulated price series with **no true edge** (`quantnb.synth.gbm_prices`, zero drift), and a
# grid of moving-average crossover rules (fast/slow lookbacks) swept over it — the parameter sweep
# that produces genuine, mechanical overfitting, not just noisy trial-to-trial luck.

# %%
N_DAYS_PBO = 252 * 10
S_BLOCKS = 10
FAST = [5, 10, 15, 20, 30]
SLOW = [50, 75, 100, 150, 200]
WARMUP = max(SLOW)  # skip the days before the slowest moving average exists

price = synth.gbm_prices(N_DAYS_PBO, mu=0.0, sigma=0.16, rng=rng_pbo)["price"]
price_arr = price.to_numpy()                        # length N_DAYS_PBO + 1
simple_ret_full = price.pct_change().iloc[1:].to_numpy()  # length N_DAYS_PBO; simple_ret_full[i] = return (i -> i+1)

remaining = N_DAYS_PBO - WARMUP
n_use = (remaining // S_BLOCKS) * S_BLOCKS  # trim to a multiple of S_BLOCKS after the warm-up
BLOCK_LEN = n_use // S_BLOCKS

rule_returns, rule_names = [], []
for f in FAST:
    ma_f = pd.Series(price_arr).rolling(f).mean().to_numpy()
    for s in SLOW:
        if f >= s:
            continue
        ma_s = pd.Series(price_arr).rolling(s).mean().to_numpy()
        # position[i] uses prices up to and including day i (flat, 0, during the warm-up NaNs);
        # it earns simple_ret_full[i], the return from day i to day i+1 — no look-ahead (cf. 6.2).
        position = np.nan_to_num(np.sign(ma_f - ma_s), nan=0.0)[:-1]
        r_full = position * simple_ret_full
        rule_returns.append(r_full[WARMUP:WARMUP + n_use])
        rule_names.append(f"{f}/{s}")
R = np.column_stack(rule_returns)  # (n_use, n_rules)
N_RULES = R.shape[1]
lab.record("n_rules_pbo", N_RULES)
lab.record("s_blocks_pbo", S_BLOCKS)

# %% [markdown]
# ### The CSCV loop
#
# Precompute each rule's sum and sum-of-squares per block so every combination is just an add.

# %%
from itertools import combinations

blocks = R[: S_BLOCKS * BLOCK_LEN].reshape(S_BLOCKS, BLOCK_LEN, N_RULES)
block_sum = blocks.sum(axis=1)             # (S_BLOCKS, N_RULES)
block_sumsq = (blocks ** 2).sum(axis=1)    # (S_BLOCKS, N_RULES)
all_blocks = set(range(S_BLOCKS))


def block_sharpe(idx: list[int]) -> np.ndarray:
    n = len(idx) * BLOCK_LEN
    s = block_sum[idx].sum(axis=0)
    ss = block_sumsq[idx].sum(axis=0)
    mean = s / n
    var = (ss / n - mean ** 2) * n / (n - 1)  # ddof=1
    return mean / np.sqrt(np.maximum(var, 1e-18))


combos = list(combinations(range(S_BLOCKS), S_BLOCKS // 2))
logits = []
for is_idx in combos:
    oos_idx = sorted(all_blocks - set(is_idx))
    is_sharpe = block_sharpe(list(is_idx))
    oos_sharpe = block_sharpe(oos_idx)
    winner = int(np.argmax(is_sharpe))
    rank = 1 + int((oos_sharpe < oos_sharpe[winner]).sum())  # 1 = worst OOS performer
    omega = rank / (N_RULES + 1)
    logits.append(np.log(omega / (1 - omega)))
logits = np.array(logits)

pbo = float((logits <= 0).mean())
n_combos = len(combos)
lab.record("n_combos_pbo", n_combos)
lab.record("pbo", pbo)
lab.record("pbo_logit_mean", float(logits.mean()))

# a random (unfit) rule should do about as well OOS as the average rule, in expectation, so PBO for
# a real parameter sweep on pure noise should sit well above 0 (mechanical overfitting is real) but
# need not approach 1 (five parameters is a mild sweep, not a thousand-feature model)
assert 0.30 < pbo < 0.95, "a real parameter sweep on pure noise should show meaningful, not total, PBO"

lab.chart("pbo_logits", charts.histogram(
    logits, title=f"CSCV logits across {n_combos} in-sample/out-of-sample splits",
    bins=30, x_fmt=charts.fmt_num(1), tail_below=0.0, tail_label="IS winner ≤ OOS median"))

# %% [markdown]
# ## 7 · Charts for the session page

# %%
lab.chart("psr_vs_dsr", charts.grouped_column_chart(
    [f"N = {n:,}" for n in TRIALS],
    [("naive PSR (ignores N)", dsr_table["psr_naive"].tolist(), "loss"),
     ("deflated Sharpe ratio", dsr_table["dsr"].tolist(), "strategy")],
    title="Naive PSR is fooled by N; DSR is not (5 years, 200 repeats)",
    y_fmt=charts.fmt_num(2), value_labels=True, y_min=0.0, y_max=1.05))

lab.chart("real_vs_fake_dsr", charts.column_chart(
    ["real edge, 5 trials", "real edge, as best-of-1,000", "no edge, best-of-1,000"],
    [dsr_honest_mean, dsr_dishonest_mean, dsr_table.loc[1000, "dsr"]],
    title="Same SR 0.9 track record, different trial count: DSR (300 repeats)",
    y_fmt=charts.fmt_num(2), roles=["strategy", "loss", "benchmark"], y_min=0.0, y_max=1.0))

# %%
lab.save()
