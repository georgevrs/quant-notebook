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
# # 6.8 · Lab: The Research Protocol — companion lab
#
# **Quant Notebook** · Unit 6 · Session 8 · [Read the session](https://georgevrs.github.io/quant-notebook/unit06-backtesting-discipline/session08-lab-research-protocol.html)
#
# Pre-register a hypothesis, log every trial, sweep parameters, score them with PBO and DSR, and
# write the go/no-go memo. This is Unit 6's capstone: it assembles Sessions 6.1–6.7 into one small,
# real, end-to-end research protocol — pre-registration (1.3) → parameter sweep (vectorbt) → trial
# log (MLflow) → overfitting statistics (6.7's PBO and DSR) → a decision, on the record.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# **Risk-free rate.** Every return below is a synthetic *excess* return; the risk-free rate is
# taken as 0 throughout, so "return" and "excess return" mean the same thing in this lab.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import tempfile
from itertools import combinations

import mlflow
import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.stats import norm

import quantnb as qn
from quantnb import charts, stats

lab = qn.Lab("6.8")  # seeds the random generators: every run gives the same numbers
PPY = 252  # trading days per year, this course's convention (AUTHORING.md §9)

# %% [markdown]
# ## 1 · The pre-registration and the kill criterion
#
# <a data-xref="1.3"></a> owns **pre-registration** and the **kill criterion**: write the hypothesis,
# the test and the number that would kill the idea before touching the data. This lab turns that
# discipline into three durable artefacts and wires them together: the pre-registration document
# below, a trial log that MLflow keeps for us (§3), and the go/no-go memo it produces (§6).
#
# The kill criterion is a pure function of three numbers this lab computes honestly: the **deflated
# Sharpe ratio** and the **probability of backtest overfitting** (both <a data-xref="6.7"></a>'s,
# reused here, not re-derived) and a fresh holdout Sharpe ratio, evaluated exactly once. All three
# gates must pass, or the idea is killed — a single flattering number never overrides the other two.

# %%
PREREG = {
    "id": "R-014",
    "date": "2026-09-27",
    "hypothesis": "A small sleeve of synthetic instruments carries positive short-lag return "
                  "autocorrelation; a moving-average crossover can harvest it net of a flat cost.",
    "universe": "40 synthetic instruments, daily bars, equal-weighted sleeve",
    "data": "20 years in-sample (the sweep never sees more) + 8 years frozen holdout",
    "signal": "close vs (fast, slow) SMA crossover, long/flat, one grid, no re-fitting",
    "primary_metric": "annualised Sharpe ratio of the equal-weighted sleeve, 0 risk-free rate",
    "variants_allowed": 45,  # every (fast, slow) pair from a 10-window ladder, C(10,2)
    "kill_criterion": "GO only if DSR >= 0.95 AND PBO < 0.50 AND holdout Sharpe > 0 — all three",
    "trial_log": "every one of the 45 variants logged to MLflow before the holdout is touched",
}
for k, v in PREREG.items():
    lab.record(f"prereg_{k}" if k != "variants_allowed" else "variants_allowed", v)
print(PREREG)


def go_no_go(dsr_value: float, pbo_value: float, holdout_sharpe: float,
             dsr_bar: float = 0.95, pbo_bar: float = 0.50) -> tuple[str, str]:
    """The pre-registered kill criterion, applied mechanically. Every gate must pass."""
    checks = {
        f"DSR at least {dsr_bar:.2f}": dsr_value >= dsr_bar,
        f"PBO below {pbo_bar:.2f}": pbo_value < pbo_bar,
        "holdout Sharpe positive": holdout_sharpe > 0,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return ("GO", "all three pre-registered gates passed") if not failed else \
           ("NO-GO", "failed " + ", ".join(failed))


DSR_BAR, PBO_BAR = 0.95, 0.50
lab.record("dsr_bar", DSR_BAR)
lab.record("pbo_bar", PBO_BAR)

# a handful of deterministic checks on the decision LOGIC itself — no randomness needed here
assert go_no_go(0.99, 0.10, 0.5) == ("GO", "all three pre-registered gates passed")
assert go_no_go(0.80, 0.10, 0.5)[0] == "NO-GO", "a DSR below the bar must kill it alone"
assert go_no_go(0.99, 0.70, 0.5)[0] == "NO-GO", "a PBO above the bar must kill it alone"
assert go_no_go(0.99, 0.10, -0.1)[0] == "NO-GO", "a negative holdout Sharpe must kill it alone"

# %% [markdown]
# ## 2 · Does the kill criterion actually work? A cheap Monte Carlo check
#
# Before trusting the three-gate rule on real trials, check it the way any classifier should be
# checked: how often does it say GO when there is truly no edge (a false positive), and how often
# does it say NO-GO when there truly is one (a false negative)? This is a **stylised** proxy — 45
# independent Gaussian "trials" and a fresh Gaussian holdout draw, not a full price simulation — used
# only to calibrate the rule cheaply before spending the compute on the real pipeline in §3–§6.

# %%
rng_null, rng_edge, rng_momentum, rng_noise = lab.rng.spawn(4)

CAL_REPEATS = 400
N_RULES = 45  # must match the real grid in §3: C(10, 2)
ANN_VOL = 0.16
DAILY_SD = ANN_VOL / np.sqrt(PPY)
T_HOLD_PROXY = 500


def sample_kurtosis(x: np.ndarray) -> float:
    """Sample kurtosis (population form, RAW — 3.0 for a normal), same convention as 6.7."""
    x = np.asarray(x, dtype=float).ravel()
    d = x - x.mean()
    return float((d ** 4).mean() / (d ** 2).mean() ** 2)


def psr(sr_hat: float, sr_star: float, T: float, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio, per-period (Bailey & Lopez de Prado, 2012); see 6.7 for the derivation."""
    sigma = np.sqrt((1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat ** 2) / T)
    return float(norm.cdf((sr_hat - sr_star) / sigma))


def dsr(sr_hat: float, T: float, skew: float, kurt: float, n_trials: int, sr_sd_p: float) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): PSR against the expected best-of-N benchmark."""
    sr0 = stats.expected_max_sharpe(n_trials, years=1.0, sr_sd=sr_sd_p) if n_trials >= 2 else 0.0
    return psr(sr_hat, sr0, T, skew, kurt)


def calibration_go_rate(sr_annual_true: float, t_periods: int, rng: np.random.Generator) -> float:
    """Fraction of CAL_REPEATS synthetic sweeps that clear DSR + holdout at the pre-registered bar."""
    mu = sr_annual_true * DAILY_SD / np.sqrt(PPY)
    go = 0
    for _ in range(CAL_REPEATS):
        ret = rng.normal(mu, DAILY_SD, (t_periods, N_RULES))
        sr_hat = ret.mean(axis=0) / ret.std(axis=0, ddof=1)
        winner = int(np.argmax(sr_hat))
        skew_w = stats.skewness(ret[:, winner])
        kurt_w = sample_kurtosis(ret[:, winner])
        sr_sd_p = float(sr_hat.std(ddof=1))
        d = dsr(sr_hat[winner], t_periods, skew_w, kurt_w, N_RULES, sr_sd_p)
        holdout = rng.normal(mu, DAILY_SD, T_HOLD_PROXY)
        holdout_sr = holdout.mean() / holdout.std(ddof=1)
        if d >= DSR_BAR and holdout_sr > 0:  # PBO needs the real block structure — checked in §5
            go += 1
    return go / CAL_REPEATS


null_go_rate = calibration_go_rate(0.0, 8 * PPY, rng_null)
lab.record("cal_null_go_rate", null_go_rate)
lab.record("cal_repeats", CAL_REPEATS)
assert null_go_rate <= 0.05, "under pure noise the DSR+holdout gate should almost never say go"

EDGE_SR_ANNUAL = 0.9
lab.record("cal_edge_sr", EDGE_SR_ANNUAL)
edge_rates = {}
for years in (3, 8, 20):
    edge_rates[years] = calibration_go_rate(EDGE_SR_ANNUAL, years * PPY, rng_edge)
    lab.record(f"cal_edge_go_rate_{years}y", edge_rates[years])
print("edge go rates by history length:", edge_rates)

assert edge_rates[3] < 0.5, "three years is not enough to reliably clear the bar, even for a real edge"
assert edge_rates[20] > 0.7, "twenty years should reliably clear the bar for a genuine SR~0.9 edge"
assert edge_rates[8] > edge_rates[3], "more history should only make a real edge easier to confirm"

# %% [markdown]
# ## 3 · Two research projects, one pipeline
#
# Two synthetic sleeves of 40 instruments each, built the way <a data-xref="6.4"></a> builds every
# exhibit: we author the truth, then run the honest process and see if it finds it. **Momentum
# pilot** has a real (and, for teaching, exaggerated) daily return autocorrelation; **noise null**
# has none. Both go through the *identical* pipeline below — nobody hand-tunes anything per project.

# %%
N_ASSETS = 40
T_IS_YEARS, T_HOLD_YEARS = 20, 8
T_IS, T_HOLD = T_IS_YEARS * PPY, T_HOLD_YEARS * PPY
PHI_MOMENTUM = 0.25
FEE = 0.0005  # a flat placeholder cost so turnover isn't free; 6.5 owns the full cost model
WINDOWS = [5, 8, 12, 18, 26, 40, 60, 90, 130, 180]  # C(10, 2) = 45 (fast, slow) pairs
S_BLOCKS = 8

lab.record("n_assets", N_ASSETS)
lab.record("t_is_years", T_IS_YEARS)
lab.record("t_hold_years", T_HOLD_YEARS)
lab.record("phi_momentum", PHI_MOMENTUM)
lab.record("fee_bp", FEE * 10_000)
lab.record("n_windows", len(WINDOWS))


def ar1_panel(t_periods: int, n_assets: int, phi: float, ann_vol: float,
              rng: np.random.Generator) -> pd.DataFrame:
    """A panel of independent AR(1) return processes, turned into prices.

    Each instrument's daily return is r_t = phi * r_(t-1) + eps_t: phi = 0 is a fair-game random
    walk (Session 2.1); phi > 0 is a KNOWN, exaggerated short-lag momentum edge, built in so we can
    test whether the protocol below finds it. Every asset is an independent draw of the same rng.
    """
    daily_sd = ann_vol / np.sqrt(PPY)
    sigma_eps = daily_sd * np.sqrt(1 - phi ** 2)
    eps = rng.normal(0.0, sigma_eps, (t_periods, n_assets))
    r = np.zeros((t_periods, n_assets))
    for t in range(1, t_periods):
        r[t] = phi * r[t - 1] + eps[t]
    price = 100 * np.exp(np.cumsum(r, axis=0))
    return pd.DataFrame(price, columns=[f"A{i}" for i in range(n_assets)])


price_momentum = ar1_panel(T_IS + T_HOLD, N_ASSETS, PHI_MOMENTUM, ANN_VOL, rng_momentum)
price_noise = ar1_panel(T_IS + T_HOLD, N_ASSETS, 0.0, ANN_VOL, rng_noise)

# know the truth we built in: the momentum panel's sample autocorrelation should sit near phi
mom_ret = np.diff(np.log(price_momentum.to_numpy()), axis=0)
r1, r2 = mom_ret[:-1].ravel(), mom_ret[1:].ravel()
phi_hat = float(np.corrcoef(r1, r2)[0, 1])
n_obs = r1.size
phi_se = 1.0 / np.sqrt(n_obs)  # standard error of a sample autocorrelation (large n)
lab.record("phi_hat", phi_hat)
lab.record("phi_se", phi_se)
assert abs(phi_hat - PHI_MOMENTUM) < 4 * phi_se, "the panel's measured autocorrelation should match phi"

# %% [markdown]
# ## 4 · The parameter sweep (vectorbt) and the trial log (MLflow)
#
# `vectorbt` runs every (fast, slow) crossover on every one of the 40 instruments in one call —
# broadcasting is its whole idea, and the reason a Commons-Clause-licensed library is worth reading
# the license on (never sell a product built on it). We average the 40 instruments' returns into one
# equal-weighted sleeve return per (fast, slow) pair, then log **every one of the 45 pairs** to
# MLflow — parameters, metrics, nothing skipped — before looking at which one won.

# %%
mlflow_dir = tempfile.mkdtemp(prefix="qn_6_8_mlflow_")  # ephemeral: never part of the repo
mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir}/mlflow.db")


def sweep_and_log(price: pd.DataFrame, project: str) -> pd.DataFrame:
    """Run the full-period crossover sweep, log every trial to MLflow, return the panel returns."""
    fast_ma, slow_ma = vbt.MA.run_combs(price, window=WINDOWS, r=2, short_names=["fast", "slow"])
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=10_000, fees=FEE, freq="D")
    ret_per_asset = pf.returns()
    panel = ret_per_asset.T.groupby(level=["fast_window", "slow_window"]).mean().T  # 40 assets -> 1 sleeve
    trades_per_combo = pf.trades.count().groupby(level=["fast_window", "slow_window"]).sum()

    mlflow.set_experiment(f"qn-6.8-{project}")
    is_ret = panel.iloc[:T_IS].to_numpy()
    sr_is = stats.annual_sharpe(is_ret, axis=0)
    for j, (fast, slow) in enumerate(panel.columns):
        with mlflow.start_run(run_name=f"{project}_f{fast}_s{slow}"):
            mlflow.log_params({"project": project, "fast": int(fast), "slow": int(slow)})
            mlflow.log_metrics({
                "sharpe_in_sample": float(sr_is[j]),
                "n_trades_sleeve": float(trades_per_combo[(fast, slow)]),
            })
    return panel


panel_momentum = sweep_and_log(price_momentum, "momentum")
panel_noise = sweep_and_log(price_noise, "noise")

# %% [markdown]
# ## 5 · Reading the trial log back, scoring with PBO and DSR
#
# The sweep table is not kept in a local variable — it is **read back from MLflow**, the way a
# teammate reopening this research six months later would. `N`, the trial count DSR needs, is
# `len(runs)`: not a guess, the row count of the log itself.

# %%
CSCV_ALL_BLOCKS = set(range(S_BLOCKS))


def cscv_pbo(returns: np.ndarray, s_blocks: int = S_BLOCKS) -> float:
    """Probability of backtest overfitting via CSCV (Bailey, Borwein, Lopez de Prado & Zhu, 2017,
    Journal of Computational Finance) — same method as 6.7 §6, applied here to the logged sleeve
    returns instead of a hand-rolled loop. Split into s_blocks contiguous blocks; for every
    equal-sized in/out split, the in-sample
    winner's out-of-sample rank sets one CSCV logit; PBO is the share of logits at or below zero
    (the in-sample winner did no better than the OOS median)."""
    n_use = (returns.shape[0] // s_blocks) * s_blocks
    r = returns[:n_use]
    block_len = n_use // s_blocks
    n_rules = r.shape[1]
    blocks = r.reshape(s_blocks, block_len, n_rules)
    block_sum, block_sumsq = blocks.sum(axis=1), (blocks ** 2).sum(axis=1)

    def block_sharpe(idx: list[int]) -> np.ndarray:
        n = len(idx) * block_len
        s, ss = block_sum[idx].sum(axis=0), block_sumsq[idx].sum(axis=0)
        mean = s / n
        var = (ss / n - mean ** 2) * n / (n - 1)
        return mean / np.sqrt(np.maximum(var, 1e-18))

    logits = []
    for is_idx in combinations(range(s_blocks), s_blocks // 2):
        oos_idx = sorted(CSCV_ALL_BLOCKS - set(is_idx))
        is_sharpe, oos_sharpe = block_sharpe(list(is_idx)), block_sharpe(oos_idx)
        winner = int(np.argmax(is_sharpe))
        rank = 1 + int((oos_sharpe < oos_sharpe[winner]).sum())
        omega = rank / (n_rules + 1)
        logits.append(np.log(omega / (1 - omega)))
    return float((np.array(logits) <= 0).mean())


def score_project(project: str, panel: pd.DataFrame) -> dict:
    runs = mlflow.search_runs(experiment_names=[f"qn-6.8-{project}"])
    n_trials = len(runs)
    winner_row = runs.loc[runs["metrics.sharpe_in_sample"].astype(float).idxmax()]
    fast_w, slow_w = int(winner_row["params.fast"]), int(winner_row["params.slow"])
    n_trades_sleeve = float(winner_row["metrics.n_trades_sleeve"])

    is_ret = panel.iloc[:T_IS].to_numpy()
    sr_is_logged = float(winner_row["metrics.sharpe_in_sample"])
    winner_col = list(panel.columns).index((fast_w, slow_w))
    sr_is_direct = float(stats.annual_sharpe(is_ret[:, winner_col]))
    assert abs(sr_is_logged - sr_is_direct) < 1e-9, "the trial log must reproduce the direct computation"

    sr_p_all = is_ret.mean(axis=0) / is_ret.std(axis=0, ddof=1)
    skew_w = stats.skewness(is_ret[:, winner_col])
    kurt_w = sample_kurtosis(is_ret[:, winner_col])
    sr_sd_p = float(sr_p_all.std(ddof=1))
    dsr_value = dsr(sr_p_all[winner_col], T_IS, skew_w, kurt_w, n_trials, sr_sd_p)
    pbo_value = cscv_pbo(is_ret)

    holdout_ret = panel[(fast_w, slow_w)].iloc[T_IS:].to_numpy()
    holdout_sharpe = float(stats.annual_sharpe(holdout_ret))

    decision, reason = go_no_go(dsr_value, pbo_value, holdout_sharpe)
    return {
        "project": project, "n_trials": n_trials, "fast": fast_w, "slow": slow_w,
        "best_is_sharpe": sr_is_direct, "dsr": dsr_value, "pbo": pbo_value,
        "holdout_sharpe": holdout_sharpe, "decision": decision, "reason": reason,
        "n_trades_sleeve": n_trades_sleeve,
    }


result_momentum = score_project("momentum", panel_momentum)
result_noise = score_project("noise", panel_noise)
for r in (result_momentum, result_noise):
    print(r)
    for k, v in r.items():
        if k != "project":
            lab.record(f"proj_{r['project']}_{k}", v)

assert result_momentum["n_trials"] == len(WINDOWS) * (len(WINDOWS) - 1) // 2 == 45
assert result_noise["n_trials"] == 45
assert result_momentum["dsr"] > result_noise["dsr"], "the real edge should score higher than noise"
assert result_momentum["decision"] == "GO", "the exaggerated real edge, over 20 years, should clear the bar"
assert result_noise["decision"] == "NO-GO", "pure noise, swept 45 ways, should not clear the bar"

# %% [markdown]
# ## 6 · The go/no-go memo
#
# The memo is nothing more than the pre-registration, the trial-log summary and the decision,
# assembled without editorial licence — every field below is `PREREG`, `result_momentum` or
# `result_noise`, not a fresh judgement call.

# %%
memo_momentum = {**PREREG, **result_momentum}
memo_noise = {**PREREG, **result_noise}
lab.record("memo_momentum_decision", memo_momentum["decision"])
lab.record("memo_noise_decision", memo_noise["decision"])
print("MOMENTUM PILOT MEMO:", memo_momentum["decision"], memo_momentum["reason"])
print("NOISE NULL MEMO:", memo_noise["decision"], memo_noise["reason"])

# %% [markdown]
# ## 7 · Charts for the session page

# %%
lab.chart("edge_go_rate", charts.column_chart(
    ["3 years", "8 years", "20 years"], [edge_rates[3], edge_rates[8], edge_rates[20]],
    title=f"How much history a real SR={EDGE_SR_ANNUAL} edge needs to clear the bar (45 trials, {CAL_REPEATS} repeats)",
    y_fmt=charts.fmt_pct(0), roles=["loss", "strategy", "strategy"], y_min=0.0, y_max=1.0))

lab.chart("honesty_gap", charts.grouped_column_chart(
    ["Momentum pilot", "Noise null"],
    [("best in-sample Sharpe", [result_momentum["best_is_sharpe"], result_noise["best_is_sharpe"]], "loss"),
     ("holdout Sharpe (never touched until now)", [result_momentum["holdout_sharpe"], result_noise["holdout_sharpe"]], "strategy")],
    title="The winning combo's in-sample Sharpe vs. its own frozen holdout",
    y_fmt=charts.fmt_num(2), value_labels=True))

lab.chart("dsr_pbo", charts.grouped_column_chart(
    ["Momentum pilot", "Noise null"],
    [("deflated Sharpe ratio", [result_momentum["dsr"], result_noise["dsr"]], "strategy"),
     ("probability of backtest overfitting", [result_momentum["pbo"], result_noise["pbo"]], "loss")],
    title="The two probability-scale verdicts the kill criterion actually reads",
    y_fmt=charts.fmt_num(2), value_labels=True, y_min=0.0, y_max=1.05))

# %%
lab.save()
