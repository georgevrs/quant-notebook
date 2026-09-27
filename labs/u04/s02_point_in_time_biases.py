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
# # 4.2 · Point-in-Time Data & Data Biases — companion lab
#
# **Quant Notebook** · Unit 4 · Session 2 · [Read the session](https://georgevrs.github.io/quant-notebook/unit04-data-engineering/session02-point-in-time-biases.html)
#
# Look-ahead, survivorship, restatements, vintages and delistings — the data biases that manufacture fake alpha, and how to engineer them out.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Risk-free rate: **0** throughout. Every experiment here compares two ways of *measuring* the
# same synthetic world, so the bias is visible without needing an excess-return benchmark.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

import quantnb as qn
from quantnb import charts
from quantnb.returns import max_drawdown_paths

lab = qn.Lab("4.2")  # seeds the random generators: every run gives the same numbers
# One independent random stream per experiment, so changing one section never moves another.
rng_surv, rng_factor, rng_backfill = lab.rng.spawn(3)

# %% [markdown]
# ## 1 · One fact, two dates — a toy point-in-time table (DuckDB)
#
# Every reported number has a **reference date** (the period it describes) and a **knowledge
# date** (the date it became public). A vendor's "current" table silently overwrites history with
# the latest **restatement** every time you re-query it; a **point-in-time data** table keeps every
# vintage and lets you ask "what did this field say, as of date X?"

# %%
con = duckdb.connect()
con.execute("""
    CREATE TABLE fundamentals_vintages (
        ticker VARCHAR, ref_quarter VARCHAR, knowledge_date DATE, value DOUBLE, is_restatement BOOLEAN
    )
""")
con.execute("""
    INSERT INTO fundamentals_vintages VALUES
        ('SYNTH', '2024Q2', DATE '2024-08-09', 0.0410, FALSE),   -- as first filed, ~40 days after quarter end
        ('SYNTH', '2024Q2', DATE '2025-11-12', 0.0270, TRUE)     -- restated ~15 months later, after audit
""")
ASOF = "2024-09-01"  # a researcher building a signal shortly after the original filing

naive_value = con.execute("""
    SELECT value FROM fundamentals_vintages
    WHERE ticker = 'SYNTH' AND ref_quarter = '2024Q2'
    ORDER BY knowledge_date DESC LIMIT 1
""").fetchone()[0]  # "give me the latest value" — silently the restated one, from 15 months in the future

pit_value = con.execute("""
    SELECT value FROM fundamentals_vintages
    WHERE ticker = 'SYNTH' AND ref_quarter = '2024Q2' AND knowledge_date <= ?
    ORDER BY knowledge_date DESC LIMIT 1
""", [ASOF]).fetchone()[0]  # "give me the latest value KNOWN as of ASOF" — the original filing

print(f"naive query (no as-of filter): {naive_value:.2%}  ·  point-in-time query (as of {ASOF}): {pit_value:.2%}")
lab.record("toy_naive_query_value", naive_value)
lab.record("toy_pit_query_value", pit_value)
assert naive_value != pit_value, "the toy table should show the restatement changed the number"
assert np.isclose(pit_value, 0.0410) and np.isclose(naive_value, 0.0270)

# %% [markdown]
# ## 2 · Survivorship bias: a look-ahead-biased vs. point-in-time-correct equity backtest
#
# `N_STOCKS` synthetic companies over `MONTHS` months. Each has its own drift, so some are
# genuinely bad businesses. A company's monthly hazard of **delisting** rises with how badly it
# has done over the trailing year — distress precedes delisting, exactly as in real markets. On
# delisting we draw a **delisting return**: mostly a harsh, distress-flavoured loss, occasionally
# a takeover premium.
#
# We then build the SAME equal-weight portfolio two ways:
#  * **point-in-time correct** — each month, equal-weight every name still listed that month
#    (including the harsh return in its delisting month), dropping it afterwards;
#  * **survivorship-filtered (look-ahead)** — take only the names still listed at the END of the
#    run and backtest their full history "since inception", exactly what "download today's index
#    members and backtest since inception" does.

# %%
N_STOCKS, N_REPS, MONTHS = 150, 200, 180          # 15 simulated years, 200 independent universes
MU_MEAN, MU_SD, SIGMA = 0.07, 0.12, 0.35          # annual drift (cross-sectional), annual vol
BASE_HAZARD, SENSITIVITY, MAX_HAZARD = 0.0025, 0.035, 0.08   # monthly delisting hazard
P_BAD_DELIST = 0.75                                # share of delistings that are distress, not a buyout


def simulate_equity_universe(rng, n_stocks, n_reps, months):
    """Vectorised across (stocks, reps). Returns monthly simple-return panels (months, reps)."""
    dt = 1.0 / 12
    mu = rng.normal(MU_MEAN, MU_SD, (n_stocks, n_reps))
    z = rng.standard_normal((months, n_stocks, n_reps))
    base_r = np.expm1((mu - 0.5 * SIGMA ** 2) * dt + SIGMA * np.sqrt(dt) * z)  # "would-be" monthly returns

    alive = np.ones((n_stocks, n_reps), dtype=bool)
    realized = np.full((months, n_stocks, n_reps), np.nan)
    for m in range(months):
        realized[m] = np.where(alive, base_r[m], np.nan)
        if m >= 11:
            trailing12 = np.nanprod(1.0 + realized[m - 11:m + 1], axis=0) - 1.0
            hazard = np.clip(BASE_HAZARD + SENSITIVITY * np.maximum(0.0, -trailing12), 0.0, MAX_HAZARD)
            delist = alive & (rng.random((n_stocks, n_reps)) < hazard)
            bad = delist & (rng.random((n_stocks, n_reps)) < P_BAD_DELIST)
            good = delist & ~bad
            bad_ret = np.clip(rng.normal(-0.35, 0.15, (n_stocks, n_reps)), -0.95, -0.02)
            good_ret = np.clip(rng.normal(0.18, 0.08, (n_stocks, n_reps)), 0.0, 0.5)
            realized[m] = np.where(bad, bad_ret, realized[m])
            realized[m] = np.where(good, good_ret, realized[m])
            alive = alive & ~delist

    survivors = alive  # never delisted over the whole horizon
    pit_return = np.nanmean(realized, axis=1)                                   # (months, reps)
    surv_return = np.where(survivors[None, :, :], base_r, np.nan)
    surv_return = np.nanmean(surv_return, axis=1)                               # (months, reps)
    return pit_return, surv_return, survivors


pit_ret, surv_ret, survivors = simulate_equity_universe(rng_surv, N_STOCKS, N_REPS, MONTHS)
survival_rate = float(survivors.mean())
print(f"average fraction of names still listed after {MONTHS / 12:.0f} years: {survival_rate:.1%}")
lab.record("surv_n_stocks", N_STOCKS)
lab.record("surv_n_reps", N_REPS)
lab.record("surv_years", MONTHS / 12)
lab.record("surv_survival_rate", survival_rate)

# %% [markdown]
# **Per-universe statistics**, then averaged across the 200 universes (annualised on 12
# periods/year, §2.3's convention) — never computed on a series averaged ACROSS universes first,
# which would diversify away the very volatility we are trying to report. The CAGR gap is a Monte
# Carlo statistic, so it is asserted at 4 standard errors, never on a single lucky draw.

# %%
def cagr_paths(simple):  # simple: (months, reps) → (reps,)
    growth = np.nanprod(1.0 + simple, axis=0)
    years = simple.shape[0] / 12
    return growth ** (1.0 / years) - 1.0


def vol_paths(simple):
    return np.nanstd(simple, axis=0, ddof=1) * np.sqrt(12)


def sharpe_paths(simple):
    return np.nanmean(simple, axis=0) / np.nanstd(simple, axis=0, ddof=1) * np.sqrt(12)


cagr_pit_paths, cagr_surv_paths = cagr_paths(pit_ret), cagr_paths(surv_ret)
gap_paths = cagr_surv_paths - cagr_pit_paths
gap_mean, gap_se = float(gap_paths.mean()), float(gap_paths.std(ddof=1) / np.sqrt(N_REPS))
print(f"mean CAGR gap (survivorship-filtered − point-in-time) = {gap_mean:.2%}, SE {gap_se:.3%}, "
      f"{gap_mean / gap_se:.1f} SE from zero")
assert gap_mean - 4 * gap_se > 0, "the look-ahead universe should overstate CAGR reliably, at 4 SE"

for name, paths in (("pit", pit_ret), ("surv", surv_ret)):
    lab.record(f"surv_cagr_{name}", float(cagr_paths(paths).mean()))
    lab.record(f"surv_vol_{name}", float(vol_paths(paths).mean()))
    lab.record(f"surv_sharpe_{name}", float(sharpe_paths(paths).mean()))
    lab.record(f"surv_maxdd_{name}", float(max_drawdown_paths(paths).mean()))
lab.record("surv_cagr_gap", gap_mean)
lab.record("surv_cagr_gap_se", gap_se)
lab.record("surv_cagr_gap_pp", gap_mean * 100)  # in percentage POINTS, for prose that says so explicitly
print(f"point-in-time CAGR {lab.results['surv_cagr_pit']:.2%} · "
      f"survivorship-filtered CAGR {lab.results['surv_cagr_surv']:.2%}")

# %% [markdown]
# **Chart**: one representative universe — the one whose CAGR gap is closest to the median of all
# 200 — so the picture shows a typical run, not an average-of-averages that would smooth away the
# noise a real backtest actually has. The statistical claim above already covers all 200 universes.

# %%
def in_years_monthly(s: pd.Series) -> pd.Series:
    return pd.Series(s.to_numpy(), index=np.arange(1, len(s) + 1) / 12)


typical = int(np.argmin(np.abs(gap_paths - np.median(gap_paths))))
growth_pit = in_years_monthly((1.0 + pd.Series(pit_ret[:, typical])).cumprod())
growth_surv = in_years_monthly((1.0 + pd.Series(surv_ret[:, typical])).cumprod())
lab.chart("growth", charts.line_chart(
    [charts.Series("Point-in-time correct", growth_pit, role="strategy", end_label="Point-in-time"),
     charts.Series("Survivorship-filtered (look-ahead)", growth_surv, role="loss", end_label="Survivorship")],
    title="Growth of $1 in one representative simulated universe (log scale)",
    logy=True, y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

# %% [markdown]
# ## 3 · Restatement bias: sorting on a noisy-but-timely signal vs. a clean-but-late one
#
# A synthetic "quality" score drives next quarter's return. Nobody observes true quality directly:
# at the time, you only see the **as-reported** figure (noisy — normal reporting/estimation error);
# years later a **restatement** replaces it with a cleaner number, closer to the truth, because it
# was corrected with hindsight. Sorting a long-short portfolio on the *restated* number — as a
# vendor's "current" database invites you to do, even for a historical date — is **look-ahead
# bias**: identical timing, but a more accurate ruler than anyone had at the time.

# %%
N_ASSETS, N_REPS_B, QUARTERS = 60, 200, 60                        # 15 years, quarterly rebalancing
BETA, SIGMA_E = 0.008, 0.10                                       # true quarterly return per unit of quality
SIGMA_REPORT, SIGMA_RESTATE = 2.0, 0.3                            # noise added to true quality (unit variance)
IC_TRUE_THEORY = BETA / np.sqrt(BETA ** 2 + SIGMA_E ** 2)         # attainable IC with PERFECT knowledge (unreachable)
IC_REPORTED_THEORY = IC_TRUE_THEORY / np.sqrt(1 + SIGMA_REPORT ** 2)   # classical attenuation by measurement noise
IC_RESTATED_THEORY = IC_TRUE_THEORY / np.sqrt(1 + SIGMA_RESTATE ** 2)


def row_corr(a, b):
    """Pearson correlation computed per row (per quarter), vectorised over the assets axis."""
    a, b = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    return (a * b).mean(axis=1) / (a.std(axis=1) * b.std(axis=1))


def long_short_return(signal, forward_return, quintile=0.2):
    """Top-quintile minus bottom-quintile average forward return, per quarter (vectorised)."""
    k = int(round(signal.shape[1] * quintile))
    order = np.argsort(signal, axis=1)
    sorted_fwd = np.take_along_axis(forward_return, order, axis=1)
    return sorted_fwd[:, -k:].mean(axis=1) - sorted_fwd[:, :k].mean(axis=1)


ic_reported_reps, ic_restated_reps = np.empty(N_REPS_B), np.empty(N_REPS_B)
sharpe_reported_reps, sharpe_restated_reps = np.empty(N_REPS_B), np.empty(N_REPS_B)
for i in range(N_REPS_B):
    true_quality = rng_factor.standard_normal((QUARTERS, N_ASSETS))
    forward_return = BETA * true_quality + SIGMA_E * rng_factor.standard_normal((QUARTERS, N_ASSETS))
    reported = true_quality + SIGMA_REPORT * rng_factor.standard_normal((QUARTERS, N_ASSETS))
    restated = true_quality + SIGMA_RESTATE * rng_factor.standard_normal((QUARTERS, N_ASSETS))

    ic_reported_reps[i] = row_corr(reported, forward_return).mean()
    ic_restated_reps[i] = row_corr(restated, forward_return).mean()
    ls_reported = long_short_return(reported, forward_return)
    ls_restated = long_short_return(restated, forward_return)
    sharpe_reported_reps[i] = ls_reported.mean() / ls_reported.std(ddof=1) * np.sqrt(4)   # 4 quarters/year
    sharpe_restated_reps[i] = ls_restated.mean() / ls_restated.std(ddof=1) * np.sqrt(4)

for label, empirical, theory in (("reported", ic_reported_reps, IC_REPORTED_THEORY),
                                 ("restated", ic_restated_reps, IC_RESTATED_THEORY)):
    mean_ic, se_ic = float(empirical.mean()), float(empirical.std(ddof=1) / np.sqrt(N_REPS_B))
    print(f"IC {label}: empirical {mean_ic:.4f} ± {se_ic:.4f} SE · theory {theory:.4f}")
    assert abs(mean_ic - theory) < 4 * se_ic, f"empirical {label} IC should match the attenuation formula at 4 SE"
    lab.record(f"qf_ic_{label}_empirical", mean_ic)
    lab.record(f"qf_ic_{label}_theory", float(theory))

sharpe_gap = sharpe_restated_reps - sharpe_reported_reps
gap_mean_b, gap_se_b = float(sharpe_gap.mean()), float(sharpe_gap.std(ddof=1) / np.sqrt(N_REPS_B))
print(f"long-short Sharpe gap (restated − reported) = {gap_mean_b:.2f} ± {gap_se_b:.3f} SE")
assert gap_mean_b - 4 * gap_se_b > 0, "restated data should inflate the long-short Sharpe reliably, at 4 SE"

lab.record("qf_n_assets", N_ASSETS)
lab.record("qf_quarters", QUARTERS)
lab.record("qf_n_reps", N_REPS_B)
lab.record("qf_ic_true_theory", float(IC_TRUE_THEORY))
lab.record("qf_sharpe_reported", float(sharpe_reported_reps.mean()))
lab.record("qf_sharpe_restated", float(sharpe_restated_reps.mean()))
lab.record("qf_sharpe_gap", gap_mean_b)

lab.chart("quality_sharpe", charts.column_chart(
    ["As-reported (point-in-time)", "Restated (look-ahead)"],
    [float(sharpe_reported_reps.mean()), float(sharpe_restated_reps.mean())],
    title="Long-short Sharpe of the SAME quality signal, sorted on two versions of the same data",
    y_fmt=charts.fmt_num(2), roles=["strategy", "loss"]))

# %% [markdown]
# ## 4 · Backfill bias: selecting on a track record with zero true skill
#
# `M_CANDIDATES` hypothetical strategies each run a 24-month incubation with **zero** true edge. A
# database only backfills (lists the history of) the best `SELECT_TOP_P` by incubation track
# record — exactly how a hedge-fund or coin database only ever shows you the ones good enough to
# get listed. The selected subset's average return is a biased estimate of "what these strategies
# do", even though nothing about any individual number is wrong.

# %%
M_CANDIDATES, N_REPS_C, INCUBATION_MONTHS, SELECT_TOP_P = 3000, 200, 24, 0.15
TRUE_MU, TRUE_SIGMA = 0.0, 0.04                                  # monthly; zero true skill, by construction

sigma_mean = TRUE_SIGMA / np.sqrt(INCUBATION_MONTHS)
z_p = norm.ppf(1 - SELECT_TOP_P)
theory_selected_mean = TRUE_MU + sigma_mean * norm.pdf(z_p) / SELECT_TOP_P   # truncated-normal selection mean

pop_means, sel_means = np.empty(N_REPS_C), np.empty(N_REPS_C)
for i in range(N_REPS_C):
    incubation = rng_backfill.normal(TRUE_MU, TRUE_SIGMA, (M_CANDIDATES, INCUBATION_MONTHS))
    track_record = incubation.mean(axis=1)
    cutoff = np.quantile(track_record, 1 - SELECT_TOP_P)
    pop_means[i] = track_record.mean()
    sel_means[i] = track_record[track_record >= cutoff].mean()

pop_mean, pop_se = float(pop_means.mean()), float(pop_means.std(ddof=1) / np.sqrt(N_REPS_C))
sel_mean, sel_se = float(sel_means.mean()), float(sel_means.std(ddof=1) / np.sqrt(N_REPS_C))
print(f"population mean monthly return {pop_mean:.4%} ± {pop_se:.4%} SE (true value 0)")
print(f"backfilled (top {SELECT_TOP_P:.0%}) mean monthly return {sel_mean:.4%} ± {sel_se:.4%} SE, "
      f"theory {theory_selected_mean:.4%}")
assert abs(pop_mean - TRUE_MU) < 4 * pop_se
assert abs(sel_mean - theory_selected_mean) < 4 * sel_se, "the selected mean should match the truncated-normal formula"

lab.record("bf_m_candidates", M_CANDIDATES)
lab.record("bf_n_reps", N_REPS_C)
lab.record("bf_incubation_months", INCUBATION_MONTHS)
lab.record("bf_select_pct", SELECT_TOP_P)
lab.record("bf_population_mean_monthly", pop_mean)
lab.record("bf_backfilled_mean_monthly", sel_mean)
lab.record("bf_backfilled_mean_theory", float(theory_selected_mean))
lab.record("bf_bias_monthly", sel_mean - pop_mean)
lab.record("bf_bias_monthly_pp", (sel_mean - pop_mean) * 100)
lab.record("bf_bias_annualized", (1 + sel_mean) ** 12 - (1 + pop_mean) ** 12)

lab.chart("backfill", charts.column_chart(
    ["Full population (unlisted included)", "Backfilled subset only"],
    [pop_mean, sel_mean],
    title="Mean incubation-period return: everyone who tried, vs. only those who got listed",
    y_fmt=charts.fmt_pct(2), roles=["strategy", "loss"]))

# %%
lab.save()
