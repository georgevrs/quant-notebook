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
# # 1.2 · Where Returns Come From — companion lab
#
# **Quant Notebook** · Unit 1 · Session 2 · [Read the session](https://georgevrs.github.io/quant-notebook/unit01-quant-landscape/session02-where-returns-come-from.html)
#
# The five honest sources of edge — risk premia, behavioural anomalies, liquidity provision, information and speed — and why most apparent edges are none of them.
#
# We build a synthetic market where we KNOW the truth — which factors carry a premium, how often
# the market crashes, and which strategy (if any) has real skill. Then we run four strategies
# through it. Three of them look like alpha and are not; one looks dull and is the only real edge.
# A regression on the right factors tells them apart, and the shape of the returns (skewness,
# drawdown) tells you what the regression cannot. Last, we check Sharpe's arithmetic of active
# management: before costs, active investors as a group earn exactly the market.
#
# Synthetic **excess** returns throughout: the risk-free rate is 0 in this world.
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
from scipy.stats import norm, skew

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR, annualised_return, max_drawdown, sharpe_ratio
from quantnb.stats import factor_regression, sharpe_se

lab = qn.Lab("1.2")  # seeds the random generators: every run gives the same numbers
# One independent random stream per experiment: editing one section never changes another's numbers.
rng_world, rng_crash, rng_picker, rng_insurer, rng_managers, rng_theory = lab.rng.spawn(6)

# %% [markdown]
# ## 1 · A market where we know the truth
#
# `quantnb.synth.factor_model_returns` gives us the raw material of a linear factor model,
# R = B f + e: zero-mean factor shocks f, each stock's loadings B on those factors, and
# stock-specific (idiosyncratic) noise e. We then decide what the factors *pay*:
#
# * **MKT** — the market factor. It earns a 6% premium a year for bearing equity risk, and on
#   rare days it crashes (about 0.4 times a year, −7% on average). Part of the premium is the
#   price of that crash risk.
# * **STYLE** — a value-like style factor with a 4% premium (think "cheap stocks beat expensive
#   ones", Unit 8). Less volatile than the market.
# * **IND** — an industry-like factor that moves prices but carries no premium at all.
#
# Time is simulated, so series are indexed by elapsed years, not calendar dates.

# %%
DAYS = PERIODS_PER_YEAR                  # 252 trading days a year
YEARS = 20
T = DAYS * YEARS
N_ASSETS = 300
FACTORS = ["MKT", "STYLE", "IND"]
PREMIUM = {"MKT": 0.06, "STYLE": 0.04, "IND": 0.00}   # annual expected excess return
VOL = {"MKT": 0.15, "STYLE": 0.08, "IND": 0.15}       # annual volatility of the diffusive part
CRASH_RATE = 0.4                          # expected market crashes per year
CRASH_MEAN, CRASH_SD = -0.07, 0.02        # size of a one-day crash
IDIO_VOL = 0.25
mkt_drift = PREMIUM["MKT"] - CRASH_RATE * CRASH_MEAN   # diffusive drift: crashes included, MKT earns 6%


def market_factor(n_days: int, shocks: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Daily market excess returns (drift + diffusion + rare one-day crashes), and the crash-day flags."""
    crashed = rng.random(n_days) < CRASH_RATE / DAYS
    jumps = np.where(crashed, rng.normal(CRASH_MEAN, CRASH_SD, n_days), 0.0)
    return mkt_drift / DAYS + VOL["MKT"] / np.sqrt(DAYS) * shocks + jumps, crashed


raw = qn.synth.factor_model_returns(T, N_ASSETS, n_factors=3, factor_vol=0.15,
                                    idio_vol=IDIO_VOL, rng=rng_world)
shocks = raw["factors"].to_numpy() / (0.15 / np.sqrt(DAYS))        # unit-variance daily shocks
B = raw["loadings"].to_numpy()                                    # N × 3 loadings, the truth
idio = (raw["returns"] - raw["factors"] @ raw["loadings"].T).to_numpy()  # e = R − f B'
elapsed = pd.Index(np.arange(1, T + 1) / DAYS, name="years")      # simulated time, in years

F = np.empty((T, 3))
F[:, 0], crash_day = market_factor(T, shocks[:, 0], rng_crash)
F[:, 1] = PREMIUM["STYLE"] / DAYS + VOL["STYLE"] / np.sqrt(DAYS) * shocks[:, 1]
F[:, 2] = PREMIUM["IND"] / DAYS + VOL["IND"] / np.sqrt(DAYS) * shocks[:, 2]
factors = pd.DataFrame(F, index=elapsed, columns=FACTORS)
R = idio + F @ B.T                        # every stock's daily excess return: R = B F + e

mkt = factors["MKT"]
n_crash_days = int(crash_day.sum())
print(f"market: mean {mkt.mean() * DAYS:.2%} a year · vol {mkt.std() * np.sqrt(DAYS):.2%} · "
      f"{n_crash_days} crash days in {YEARS} years · worst day {mkt.min():.1%}")
lab.record("years", YEARS)
lab.record("n_assets", N_ASSETS)
lab.record("mkt_premium", PREMIUM["MKT"])
lab.record("style_premium", PREMIUM["STYLE"])
lab.record("crash_rate", CRASH_RATE)
lab.record("crash_mean", CRASH_MEAN)
lab.record("n_crash_days", n_crash_days)
lab.record("expected_crash_days", CRASH_RATE * YEARS)
lab.record("mkt_worst_day", float(mkt.min()))

# %% [markdown]
# ## 2 · Four strategies, one of them skilled
#
# * **Rocket** — buys the 30 most market-sensitive stocks and levers the book 1.2×.
#   No skill: it is market beta with a costume on.
# * **Picker** — a factor-neutral stock picker (market, style and industry hedged) with a genuine
#   *informational edge*: each month its forecast of every stock's stock-specific return is
#   correlated **0.01** with the truth. It holds weights proportional to the forecast.
# * **Insurer** — sells one-day crash insurance on the index: every day it collects a premium,
#   and pays three times the amount by which the market's fall exceeds 3% — a one-day put struck
#   3% below the market, sold three times over — plus unrelated noise (4% a year). Insurance is
#   sold at twice its expected payout: that markup is the (volatility) risk premium. No skill.
# * **Bargain** — market-neutral: long the 30 stocks with the highest STYLE loading, short the 30
#   lowest, market beta hedged with index futures. No skill: it is STYLE beta in disguise.

# %%
ROCKET_N, ROCKET_LEVERAGE = 30, 1.2
IC, PICKER_TE = 0.01, 0.05                # forecast-truth correlation; target tracking error
INSURE_STRIKE, INSURE_MARKUP, INSURE_SIZE, INSURE_NOISE = 0.03, 2.0, 3.0, 0.04
BARGAIN_N, BARGAIN_STYLE_BETA = 30, 1.0

# Rocket: 1.2× equal-weighted in the 30 highest market loadings.
top_beta = np.argsort(B[:, 0])[-ROCKET_N:]
w_rocket = np.zeros(N_ASSETS)
w_rocket[top_beta] = ROCKET_LEVERAGE / ROCKET_N
rocket = R @ w_rocket

# Picker: monthly forecasts of next month's stock-specific return, correlated IC with the truth.
MONTH = DAYS // 12                        # 21 trading days
n_months = T // MONTH
idio_month = idio.reshape(n_months, MONTH, N_ASSETS).sum(axis=1)
sd_month = IDIO_VOL / np.sqrt(12)
z = IC * idio_month / sd_month + np.sqrt(1 - IC ** 2) * rng_picker.standard_normal((n_months, N_ASSETS))
scale = PICKER_TE / (IDIO_VOL * np.sqrt(N_ASSETS))   # makes the ex-ante tracking error ≈ 5%
w_picker = scale * np.repeat(z, MONTH, axis=0)        # weights held for the whole month
picker = (w_picker * idio).sum(axis=1)                # factor-hedged: only stock-specific return
picker_alpha_true = 12 * scale * N_ASSETS * IC * sd_month
picker_gross = float(np.abs(w_picker).sum(axis=1).mean())


# Insurer: premium θ minus the payout of a daily put struck 3% below, sized 3×, plus noise.
def expected_payout(strike: float) -> float:
    """E[max(0, −MKT − strike)] per day under the TRUE model (a normal mixture: crash / no crash)."""
    def call_on_normal(m: float, s: float) -> float:   # E[max(0, Y)] for Y ~ N(m, s²)
        return m * norm.cdf(m / s) + s * norm.pdf(m / s)
    p = CRASH_RATE / DAYS
    mu, sd = mkt_drift / DAYS, VOL["MKT"] / np.sqrt(DAYS)
    calm = call_on_normal(-mu - strike, sd)
    crashed = call_on_normal(-(mu + CRASH_MEAN) - strike, np.hypot(sd, CRASH_SD))
    return (1 - p) * calm + p * crashed


premium_daily = INSURE_MARKUP * expected_payout(INSURE_STRIKE)


def insurance(mkt_returns: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(PUT factor = one unit of sold insurance, Insurer = INSURE_SIZE units + unrelated noise)."""
    put = premium_daily - np.maximum(0.0, -mkt_returns - INSURE_STRIKE)
    noise = INSURE_NOISE / np.sqrt(DAYS) * rng.standard_normal(len(mkt_returns))
    return put, INSURE_SIZE * put + noise


put_factor, insurer = insurance(mkt.to_numpy(), rng_insurer)
factors["PUT"] = put_factor

# Bargain: long top-30 / short bottom-30 STYLE loadings, then hedge the market beta.
order = np.argsort(B[:, 1])
w_bargain = np.zeros(N_ASSETS)
w_bargain[order[-BARGAIN_N:]] = 1 / BARGAIN_N
w_bargain[order[:BARGAIN_N]] = -1 / BARGAIN_N
w_bargain *= BARGAIN_STYLE_BETA / (w_bargain @ B[:, 1])   # scale to a STYLE loading of 1.0
bargain = R @ w_bargain - (w_bargain @ B[:, 0]) * mkt.to_numpy()   # short index futures

strategies = pd.DataFrame({"Market": mkt.to_numpy(), "Rocket": rocket, "Picker": picker,
                           "Insurer": insurer, "Bargain": bargain}, index=elapsed)

# The truth, for checking the regressions later: alpha a year, and loadings on MKT/STYLE/IND/PUT.
truth = pd.DataFrame({
    "alpha": [0.0, 0.0, picker_alpha_true, 0.0, 0.0],
    "MKT":   [1.0, w_rocket @ B[:, 0], 0.0, 0.0, 0.0],
    "STYLE": [0.0, w_rocket @ B[:, 1], 0.0, 0.0, w_bargain @ B[:, 1]],
    "IND":   [0.0, w_rocket @ B[:, 2], 0.0, 0.0, w_bargain @ B[:, 2]],
    "PUT":   [0.0, 0.0, 0.0, INSURE_SIZE, 0.0],
}, index=strategies.columns)
print(truth.round(3))
lab.record("rocket_n", ROCKET_N)
lab.record("rocket_leverage", ROCKET_LEVERAGE)
lab.record("rocket_beta_true", float(truth.loc["Rocket", "MKT"]))
lab.record("picker_ic", IC)
lab.record("picker_alpha_true", float(picker_alpha_true))
lab.record("picker_te_target", PICKER_TE)
lab.record("picker_gross", picker_gross)
lab.record("insure_strike", INSURE_STRIKE)
lab.record("insure_markup", INSURE_MARKUP)
lab.record("insure_size", INSURE_SIZE)
lab.record("insure_noise", INSURE_NOISE)
lab.record("insure_premium_annual", float(INSURE_SIZE * premium_daily * DAYS))
lab.record("insure_expected_loss_annual", float(INSURE_SIZE * expected_payout(INSURE_STRIKE) * DAYS))
lab.record("bargain_style_beta", BARGAIN_STYLE_BETA)
lab.record("bargain_n", BARGAIN_N)

# %% [markdown]
# ## 3 · The truth in numbers: population values
#
# Twenty years is one sample. To know what each statistic *should* be, we compute its population
# value — analytically where we can, and otherwise from 1,000 simulated years of the same world.
# Every assert below compares the 20-year sample with these values, at 3 standard errors.

# %%
THEORY_YEARS = 1_000
p_crash = CRASH_RATE / DAYS
var_mkt_daily = VOL["MKT"] ** 2 / DAYS + p_crash * (CRASH_MEAN ** 2 + CRASH_SD ** 2) - (p_crash * CRASH_MEAN) ** 2
cov_f = np.diag([var_mkt_daily, VOL["STYLE"] ** 2 / DAYS, VOL["IND"] ** 2 / DAYS])   # independent factors
premia = np.array([PREMIUM[f] for f in FACTORS])


def linear_book(w: np.ndarray) -> dict:
    """Population mean, vol and Sharpe (annual) of a stock portfolio w, straight from the model."""
    b = w @ B
    var_daily = b @ cov_f @ b + (w ** 2).sum() * IDIO_VOL ** 2 / DAYS
    mean, vol = float(b @ premia), float(np.sqrt(var_daily * DAYS))
    return {"mean": mean, "vol": vol, "sharpe": mean / vol, "r2_mkt": b[0] ** 2 * var_mkt_daily / var_daily}


pop = {"market_mean": PREMIUM["MKT"], "market_vol": float(np.sqrt(var_mkt_daily * DAYS))}
pop["market_sharpe"] = pop["market_mean"] / pop["market_vol"]
rocket_pop = linear_book(w_rocket)
# Bargain against the market alone: its STYLE (and IND) exposure is invisible, so it shows up as alpha.
pop["bargain_mean"] = float(truth.loc["Bargain", ["STYLE", "IND"]] @ premia[1:])   # hedged: no MKT premium
pop["bargain_mkt_alpha"] = float(truth.loc["Bargain", "STYLE"] * PREMIUM["STYLE"] + truth.loc["Bargain", "IND"] * PREMIUM["IND"])
pop["rocket_mkt_alpha"] = float(truth.loc["Rocket", "STYLE"] * PREMIUM["STYLE"] + truth.loc["Rocket", "IND"] * PREMIUM["IND"])

# The Insurer is nonlinear in the market, so we measure its population values on 1,000 years.
n_long = THEORY_YEARS * DAYS
mkt_long, _ = market_factor(n_long, rng_theory.standard_normal(n_long), rng_theory)
_, insurer_long = insurance(mkt_long, rng_theory)
months_long = (1 + insurer_long).reshape(-1, MONTH).prod(axis=1) - 1
fit_long = factor_regression(insurer_long, mkt_long, names=["MKT"])
pop.update({
    "insurer_mean": float(insurer_long.mean() * DAYS),
    "insurer_vol": float(insurer_long.std() * np.sqrt(DAYS)),
    "insurer_skew": float(skew(months_long)),
    "insurer_up_months": float((months_long > 0).mean()),
    "insurer_mkt_alpha": fit_long["alpha_annual"],
    "insurer_mkt_beta": fit_long["betas"]["MKT"],
})
pop["insurer_sharpe"] = pop["insurer_mean"] / pop["insurer_vol"]
print({k: round(v, 4) for k, v in pop.items()})
print({k: round(v, 4) for k, v in rocket_pop.items()})
for key, value in pop.items():
    lab.record(f"pop_{key}", float(value))
for key, value in rocket_pop.items():
    lab.record(f"pop_rocket_{key}", float(value))
lab.record("theory_years", THEORY_YEARS)

# What theory teaches, independent of any one sample:
assert abs(rocket_pop["sharpe"] - pop["market_sharpe"]) < 0.05, "high beta + leverage should leave Sharpe ≈ the market's"
assert rocket_pop["r2_mkt"] > 0.8, "the market should explain almost all of the Rocket's variance"
assert pop["insurer_skew"] < -1.0, "selling crash insurance should be strongly negatively skewed"

# %% [markdown]
# ## 4 · The raw track records
#
# What a marketing deck would show you: average return, volatility, Sharpe ratio — plus what it
# usually leaves out: compound growth (CAGR), the skewness of monthly returns and the maximum
# drawdown. The *average (arithmetic)* return is shown because the alpha + beta split in section 5
# is exact only for averages; the money actually compounded at the CAGR (Session 2.3, §4).


# %%
def monthly(r: pd.Series) -> np.ndarray:
    """Compound daily returns into 21-day months."""
    return (1 + r.to_numpy()).reshape(-1, MONTH).prod(axis=1) - 1


def track_record(r: pd.Series) -> dict:
    m = monthly(r)
    return {
        "mean": r.mean() * DAYS,                     # average (arithmetic) return a year
        "cagr": annualised_return(r),                # what the money compounded at
        "vol": r.std() * np.sqrt(DAYS),
        "sharpe": sharpe_ratio(r),                   # cash pays 0%: returns are already excess
        "skew": float(skew(m)),
        "maxdd": max_drawdown(r),
        "worst_month": float(m.min()),
        "pct_up_months": float((m > 0).mean()),
        "growth": float((1 + r).prod()),
    }


records = pd.DataFrame({name: track_record(strategies[name]) for name in strategies}).T
print(records.round(3))
for name, row in records.iterrows():
    for col, value in row.items():
        lab.record(f"{name.lower()}_{col}", float(value))
rocket_minus_market = float(records.loc["Rocket", "mean"] - records.loc["Market", "mean"])
lab.record("rocket_minus_market", rocket_minus_market)
lab.record("rocket_minus_market_pp", 100 * rocket_minus_market)   # in percentage points

# Samples agree with the population at 3 standard errors (SE of a 20-year average = vol / √20):
se_mean = lambda vol: 3 * vol / np.sqrt(YEARS)   # noqa: E731
assert abs(records.loc["Rocket", "mean"] - rocket_pop["mean"]) < se_mean(rocket_pop["vol"])
assert abs(records.loc["Insurer", "mean"] - pop["insurer_mean"]) < se_mean(pop["insurer_vol"])
assert abs(records.loc["Market", "mean"] - pop["market_mean"]) < se_mean(pop["market_vol"])

# %% [markdown]
# ## 5 · Alpha and beta: which model you regress on decides what "alpha" means
#
# Regress each strategy's daily excess return on factor returns with
# `quantnb.stats.factor_regression`. The intercept, times 252, is the annual **alpha**; the slopes
# are the **betas**. We fit two models:
#
# * **mkt** — the market alone: what most people check (CAPM-style, Session 7.2);
# * **full** — MKT, STYLE, IND and the crash-insurance factor PUT: everything priced in this world.
#
# The sample average return splits exactly into alpha plus Σ beta × factor average — an identity
# of OLS with an intercept — so the "where did the return come from" chart adds up to the total.

# %%
MODELS = {"mkt": ["MKT"], "full": ["MKT", "STYLE", "IND", "PUT"]}
rows = []
for name in ["Rocket", "Picker", "Insurer", "Bargain"]:
    y = strategies[name]
    for model, cols in MODELS.items():
        fit = factor_regression(y, factors[cols], names=cols)
        alpha = fit["alpha_annual"]
        explained = float(sum(fit["betas"][c] * factors[c].mean() for c in cols) * DAYS)
        assert np.isclose(alpha + explained, y.mean() * DAYS)   # average = alpha + Σ beta × factor average
        rows.append({"strategy": name, "model": model, "alpha": alpha,
                     "alpha_se": alpha / fit["alpha_t"], "t_alpha": fit["alpha_t"],
                     "beta_mkt": fit["betas"]["MKT"], "r2": fit["r2"], "explained": explained,
                     **{f"b_{c}": fit["betas"][c] for c in cols},
                     **{f"se_{c}": fit["betas"][c] / fit["beta_t"][c] for c in cols}})
reg = pd.DataFrame(rows).set_index(["strategy", "model"])
print(reg[["alpha", "t_alpha", "beta_mkt", "r2", "explained"]].round(3))
for (name, model), row in reg.iterrows():
    key = f"{name.lower()}_{model}"
    for col in ["alpha", "alpha_se", "t_alpha", "beta_mkt", "r2", "explained"]:
        lab.record(f"{key}_{col}", float(row[col]))
lab.record("insurer_full_b_put", float(reg.loc[("Insurer", "full"), "b_PUT"]))
lab.record("bargain_full_b_style", float(reg.loc[("Bargain", "full"), "b_STYLE"]))


def within(estimate: float, true: float, se: float, k: float = 3.0) -> bool:
    return abs(estimate - true) < k * se


# The full model recovers the truth for every strategy: alphas and loadings within 3 standard errors.
for name in ["Rocket", "Picker", "Insurer", "Bargain"]:
    row = reg.loc[(name, "full")]
    assert within(row["alpha"], truth.loc[name, "alpha"], row["alpha_se"]), f"{name}: full-model alpha"
    for c in ["MKT", "STYLE", "PUT"]:
        assert within(row[f"b_{c}"], truth.loc[name, c], row[f"se_{c}"]), f"{name}: {c} loading"
# The market-only model is fooled: its alphas match the population values of the hidden exposures.
for name, key in [("Rocket", "rocket_mkt_alpha"), ("Bargain", "bargain_mkt_alpha"), ("Insurer", "insurer_mkt_alpha")]:
    row = reg.loc[(name, "mkt")]
    assert within(row["alpha"], pop[key], row["alpha_se"]), f"{name}: market-only alpha vs its population value"

# %% [markdown]
# ## 6 · The Picker's edge: tiny per bet, large in aggregate
#
# The Picker's forecasts are barely better than a coin: with a correlation of 0.01, it calls the
# direction of a stock's specific return right about 50.3% of the time. But it makes that call on
# 300 stocks, 12 times a year. The **fundamental law of active management** (Grinold & Kahn;
# Session 8.7) says the information ratio is about IC × √breadth.

# %%
breadth = N_ASSETS * 12
ir_theory = IC * np.sqrt(breadth)
picker_te = picker.std() * np.sqrt(DAYS)
ir_measured = reg.loc[("Picker", "full"), "alpha"] / picker_te
hit_rate = 0.5 + np.arcsin(IC) / np.pi                 # P(forecast sign = outcome sign), bivariate normal
realised_ic = float(np.mean([np.corrcoef(z[m], idio_month[m])[0, 1] for m in range(n_months)]))
print(f"IC {IC} · breadth {breadth:,} · IR theory {ir_theory:.2f} · measured {ir_measured:.2f} · "
      f"hit rate {hit_rate:.2%} · realised IC {realised_ic:.4f}")
lab.record("picker_breadth", breadth)
lab.record("picker_ir_theory", float(ir_theory))
lab.record("picker_ir_measured", float(ir_measured))
lab.record("picker_te", float(picker_te))
lab.record("picker_hit_rate", float(hit_rate))
lab.record("picker_realised_ic", realised_ic)
# 20 years pin an information ratio down only to about ±1/√20 — assert at 3 standard errors.
assert abs(ir_measured - ir_theory) < 3 * sharpe_se(ir_theory, YEARS)

# %% [markdown]
# ## 7 · The arithmetic of active management (Sharpe, 1991)
#
# Every share in the market is owned by someone. Split the owners into **passive** investors, who
# hold every stock in proportion to its market value, and **active** investors, who hold anything
# else. Passive investors earn the market return. The market return is the value-weighted average
# of everyone's return. So active investors, *as a group*, must also earn the market return —
# before costs. After costs (fees, spreads, impact), they must earn less.
#
# We check it: 1,000 active managers hold random tilted portfolios, re-drawn every year, that add
# up (together with the passive 40%) to exactly the market.

# %%
N_MANAGERS = 1_000
PASSIVE_SHARE = 0.40                       # of every stock, owned by index funds (illustrative)
TILT = 1.0                                 # how far active managers stray from market weights
ACTIVE_COST, PASSIVE_COST = 0.010, 0.0005  # annual all-in costs: fees + trading (illustrative)

caps = rng_managers.lognormal(0.0, 1.0, N_ASSETS)
w_mkt = caps / caps.sum()                               # the market portfolio: value weights
size = rng_managers.lognormal(0.0, 1.0, N_MANAGERS)     # managers differ in size
growth = (1 + R).reshape(YEARS, DAYS, N_ASSETS).prod(axis=1)   # each stock's gross return, per year
market_year = growth @ w_mkt - 1                        # what passive investors earn, before costs


def active_holdings() -> np.ndarray:
    """Dollar holdings (managers × stocks) of the active investors.

    Every manager tilts away from market weights at random; then each stock's column is rescaled
    so that active + passive investors together own exactly 100% of it — as in the real market.
    """
    tilts = np.exp(TILT * rng_managers.standard_normal((N_MANAGERS, N_ASSETS)))
    H = size[:, None] * w_mkt[None, :] * tilts
    return H * ((1 - PASSIVE_SHARE) * w_mkt / H.sum(axis=0))


active = np.empty((YEARS, N_MANAGERS))     # each manager's return, each year, before costs
aggregate = np.empty(YEARS)                # the return on the average active DOLLAR
for y in range(YEARS):
    H = active_holdings()                  # managers re-draw their bets every year
    dollars = H.sum(axis=1)
    active[y] = (H / dollars[:, None]) @ growth[y] - 1
    aggregate[y] = dollars @ active[y] / dollars.sum()

# Sharpe's identity: before costs, the average active dollar earns EXACTLY the market return.
identity_gap = float(np.abs(aggregate - market_year).max())
assert identity_gap < 1e-12
print(f"largest gap between the average active dollar and the market, any year: {identity_gap:.1e}")

# After costs: the average active dollar trails the average passive dollar by the cost difference.
active_net = active - ACTIVE_COST
passive_net = market_year - PASSIVE_COST
aggregate_net_gap = float(np.mean(aggregate - ACTIVE_COST - passive_net))
assert np.isclose(aggregate_net_gap, -(ACTIVE_COST - PASSIVE_COST))


def cagr(yearly: np.ndarray, axis: int = 0) -> np.ndarray:
    return (1 + yearly).prod(axis=axis) ** (1 / YEARS) - 1


beat_1y_gross = float((active > market_year[:, None]).mean())        # share of manager-years
beat_1y_net = float((active_net > passive_net[:, None]).mean())
excess_20y_gross = cagr(active) - cagr(market_year)                  # per manager, a year
excess_20y_net = cagr(active_net) - cagr(passive_net)
beat_20y_gross = float((excess_20y_gross > 0).mean())
beat_20y_net = float((excess_20y_net > 0).mean())
tracking_error = np.median((active - market_year[:, None]).std(axis=0, ddof=1))
print(f"beat the index fund — 1 year: {beat_1y_gross:.0%} before costs, {beat_1y_net:.0%} after; "
      f"{YEARS} years: {beat_20y_gross:.0%} before, {beat_20y_net:.0%} after · median TE {tracking_error:.1%}")

lab.record("n_managers", N_MANAGERS)
lab.record("passive_share", PASSIVE_SHARE)
lab.record("active_cost", ACTIVE_COST)
lab.record("passive_cost", PASSIVE_COST)
lab.record("identity_gap", identity_gap)
lab.record("aggregate_net_gap", aggregate_net_gap)
lab.record("beat_1y_gross", beat_1y_gross)
lab.record("beat_1y_net", beat_1y_net)
lab.record("beat_20y_gross", beat_20y_gross)
lab.record("beat_20y_net", beat_20y_net)
lab.record("manager_te", float(tracking_error))
lab.record("median_excess_20y_net", float(np.median(excess_20y_net)))

assert 0.40 < beat_1y_gross < 0.60          # before costs, averaged over 20,000 manager-years: a coin toss
assert beat_20y_net < beat_1y_net < 0.5     # after costs, and over time, the odds shrink

# %% [markdown]
# ## 8 · Charts for the session page

# %%
paths = (1 + strategies).cumprod()
lab.chart("equity", charts.line_chart(
    [charts.Series("Market", paths["Market"], role="benchmark"),
     charts.Series("Rocket", paths["Rocket"], role="alt1"),
     charts.Series("Insurer", paths["Insurer"], role="alt2"),
     charts.Series("Picker", paths["Picker"], role="strategy")],
    title=f"Growth of $1 over {YEARS} simulated years (log scale; x-axis in years)", logy=True,
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

# Where the return comes from, in the long run (population values, not this one sample): what a
# market-only regression calls alpha, vs the full model's split into factor exposure and alpha.
names = ["Rocket", "Picker", "Insurer", "Bargain"]
long_run = pd.DataFrame({
    "mkt_alpha": [pop["rocket_mkt_alpha"], picker_alpha_true, pop["insurer_mkt_alpha"], pop["bargain_mkt_alpha"]],
    "full_beta": [rocket_pop["mean"], 0.0, pop["insurer_mean"], pop["bargain_mean"]],   # all of it is β × premia
    "full_alpha": [0.0, picker_alpha_true, 0.0, 0.0],                                  # the truth
}, index=names)
lab.chart("decomposition", charts.grouped_column_chart(
    names,
    [("α, market-only model", long_run["mkt_alpha"].tolist(), "alt1"),
     ("β × factor premia, full model", long_run["full_beta"].tolist(), "benchmark"),
     ("α, full model (the truth)", long_run["full_alpha"].tolist(), "strategy")],
    title="Long-run average return a year, split two ways (population values)", y_fmt=charts.fmt_pct(0),
    value_labels=True))

# Skew made visible: every month, sorted from worst to best. A symmetric strategy traces a gentle
# S; a short-insurance strategy is flat and slightly positive — until its worst handful of months.
# (A histogram hides those months: each is a single observation, a bar one pixel tall.)
ranked = pd.DataFrame({name: np.sort(monthly(strategies[name])) for name in ["Market", "Picker", "Insurer"]},
                      index=np.arange(1, n_months + 1))
lab.chart("sorted_months", charts.line_chart(
    [charts.Series("Market", ranked["Market"], role="benchmark", label_end=False),
     charts.Series("Picker", ranked["Picker"], role="strategy", label_end=False),
     charts.Series("Insurer", ranked["Insurer"], role="alt2", label_end=False)],
    title="Every monthly return, sorted from worst (left) to best (right)",
    y_fmt=charts.fmt_pct(0), hline=0.0))
CRASH_MONTH = -0.04
lab.record("insurer_crash_months", int((ranked["Insurer"] < CRASH_MONTH).sum()))
lab.record("insurer_median_month", float(ranked["Insurer"].median()))

lab.chart("managers", charts.histogram(
    excess_20y_net, bins=40, x_fmt=charts.fmt_pct(1),
    title=f"{N_MANAGERS:,} active managers: annual return minus the index fund, after costs, {YEARS} years",
    tail_below=0.0, tail_label="trailed the index fund"))

# %%
lab.save()
