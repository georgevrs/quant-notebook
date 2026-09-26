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
# # 2.2 · Equities, Indices & ETFs — companion lab
#
# **Quant Notebook** · Unit 2 · Session 2 · [Read the session](https://georgevrs.github.io/quant-notebook/unit02-markets-instruments/session02-equities-indices-etfs.html)
#
# Shares, dividends, splits, index construction, ETF creation/redemption, and the mechanics and costs of shorting.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Everything here is **synthetic and illustrative**: four made-up companies, made-up dividend
# yields, borrow fees and arbitrage costs. The risk-free rate is 0 throughout (so no interest is
# earned on cash or collateral). Simulated time is elapsed trading time (252 days = 1 year),
# never a calendar date. Each experiment draws from its own random stream, so editing one section
# never changes another section's numbers.

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
from scipy.stats import norm, poisson

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR

lab = qn.Lab("2.2")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment (§12a of AUTHORING.md).
rng_path, rng_mc, rng_div, rng_exdate, rng_etf, rng_short = lab.rng.spawn(6)
DT = 1 / PERIODS_PER_YEAR


def in_years(values, start: float = 0.0) -> pd.Series:
    """Index a simulated daily series by elapsed years, so charts never imply calendar dates."""
    v = np.asarray(values, dtype=float)
    return pd.Series(v, index=start + np.arange(len(v)) / PERIODS_PER_YEAR)


# %% [markdown]
# ## 1 · A toy stock market: four companies, three index rules
#
# **Market capitalisation** = price × shares outstanding. **Free float** is the fraction of
# shares that can actually trade (not locked up by founders, families or governments).
# The same four stocks get very different weights under different rules:
#
# * **float-adjusted cap weighting** — weight ∝ price × shares × float (the S&P 500's rule);
# * **equal weighting** — 1/N each, restored at every quarterly rebalance;
# * **price weighting** — weight ∝ the share price alone (the Dow's rule).

# %%
UNIVERSE = pd.DataFrame(
    {
        "price": [50.0, 400.0, 30.0, 120.0],      # $ per share at the start
        "shares_m": [2_000.0, 100.0, 800.0, 25.0],  # shares outstanding, millions
        "float": [0.95, 0.90, 0.40, 1.00],        # fraction of shares that trade freely
        "beta": [1.0, 1.1, 0.9, 1.2],             # exposure to the common market factor
        "sigma": [0.22, 0.28, 0.32, 0.45],        # total volatility, per year
    },
    index=["Mega", "Pricey", "Family", "Small"],
)
MU = 0.08          # every stock has the SAME expected return: differences are weighting or luck
N_STOCKS = len(UNIVERSE)
SIGMA_M = 0.18     # volatility of the common market factor
UNIVERSE["cap_bn"] = UNIVERSE.price * UNIVERSE.shares_m / 1_000
UNIVERSE["float_cap_bn"] = UNIVERSE.cap_bn * UNIVERSE["float"]
UNIVERSE["w_cap"] = UNIVERSE.cap_bn / UNIVERSE.cap_bn.sum()
UNIVERSE["w_float"] = UNIVERSE.float_cap_bn / UNIVERSE.float_cap_bn.sum()
UNIVERSE["w_equal"] = 1 / len(UNIVERSE)
UNIVERSE["w_price"] = UNIVERSE.price / UNIVERSE.price.sum()
UNIVERSE["idio"] = np.sqrt(UNIVERSE.sigma ** 2 - (UNIVERSE.beta * SIGMA_M) ** 2)  # stock-specific vol
print(UNIVERSE.round(3))
for name, row in UNIVERSE.iterrows():
    k = name.lower()
    for col in ("price", "shares_m", "float", "cap_bn", "float_cap_bn", "w_cap", "w_float", "w_equal", "w_price", "sigma"):
        lab.record(f"u_{k}_{col}", float(row[col]))
lab.record("u_mu", MU)
lab.record("u_sigma_m", SIGMA_M)
lab.record("u_total_cap_bn", float(UNIVERSE.cap_bn.sum()))
for w in ("w_cap", "w_float", "w_equal", "w_price"):
    assert abs(UNIVERSE[w].sum() - 1) < 1e-12

# %% [markdown]
# ## 2 · One simulated decade, one split
#
# A one-factor model: each stock's daily log return is a common market shock times its beta plus
# its own shock, with total volatility `sigma` and expected return `MU`. Halfway through, *Pricey*
# splits **4-for-1**: every share becomes four, and the quoted price divides by four. Nothing
# economic happens — but a price series that ignores the split shows a 75% "crash".

# %%
YEARS_PATH = 10
N_DAYS = YEARS_PATH * PERIODS_PER_YEAR
SPLIT_DAY = N_DAYS // 2        # the split takes effect at the open of this trading day
SPLIT_RATIO = 4
REBAL_DAYS = 63                # equal-weight index restores 1/N every quarter (63 trading days)


def simulate_log_returns(rng: np.random.Generator, n_steps: int, dt: float, n_paths: int = 1) -> np.ndarray:
    """Log returns (n_steps, n_paths, n_stocks) from the one-factor model; E[simple growth] = MU a year."""
    beta, idio, sig = (UNIVERSE[c].to_numpy() for c in ("beta", "idio", "sigma"))
    z_m = rng.standard_normal((n_steps, n_paths, 1))
    z_i = rng.standard_normal((n_steps, n_paths, len(UNIVERSE)))
    return (MU - 0.5 * sig ** 2) * dt + np.sqrt(dt) * (beta * SIGMA_M * z_m + idio * z_i)


lr = simulate_log_returns(rng_path, N_DAYS, DT)[:, 0, :]                 # (days, stocks)
p0 = UNIVERSE.price.to_numpy()
econ = p0 * np.exp(np.vstack([np.zeros((1, N_STOCKS)), np.cumsum(lr, axis=0)]))  # split-adjusted ("economic") prices
split_factor = np.ones_like(econ)
i_pricey = UNIVERSE.index.get_loc("Pricey")
split_factor[SPLIT_DAY:, i_pricey] = SPLIT_RATIO
raw = econ / split_factor                          # what the exchange quotes: /4 from the split day on
shares = UNIVERSE.shares_m.to_numpy() * split_factor  # shares outstanding ×4 from the split day on
R = econ[1:] / econ[:-1] - 1                        # true daily simple returns (split-adjusted)

# --- float-adjusted cap-weighted index: sum of float market values / a constant divisor -------------
float_mv = UNIVERSE["float"].to_numpy() * shares * raw
cw_level = float_mv.sum(axis=1) / float_mv[0].sum() * 100      # divisor chosen so the index starts at 100
# its daily return is exactly the start-of-day-weighted average of the stocks' returns
w_prev = float_mv[:-1] / float_mv[:-1].sum(axis=1, keepdims=True)
assert np.allclose(cw_level[1:] / cw_level[:-1] - 1, (w_prev * R).sum(axis=1), atol=1e-12)
# and a cap-weighted index never needs to trade: yesterday's weights drift into today's by themselves
drifted = w_prev * (1 + R)
drifted /= drifted.sum(axis=1, keepdims=True)
assert np.allclose(drifted, float_mv[1:] / float_mv[1:].sum(axis=1, keepdims=True), atol=1e-12)

# --- equal-weighted index, rebalanced quarterly --------------------------------------------------------
ew_level = np.empty(N_DAYS + 1)
ew_level[0] = 100.0
turnover = []                                         # one-way turnover at each rebalance
anchor = 0
for t in range(1, N_DAYS + 1):
    growth = econ[t] / econ[anchor]                   # buy-and-hold since the last rebalance
    ew_level[t] = ew_level[anchor] * growth.mean()
    if t % REBAL_DAYS == 0 and t < N_DAYS:            # restore 1/N at the close
        w_before = growth / growth.sum()
        turnover.append(0.5 * np.abs(w_before - 1 / N_STOCKS).sum())
        anchor = t
ew_turnover_year = float(np.sum(turnover) / YEARS_PATH)

# --- price-weighted index (the Dow's rule): sum of quoted prices / a divisor --------------------------
divisor = np.full(N_DAYS + 1, raw[0].sum() / 100)     # start at 100
for t in range(1, N_DAYS + 1):
    divisor[t] = divisor[t - 1]
    if t == SPLIT_DAY:                                # re-set the divisor so the split moves nothing
        prev_post = raw[t - 1] / (split_factor[t] / split_factor[t - 1])   # yesterday's prices, split-adjusted
        divisor[t] = divisor[t - 1] * prev_post.sum() / raw[t - 1].sum()
pw_level = raw.sum(axis=1) / divisor
pw_naive = raw.sum(axis=1) / divisor[0]               # the bug: forget to adjust the divisor
# on the split day the correct index moves by the price-weighted average of TRUE returns
prev_post = raw[SPLIT_DAY - 1] / (split_factor[SPLIT_DAY] / split_factor[SPLIT_DAY - 1])
w_pw_post = prev_post / prev_post.sum()
assert np.isclose(pw_level[SPLIT_DAY] / pw_level[SPLIT_DAY - 1] - 1, (w_pw_post * R[SPLIT_DAY - 1]).sum(), atol=1e-12)

# weights of Pricey in the price-weighted index just before and just after the split
w_pw_before = raw[SPLIT_DAY - 1, i_pricey] / raw[SPLIT_DAY - 1].sum()
w_pw_after = w_pw_post[i_pricey]
# ... and in the cap-weighted index (a split changes nothing)
w_cw_before = float_mv[SPLIT_DAY - 1, i_pricey] / float_mv[SPLIT_DAY - 1].sum()
w_cw_after = float_mv[SPLIT_DAY, i_pricey] / float_mv[SPLIT_DAY].sum()


def cagr_of(level: np.ndarray, years: float) -> float:
    return float((level[-1] / level[0]) ** (1 / years) - 1)


def ann_vol_of(level: np.ndarray) -> float:
    return float(np.std(np.diff(np.log(level)), ddof=1) * np.sqrt(PERIODS_PER_YEAR))


for key, level in (("cw", cw_level), ("ew", ew_level), ("pw", pw_level), ("pw_naive", pw_naive)):
    lab.record(f"path_{key}_cagr", cagr_of(level, YEARS_PATH))
    lab.record(f"path_{key}_vol", ann_vol_of(level))
    lab.record(f"path_{key}_final", float(level[-1] / level[0]))
lab.record("path_years", YEARS_PATH)
lab.record("ew_turnover_year", ew_turnover_year)
lab.record("pw_pricey_w_before", float(w_pw_before))
lab.record("pw_pricey_w_after", float(w_pw_after))
lab.record("cw_pricey_w_before", float(w_cw_before))
lab.record("cw_pricey_w_after", float(w_cw_after))
lab.record("split_ratio", SPLIT_RATIO)
lab.record("split_year", SPLIT_DAY / PERIODS_PER_YEAR)
lab.record("pw_naive_split_day", float(pw_naive[SPLIT_DAY] / pw_naive[SPLIT_DAY - 1] - 1))
assert abs(w_cw_after - w_cw_before) < 0.02            # cap weight: only the day's price move changes it
x_pricey = raw[SPLIT_DAY - 1, i_pricey]                # price weight: the split slashes it, by formula
rest = raw[SPLIT_DAY - 1].sum() - x_pricey
assert np.isclose(w_pw_after, (x_pricey / SPLIT_RATIO) / (rest + x_pricey / SPLIT_RATIO))
assert w_pw_after < w_pw_before
for i, name in enumerate(UNIVERSE.index):
    lab.record(f"path_{name.lower()}_cagr", cagr_of(econ[:, i], YEARS_PATH))
print(f"CAGR over {YEARS_PATH} years — cap-weighted {lab.results['path_cw_cagr']:.2%}, "
      f"equal-weighted {lab.results['path_ew_cagr']:.2%}, price-weighted {lab.results['path_pw_cagr']:.2%}")
print(f"Pricey's weight in the price-weighted index: {w_pw_before:.1%} → {w_pw_after:.1%} at the split")

# %% [markdown]
# **The split, mishandled vs handled.** Compute returns from quoted prices and the split day shows
# a loss of about 75%. Every statistic downstream inherits it.

# %%
raw_r = raw[1:, i_pricey] / raw[:-1, i_pricey] - 1         # the bug: returns from quoted prices
adj_r = R[:, i_pricey]                                     # handled: split-adjusted returns
naive_split_day = raw_r[SPLIT_DAY - 1]
true_split_day = adj_r[SPLIT_DAY - 1]
assert np.isclose(1 + naive_split_day, (1 + true_split_day) / SPLIT_RATIO)


def max_dd(simple: np.ndarray) -> float:
    w = np.cumprod(1 + simple)
    return float((w / np.maximum.accumulate(w) - 1).min())


lab.record("split_naive_day", float(naive_split_day))
lab.record("split_true_day", float(true_split_day))
lab.record("split_naive_cagr", float(np.prod(1 + raw_r) ** (1 / YEARS_PATH) - 1))
lab.record("split_true_cagr", float(np.prod(1 + adj_r) ** (1 / YEARS_PATH) - 1))
lab.record("split_naive_maxdd", max_dd(raw_r))
lab.record("split_true_maxdd", max_dd(adj_r))
lab.record("split_naive_vol", float(np.std(raw_r, ddof=1) * np.sqrt(PERIODS_PER_YEAR)))
lab.record("split_true_vol", float(np.std(adj_r, ddof=1) * np.sqrt(PERIODS_PER_YEAR)))

lab.chart("index_growth", charts.line_chart(
    [charts.Series("cap-weighted (float)", in_years(cw_level / 100), role="strategy", end_label="cap"),
     charts.Series("equal-weighted", in_years(ew_level / 100), role="alt1", end_label="equal"),
     charts.Series("price-weighted", in_years(pw_level / 100), role="alt2", end_label="price")],
    title=f"Same four stocks, three index rules: growth of $1 over {YEARS_PATH} simulated years",
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

# %% [markdown]
# ## 3 · Luck or rule? 2,000 simulated decades
#
# One path cannot tell you whether a weighting rule is better. Simulate 2,000 decades at quarterly
# steps (exact for this model, because between rebalances both indices simply hold their stocks).
#
# **Theory.** With every stock expected to grow at `MU`, *any* fixed rule has the same expected
# wealth: E[terminal wealth] = e^(MU·Y) over Y years, for buy-and-hold cap weighting and for
# quarterly-rebalanced equal weighting alike. What differs is compounding. Session 2.3's volatility drag applies to a whole
# portfolio: a portfolio held at constant weights w compounds at about MU − ½·w'Σw, where Σ is the
# covariance matrix of the stocks. So the rule whose portfolio is *less volatile* compounds faster —
# and every diversified index compounds faster than its average stock (MU − ½σᵢ²).

# %%
N_MC, Q_PER_YEAR = 2_000, 4
DQ = 1 / Q_PER_YEAR
lq = simulate_log_returns(rng_mc, YEARS_PATH * Q_PER_YEAR, DQ, N_MC)   # (quarters, paths, stocks)
G = np.exp(lq)                                                           # quarterly gross returns
w0 = UNIVERSE.w_float.to_numpy()
W_cw = (w0 * np.exp(lq.sum(axis=0))).sum(axis=1)                         # buy and hold
W_ew = np.prod(G.mean(axis=2), axis=0)                                   # 1/N restored each quarter
Y = YEARS_PATH                                   # horizon in years (T is reserved for a number of periods)
theory_EW = np.exp(MU * Y)
for key, W in (("cw", W_cw), ("ew", W_ew)):
    se = W.std(ddof=1) / np.sqrt(N_MC)
    lab.record(f"mc_{key}_mean_wealth", float(W.mean()))
    lab.record(f"mc_{key}_mean_wealth_se", float(se))
    assert abs(W.mean() - theory_EW) < 4 * se, f"{key}: mean terminal wealth disagrees with e^(MU Y)"
lab.record("mc_theory_wealth", float(theory_EW))
g_cw, g_ew = np.log(W_cw) / Y, np.log(W_ew) / Y          # annual log growth of each simulated decade
beta, idio = UNIVERSE.beta.to_numpy(), UNIVERSE.idio.to_numpy()
cov = np.outer(beta, beta) * SIGMA_M ** 2 + np.diag(idio ** 2)
w_eq = np.full(N_STOCKS, 1 / N_STOCKS)
g_ew_theory = MU - 0.5 * w_eq @ cov @ w_eq               # continuous-rebalancing approximation
g_ew_se = g_ew.std(ddof=1) / np.sqrt(N_MC)
# quarterly (not continuous) rebalancing biases this negligibly (< 0.02 pp a year)
assert abs(g_ew.mean() - g_ew_theory) < 4 * g_ew_se
gap = g_cw - g_ew
gap_se = gap.std(ddof=1) / np.sqrt(N_MC)
lab.record("mc_n", N_MC)
lab.record("mc_cw_growth", float(np.expm1(g_cw.mean())))    # CAGR of the typical (log-average) decade
lab.record("mc_ew_growth", float(np.expm1(g_ew.mean())))
lab.record("mc_ew_growth_theory", float(np.expm1(g_ew_theory)))
lab.record("mc_growth_gap_pp", float(100 * gap.mean()))
lab.record("mc_growth_gap_se_pp", float(100 * gap_se))
lab.record("mc_p_ew_beats_cw", float((W_ew > W_cw).mean()))
lab.record("vol_ew", float(np.sqrt(w_eq @ cov @ w_eq)))
lab.record("vol_cw0", float(np.sqrt(w0 @ cov @ w0)))
lab.record("avg_stock_growth", float(np.expm1(MU - 0.5 * (UNIVERSE.sigma ** 2).mean())))
# When one rule clearly compounds faster, it should be the one whose portfolio starts out calmer.
# (In THIS universe that is the cap-weighted index: most of its money sits in Mega, the calmest stock.)
if abs(gap.mean()) > 4 * gap_se:
    assert (gap.mean() > 0) == (w0 @ cov @ w0 < w_eq @ cov @ w_eq)
print(f"typical growth: equal {lab.results['mc_ew_growth']:.2%} (theory {np.expm1(g_ew_theory):.2%}) vs "
      f"cap {lab.results['mc_cw_growth']:.2%} a year; equal beats cap in {lab.results['mc_p_ew_beats_cw']:.0%} of decades")

# %% [markdown]
# ## 4 · Dividends: price return vs total return over 20 years
#
# An index-like asset with an expected **total** return of 8% a year, 18% volatility and a 2%
# dividend yield paid quarterly. On each ex-dividend date the price drops by exactly the dividend
# (this lab's simplifying assumption — real drops are noisier and, on average, a little smaller).
# The **price return** ignores the dividends; the **total return** reinvests each one at the
# ex-date close. Dividends are paid as a fixed fraction of price here, so the gap between the two
# is deterministic: (1 − q/4)^(−4·years).

# %%
YEARS_DIV, MU_TR, SIGMA_TR, DIV_YIELD = 20, 0.08, 0.18, 0.02
EX_EVERY = PERIODS_PER_YEAR // 4                   # an ex-dividend date every quarter
n_div_days = YEARS_DIV * PERIODS_PER_YEAR
z = rng_div.standard_normal(n_div_days)
price = np.empty(n_div_days + 1)
tr = np.empty(n_div_days + 1)
price[0] = tr[0] = 100.0
divs_paid = 0.0
for t in range(1, n_div_days + 1):
    cum = price[t - 1] * np.exp((MU_TR - 0.5 * SIGMA_TR ** 2) * DT + SIGMA_TR * np.sqrt(DT) * z[t - 1])
    D = DIV_YIELD / 4 * cum if t % EX_EVERY == 0 else 0.0   # the dividend detaches on the ex-date
    price[t] = cum - D                                      # price drops by the dividend
    tr[t] = tr[t - 1] * (price[t] + D) / price[t - 1]       # total return: dividend reinvested
    divs_paid += D
gap_theory = (1 - DIV_YIELD / 4) ** (-4 * YEARS_DIV)
assert np.isclose(tr[-1] / price[-1], gap_theory)           # identity, not luck
lab.record("div_years", YEARS_DIV)
lab.record("div_mu", MU_TR)
lab.record("div_sigma", SIGMA_TR)
lab.record("div_yield", DIV_YIELD)
lab.record("div_price_cagr", cagr_of(price, YEARS_DIV))
lab.record("div_tr_cagr", cagr_of(tr, YEARS_DIV))
lab.record("div_price_final", float(price[-1] / price[0]))
lab.record("div_tr_final", float(tr[-1] / tr[0]))
lab.record("div_gap_ratio", float(gap_theory))
lab.record("div_missing_share", float(1 - 1 / gap_theory))   # share of the total-return wealth a price-only view misses
lab.record("div_cagr_gap_pp", 100 * (lab.results["div_tr_cagr"] - lab.results["div_price_cagr"]))
lab.chart("total_vs_price", charts.line_chart(
    [charts.Series("total return (dividends reinvested)", in_years(tr / 100), role="strategy", end_label="total"),
     charts.Series("price return", in_years(price / 100), role="benchmark", end_label="price")],
    title=f"Growth of $1 over {YEARS_DIV} simulated years: total vs price return (log scale)", logy=True,
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

# %% [markdown]
# **Can you see the ex-date drop?** On one stock, on one day, barely: the day's random move is several
# times the dividend. Across many ex-dates it averages out. Simulate 2,000 ex-dates with quarterly
# dividends of 0.25%–1.25% of the price and a daily volatility of 2%, where the price drops by
# θ = 1 × the dividend, and estimate θ by least squares through the origin.

# %%
N_EX, DAILY_SD, THETA = 2_000, 0.02, 1.0
y = rng_exdate.uniform(0.0025, 0.0125, N_EX)                 # dividend as a fraction of the cum price
r_ex = -THETA * y + DAILY_SD * rng_exdate.standard_normal(N_EX)
theta_hat = float(-(r_ex @ y) / (y @ y))
resid = r_ex + theta_hat * y
theta_se = float(np.sqrt(resid @ resid / (N_EX - 1) / (y @ y)))
lab.record("ex_n", N_EX)
lab.record("ex_daily_sd", DAILY_SD)
lab.record("ex_theta_hat", theta_hat)
lab.record("ex_theta_se", theta_se)
lab.record("ex_noise_to_signal", DAILY_SD / float(y.mean()))
assert abs(theta_hat - THETA) < 4 * theta_se
print(f"estimated drop ratio {theta_hat:.2f} ± {theta_se:.2f} from {N_EX:,} ex-dates")

# %% [markdown]
# ## 5 · ETFs: why the price stays near NAV — and when it doesn't
#
# The ETF's market price is P = NAV × (1 + π), where π is the premium (negative: a discount).
# Order flow pushes π around: π_t = φ·π_(t−1) + ε_t (the code calls it p).
# **Authorised participants** (APs) arbitrage any premium beyond their all-in cost c: above +c they *create* (deliver the basket, receive new
# ETF shares, sell them), below −c they *redeem* (buy ETF shares, hand them in for the basket).
# That caps |π| at c. Without APs the ETF behaves like a closed-end fund and π wanders freely.
#
# All parameters are illustrative. Theory for the no-AP case: π is AR(1) with stationary standard
# deviation s/√(1 − φ²).

# %%
PHI, S_BP, C_BP = 0.98, 10.0, 15.0          # persistence, daily shock sd (bp), AP cost band (bp)
N_ETF, ETF_DAYS = 2_000, PERIODS_PER_YEAR
sd_theory = S_BP / np.sqrt(1 - PHI ** 2)


def premium_paths(eps: np.ndarray, band: np.ndarray | None, p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Premium paths (days+1, paths) in bp; band = AP cost per day (None: no APs). Also returns AP actions."""
    n, m = eps.shape
    p = np.empty((n + 1, m))
    act = np.zeros((n, m), dtype=int)           # +1 creation, −1 redemption
    p[0] = p0 if band is None else np.clip(p0, -band[0], band[0])
    for t in range(n):
        x = PHI * p[t] + eps[t]
        if band is not None:
            act[t] = np.where(x > band[t], 1, np.where(x < -band[t], -1, 0))
            x = np.clip(x, -band[t], band[t])    # the AP trade pulls the premium back to the band edge
        p[t + 1] = x
    return p, act


eps = S_BP * rng_etf.standard_normal((ETF_DAYS, N_ETF))
start = sd_theory * rng_etf.standard_normal(N_ETF)            # start from the stationary distribution
p_free, _ = premium_paths(eps, None, start)
p_ap, act = premium_paths(eps, np.full(ETF_DAYS, C_BP), start)
# theory check at the last day (independent across paths)
sd_hat = float(p_free[-1].std(ddof=1))
sd_se = sd_theory / np.sqrt(2 * (N_ETF - 1))
assert abs(sd_hat - sd_theory) < 4 * sd_se
P_WIDE = 50.0                                                 # a "wide" deviation: 50 bp
p_wide_theory = float(2 * norm.sf(P_WIDE / sd_theory))
p_wide_hat = float((np.abs(p_free[-1]) > P_WIDE).mean())
assert abs(p_wide_hat - p_wide_theory) < 4 * np.sqrt(p_wide_theory * (1 - p_wide_theory) / N_ETF)
assert np.abs(p_ap).max() <= C_BP + 1e-9                      # the arbitrage band, by construction
# tracking: the ETF's log return and the NAV's differ only by the change in log(1 + p), so at most ~2c
track_gap = np.abs(np.log1p(p_ap[-1] / 1e4) - np.log1p(p_ap[0] / 1e4))
assert track_gap.max() <= 2 * C_BP / 1e4 * 1.001
lab.record("etf_phi", PHI)
lab.record("etf_s_bp", S_BP)
lab.record("etf_c_bp", C_BP)
lab.record("etf_sd_theory_bp", float(sd_theory))
lab.record("etf_sd_hat_bp", sd_hat)
lab.record("etf_wide_bp", P_WIDE)
lab.record("etf_p_wide_theory", p_wide_theory)
lab.record("etf_p_wide_hat", p_wide_hat)
lab.record("etf_mean_abs_free_bp", float(np.abs(p_free[1:]).mean()))
lab.record("etf_mean_abs_ap_bp", float(np.abs(p_ap[1:]).mean()))
lab.record("etf_ap_days_share", float((act != 0).mean()))
lab.record("etf_track_max_bp", float(track_gap.max() * 1e4))
lab.record("etf_n", N_ETF)

# %% [markdown]
# **A stress episode.** Same model, one year, but for 20 trading days the underlying market seizes
# up: sellers hit the ETF every day (a drift of −20 bp a day in the premium), shocks triple and the
# APs' cost of arbitrage — buying ETF shares, redeeming them for a basket and selling that basket into
# a market with no dealer capacity — jumps from 15 bp to 400 bp. The band widens, so the discount can too.

# %%
STRESS_START, STRESS_LEN = 120, 20
STRESS_C_BP, STRESS_DRIFT_BP, STRESS_SHOCK = 400.0, -20.0, 3.0
band = np.full(ETF_DAYS, C_BP)
band[STRESS_START:STRESS_START + STRESS_LEN] = STRESS_C_BP
shocks = S_BP * rng_etf.standard_normal((ETF_DAYS, 1))
shocks[STRESS_START:STRESS_START + STRESS_LEN] = (STRESS_DRIFT_BP
                                                  + STRESS_SHOCK * shocks[STRESS_START:STRESS_START + STRESS_LEN])
p_stress, act_stress = premium_paths(shocks, band, np.zeros(1))
calm = np.r_[np.arange(1, STRESS_START + 1), np.arange(STRESS_START + STRESS_LEN + 1, ETF_DAYS + 1)]
lab.record("stress_len", STRESS_LEN)
lab.record("stress_c_bp", STRESS_C_BP)
lab.record("stress_max_discount_pct", float(-p_stress.min() / 100))
lab.record("stress_calm_max_abs_bp", float(np.abs(p_stress[calm]).max()))
lab.record("stress_redemption_days", int((act_stress[:, 0] == -1).sum()))
assert np.abs(p_stress[:, 0]).max() <= band.max() + 1e-9   # only the band limits the discount
assert np.abs(p_stress[calm, 0]).max() <= C_BP + 1e-9       # outside the episode APs hold the normal band
band_series = np.r_[band[0], band]
lab.chart("etf_premium", charts.line_chart(
    [charts.Series("premium (+) / discount (−)", in_years(p_stress[:, 0] / 1e4), role="strategy", label_end=False),
     charts.Series("AP band, lower edge", in_years(-band_series / 1e4), role="benchmark", label_end=False)],
    title=f"ETF price vs NAV over one simulated year, with a {STRESS_LEN}-day stress episode",
    y_fmt=charts.fmt_pct(1), hline=0.0))

# %% [markdown]
# ## 6 · Short selling: a worked example
#
# Short 1,000 shares at $50 and buy them back a year later at $45. You were right: +$5,000 on the
# price. But you also owe the lender every dividend ($1.00 a share over the year, as "manufactured"
# dividends) and a borrow fee on the value of the loan. Fees accrue daily on the current value;
# here they are charged on the $50,000 starting value for simplicity. Fee levels are illustrative:
# general-collateral stocks cost tens of basis points a year, "specials" several percent,
# the hardest-to-borrow names far more (see the session page for sources).

# %%
N_SH, P_IN, P_OUT, DIV_PS = 1_000, 50.0, 45.0, 1.00
FEES = {"gc": 0.0025, "special": 0.05, "htb": 0.15}
price_pnl = (P_IN - P_OUT) * N_SH
div_owed = DIV_PS * N_SH
lab.record("ex_price_pnl", price_pnl)
lab.record("ex_div_owed", div_owed)
lab.record("ex_notional", P_IN * N_SH)
lab.record("ex_shares", N_SH)
lab.record("ex_p_in", P_IN)
lab.record("ex_p_out", P_OUT)
for k, f in FEES.items():
    fee = f * P_IN * N_SH
    lab.record(f"ex_fee_{k}_rate", f)
    lab.record(f"ex_fee_{k}", fee)
    lab.record(f"ex_net_{k}", price_pnl - div_owed - fee)
assert lab.results["ex_net_htb"] < 0 < lab.results["ex_net_gc"]    # right on the stock, wrong on the trade
lab.record("ex_loss_htb", -lab.results["ex_net_htb"])
# The US margin arithmetic (Reg T + FINRA Rule 4210, as of 2026): deposit 150% of the short's value
# (the sale proceeds + 50%); a call comes when equity < 30% of the CURRENT value of the shares owed.
REG_T, MAINT = 1.50, 0.30
call_rise = REG_T / (1 + MAINT) - 1                    # 1.5·P0 − P = 0.3·P  ⇒  P = 1.5/1.3 · P0
lab.record("short_call_rise", call_rise)
lab.record("short_regt", REG_T)
lab.record("short_maint", MAINT)

# %% [markdown]
# ## 7 · Short P&L by Monte Carlo: fees, dividends and squeezes
#
# A crowded short: price $50, expected total return 8% a year, 2% dividend yield, 50% volatility,
# a 5% borrow fee. Dividends and fees accrue daily on the current price. Two worlds on 20,000
# one-year paths:
#
# * **A — diffusion only** (geometric Brownian motion);
# * **B — with squeezes**: on top of the diffusion, squeezes arrive at random (on average 0.3 a
#   year) and each lifts the price by 80% in a day; the drift is lowered to compensate, so the
#   *expected* price path is identical in both worlds. Only the tail differs.
#
# **Theory.** With dividend yield q (`Q_S`), borrow fee κ (`FEE_S`) and a = e^((μ − q)/252), the
# expected P&L per share over n days is exact: E[P&L] = P0 − P0·aⁿ − (q + κ)/252 · P0 · Σ_(t<n) aᵗ.
# P(price doubles by year-end) is a normal tail (world A) or a Poisson mixture of normal tails (B).
# P(margin call within the year) in world A is the reflection-principle barrier formula of Session 2.4
# with the Broadie–Glasserman–Kou correction for daily checks.

# %%
P0, MU_S, Q_S, SIGMA_S, FEE_S = 50.0, 0.08, 0.02, 0.50, 0.05
LAMBDA_SQ, JUMP = 0.3, 1.8                            # squeezes per year, price multiple on a squeeze day
N_SHORT = 20_000
n = PERIODS_PER_YEAR
z_s = rng_short.standard_normal((n, N_SHORT))
k_s = rng_short.poisson(LAMBDA_SQ * DT, (n, N_SHORT))  # number of squeezes each day (almost always 0 or 1)
nu_a = (MU_S - Q_S - 0.5 * SIGMA_S ** 2) * DT
nu_b = nu_a - LAMBDA_SQ * (JUMP - 1) * DT             # compensated drift: E[P] is the same as in A
paths = {
    "a": P0 * np.exp(np.vstack([np.zeros(N_SHORT), np.cumsum(nu_a + SIGMA_S * np.sqrt(DT) * z_s, axis=0)])),
    "b": P0 * np.exp(np.vstack([np.zeros(N_SHORT), np.cumsum(nu_b + SIGMA_S * np.sqrt(DT) * z_s
                                                             + np.log(JUMP) * k_s, axis=0)])),
}
a_day = np.exp((MU_S - Q_S) * DT)
theory_pnl = P0 - P0 * a_day ** n - (Q_S + FEE_S) * DT * P0 * np.sum(a_day ** np.arange(n))
barrier = REG_T / (1 + MAINT) * P0                   # price at which the margin call arrives


def p_double(world: str) -> float:
    """P(P_T ≥ 2·P0) after one year: normal tail, mixed over the number of squeezes in world B."""
    m = (MU_S - Q_S - 0.5 * SIGMA_S ** 2)
    if world == "a":
        return float(norm.sf((np.log(2) - m) / SIGMA_S))
    m -= LAMBDA_SQ * (JUMP - 1)
    ks = np.arange(0, 30)
    return float(np.sum(poisson.pmf(ks, LAMBDA_SQ) * norm.sf((np.log(2) - m - ks * np.log(JUMP)) / SIGMA_S)))


def p_touch_up(h_ratio: float, mu: float, sigma: float, tau: float, dt: float) -> float:
    """P(a GBM with drift mu closes at or above h_ratio × its start on some day within tau years)."""
    h = np.log(h_ratio) + 0.5826 * sigma * np.sqrt(dt)     # discrete monitoring ≈ a barrier further away
    nu = mu - 0.5 * sigma ** 2
    s = sigma * np.sqrt(tau)
    return float(norm.sf((h - nu * tau) / s) + np.exp(2 * nu * h / sigma ** 2) * norm.cdf((-h - nu * tau) / s))


short_rows = {}
for world, P in paths.items():
    pnl = P0 - P[-1] - (Q_S + FEE_S) * DT * P[:-1].sum(axis=0)       # per share
    ret = pnl / P0                                                    # return on the initial short value
    called = (P[1:] >= barrier).any(axis=0)                          # price-only test (matches the theory)
    # the real test: accrued fees and dividends reduce the credit balance, so calls come a little sooner
    costs_all = np.vstack([np.zeros(N_SHORT), np.cumsum((Q_S + FEE_S) * DT * P[:-1], axis=0)])
    eq_all = REG_T * P0 - P - costs_all                              # equity per share, every day
    called_net = (eq_all[1:] < MAINT * P[1:]).any(axis=0)
    se = pnl.std(ddof=1) / np.sqrt(N_SHORT)
    assert abs(pnl.mean() - theory_pnl) < 4 * se, f"world {world}: mean P&L disagrees with theory"
    pd_hat, pd_th = float((P[-1] >= 2 * P0).mean()), p_double(world)
    assert abs(pd_hat - pd_th) < 4 * np.sqrt(pd_th * (1 - pd_th) / N_SHORT)
    short_rows[world] = {
        "mean_ret": float(ret.mean()), "mean_ret_se": float(se / P0),
        "p_profit": float((pnl > 0).mean()),
        "p_loss50": float((ret < -0.5).mean()),
        "p_loss100": float((ret < -1.0).mean()),
        "p_double": pd_hat, "p_double_theory": pd_th,
        "p_call": float(called.mean()),
        "p_call_net": float(called_net.mean()),
        "p_right_but_called": float((called_net & (P[-1] < P0)).mean()),
        "worst": float(ret.min()), "best": float(ret.max()),
        "q01": float(np.quantile(ret, 0.01)),
    }
    for key, v in short_rows[world].items():
        lab.record(f"short_{world}_{key}", v)
    short_rows[world]["ret"] = ret
    short_rows[world]["called_net"] = called_net
    short_rows[world]["eq_all"] = eq_all
lab.record("short_theory_ret", float(theory_pnl / P0))
p_call_theory = p_touch_up(barrier / P0, MU_S - Q_S, SIGMA_S, 1.0, DT)
lab.record("short_a_p_call_theory", p_call_theory)
assert abs(short_rows["a"]["p_call"] - p_call_theory) < 4 * np.sqrt(p_call_theory * (1 - p_call_theory) / N_SHORT)
assert short_rows["a"]["best"] < 1.0 and short_rows["b"]["best"] < 1.0   # a short can never make more than 100%
# Squeezes make a short win MORE often and lose BIGGER: the same mean, a fatter left tail.
pa, pb = short_rows["a"]["p_profit"], short_rows["b"]["p_profit"]
assert pb - pa > 4 * np.sqrt(pa * (1 - pa) / N_SHORT + pb * (1 - pb) / N_SHORT)
# The fatter tail, tested on the price alone so the borrow fee cannot blur it: P(price doubles).
# (p_loss100 also includes fees and dividends; it is reported, not asserted.)
da, db = short_rows["a"]["p_double"], short_rows["b"]["p_double"]
assert db - da > 4 * np.sqrt(da * (1 - da) / N_SHORT + db * (1 - db) / N_SHORT)
lab.record("short_p0", P0)
lab.record("short_mu", MU_S)
lab.record("short_q", Q_S)
lab.record("short_sigma", SIGMA_S)
lab.record("short_fee", FEE_S)
lab.record("short_lambda", LAMBDA_SQ)
lab.record("short_jump", JUMP - 1)
lab.record("short_n", N_SHORT)
lab.record("short_barrier", float(barrier))
lab.record("short_p_any_squeeze", float(1 - np.exp(-LAMBDA_SQ)))
lab.record("short_p_any_squeeze_hat", float((k_s.sum(axis=0) > 0).mean()))
print(pd.DataFrame({w: {k: v for k, v in r.items() if k not in ("ret", "called_net", "eq_all")}
                    for w, r in short_rows.items()}).round(4))

RET_FLOOR = -3.0                                      # display floor: bigger losses are shown in the last bin
ret_b = np.maximum(short_rows["b"]["ret"], RET_FLOOR)
lab.record("short_b_share_below_floor", float((short_rows["b"]["ret"] < RET_FLOOR).mean()))
lab.chart("short_hist", charts.histogram(
    ret_b, bins=64, x_fmt=charts.fmt_pct(0), tail_below=-1.0, tail_label="lost more than the initial value",
    title=f"One-year return of a short, world B ({N_SHORT:,} paths; losses below {RET_FLOOR:.0%} "
          f"shown at {RET_FLOOR:.0%})".replace("-", "−")))

# %% [markdown]
# **One squeeze, up close.** Pick (deliberately — an illustration, not a typical path) the first
# world-B path on which the short is *winning* (price below entry) when a squeeze hits in the second
# quarter, the squeeze itself triggers the first margin call, and the price still ends the year below
# the entry price by at least 10%: right about the stock, and forced out anyway if there was no spare cash.
# Calls are judged on equity net of accrued fees and dividends, the same basis as the assert below.
# If no path qualifies (possible on other seeds), the criteria are relaxed step by step.
# Plot the account's equity (if you could hold on) against the maintenance requirement.

# %%
Pb = paths["b"]
sq_day = np.argmax(k_s > 0, axis=0)                   # index of the first squeeze step (0 if none)
has_sq = k_s.sum(axis=0) > 0
cols = np.arange(N_SHORT)
eq_b, called_b = short_rows["b"]["eq_all"], short_rows["b"]["called_net"]
first_call_all = np.argmax(eq_b[1:] < MAINT * Pb[1:], axis=0) + 1   # first day below maintenance (net)
squeeze_calls = has_sq & called_b & (first_call_all == sq_day + 1)  # the first squeeze day is the first call
TIERS = [
    squeeze_calls & (sq_day >= n // 4) & (sq_day < n // 2) & (Pb[sq_day, cols] < P0) & (Pb[-1] < 0.9 * P0),
    squeeze_calls & (Pb[-1] < P0),                                   # relaxed: just ends below entry
    squeeze_calls,                                                   # relaxed: any squeeze-triggered call
]
tier = next((k for k, t in enumerate(TIERS) if t.any()), None)
assert tier is not None, "no squeeze triggered a margin call on any path: raise N_SHORT or LAMBDA_SQ"
cands = np.flatnonzero(TIERS[tier])
j = int(cands[0])
costs = np.r_[0.0, np.cumsum((Q_S + FEE_S) * DT * Pb[:-1, j])]
equity = REG_T * P0 - Pb[:, j] - costs                # credit balance minus the value of the shares owed
maint_req = MAINT * Pb[:, j]
call_day = int(first_call_all[j])
SQ_SHARES = 100                                       # the chart shows a 100-share position
assert equity[call_day] < maint_req[call_day] and (equity[:call_day] >= maint_req[:call_day]).all()
lab.record("sq_path_day", call_day)
lab.record("sq_path_pre", float(Pb[call_day - 1, j] / P0 - 1))           # price vs entry the day before
lab.record("sq_path_jump_day_ret", float(Pb[call_day, j] / Pb[call_day - 1, j] - 1))
lab.record("sq_path_bought_in_ret", float(equity[call_day] / P0 - (REG_T - 1)))  # P&L / stake if closed at the call
lab.record("sq_path_end", float(Pb[-1, j] / P0 - 1))
lab.record("sq_path_pre_drop", float(1 - Pb[call_day - 1, j] / P0))     # the same two, as positive falls
lab.record("sq_path_end_drop", float(1 - Pb[-1, j] / P0))
lab.record("sq_path_held_ret", float(equity[-1] / P0 - (REG_T - 1)))
lab.record("sq_path_n_candidates", int(cands.size))
lab.record("sq_path_tier", tier)
lab.record("sq_shares", SQ_SHARES)
days = np.arange(n + 1)
lab.chart("squeeze_path", charts.line_chart(
    [charts.Series("equity, if you could hold on", pd.Series(equity * SQ_SHARES, index=days), role="strategy",
                   end_label="equity"),
     charts.Series("maintenance requirement", pd.Series(maint_req * SQ_SHARES, index=days), role="loss",
                   end_label="maintenance")],
    title=f"Short {SQ_SHARES} shares at ${P0:.0f} with {REG_T:.0%} collateral: one squeeze path (x = trading days)",
    y_fmt=charts.fmt_usd_compact(1)))

# %%
lab.save()
