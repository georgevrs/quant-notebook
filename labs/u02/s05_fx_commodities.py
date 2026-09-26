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
# # 2.5 · FX & Commodities — companion lab
#
# **Quant Notebook** · Unit 2 · Session 5 · [Read the session](https://georgevrs.github.io/quant-notebook/unit02-markets-instruments/session05-fx-commodities.html)
#
# Currency quoting, forwards and covered interest parity; commodity term structures, storage and convenience yield.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# All quotes, interest rates, costs and model parameters below are **illustrative** round numbers,
# not market data. Simulations use synthetic data; time is elapsed time (12 months = 1 year),
# never a calendar date. The carry-trade returns in section 4 are already excess returns (you borrow
# one currency to hold another), so no separate risk-free rate is subtracted.

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
from scipy.integrate import quad
from scipy.stats import norm, poisson

import quantnb as qn
from quantnb import charts

lab = qn.Lab("2.5")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
rng_cip, rng_carry = lab.rng.spawn(2)

# %% [markdown]
# ## 1 · Reading a quote: base, quote, pips and crosses
#
# `EUR/USD 1.0850 / 1.0851` means: one euro (the **base** currency) costs 1.0850 US dollars (the
# **quote** currency) if you sell euros to the dealer (the bid) and 1.0851 if you buy them (the ask).
# A **pip** is the fourth decimal for most pairs and the second for yen pairs. Pairs that do not
# involve the dollar (**crosses**) are usually built from two dollar pairs, so you pay two spreads.

# %%
PIP = {"EURUSD": 1e-4, "USDJPY": 1e-2, "EURJPY": 1e-2}   # size of one pip, in quote-currency units
QUOTES = {                                               # illustrative dealer quotes: (bid, ask)
    "EURUSD": (1.0850, 1.0851),
    "USDJPY": (150.00, 150.02),
}
NOTIONAL_BASE = 1_000_000                                # "one million" of the base currency


def spread_pips(pair: str, bid: float, ask: float) -> float:
    """Bid-ask spread measured in pips of that pair."""
    return (ask - bid) / PIP[pair]


def rel_spread(bid: float, ask: float) -> float:
    """Spread as a fraction of the mid price (what a round trip costs you)."""
    return (ask - bid) / ((ask + bid) / 2)


eu_b, eu_a = QUOTES["EURUSD"]
uj_b, uj_a = QUOTES["USDJPY"]
# Cross EUR/JPY: to buy euros with yen you buy dollars with yen (at the USD/JPY ask) and then buy
# euros with those dollars (at the EUR/USD ask) — so the cross ask multiplies the two asks.
ej_b, ej_a = eu_b * uj_b, eu_a * uj_a
QUOTES["EURJPY"] = (ej_b, ej_a)
for pair, (b, a) in QUOTES.items():
    lab.record(f"{pair.lower()}_bid", b)
    lab.record(f"{pair.lower()}_ask", a)
    lab.record(f"{pair.lower()}_spread_pips", spread_pips(pair, b, a))
    lab.record(f"{pair.lower()}_spread_bp", 1e4 * rel_spread(b, a))
    print(f"{pair}: {b:.4f} / {a:.4f}  spread {spread_pips(pair, b, a):.2f} pips = {1e4 * rel_spread(b, a):.2f} bp")

# a cross costs (almost exactly) the sum of the two relative spreads: (1 + s1)(1 + s2) − 1 on the asks
assert abs(rel_spread(ej_b, ej_a) - (rel_spread(eu_b, eu_a) + rel_spread(uj_b, uj_a))) < 1e-6
assert rel_spread(ej_b, ej_a) > max(rel_spread(eu_b, eu_a), rel_spread(uj_b, uj_a))

# What one pip is worth on one million of the base currency, in dollars
uj_mid = (uj_b + uj_a) / 2
pip_usd = {
    "EURUSD": NOTIONAL_BASE * PIP["EURUSD"],               # already in dollars
    "USDJPY": NOTIONAL_BASE * PIP["USDJPY"] / uj_mid,      # yen → dollars
    "EURJPY": NOTIONAL_BASE * PIP["EURJPY"] / uj_mid,
}
for pair, v in pip_usd.items():
    lab.record(f"{pair.lower()}_pip_usd_per_mio", v)
lab.record("usdjpy_pip_jpy_per_mio", NOTIONAL_BASE * PIP["USDJPY"])
lab.record("usdcad_pip_cad_per_mio", NOTIONAL_BASE * 1e-4)          # USD/CAD: a 4-decimal pair, priced in CAD
lab.record("notional_base", NOTIONAL_BASE)
lab.record("usdjpy_mid", uj_mid)
# a round trip (buy at the ask, sell at the bid) of €10 million EUR/USD
lab.record("roundtrip_10m_usd", 10 * NOTIONAL_BASE * (eu_a - eu_b))

# %% [markdown]
# ## 2 · FX forwards and covered interest parity (CIP)
#
# Convention for the whole lab: a pair BASE/QUOTE, the price S = units of QUOTE per 1 BASE,
# i_q = the quote currency's interest rate, i_b = the base currency's, both **simple money-market
# rates** (ACT/360 for USD and EUR, so τ = days/360). Two routes turn dollars today into dollars
# at τ: deposit them at i_q, or convert to euros, deposit at i_b and sell the euros forward.
# No arbitrage makes the two routes pay the same:
#
#     F = S · (1 + i_q τ) / (1 + i_b τ)
#
# The higher-rate currency trades at a forward discount. **Forward points** = (F − S)/pip.
# With continuous rates this is Session 2.4's F = S·exp((r_f − q)τ), with r_f = i_q and q = i_b.

# %%
S_EURUSD = (eu_b + eu_a) / 2     # mid
I_USD, I_EUR = 0.04, 0.02        # illustrative money-market rates (simple, ACT/360)
# ACT/360 counts ACTUAL calendar days ÷ 360, so a one-year forward spans ~365 days, τ ≈ 1.014.
# Actual days from an illustrative spot date (calendars differ by a day or two month to month).
TENORS = {"1m": 31, "3m": 92, "6m": 183, "1y": 365}   # actual days to the forward date


def cip_forward(spot: float, i_quote: float, i_base: float, tau: float) -> float:
    """Covered-interest-parity forward: QUOTE per 1 BASE, simple money-market rates, tau in years."""
    return spot * (1 + i_quote * tau) / (1 + i_base * tau)


lab.record("s_eurusd", S_EURUSD)
lab.record("i_usd", I_USD)
lab.record("i_eur", I_EUR)
for name, days in TENORS.items():
    tau = days / 360
    F = cip_forward(S_EURUSD, I_USD, I_EUR, tau)
    pts = (F - S_EURUSD) / PIP["EURUSD"]
    lab.record(f"fwd_{name}", F)
    lab.record(f"fwd_{name}_points", pts)
    lab.record(f"fwd_{name}_days", days)
    print(f"{name}: F = {F:.5f}  forward points {pts:+.1f}")
    assert F > S_EURUSD    # the euro has the LOWER rate, so it trades at a forward premium

# rule of thumb: forward points ≈ S (i_q − i_b) τ / pip; and the continuous-rate version
tau_1y = TENORS["1y"] / 360
approx_pts = S_EURUSD * (I_USD - I_EUR) * tau_1y / PIP["EURUSD"]
cont_pts = (S_EURUSD * np.exp((I_USD - I_EUR) * tau_1y) - S_EURUSD) / PIP["EURUSD"]
lab.record("fwd_1y_points_approx", approx_pts)
lab.record("fwd_1y_points_cont", cont_pts)
# Plugging money-market (simple) rates into the continuous formula misprices the forward:
lab.record("fwd_1y_cont_error_pips", cont_pts - lab.results["fwd_1y_points"])

# Check-yourself question 1: sell ¥300 million for dollars. You BUY the base (USD), so you pay the ask.
Q1_JPY = 300e6
lab.record("q1_jpy", Q1_JPY)
lab.record("q1_usd_at_ask", Q1_JPY / uj_a)
lab.record("q1_halfspread_usd", Q1_JPY / uj_mid - Q1_JPY / uj_a)

# %% [markdown]
# **The CIP loop, in cash flows.** If the market forward is ABOVE the CIP price: borrow dollars,
# buy euros spot, deposit them, and sell the euros forward. Everything is fixed today, so the profit
# must not depend on where EUR/USD ends up. We check that by brute force on simulated end-rates.

# %%
TAU3 = TENORS["3m"] / 360
F_CIP_3M = cip_forward(S_EURUSD, I_USD, I_EUR, TAU3)
DEVIATION_PIPS = 8.0                                   # the market forward sits this far above CIP
F_MKT_3M = F_CIP_3M + DEVIATION_PIPS * PIP["EURUSD"]
USD_NOTIONAL = 100e6                                   # dollars borrowed


def implied_basis(spot: float, fwd: float, i_quote: float, i_base: float, tau: float) -> float:
    """Cross-currency basis x (Du, Tepper & Verdelhan's sign): F = S(1 + i_q τ)/(1 + (i_b + x) τ)."""
    return ((spot * (1 + i_quote * tau) / fwd) - 1) / tau - i_base


def loop_profit(usd: float, spot_ask: float, fwd_bid: float, i_q_borrow: float, i_b_lend: float,
                tau: float) -> float:
    """Borrow USD, buy EUR spot at the ask, deposit EUR, sell EUR forward at the bid: USD profit at τ."""
    eur = usd / spot_ask * (1 + i_b_lend * tau)       # euros you will have at τ
    return eur * fwd_bid - usd * (1 + i_q_borrow * tau)


x_mkt = implied_basis(S_EURUSD, F_MKT_3M, I_USD, I_EUR, TAU3)
gross = loop_profit(USD_NOTIONAL, S_EURUSD, F_MKT_3M, I_USD, I_EUR, TAU3)   # everything at mid
lab.record("f_cip_3m", F_CIP_3M)
lab.record("f_mkt_3m", F_MKT_3M)
lab.record("dev_pips", DEVIATION_PIPS)
lab.record("basis_bp", 1e4 * x_mkt)
lab.record("usd_notional", USD_NOTIONAL)
lab.record("gross_profit", gross)
lab.record("gross_profit_bp_pa", 1e4 * gross / USD_NOTIONAL / TAU3)
assert (x_mkt < 0) == (DEVIATION_PIPS > 0)   # forward too high ⇔ negative basis
assert abs(loop_profit(USD_NOTIONAL, S_EURUSD, F_CIP_3M, I_USD, I_EUR, TAU3)) < 1e-6   # zero at CIP

# Brute force: whatever EUR/USD does over the 3 months, the covered loop pays the same.
N_SCENARIOS = 10_000
S_T = S_EURUSD * np.exp(rng_cip.normal(0.0, 0.08 * np.sqrt(TAU3), N_SCENARIOS))
lab.record("cip_mc_n", N_SCENARIOS)
eur_T = USD_NOTIONAL / S_EURUSD * (1 + I_EUR * TAU3)
package = eur_T * S_T + eur_T * (F_MKT_3M - S_T) - USD_NOTIONAL * (1 + I_USD * TAU3)  # deposit + short forward − loan
assert np.allclose(package, gross), "the covered loop's profit must be the same in every scenario"
uncovered = eur_T * S_T - USD_NOTIONAL * (1 + I_USD * TAU3)    # drop the forward: now it is a currency bet
lab.record("uncovered_sd", float(uncovered.std()))
lab.record("uncovered_sd_m", float(uncovered.std()) / 1e6)       # in millions: no false precision
print(f"CIP 3m {F_CIP_3M:.5f}, market {F_MKT_3M:.5f}: basis {1e4 * x_mkt:.1f} bp, "
      f"gross profit ${gross:,.0f} on ${USD_NOTIONAL / 1e6:.0f}m; uncovered sd ${uncovered.std():,.0f}")

# %% [markdown]
# ## 3 · Arbitrage net of costs: who can actually trade it?
#
# Real traders buy at the ask, sell at the bid, borrow above and lend below the mid rate. A bank
# also pays for the balance sheet the trade uses: Du, Tepper & Verdelhan's back-of-envelope is a 3%
# leverage ratio × a 10% required return on equity ≈ 30 bp a year. The costs turn the single CIP
# price into a **no-arbitrage band**; only a forward outside the band pays.
#
#     lower = S_bid (1 + i_q,lend τ)/(1 + i_b,borrow τ)   ≤   F   ≤   S_ask (1 + i_q,borrow τ)/(1 + i_b,lend τ) = upper

# %%
TIERS = {  # illustrative costs: half-spreads in pips, rate spreads (vs mid) and balance-sheet charge per year
    "bank": dict(spot_half=0.1, points_half=0.1, borrow=0.0002, lend=0.0002, balance_sheet=0.03 * 0.10),
    "fund": dict(spot_half=0.25, points_half=0.25, borrow=0.0025, lend=0.0010, balance_sheet=0.0),
    "retail": dict(spot_half=1.0, points_half=1.0, borrow=0.0300, lend=0.0150, balance_sheet=0.0),
}


def no_arb_band(spot_mid: float, t: dict, i_q: float, i_b: float, tau: float) -> tuple[float, float]:
    """Lower and upper forward prices between which no covered loop is profitable for this trader."""
    s_bid = spot_mid - t["spot_half"] * PIP["EURUSD"]
    s_ask = spot_mid + t["spot_half"] * PIP["EURUSD"]
    bs = t["balance_sheet"]                                   # borrowing expands the balance sheet
    upper = s_ask * (1 + (i_q + t["borrow"] + bs) * tau) / (1 + (i_b - t["lend"]) * tau)
    lower = s_bid * (1 + (i_q - t["lend"]) * tau) / (1 + (i_b + t["borrow"] + bs) * tau)
    return lower, upper


tier_rows = []
for name, t in TIERS.items():
    lo, hi = no_arb_band(S_EURUSD, t, I_USD, I_EUR, TAU3)
    # outright = spot + points, so its half-spread is spot's plus the points'. (Dealers usually run
    # the loop as ONE FX swap, paying only the points spread; paying the spot spread twice, as
    # here, slightly overstates the cost — it does not change which traders profit.)
    fwd_bid = F_MKT_3M - (t["spot_half"] + t["points_half"]) * PIP["EURUSD"]
    s_ask = S_EURUSD + t["spot_half"] * PIP["EURUSD"]
    net_spreads = loop_profit(USD_NOTIONAL, s_ask, fwd_bid, I_USD + t["borrow"], I_EUR - t["lend"], TAU3)
    bs_cost = USD_NOTIONAL * t["balance_sheet"] * TAU3
    net = net_spreads - bs_cost
    tier_rows.append({"tier": name, "upper_pips": (hi - F_CIP_3M) / PIP["EURUSD"],
                      "lower_pips": (lo - F_CIP_3M) / PIP["EURUSD"], "net_spreads": net_spreads,
                      "bs_cost": bs_cost, "net": net})
    lab.record(f"{name}_upper_pips", (hi - F_CIP_3M) / PIP["EURUSD"])
    lab.record(f"{name}_net_spreads", net_spreads)
    lab.record(f"{name}_bs_cost", bs_cost)
    lab.record(f"{name}_net", net)
    lab.record(f"{name}_loss", -net)                       # positive when the loop loses money
    for k in ("spot_half", "points_half", "borrow", "lend", "balance_sheet"):
        lab.record(f"{name}_{k}", t[k])
    for k in ("borrow", "lend", "balance_sheet"):
        lab.record(f"{name}_{k}_bp", 1e4 * t[k])                 # the same, in basis points
    # the band and the profit tell the same story: the loop pays iff the forward bid clears the upper bound
    assert (net > 0) == (fwd_bid > hi)
    assert lo < F_CIP_3M < hi
tiers = pd.DataFrame(tier_rows).set_index("tier")
print(tiers.round(1))
lab.record("bs_cost_bp_pa", 1e4 * TIERS["bank"]["balance_sheet"])
# At mid prices a forward above CIP is always a clean arbitrage (the per-tier asserts above check
# that each trader profits exactly when the forward bid clears the top of their band).
assert (gross > 0) == (DEVIATION_PIPS > 0)
# The bank's band is almost all balance sheet: without the 30 bp charge it would be far narrower.
_, hi_no_bs = no_arb_band(S_EURUSD, {**TIERS["bank"], "balance_sheet": 0.0}, I_USD, I_EUR, TAU3)
bank_no_bs_pips = (hi_no_bs - F_CIP_3M) / PIP["EURUSD"]
lab.record("bank_upper_pips_no_bs", bank_no_bs_pips)
assert tiers.loc["bank", "upper_pips"] - bank_no_bs_pips > bank_no_bs_pips

# %% [markdown]
# ## 4 · FX carry: steady gains, occasional crashes — Monte Carlo
#
# A **carry trade** borrows a low-rate currency and holds a high-rate one. Under CIP this is the same
# as buying the high-rate currency forward at its discount, so the carry is the interest differential.
# Whether you keep it depends on the exchange rate. A stylised model, monthly:
#
#     log return = (δ + g − σ²/2)Δ + σ√Δ·z + (sum of N crash jumps),  N ~ Poisson(λΔ),  jump ~ N(μ_J, s_J²)
#
# δ = carry, g = the currency's drift between crashes, λ = crashes per year. All the moments below
# have closed forms (the gross return e^r is a Poisson mixture of lognormals), so we can test the
# simulation against theory. We compare it with a **Gaussian twin**: same mean and volatility, no crashes.

# %%
CARRY = 0.05                         # δ: interest differential per year (illustrative)
CALM_DRIFT = 0.0                     # g: the high-yield currency's drift between crashes
SIGMA_FX = 0.09                      # everyday volatility of the exchange rate
CRASH_RATE = 0.20                    # λ: crashes per year — one every five years on average
CRASH_MEAN, CRASH_SD = -0.10, 0.04   # log size of one crash
N_PATHS, YEARS_C = 10_000, 20
DT = 1 / 12
N_MONTHS = 12 * YEARS_C
M_LOG = CARRY + CALM_DRIFT - 0.5 * SIGMA_FX ** 2


def gross_moment(k: int, lam: float = CRASH_RATE) -> float:
    """E[(1 + R)^k] for one month: E[exp(k r)] of the jump-diffusion log return."""
    jump = lam * DT * (np.exp(k * CRASH_MEAN + 0.5 * k ** 2 * CRASH_SD ** 2) - 1)
    return float(np.exp(k * M_LOG * DT + 0.5 * k ** 2 * SIGMA_FX ** 2 * DT + jump))


def theory_stats(lam: float = CRASH_RATE) -> dict[str, float]:
    """Mean, sd, skewness and annualised Sharpe of MONTHLY simple returns, from the raw moments."""
    m1, m2, m3 = (gross_moment(k, lam) for k in (1, 2, 3))
    var = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    return {"mean": m1 - 1, "sd": np.sqrt(var), "skew": mu3 / var ** 1.5,
            "sharpe": (m1 - 1) / np.sqrt(var) * np.sqrt(12)}


def p_log_below(x: float, lam: float = CRASH_RATE, n_max: int = 30) -> float:
    """P(monthly log return < x): a Poisson mixture of normals."""
    n = np.arange(n_max)
    w = poisson.pmf(n, lam * DT)
    return float(np.sum(w * norm.cdf((x - M_LOG * DT - n * CRASH_MEAN) / np.sqrt(SIGMA_FX ** 2 * DT + n * CRASH_SD ** 2))))


# --- simulate: rows = months, columns = paths ---------------------------------------------------
z = rng_carry.standard_normal((N_MONTHS, N_PATHS))
n_jumps = rng_carry.poisson(CRASH_RATE * DT, (N_MONTHS, N_PATHS))
jumps = n_jumps * CRASH_MEAN + np.sqrt(n_jumps) * CRASH_SD * rng_carry.standard_normal((N_MONTHS, N_PATHS))
log_r = M_LOG * DT + SIGMA_FX * np.sqrt(DT) * z + jumps
R = np.expm1(log_r)                                  # monthly simple excess returns

th = theory_stats()
th_calm = theory_stats(lam=0.0)
# Gaussian twin: normal monthly SIMPLE returns with the same mean and sd
R_twin = th["mean"] + th["sd"] * rng_carry.standard_normal((N_MONTHS, N_PATHS))


def skew(x: np.ndarray) -> float:
    d = x - x.mean()
    return float((d ** 3).mean() / (d ** 2).mean() ** 1.5)


# Every Monte Carlo result is asserted against theory at 4 standard errors. With a dozen asserts,
# 3 SE would fail on a few per cent of seeds by chance alone (a multiple-testing problem).
SE_TOL = 4
lab.record("se_tol", SE_TOL)

# Standard errors by batch means: split the paths into 50 batches, estimate in each, take sd/√50.
N_BATCH = 50
batches = np.array_split(np.arange(N_PATHS), N_BATCH)


def batch_se(stat, x: np.ndarray) -> float:
    vals = np.array([stat(x[:, b]) for b in batches])
    return float(vals.std(ddof=1) / np.sqrt(N_BATCH))


mc_mean, mc_mean_se = float(R.mean()), float(R.std(ddof=1) / np.sqrt(R.size))
mc_skew, mc_skew_se = skew(R), batch_se(skew, R)
mc_sd = float(R.std(ddof=1))
mc_sharpe = mc_mean / mc_sd * np.sqrt(12)
print(f"mean {mc_mean:.5f} (theory {th['mean']:.5f} ± {mc_mean_se:.5f}) · skew {mc_skew:.3f} "
      f"(theory {th['skew']:.3f} ± {mc_skew_se:.3f}) · Sharpe {mc_sharpe:.3f} (theory {th['sharpe']:.3f})")
assert abs(mc_mean - th["mean"]) < SE_TOL * mc_mean_se
assert abs(mc_skew - th["skew"]) < SE_TOL * mc_skew_se
assert abs(float(R.std(ddof=1)) - th["sd"]) < SE_TOL * batch_se(lambda a: float(a.std(ddof=1)), R)

# --- crash months: how often is a month worse than 3 standard deviations below the mean? ---------
THRESH = th["mean"] - 3 * th["sd"]
p_tail_th = p_log_below(np.log1p(THRESH))
p_tail_mc = float((R < THRESH).mean())
p_tail_se = np.sqrt(p_tail_th * (1 - p_tail_th) / R.size)
p_tail_twin = float((R_twin < THRESH).mean())
assert abs(p_tail_mc - p_tail_th) < SE_TOL * p_tail_se

# --- a five-year track record with no crash in it --------------------------------------------------
WINDOW_Y = 5
w = 12 * WINDOW_Y
calm = n_jumps[:w].sum(axis=0) == 0                              # paths whose first 5 years had no crash
p_calm_th = float(np.exp(-CRASH_RATE * WINDOW_Y))
p_calm_se = np.sqrt(p_calm_th * (1 - p_calm_th) / N_PATHS)
assert abs(calm.mean() - p_calm_th) <= SE_TOL * p_calm_se + 1e-12
R_calm = R[:w, calm]
calm_sharpe = float(R_calm.mean() / R_calm.std(ddof=1) * np.sqrt(12))  # pooled over all calm windows
calm_se = qn.stats.sharpe_se(calm_sharpe, R_calm.size / 12, periods_per_year=12)
assert abs(calm_sharpe - th_calm["sharpe"]) < SE_TOL * calm_se
# ...and the typical single calm window: its own 5-year Sharpe, path by path
per_path_calm = R_calm.mean(axis=0) / R_calm.std(axis=0, ddof=1) * np.sqrt(12)

# --- growth and drawdowns, path by path ------------------------------------------------------------
def max_dd(simple: np.ndarray) -> np.ndarray:
    """Maximum drawdown of each column's wealth path: a NEGATIVE fraction, as Session 2.3 defines it."""
    wealth = np.vstack([np.ones(simple.shape[1]), np.cumprod(1 + simple, axis=0)])
    return (wealth / np.maximum.accumulate(wealth, axis=0)).min(axis=0) - 1


log_growth = log_r.sum(axis=0) / YEARS_C                          # per path, per year
g_th = M_LOG + CRASH_RATE * CRASH_MEAN                            # E[log growth per year]
assert abs(log_growth.mean() - g_th) < SE_TOL * log_growth.std(ddof=1) / np.sqrt(N_PATHS)
mdd, mdd_twin = max_dd(R), max_dd(R_twin)
worst_month = R.min(axis=0)
worst_month_twin = R_twin.min(axis=0)

for k, v in {"carry": CARRY, "calm_drift": CALM_DRIFT, "sigma_fx": SIGMA_FX, "crash_rate": CRASH_RATE,
             "crash_mean": CRASH_MEAN, "crash_sd": CRASH_SD, "n_paths": N_PATHS, "years_c": YEARS_C}.items():
    lab.record(k, v)
if CRASH_RATE > 0:
    lab.record("crash_every", 1 / CRASH_RATE)                   # average years between crashes
lab.record("carry_cagr_typical", float(np.expm1(g_th)))           # growth is CAGR
lab.record("carry_cagr_median_mc", float(np.expm1(np.median(log_growth))))
lab.record("carry_mean_m", mc_mean)
lab.record("carry_mean_m_theory", th["mean"])
lab.record("carry_vol_a", mc_sd * np.sqrt(12))
lab.record("carry_sharpe", mc_sharpe)
lab.record("carry_sharpe_theory", th["sharpe"])
lab.record("carry_skew", mc_skew)
lab.record("carry_skew_theory", th["skew"])
lab.record("carry_skew_se", mc_skew_se)
lab.record("twin_skew", skew(R_twin))
lab.record("twin_sharpe", float(R_twin.mean() / R_twin.std(ddof=1) * np.sqrt(12)))
lab.record("tail_thresh", THRESH)
lab.record("p_tail_mc", p_tail_mc)
lab.record("p_tail_theory", p_tail_th)
lab.record("p_tail_twin", p_tail_twin)
lab.record("p_tail_normal", float(norm.cdf(-3)))
lab.record("tail_ratio", p_tail_mc / float(norm.cdf(-3)))
lab.record("p_calm_5y", float(calm.mean()))
lab.record("p_calm_5y_theory", p_calm_th)
lab.record("calm_sharpe", calm_sharpe)
lab.record("calm_sharpe_theory", th_calm["sharpe"])
lab.record("calm_sharpe_path_median", float(np.median(per_path_calm)))
lab.record("mdd_median", float(np.median(mdd)))                  # drawdowns are negative numbers
lab.record("mdd_worst5", float(np.quantile(mdd, 0.05)))           # the worst 5% of paths
lab.record("mdd_twin_median", float(np.median(mdd_twin)))
lab.record("mdd_twin_worst5", float(np.quantile(mdd_twin, 0.05)))
lab.record("window_y", WINDOW_Y)
lab.record("worst_month_median", float(np.median(worst_month)))
lab.record("worst_month_twin_median", float(np.median(worst_month_twin)))
lab.record("p_loss_20y", float((log_growth < 0).mean()))
# What a calm five-year record teaches you about drawdowns, versus what the next 20 years deliver
mdd_calm5 = max_dd(R_calm)
lab.record("mdd_calm5_median", float(np.median(mdd_calm5)))
lab.record("mdd_after_calm_median", float(np.median(mdd[calm])))
# Same mean and volatility, so over 20 years the twin's drawdowns are similar in depth; the crash
# model's losses arrive in single months instead (its median worst month is far deeper).
if CRASH_RATE * YEARS_C >= 1:                      # only meaningful if a typical path sees a crash
    assert np.median(worst_month) < np.median(worst_month_twin) - 0.02

# How often is a month worse than a given loss? Expressed as months per century (× 1,200).
LOSSES = [-0.04, -0.08, -0.12, -0.16]
per_century, per_century_twin = [], []
for x in LOSSES:
    p_th = p_log_below(np.log1p(x))
    p_mc = float((R < x).mean())
    assert abs(p_mc - p_th) < SE_TOL * np.sqrt(p_th * (1 - p_th) / R.size) + 1e-12
    p_twin = float(norm.cdf((x - th["mean"]) / th["sd"]))          # the twin is exactly normal
    per_century.append(1200 * p_mc)
    per_century_twin.append(1200 * p_twin)
    tag = f"{-int(round(100 * x))}"
    lab.record(f"century_{tag}", 1200 * p_mc)
    lab.record(f"century_{tag}_theory", 1200 * p_th)
    lab.record(f"century_{tag}_twin", 1200 * p_twin)
lab.chart("carry_tail", charts.grouped_column_chart(
    [f"worse than {x:.0%}".replace("-", "−") for x in LOSSES],
    [("carry with crashes", per_century, "strategy"), ("Gaussian twin", per_century_twin, "benchmark")],
    title="Months per century worse than each loss: carry with crashes vs its Gaussian twin",
    y_fmt=charts.fmt_num(1), value_labels=True))

# %% [markdown]
# ## 5 · Storage and convenience yield: a simple storage model
#
# Holding a barrel costs financing r_f and storage u (both per year, as fractions of spot) and pays
# a **convenience yield** y: the value of having the physical when you need it. The theory of
# storage (Kaldor 1939, Working 1949, Brennan 1958) says y is high when inventories are scarce and
# near zero when they are ample. Gorton, Hayashi & Rouwenhorst (2013) find a decreasing, non-linear
# relation between convenience yield and inventories in 31 commodities; the exponential below is
# our teaching choice of such a curve, not their estimate. Model:
#
#     y(I) = Y_MAX · exp(−I / I_SCALE),   I = inventory ÷ its normal level
#
# and inventories expected to drift back to normal with a half-life. The futures curve integrates
# the net carry along the expected inventory path:  F(τ) = S · exp(∫₀^τ [r_f + u − y(I(s))] ds).

# %%
# Session 2.4's crude-oil inputs, copied so the two sessions tell one story. They live in
# labs/u02/s04_futures.py, CARRY_CASES["glut"] and ["tight"]: keep these in step if those change.
S04_RF, S04_STORAGE = 0.04, 0.05                 # r_f and u of both 2.4 crude cases
S04_Y = {"glut24": 0.01, "tight24": 0.21}        # 2.4's convenience yields: tanks filling / shortage
R_F, U_STORE = S04_RF, S04_STORAGE     # financing and storage, as in Session 2.4's crude examples
Y_MAX, I_SCALE = 0.60, 0.35            # the convenience-yield curve (illustrative)
HALF_LIFE = 0.75                       # years for inventories to close half the gap to normal
KAPPA = np.log(2) / HALF_LIFE
STATES = {"scarce": 0.30, "normal": 1.00, "glut": 1.80}   # inventory today ÷ normal
MONTHS = np.arange(0, 37)              # 0–36 months: long enough to see a slow recovery bottom out


def convenience_yield(inv: float) -> float:
    """Convenience yield per year as a function of inventory relative to normal (Working-curve shape)."""
    return Y_MAX * np.exp(-inv / I_SCALE)


def expected_inventory(inv0: float, t: float) -> float:
    return 1.0 + (inv0 - 1.0) * np.exp(-KAPPA * t)


def log_premium(inv0: float, tau: float) -> float:
    """ln(F(τ)/S): the net carry r_f + u − y integrated along the expected inventory path."""
    if tau == 0:
        return 0.0
    val, _ = quad(lambda s: R_F + U_STORE - convenience_yield(expected_inventory(inv0, s)), 0.0, tau,
                  epsabs=1e-12, epsrel=1e-12)
    return val


curves = {}
for name, inv0 in STATES.items():
    lp = np.array([log_premium(inv0, m / 12) for m in MONTHS])
    curves[name] = lp
    lab.record(f"st_{name}_inv", inv0)
    lab.record(f"st_{name}_y0", convenience_yield(inv0))
    lab.record(f"st_{name}_slope_1m", 12 * lp[1])                # annualised 1-month calendar spread
    lab.record(f"st_{name}_prem_12m", float(np.expm1(lp[12])))   # F(12m)/S − 1
    lab.record(f"st_{name}_prem_24m", float(np.expm1(lp[24])))
    lab.record(f"st_{name}_roll_1m", -12 * lp[1])                 # roll yield, annualised (Session 2.4)
    # the full-carry ceiling: convenience yield is never negative, so F ≤ S·exp((r_f + u)τ)
    assert np.all(lp <= (R_F + U_STORE) * MONTHS / 12 + 1e-12)
# normal inventories stay normal, so the curve is the constant-carry curve of Session 2.4
assert np.allclose(curves["normal"], (R_F + U_STORE - convenience_yield(1.0)) * MONTHS / 12, atol=1e-10)
# scarcity: backwardation at the front, turning up once inventories are expected to recover
bottom = int(np.argmin(curves["scarce"]))          # the month where the scarce curve stops falling
assert curves["scarce"][1] < 0
lab.record("st_scarce_bottom_month", bottom)
lab.record("full_carry", R_F + U_STORE)
lab.record("st_rf", R_F)
lab.record("st_u", U_STORE)
lab.record("st_half_life", HALF_LIFE)
# the inventory level where y = r_f + u: below it the front of the curve inverts
inv_star = -I_SCALE * np.log((R_F + U_STORE) / Y_MAX)
lab.record("inv_star", inv_star)
# The scarce curve stops falling when expected inventory climbs back to inv_star, at
# t* = ln((1 − I0)/(1 − inv_star))/κ. Check the grid minimum sits within a month of it — unless t*
# lies beyond the grid, in which case the curve has not bottomed out yet (no fake answer).
t_star_m = 12 * np.log((1 - STATES["scarce"]) / (1 - inv_star)) / KAPPA
lab.record("st_scarce_bottom_theory_m", t_star_m)
if t_star_m < MONTHS[-1]:
    assert abs(bottom - t_star_m) <= 1
else:
    print(f"the scarce curve has not bottomed within {MONTHS[-1]} months (t* = {t_star_m:.1f})")
# Session 2.4's two crude cases are two points on this curve: y = 1% (glut) and y = 21% (shortage)
for tag, y24 in S04_Y.items():
    inv = -I_SCALE * np.log(y24 / Y_MAX)
    assert abs(convenience_yield(inv) - y24) < 1e-12
    lab.record(f"inv_{tag}", inv)
    lab.record(f"y_{tag}", y24)

lab.chart("storage_curves", charts.line_chart(
    [charts.Series("full-carry ceiling", pd.Series(np.expm1((R_F + U_STORE) * MONTHS / 12), index=MONTHS),
                   role="benchmark", end_label="ceiling"),
     charts.Series("glut", pd.Series(np.expm1(curves["glut"]), index=MONTHS), role="alt2"),
     charts.Series("normal", pd.Series(np.expm1(curves["normal"]), index=MONTHS), role="strategy"),
     charts.Series("scarce", pd.Series(np.expm1(curves["scarce"]), index=MONTHS), role="alt1")],
    title="Storage model: futures premium over spot by months to expiry, three inventory states",
    y_fmt=charts.fmt_pct(0), hline=0.0))

# Working's "supply of storage" curve: the price of storage (front spread) against inventories
inv_grid = np.linspace(0.10, 2.50, 97)
spread_1m = np.array([12 * log_premium(i0, 1 / 12) for i0 in inv_grid])
assert np.all(np.diff(spread_1m) > 0)                  # more inventory → a steeper contango, always
assert np.all(spread_1m < R_F + U_STORE)               # ...but never beyond full carry
root = float(np.interp(0.0, spread_1m, inv_grid))
assert root < inv_star < root + 0.1                    # inventories recover within the month, so the 1-month spread turns first
lab.record("spread_root_inv", root)
lab.record("spread_min", float(spread_1m.min()))
lab.chart("supply_of_storage", charts.line_chart(
    [charts.Series("1-month calendar spread", pd.Series(spread_1m, index=inv_grid), role="strategy",
                   label_end=False)],
    title="Price of storage vs inventory (1 = normal): annualised 1-month spread, storage model",
    y_fmt=charts.fmt_pct(0), hline=R_F + U_STORE, hline_label="full carry (r_f + u)"))

# %% [markdown]
# ## 6 · A seasonal, natural-gas-style curve
#
# Gas is burned mostly in winter but produced all year, so it is injected into storage from April
# to October and withdrawn from November to March (the EIA's seasons). The curve carries a seasonal
# factor on top of a smooth carry. Storage arbitrage caps how fast the curve may RISE:
# buy month i, store, deliver into month j, so F_j ≤ F_i·exp(r_f Δ) + U·Δ. Nothing caps the FALL
# after winter — you cannot store gas backwards in time.

# %%
GAS_BASE = 3.50        # $/MMBtu, the deseasonalised level today (illustrative)
GAS_CARRY = 0.02       # smooth (non-seasonal) slope of the curve, per year
GAS_AMP = 0.07         # seasonal amplitude in logs: winter ≈ +7%, summer ≈ −7%
PEAK_MONTH = 1         # January
START_MONTH = 5        # the front contract delivers in May
GAS_RF = 0.04          # financing rate
GAS_STORAGE = 1.80     # $/MMBtu per year to hold gas in storage (illustrative, includes capacity rent)
N_GAS = 36
k_gas = np.arange(1, N_GAS + 1)                        # months to delivery
deliv_month = (START_MONTH - 1 + k_gas - 1) % 12 + 1   # calendar month of delivery, 1..12


def gas_curve(amp: float) -> np.ndarray:
    """Futures price for delivery in k months: smooth carry × seasonal factor peaking in PEAK_MONTH."""
    season = amp * np.cos(2 * np.pi * (deliv_month - PEAK_MONTH) / 12)
    return GAS_BASE * np.exp(GAS_CARRY * k_gas / 12 + season)


def storage_violations(F: np.ndarray, r: float = GAS_RF, U: float = GAS_STORAGE) -> int:
    """Pairs (i < j) where buying month i, storing and delivering into month j is a riskless profit."""
    i, j = np.triu_indices(len(F), k=1)
    dt = (j - i) / 12
    return int(np.sum(F[j] > F[i] * np.exp(r * dt) + U * dt + 1e-12))


F_gas = gas_curve(GAS_AMP)
flat_gas = GAS_BASE * np.exp(GAS_CARRY * k_gas / 12)
n_viol = storage_violations(F_gas)
if n_viol:
    print(f"warning: {n_viol} storage-arbitrage violations — the seasonal amplitude is too large for GAS_STORAGE")
# the checker works: a curve with triple the seasonality must violate the storage bound
assert storage_violations(gas_curve(3 * GAS_AMP)) > 0
# the largest amplitude storage arbitrage allows (bisection), given the storage cost
lo_a, hi_a = 0.0, 1.0
for _ in range(60):
    mid = (lo_a + hi_a) / 2
    lo_a, hi_a = (mid, hi_a) if storage_violations(gas_curve(mid)) == 0 else (lo_a, mid)
amp_max = lo_a


def storage_slack(F: np.ndarray, r: float = GAS_RF, U: float = GAS_STORAGE) -> np.ndarray:
    """Bound minus price for every pair (i < j): F_i·e^{rΔ} + UΔ − F_j (storage paid at delivery;
    Session 2.4's (S + U)·e^{rτ} prepays it — a small difference). Negative slack = arbitrage."""
    i, j = np.triu_indices(len(F), k=1)
    dt = (j - i) / 12
    return i, j, F[i] * np.exp(r * dt) + U * dt - F[j]


# Which trade binds first as seasonality grows? At the largest allowed amplitude, the pair with the
# least slack is the steepest single step of the autumn climb — not the summer-to-winter spread.
bi, bj, bslack = storage_slack(gas_curve(amp_max))
b = int(np.argmin(bslack))
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
lab.record("gas_bind_from", MONTH_NAMES[deliv_month[bi[b]] - 1])
lab.record("gas_bind_to", MONTH_NAMES[deliv_month[bj[b]] - 1])
lab.record("gas_bind_months", int(bj[b] - bi[b]))
assert bj[b] - bi[b] == 1                      # the binding trade is a one-month step
# the July-to-January spread and its own storage bound, on the default curve
jul0 = int(np.flatnonzero(deliv_month == 7)[0])
jan0 = int(np.flatnonzero((deliv_month == 1) & (k_gas > k_gas[jul0]))[0])
dt_js = (jan0 - jul0) / 12
js_bound = F_gas[jul0] * np.exp(GAS_RF * dt_js) + GAS_STORAGE * dt_js - F_gas[jul0]
lab.record("gas_js_spread", float(F_gas[jan0] - F_gas[jul0]))
lab.record("gas_js_bound", float(js_bound))
assert F_gas[jan0] - F_gas[jul0] < js_bound

# the naive "carry" of a seasonal curve: each month's annualised roll between neighbouring contracts
naive_carry = 12 * np.log(F_gas[:-1] / F_gas[1:])
# over any 12 consecutive months the seasonal part cancels exactly: the average is the smooth −carry
for s in range(len(naive_carry) - 11):
    assert abs(naive_carry[s:s + 12].mean() - (-GAS_CARRY)) < 1e-12

winter = np.isin(deliv_month, (12, 1, 2))
jan = np.flatnonzero(deliv_month == 1)[0]
jul = np.flatnonzero(deliv_month == 7)[0]
lab.record("gas_base", GAS_BASE)
lab.record("gas_amp", GAS_AMP)
lab.record("gas_carry", GAS_CARRY)
lab.record("gas_storage", GAS_STORAGE)
lab.record("gas_rf", GAS_RF)
lab.record("gas_violations", n_viol)
lab.record("gas_amp_max", amp_max)
lab.record("gas_jan", float(F_gas[jan]))
lab.record("gas_jul", float(F_gas[jul]))
lab.record("gas_winter_summer", float(F_gas[jan] - F_gas[jul]))
lab.record("gas_naive_carry_max", float(naive_carry.max()))
lab.record("gas_naive_carry_min", float(naive_carry.min()))
lab.record("gas_naive_carry_mean", float(naive_carry[:12].mean()))
print(f"gas: Jan {F_gas[jan]:.2f}, Jul {F_gas[jul]:.2f}; violations {n_viol}; max amplitude {amp_max:.3f}; "
      f"naive carry {naive_carry.min():+.1%} … {naive_carry.max():+.1%}, 12-month mean {naive_carry[:12].mean():+.2%}")

# shade the winter deliveries (Dec–Feb) so the seasonal humps are easy to see
bands, k = [], 0
while k < N_GAS:
    if winter[k]:
        e = k
        while e + 1 < N_GAS and winter[e + 1]:
            e += 1
        bands.append((float(k_gas[k]) - 0.5, float(k_gas[e]) + 0.5, "winter"))
        k = e + 1
    else:
        k += 1
lab.chart("gas_curve", charts.line_chart(
    [charts.Series("no seasonality", pd.Series(flat_gas, index=k_gas), role="benchmark", label_end=False),
     charts.Series("futures curve", pd.Series(F_gas, index=k_gas), role="strategy", end_label="curve")],
    title="A natural-gas-style futures curve ($/MMBtu) by months to delivery; front = May",
    y_fmt=charts.fmt_num(2, prefix="$"), bands=bands))

# %%
lab.save()
