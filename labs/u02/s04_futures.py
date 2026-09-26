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
# # 2.4 · Futures: Carry, Basis, Roll & Margin — companion lab
#
# **Quant Notebook** · Unit 2 · Session 4 · [Read the session](https://georgevrs.github.io/quant-notebook/unit02-markets-instruments/session04-futures.html)
#
# How futures are priced from cost of carry, why curves slope, what roll yield really is, and how margin creates leverage and liquidation risk.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# All prices and margin levels below are **illustrative** round numbers, not market data. Contract
# multipliers and tick sizes are the real CME specifications (checked on cmegroup.com, 2026).
# CME sets margins in dollars per contract; the percentages here are illustrative.
# Simulations use synthetic prices; time is elapsed trading time (252 days = 1 year), never a date.

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

lab = qn.Lab("2.4")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
rng_arb, rng_roll, rng_worlds, rng_margin = lab.rng.spawn(4)

# %% [markdown]
# ## 1 · Reading a contract: multiplier, tick, notional
#
# A futures price is quoted per unit of the underlying (index points, dollars per barrel). The
# **contract multiplier** turns it into money: notional value = futures price × multiplier.
# The tick is the smallest price step; its dollar value is tick × multiplier.

# %%
SPECS = {  # name: (multiplier in $ per point/unit, tick size, illustrative price)
    "es": (50.0, 0.25, 5_000.0),     # E-mini S&P 500: $50 × index, tick 0.25 index points
    "mes": (5.0, 0.25, 5_000.0),     # Micro E-mini S&P 500: $5 × index
    "cl": (1_000.0, 0.01, 70.0),     # WTI crude oil (CL): 1,000 barrels, tick $0.01 per barrel
}
for name, (mult, tick, price) in SPECS.items():
    lab.record(f"{name}_mult", mult)
    lab.record(f"{name}_tick_value", tick * mult)
    lab.record(f"{name}_price", price)
    lab.record(f"{name}_notional", price * mult)
    print(f"{name.upper():>4}: tick worth ${tick * mult:,.2f} · notional ${price * mult:,.0f}")

# Margins are fixed dollars per contract; we express them as fractions of the notional AT ENTRY.
# (Illustrative: CME says futures margin is typically 3–12% of notional.)
MAINT_PCT = 0.05                      # maintenance margin, 5% of entry notional
INIT_PCT = 1.10 * MAINT_PCT           # initial = 110% of maintenance (CME's rule for higher-risk accounts)
lab.record("maint_pct", MAINT_PCT)
lab.record("init_pct", INIT_PCT)
lab.record("max_leverage", 1 / INIT_PCT)  # the most exposure one dollar of initial margin can buy here
lab.record("es_init_margin", INIT_PCT * SPECS["es"][0] * SPECS["es"][2])
lab.record("es_maint_margin", MAINT_PCT * SPECS["es"][0] * SPECS["es"][2])
assert abs(SPECS["es"][0] * SPECS["es"][1] - 12.5) < 1e-12   # $12.50 a tick, as CME specifies

# Session 2.3's asset (μ = 8%, σ = 20%, r_f = 3%) has growth-optimal leverage L* = (μ − r_f)/σ².
MU_23, SIGMA_23, RF_23 = 0.08, 0.20, 0.03
l_star_23 = (MU_23 - RF_23) / SIGMA_23 ** 2
lab.record("l_star_23", l_star_23)
lab.record("max_lev_over_lstar", (1 / INIT_PCT) / l_star_23)
print(f"max leverage at initial margin: {1 / INIT_PCT:.1f}× — {(1 / INIT_PCT) / l_star_23:.1f}× the 2.3 optimum")

# %% [markdown]
# ## 2 · Pricing a futures curve by cost of carry
#
# Owning the asset until expiry costs financing (r_f) and storage (u) and pays income (q) and a
# convenience yield (y). No-arbitrage forces the futures price for a contract expiring in τ years
# to equal the spot price grown at the **net cost of carry** c = r_f + u − q − y:
# F = S · exp(c · τ). (All rates continuously compounded, per year.)

# %%
CARRY_CASES = {  # illustrative parameters, not market data
    "index": dict(S=5_000.0, r_f=0.04, u=0.00, q=0.015, y=0.00),   # equity index: financing − dividends
    "glut": dict(S=70.0, r_f=0.04, u=0.05, q=0.00, y=0.01),        # crude, tanks nearly full: contango
    "tight": dict(S=70.0, r_f=0.04, u=0.05, q=0.00, y=0.21),       # crude, shortage: backwardation
}
MATURITIES_M = [0, 1, 3, 6, 12]  # months to expiry


def net_carry(r_f: float, u: float, q: float, y: float) -> float:
    """Net cost of carry c = financing + storage − income − convenience yield (per year)."""
    return r_f + u - q - y


def futures_price(S: float, c: float, tau: float) -> float:
    """Cost-of-carry futures price for a contract expiring in tau years."""
    return S * np.exp(c * tau)


curve_rows = []
for name, p in CARRY_CASES.items():
    c = net_carry(p["r_f"], p["u"], p["q"], p["y"])
    lab.record(f"{name}_c", c)
    for param, value in p.items():                 # the inputs, so the page can quote them
        lab.record(f"{name}_{param}", value)
    for m in MATURITIES_M:
        F = futures_price(p["S"], c, m / 12)
        basis = p["S"] - F                     # our sign convention: basis = spot − futures
        curve_rows.append({"case": name, "months": m, "F": F, "basis": basis})
        lab.record(f"{name}_F_{m}m", F)
        lab.record(f"{name}_basis_{m}m", basis)
curve = pd.DataFrame(curve_rows)
print(curve.pivot(index="months", columns="case", values="F").round(2))

# convergence: at expiry (tau = 0) the futures price IS the spot price, so the basis is zero
assert (curve.loc[curve.months == 0, "basis"].abs() < 1e-12).all()
# the sign of the basis tells you the shape: negative in contango, positive in backwardation
assert (curve.loc[(curve.case == "glut") & (curve.months > 0), "basis"] < 0).all()
assert (curve.loc[(curve.case == "tight") & (curve.months > 0), "basis"] > 0).all()

# %% [markdown]
# **Cash-and-carry arbitrage.** Suppose the 3-month index future trades at 5,060 while the
# cost-of-carry price is lower. Borrow, buy the index, sell the future: the profit at expiry is
# F_market − S·exp((r_f − q)τ) per index point, whatever the index does. With the E-mini's $50
# multiplier that is real money — which is exactly why index arbitrageurs keep it from happening.

# %%
F_MARKET = 5_060.0
p = CARRY_CASES["index"]
TAU3 = 3 / 12
fair = futures_price(p["S"], net_carry(p["r_f"], p["u"], p["q"], p["y"]), TAU3)
arb_points = F_MARKET - fair
lab.record("arb_F_market", F_MARKET)
lab.record("arb_fair", fair)
lab.record("arb_points", arb_points)
lab.record("arb_dollars_es", arb_points * SPECS["es"][0])

# Check the arbitrage by brute force: simulate the index at expiry many ways; the locked-in profit
# of the cash-and-carry package must not depend on where the index ends.
S_end = p["S"] * np.exp(rng_arb.normal(0.0, 0.2 * np.sqrt(TAU3), 1_000))
units = np.exp(-p["q"] * TAU3)                     # buy e^{-q tau} units; dividends reinvested → 1 unit
loan = units * p["S"]                              # borrowed today
package = (S_end - loan * np.exp(p["r_f"] * TAU3)) + (F_MARKET - S_end)   # asset leg + short-futures leg
assert np.allclose(package, arb_points), "cash-and-carry profit must be the same in every scenario"
print(f"fair 3m price {fair:,.2f}; market {F_MARKET:,.0f} → locked-in {arb_points:.2f} points "
      f"= ${arb_points * SPECS['es'][0]:,.0f} per E-mini")

# For a consumption commodity the reverse trade (short the physical) is not available, so only the
# upper bound F ≤ S·exp((r_f + u)τ) holds. The convenience yield measures how far below it F sits.
pt = CARRY_CASES["tight"]
upper = pt["S"] * np.exp((pt["r_f"] + pt["u"]) * TAU3)
lab.record("tight_upper_3m", upper)
assert lab.results["tight_F_3m"] < upper

# %% [markdown]
# ## 3 · What a rolled futures position earns: spot + roll + collateral
#
# One spot path, two curves. We hold the front contract for its last month (21 trading days),
# let it converge to spot at expiry, and roll into the next one. Fully collateralised: the notional
# sits in T-bills earning r_f. Koijen, Moskowitz, Pedersen & Vrugt (2018) write the excess return
# over one contract life (bought at F_0, held to expiry)
#
#     R = (S_exp − F_0)/F_0 = (S_0 − F_0)/F_0 + (S_exp − S_0)/F_0 = carry + spot move,
#
# and with a constant-slope curve F = S·exp(c τ) the daily log futures return is the log spot
# return minus c/252: roll yield accrues every day, not on roll day.

# %%
YEARS = 10
ROLL_DAYS = 21                         # hold each contract for its final month
MU_SPOT, SIGMA_SPOT, RF = 0.03, 0.25, 0.03
C_CONTANGO, C_BACKW = 0.06, -0.06      # net carry of the two curves (per year)
n_days = PERIODS_PER_YEAR * YEARS
assert n_days % ROLL_DAYS == 0
spot = qn.synth.gbm_prices(n_days, mu=MU_SPOT, sigma=SIGMA_SPOT, s0=70.0, rng=rng_roll)["price"]
spot.index = np.arange(n_days + 1) / PERIODS_PER_YEAR   # elapsed years, not calendar dates


def rolled_futures(spot: pd.Series, c: float, roll_days: int = ROLL_DAYS) -> dict[str, pd.Series]:
    """Hold the front contract for its last `roll_days` days, then roll; constant-slope curve.

    Returns {'er': daily excess returns of the rolled position,
             'front': the unadjusted front-month price series (what a spliced chart shows)}.
    """
    s = spot.to_numpy()
    t = np.arange(len(s))
    days_left = (-t) % roll_days                        # trading days to the next expiry (0 on expiry days)
    tau_close = np.where(days_left == 0, roll_days, days_left) / PERIODS_PER_YEAR  # contract held overnight
    front = s * np.exp(c * days_left / PERIODS_PER_YEAR)          # expiring contract = spot
    held_prev = s[:-1] * np.exp(c * tau_close[:-1])               # price paid at yesterday's close
    held_now = s[1:] * np.exp(c * (tau_close[:-1] - 1 / PERIODS_PER_YEAR))   # same contract today
    er = held_now / held_prev - 1.0
    return {"er": pd.Series(er, index=spot.index[1:]), "front": pd.Series(front, index=spot.index)}


def cagr(simple: pd.Series) -> float:
    return float((1 + simple).prod() ** (1 / YEARS) - 1)


def growth_index(simple: pd.Series) -> pd.Series:
    """Growth of $1, starting at 1.0 at year 0."""
    g = (1 + simple).cumprod()
    return pd.concat([pd.Series([1.0], index=[0.0]), g])


spot_r = spot.pct_change().iloc[1:]
legs = {name: rolled_futures(spot, c) for name, c in (("contango", C_CONTANGO), ("backw", C_BACKW))}
for name, c in (("contango", C_CONTANGO), ("backw", C_BACKW)):
    er = legs[name]["er"]
    tr = er + RF / PERIODS_PER_YEAR                       # collateral interest on the notional
    # the decomposition is an identity: log futures = log spot − c per year, to machine precision
    log_gap = np.log1p(er).sum() - np.log1p(spot_r).sum()
    assert abs(log_gap - (-c * YEARS)) < 1e-9
    lab.record(f"{name}_er_cagr", cagr(er))
    lab.record(f"{name}_tr_cagr", cagr(tr))
    lab.record(f"{name}_roll_log", -c)                    # roll yield per year, log terms
    lab.record(f"{name}_growth_er", float((1 + er).prod()))
lab.record("spot_cagr", cagr(spot_r))
lab.record("spot_growth", float(spot.iloc[-1] / spot.iloc[0]))
lab.record("roll_mu_spot", MU_SPOT)
lab.record("roll_sigma_spot", SIGMA_SPOT)
lab.record("roll_rf", RF)
lab.record("c_contango", C_CONTANGO)
lab.record("c_backw", C_BACKW)
print(f"spot CAGR {cagr(spot_r):.2%} · futures ER: contango {lab.results['contango_er_cagr']:.2%}, "
      f"backwardation {lab.results['backw_er_cagr']:.2%}")

# KMPV's per-contract identity: excess return = carry + spot change, exactly, contract by contract
s = spot.to_numpy()
starts = s[:-1:ROLL_DAYS]
ends = s[ROLL_DAYS::ROLL_DAYS]
F0 = starts * np.exp(C_CONTANGO * ROLL_DAYS / PERIODS_PER_YEAR)  # price paid for each contract
R_contract = (ends - F0) / F0      # excess return, contract by contract
carry = (starts - F0) / F0         # known the day you buy: the roll yield
spot_part = (ends - starts) / F0   # what the spot price did
assert np.allclose(R_contract, carry + spot_part, atol=1e-14)
# ... and compounding the daily excess returns inside each contract gives the same number
daily = legs["contango"]["er"].to_numpy().reshape(-1, ROLL_DAYS)
assert np.allclose(np.prod(1 + daily, axis=1) - 1, R_contract, atol=1e-12)
lab.record("carry_per_contract_contango", float(carry[0]))

# The naive mistake: reading "the oil futures return" off a spliced front-month price chart.
# The spliced series starts and ends at spot, so its log growth is the SPOT's; the investable
# position's is lower by exactly c per year. An identity, not a property of this path.
front = legs["contango"]["front"]
naive = cagr(front.pct_change().iloc[1:])
assert abs(np.log(front.iloc[-1] / front.iloc[0]) - np.log1p(legs["contango"]["er"]).sum() - C_CONTANGO * YEARS) < 1e-9
lab.record("naive_front_cagr", naive)
lab.record("naive_gap_pp", 100 * (naive - lab.results["contango_er_cagr"]))

# A no-free-lunch check: a collateralised EQUITY-INDEX future priced at carry (c = r_f − q) earns the
# index's total return (price + dividends): the dividend the futures holder misses is exactly the carry.
Q_DIV = 0.02
er_idx = rolled_futures(spot, RF - Q_DIV)["er"]
tr_idx = er_idx + RF / PERIODS_PER_YEAR
index_tr = (1 + spot_r) * (1 + Q_DIV / PERIODS_PER_YEAR) - 1   # dividends reinvested daily
assert abs(cagr(tr_idx) - cagr(index_tr)) < 0.001

lab.chart("rolled_growth", charts.line_chart(
    [charts.Series("spot", spot / spot.iloc[0], role="benchmark"),
     charts.Series("contango", growth_index(legs["contango"]["er"]), role="alt1"),
     charts.Series("backwardated", growth_index(legs["backw"]["er"]), role="alt2")],
    title="Growth of $1: spot vs rolled futures on two curves, same spot path (log scale, years)", logy=True,
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

# %% [markdown]
# ## 4 · Is roll yield a return? Two worlds, 2,000 paths each
#
# The decomposition above is an accounting identity. Whether the carry shows up as *expected*
# return depends on how the spot price moves:
#
# * **World A — spot ignores the curve.** Spot drifts at 3% a year whatever the curve says.
#   Then the expected futures excess return is (spot drift − c): carry is a return forecast.
# * **World B — the curve is an unbiased forecast.** Spot drifts at exactly c, so E[S_exp] = F.
#   The carry is then fully offset by expected spot appreciation, and every curve shape has an
#   expected futures excess return of zero.
#
# Same random shocks in both worlds and for every curve (common random numbers), so the only
# difference is the drift. Returns are simple; the average (arithmetic) monthly return × 12 is the
# expected return we compare with theory.

# %%
N_PATHS, SIM_YEARS = 2_000, 10
SHAPES = [("contango +12%", 0.12), ("contango +6%", 0.06), ("flat 0%", 0.0),
          ("backw. −6%", -0.06), ("backw. −12%", -0.12)]
n_months = 12 * SIM_YEARS
dt = 1 / 12
z = rng_worlds.standard_normal((n_months, N_PATHS))


def monthly_futures_excess(spot_drift: float, c: float) -> np.ndarray:
    """Excess return of each 1-month contract held to expiry: (S_exp − F_0)/F_0 with F_0 = S_0 e^{c/12}."""
    growth = np.exp((spot_drift - 0.5 * SIGMA_SPOT ** 2) * dt + SIGMA_SPOT * np.sqrt(dt) * z)  # S_exp / S_0
    return growth * np.exp(-c * dt) - 1.0


two_worlds = []
for label, c in SHAPES:
    roll = 12 * (np.exp(-c * dt) - 1)                      # annualised carry, KMPV (S − F)/F × 12
    a = monthly_futures_excess(MU_SPOT, c)
    b = monthly_futures_excess(c, c)
    two_worlds.append({"shape": label, "c": c, "roll_yield": roll, "world_a": 12 * a.mean(), "world_b": 12 * b.mean(),
                       "se": 12 * a.std(ddof=1) / np.sqrt(a.size),
                       "theory_a": 12 * (np.exp((MU_SPOT - c) * dt) - 1), "theory_b": 0.0})
tw = pd.DataFrame(two_worlds)
print(tw.round(4))
for row in tw.itertuples():
    key = f"{row.c:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
    lab.record(f"tw_{key}_roll", row.roll_yield)
    lab.record(f"tw_{key}_a", row.world_a)
    lab.record(f"tw_{key}_b", row.world_b)
    lab.record(f"tw_{key}_theory_a", row.theory_a)
# simulation agrees with theory within 4 standard errors (≈ 0.25 percentage points each)
assert ((tw.world_a - tw.theory_a).abs() < 4 * tw.se).all()
assert (tw.world_b.abs() < 4 * tw.se).all()
lab.record("tw_n_paths", N_PATHS)
lab.record("tw_se_pp", 100 * float(tw.se.max()))

lab.chart("two_worlds", charts.grouped_column_chart(
    [s for s, _ in SHAPES],
    [("roll yield (carry)", tw.roll_yield.tolist(), "benchmark"),
     ("excess return, world A", tw.world_a.tolist(), "strategy"),
     ("excess return, world B", tw.world_b.tolist(), "alt1")],
    title="Roll yield vs average futures excess return, by curve shape (2,000 paths)",
    y_fmt=charts.fmt_pct(0), value_labels=True))

# %% [markdown]
# ## 5 · The margin ledger: one position, six trading days
#
# Long 2 E-mini contracts at 5,000 with $40,000 in the account. Every evening the exchange sets a
# settlement price; the day's gain or loss moves between accounts as **variation margin**. If
# equity drops below the maintenance level, the broker issues a **margin call** to restore the
# account to the *initial* level.

# %%
N_ES, ENTRY, DEPOSIT = 2, 5_000.0, 40_000.0
SETTLES = [4_985.0, 4_940.0, 4_880.0, 4_845.0, 4_905.0, 4_930.0]
mult = SPECS["es"][0]
im_total = N_ES * INIT_PCT * mult * ENTRY           # dollars, fixed at entry
mm_total = N_ES * MAINT_PCT * mult * ENTRY
equity, prev, deposits = DEPOSIT, ENTRY, 0.0
ledger = []
for day, settle in enumerate(SETTLES, start=1):
    vm = N_ES * mult * (settle - prev)                  # variation margin: today's mark-to-market P&L
    equity += vm
    call = max(im_total - equity, 0.0) if equity < mm_total else 0.0
    if call:
        lab.record("led_call_equity_pre", equity)       # equity before the top-up
        lab.record("led_call_drop_pts", ENTRY - settle)
        lab.record("led_call_drop_pct", (ENTRY - settle) / ENTRY)
    equity += call
    deposits += call
    ledger.append({"day": day, "settle": settle, "vm": vm, "call": call, "equity": equity})
    prev = settle
led = pd.DataFrame(ledger)
print(led)
for row in led.itertuples():
    lab.record(f"led_{row.day}_settle", row.settle)
    lab.record(f"led_{row.day}_vm", row.vm)
    lab.record(f"led_{row.day}_call", row.call)
    lab.record(f"led_{row.day}_equity", row.equity)
lab.record("led_entry", ENTRY)
lab.record("led_n", N_ES)
lab.record("led_notional", N_ES * mult * ENTRY)
lab.record("led_deposit", DEPOSIT)
lab.record("led_leverage", N_ES * mult * ENTRY / DEPOSIT)
lab.record("led_im", im_total)
lab.record("led_mm", mm_total)
lab.record("led_trigger_pts", (DEPOSIT - mm_total) / (N_ES * mult))
lab.record("led_trigger_pct", (DEPOSIT - mm_total) / (N_ES * mult * ENTRY))
lab.record("led_inv_l", DEPOSIT / (N_ES * mult * ENTRY))     # 1/L
lab.record("led_total_loss", -N_ES * mult * (SETTLES[-1] - ENTRY))
lab.record("led_deposits", deposits)
# accounting check: final equity = deposit + total P&L + margin-call deposits
assert np.isclose(led.equity.iloc[-1], DEPOSIT + N_ES * mult * (SETTLES[-1] - ENTRY) + deposits)
assert led.call.gt(0).sum() == 1
# the trigger formula d* = 1/L − m matches the ledger arithmetic
assert np.isclose(lab.results["led_trigger_pct"], 1 / lab.results["led_leverage"] - MAINT_PCT)

# Check-yourself question 4: 10 Micro E-minis at 5,000 with $20,000 in the account.
Q4_N, Q4_EQ = 10, 20_000.0
q4_notional = Q4_N * SPECS["mes"][0] * ENTRY
q4_lev = q4_notional / Q4_EQ
q4_dstar = 1 / q4_lev - MAINT_PCT
lab.record("q4_notional", q4_notional)
lab.record("q4_leverage", q4_lev)
lab.record("q4_dstar", q4_dstar)
lab.record("q4_points", q4_dstar * ENTRY)
lab.record("q4_level", ENTRY * (1 - q4_dstar))
lab.record("week_sd", 0.20 / np.sqrt(52))                  # one week's sd at 20% annual volatility

# %% [markdown]
# ## 6 · Leverage, margin calls and forced liquidation — Monte Carlo
#
# A trader with $100,000 in the account buys Micro E-mini contracts on a futures price that drifts
# at 5% a year with 20% volatility (the asset of Session 2.3: μ − r_f = 8% − 3%). Leverage
# L = notional at entry / account equity. With the maintenance margin m a fixed dollar amount per
# contract, written as a fraction of the entry notional, a margin call comes after an adverse move of
#
#     d* = 1/L − m
#
# Two traders face the same 20,000 one-year paths: one with no spare cash (the first margin call
# is a forced liquidation) and one with a reserve of 20% of the account to meet calls.
# Margin is checked at the daily settlement only; brokers often demand more than the exchange
# minimum and may liquidate intraday, so these liquidation probabilities are a LOWER bound.

# %%
CAPITAL, RESERVE_PCT = 100_000.0, 0.20
MU_F, SIGMA_F, N_MC, HORIZON = 0.05, 0.20, 20_000, PERIODS_PER_YEAR
LEVELS = [1, 3, 5, 10, 15]
mult_m, F0_m = SPECS["mes"][0], SPECS["mes"][2]
notional_1 = mult_m * F0_m
z_f = rng_margin.standard_normal((HORIZON, N_MC))
dt_d = 1 / PERIODS_PER_YEAR
logF = np.vstack([np.zeros(N_MC), np.cumsum((MU_F - 0.5 * SIGMA_F ** 2) * dt_d + SIGMA_F * np.sqrt(dt_d) * z_f, axis=0)])
F_paths = F0_m * np.exp(logF)                           # (HORIZON + 1, N_MC)


def simulate_account(n_contracts: int, reserve: float, keep_path: bool = False) -> dict[str, np.ndarray]:
    """Daily mark-to-market of a long position; meet calls from `reserve`, else forced liquidation.

    Margins are fixed dollars per contract (set at entry). A call restores equity to the initial
    margin. Liquidation closes the position at that day's settlement price; the account stops moving.
    """
    mm = n_contracts * MAINT_PCT * notional_1
    im = n_contracts * INIT_PCT * notional_1
    eq = np.full(N_MC, CAPITAL)
    res = np.full(N_MC, reserve)
    alive = np.ones(N_MC, dtype=bool)
    first_call = np.full(N_MC, -1)
    liq_day = np.full(N_MC, -1)
    eq_path = np.empty((HORIZON + 1, N_MC)) if keep_path else np.empty((HORIZON + 1, 0))
    if keep_path:
        eq_path[0] = eq
    for t in range(1, HORIZON + 1):
        eq = eq + alive * n_contracts * mult_m * (F_paths[t] - F_paths[t - 1])   # variation margin
        call = alive & (eq < mm)
        first_call[call & (first_call < 0)] = t
        need = im - eq
        pay = call & (need <= res)
        eq = np.where(pay, eq + need, eq)
        res = np.where(pay, res - need, res)
        fail = call & ~pay
        liq_day[fail] = t
        alive &= ~fail
        if keep_path:
            eq_path[t] = eq
    return {"first_call": first_call, "liq_day": liq_day, "eq_path": eq_path, "mm": mm, "im": im}


# %% [markdown]
# **The theory column.** For a lognormal price the chance of touching a level d* below the start
# within τ years has a closed form — the reflection principle for Brownian motion with drift
# ν = μ − σ²/2 and a lower barrier h = ln(1 − d*) < 0:
#
#     P(call) = Φ((h − ντ)/(σ√τ)) + exp(2νh/σ²) · Φ((h + ντ)/(σ√τ))
#
# Margin is checked only at the daily close, not continuously, so daily monitoring misses some
# crossings. The Broadie–Glasserman–Kou (1997) continuity correction handles this by moving the
# barrier away by 0.5826·σ·√(1/252). Session 3.5 covers the Monte Carlo side properly.

# %%
def p_touch(d_star: float, mu: float, sigma: float, tau: float, dt: float) -> float:
    """P(a GBM closes d_star below its start on some day within tau years), daily monitoring."""
    if d_star >= 1:
        return 0.0
    h = np.log(1 - d_star) - 0.5826 * sigma * np.sqrt(dt)    # discrete monitoring ≈ a barrier further away
    nu = mu - 0.5 * sigma ** 2
    s = sigma * np.sqrt(tau)
    return float(norm.cdf((h - nu * tau) / s) + np.exp(2 * nu * h / sigma ** 2) * norm.cdf((h + nu * tau) / s))


rows, runs = [], {}
up = F_paths[-1] > F_paths[0]
for L in LEVELS:
    n_c = int(round(L * CAPITAL / notional_1))
    d_star = 1 / L - MAINT_PCT
    broke = simulate_account(n_c, 0.0, keep_path=(L == 10))
    reserved = simulate_account(n_c, RESERVE_PCT * CAPITAL)
    runs[L] = broke
    called = broke["first_call"] >= 0
    rows.append({"L": L, "contracts": n_c, "d_star": d_star,
                 "p_call": called.mean(), "p_call_theory": p_touch(d_star, MU_F, SIGMA_F, 1.0, dt_d),
                 "p_liq_reserve": (reserved["liq_day"] >= 0).mean(),
                 "p_right_but_out": (called & up).mean(),
                 "p_right_given_out": (called & up).sum() / max(called.sum(), 1)})
mc = pd.DataFrame(rows).set_index("L")
print(mc.round(4))
for L, row in mc.iterrows():
    lab.record(f"mc_{L}_contracts", int(row.contracts))
    lab.record(f"mc_{L}_d_star", row.d_star)
    lab.record(f"mc_{L}_p_call", row.p_call)
    lab.record(f"mc_{L}_p_call_theory", row.p_call_theory)
    lab.record(f"mc_{L}_p_liq_reserve", row.p_liq_reserve)
    lab.record(f"mc_{L}_p_right_but_out", row.p_right_but_out)
    lab.record(f"mc_{L}_p_right_given_out", row.p_right_given_out)
lab.record("mc_paths", N_MC)
lab.record("mc_mu", MU_F)
lab.record("mc_sigma", SIGMA_F)
lab.record("mc_capital", CAPITAL)
lab.record("mc_reserve", RESERVE_PCT * CAPITAL)
lab.record("p_up", up.mean())
# the simulation matches the barrier formula: the tolerance, 1.5 percentage points, is about four
# Monte Carlo standard errors at p = 0.5 (0.35 pp with 20,000 paths), plus room for the
# approximate continuity correction
MC_TOL = 0.015
mc_max_gap = float((mc.p_call - mc.p_call_theory).abs().max())
lab.record("mc_max_gap_pp", 100 * mc_max_gap)
lab.record("mc_tol_pp", 100 * MC_TOL)
lab.record("mc_se_pp", 100 * np.sqrt(0.25 / N_MC))
assert mc_max_gap < MC_TOL
# a reserve postpones liquidation but never makes it more likely
assert (mc.p_liq_reserve <= mc.p_call + 1e-12).all()
assert mc.p_call.is_monotonic_increasing

# %% [markdown]
# **One path, up close.** Pick (deliberately — it is an illustration, not a typical path) the first
# path on which the 10× trader without a reserve was forced out in the second quarter, even though
# the futures price ended the year *higher* than the entry price. Plot the account against the
# maintenance requirement, and against what the same position would have been worth had the
# trader been able to hold on.

# %%
L_SHOW = 10
r10 = runs[L_SHOW]
Q1, Q2 = HORIZON // 4, HORIZON // 2
candidates = np.flatnonzero((r10["liq_day"] > Q1) & (r10["liq_day"] <= Q2) & up)
k = int(candidates[0])
n_c = int(mc.loc[L_SHOW, "contracts"])
days = np.arange(HORIZON + 1)
held = CAPITAL + n_c * mult_m * (F_paths[:, k] - F_paths[0, k])
eq_k = r10["eq_path"][:, k]
mm_line = np.where(days <= r10["liq_day"][k], r10["mm"], np.nan)
lab.record("path_liq_day", int(r10["liq_day"][k]))
lab.record("path_eq_at_liq", float(eq_k[r10["liq_day"][k]]))
lab.record("path_held_end", float(held[-1]))
lab.record("path_F_change", float(F_paths[-1, k] / F_paths[0, k] - 1))
lab.record("path_drop_at_liq", float(1 - F_paths[r10["liq_day"][k], k] / F_paths[0, k]))
lab.record("path_mm", float(r10["mm"]))
# Holding on is not free either: the unconstrained account's low point shows the cash it would have needed.
lab.record("path_held_min", float(held.min()))
lab.record("path_held_min_day", int(held.argmin()))
lab.chart("margin_path", charts.line_chart(
    [charts.Series("account", pd.Series(eq_k, index=days), role="strategy"),
     charts.Series("maintenance margin", pd.Series(mm_line, index=days), role="loss", label_end=False),
     charts.Series("held on", pd.Series(held, index=days), role="benchmark")],
    title="10× long futures, no spare cash: one path (x = trading days)",
    y_fmt=charts.fmt_usd_compact(0)))

# %% [markdown]
# ## 7 · 20 April 2020, in dollars
#
# The May 2020 WTI contract settled at $18.27 on Friday 17 April and at −$37.63 on Monday
# 20 April, the day before it expired (CFTC staff report, 2020); it fell $55.90 that day, while
# the June contract settled at $20.43, down $4.60 (CRS Insight IN11354, 2020). One contract is
# 1,000 barrels, and variation margin runs from settlement to settlement.

# %%
WTI_PREV_SETTLE, WTI_SETTLE = 18.27, -37.63          # May contract, 17 and 20 April 2020
JUNE_SETTLE, JUNE_CHANGE = 20.43, -4.60               # June contract on 20 April, and its change
vm_day = (WTI_SETTLE - WTI_PREV_SETTLE) * SPECS["cl"][0]
lab.record("wti_prev_settle", WTI_PREV_SETTLE)
lab.record("wti_settle", WTI_SETTLE)
lab.record("wti_loss_abs", -vm_day)
lab.record("wti_notional_prev", WTI_PREV_SETTLE * SPECS["cl"][0])
lab.record("wti_loss_vs_notional", -vm_day / (WTI_PREV_SETTLE * SPECS["cl"][0]))
june_prev = JUNE_SETTLE - JUNE_CHANGE
lab.record("june_settle", JUNE_SETTLE)
lab.record("spread_prev", WTI_PREV_SETTLE - june_prev)   # May − June on Friday
lab.record("spread_day", WTI_SETTLE - JUNE_SETTLE)       # May − June on Monday
assert abs(-vm_day - 55_900) < 1e-6                     # the CRS figure: a $55.90 fall × 1,000 barrels
print(f"one long CL contract paid ${-vm_day:,.0f} of variation margin in a single session; "
      f"May–June spread went from {WTI_PREV_SETTLE - june_prev:+.2f} to {WTI_SETTLE - JUNE_SETTLE:+.2f}")

# %%
lab.save()
