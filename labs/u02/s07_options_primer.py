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
# # 2.7 · Options Primer — companion lab
#
# **Quant Notebook** · Unit 2 · Session 7 · [Read the session](https://georgevrs.github.io/quant-notebook/unit02-markets-instruments/session07-options-primer.html)
#
# Calls, puts, payoffs, put-call parity, intrinsic vs time value, and a first look at implied volatility — enough to read any options discussion.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Everything here is **synthetic and illustrative**: a $100 stock, round-number rates and volatilities,
# made-up quotes. Prices are per share; a standard US equity option covers 100 shares (OCC), so
# multiply by 100 for the cost of one contract. The risk-free rate is 4% a year throughout; all
# rates are continuously compounded, per year. Pricing uses the risk-neutral drift (r_f − q); only
# section 7 uses a real-world drift (8%). Time is elapsed time to expiry in years (τ), never a
# calendar date.

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

lab = qn.Lab("2.7")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
rng_parity, rng_crash, rng_arb, rng_cc = lab.rng.spawn(4)

# The running example: a $100 stock, options with 3 months to expiry.
S0 = 100.0          # spot price today
K_ATM = 100.0       # at-the-money strike
TAU = 0.25          # time to expiry in years (3 months)
R_F = 0.04          # risk-free rate, continuously compounded
SIGMA = 0.20        # volatility used to price the options (annualised)
MULT = 100          # shares per standard US equity option contract
for key, value in {"S0": S0, "K_atm": K_ATM, "tau": TAU, "tau_months": 12 * TAU, "r_f": R_F,
                   "sigma": SIGMA, "mult": MULT}.items():
    lab.record(key, value)

# %% [markdown]
# ## 0 · A pricing black box
#
# To draw P&L diagrams we need premiums, and premiums need a pricing model. The function below is
# the **Black-Scholes-Merton** price of a European option. Treat it as a black box for now:
# Session 10.3 derives it (twice), and Session 10.4 takes it apart. All this session needs is that
# it turns (spot, strike, time, rate, dividend yield, **volatility**) into a price — and that the
# price rises with volatility. Section 3 checks it against a Monte Carlo simulation.

# %%
def bs_price(S, K, tau, r_f, sigma, q=0.0, kind="call"):
    """BLACK BOX (derived in Session 10.3): Black-Scholes-Merton price of a European option.

    S spot, K strike, tau years to expiry, r_f risk-free rate, sigma volatility, q dividend yield
    (all continuously compounded, per year). Works on numpy arrays.
    """
    S, K = np.asarray(S, dtype=float), np.asarray(K, dtype=float)
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r_f - q + 0.5 * sigma ** 2) * tau) / vol
    d2 = d1 - vol
    if kind == "call":
        return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r_f * tau) * norm.cdf(d2)
    return K * np.exp(-r_f * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)


def call_payoff(S_exp, K):
    """What a call pays at expiry: the right to buy at K is worth S_exp − K, or nothing."""
    return np.maximum(S_exp - K, 0.0)


def put_payoff(S_exp, K):
    """What a put pays at expiry: the right to sell at K is worth K − S_exp, or nothing."""
    return np.maximum(K - S_exp, 0.0)


# %% [markdown]
# ## 1 · Calls and puts: premium, payoff, P&L at expiry
#
# **Payoff** is what the option is worth at expiry. **P&L at expiry** is payoff minus the premium
# you paid (for a buyer) or premium received minus payoff (for a seller). Like most payoff diagrams,
# we ignore the interest on the premium over the three months (about 1% of it here).

# %%
C_atm = float(bs_price(S0, K_ATM, TAU, R_F, SIGMA, kind="call"))
P_atm = float(bs_price(S0, K_ATM, TAU, R_F, SIGMA, kind="put"))
lab.record("C_atm", C_atm)
lab.record("P_atm", P_atm)
lab.record("C_atm_contract", C_atm * MULT)
lab.record("P_atm_contract", P_atm * MULT)
lab.record("call_breakeven", K_ATM + C_atm)
lab.record("put_breakeven", K_ATM - P_atm)
print(f"3-month ATM call {C_atm:.2f} (${C_atm * MULT:,.0f} a contract), put {P_atm:.2f}")

SCENARIOS = [80.0, 90.0, 100.0, 110.0, 120.0]
rows = []
for s_exp in SCENARIOS:
    rows.append({"S_exp": s_exp,
                 "long_call": float(call_payoff(s_exp, K_ATM) - C_atm),
                 "short_call": float(C_atm - call_payoff(s_exp, K_ATM)),
                 "long_put": float(put_payoff(s_exp, K_ATM) - P_atm),
                 "short_put": float(P_atm - put_payoff(s_exp, K_ATM)),
                 "stock": s_exp - S0})
pnl_table = pd.DataFrame(rows).set_index("S_exp")
print(pnl_table.round(2))
for s_exp, row in pnl_table.iterrows():
    for col, v in row.items():
        lab.record(f"pnl_{col}_{int(s_exp)}", v)
# options are zero-sum: every buyer's P&L is exactly a seller's loss
assert np.allclose(pnl_table.long_call, -pnl_table.short_call)
assert np.allclose(pnl_table.long_put, -pnl_table.short_put)
# the buyer's loss is capped at the premium; the call seller's is not
assert pnl_table.long_call.min() >= -C_atm - 1e-12 and pnl_table.long_put.min() >= -P_atm - 1e-12

# Options as leverage (Session 2.3): the same +10% stock move, two very different returns.
lab.record("lev_stock_ret", 110.0 / S0 - 1)
lab.record("lev_call_ret", float(call_payoff(110.0, K_ATM) / C_atm - 1))
lab.record("lev_call_ret_flat", float(call_payoff(100.0, K_ATM) / C_atm - 1))   # stock unchanged: −100%

# Check-yourself question 1: the stock ends at 103 — right direction, and still a loss.
S_Q1 = 103.0
lab.record("q1_S", S_Q1)
lab.record("q1_payoff", float(call_payoff(S_Q1, K_ATM)))
lab.record("q1_pnl", float(call_payoff(S_Q1, K_ATM) - C_atm))
assert lab.results["q1_pnl"] < 0 < lab.results["q1_payoff"]
assert lab.results["lev_call_ret"] > 10 * lab.results["lev_stock_ret"]

# %% [markdown]
# ## 2 · Moneyness, intrinsic value and time value
#
# **Intrinsic value** is what the option would pay if it expired right now: max(S − K, 0) for a
# call, max(K − S, 0) for a put. **Time value** is the rest of the price. It is largest at the
# money, where the outcome is most uncertain, and it shrinks as expiry approaches.

# %%
MONEYNESS_STRIKES = [90.0, 100.0, 110.0]   # with the stock at $100: the 90 call is in the money, the 90 put out of it
for K in MONEYNESS_STRIKES:
    c = float(bs_price(S0, K, TAU, R_F, SIGMA, kind="call"))
    p = float(bs_price(S0, K, TAU, R_F, SIGMA, kind="put"))
    k = f"k{int(K)}"
    lab.record(f"{k}_call", c)
    lab.record(f"{k}_call_intr", max(S0 - K, 0.0))
    lab.record(f"{k}_call_tv", c - max(S0 - K, 0.0))
    lab.record(f"{k}_put", p)
    lab.record(f"{k}_put_intr", max(K - S0, 0.0))
    lab.record(f"{k}_put_tv", p - max(K - S0, 0.0))
for side in ("call", "put"):
    tv90, tv100, tv110 = (lab.results[f"k{int(K)}_{side}_tv"] for K in MONEYNESS_STRIKES)
    assert tv100 > tv90 and tv100 > tv110, "time value peaks at the money"

# Where does time value come from? Switch the uncertainty off (σ → 0): the call is then worth only
# the discounted-forward intrinsic value max(S − K·e^{−r τ}, 0). What remains is interest, not optionality.
c_novol = float(bs_price(S0, K_ATM, TAU, R_F, 1e-9))
lab.record("C_atm_novol", c_novol)
assert abs(c_novol - (S0 - K_ATM * np.exp(-R_F * TAU))) < 1e-9
lab.record("C_atm_optionality", C_atm - c_novol)

# A European put deep in the money can be worth LESS than its intrinsic value: you receive K only at
# expiry, so its value is about K·e^{−r τ} − S, which is below K − S. (An American put would be
# exercised instead — one reason American puts are worth more.)
S_DEEP = 60.0
p_deep = float(bs_price(S_DEEP, K_ATM, TAU, R_F, SIGMA, kind="put"))
lab.record("deep_put_S", S_DEEP)
lab.record("deep_put_price", p_deep)
lab.record("deep_put_intr", K_ATM - S_DEEP)
lab.record("deep_put_tv", p_deep - (K_ATM - S_DEEP))
assert p_deep < K_ATM - S_DEEP, "negative time value for a deep in-the-money European put"

# Chart: call value against spot, three months and one month before expiry, and at expiry.
TAU_1M = 1 / 12
spots = np.linspace(80.0, 120.0, 161)
lab.chart("time_value", charts.line_chart(
    [charts.Series("3 months to expiry", pd.Series(bs_price(spots, K_ATM, TAU, R_F, SIGMA), index=spots),
                   role="strategy", end_label="3 months"),
     charts.Series("1 month to expiry", pd.Series(bs_price(spots, K_ATM, TAU_1M, R_F, SIGMA), index=spots),
                   role="alt1", end_label="1 month"),
     charts.Series("at expiry = intrinsic value", pd.Series(call_payoff(spots, K_ATM), index=spots),
                   role="benchmark", end_label="expiry")],
    title="Value of a 100-strike call against the stock price (x = stock price, $)",
    y_fmt=charts.fmt_num(0, prefix="$"), height=300))
lab.record("C_atm_1m", float(bs_price(S0, K_ATM, TAU_1M, R_F, SIGMA)))

# %% [markdown]
# ## 3 · Put-call parity, checked by Monte Carlo
#
# Portfolio A: one call + a zero-coupon bond paying K at expiry. Portfolio B: one put + one share
# (with dividends reinvested, start with e^{−qτ} shares). Both are worth max(S_exp, K) at expiry, in
# every state of the world, so they must cost the same today:
#
#     C − P = S·e^{−qτ} − K·e^{−r τ}
#
# We price the call and the put separately by simulation — the average discounted payoff over
# 400,000 paths of a risk-neutral GBM (drift r − q; Session 10.1 explains why that drift) — and
# check that the difference matches the right-hand side within Monte Carlo error. Session 3.5
# covers Monte Carlo and its standard errors properly.

# %%
N_PATHS = 400_000
STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
DIV_CASES = {"q0": 0.0, "q2": 0.02}   # no dividends, and a 2% dividend yield


def terminal_gbm(n, s0, tau, r_f, sigma, q, rng):
    """S_exp under a risk-neutral GBM: one exact step, E[S_exp] = s0·e^{(r − q)τ}."""
    z = rng.standard_normal(n)
    return s0 * np.exp((r_f - q - 0.5 * sigma ** 2) * tau + sigma * np.sqrt(tau) * z)


def mc_prices(S_exp, strikes, tau, r_f):
    """Monte Carlo call and put prices (discounted mean payoff) and the standard error of C − P."""
    disc = np.exp(-r_f * tau)
    C = np.array([disc * call_payoff(S_exp, k).mean() for k in strikes])
    P = np.array([disc * put_payoff(S_exp, k).mean() for k in strikes])
    C_se = np.array([disc * call_payoff(S_exp, k).std(ddof=1) for k in strikes]) / np.sqrt(S_exp.size)
    P_se = np.array([disc * put_payoff(S_exp, k).std(ddof=1) for k in strikes]) / np.sqrt(S_exp.size)
    # path by path, call payoff − put payoff = S_exp − K, so C − P has the standard error of disc·S_exp
    diff_se = disc * S_exp.std(ddof=1) / np.sqrt(S_exp.size)
    return C, P, C_se, P_se, diff_se


parity_rows = []
for case, q in DIV_CASES.items():
    S_exp = terminal_gbm(N_PATHS, S0, TAU, R_F, SIGMA, q, rng_parity)
    C, P, C_se, P_se, diff_se = mc_prices(S_exp, STRIKES, TAU, R_F)
    rhs = S0 * np.exp(-q * TAU) - STRIKES * np.exp(-R_F * TAU)
    for k, c, p, cse, pse, r in zip(STRIKES, C, P, C_se, P_se, rhs):
        parity_rows.append({"case": case, "K": k, "C_mc": c, "P_mc": p, "C_bs": float(bs_price(S0, k, TAU, R_F, SIGMA, q)),
                            "P_bs": float(bs_price(S0, k, TAU, R_F, SIGMA, q, "put")),
                            "C_se": cse, "P_se": pse, "lhs": c - p, "rhs": r, "gap": c - p - r, "se": diff_se})
parity = pd.DataFrame(parity_rows)
parity["z"] = parity.gap / parity.se
print(parity[["case", "K", "C_mc", "P_mc", "lhs", "rhs", "gap", "se", "z"]].round(4))

# The payoff identity behind parity holds path by path, whatever the model:
assert np.allclose(call_payoff(S_exp, 100.0) - put_payoff(S_exp, 100.0), S_exp - 100.0)
# ... so the simulated C − P matches S·e^{−qτ} − K·e^{−rτ} within Monte Carlo error (4 standard errors),
PARITY_TOL_SE = 4.0
assert (parity.z.abs() < PARITY_TOL_SE).all()
# ... and the black box agrees with the simulation, option by option, within 4 standard errors.
assert ((parity.C_mc - parity.C_bs).abs() < PARITY_TOL_SE * parity.C_se).all()
assert ((parity.P_mc - parity.P_bs).abs() < PARITY_TOL_SE * parity.P_se).all()
atm0 = parity[(parity.case == "q0") & (parity.K == 100.0)].iloc[0]
atm2 = parity[(parity.case == "q2") & (parity.K == 100.0)].iloc[0]
lab.record("mc_paths", N_PATHS)
lab.record("mc_n_strikes", len(STRIKES))
lab.record("mc_C_atm", atm0.C_mc)
lab.record("mc_P_atm", atm0.P_mc)
lab.record("mc_C_atm_se", atm0.C_se)
lab.record("mc_lhs_atm", atm0.lhs)
lab.record("mc_rhs_atm", atm0.rhs)
lab.record("mc_se_diff_atm", atm0.se)
lab.record("mc_lhs_atm_q2", atm2.lhs)
lab.record("mc_rhs_atm_q2", atm2.rhs)
lab.record("div_q", DIV_CASES["q2"])
lab.record("mc_max_abs_z", float(parity.z.abs().max()))
lab.record("mc_z_q0", float(atm0.z))
lab.record("mc_z_q2", float(atm2.z))
# The gap is identical at every strike: path by path, call payoff − put payoff = S_exp − K, so the
# simulated C − P is e^{−rτ}(mean S_exp − K) and the only thing it can get wrong is the forward, mean S_exp.
assert parity.groupby("case").gap.std().max() < 1e-9
lab.record("parity_tol_se", PARITY_TOL_SE)

# %% [markdown]
# **Parity does not care about the model.** Replace the lognormal world with one that has crash
# risk: most of the time the stock diffuses with 15% volatility, but with probability 8% over the
# three months it also gaps down 25% (in log terms). We rescale so the forward price is unchanged.
# The individual option prices move — out-of-the-money puts get dearer — but C − P does not.

# %%
CRASH_SIGMA, CRASH_P, CRASH_JUMP = 0.15, 0.08, -0.25


def terminal_crash(n, s0, tau, r_f, q, rng):
    """S_exp from a diffusion plus one possible downward gap, rescaled so that E[S_exp] = s0·e^{(r − q)τ}."""
    z = rng.standard_normal(n)
    jump = rng.random(n) < CRASH_P
    mean_jump = CRASH_P * np.exp(CRASH_JUMP) + (1 - CRASH_P)          # E[gap multiplier]
    diffusion = np.exp(-0.5 * CRASH_SIGMA ** 2 * tau + CRASH_SIGMA * np.sqrt(tau) * z)
    return s0 * np.exp((r_f - q) * tau) * diffusion * np.where(jump, np.exp(CRASH_JUMP), 1.0) / mean_jump


S_exp_crash = terminal_crash(N_PATHS, S0, TAU, R_F, 0.0, rng_crash)
Cc, Pc, Cc_se, Pc_se, dse_c = mc_prices(S_exp_crash, STRIKES, TAU, R_F)
gap_crash = (Cc - Pc) - (S0 - STRIKES * np.exp(-R_F * TAU))
assert (np.abs(gap_crash) < PARITY_TOL_SE * dse_c).all(), "parity holds in the crash world too"
lab.record("crash_max_abs_z", float(np.abs(gap_crash / dse_c).max()))
lab.record("crash_sigma", CRASH_SIGMA)
lab.record("crash_p", CRASH_P)
lab.record("crash_jump", CRASH_JUMP)
lab.record("crash_jump_pct", float(np.exp(CRASH_JUMP) - 1))

# %% [markdown]
# ## 4 · A parity violation, and the arbitrage it hands you
#
# Suppose the market quotes the 3-month 100 call $0.60 above the price parity allows, with the put
# fairly priced. Then C − P > S − K·e^{−rτ}: the call side is rich. A **conversion** sells the rich
# side and buys the cheap side: sell the call, buy the put, buy the stock, and borrow K·e^{−rτ}.
# At expiry the call-put-stock package is worth exactly K, which repays the loan. What you kept
# today is profit, in every state of the world — before costs.

# %%
MISPRICING = 0.60                     # the call is quoted this much too high (per share)
HALF_SPREAD_OPT = 0.05                # you sell at the bid and buy at the ask: half a spread per option leg
HALF_SPREAD_STK = 0.01                # half the stock's bid-ask spread
COMMISSION_OPT = 0.65 / MULT          # illustrative $0.65 per option contract, per share
C_mkt = C_atm + MISPRICING
P_mkt = P_atm
loan = K_ATM * np.exp(-R_F * TAU)     # borrow today what repays exactly K at expiry
cash_today = C_mkt - P_mkt - S0 + loan
costs = 2 * (HALF_SPREAD_OPT + COMMISSION_OPT) + HALF_SPREAD_STK

# Brute-force check: wherever the stock ends, the package is worth K and the loan costs K.
S_exp_arb = S0 * np.exp(rng_arb.normal(0.0, 0.4 * np.sqrt(TAU), 10_000))   # any distribution will do
at_expiry = -call_payoff(S_exp_arb, K_ATM) + put_payoff(S_exp_arb, K_ATM) + S_exp_arb - K_ATM
profit_at_expiry = cash_today * np.exp(R_F * TAU) + at_expiry
assert np.allclose(at_expiry, 0.0), "short call + long put + stock = K in every state"
assert np.ptp(profit_at_expiry) < 1e-9, "the profit does not depend on where the stock ends"
lab.record("arb_mispricing", MISPRICING)
lab.record("arb_C_mkt", C_mkt)
lab.record("arb_lhs", C_mkt - P_mkt)
lab.record("arb_rhs", S0 - loan)
lab.record("arb_loan", loan)
lab.record("arb_cash_today", cash_today)
lab.record("arb_cash_today_contract", cash_today * MULT)
lab.record("arb_profit_expiry", float(profit_at_expiry.mean()))
lab.record("arb_costs", costs)
lab.record("arb_costs_contract", costs * MULT)
lab.record("arb_net_contract", (cash_today - costs) * MULT)
# Financing is a cost too: these numbers assume you borrow at r_f, as a dealer can. Each extra
# percentage point on the loan costs this much per contract over the life of the trade.
lab.record("arb_fin_1pp_contract", MULT * loan * (np.exp(0.01 * TAU) - 1))
lab.record("arb_fin_2pp_contract", MULT * loan * (np.exp(0.02 * TAU) - 1))
assert lab.results["arb_fin_1pp_contract"] > (0.25 - costs) * MULT, "1 pp of extra financing erases a $0.25 gap"
lab.record("arb_half_spread_opt", HALF_SPREAD_OPT)
lab.record("arb_half_spread_stk", HALF_SPREAD_STK)
lab.record("arb_commission", COMMISSION_OPT * MULT)
lab.record("arb_n_scen", S_exp_arb.size)
assert abs(cash_today - MISPRICING) < 1e-12

# Which violations survive costs? Anything smaller than `costs` is not an arbitrage for you.
for m in (0.05, 0.10, 0.25, 0.60):
    key = f"{int(round(m * 100)):02d}"
    lab.record(f"arb_gap_{key}", m)
    lab.record(f"arb_gross_{key}", m * MULT)
    lab.record(f"arb_net_{key}", (m - costs) * MULT)

# %% [markdown]
# ## 5 · Structures: payoff diagrams for four classic positions
#
# Every structure is a sum of calls, puts, stock and cash, so its P&L at expiry is the sum of the
# pieces. Strikes: covered call sells the 105 call; protective put buys the 95 put; the straddle
# buys the 100 call and the 100 put; the bull call spread buys the 100 call and sells the 110 call.

# %%
K_CC, K_PP, K_LO, K_HI = 105.0, 95.0, 100.0, 110.0
# premiums: the call sold in the covered call, the put bought for protection, the call sold in the spread
prem = {"cc": float(bs_price(S0, K_CC, TAU, R_F, SIGMA)),
        "pp": float(bs_price(S0, K_PP, TAU, R_F, SIGMA, kind="put")),
        "hi": float(bs_price(S0, K_HI, TAU, R_F, SIGMA))}


def structure_pnl(S_exp):
    """P&L at expiry, per share, of each structure (premiums not grown at r_f)."""
    return {
        "stock": S_exp - S0,
        "covered_call": S_exp - S0 - call_payoff(S_exp, K_CC) + prem["cc"],
        "protective_put": S_exp - S0 + put_payoff(S_exp, K_PP) - prem["pp"],
        "long_call": call_payoff(S_exp, K_LO) - C_atm,
        "straddle": call_payoff(S_exp, K_LO) + put_payoff(S_exp, K_LO) - C_atm - P_atm,
        "bull_spread": call_payoff(S_exp, K_LO) - call_payoff(S_exp, K_HI) - C_atm + prem["hi"],
    }


grid = np.linspace(70.0, 130.0, 241)
pnl = {k: pd.Series(v, index=grid) for k, v in structure_pnl(grid).items()}
debit_spread = C_atm - prem["hi"]
debit_straddle = C_atm + P_atm
facts = {
    "cc_premium": prem["cc"], "cc_max_gain": K_CC - S0 + prem["cc"], "cc_breakeven": S0 - prem["cc"],
    "pp_premium": prem["pp"], "pp_max_loss": S0 - K_PP + prem["pp"], "pp_breakeven": S0 + prem["pp"],
    "straddle_cost": debit_straddle, "straddle_be_lo": K_LO - debit_straddle, "straddle_be_hi": K_LO + debit_straddle,
    "straddle_move": debit_straddle / S0,
    "spread_cost": debit_spread, "spread_max_gain": K_HI - K_LO - debit_spread, "spread_breakeven": K_LO + debit_spread,
    "c110": prem["hi"],
}
for key, v in facts.items():
    lab.record(key, v)
# sanity: the diagrams agree with the formulas
assert np.isclose(pnl["covered_call"].max(), facts["cc_max_gain"])
assert np.isclose(pnl["protective_put"].min(), -facts["pp_max_loss"])
assert np.isclose(pnl["bull_spread"].max(), facts["spread_max_gain"])
assert np.isclose(pnl["straddle"].min(), -debit_straddle)
# Homework answer key: a box (bull call spread + bear put spread, 100/110) pays K_HI − K_LO in every state,
# so it must cost (K_HI − K_LO)·e^{−r τ} — put-call parity applied at both strikes.
box = (C_atm - prem["hi"]) + (float(bs_price(S0, K_HI, TAU, R_F, SIGMA, kind="put")) - P_atm)
lab.record("box_width", K_HI - K_LO)
lab.record("box_price", box)
assert abs(box - (K_HI - K_LO) * np.exp(-R_F * TAU)) < 1e-9
# by parity, a covered call has the SAME payoff as a short put at the same strike plus cash
short_put_cc = put_payoff(grid, K_CC)
assert np.allclose((grid - call_payoff(grid, K_CC)) - (K_CC - short_put_cc), 0.0)

lab.chart("hedges", charts.line_chart(
    [charts.Series("stock alone", pnl["stock"], role="benchmark", end_label="stock"),
     charts.Series(f"covered call (sell the {K_CC:.0f} call)", pnl["covered_call"], role="strategy",
                   end_label="covered call"),
     charts.Series(f"protective put (buy the {K_PP:.0f} put)", pnl["protective_put"], role="alt1",
                   end_label="protective put")],
    title="P&L at expiry per share: stock with a covered call or a protective put (x = stock price, $)",
    y_fmt=charts.fmt_num(0, prefix="$"), hline=0.0, height=320))
lab.chart("views", charts.line_chart(
    [charts.Series(f"long {K_LO:.0f} call", pnl["long_call"], role="benchmark", end_label="long call"),
     charts.Series(f"long straddle ({K_LO:.0f} call + {K_LO:.0f} put)", pnl["straddle"], role="strategy",
                   end_label="straddle"),
     charts.Series(f"bull call spread ({K_LO:.0f}/{K_HI:.0f})", pnl["bull_spread"], role="alt1",
                   end_label="bull spread")],
    title="P&L at expiry per share: straddle, bull call spread and a plain call (x = stock price, $)",
    y_fmt=charts.fmt_num(0, prefix="$"), hline=0.0, height=320))

# %% [markdown]
# ## 6 · Implied volatility: invert the black box by bisection
#
# The black box maps volatility → price, and the map is strictly increasing. So for any market
# price there is exactly one volatility that reproduces it: the **implied volatility**. Bisection
# finds it: bracket the answer, halve the bracket, keep the half whose price straddles the target.

# %%
VOLS = [0.10, 0.20, 0.30, 0.40]
prices_by_vol = [float(bs_price(S0, K_ATM, TAU, R_F, v)) for v in VOLS]
for v, p in zip(VOLS, prices_by_vol):
    lab.record(f"C_vol_{int(round(v * 100))}", p)
    lab.record(f"vol_{int(round(v * 100))}", v)
assert all(np.diff(prices_by_vol) > 0), "a call is worth more when volatility is higher"


def implied_vol(price, S, K, tau, r_f, q=0.0, kind="call", lo=1e-4, hi=5.0, tol=1e-8):
    """Bisection: the σ at which bs_price(σ) = price. Returns (σ, iterations)."""
    f = lambda s: bs_price(S, K, tau, r_f, s, q, kind) - price   # noqa: E731 — increasing in s
    if not f(lo) <= 0 <= f(hi):
        raise ValueError("price outside the range the model can produce: check no-arbitrage bounds")
    n = 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid          # model price too high → the answer is below mid
        else:
            lo = mid
        n += 1
    return 0.5 * (lo + hi), n


QUOTE = 5.25                                       # a made-up market price for the 3-month 100 call
iv, n_iter = implied_vol(QUOTE, S0, K_ATM, TAU, R_F)
lab.record("iv_quote", QUOTE)
lab.record("iv_quote_contract", QUOTE * MULT)
lab.record("iv", iv)
lab.record("iv_iters", n_iter)
assert abs(float(bs_price(S0, K_ATM, TAU, R_F, iv)) - QUOTE) < 1e-6, "the implied vol reproduces the price"
# round trip: price at 20% → invert → 20%
assert abs(implied_vol(C_atm, S0, K_ATM, TAU, R_F)[0] - SIGMA) < 1e-7

# Trader's rule of thumb (Brenner & Subrahmanyam 1988): an at-the-money straddle costs about
# 0.8 × S × σ × √τ — the market's price for the expected size of the move.
straddle_rule = 0.8 * S0 * SIGMA * np.sqrt(TAU)
lab.record("straddle_rule", straddle_rule)
lab.record("straddle_rule_iv", debit_straddle / (0.8 * S0 * np.sqrt(TAU)))
assert abs(debit_straddle - straddle_rule) / straddle_rule < 0.02

# Implied vol makes prices comparable. Which of these two ATM calls is "expensive"?
QUOTES_CMP = {"short": (1 / 12, 2.60), "long": (1.0, 9.90)}   # (tau, made-up price)
for name, (tau_q, price_q) in QUOTES_CMP.items():
    lab.record(f"cmp_{name}_tau_m", 12 * tau_q)
    lab.record(f"cmp_{name}_price", price_q)
    lab.record(f"cmp_{name}_iv", implied_vol(price_q, S0, K_ATM, tau_q, R_F)[0])
assert lab.results["cmp_short_iv"] > lab.results["cmp_long_iv"]

# %% [markdown]
# **A first look at the smile.** Invert the crash-world prices of Section 3, strike by strike.
# One model volatility cannot fit them all: out-of-the-money puts need a higher implied volatility.
# (We invert the out-of-the-money side at each strike: puts below 100, calls from 100 up.)
# To respect Monte Carlo error, each implied vol comes with the range implied by price ± 4 standard errors.

# %%
def iv_curve(C, P, C_se, P_se):
    out = []
    for k, c, p, cse, pse in zip(STRIKES, C, P, C_se, P_se):
        kind, price, se = ("put", p, pse) if k < 100.0 else ("call", c, cse)
        mid = implied_vol(price, S0, k, TAU, R_F, kind=kind)[0]
        lo = implied_vol(price - PARITY_TOL_SE * se, S0, k, TAU, R_F, kind=kind)[0]
        hi = implied_vol(price + PARITY_TOL_SE * se, S0, k, TAU, R_F, kind=kind)[0]
        out.append((mid, lo, hi))
    return np.array(out)


gbm0 = parity[parity.case == "q0"]
iv_gbm = iv_curve(gbm0.C_mc.to_numpy(), gbm0.P_mc.to_numpy(), gbm0.C_se.to_numpy(), gbm0.P_se.to_numpy())
iv_crash = iv_curve(Cc, Pc, Cc_se, Pc_se)
print(pd.DataFrame({"K": STRIKES, "iv_gbm": iv_gbm[:, 0], "iv_crash": iv_crash[:, 0]}).round(4))
# lognormal world: every strike's implied vol is 20%, within Monte Carlo error
assert ((iv_gbm[:, 1] <= SIGMA) & (SIGMA <= iv_gbm[:, 2])).all()
i85, i100 = 0, int(np.flatnonzero(STRIKES == 100.0)[0])
if CRASH_P > 0 and CRASH_JUMP < 0:
    # crash world: the lowest strike's implied vol exceeds the ATM one by far more than the MC error
    assert iv_crash[i85, 1] > iv_crash[i100, 2] + 0.02
else:
    # no crash risk (a Try-it edit): the crash world is a plain lognormal world with CRASH_SIGMA
    assert ((iv_crash[:, 1] <= CRASH_SIGMA) & (CRASH_SIGMA <= iv_crash[:, 2])).all()
lab.record("smile_iv_85", iv_crash[i85, 0])
lab.record("smile_iv_100", iv_crash[i100, 0])
lab.record("smile_iv_115", iv_crash[-1, 0])
lab.record("smile_gbm_max_dev_pp", 100 * float(np.abs(iv_gbm[:, 0] - SIGMA).max()))
lab.chart("smile", charts.line_chart(
    [charts.Series("lognormal world (GBM, σ = 20%)", pd.Series(iv_gbm[:, 0], index=STRIKES), role="benchmark",
                   end_label="lognormal"),
     charts.Series("crash-risk world", pd.Series(iv_crash[:, 0], index=STRIKES), role="strategy",
                   end_label="crash risk")],
    title="Implied volatility by strike, 3-month options (x = strike, $; spot = $100)",
    y_fmt=charts.fmt_pct(0), y_min=0.10, y_max=0.30, height=280))

# %% [markdown]
# ## 7 · Covered call vs holding the stock: the P&L distribution
#
# Now the *real-world* distribution matters, not the pricing one. The stock drifts at 8% a year
# with 20% volatility (the asset of Session 2.3); you own it for three months, with or without a
# 105 call sold against it at the black-box price (implied volatility = the true 20%, so the call
# is fairly priced). P&L is per share; interest on the premium is ignored.

# %%
MU_REAL, N_CC = 0.08, 200_000
S_exp_real = S0 * np.exp((MU_REAL - 0.5 * SIGMA ** 2) * TAU + SIGMA * np.sqrt(TAU) * rng_cc.standard_normal(N_CC))
pl = structure_pnl(S_exp_real)
pl_stock, pl_cc = pl["stock"], pl["covered_call"]
summary = {}
for name, x in (("stock", pl_stock), ("cc", pl_cc)):
    summary[name] = {"mean": x.mean(), "sd": x.std(ddof=1), "p_profit": (x > 0).mean(),
                     "p05": np.percentile(x, 5), "p95": np.percentile(x, 95), "max": x.max()}
    for stat, v in summary[name].items():
        lab.record(f"dist_{name}_{stat}", float(v))
print(pd.DataFrame(summary).round(3))
lab.record("dist_mu", MU_REAL)
lab.record("dist_n", N_CC)
lab.record("dist_p_called", float((S_exp_real > K_CC).mean()))


def expected_call_payoff(S, K, tau, mu, sigma):
    """E[max(S_exp − K, 0)] when S drifts at mu (not r_f): the black-box algebra with r → mu, undiscounted."""
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (mu + 0.5 * sigma ** 2) * tau) / vol
    return S * np.exp(mu * tau) * norm.cdf(d1) - K * norm.cdf(d1 - vol)


# The covered call gives up E[call payoff] and collects the premium. With drift above r_f, the call
# pays off more on average than its (risk-neutral) price, so the covered call earns less on average.
theory_gap = float(expected_call_payoff(S0, K_CC, TAU, MU_REAL, SIGMA)) - prem["cc"]
mc_gap = pl_stock - pl_cc
lab.record("dist_mean_gap", float(mc_gap.mean()))
lab.record("dist_mean_gap_theory", theory_gap)
assert abs(mc_gap.mean() - theory_gap) < 4 * mc_gap.std(ddof=1) / np.sqrt(N_CC)
assert summary["cc"]["sd"] < summary["stock"]["sd"]
assert summary["cc"]["p_profit"] > summary["stock"]["p_profit"]
# below the strike the covered call is the stock plus the premium, so the left tail shifts by exactly C
assert np.isclose(summary["cc"]["p05"] - summary["stock"]["p05"], prem["cc"])

EDGES = np.arange(-15.0, 16.0, 5.0)   # $5 buckets from −15 to 15, plus one bucket for each tail
labels = (["< −15"] + [f"{int(a)} to {int(b)}".replace("-", "−") for a, b in zip(EDGES[:-1], EDGES[1:])]
          + ["> 15"])


def bucket_probs(x):
    """Share of outcomes in each bucket: below −15, [−15, −10), …, [10, 15), 15 and above."""
    idx = np.searchsorted(EDGES, x, side="right")          # 0 = left tail … len(EDGES) = right tail
    return np.bincount(idx, minlength=len(EDGES) + 1) / x.size


probs_stock, probs_cc = bucket_probs(pl_stock), bucket_probs(pl_cc)
assert np.isclose(probs_stock.sum(), 1.0) and np.isclose(probs_cc.sum(), 1.0)
lab.chart("cc_hist", charts.grouped_column_chart(
    labels,
    [("hold the stock", probs_stock.tolist(), "benchmark"),
     (f"covered call (short the {K_CC:.0f} call)", probs_cc.tolist(), "strategy")],
    title="P&L after 3 months per share, $5 buckets (200,000 simulated paths)",
    y_fmt=charts.fmt_pct(0), height=300))
lab.record("cc_bucket_top", float(probs_cc.max()))

# %%
lab.save()
