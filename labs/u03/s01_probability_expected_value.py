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
# # 3.1 · Probability & Expected Value — companion lab
#
# **Quant Notebook** · Unit 3 · Session 1 · [Read the session](https://georgevrs.github.io/quant-notebook/unit03-math-toolkit/session01-probability-expected-value.html)
#
# Random variables, expectation, variance, conditioning and Bayes — the language of every bet and every interview question.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# Every experiment is solved twice: once exactly, with a formula, and once by simulation. The
# asserts check that the two agree within 4 standard errors of the simulation (the standard
# deviation of a simulated average over n draws is σ/√n — §5 of the session explains why).
# Why 4 and not 3: this lab makes about 30 such checks, and at 3 SE the chance that at least one
# fails by pure luck is several percent. At 4 SE it is well under 1%. (Never hunt for a seed that
# passes: that is p-hacking your own test suite.)
# All bets and markets here are toy models with round-number inputs, not market data; P&L is in
# dollars per bet, and there is no interest rate anywhere (risk-free rate 0).

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
from scipy import stats

import quantnb as qn
from quantnb import charts

lab = qn.Lab("3.1")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
(rng_bets, rng_sum, rng_regime, rng_bayes, rng_lln,
 rng_clt, rng_coin, rng_puzzles) = lab.rng.spawn(8)

# %% [markdown]
# ## 1 · Two bets: expected value and variance, exactly and by simulation
#
# A **random variable** is a number whose value depends on chance — here, the P&L of one trade.
# A discrete bet is fully described by its outcomes and their probabilities. Its **expected
# value** is the probability-weighted average outcome; its **variance** is the probability-weighted
# average squared distance from that expected value.
#
# Bet A wins rarely but big; bet B wins often but small. Stake: $10,000 per trade.

# %%
STAKE = 10_000
BETS = {  # name: (outcomes in $ per trade, probabilities)
    "a": (np.array([+300.0, -150.0]), np.array([0.40, 0.60])),   # +3% / −1.5% of the stake
    "b": (np.array([+100.0, -250.0]), np.array([0.70, 0.30])),   # +1% / −2.5% of the stake
}
COST = 10.0          # $ per trade (0.10% of the stake, round trip) — illustrative
N_SIM = 1_000_000    # simulated trades per bet


def ev_and_var(outcomes: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Closed-form expected value and variance of a discrete bet."""
    ev = np.sum(probs * outcomes)
    var = np.sum(probs * (outcomes - ev) ** 2)
    return float(ev), float(var)


def central_moment4(outcomes: np.ndarray, probs: np.ndarray) -> float:
    """E[(X − EV)⁴]: needed only for the standard error of a simulated variance."""
    ev, _ = ev_and_var(outcomes, probs)
    return float(np.sum(probs * (outcomes - ev) ** 4))


for name, (x, p) in BETS.items():
    ev, var = ev_and_var(x, p)
    draws = rng_bets.choice(x, size=N_SIM, p=p)          # simulate the same bet a million times
    sim_ev, sim_var = draws.mean(), draws.var()
    se_ev = np.sqrt(var / N_SIM)
    se_var = np.sqrt((central_moment4(x, p) - var ** 2) / N_SIM)
    assert abs(sim_ev - ev) < 4 * se_ev, f"bet {name}: simulated EV off by more than 4 SE"
    assert abs(sim_var - var) < 4 * se_var, f"bet {name}: simulated variance off by more than 4 SE"
    print(f"bet {name.upper()}: EV ${ev:+.2f} (sim ${sim_ev:+.2f} ± {se_ev:.2f}) · "
          f"sd ${np.sqrt(var):.2f} (sim ${np.sqrt(sim_var):.2f}) · hit rate {p[0]:.0%}")
    lab.record(f"{name}_win", x[0])
    lab.record(f"{name}_loss", x[1])
    lab.record(f"{name}_win_pct", x[0] / STAKE)
    lab.record(f"{name}_loss_pct", x[1] / STAKE)
    lab.record(f"{name}_p_win", p[0])
    lab.record(f"{name}_p_loss", p[1])
    lab.record(f"{name}_ev", ev)
    lab.record(f"{name}_var", var)
    lab.record(f"{name}_sd", np.sqrt(var))
    lab.record(f"{name}_ev_sim", sim_ev)
    lab.record(f"{name}_sd_sim", np.sqrt(sim_var))
    lab.record(f"{name}_ev_se", se_ev)
    lab.record(f"{name}_ev_net", ev - COST)
    lab.record(f"{name}_loss_abs", -x[1])            # losses as positive dollars, for prose
lab.record("stake", STAKE)
lab.record("cost", COST)
lab.record("n_sim", N_SIM)
# Bet A loses more often than it wins, yet has the positive EV; bet B is the reverse.
assert ev_and_var(*BETS["a"])[0] > 0 > ev_and_var(*BETS["b"])[0]

# %% [markdown]
# ## 2 · Adding bets: expectation is linear, variance is not
#
# **Linearity of expectation**: the EV of a sum is the sum of the EVs — always, whether or not
# the bets are related. Variance adds only for uncorrelated bets; in general
# Var(X + Y) = Var X + Var Y + 2 Cov(X, Y). For n bets with equal variance σ² and pairwise
# correlation ρ, Var(sum) = σ²(n + n(n − 1)ρ).
#
# Ten simultaneous copies of bet A, one set per trading day. To make them correlated we use a
# simple trick: each day, with probability ρ all ten share one coin (they win or lose together);
# otherwise each has its own. Two bets then have correlation exactly ρ.

# %%
DAYS_PER_YEAR = 252
N_POS = 10            # simultaneous positions per day
RHO = 0.3             # pairwise correlation between positions
N_DAYS_SUM = 200_000  # simulated days

x_a, p_a = BETS["a"]
ev_a, var_a = ev_and_var(x_a, p_a)
sd_a = np.sqrt(var_a)

# a year of one bet per trading day: EV and sd of the total
lab.record("days_per_year", DAYS_PER_YEAR)
lab.record("year_ev", DAYS_PER_YEAR * ev_a)
lab.record("year_sd", np.sqrt(DAYS_PER_YEAR) * sd_a)
lab.record("year_p_loss_clt", float(stats.norm.cdf(-DAYS_PER_YEAR * ev_a / (np.sqrt(DAYS_PER_YEAR) * sd_a))))
# Exact: the year loses money when the number of wins W satisfies W·win + (252 − W)·loss < 0.
w_even = DAYS_PER_YEAR * (-x_a[1]) / (x_a[0] - x_a[1])       # break-even number of wins (84)
year_p_loss_exact = float(stats.binom.cdf(np.ceil(w_even) - 1, DAYS_PER_YEAR, p_a[0]))
lab.record("year_p_loss_exact", year_p_loss_exact)
lab.record("year_wins_to_lose", int(np.ceil(w_even) - 1))     # 83 or fewer wins → a losing year


def daily_pnl(n_days: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """P&L of N_POS copies of bet A per day; any two positions have correlation rho."""
    shared = rng.random(n_days) < rho                      # days on which all positions move together
    common = rng.random(n_days) < p_a[0]                   # the shared coin (True = win)
    own = rng.random((n_days, N_POS)) < p_a[0]             # each position's own coin
    wins = np.where(shared[:, None], common[:, None], own)
    return np.where(wins, x_a[0], x_a[1])                  # shape (n_days, N_POS)


def var_of_sum(n: int, sd: float, rho: float) -> float:
    """Variance of the sum of n bets with equal sd and pairwise correlation rho."""
    return sd ** 2 * (n + n * (n - 1) * rho)


rows = []
for tag, rho in (("indep", 0.0), ("corr", RHO)):
    pnl = daily_pnl(N_DAYS_SUM, rho, rng_sum)
    total = pnl.sum(axis=1)
    theory_var = var_of_sum(N_POS, sd_a, rho)
    se_mean = np.sqrt(theory_var / N_DAYS_SUM)
    m4 = np.mean((total - total.mean()) ** 4)                # sample 4th central moment
    se_var = np.sqrt((m4 - total.var() ** 2) / N_DAYS_SUM)
    pair_corr = np.corrcoef(pnl[:, 0], pnl[:, 1])[0, 1]
    assert abs(total.mean() - N_POS * ev_a) < 4 * se_mean, "linearity: EV of the sum is the sum of EVs"
    assert abs(total.var() - theory_var) < 4 * se_var, "variance of a sum = σ²(n + n(n−1)ρ)"
    assert abs(pair_corr - rho) < 4 / np.sqrt(N_DAYS_SUM)   # SE of a sample correlation ≈ 1/√n
    rows.append({"case": tag, "rho": rho, "ev": total.mean(), "ev_theory": N_POS * ev_a,
                 "sd": total.std(), "sd_theory": np.sqrt(theory_var), "pair_corr": pair_corr})
sums = pd.DataFrame(rows).set_index("case")
print(sums.round(3))
lab.record("n_pos", N_POS)
lab.record("rho", RHO)
lab.record("sum_ev", N_POS * ev_a)
lab.record("n_days_sum", N_DAYS_SUM)
for tag in ("indep", "corr"):
    lab.record(f"sum_{tag}_ev_sim", float(sums.loc[tag, "ev"]))
    lab.record(f"sum_{tag}_sd", float(sums.loc[tag, "sd_theory"]))
    lab.record(f"sum_{tag}_sd_sim", float(sums.loc[tag, "sd"]))
    lab.record(f"sum_{tag}_pair_corr_sim", float(sums.loc[tag, "pair_corr"]))
lab.record("sum_sd_ratio", float(sums.loc["corr", "sd_theory"] / sums.loc["indep", "sd_theory"]))
lab.record("sum_var_factor", N_POS + N_POS * (N_POS - 1) * RHO)   # σ² × this = variance of the sum

# %% [markdown]
# ## 3 · Conditioning and independence: zero correlation is not independence
#
# A toy market with two volatility regimes. Each day is *calm* (daily sd 0.8%) or *stormy*
# (2.4%); regimes persist (a Markov chain). Given the regime, the return is normal with mean 0.
# Because the sign of every return is a fair coin, today's return tells you nothing about the
# *direction* of tomorrow's: their correlation is exactly 0. But a big move today makes a big
# move tomorrow far more likely — the two days are **not independent**.
#
# Closed forms (π = stationary probabilities, λ = 1 − a − b = persistence of the chain):
# * P(|r| > c) = Σ_s π_s · 2Φ(−c/σ_s)
# * P(both days big) = Σ_{s,s'} π_s P(s→s') · 2Φ(−c/σ_s) · 2Φ(−c/σ_s')
# * corr(r_t², r_{t+1}²) = λ Var(σ²) / (3 E[σ⁴] − E[σ²]²)

# %%
SIG = np.array([0.008, 0.024])        # daily sd in the calm and the stormy regime
P_CALM_TO_STORM, P_STORM_TO_CALM = 0.02, 0.08
BIG = 0.03                            # a "big move": |return| above 3%
N_PATHS_REG, N_DAYS_REG = 1_000, 2_520   # 1,000 independent 10-year histories

TRANS = np.array([[1 - P_CALM_TO_STORM, P_CALM_TO_STORM],
                  [P_STORM_TO_CALM, 1 - P_STORM_TO_CALM]])
pi = np.array([P_STORM_TO_CALM, P_CALM_TO_STORM]) / (P_CALM_TO_STORM + P_STORM_TO_CALM)  # stationary
lam = 1 - P_CALM_TO_STORM - P_STORM_TO_CALM
p_big_state = 2 * stats.norm.sf(BIG / SIG)                         # P(big | regime)
p_big = float(pi @ p_big_state)
p_both_big = float(np.sum(pi[:, None] * TRANS * np.outer(p_big_state, p_big_state)))
p_big_given_big = p_both_big / p_big
s2 = SIG ** 2
e_s2, e_s4 = pi @ s2, pi @ s2 ** 2
corr_sq_theory = float(lam * (e_s4 - e_s2 ** 2) / (3 * e_s4 - e_s2 ** 2))


def simulate_regime_returns(n_paths: int, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Daily returns (n_days, n_paths) from the two-regime model, starting in the stationary mix."""
    state = (rng.random(n_paths) < pi[1]).astype(int)
    out = np.empty((n_days, n_paths))
    for t in range(n_days):
        out[t] = SIG[state] * rng.standard_normal(n_paths)
        u = rng.random(n_paths)
        state = np.where(state == 0, (u < P_CALM_TO_STORM).astype(int), (u >= P_STORM_TO_CALM).astype(int))
    return out


def ratio_and_se(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Pooled ratio Σnum/Σden over independent paths, with its delta-method standard error."""
    r = num.sum() / den.sum()
    return float(r), float(np.sqrt(np.sum((num - r * den) ** 2)) / den.sum())


r = simulate_regime_returns(N_PATHS_REG, N_DAYS_REG, rng_regime)
today, tomorrow = r[:-1], r[1:]
big_today, big_tomorrow = np.abs(today) > BIG, np.abs(tomorrow) > BIG
# P(big tomorrow) and P(big tomorrow | big today), pooled over paths
p_big_sim, p_big_se = ratio_and_se(big_tomorrow.sum(axis=0), np.full(N_PATHS_REG, N_DAYS_REG - 1.0))
p_bb_sim, p_bb_se = ratio_and_se((big_today & big_tomorrow).sum(axis=0), big_today.sum(axis=0))
# lag-1 correlation of returns (true mean 0) and of squared returns (per path, then averaged)
corr_r_sim, corr_r_se = ratio_and_se((today * tomorrow).sum(axis=0), (today ** 2).sum(axis=0))
sq = r ** 2
corr_sq_paths = np.array([np.corrcoef(sq[:-1, i], sq[1:, i])[0, 1] for i in range(N_PATHS_REG)])
corr_sq_sim, corr_sq_se = corr_sq_paths.mean(), corr_sq_paths.std(ddof=1) / np.sqrt(N_PATHS_REG)

print(f"P(big) {p_big:.4f} (sim {p_big_sim:.4f}) · P(big | big yesterday) {p_big_given_big:.4f} (sim {p_bb_sim:.4f})")
print(f"corr(r_t, r_t+1) 0 (sim {corr_r_sim:+.4f} ± {corr_r_se:.4f}) · "
      f"corr(r², r²) {corr_sq_theory:.3f} (sim {corr_sq_sim:.3f} ± {corr_sq_se:.3f})")
assert abs(p_big_sim - p_big) < 4 * p_big_se
assert abs(p_bb_sim - p_big_given_big) < 4 * p_bb_se
assert abs(corr_r_sim) < 4 * corr_r_se, "returns are uncorrelated day to day"
# Per-path sample autocorrelations of a persistent series are biased by O(1/T): with T = 2,520 days
# and regimes lasting ~10 days the expected bias is about 0.002, so we allow 0.003 on top of 4 SE.
CORR_SQ_BIAS_ALLOWANCE = 0.003
assert abs(corr_sq_sim - corr_sq_theory) < 4 * corr_sq_se + CORR_SQ_BIAS_ALLOWANCE, "…but squared returns are not"
assert p_big_given_big > 3 * p_big, "a big move today makes a big move tomorrow far more likely"

lab.record("reg_sig_calm", SIG[0])
lab.record("reg_sig_storm", SIG[1])
lab.record("reg_p_cs", P_CALM_TO_STORM)
lab.record("reg_p_sc", P_STORM_TO_CALM)
lab.record("reg_pi_storm", float(pi[1]))
lab.record("reg_big", BIG)
lab.record("reg_years", N_DAYS_REG / DAYS_PER_YEAR)
lab.record("reg_paths", N_PATHS_REG)
lab.record("reg_p_big", p_big)
lab.record("reg_p_big_sim", p_big_sim)
lab.record("reg_p_big_given_big", p_big_given_big)
lab.record("reg_p_big_given_big_sim", p_bb_sim)
lab.record("reg_big_ratio", p_big_given_big / p_big)
lab.record("reg_corr_r_sim", corr_r_sim)
lab.record("reg_corr_r_se", corr_r_se)
lab.record("reg_corr_sq", corr_sq_theory)
lab.record("reg_corr_sq_sim", float(corr_sq_sim))

# %% [markdown]
# ## 4 · Bayes' rule: the model said long — how likely is an up-day?
#
# A deliberately strong toy classifier. 52% of days are up days. On up days the model says
# "long" 70% of the time (its *recall* for up days); on down days it still says "long" 55% of the
# time. Bayes' rule turns P(long | up) — what the model's builder measured — into P(up | long) —
# what the trader needs:
#
# P(up | long) = P(long | up) P(up) / [P(long | up) P(up) + P(long | down) P(down)]
#
# The size of each day's move is exponential with mean 1%, whatever the signal said (the signal
# only knows the direction), so E[R | long] = P(up | long)·1% − P(down | long)·1%.

# %%
P_UP = 0.52
P_LONG_GIVEN_UP = 0.70
P_LONG_GIVEN_DOWN = 0.55
MEAN_MOVE = 0.01          # average absolute daily move
N_DAYS_BAYES = 1_000_000


def posterior(prior: float, p_signal_if_true: float, p_signal_if_false: float) -> float:
    """Bayes' rule for a yes/no hypothesis: P(true | signal)."""
    joint_true = p_signal_if_true * prior
    return joint_true / (joint_true + p_signal_if_false * (1 - prior))


p_long = P_LONG_GIVEN_UP * P_UP + P_LONG_GIVEN_DOWN * (1 - P_UP)
post_up_long = posterior(P_UP, P_LONG_GIVEN_UP, P_LONG_GIVEN_DOWN)
post_down_short = posterior(1 - P_UP, 1 - P_LONG_GIVEN_DOWN, 1 - P_LONG_GIVEN_UP)
accuracy = P_LONG_GIVEN_UP * P_UP + (1 - P_LONG_GIVEN_DOWN) * (1 - P_UP)
ev_given_long = (2 * post_up_long - 1) * MEAN_MOVE          # E[R | long]
ev_given_short = (1 - 2 * post_down_short) * MEAN_MOVE      # E[R | short] (negative: shorting pays)
ev_all = (2 * P_UP - 1) * MEAN_MOVE                         # E[R], no signal
# law of total expectation: E[R] = P(long) E[R | long] + P(short) E[R | short]
assert np.isclose(ev_all, p_long * ev_given_long + (1 - p_long) * ev_given_short)

up = rng_bayes.random(N_DAYS_BAYES) < P_UP
says_long = rng_bayes.random(N_DAYS_BAYES) < np.where(up, P_LONG_GIVEN_UP, P_LONG_GIVEN_DOWN)
ret = np.where(up, 1.0, -1.0) * rng_bayes.exponential(MEAN_MOVE, N_DAYS_BAYES)

n_long = says_long.sum()
sim_post_long = up[says_long].mean()
sim_post_short = (~up[~says_long]).mean()
sim_ev_long = ret[says_long].mean()
sim_ev_short = ret[~says_long].mean()
se_post_long = np.sqrt(post_up_long * (1 - post_up_long) / n_long)
se_post_short = np.sqrt(post_down_short * (1 - post_down_short) / (N_DAYS_BAYES - n_long))
assert abs(sim_post_long - post_up_long) < 4 * se_post_long
assert abs(sim_post_short - post_down_short) < 4 * se_post_short
assert abs(sim_ev_long - ev_given_long) < 4 * ret[says_long].std() / np.sqrt(n_long)
assert abs(sim_ev_short - ev_given_short) < 4 * ret[~says_long].std() / np.sqrt(N_DAYS_BAYES - n_long)
print(f"P(up | long) {post_up_long:.4f} (sim {sim_post_long:.4f}) · P(down | short) {post_down_short:.4f} "
      f"(sim {sim_post_short:.4f}) · E[R | long] {ev_given_long:.4%} (sim {sim_ev_long:.4%})")

lab.record("bay_p_up", P_UP)
lab.record("bay_p_down", 1 - P_UP)
lab.record("bay_p_long_up", P_LONG_GIVEN_UP)
lab.record("bay_p_long_down", P_LONG_GIVEN_DOWN)
lab.record("bay_p_short_down", 1 - P_LONG_GIVEN_DOWN)
lab.record("bay_p_long", p_long)
lab.record("bay_post_up_long", post_up_long)
lab.record("bay_post_up_long_sim", sim_post_long)
lab.record("bay_post_down_short", post_down_short)
lab.record("bay_post_down_short_sim", sim_post_short)
lab.record("bay_accuracy", accuracy)
lab.record("bay_mean_move", MEAN_MOVE)
lab.record("bay_ev_long", ev_given_long)
lab.record("bay_ev_long_sim", sim_ev_long)
lab.record("bay_ev_short", ev_given_short)
lab.record("bay_ev_short_sim", sim_ev_short)
lab.record("bay_ev_short_trade", -ev_given_short)     # what a short position earns on a short signal
lab.record("bay_ev_all", ev_all)
lab.record("bay_n_days", N_DAYS_BAYES)
# the same arithmetic as natural frequencies: out of 1,000 days
NAT_DAYS = 1_000
lab.record("nat_days", NAT_DAYS)
for key, value in {"up": P_UP, "down": 1 - P_UP,
                   "up_long": P_UP * P_LONG_GIVEN_UP, "down_long": (1 - P_UP) * P_LONG_GIVEN_DOWN,
                   "long": p_long}.items():
    lab.record(f"nat_{key}", NAT_DAYS * value)

# %% [markdown]
# ## 5 · The law of large numbers: how many bets before an edge shows?
#
# A 51/49 bet: win or lose 1% of the stake ($100). The **law of large numbers** says the average
# P&L per bet converges to the EV ($2). How fast? The average of n independent bets has standard
# deviation σ/√n (its standard error). Two different questions have two different answers:
#
# * When does the 95% band of luck around the TRUE edge stop covering break-even?
#   When EV > 1.96 σ/√n, i.e. after n* = (1.96 σ / EV)² bets. By then about 97.5% of players are ahead.
# * When will a player's OWN significance test (t = average / (s/√n) > 1.96) be likely to confirm
#   the edge? At n* only about half the time: a player who earned exactly the edge would just
#   pass, and half of players earn less. An 80% chance needs n₈₀ = ((1.96 + 0.84) σ / EV)², about
#   twice n*. (And lucky players can pass long before n*.)

# %%
P_WIN_LLN = 0.51
WIN_LLN = 0.01 * STAKE        # $100
N_BETS_LLN = 20_000           # bets per simulated player (chart)
N_PLAYERS_LLN = 3             # players drawn on the chart
FIRST_SHOWN = 1_000           # the chart starts after 1,000 bets, when the average has settled a little
N_PLAYERS_P = 200_000         # players for the P(ahead) check
N_GRID = [100, 1_000, 5_000, 10_000, 20_000]

ev_lln = WIN_LLN * (2 * P_WIN_LLN - 1)
sd_lln = WIN_LLN * np.sqrt(1 - (2 * P_WIN_LLN - 1) ** 2)
Z_95 = stats.norm.ppf(0.975)                   # 1.96: two-sided 5% test
Z_POWER80 = stats.norm.ppf(0.80)               # 0.84: an 80% chance of passing it
n_star = (Z_95 * sd_lln / ev_lln) ** 2
n_80 = ((Z_95 + Z_POWER80) * sd_lln / ev_lln) ** 2
lab.record("lln_n_80", n_80)
lab.record("lln_years_80", n_80 / DAYS_PER_YEAR)
lab.record("lln_z80", Z_95 + Z_POWER80)
lab.record("lln_p_win", P_WIN_LLN)
lab.record("lln_p_loss", 1 - P_WIN_LLN)
lab.record("lln_win", WIN_LLN)
lab.record("lln_ev", ev_lln)
lab.record("lln_sd", sd_lln)
lab.record("lln_n_star", n_star)
lab.record("lln_edge_ratio", ev_lln / sd_lln)          # the edge as a fraction of one bet's noise
lab.record("lln_years_star", n_star / DAYS_PER_YEAR)
lab.record("lln_n_bets", N_BETS_LLN)

steps = np.where(rng_lln.random((N_PLAYERS_LLN, N_BETS_LLN)) < P_WIN_LLN, WIN_LLN, -WIN_LLN)
running = steps.cumsum(axis=1) / np.arange(1, N_BETS_LLN + 1)
for i in range(N_PLAYERS_LLN):
    lab.record(f"lln_player{i + 1}_final", running[i, -1])
lab.record("lln_final_min", running[:, -1].min())
lab.record("lln_final_max", running[:, -1].max())

def t_stat_from_wins(wins: np.ndarray, n: int) -> np.ndarray:
    """A player's own t-statistic after n bets of ±WIN_LLN: average / (sample sd / √n)."""
    mean = WIN_LLN * (2 * wins / n - 1)
    sd = np.sqrt(n / (n - 1) * np.maximum(WIN_LLN ** 2 - mean ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mean * np.sqrt(n) / sd, np.sign(mean) * np.inf)


def p_ahead_exact(n: int) -> float:
    """P(total P&L > 0 after n bets) = P(wins > n/2), exactly, from the binomial distribution."""
    return float(stats.binom.sf(n // 2, n, P_WIN_LLN))


def p_significant_exact(n: int) -> float:
    """P(the player's own t-statistic exceeds 1.96 after n bets), exactly: sum over all win counts."""
    w = np.arange(n + 1)
    return float(stats.binom.pmf(w, n, P_WIN_LLN)[t_stat_from_wins(w, n) > Z_95].sum())


# P(ahead) and P(own test says "edge") after n bets — exact binomial vs simulation
lln_rows = []
for n in N_GRID:
    wins = rng_lln.binomial(n, P_WIN_LLN, size=N_PLAYERS_P)
    ahead, ahead_sim = p_ahead_exact(n), float((wins > n / 2).mean())
    signif, signif_sim = p_significant_exact(n), float((t_stat_from_wins(wins, n) > Z_95).mean())
    assert abs(ahead_sim - ahead) < 4 * np.sqrt(ahead * (1 - ahead) / N_PLAYERS_P)
    assert abs(signif_sim - signif) < 4 * np.sqrt(signif * (1 - signif) / N_PLAYERS_P)
    lln_rows.append({"n": n, "ev_total": n * ev_lln, "sd_total": np.sqrt(n) * sd_lln,
                     "p_ahead": ahead, "p_ahead_sim": ahead_sim, "p_signif": signif, "p_signif_sim": signif_sim})
    lab.record(f"lln_{n}_bets", n)
    lab.record(f"lln_{n}_ev_total", n * ev_lln)
    lab.record(f"lln_{n}_sd_total", np.sqrt(n) * sd_lln)
    lab.record(f"lln_{n}_p_ahead", ahead)
    lab.record(f"lln_{n}_p_ahead_sim", ahead_sim)
    lab.record(f"lln_{n}_p_signif", signif)
    lab.record(f"lln_{n}_p_signif_sim", signif_sim)
print(pd.DataFrame(lln_rows).set_index("n").round(4))
# with a positive edge, both chances only grow as the bets pile up
assert all(a["p_ahead"] < b["p_ahead"] for a, b in zip(lln_rows, lln_rows[1:]))
assert all(a["p_signif"] < b["p_signif"] for a, b in zip(lln_rows, lln_rows[1:]))

# at n*, the band clears break-even (≈97.5% of players ahead) but a player's own test is a coin flip
n_star_int = int(round(n_star))
lab.record("lln_p_ahead_at_nstar", p_ahead_exact(n_star_int))
lab.record("lln_p_signif_at_nstar", p_significant_exact(n_star_int))
lab.record("lln_p_signif_at_n80", p_significant_exact(int(round(n_80))))
# The normal approximation predicts 97.5%, 50% and 80%; the exact binomial answers differ only by
# discreteness and by using the sample sd, hence these small (non-Monte-Carlo) tolerances.
assert abs(p_ahead_exact(n_star_int) - 0.975) < 0.01
assert abs(p_significant_exact(n_star_int) - 0.5) < 0.03
assert abs(p_significant_exact(int(round(n_80))) - 0.8) < 0.03
print(f"n* = {n_star:,.0f}: P(ahead) {p_ahead_exact(n_star_int):.3f}, P(own test passes) "
      f"{p_significant_exact(n_star_int):.3f} · n80 = {n_80:,.0f}: P(pass) {p_significant_exact(int(round(n_80))):.3f}")

x_bets = np.arange(1, N_BETS_LLN + 1)
shown = slice(FIRST_SHOWN - 1, None)
lab.record("lln_first_shown", FIRST_SHOWN)
band = 1.96 * sd_lln / np.sqrt(x_bets)
roles = ["strategy", "alt1", "alt2"]
lln_series = [charts.Series(f"player {i + 1}", pd.Series(running[i, shown], index=x_bets[shown]),
                            role=roles[i], end_label=f"player {i + 1}") for i in range(N_PLAYERS_LLN)]
lln_series += [
    charts.Series("edge + 1.96 SE", pd.Series(ev_lln + band[shown], index=x_bets[shown]), role="benchmark",
                  end_label="+1.96 SE"),
    charts.Series("edge − 1.96 SE", pd.Series(ev_lln - band[shown], index=x_bets[shown]), role="benchmark",
                  end_label="−1.96 SE"),
]
lab.chart("lln_paths", charts.line_chart(
    lln_series, title="Average P&L per bet as bets pile up — a 51/49 bet of ±$100",
    y_fmt=charts.fmt_num(2, prefix="$"), hline=0.0, height=320))

# %% [markdown]
# ## 6 · The central limit theorem — and when it is slow
#
# The **central limit theorem**: the average of n independent draws with finite variance,
# standardised as Z = (average − μ)/(σ/√n), looks more and more like a standard normal as n grows.
# How fast depends on the tails. We compare a thin-tailed uniform with a heavy-tailed Student-t
# with 2.5 degrees of freedom (finite variance ν/(ν − 2) = 5, but no finite third moment).
# Yardsticks: a standard normal lands within ±1 with probability 68.3% and beyond ±3 with 0.27%.

# %%
NU = 2.5
N_CLT = [1, 10, 30, 100, 1_000]
N_REPS = 100_000               # sample means per (distribution, n)
N_HIST = 30                    # the n shown in the two histograms
HIST_EDGES = np.linspace(-5, 5, 51)
# The uniform's average is not exactly normal at finite n. The Edgeworth expansion puts its gap
# from the normal (excess kurtosis −1.2/n) below these for n ≥ 30, so we allow them on top of 4 SE:
EDGEWORTH_WITHIN1 = 0.002      # for P(|Z| ≤ 1): the gap is about 0.0016 at n = 30
EDGEWORTH_BEYOND3 = 0.0004     # for P(|Z| > 3): the gap is about 0.0003 at n = 30

sd_uniform, sd_t = np.sqrt(1 / 12), np.sqrt(NU / (NU - 2))
p_within1_normal = float(2 * stats.norm.cdf(1) - 1)
p_beyond3_normal = float(2 * stats.norm.sf(3))


def standardised_means(draw, sd: float, mu: float, n: int, reps: int) -> np.ndarray:
    """reps sample means of n draws, standardised by the CLT scale sd/√n (computed in chunks)."""
    out, chunk = [], max(1, 2_000_000 // n)
    for start in range(0, reps, chunk):
        m = min(chunk, reps - start)
        out.append((draw((m, n)).mean(axis=1) - mu) / (sd / np.sqrt(n)))
    return np.concatenate(out)


def tail_stats(z: np.ndarray) -> tuple[float, float]:
    return float((np.abs(z) <= 1).mean()), float((np.abs(z) > 3).mean())


clt_rows, hist_z = [], {}
for n in N_CLT:
    zu = standardised_means(rng_clt.random, sd_uniform, 0.5, n, N_REPS)
    zt = standardised_means(lambda size: rng_clt.standard_t(NU, size), sd_t, 0.0, n, N_REPS)
    (wu, bu), (wt, bt) = tail_stats(zu), tail_stats(zt)
    clt_rows.append({"n": n, "unif_within1": wu, "unif_beyond3": bu, "t_within1": wt, "t_beyond3": bt})
    for key, v in {"unif_within1": wu, "unif_beyond3": bu, "t_within1": wt, "t_beyond3": bt}.items():
        lab.record(f"clt_{n}_{key}", v)
    lab.record(f"clt_{n}_n", n)
    if n == N_HIST:
        hist_z = {"uniform": zu, "t": zt}
clt = pd.DataFrame(clt_rows).set_index("n")
print(clt.round(4))


def se_p(p: float) -> float:
    return np.sqrt(p * (1 - p) / N_REPS)


# n = 1 is exact: a standardised uniform lives in ±√3, so it is never beyond ±3
assert np.isclose(clt.loc[1, "unif_within1"], 1 / np.sqrt(3), atol=4 * se_p(1 / np.sqrt(3)))
assert clt.loc[1, "unif_beyond3"] == 0.0
exact_t_within1 = float(2 * stats.t.cdf(sd_t, NU) - 1)
exact_t_beyond3 = float(2 * stats.t.sf(3 * sd_t, NU))
assert abs(clt.loc[1, "t_within1"] - exact_t_within1) < 4 * se_p(exact_t_within1)
assert abs(clt.loc[1, "t_beyond3"] - exact_t_beyond3) < 4 * se_p(exact_t_beyond3)
# uniform: normal already at n = 30
for n in (30, 100, 1_000):
    assert abs(clt.loc[n, "unif_within1"] - p_within1_normal) < 4 * se_p(p_within1_normal) + EDGEWORTH_WITHIN1
    assert abs(clt.loc[n, "unif_beyond3"] - p_beyond3_normal) < 4 * se_p(p_beyond3_normal) + EDGEWORTH_BEYOND3
if NU <= 3:  # no finite third moment: the Berry–Esseen 1/√n guarantee does not apply
    # still clearly not normal at n = 1,000 — too peaked in the middle, too many extremes
    assert clt.loc[1_000, "t_within1"] - p_within1_normal > 4 * se_p(p_within1_normal)
    assert clt.loc[1_000, "t_beyond3"] - p_beyond3_normal > 4 * se_p(p_beyond3_normal)
    assert clt.loc[1_000, "t_within1"] < clt.loc[10, "t_within1"] - 4 * se_p(clt.loc[10, "t_within1"])  # …but converging

lab.record("clt_nu", NU)
lab.record("clt_var_t", NU / (NU - 2))
lab.record("clt_reps", N_REPS)
lab.record("clt_n_hist", N_HIST)
lab.record("clt_normal_within1", p_within1_normal)
lab.record("clt_normal_beyond3", p_beyond3_normal)
lab.record("clt_t_ratio_beyond3_1000", clt.loc[1_000, "t_beyond3"] / p_beyond3_normal)
lab.record("clt_t_ratio_beyond3_30", clt.loc[30, "t_beyond3"] / p_beyond3_normal)
lab.record("clt_t_offchart", float((np.abs(hist_z["t"]) > 5).mean()))

# %% [markdown]
# **What an analyst actually computes.** Above, averages were scaled by the TRUE σ. A real
# confidence interval scales by the SAMPLE standard deviation, which the same wild draws inflate —
# so the t-statistic mean / (s/√n) is far better behaved than Z_n. The real danger with fat tails
# is quoting normal tail probabilities for single returns: the chance that one month falls more
# than 3.09 standard deviations below its mean (a 1-in-1,000 event for a normal).

# %%
t_draws = rng_clt.standard_t(NU, (N_REPS, N_HIST))
t_stat = t_draws.mean(axis=1) / (t_draws.std(axis=1, ddof=1) / np.sqrt(N_HIST))
tstat_cover196 = float((np.abs(t_stat) <= 1.96).mean())
tstat_beyond3 = float((np.abs(t_stat) > 3).mean())
if NU <= 3:  # with very fat tails, the sample-sd version is far less extreme than the true-σ version
    assert clt.loc[N_HIST, "t_beyond3"] - tstat_beyond3 > 4 * se_p(clt.loc[N_HIST, "t_beyond3"])
Z_1IN1000 = stats.norm.ppf(0.001)                                   # −3.09
month_tail_t = float(stats.t.cdf(Z_1IN1000 * sd_t, NU))            # unit-variance Student-t, exact
print(f"t-statistic with sample sd, n = {N_HIST}: |t| ≤ 1.96 {tstat_cover196:.1%}, |t| > 3 {tstat_beyond3:.2%} · "
      f"single-return P(< −3.09 sd): Student-t {month_tail_t:.2%} vs normal 0.10%")
lab.record("clt_tstat_cover196", tstat_cover196)
lab.record("clt_tstat_beyond3", tstat_beyond3)
lab.record("clt_month_tail_t", month_tail_t)
lab.record("clt_month_tail_normal", 0.001)
lab.record("clt_month_tail_ratio", month_tail_t / 0.001)
lab.record("clt_z_1in1000", -Z_1IN1000)

grid = np.linspace(-5, 5, 201)
normal_pdf = (grid, stats.norm.pdf(grid))
lab.chart("clt_uniform", charts.histogram(
    hist_z["uniform"], title=f"Averages of {N_HIST} uniform draws, standardised",
    bins=HIST_EDGES, x_fmt=charts.fmt_num(0), density_overlay=normal_pdf, overlay_label="normal (CLT)"))
lab.chart("clt_t", charts.histogram(
    hist_z["t"], title=f"Averages of {N_HIST} Student-t({NU:g}) draws, standardised",
    bins=HIST_EDGES, x_fmt=charts.fmt_num(0), density_overlay=normal_pdf, overlay_label="normal (CLT)"))

# %% [markdown]
# ## 7 · Expected value is not what you experience
#
# The coin toss from Peters & Gell-Mann (2016): heads, your wealth grows 50%; tails, it shrinks
# 40%. Each round you stake everything you have. The EV per round is +5%, so expected wealth
# grows as 1.05^t. But what compounds is the average *log* growth, ½ ln 1.5 + ½ ln 0.6 = ½ ln 0.9,
# which is negative: over many rounds the typical player shrinks by about 5% a round. The mean is
# carried by the few players who flip many more heads than tails (wealth depends on the count of
# heads, not their order).

# %%
UP_MULT, DOWN_MULT = 1.5, 0.6
N_PLAYERS_COIN = 100_000
ROUNDS = 20

ev_round = 0.5 * UP_MULT + 0.5 * DOWN_MULT - 1                       # +5%
g_round = 0.5 * np.log(UP_MULT) + 0.5 * np.log(DOWN_MULT)            # ½ ln 0.9 < 0
sd_round = 0.5 * (UP_MULT - DOWN_MULT)                               # sd of the one-round return
heads = rng_coin.random((N_PLAYERS_COIN, ROUNDS)) < 0.5
wealth = np.cumprod(np.where(heads, UP_MULT, DOWN_MULT), axis=1)
wealth = np.hstack([np.ones((N_PLAYERS_COIN, 1)), wealth])          # round 0: everyone has $1
mean_path, median_path = wealth.mean(axis=0), np.median(wealth, axis=0)

mean_theory = (1 + ev_round) ** ROUNDS
median_theory = (UP_MULT * DOWN_MULT) ** (ROUNDS / 2)                # 10 heads, 10 tails
k_even = np.log(1 / DOWN_MULT) / np.log(UP_MULT / DOWN_MULT)         # share of heads needed to break even
p_lose = float(stats.binom.cdf(np.ceil(k_even * ROUNDS) - 1, ROUNDS, 0.5))
final = wealth[:, -1]
p_lose_sim = float((final < 1).mean())
assert abs(final.mean() - mean_theory) < 4 * final.std() / np.sqrt(N_PLAYERS_COIN)
assert np.isclose(np.median(final), median_theory)
assert abs(p_lose_sim - p_lose) < 4 * np.sqrt(p_lose * (1 - p_lose) / N_PLAYERS_COIN)
top1 = np.sort(final)[-N_PLAYERS_COIN // 100:].sum() / final.sum()
print(f"EV/round {ev_round:+.1%} · log growth/round {g_round:+.4f} · after {ROUNDS} rounds: mean "
      f"{final.mean():.3f} (theory {mean_theory:.3f}), median {np.median(final):.3f}, P(lose) {p_lose_sim:.3f} "
      f"(exact {p_lose:.3f}), top 1% hold {top1:.0%}")

lab.record("coin_up", UP_MULT - 1)
lab.record("coin_down", 1 - DOWN_MULT)
lab.record("coin_ev_round", ev_round)
lab.record("coin_g_round", g_round)
lab.record("coin_typical_round", np.expm1(g_round))            # typical growth per round
lab.record("coin_sd_round", sd_round)
lab.record("coin_drag_approx", ev_round - sd_round ** 2 / 2)   # 2.3's μ − σ²/2 approximation
lab.record("coin_rounds", ROUNDS)
lab.record("coin_players", N_PLAYERS_COIN)
lab.record("coin_mean_theory", mean_theory)
lab.record("coin_mean_sim", float(final.mean()))
lab.record("coin_median", median_theory)
lab.record("coin_p_lose", p_lose)
lab.record("coin_p_lose_sim", p_lose_sim)
lab.record("coin_top1_share", float(top1))
lab.record("coin_heads_needed", int(np.ceil(k_even * ROUNDS)))

rounds_ix = np.arange(ROUNDS + 1)
even = rounds_ix % 2 == 0   # after an even number of rounds the median player has exactly half heads
lab.chart("coin_growth", charts.line_chart(
    [charts.Series("average across players", pd.Series(mean_path, index=rounds_ix), role="strategy",
                   end_label="average"),
     charts.Series("median player", pd.Series(median_path[even], index=rounds_ix[even]), role="alt1",
                   end_label="median")],
    title=f"The +50% / −40% coin: wealth of {N_PLAYERS_COIN:,} players (log scale)", logy=True,
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

# %% [markdown]
# ## 8 · Four interview classics, solved exactly and by Monte Carlo
#
# 1. **Die with one re-roll.** You are paid the face of a die; after the first roll you may re-roll
#    once. Keep the first roll only if it beats the EV of a fresh roll (3.5): EV = ½·5 + ½·3.5 = 4.25.
# 2. **HH vs HT.** Expected fair-coin flips until HH appears: 6. Until HT appears: 4.
# 3. **First ace.** Expected position of the first ace in a shuffled deck: 1 + 48/5 = 10.6
#    (each of the 48 other cards precedes all four aces with probability 1/5 — linearity).
# 4. **Coin from a bag.** One of 10 coins has two heads. You pick one at random, flip it 3 times and
#    see 3 heads. P(two-headed) = 1 / (1 + 9/8) = 8/17.

# %%
N_PUZZLE = 400_000


def mc_mean(x: np.ndarray) -> tuple[float, float]:
    """Monte Carlo estimate and its standard error."""
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


# 1 · die with one re-roll: keep 4, 5, 6; re-roll 1, 2, 3
first = rng_puzzles.integers(1, 7, N_PUZZLE)
second = rng_puzzles.integers(1, 7, N_PUZZLE)
payout = np.where(first >= 4, first, second)
exact_reroll = 0.5 * np.mean([4, 5, 6]) + 0.5 * 3.5


def flips_until(pattern: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Number of fair-coin flips until `pattern` (two letters of H/T) first appears, for n players."""
    want_first, want_second = pattern[0] == "H", pattern[1] == "H"
    count = np.zeros(n, dtype=int)
    prev = np.zeros(n, dtype=bool)          # was the previous flip heads?
    done = np.zeros(n, dtype=bool)
    while not done.all():
        flip = rng.random(n) < 0.5          # True = heads
        live = ~done
        count[live] += 1
        done |= live & (count >= 2) & (prev == want_first) & (flip == want_second)
        prev = np.where(live, flip, prev)
    return count


hh = flips_until("HH", N_PUZZLE, rng_puzzles)
ht = flips_until("HT", N_PUZZLE, rng_puzzles)

# 3 · first ace: cards 0–3 are the aces
decks = rng_puzzles.permuted(np.tile(np.arange(52), (N_PUZZLE, 1)), axis=1)
first_ace = np.argmax(decks < 4, axis=1) + 1
exact_first_ace = 1 + 48 / 5

# 4 · coin from a bag, three heads seen
two_headed = rng_puzzles.random(N_PUZZLE) < 0.1
three_heads = two_headed | (rng_puzzles.random((N_PUZZLE, 3)) < 0.5).all(axis=1)
exact_bag = posterior(0.1, 1.0, 0.5 ** 3)

puzzles = {  # key: (exact, Monte Carlo draws)
    "reroll": (exact_reroll, payout),
    "hh": (6.0, hh),
    "ht": (4.0, ht),
    "first_ace": (exact_first_ace, first_ace),
    "bag": (exact_bag, two_headed[three_heads].astype(float)),
}
gaps = {}
for key, (exact, draws) in puzzles.items():
    est, se = mc_mean(draws)
    assert abs(est - exact) < 4 * se, f"{key}: Monte Carlo disagrees with the exact answer"
    gaps[key] = abs(est - exact) / se                       # the gap, in standard errors
    lab.record(f"pz_{key}_exact", float(exact))
    lab.record(f"pz_{key}_mc", est)
    lab.record(f"pz_{key}_se", se)
    print(f"{key:>9}: exact {exact:.4f} · Monte Carlo {est:.4f} ± {se:.4f} ({gaps[key]:.1f} SE away)")
assert np.isclose(exact_bag, 8 / 17)
lab.record("pz_max_gap_se", max(gaps.values()))            # honest simulation: a gap near 3 SE happens
lab.record("pz_n", N_PUZZLE)
lab.record("pz_bag_n_cond", int(three_heads.sum()))

# %%
lab.save()
