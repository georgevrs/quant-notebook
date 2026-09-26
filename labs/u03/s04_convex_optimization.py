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
# # 3.4 · Convex Optimization — companion lab
#
# **Quant Notebook** · Unit 3 · Session 4 · [Read the session](https://georgevrs.github.io/quant-notebook/unit03-math-toolkit/session04-convex-optimization.html)
#
# Objectives, constraints, convexity, quadratic programs and Lagrange multipliers — and how to state a portfolio problem so cvxpy can solve it.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# What you will do:
#
# 1. Build a covariance matrix from synthetic returns (a known 3-factor model).
# 2. Solve the minimum-variance portfolio in closed form and with cvxpy — and check they agree.
# 3. Add long-only and position-cap constraints and see which ones bind.
# 4. Read the Lagrange multipliers as prices, and check them against finite differences.
# 5. Verify the KKT conditions yourself instead of trusting the solver.
# 6. Watch cvxpy reject non-DCP formulations, and rewrite one so it is accepted.
# 7. Provoke `infeasible` and `unbounded`, and measure solver tolerance.
# 8. Rebalance with an L1 turnover penalty (a preview of Session 9.6).
#
# Synthetic excess returns, risk-free rate 0. Timings are printed, never recorded: they depend on
# your machine.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)
if importlib.util.find_spec("cvxpy") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cvxpy"], check=True)

# %%
import time

import cvxpy as cp
import numpy as np
import pandas as pd

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR

lab = qn.Lab("3.4")  # seeds the random generators: every run gives the same numbers
# One independent random stream per experiment, so editing one section never changes another.
rng_market, rng_forecast = lab.rng.spawn(2)

SOLVER = cp.CLARABEL  # interior-point; ships with cvxpy. Section 7 compares OSQP and SCS.
# Tight tolerances for the analysis sections, so "is this constraint binding?" has a clean answer.
# Section 7 shows what the default tolerances do to the same problem.
TIGHT = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
# The L1 rebalance of section 8 is harder for Clarabel at 1e-12 (it reports optimal_inaccurate);
# OSQP at tight tolerance, with its polishing step, recovers the exact set of traded names.
OSQP_TIGHT = dict(eps_abs=1e-10, eps_rel=1e-10, max_iter=200_000)

# %% [markdown]
# ## 1 · The input: a covariance matrix
#
# Twenty assets driven by three common factors plus their own noise (`quantnb.synth`). We simulate
# five years and one quarter of daily returns and estimate two annualised covariance matrices on
# rolling five-year windows: `S_old` (a quarter ago) and `S` (today). Every section uses today's
# `S`; `S_old` appears only in section 8, to build last quarter's portfolio. Estimation error is
# Session 9.2's subject; here the matrix is simply the input.

# %%
N_ASSETS, YEARS, QUARTER = 20, 5, 63
WINDOW = PERIODS_PER_YEAR * YEARS

market = qn.synth.factor_model_returns(WINDOW + QUARTER, N_ASSETS, n_factors=3, rng=rng_market)
R = market["returns"].to_numpy()
S_old = np.cov(R[:WINDOW], rowvar=False) * PERIODS_PER_YEAR       # annualised, a quarter ago
S = np.cov(R[QUARTER:], rowvar=False) * PERIODS_PER_YEAR          # annualised, today

# Sort the assets by their unconstrained minimum-variance weight (largest first) and number them
# 1..20, so every chart below reads left to right from "the optimiser loves it" to "it shorts it".
ones = np.ones(N_ASSETS)
_x = np.linalg.solve(S, ones)
order = np.argsort(-_x / _x.sum())
S, S_old = S[np.ix_(order, order)], S_old[np.ix_(order, order)]
labels = [str(i + 1) for i in range(N_ASSETS)]

vols = np.sqrt(np.diag(S))
cond = np.linalg.cond(S)
print(f"asset vols {vols.min():.1%}–{vols.max():.1%} · condition number of S {cond:.1f}")
assert np.all(np.linalg.eigvalsh(S) > 0), "S must be positive definite for the closed form"
lab.record("n_assets", N_ASSETS)
lab.record("years", YEARS)
lab.record("vol_lo_asset", float(vols.min()))
lab.record("vol_hi_asset", float(vols.max()))
lab.record("cond_S", float(cond))

# %% [markdown]
# ## 2 · Minimum variance: closed form vs cvxpy
#
# Minimise w′Σw subject to the budget 1′w = 1. Setting the gradient of the Lagrangian to zero gives
# w* = Σ⁻¹1 / (1′Σ⁻¹1), with variance 1 / (1′Σ⁻¹1). cvxpy must find the same point.

# %%
def min_var_closed_form(S: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum-variance weights under the budget only: w = S⁻¹1 / 1′S⁻¹1 (solve, never invert)."""
    x = np.linalg.solve(S, np.ones(len(S)))
    return x / x.sum(), 1.0 / x.sum()


w_cf, var_cf = min_var_closed_form(S)

w = cp.Variable(N_ASSETS)
budget = cp.sum(w) == 1
prob = cp.Problem(cp.Minimize(cp.quad_form(w, S)), [budget])
t0 = time.perf_counter()
prob.solve(solver=SOLVER)
print(f"status {prob.status} in {1e3 * (time.perf_counter() - t0):.1f} ms (timing printed, not recorded)")
w_mv = w.value.copy()

gap_cf = np.abs(w_mv - w_cf).max()
print(f"closed-form vol {np.sqrt(var_cf):.3%} · cvxpy vol {np.sqrt(prob.value):.3%} · max |Δw| {gap_cf:.1e}")
# Clarabel stops at a relative duality gap of 1e-8; on a matrix this well conditioned the weights
# agree far more tightly than 1e-6, so a failure here means the model changed, not the rounding.
assert prob.status == cp.OPTIMAL
assert gap_cf < 1e-6
assert abs(prob.value - var_cf) < 1e-9

lab.record("vol_mv", float(np.sqrt(var_cf)))
lab.record("gap_cf", float(gap_cf))
lab.record("n_short_mv", int((w_mv < -1e-8).sum()))
lab.record("gross_mv", float(np.abs(w_mv).sum()))
lab.record("w_mv_min", float(w_mv.min()))
lab.record("w_mv_max", float(w_mv.max()))

# %% [markdown]
# ## 3 · Constraints that bind: long-only, then a 10% cap
#
# Each rule shrinks the feasible set, so the minimum variance can only rise. The interesting part
# is *which* constraints end up binding — holding with equality at the optimum.

# %%
CAP = 0.10          # maximum weight per asset
TOL_BIND = 1e-6     # "at the bound" means within this of it (solvers never hit a bound exactly)


def min_var(S: np.ndarray, long_only: bool = False, cap: float | None = None, solver=SOLVER, **opts):
    """Minimum-variance QP with optional long-only and per-asset cap. Returns (problem, w, constraints)."""
    n = len(S)
    w = cp.Variable(n)
    cons = {"budget": cp.sum(w) == 1}
    if long_only:
        cons["floor"] = w >= 0
    if cap is not None:
        cons["cap"] = w <= cap
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, S)), list(cons.values()))
    prob.solve(solver=solver, **opts)
    return prob, w, cons


p_lo, w_lo_var, c_lo = min_var(S, long_only=True, **TIGHT)
p_box, w_box_var, c_box = min_var(S, long_only=True, cap=CAP, **TIGHT)
w_lo, w_box = w_lo_var.value.copy(), w_box_var.value.copy()
vol_lo, vol_box = np.sqrt(p_lo.value), np.sqrt(p_box.value)

at_floor_lo = w_lo < TOL_BIND
at_floor_box = w_box < TOL_BIND
at_cap_box = w_box > CAP - TOL_BIND
print(f"no limits  vol {np.sqrt(var_cf):.2%} · shorts {(w_mv < 0).sum()}")
print(f"long-only  vol {vol_lo:.2%} · held {(~at_floor_lo).sum()} · at zero {at_floor_lo.sum()}")
print(f"+ cap {CAP:.0%}  vol {vol_box:.2%} · at zero {at_floor_box.sum()} · at cap {at_cap_box.sum()}")

# adding constraints can never lower the minimum
assert np.sqrt(var_cf) <= vol_lo + 1e-9 <= vol_box + 2e-9
assert at_cap_box.sum() > 0 and at_floor_box.sum() > 0, "both kinds of constraint should bind here"

lab.record("cap", CAP)
lab.record("vol_lo", float(vol_lo))
lab.record("vol_box", float(vol_box))
lab.record("n_held_lo", int((~at_floor_lo).sum()))
lab.record("n_zero_lo", int(at_floor_lo.sum()))
lab.record("n_zero_box", int(at_floor_box.sum()))
lab.record("n_cap_box", int(at_cap_box.sum()))
lab.record("n_held_box", int((~at_floor_box).sum()))
lab.record("n_interior_box", int((~at_floor_box & ~at_cap_box).sum()))
lab.record("vol_cost_lo_bp", float(1e4 * (vol_lo - np.sqrt(var_cf))))
lab.record("vol_cost_cap_bp", float(1e4 * (vol_box - vol_lo)))

lab.chart("weights", charts.grouped_column_chart(
    labels,
    [("no limits", list(w_mv), "strategy"),
     ("long-only", list(w_lo), "alt1"),
     (f"long-only, cap {CAP:.0%}", list(w_box), "alt2")],
    title="Minimum-variance weights under three rule sets (assets 1–20)",
    y_fmt=charts.fmt_pct(0), height=280))

# %% [markdown]
# ## 4 · Multipliers are prices
#
# cvxpy stores each constraint's Lagrange multiplier in `constraint.dual_value`. Its convention:
# raising a constraint's right-hand side by δ changes the optimal value by about −(dual value)·δ.
# We check that claim twice with finite differences — the budget, and a gross-leverage cap.
#
# **(a) The budget.** Minimum variance with 1′w = b has optimal value b²/(1′Σ⁻¹1), so its slope at
# b = 1 is 2σ²_min. The multiplier should say the same.

# %%
H = 1e-3  # finite-difference step; the value function is exactly quadratic in b, so only solver noise matters


def tight(prob: cp.Problem) -> None:
    """Re-solve with Clarabel at tighter tolerances, so finite differences see the value, not the noise."""
    prob.solve(solver=cp.CLARABEL, **TIGHT)


def pstar_budget(b: float) -> float:
    w = cp.Variable(N_ASSETS)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, S)), [cp.sum(w) == b])
    tight(prob)
    return prob.value


nu_budget = -budget.dual_value            # sign flipped: cvxpy's dual is −(slope) for `expr == b`
fd_budget = (pstar_budget(1 + H) - pstar_budget(1 - H)) / (2 * H)
print(f"budget: multiplier {nu_budget:.6f} · finite difference {fd_budget:.6f} · 2σ² {2 * var_cf:.6f}")
assert abs(nu_budget - 2 * var_cf) < 1e-8
assert abs(fd_budget - nu_budget) < 1e-7
lab.record("var_mv", float(var_cf))
lab.record("nu_budget", float(nu_budget))
lab.record("fd_budget", float(fd_budget))

# %% [markdown]
# **(b) A gross-leverage cap.** Allow shorts but cap gross exposure: ‖w‖₁ = Σ|wᵢ| ≤ L. The
# unconstrained minimum-variance portfolio has gross leverage above 1, so a cap between 1 and that
# level binds. The multiplier λ_L says how much variance one more unit of L would buy.

# %%
L_CAP = round(1 + 0.5 * (np.abs(w_mv).sum() - 1), 2)   # halfway between long-only and "no limit"
L_STEPS = (0.02, 0.10)                                 # "what if we loosened it by…"


def min_var_gross(L: float, precise: bool = False):
    w = cp.Variable(N_ASSETS)
    gross = cp.norm1(w) <= L
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, S)), [cp.sum(w) == 1, gross])
    tight(prob) if precise else prob.solve(solver=SOLVER)
    return prob, gross, w


p_L, gross_con, w_L = min_var_gross(L_CAP, precise=True)
lam_L = float(gross_con.dual_value)
fd_L = -(min_var_gross(L_CAP + H, True)[0].value - min_var_gross(L_CAP - H, True)[0].value) / (2 * H)
vol_L = np.sqrt(p_L.value)
print(f"gross cap {L_CAP}: vol {vol_L:.3%} · λ_L {lam_L:.6f} · finite difference {fd_L:.6f}")
assert lam_L > 1e-6, "the gross cap should bind"
# Central difference on a smooth value function: error is O(H²) plus solver noise ~1e-12/H.
assert abs(fd_L - lam_L) < 1e-3 * lam_L + 1e-8
for step in L_STEPS:
    vol_next = np.sqrt(min_var_gross(L_CAP + step, precise=True)[0].value)
    pred_bp = 1e4 * lam_L * step / (2 * vol_L)          # dσ = dσ²/(2σ): first order
    actual_bp = 1e4 * (vol_L - vol_next)
    print(f"loosen by {step}: vol falls {actual_bp:.2f} bp (multiplier predicts {pred_bp:.2f} bp)")
    # Theorem (Boyd & Vandenberghe §5.6.2, the global sensitivity inequality): loosening a constraint
    # by δ can never cut the optimal value by more than λ·δ. The tangent is an upper bound on the gain.
    assert p_L.value - vol_next ** 2 <= lam_L * step + 1e-12
    # In volatility units the square root bends the other way a little; here the tangent still wins.
    assert actual_bp <= pred_bp + 1e-9
    tag = f"{round(step * 100):02d}"
    lab.record(f"step_{tag}", step)
    lab.record(f"pred_drop_{tag}_bp", float(pred_bp))
    lab.record(f"actual_drop_{tag}_bp", float(actual_bp))
    lab.record(f"var_drop_{tag}", float(p_L.value - vol_next ** 2))
# The tangent is a local statement: its relative error must shrink as the step shrinks.
rel_err = {t: 1 - lab.results[f"actual_drop_{t}_bp"] / lab.results[f"pred_drop_{t}_bp"] for t in ("02", "10")}
print(f"tangent over-prediction: {rel_err['02']:.0%} for the small step, {rel_err['10']:.0%} for the large one")
assert 0 <= rel_err["02"] < rel_err["10"]
lab.record("tangent_err_02", float(rel_err["02"]))
lab.record("tangent_err_10", float(rel_err["10"]))

# At L = 1 the cap forces Σ|w| = Σw = 1, i.e. no shorts: the long-only portfolio of section 3.
vol_L1 = np.sqrt(min_var_gross(1.0)[0].value)
assert abs(vol_L1 - vol_lo) < 1e-6
# Past the unconstrained gross leverage the cap no longer binds and its price is zero.
p_loose, gross_loose, _ = min_var_gross(np.abs(w_mv).sum() + 0.2)
assert abs(gross_loose.dual_value) < 1e-6 and abs(np.sqrt(p_loose.value) - np.sqrt(var_cf)) < 1e-6

lab.record("l_cap", L_CAP)
lab.record("lam_L", lam_L)
lab.record("fd_L", float(fd_L))
lab.record("vol_L", float(vol_L))
lab.record("n_short_L", int((w_L.value < -TOL_BIND).sum()))

# The whole value curve, and the multiplier's tangent line at L_CAP (drawn in volatility units).
L_grid = np.round(np.arange(1.0, np.abs(w_mv).sum() + 0.15, 0.01), 2)
vol_grid = np.array([np.sqrt(min_var_gross(L)[0].value) for L in L_grid])
near = L_grid[(L_grid >= L_CAP - 0.12) & (L_grid <= L_CAP + 0.12)]
tangent = vol_L - lam_L / (2 * vol_L) * (near - L_CAP)
lab.chart("gross_cap", charts.line_chart(
    [charts.Series("lowest achievable volatility", pd.Series(vol_grid, index=L_grid), role="strategy",
                   end_label="lowest vol"),
     charts.Series("tangent from the multiplier", pd.Series(tangent, index=near), role="alt1",
                   label_end=False)],
    title="Lowest volatility vs the gross-leverage cap L", y_fmt=charts.fmt_pct(1), height=280))

# %% [markdown]
# ## 5 · Verify the KKT conditions yourself
#
# For the long-only, capped problem write the Lagrangian as
# ℒ = w′Σw − ν(1′w − 1) − λ′w + η′(w − u), with λ, η ≥ 0 the multipliers of the floor and the cap.
# The KKT conditions are four checks you can run on any solution — no trust in the solver needed.

# %%
nu = -c_box["budget"].dual_value        # budget price (sign flipped to our convention)
lam = c_box["floor"].dual_value         # ≥ 0, non-zero only where w = 0
eta = c_box["cap"].dual_value           # ≥ 0, non-zero only where w = cap
marg_var = 2 * S @ w_box                # ∂(w′Σw)/∂wᵢ: each asset's marginal variance

kkt = {
    "primal": max(abs(w_box.sum() - 1), (-w_box).max(), (w_box - CAP).max(), 0.0),
    "dual": max(-lam.min(), -eta.min(), 0.0),
    "slack": max(np.abs(lam * w_box).max(), np.abs(eta * (CAP - w_box)).max()),
    "stationarity": np.abs(marg_var - nu - lam + eta).max(),
}
for k, v in kkt.items():
    print(f"KKT {k:<13} residual {v:.1e}")
    lab.record(f"kkt_{k}", float(v))
# Solved at 1e-12 tolerances, residuals of 1e-9 or less are "zero" (section 7 shows why not 0).
assert all(v < 1e-9 for v in kkt.values())

# In words: held assets share one marginal variance, ν. Excluded assets sit above it by λᵢ,
# capped assets below it by ηᵢ. The gap is exactly the multiplier.
gap = (marg_var - nu) / nu
interior = ~at_floor_box & ~at_cap_box
assert np.abs(gap[interior]).max() < 1e-6
assert np.all(gap[at_floor_box] > -1e-6) and np.all(gap[at_cap_box] < 1e-6)
# Euler's identity ties the budget price to the risk: Σ wᵢ·marg_varᵢ = 2σ², so ν = 2σ² + u·Σηᵢ.
assert abs(nu - (2 * p_box.value + CAP * eta.sum())) < 1e-7
# Without caps the budget price is exactly 2σ² again (section 4a, but now with 0 ≤ w).
assert abs(-c_lo["budget"].dual_value - 2 * p_lo.value) < 1e-7

roles = ["alt1" if f else ("alt2" if c else "strategy") for f, c in zip(at_floor_box, at_cap_box)]
lab.chart("kkt_gap", charts.column_chart(
    labels, list(gap), roles=roles, value_labels=False,
    title="Each asset's marginal variance vs the budget price (long-only, capped)",
    y_fmt=charts.fmt_pct(0), height=260))
lab.record("nu_box", float(nu))
lab.record("gap_max_floor", float(gap[at_floor_box].max()))
lab.record("gap_min_cap", float(gap[at_cap_box].min()))
lab.record("eta_max", float(eta.max()))
lab.record("lam_max", float(lam.max()))

# %% [markdown]
# ## 6 · DCP: what cvxpy accepts, what it rejects, and how to rewrite
#
# cvxpy only accepts problems it can *prove* convex from its composition rules (Disciplined Convex
# Programming). The rules are sufficient, not necessary: a convex function written the wrong way is
# rejected. A genuinely non-convex request is rejected however you write it.

# %%
Lc = np.linalg.cholesky(S)               # S = Lc Lc′, so w′Sw = ‖Lc′w‖²
mu_fc = rng_forecast.normal(0.05, 0.03, N_ASSETS)   # someone's return forecasts, used in section 7 too
w = cp.Variable(N_ASSETS)
w_prev_demo = np.full(N_ASSETS, 1 / N_ASSETS)

gallery = [  # (label, problem, DCP expected?)
    ("minimise variance: quad_form(w, S)",
     cp.Problem(cp.Minimize(cp.quad_form(w, S)), [cp.sum(w) == 1]), True),
    ("minimise volatility: sqrt(quad_form(w, S))",
     cp.Problem(cp.Minimize(cp.sqrt(cp.quad_form(w, S))), [cp.sum(w) == 1]), False),
    ("minimise volatility: norm(Lc.T @ w, 2)",
     cp.Problem(cp.Minimize(cp.norm(Lc.T @ w, 2)), [cp.sum(w) == 1]), True),
    ("volatility cap: quad_form(w, S) <= 0.12**2",
     cp.Problem(cp.Maximize(mu_fc @ w), [cp.sum(w) == 1, cp.quad_form(w, S) <= 0.12 ** 2]), True),
    ("volatility floor: quad_form(w, S) >= 0.15**2",
     cp.Problem(cp.Minimize(cp.sum_squares(w)), [cp.sum(w) == 1, cp.quad_form(w, S) >= 0.15 ** 2]), False),
    ("gross cap: norm1(w) <= 1.5",
     cp.Problem(cp.Minimize(cp.quad_form(w, S)), [cp.sum(w) == 1, cp.norm1(w) <= 1.5]), True),
    ("gross exactly: norm1(w) == 1.5",
     cp.Problem(cp.Minimize(cp.quad_form(w, S)), [cp.sum(w) == 1, cp.norm1(w) == 1.5]), False),
    ("turnover penalty: + norm1(w - w_prev)",
     cp.Problem(cp.Minimize(cp.quad_form(w, S) + 0.01 * cp.norm1(w - w_prev_demo)), [cp.sum(w) == 1]), True),
    ("Sharpe ratio: (mu @ w) / norm(Lc.T @ w)",
     cp.Problem(cp.Maximize((mu_fc @ w) / cp.norm(Lc.T @ w, 2)), [cp.sum(w) == 1]), False),
]
for label, prob, expected in gallery:
    ok = prob.is_dcp()
    print(f"{'accepted' if ok else 'REJECTED':>8} · {label}")
    assert ok == expected, label
lab.record("dcp_n_tested", len(gallery))

# Solving a rejected problem raises DCPError — catch it and read the message.
try:
    gallery[1][1].solve(solver=SOLVER)
    raise AssertionError("cvxpy should have refused the sqrt formulation")
except cp.error.DCPError as err:
    print("DCPError:", str(err).splitlines()[0])

# The rewrite: volatility IS a norm, ‖Lc′w‖₂, and norms are convex atoms cvxpy knows.
prob_norm = gallery[2][1]
prob_norm.solve(solver=SOLVER)
vol_norm = prob_norm.value
w_norm = prob_norm.variables()[0].value
print(f"norm formulation: vol {vol_norm:.4%} vs closed form {np.sqrt(var_cf):.4%}")
assert abs(vol_norm - np.sqrt(var_cf)) < 1e-7 and np.abs(w_norm - w_cf).max() < 1e-6
lab.record("vol_norm", float(vol_norm))

# %% [markdown]
# **Cardinality is not convex.** "At most 10 names" fails the definition of a convex set: average
# two 10-name portfolios with different names and you hold 20.

# %%
K = 10
a = np.r_[np.full(K, 1 / K), np.zeros(N_ASSETS - K)]      # names 1–10
b = np.r_[np.zeros(N_ASSETS - K), np.full(K, 1 / K)]      # names 11–20
mid = 0.5 * a + 0.5 * b
names = lambda v: int((np.abs(v) > 1e-12).sum())          # noqa: E731
print(f"names held: a {names(a)}, b {names(b)}, their average {names(mid)}")
assert names(a) <= K and names(b) <= K and names(mid) > K
lab.record("card_k", K)
lab.record("card_mid", names(mid))

# %% [markdown]
# ## 7 · When the solver says no — status, infeasibility, unboundedness, tolerance
#
# Always read `prob.status` before `w.value`. Two deliberate failures, then the fix for each.

# %%
# Infeasible: cap every asset so tightly that all of them together hold only 80% of the budget
# (20 assets × 4%).
TIGHT_CAP = 0.8 / N_ASSETS
p_inf, w_inf, _ = min_var(S, long_only=True, cap=TIGHT_CAP)
print(f"cap {TIGHT_CAP:.0%}: status {p_inf.status}, value {p_inf.value}, w.value {w_inf.value}")
assert p_inf.status == cp.INFEASIBLE and w_inf.value is None and np.isinf(p_inf.value)
lab.record("tight_cap", TIGHT_CAP)
lab.record("tight_cap_max_invested", TIGHT_CAP * N_ASSETS)
lab.record("status_infeasible", p_inf.status)

# Unbounded: maximise forecast return with only a budget. Short the worst forecast, buy the best,
# repeat forever — the objective has no ceiling.
w = cp.Variable(N_ASSETS)
p_unb = cp.Problem(cp.Maximize(mu_fc @ w), [cp.sum(w) == 1])
p_unb.solve(solver=SOLVER)
print(f"no risk term, no leverage limit: status {p_unb.status}, value {p_unb.value}")
assert p_unb.status == cp.UNBOUNDED and np.isinf(p_unb.value)
lab.record("status_unbounded", p_unb.status)

# The fix is a modelling fix: add the missing constraint (here a gross cap) and it becomes a small LP.
p_fix = cp.Problem(cp.Maximize(mu_fc @ w), [cp.sum(w) == 1, cp.norm1(w) <= 1.5])
p_fix.solve(solver=SOLVER)
assert p_fix.status == cp.OPTIMAL
print(f"with a 1.5 gross cap: status {p_fix.status}, long {w.value[w.value > 1e-6].sum():.2f}, "
      f"short {w.value[w.value < -1e-6].sum():.2f}")
lab.record("status_fixed", p_fix.status)

# %% [markdown]
# **Tolerance.** Solvers stop when their residuals fall below a tolerance, so "zero" comes back as
# 1e-9, −5e-9 or 2e-4. Compare three solvers, at cvxpy's default settings, on the long-only capped
# problem against a reference solved at much tighter tolerance — once in annual units, once in
# daily units (the same problem divided by 252). The weight error and the volatility error tell
# very different stories.

# %%
def solve_box(S_: np.ndarray, solver, **opts):
    w = cp.Variable(len(S_))
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, S_)), [cp.sum(w) == 1, w >= 0, w <= CAP])
    t0 = time.perf_counter()
    prob.solve(solver=solver, **opts)
    ms = 1e3 * (time.perf_counter() - t0)
    assert prob.status == cp.OPTIMAL, (solver, prob.status)
    return w.value.copy(), ms


def ann_vol(v: np.ndarray) -> float:
    return float(np.sqrt(v @ S @ v))      # always measured in annual units, whatever was solved


w_ref, _ = solve_box(S, cp.CLARABEL, tol_gap_abs=1e-13, tol_gap_rel=1e-13, tol_feas=1e-13)
assert np.abs(w_ref - w_box).max() < 1e-7          # agrees with the tight solve of section 3
rows = []
for units, S_ in (("annual", S), ("daily", S / PERIODS_PER_YEAR)):
    for name, solver in (("CLARABEL", cp.CLARABEL), ("OSQP", cp.OSQP), ("SCS", cp.SCS)):
        w_s, ms = solve_box(S_, solver)
        rows.append({"units": units, "solver": name, "max_dw": np.abs(w_s - w_ref).max(),
                     "dvol_bp": 1e4 * (ann_vol(w_s) - ann_vol(w_ref)), "min_w": w_s.min(),
                     "budget_err": abs(w_s.sum() - 1), "ms (printed only)": ms})
tol_table = pd.DataFrame(rows)
print(tol_table.to_string(float_format=lambda v: f"{v:.1e}"))
for r in rows:
    key = f"tol_{r['units']}_{r['solver'].lower()}"
    lab.record(f"{key}_dw", float(r["max_dw"]))
    lab.record(f"{key}_dvol_bp", float(r["dvol_bp"]))
    lab.record(f"{key}_minw", float(r["min_w"]))
# Every default solve is "optimal", and every one is within 0.01 of the reference weights. That bound
# is loose on purpose: one asset (found and recorded just below) has a multiplier near zero, so the objective
# is almost flat along it (moving it costs only about λ per unit), so a solver can stop well short of it.
assert tol_table["max_dw"].max() < 1e-2
# ...while the volatility is right to a small fraction of a basis point: flat means cheap to miss.
assert tol_table["dvol_bp"].abs().max() < 0.1
# The narrative: the same problem in daily units (every number 252 times smaller) comes back less
# accurate from Clarabel at its defaults. Solvers' stopping rules and internal scaling are tuned for
# numbers near 1, so scale your problem (annual variances, weights not dollars) before blaming them.
clar = tol_table.set_index(["units", "solver"])["max_dw"]
assert clar[("daily", "CLARABEL")] > 3 * clar[("annual", "CLARABEL")]
lab.record("tol_clarabel_units_ratio", float(clar[("daily", "CLARABEL")] / clar[("annual", "CLARABEL")]))
worst = int(np.argmax(np.abs(solve_box(S, cp.CLARABEL)[0] - w_ref)))
lab.record("tol_worst_asset", worst + 1)
lab.record("tol_worst_lam", float(lam[worst]))
lab.record("var_annual", float(p_box.value))
lab.record("var_daily", float(p_box.value / PERIODS_PER_YEAR))

# %% [markdown]
# ## 8 · A turnover-penalised rebalance (a preview of Session 9.6)
#
# A quarter ago you built the long-only, capped portfolio from `S_old`. Today's `S` is slightly
# different, so re-optimising from scratch trades several names — for a sliver of volatility.
# Adding θ‖w − w_prev‖₁ to the objective prices each unit of trading; the kink of the absolute value
# at zero makes many trades exactly zero. (We call the penalty weight θ because κ is already the
# borrow fee of Session 2.2.)

# %%
THETAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]   # variance units per unit of turnover
TOL_TRADE = 1e-6

w_prev = min_var(S_old, long_only=True, cap=CAP, **TIGHT)[1].value.copy()


def rebalance(theta: float):
    w = cp.Variable(N_ASSETS)
    obj = cp.quad_form(w, S) + theta * cp.norm1(w - w_prev)
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1, w >= 0, w <= CAP])
    prob.solve(solver=cp.OSQP, **OSQP_TIGHT)  # tight, so "traded" means traded, not solver noise
    assert prob.status == cp.OPTIMAL
    trade = w.value - w_prev
    return {"theta": theta, "turnover": float(np.abs(trade).sum()),
            "n_trades": int((np.abs(trade) > TOL_TRADE).sum()),
            "vol": float(np.sqrt(w.value @ S @ w.value))}


reb = pd.DataFrame([rebalance(theta) for theta in THETAS])
print(reb.to_string(index=False))
vol_prev = float(np.sqrt(w_prev @ S @ w_prev))
# A heavier penalty can never buy more turnover (standard exchange argument for penalty weights).
assert np.all(np.diff(reb["turnover"]) <= 1e-7)
assert reb["n_trades"].iloc[0] > reb["n_trades"].iloc[3] > reb["n_trades"].iloc[-1] == 0
assert reb["turnover"].iloc[-1] < 1e-6, "at the largest θ the old portfolio should be kept"
# θ = 0 is just today's minimum-variance problem: it must reproduce section 3.
assert abs(reb["vol"].iloc[0] - vol_box) < 1e-6
assert reb["vol"].iloc[0] <= vol_prev + 1e-9

lab.record("vol_prev", vol_prev)
for i, r in reb.iterrows():
    lab.record(f"reb_{i}_theta", float(r["theta"]))
    lab.record(f"reb_{i}_turnover", float(r["turnover"]))
    lab.record(f"reb_{i}_trades", int(r["n_trades"]))
    lab.record(f"reb_{i}_vol", float(r["vol"]))
lab.record("reb_full_turnover", float(reb["turnover"].iloc[0]))
lab.record("reb_vol_gain_bp", 1e4 * (vol_prev - float(reb["vol"].iloc[0])))

lab.chart("turnover", charts.column_chart(
    [f"{k:g}" for k in THETAS], list(reb["turnover"]),
    title="Turnover of the rebalance as the penalty θ grows", y_fmt=charts.fmt_pct(1), height=240))

# %%
lab.save()
