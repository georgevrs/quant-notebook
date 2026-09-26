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
# # 3.3 · Linear Algebra for Portfolios — companion lab
#
# **Quant Notebook** · Unit 3 · Session 3 · [Read the session](https://georgevrs.github.io/quant-notebook/unit03-math-toolkit/session03-linear-algebra-portfolios.html)
#
# Weights as vectors, risk as a quadratic form, PCA as the hidden structure of a market — linear algebra through the portfolio lens.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# All data are synthetic daily returns from a factor model whose true covariance matrix we know,
# so every estimate can be checked against the truth. There is no risk-free rate in this lab:
# nothing here depends on the level of returns, only on how they move together.

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

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR

lab = qn.Lab("3.3")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
rng_market, rng_pca_mc, rng_tn, rng_missing = lab.rng.spawn(4)

# %% [markdown]
# ## 1 · A portfolio is a vector; its risk is a quadratic form
#
# A classic 60/40 portfolio: 60% equities, 40% bonds. The volatilities and the correlation below
# are **illustrative annual assumptions**, not estimates. The weight vector w meets the covariance
# matrix Σ twice: the portfolio variance is w'Σw. The covariance matrix is the correlation matrix
# scaled by the volatilities on both sides: Σ = D C D, with D = diag(σ).

# %%
W_6040 = np.array([0.60, 0.40])        # equities, bonds
VOLS_6040 = np.array([0.16, 0.06])     # annual volatilities (illustrative)
RHO_SB = 0.20                          # stock-bond correlation (illustrative; it changes sign over decades)

C_6040 = np.array([[1.0, RHO_SB], [RHO_SB, 1.0]])
D_6040 = np.diag(VOLS_6040)
SIGMA_6040 = D_6040 @ C_6040 @ D_6040                    # Σ = D C D
port_var = W_6040 @ SIGMA_6040 @ W_6040                  # the quadratic form w'Σw
# Exact identity: the matrix form is the two-asset formula from Session 3.1, written compactly.
w1, w2 = W_6040
s1, s2 = VOLS_6040
assert np.isclose(port_var, w1**2 * s1**2 + w2**2 * s2**2 + 2 * w1 * w2 * RHO_SB * s1 * s2)
assert np.allclose(SIGMA_6040, np.outer(VOLS_6040, VOLS_6040) * C_6040)

# Risk contributions (a preview of Session 9.3): RC_i = w_i (Σw)_i. They add up to w'Σw exactly
# (Euler's theorem for a function that scales like the weights squared).
rc = W_6040 * (SIGMA_6040 @ W_6040)
assert np.isclose(rc.sum(), port_var)
rc_share = rc / port_var
avg_vol = W_6040 @ VOLS_6040                             # weighted-average volatility
print(f"60/40: vol {np.sqrt(port_var):.2%} · average asset vol {avg_vol:.2%} · "
      f"risk shares equities {rc_share[0]:.1%}, bonds {rc_share[1]:.1%}")

lab.record("w_eq", W_6040[0])
lab.record("w_bd", W_6040[1])
lab.record("vol_eq", VOLS_6040[0])
lab.record("vol_bd", VOLS_6040[1])
lab.record("rho_sb", RHO_SB)
lab.record("var_6040", float(port_var))
lab.record("vol_6040", float(np.sqrt(port_var)))
lab.record("avg_vol_6040", float(avg_vol))
lab.record("dr_6040", float(avg_vol / np.sqrt(port_var)))   # diversification ratio
lab.record("rc_share_eq", float(rc_share[0]))
lab.record("rc_share_bd", float(rc_share[1]))

# %% [markdown]
# ## 2 · Diversification has a floor
#
# N assets, each with volatility σ, every pair with correlation ρ, held in equal weights 1/N.
# The covariance matrix is Σ = σ²[(1 − ρ)I + ρ11'], and the quadratic form collapses to
#
#     w'Σw = σ² (ρ + (1 − ρ)/N)  →  ρσ²  as N → ∞.
#
# The part (1 − ρ)σ²/N diversifies away; the part ρσ² never does. We check the closed form
# against the matrix computation for every N — an exact identity, not a simulation.

# %%
RHOS = (0.0, 0.3, 0.6)
N_MAX = 100
N_SHOW = (1, 10, 30, 100)
SIGMA_ONE = 0.30                     # one asset's annual volatility (illustrative: a single stock)


def equal_corr_cov(n: int, sigma: float, rho: float) -> np.ndarray:
    """Covariance matrix of n assets with equal volatility sigma and equal pairwise correlation rho."""
    return sigma**2 * ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))


def vol_fraction(n: int, rho: float) -> float:
    """Equal-weight portfolio volatility as a fraction of one asset's volatility (closed form)."""
    return float(np.sqrt(rho + (1 - rho) / n))


curves = {}
for rho in RHOS:
    fr = []
    for n in range(1, N_MAX + 1):
        w = np.full(n, 1 / n)
        via_matrix = np.sqrt(w @ equal_corr_cov(n, SIGMA_ONE, rho) @ w) / SIGMA_ONE
        assert np.isclose(via_matrix, vol_fraction(n, rho))     # exact identity
        fr.append(via_matrix)
    curves[rho] = pd.Series(fr, index=np.arange(1, N_MAX + 1))
    key = f"{rho:.1f}".replace(".", "")
    for n in N_SHOW:
        f_n = vol_fraction(n, rho)
        lab.record(f"div_frac_{key}_n{n}", f_n)
        lab.record(f"div_dr_{key}_n{n}", 1 / f_n)                # diversification ratio
        lab.record(f"div_bets_{key}_n{n}", 1 / f_n**2)           # effective number of independent bets
    if rho > 0:
        lab.record(f"div_frac_{key}_inf", np.sqrt(rho))
        lab.record(f"div_dr_{key}_inf", 1 / np.sqrt(rho))
        lab.record(f"div_bets_{key}_inf", 1 / rho)
        lab.record(f"floor_vol_{key}", np.sqrt(rho) * SIGMA_ONE)
    print(f"ρ = {rho:.1f}: vol fraction at N = 10 / 100: {vol_fraction(10, rho):.3f} / {vol_fraction(100, rho):.3f}")
lab.record("sigma_one", SIGMA_ONE)
lab.record("rho_mid", RHOS[1])
lab.record("rho_hi", RHOS[2])
# Link to Session 3.1: ten positions at ρ = 0.3 are sqrt(1 + 9ρ) times as risky as ten independent ones.
lab.record("ratio_10_vs_indep", float(np.sqrt(1 + 9 * 0.3)))

# %% [markdown]
# ## 3 · A synthetic market with a known covariance matrix
#
# Fifty assets driven by three factors — one "market" factor that every asset loads on
# (loadings around 1) and two "style" factors with loadings around 0 — plus independent noise.
# Ten years of daily returns. Because the loadings B are known, the TRUE covariance matrix is too:
#
#     Σ = B diag(σ_f²) B' + σ_e² I.
#
# The sample covariance matrix S is an estimate of it.

# %%
N_ASSETS, N_FACTORS, YEARS = 50, 3, 10
T_DAYS = PERIODS_PER_YEAR * YEARS
FACTOR_VOL, IDIO_VOL = 0.15, 0.25          # annual volatilities of each factor and of the noise

mkt = qn.synth.factor_model_returns(T_DAYS, N_ASSETS, N_FACTORS, factor_vol=FACTOR_VOL,
                                    idio_vol=IDIO_VOL, rng=rng_market)
R = mkt["returns"]                          # T × N daily simple returns
B = mkt["loadings"].to_numpy()              # N × K true loadings
F_D = FACTOR_VOL / np.sqrt(PERIODS_PER_YEAR)   # daily factor volatility
E_D = IDIO_VOL / np.sqrt(PERIODS_PER_YEAR)     # daily idiosyncratic volatility


def true_cov(loadings: np.ndarray) -> np.ndarray:
    """Daily covariance matrix implied by the factor model: B diag(σ_f²) B' + σ_e² I."""
    return (loadings * F_D**2) @ loadings.T + E_D**2 * np.eye(loadings.shape[0])


SIGMA = true_cov(B)
S = R.cov().to_numpy()                      # sample covariance (divides by T − 1)
C = R.corr().to_numpy()                     # sample correlation matrix
d = np.sqrt(np.diag(S))
assert np.allclose(S, np.diag(d) @ C @ np.diag(d))      # Σ = D C D, exactly, for the estimates too

# Portfolio return is a dot product, R_p = R w; its sample variance IS w'Sw (same T − 1 divisor).
w_ew = np.full(N_ASSETS, 1 / N_ASSETS)
port = R.to_numpy() @ w_ew
assert np.isclose(port.var(ddof=1), w_ew @ S @ w_ew)

rel_frob = np.linalg.norm(S - SIGMA) / np.linalg.norm(SIGMA)
# For normal returns the expected squared error is exact: E||S − Σ||² = (tr Σ² + (tr Σ)²)/(T − 1).
frob2_theory = (np.trace(SIGMA @ SIGMA) + np.trace(SIGMA) ** 2) / (T_DAYS - 1)
print(f"relative error of S: {rel_frob:.2%} (typical, from theory: {np.sqrt(frob2_theory) / np.linalg.norm(SIGMA):.2%})")
lab.record("n_assets", N_ASSETS)
lab.record("n_factors", N_FACTORS)
lab.record("years", YEARS)
lab.record("t_days", T_DAYS)
lab.record("factor_vol", FACTOR_VOL)
lab.record("idio_vol", IDIO_VOL)
lab.record("n_cov_entries", N_ASSETS * (N_ASSETS + 1) // 2)
lab.record("rel_frob_err", float(rel_frob))
lab.record("rel_frob_err_theory", float(np.sqrt(frob2_theory) / np.linalg.norm(SIGMA)))
lab.record("ew_vol_true", float(np.sqrt(w_ew @ SIGMA @ w_ew * PERIODS_PER_YEAR)))
lab.record("ew_vol_sample", float(np.sqrt(w_ew @ S @ w_ew * PERIODS_PER_YEAR)))

# %% [markdown]
# ## 4 · Eigen-decomposition: the portfolios of risk
#
# A symmetric matrix factors as Σ = V Λ V' with orthonormal eigenvectors (columns of V) and real
# eigenvalues (the diagonal of Λ). Read each eigenvector as a portfolio: the eigen-portfolios are
# uncorrelated with each other, and each eigenvalue is its portfolio's variance. We sort them from
# largest to smallest and fix each eigenvector's arbitrary sign so its weights sum to a positive number.


def eig_desc(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (largest first) and eigenvectors of a symmetric matrix, signs fixed so sum(v) ≥ 0."""
    lam, vec = np.linalg.eigh(m)            # eigh: for symmetric matrices; ascending order
    lam, vec = lam[::-1], vec[:, ::-1]
    vec = vec * np.where(vec.sum(axis=0) < 0, -1.0, 1.0)
    return lam, vec


# %%
lam_true, V_true = eig_desc(SIGMA)
lam_s, V_s = eig_desc(S)

# Exact identities (to floating-point precision):
assert np.allclose(V_s @ np.diag(lam_s) @ V_s.T, S)                 # S = V Λ V'
assert np.allclose(V_s.T @ V_s, np.eye(N_ASSETS))                   # orthonormal eigenvectors
assert np.isclose(lam_s.sum(), np.trace(S))                         # total variance = sum of eigenvalues
pcs = (R - R.mean()).to_numpy() @ V_s                               # returns of the eigen-portfolios
pc_cov = np.cov(pcs, rowvar=False)
assert np.allclose(pc_cov, np.diag(lam_s), atol=1e-12)             # uncorrelated, variance = eigenvalue
# Any portfolio's variance splits across the eigen-portfolios: w'Sw = Σ_k λ_k (v_k'w)².
w_ls = np.where(np.arange(N_ASSETS) < N_ASSETS // 2, 1.0, -1.0) / N_ASSETS   # a long-short book
for w in (w_ew, w_ls):
    assert np.isclose(w @ S @ w, np.sum(lam_s * (V_s.T @ w) ** 2))
ew_pc_share = lam_s * (V_s.T @ w_ew) ** 2 / (w_ew @ S @ w_ew)
assert lam_s.min() > 0                                              # full-data sample Σ is positive definite

share_true = lam_true / lam_true.sum()
share_s = lam_s / lam_s.sum()
cond_true = lam_true[0] / lam_true[-1]
cond_s = lam_s[0] / lam_s[-1]
print("variance share, first 5 components — sample:", share_s[:5].round(3), " true:", share_true[:5].round(3))
for k in range(4):
    lab.record(f"pc{k + 1}_share_true", float(share_true[k]))
    lab.record(f"pc{k + 1}_share_sample", float(share_s[k]))
lab.record("top3_share_true", float(share_true[:3].sum()))
lab.record("top3_share_sample", float(share_s[:3].sum()))
lab.record("idio_eig_share_true", float(share_true[-1]))        # each of the N − K noise directions
lab.record("n_noise_dirs", N_ASSETS - N_FACTORS)
lab.record("ew_pc1_share", float(ew_pc_share[0]))
lab.record("pc1_vol_sample", float(np.sqrt(lam_s[0] * PERIODS_PER_YEAR)))
lab.record("cond_true", float(cond_true))
lab.record("cond_sample", float(cond_s))

# %% [markdown]
# ## 5 · PCA recovers the planted structure
#
# PCA is the eigen-decomposition of the covariance (or correlation) matrix of returns. Does the
# first sample eigenvector find the market factor we planted? And do the top three span the same
# space as the three true loading vectors? Then: how close is "close"? Sampling theory for PCA
# (Anderson, 1963) predicts the expected squared angle between a sample eigenvector and the true
# one, E[sin²θ] ≈ (1/n) Σ_{j≠1} λ₁λ_j/(λ₁ − λ_j)² with n = T − 1; the same argument gives the
# expected share of each loading vector that falls OUTSIDE the estimated 3-dimensional subspace.
# We check both against 300 simulated decades from the same true market, at 4 standard errors.

# %%
pc1 = V_s[:, 0]
corr_pc1_mkt = float(np.corrcoef(pc1, B[:, 0])[0, 1])
corr_true_pc1_mkt = float(np.corrcoef(V_true[:, 0], B[:, 0])[0, 1])
cos_pc1 = float(abs(pc1 @ V_true[:, 0]))
P3 = V_s[:, :N_FACTORS]


def outside_share(basis: np.ndarray, b: np.ndarray) -> float:
    """Share of vector b's squared length lying outside the span of the orthonormal columns of basis."""
    return float(1 - np.linalg.norm(basis.T @ b) ** 2 / np.linalg.norm(b) ** 2)


subspace_r2 = [1 - outside_share(P3, B[:, k]) for k in range(N_FACTORS)]
# PCA finds the subspace, not the labels: PC2 and PC3 are mixtures of the two style factors.
corr_pc2 = [float(abs(np.corrcoef(V_s[:, 1], B[:, k])[0, 1])) for k in (1, 2)]
print(f"corr(PC1, market loadings) {corr_pc1_mkt:.3f} · |cos(PC1, true PC1)| {cos_pc1:.5f} · "
      f"subspace R² {np.round(subspace_r2, 4)} · |corr(PC2, style 1/2)| {np.round(corr_pc2, 2)}")
lab.record("corr_pc1_mkt", corr_pc1_mkt)
lab.record("corr_true_pc1_mkt", corr_true_pc1_mkt)
lab.record("cos_pc1", cos_pc1)
lab.record("pc1_pos_share", float((pc1 > 0).mean()))
for k in range(N_FACTORS):
    lab.record(f"subspace_r2_f{k}", subspace_r2[k])
lab.record("subspace_r2_min", float(min(subspace_r2)))
lab.record("corr_pc2_style1", corr_pc2[0])
lab.record("corr_pc2_style2", corr_pc2[1])

# %%
N_MC_PCA = 300
n_dof = T_DAYS - 1
sin2_theory = sum(lam_true[0] * lam_true[j] / (lam_true[0] - lam_true[j]) ** 2
                  for j in range(1, N_ASSETS)) / n_dof
cross = np.array([[lam_true[i] * lam_true[j] / (lam_true[i] - lam_true[j]) ** 2
                   for j in range(N_FACTORS, N_ASSETS)] for i in range(N_FACTORS)]).sum(axis=1)
outside_theory = []
for k in range(N_FACTORS):
    c2 = (V_true[:, :N_FACTORS].T @ B[:, k]) ** 2
    outside_theory.append(float((c2 / c2.sum()) @ cross / n_dof))


def simulate_returns(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Fresh daily returns from the SAME true factor model (same loadings B), as a T × N array."""
    f = rng.standard_normal((n_days, N_FACTORS)) * F_D
    e = rng.standard_normal((n_days, N_ASSETS)) * E_D
    return f @ B.T + e


sin2_mc, outside_mc, frob2_mc = [], [], []
for _ in range(N_MC_PCA):
    S_rep = np.cov(simulate_returns(T_DAYS, rng_pca_mc), rowvar=False)
    _, V_rep = eig_desc(S_rep)
    sin2_mc.append(1 - (V_rep[:, 0] @ V_true[:, 0]) ** 2)
    outside_mc.append([outside_share(V_rep[:, :N_FACTORS], B[:, k]) for k in range(N_FACTORS)])
    frob2_mc.append(np.linalg.norm(S_rep - SIGMA) ** 2)
sin2_mc, outside_mc, frob2_mc = np.array(sin2_mc), np.array(outside_mc), np.array(frob2_mc)
se = lambda a: a.std(axis=0, ddof=1) / np.sqrt(len(a))  # noqa: E731 — standard error of a mean

assert abs(sin2_mc.mean() - sin2_theory) < 4 * se(sin2_mc), "PC1 angle: MC vs Anderson off by > 4 SE"
assert np.all(np.abs(outside_mc.mean(axis=0) - outside_theory) < 4 * se(outside_mc)), "subspace: > 4 SE"
assert abs(frob2_mc.mean() - frob2_theory) < 4 * se(frob2_mc), "Frobenius error: MC vs exact off by > 4 SE"
print(f"E[sin²θ] PC1: MC {sin2_mc.mean():.5f} ± {se(sin2_mc):.5f} vs theory {sin2_theory:.5f}")
print(f"outside share: MC {outside_mc.mean(axis=0).round(5)} vs theory {np.round(outside_theory, 5)}")
lab.record("n_mc_pca", N_MC_PCA)
lab.record("pc1_sin2_theory", float(sin2_theory))
lab.record("pc1_sin2_mc", float(sin2_mc.mean()))
lab.record("pc1_sin2_se", float(se(sin2_mc)))
lab.record("pc1_angle_deg_theory", float(np.degrees(np.arcsin(np.sqrt(sin2_theory)))))
lab.record("pc1_angle_deg_sample", float(np.degrees(np.arccos(min(cos_pc1, 1.0)))))
lab.record("mkt_outside_theory", outside_theory[0])
lab.record("mkt_outside_mc", float(outside_mc[:, 0].mean()))
lab.record("style_outside_theory_max", max(outside_theory[1:]))

# %% [markdown]
# ## 6 · The condition number, and what inverting Σ does to noise
#
# The condition number cond(Σ) = λ_max/λ_min measures how close a matrix is to singular. (We write
# cond, not the usual κ: the course reserves κ for the borrow fee of Session 2.2.) Anything
# that uses Σ⁻¹ — the minimum-variance portfolio w ∝ Σ⁻¹1, mean-variance weights, regressions —
# divides by the small eigenvalues, and the small eigenvalues of a sample Σ are exactly the ones
# estimated worst: they are biased DOWN, more so as T/N falls.
#
# For each ratio T/N we draw 500 samples from the same true market (N = 50), estimate S, and form
# the sample global minimum-variance (GMV) portfolio ŵ = S⁻¹1 / (1'S⁻¹1). For normal returns two
# results are exact (derived below; Kan & Smith, 2008, give the general theory):
#
#   promised (in-sample)  E[ŵ'Sŵ] / σ²_min = (T − N)/(T − 1)
#   realised (true)       E[ŵ'Σŵ] / σ²_min = (T − 2)/(T − N − 1)
#
# Derivation sketch: (T − 1)S is Wishart with n = T − 1 degrees of freedom. Write a = Σ^(−1/2)1 and
# rotate so a points along the first axis. The in-sample variance is σ²_min · χ²(n − N + 1)/n.
# The realised ratio is 1 + ‖b‖², where b = W₂₂⁻¹W₂₁ is a regression coefficient that, given W₂₂,
# is normal with covariance W₂₂⁻¹ — and E[tr W₂₂⁻¹] = (N − 1)/(n − N) for an inverse Wishart.

# %%
TN_RATIOS = (1.2, 1.5, 2, 3, 5, 10, 20, 50)
N_MC_TN = 500
ones = np.ones(N_ASSETS)
Sigma_inv_1 = np.linalg.solve(SIGMA, ones)
w_gmv = Sigma_inv_1 / Sigma_inv_1.sum()                 # the TRUE minimum-variance portfolio
var_min = 1 / Sigma_inv_1.sum()                         # its variance, σ²_min = 1/(1'Σ⁻¹1)
assert np.isclose(w_gmv @ SIGMA @ w_gmv, var_min)

tn_rows = []
for ratio in TN_RATIOS:
    T = int(round(ratio * N_ASSETS))
    cnd, v_in, v_out, w_err = [], [], [], []
    for _ in range(N_MC_TN):
        S_rep = np.cov(simulate_returns(T, rng_tn), rowvar=False)
        lam_rep = np.linalg.eigvalsh(S_rep)
        cnd.append(lam_rep[-1] / lam_rep[0])
        x = np.linalg.solve(S_rep, ones)
        w_hat = x / x.sum()
        v_in.append(w_hat @ S_rep @ w_hat / var_min)
        v_out.append(w_hat @ SIGMA @ w_hat / var_min)
        w_err.append(np.abs(w_hat - w_gmv).sum() / np.abs(w_gmv).sum())
    cnd, v_in, v_out, w_err = map(np.array, (cnd, v_in, v_out, w_err))
    row = {"ratio": ratio, "T": T, "cond": cnd.mean(), "cond_ratio": cnd.mean() / cond_true,
           "v_in": v_in.mean(), "v_in_se": se(v_in), "v_in_theory": (T - N_ASSETS) / (T - 1),
           "v_out": v_out.mean(), "v_out_se": se(v_out), "v_out_theory": (T - 2) / (T - N_ASSETS - 1),
           "w_err": w_err.mean()}
    # Monte Carlo against exact theory, at 4 standard errors
    assert abs(row["v_in"] - row["v_in_theory"]) < 4 * row["v_in_se"], f"T/N={ratio}: in-sample ratio off"
    assert abs(row["v_out"] - row["v_out_theory"]) < 4 * row["v_out_se"], f"T/N={ratio}: realised ratio off"
    assert np.all(v_out >= 1 - 1e-12)   # exact: no portfolio beats the true minimum variance
    tn_rows.append(row)
tn = pd.DataFrame(tn_rows).set_index("ratio")
print(tn[["T", "cond", "cond_ratio", "v_in", "v_in_theory", "v_out", "v_out_theory", "w_err"]].round(3))
# The fewer days per asset, the worse the conditioning (differences are huge relative to MC noise).
assert tn["cond_ratio"].is_monotonic_decreasing and (tn["cond_ratio"] > 1).all()

lab.record("n_mc_tn", N_MC_TN)
lab.record("gmv_vol_true", float(np.sqrt(var_min * PERIODS_PER_YEAR)))
lab.record("gmv_gross_true", float(np.abs(w_gmv).sum()))
for ratio, row in tn.iterrows():
    key = f"{ratio:g}".replace(".", "_")
    lab.record(f"tn_{key}_ratio", float(ratio))
    lab.record(f"tn_{key}_T", int(row["T"]))
    lab.record(f"tn_{key}_cond", float(row["cond"]))
    lab.record(f"tn_{key}_cond_ratio", float(row["cond_ratio"]))
    lab.record(f"tn_{key}_vin", float(row["v_in"]))
    lab.record(f"tn_{key}_vin_theory", float(row["v_in_theory"]))
    lab.record(f"tn_{key}_vout", float(row["v_out"]))
    lab.record(f"tn_{key}_vout_theory", float(row["v_out_theory"]))
    lab.record(f"tn_{key}_vol_gap", float(np.sqrt(row["v_out"] / row["v_in"])))   # realised ÷ promised vol
    lab.record(f"tn_{key}_werr", float(row["w_err"]))

# %% [markdown]
# ## 7 · When a "covariance matrix" is not one: missing data and positive semi-definiteness
#
# A real covariance matrix is positive semi-definite (PSD): w'Σw ≥ 0 for every w, because it is the
# variance of a portfolio. Equivalently, all eigenvalues are ≥ 0. Correlations cannot be chosen
# freely: if A is 0.9-correlated with both B and C, then B and C must be at least
# ρ₁₂ρ₁₃ − √((1 − ρ₁₂²)(1 − ρ₁₃²)) = 0.62 correlated. Declare them −0.9 and the "matrix" has a
# negative eigenvalue: a portfolio with negative variance.

# %%
R12, R13, R23_BAD = 0.9, 0.9, -0.9
C_bad = np.array([[1, R12, R13], [R12, 1, R23_BAD], [R13, R23_BAD, 1.0]])
lam_bad, V_bad = np.linalg.eigh(C_bad)
w_neg = V_bad[:, 0]                                           # eigenvector of the negative eigenvalue
assert lam_bad[0] < 0 and np.isclose(w_neg @ C_bad @ w_neg, lam_bad[0])   # "negative variance"
r23_min = R12 * R13 - np.sqrt((1 - R12**2) * (1 - R13**2))
C_edge = C_bad.copy()
C_edge[1, 2] = C_edge[2, 1] = r23_min
assert abs(np.linalg.eigvalsh(C_edge)[0]) < 1e-12              # at the bound the matrix is just singular
lab.record("bad_r12", R12)
lab.record("bad_r23", R23_BAD)
lab.record("bad_eig_min", float(lam_bad[0]))
lab.record("r23_min", float(r23_min))

# %% [markdown]
# **The realistic version.** One year of daily returns on the same 50 assets, but each asset is
# missing on a random 40% of days (illiquid assets, different holidays, feed gaps). Listwise
# deletion — keep only the days on which every asset has a price — leaves nothing. pandas'
# `DataFrame.cov()` silently uses **pairwise** deletion instead: each entry is estimated from the
# days on which that particular pair overlaps. Every entry is a reasonable estimate; the matrix as
# a whole need not be a covariance matrix of anything.

# %%
T_MISS, P_MISS = PERIODS_PER_YEAR, 0.40
X = pd.DataFrame(simulate_returns(T_MISS, rng_missing))
X_obs = X.mask(rng_missing.random(X.shape) < P_MISS)
n_complete = int(X_obs.dropna().shape[0])               # listwise deletion
obs = X_obs.notna().to_numpy().astype(float)
overlap = obs.T @ obs                                     # days each pair is observed together
mean_overlap = float(overlap[np.triu_indices(N_ASSETS, 1)].mean())
S_pw = X_obs.cov().to_numpy()                             # pandas default: pairwise-complete
lam_pw, V_pw = eig_desc(S_pw)
n_neg = int((lam_pw < 0).sum())
assert n_neg >= 1, "pairwise estimate should not be PSD in this design"

w_bad = V_pw[:, -1] / np.abs(V_pw[:, -1]).sum()          # most negative eigen-portfolio, gross 100%
est_var_bad = w_bad @ S_pw @ w_bad                        # the matrix claims a NEGATIVE variance
true_vol_bad = np.sqrt(w_bad @ SIGMA @ w_bad * PERIODS_PER_YEAR)
assert est_var_bad < 0

# Repair by eigenvalue clipping: set negative eigenvalues to zero and rebuild. This is the
# Frobenius-nearest PSD matrix, a projection onto a convex set that contains the truth — so it can
# only move the estimate CLOSER to Σ (exact inequality, not a statistical claim).
S_clip = V_pw @ np.diag(np.clip(lam_pw, 0, None)) @ V_pw.T
lam_clip = np.linalg.eigvalsh(S_clip)
assert lam_clip.min() > -1e-12 * lam_clip.max()
err_before = np.linalg.norm(S_pw - SIGMA) / np.linalg.norm(SIGMA)
err_after = np.linalg.norm(S_clip - SIGMA) / np.linalg.norm(SIGMA)
assert err_after <= err_before
# The caveat: every clipped direction now claims ZERO risk — yet it is a real, risky portfolio.
est_var_clip = w_bad @ S_clip @ w_bad
assert abs(est_var_clip) < 1e-12
var_change = np.abs(np.diag(S_clip) / np.diag(S_pw) - 1).max()   # clipping moved the variances too
print(f"listwise rows left {n_complete} · mean pair overlap {mean_overlap:.0f} days · negative eigenvalues {n_neg} · "
      f"'variance' of worst portfolio {est_var_bad * PERIODS_PER_YEAR:.4f}/yr, true vol {true_vol_bad:.1%} · "
      f"error vs truth {err_before:.1%} → {err_after:.1%}")
lab.record("miss_days", T_MISS)
lab.record("miss_p", P_MISS)
lab.record("miss_complete_rows", n_complete)
lab.record("miss_mean_overlap", mean_overlap)
lab.record("miss_n_neg", n_neg)
lab.record("miss_eig_min_ann", float(lam_pw[-1] * PERIODS_PER_YEAR))
lab.record("miss_bad_var_ann", float(est_var_bad * PERIODS_PER_YEAR))
lab.record("miss_bad_true_vol", float(true_vol_bad))
lab.record("miss_err_before", float(err_before))
lab.record("miss_err_after", float(err_after))
lab.record("miss_var_change_max", float(var_change))

# %% [markdown]
# ## 8 · Charts for the session page

# %%
lab.chart("diversification", charts.line_chart(
    [charts.Series("ρ = 0", curves[0.0], role="strategy", end_label="ρ = 0"),
     charts.Series("ρ = 0.3", curves[0.3], role="alt1", end_label="ρ = 0.3"),
     charts.Series("ρ = 0.6", curves[0.6], role="alt2", end_label="ρ = 0.6")],
    title="Equal-weight portfolio volatility as a share of one asset's, by number of assets N",
    y_fmt=charts.fmt_pct(0), y_min=0.0, y_max=1.0))

cats = [f"PC{k}" for k in range(1, 9)]
lab.chart("spectrum", charts.grouped_column_chart(
    cats, [("sample, 10 years of data", share_s[:8].tolist(), "strategy"),
           ("truth", share_true[:8].tolist(), "benchmark")],
    title="Share of total variance carried by each principal component (50 assets)",
    y_fmt=charts.fmt_pct(0)))

lab.chart("pc1_loadings", charts.scatter_chart(
    B[:, 0], pc1, title="PC1 weight vs the planted market loading, one dot per asset",
    x_fmt=charts.fmt_num(2), y_fmt=charts.fmt_num(2), fit_line=True,
    x_label="true market loading (β)", y_label="PC1 weight"))

lab.chart("gmv_risk", charts.grouped_column_chart(
    [f"{r:g}" for r in TN_RATIOS],
    [("promised (in-sample)", tn["v_in"].tolist(), "alt1"),
     ("realised (true)", tn["v_out"].tolist(), "loss")],
    title="Min-variance portfolio: variance ÷ true minimum, by days per asset (T/N)",
    y_fmt=charts.fmt_num(1, suffix="×")))

# %%
lab.save()
