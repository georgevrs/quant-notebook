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
# # 6.6 · Validation: Walk-Forward, Purged K-Fold & CPCV — companion lab
#
# **Quant Notebook** · Unit 6 · Session 6 · [Read the session](https://georgevrs.github.io/quant-notebook/unit06-backtesting-discipline/session06-validation-schemes.html)
#
# Walk-forward analysis, purging and embargo, and combinatorial purged cross-validation — train/test splits that respect time.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# All returns in this lab are synthetic **excess** returns with a risk-free rate of 0 — there is
# nothing to subtract, but we say so once, per convention (Session 2.3).

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import math
from itertools import combinations

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from skfolio.model_selection import CombinatorialPurgedCV

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR, cumulative_growth, in_years

lab = qn.Lab("6.6")  # seeds the random generators: every run gives the same numbers
# One independent random stream per experiment (AUTHORING.md §12a) — never a shared stream.
rng_wf, rng_leak = lab.rng.spawn(2)

# %% [markdown]
# ## 1 · Walk-forward analysis: anchored vs rolling
#
# A synthetic return series with a small, real, first-order autocorrelation `PHI` — a momentum
# edge, not a mean-reversion one. The only "model" is: pick a lookback `L` from a grid, and go
# with the sign of the trailing `L`-day sum of returns. We know the truth (`PHI`), so we can
# check whether a validation scheme reports it honestly or flatters it.

# %%
PHI, SIGMA, N_DAYS = 0.06, 0.01, 2_016          # ~8 simulated years of daily excess returns
GRID = [5, 10, 20, 40, 60, 80]                  # candidate lookbacks, in trading days
N_FOLDS, TEST_LEN = 8, 126                      # 8 walk-forward folds of half a simulated year
WARMUP = max(GRID)                              # no signal until the longest lookback has data


def make_ar1_returns(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Daily excess returns r_t = phi * r_{t-1} + eps_t, eps ~ N(0, sigma^2)."""
    r = np.zeros(n)
    eps = rng.standard_normal(n) * sigma
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return r


def lookback_returns(r: np.ndarray, L: int) -> np.ndarray:
    """Strategy return at t: sign of the trailing L-day sum (known at t), times r[t]."""
    n = len(r)
    pos = np.zeros(n)
    for t in range(L, n):
        pos[t] = np.sign(r[t - L:t].sum())
    return pos * r


def ann_sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))


def best_lookback_sharpe(r: np.ndarray, grid: list[int], warmup: int) -> tuple[int, float]:
    sharpes = {L: ann_sharpe(lookback_returns(r, L)[warmup:]) for L in grid}
    best_L = max(sharpes, key=sharpes.get)
    return best_L, sharpes[best_L]


def walk_forward(r: np.ndarray, grid: list[int], n_folds: int, test_len: int, warmup: int,
                 anchored: bool) -> np.ndarray:
    """Roll forward: pick the best lookback on the TRAIN window only, apply it on the next
    TEST window, then move on. Anchored = train window grows; rolling = train window slides,
    same length every fold (fig 6.6.2 shows both)."""
    n = len(r)
    total_test = n_folds * test_len
    win_len = n - total_test - warmup            # length of the first (anchored) train window
    oos = []
    for i in range(n_folds):
        test_start = n - total_test + i * test_len
        test_end = test_start + test_len
        train_start = warmup if anchored else max(warmup, test_start - win_len)
        best_L, _ = best_lookback_sharpe(r[train_start:test_start], grid, 0)
        fold_returns = lookback_returns(r[:test_end], best_L)
        oos.append(fold_returns[test_start:test_end])
    return np.concatenate(oos)


# %% [markdown]
# **One illustrative path.** Compare the Sharpe ratio you would report if you (wrongly) picked
# the lookback using the WHOLE sample and backtested on that same sample, against the honest
# walk-forward Sharpe that never lets a fold see its own test window while choosing `L`.

# %%
r_demo = make_ar1_returns(N_DAYS, PHI, SIGMA, rng_wf)
is_best_L, is_sharpe = best_lookback_sharpe(r_demo, GRID, WARMUP)
oos_anchored = walk_forward(r_demo, GRID, N_FOLDS, TEST_LEN, WARMUP, anchored=True)
oos_rolling = walk_forward(r_demo, GRID, N_FOLDS, TEST_LEN, WARMUP, anchored=False)
sh_anchored, sh_rolling = ann_sharpe(oos_anchored), ann_sharpe(oos_rolling)
print(f"in-sample L={is_best_L} Sharpe={is_sharpe:.2f} | walk-forward anchored={sh_anchored:.2f} "
      f"rolling={sh_rolling:.2f}")

lab.record("wf_phi", PHI)
lab.record("wf_n_days", N_DAYS)
lab.record("wf_n_folds", N_FOLDS)
lab.record("wf_test_len", TEST_LEN)
lab.record("wf_is_best_L", is_best_L)
lab.record("wf_is_sharpe", is_sharpe)
lab.record("wf_anchored_sharpe", sh_anchored)
lab.record("wf_rolling_sharpe", sh_rolling)
# a single path is a demo, not a proof: the Monte Carlo assert below is the real claim
is_returns_oos = lookback_returns(r_demo, is_best_L)[-len(oos_anchored):]  # same window, look-ahead L
lab.chart("wf_equity", charts.line_chart(
    [charts.Series("walk-forward (honest)", in_years(cumulative_growth(oos_anchored)), role="strategy",
                   end_label="honest"),
     charts.Series("in-sample L (look-ahead)", in_years(cumulative_growth(is_returns_oos)),
                   role="loss", end_label="look-ahead")],
    title="Growth of $1 over the out-of-sample window: honest vs look-ahead",
    y_fmt=charts.fmt_num(2, prefix="$"), hline=1.0))

# %% [markdown]
# **Is the gap real, or one lucky path?** Repeat with fresh synthetic data and check the average
# gap against its Monte Carlo standard error — at 4 SE, never 3, never by hunting for a seed.

# %%
N_MC_WF = 60
gap_anchored = np.empty(N_MC_WF)
gap_rolling = np.empty(N_MC_WF)
for i in range(N_MC_WF):
    r_i = make_ar1_returns(N_DAYS, PHI, SIGMA, rng_wf)
    _, is_sh_i = best_lookback_sharpe(r_i, GRID, WARMUP)
    gap_anchored[i] = is_sh_i - ann_sharpe(walk_forward(r_i, GRID, N_FOLDS, TEST_LEN, WARMUP, True))
    gap_rolling[i] = is_sh_i - ann_sharpe(walk_forward(r_i, GRID, N_FOLDS, TEST_LEN, WARMUP, False))

se_anchored = gap_anchored.std(ddof=1) / np.sqrt(N_MC_WF)
se_rolling = gap_rolling.std(ddof=1) / np.sqrt(N_MC_WF)
lab.record("wf_mc_n", N_MC_WF)
lab.record("wf_gap_anchored_mean", float(gap_anchored.mean()))
lab.record("wf_gap_anchored_se", float(se_anchored))
lab.record("wf_gap_rolling_mean", float(gap_rolling.mean()))
lab.record("wf_gap_rolling_se", float(se_rolling))
print(f"anchored gap {gap_anchored.mean():.3f} (SE {se_anchored:.3f}, "
      f"{gap_anchored.mean() / se_anchored:.1f} SE) | rolling gap {gap_rolling.mean():.3f} "
      f"(SE {se_rolling:.3f}, {gap_rolling.mean() / se_rolling:.1f} SE)")
assert gap_anchored.mean() > 4 * se_anchored, "in-sample selection should beat honest walk-forward by > 4 SE"
assert gap_rolling.mean() > 4 * se_rolling, "in-sample selection should beat honest walk-forward by > 4 SE"

# %% [markdown]
# ## 2 · Purging and embargo, precisely
#
# A label with a forward-looking horizon `H` depends on information in `[t+1, t+H]`. Two labels
# overlap whenever their horizons share a day. **Purging** removes every training observation
# whose horizon overlaps the test set's; **embargo** removes a further block right after the test
# set, because features are often serially correlated even where labels no longer overlap
# (skfolio's docs state this exactly the way AFML does — see §6.6 references).

# %%
def purge_and_embargo(n: int, test_blocks: list[tuple[int, int]], H: int, embargo: int) -> np.ndarray:
    """Boolean train mask: True = usable for training.

    For a test block [start, stop), any observation i with a horizon [i+1, i+H] that intersects
    some test observation's horizon is purged; that is every i in [start-H+1, stop+H-2]. Embargo
    then removes `embargo` MORE observations right after the block (one-sided: AFML purges
    symmetrically for label overlap, but embargoes only forward, because time only flows forward).
    """
    mask = np.ones(n, dtype=bool)
    for start, stop in test_blocks:
        mask[start:stop] = False
        lo = max(0, start - (H - 1))
        hi = min(n, stop + (H - 1))
        mask[lo:hi] = False
        mask[stop:min(n, stop + embargo)] = False
    return mask


# a small worked example, sized to fit a diagram: 20 observations, one test fold in the middle
TOY_N, TOY_H, TOY_EMBARGO = 20, 4, 6
toy_test = (8, 12)
toy_mask = purge_and_embargo(TOY_N, [toy_test], TOY_H, TOY_EMBARGO)
toy_train_idx = np.where(toy_mask)[0]
lab.record("toy_n", TOY_N)
lab.record("toy_H", TOY_H)
lab.record("toy_embargo", TOY_EMBARGO)
lab.record("toy_test_start", toy_test[0])
lab.record("toy_test_stop", toy_test[1])          # exclusive bound (numpy/Python slice convention)
lab.record("toy_test_last", toy_test[1] - 1)       # last INCLUDED test index, for prose
lab.record("toy_purge_lo", int(max(0, toy_test[0] - (TOY_H - 1))))
lab.record("toy_purge_hi", int(min(TOY_N, toy_test[1] + (TOY_H - 1))) - 1)     # last purged index
lab.record("toy_embargo_hi", int(min(TOY_N, toy_test[1] + TOY_EMBARGO)) - 1)  # last excluded index
lab.record("toy_train_count", int(toy_mask.sum()))
print(f"toy example: train indices kept = {toy_train_idx.tolist()}")
# every purged/embargoed index is provably excluded, and nothing outside that zone is touched
assert not toy_mask[5:18].any(), "purge+embargo should clear indices 5..17 for this toy example"
assert toy_mask[:5].all() and toy_mask[18:].all(), "untouched indices must remain trainable"

# %% [markdown]
# ## 3 · Overlapping labels: why naive k-fold lies
#
# Build a labeled panel the way a triple-barrier-style label (Session 12.2) actually behaves: a
# feature observed now, and a label that sums a LATENT, autocorrelated process over the next `H`
# days. Consecutive labels overlap almost completely when `H` is large relative to the sampling
# step — exactly the situation `sklearn.model_selection.KFold(shuffle=True)` was not designed for.

# %%
N_LEAK, H_LEAK, RHO, M_LAGS = 2_000, 40, 0.90, 5
FEAT_NOISE, LABEL_NOISE = 0.30, 0.20
K_NEIGHBORS, N_SPLITS_LEAK = 3, 5


def make_overlap_dataset(n: int, H: int, rho: float, m_lags: int, feat_noise: float,
                         label_noise: float, rng: np.random.Generator):
    """Features = m_lags noisy readings of an AR(1) latent regime; label = sign of its forward
    H-day sum. Consecutive samples' labels overlap by up to H-1 days (fig 6.6.3)."""
    start = m_lags - 1
    total = n + start + H + 1
    z = np.empty(total)
    eps = rng.standard_normal(total)
    z[0] = eps[0]
    for t in range(1, total):
        z[t] = rho * z[t - 1] + math.sqrt(1 - rho ** 2) * eps[t]
    lags = np.stack([z[start - i: start - i + n] for i in range(m_lags)], axis=1)
    X = lags + feat_noise * rng.standard_normal(lags.shape)
    csum = np.cumsum(z)
    t_idx = np.arange(start, start + n)
    S = csum[t_idx + H] - csum[t_idx] + label_noise * rng.standard_normal(n)  # sum z[t+1..t+H]
    y = (S > 0).astype(int)
    return X, y


def bayes_optimal_accuracy(H: int, rho: float, m_lags: int, feat_noise: float,
                           label_noise: float) -> float:
    """The best ANY classifier could do from these features (arcsin law for jointly normal
    variables: P(sign(w'x) = sign(S)) = 1/2 + arcsin(r)/pi at the optimal linear projection w)."""
    C1 = sum(rho ** j for j in range(1, H + 1))
    b = np.array([rho ** i for i in range(m_lags)])
    idx = np.arange(m_lags)
    Sigma_x = rho ** np.abs(idx[:, None] - idx[None, :]) + feat_noise ** 2 * np.eye(m_lags)
    Var_S = H + 2 * sum((H - k) * rho ** k for k in range(1, H)) + label_noise ** 2
    r2 = (C1 ** 2) * (b @ np.linalg.solve(Sigma_x, b)) / Var_S
    return float(0.5 + math.asin(math.sqrt(r2)) / math.pi)


class PurgedKFold:
    """Contiguous k-fold with purging and embargo — the from-scratch splitter this session
    contributes to quantnb (see the handoff)."""
    def __init__(self, n_splits: int, H: int, embargo: int):
        self.n_splits, self.H, self.embargo = n_splits, H, embargo

    def split(self, n: int):
        sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        sizes[: n % self.n_splits] += 1
        bounds = np.cumsum(sizes)
        starts = np.concatenate([[0], bounds[:-1]])
        for start, stop in zip(starts, bounds):
            mask = purge_and_embargo(n, [(start, stop)], self.H, self.embargo)
            yield np.where(mask)[0], np.arange(start, stop)


def cv_accuracy(splits, X: np.ndarray, y: np.ndarray, k: int) -> float:
    accs = []
    for tr, te in splits:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


bayes_acc = bayes_optimal_accuracy(H_LEAK, RHO, M_LAGS, FEAT_NOISE, LABEL_NOISE)
cut = int(N_LEAK * 0.7)
holdout_train = np.arange(0, cut - H_LEAK)
holdout_test = np.arange(cut + H_LEAK, N_LEAK)

# %% [markdown]
# **Is one dataset's number a fluke?** A single random draw is noisy (kNN accuracy here has a
# sample standard deviation of a few percentage points), so redraw the whole labeled panel from
# scratch many times and report the AVERAGE of each scheme, plus the (naive − purged) gap against
# its Monte Carlo standard error — 4 SE, never 3, never by hunting for a seed. The first redraw
# doubles as `X_demo, y_demo`, the concrete dataset the rest of this section and CPCV (§4) use.

# %%
N_MC_LEAK = 50
naive_runs = np.empty(N_MC_LEAK)
purged_runs = np.empty(N_MC_LEAK)
holdout_runs = np.empty(N_MC_LEAK)
X_demo = y_demo = None
for i in range(N_MC_LEAK):
    X_i, y_i = make_overlap_dataset(N_LEAK, H_LEAK, RHO, M_LAGS, FEAT_NOISE, LABEL_NOISE, rng_leak)
    if i == 0:
        X_demo, y_demo = X_i, y_i
    naive_runs[i] = cv_accuracy(KFold(n_splits=N_SPLITS_LEAK, shuffle=True, random_state=i).split(X_i),
                               X_i, y_i, K_NEIGHBORS)
    purged_runs[i] = cv_accuracy(PurgedKFold(N_SPLITS_LEAK, H_LEAK, H_LEAK).split(N_LEAK),
                                X_i, y_i, K_NEIGHBORS)
    holdout_runs[i] = cv_accuracy([(holdout_train, holdout_test)], X_i, y_i, K_NEIGHBORS)

naive_acc, purged_acc, holdout_acc = naive_runs.mean(), purged_runs.mean(), holdout_runs.mean()
gap_leak = naive_runs - purged_runs
se_leak = gap_leak.std(ddof=1) / np.sqrt(N_MC_LEAK)
print(f"Bayes bound {bayes_acc:.3f} | naive CV {naive_acc:.3f} | purged CV {purged_acc:.3f} "
      f"| holdout {holdout_acc:.3f} (means over {N_MC_LEAK} redraws)")
print(f"naive − purged gap = {gap_leak.mean():.4f} (SE {se_leak:.4f}, "
      f"{gap_leak.mean() / se_leak:.1f} SE)")
assert gap_leak.mean() > 4 * se_leak, "naive k-fold should beat purged k-fold by > 4 SE — that IS the leak"

lab.record("leak_n", N_LEAK)
lab.record("leak_H", H_LEAK)
lab.record("leak_rho", RHO)
lab.record("leak_m_lags", M_LAGS)
lab.record("leak_k_neighbors", K_NEIGHBORS)
lab.record("leak_mc_n", N_MC_LEAK)
lab.record("leak_bayes_acc", bayes_acc)
lab.record("leak_naive_acc", naive_acc)
lab.record("leak_purged_acc", purged_acc)
lab.record("leak_holdout_acc", holdout_acc)
lab.record("leak_gap_mean", float(gap_leak.mean()))
lab.record("leak_gap_se", float(se_leak))
lab.chart("leak_scores", charts.column_chart(
    ["Naive k-fold", "Purged k-fold", "Holdout", "Bayes bound"],
    [naive_acc, purged_acc, holdout_acc, bayes_acc],
    title="Same data, same classifier, four validation scores (means over 50 redraws)",
    y_fmt=charts.fmt_pct(1), roles=["loss", "strategy", "alt2", "benchmark"]))
lab.chart("leak_mc_hist", charts.histogram(
    gap_leak, title="Naive-minus-purged accuracy gap across 50 redrawn datasets",
    x_fmt=charts.fmt_pct(1), tail_below=0.0, tail_label="purged wins (no leak)"))

# %% [markdown]
# ## 4 · Combinatorial purged cross-validation (CPCV)
#
# Partition into `N` groups, test on every combination of `k` of them (purging and embargo applied
# to each), and stitch the out-of-fold predictions back into complete paths. Each of the `N` groups
# appears as a test group in exactly `C(N-1, k-1)` of the `C(N, k)` splits — and that count is also
# the number of complete paths you can reconstruct (López de Prado, 2018, ch. 12).

# %%
def n_paths(N: int, k: int) -> int:
    return math.comb(N - 1, k - 1)


# the book's own two examples — pure combinatorics, no simulation needed
lab.record("cpcv_book_small_N", 6)
lab.record("cpcv_book_small_k", 2)
lab.record("cpcv_book_small_paths", n_paths(6, 2))
lab.record("cpcv_book_example_N", 15)
lab.record("cpcv_book_example_k", 11)
lab.record("cpcv_book_example_paths", n_paths(15, 11))
assert n_paths(6, 2) == 5
assert n_paths(15, 11) == 1_001, "reproduces AFML's own (N=15, k=11) -> 1,001-path example"

# %%
def round_robin_matchings(n: int) -> list[list[tuple[int, int]]]:
    """Circle method: a 1-factorization of K_n (n even) into n-1 perfect matchings — for k=2,
    each matching IS one reconstructed CPCV path (every group tested exactly once per matching,
    using a different, disjoint split each time)."""
    players = list(range(n))
    matchings = []
    for _ in range(n - 1):
        matchings.append([tuple(sorted((players[i], players[n - 1 - i]))) for i in range(n // 2)])
        players = [players[0]] + [players[-1]] + players[1:-1]
    return matchings


N_GROUPS, K_TEST = 10, 2
matchings = round_robin_matchings(N_GROUPS)
all_pairs = set(combinations(range(N_GROUPS), 2))
seen_pairs = [p for m in matchings for p in m]
assert sorted(seen_pairs) == sorted(all_pairs), "every pair of groups must be tested exactly once"
assert all(sorted(x for pair in m for x in pair) == list(range(N_GROUPS)) for m in matchings), \
    "each matching must be a perfect matching (every group appears exactly once)"
assert len(matchings) == n_paths(N_GROUPS, K_TEST) == math.comb(N_GROUPS - 1, K_TEST - 1)

# cross-check the split/path counts against skfolio's independent implementation (deps: portfolio)
skf_cv = CombinatorialPurgedCV(n_folds=N_GROUPS, n_test_folds=K_TEST)
assert skf_cv.get_n_splits() == math.comb(N_GROUPS, K_TEST)
assert skf_cv.n_test_paths == n_paths(N_GROUPS, K_TEST)
lab.record("cpcv_N", N_GROUPS)
lab.record("cpcv_k", K_TEST)
lab.record("cpcv_n_splits", math.comb(N_GROUPS, K_TEST))
lab.record("cpcv_n_paths", n_paths(N_GROUPS, K_TEST))

# %% [markdown]
# Now run it on the SAME overlapping-label dataset from §3: partition into 10 contiguous groups,
# purge and embargo every one of the 45 splits with respect to BOTH held-out groups, and stitch
# the 9 reconstructed paths' accuracies into a distribution instead of one number.

# %%
group_bounds = np.array_split(np.arange(N_LEAK), N_GROUPS)
group_range = [(int(g[0]), int(g[-1]) + 1) for g in group_bounds]
pair_accuracy: dict[tuple[int, int], tuple[float, float]] = {}
for g, h in combinations(range(N_GROUPS), 2):
    test_blocks = [group_range[g], group_range[h]]
    mask = purge_and_embargo(N_LEAK, test_blocks, H_LEAK, H_LEAK)
    train_idx = np.where(mask)[0]
    clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS).fit(X_demo[train_idx], y_demo[train_idx])
    acc_g = clf.score(X_demo[group_range[g][0]:group_range[g][1]], y_demo[group_range[g][0]:group_range[g][1]])
    acc_h = clf.score(X_demo[group_range[h][0]:group_range[h][1]], y_demo[group_range[h][0]:group_range[h][1]])
    pair_accuracy[(g, h)] = (acc_g, acc_h)

path_accuracies = []
for matching in matchings:
    per_group = np.empty(N_GROUPS)
    for g, h in matching:
        acc_g, acc_h = pair_accuracy[(g, h)]
        per_group[g], per_group[h] = acc_g, acc_h
    path_accuracies.append(float(per_group.mean()))
path_accuracies = np.array(path_accuracies)

lab.record("cpcv_path_acc_mean", float(path_accuracies.mean()))
lab.record("cpcv_path_acc_sd", float(path_accuracies.std(ddof=1)))
lab.record("cpcv_path_acc_min", float(path_accuracies.min()))
lab.record("cpcv_path_acc_max", float(path_accuracies.max()))
print(f"9 CPCV paths: mean {path_accuracies.mean():.3f}, sd {path_accuracies.std(ddof=1):.3f}, "
      f"range [{path_accuracies.min():.3f}, {path_accuracies.max():.3f}]")
lab.chart("cpcv_paths", charts.column_chart(
    [f"path {i + 1}" for i in range(len(path_accuracies))], path_accuracies.tolist(),
    title="Nine reconstructed CPCV paths, same dataset (N=10, k=2)",
    y_fmt=charts.fmt_pct(1), roles=["strategy"] * len(path_accuracies)))

# %%
lab.save()
