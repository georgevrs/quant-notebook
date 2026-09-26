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
# # 1.5 · Set Up Your Quant Lab — companion lab
#
# **Quant Notebook** · Unit 1 · Session 5 · [Read the session](https://georgevrs.github.io/quant-notebook/unit01-quant-landscape/session05-quant-lab-setup.html)
#
# A reproducible workbench — uv, Python 3.12, Jupyter, DuckDB, the quantnb package, a research repo
# layout and a data cache. This lab is the workbench's acceptance test. It checks your environment
# against the lockfile, proves that the same seed gives the same numbers, shows three ways
# randomness leaks into results anyway, builds a cached, fingerprinted Parquet dataset, and
# queries it with pandas, DuckDB SQL and a polars lazy frame to check they give the same answer.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
# Machine-dependent facts (versions, paths, timings) are **printed, never recorded**: the recorded
# results must be identical on every machine.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import hashlib
import json
import math
import os
import platform
import time
import tomllib
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR, sharpe_ratio, simple_returns

lab = qn.Lab("1.5")  # seeds the random generators: every run gives the same numbers
rng = lab.rng

# %% [markdown]
# ## 1 · Environment self-check
#
# Is this the environment the labs were tested in? Inside the repository, `uv sync` installed
# exactly the versions pinned in `uv.lock`; this cell compares what is installed with what is
# locked. In Colab there is no lockfile: you get whatever that runtime ships, which is fine for
# exploring and not fine for results you want to reproduce.
#
# Everything here is printed, not recorded: versions and paths differ between machines, and the
# recorded results must not.

# %%
KEY_PACKAGES = ["numpy", "pandas", "polars", "pyarrow", "duckdb", "scipy", "statsmodels", "quantnb"]


def installed_version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def locked_versions(root: Path | None) -> dict[str, str]:
    """Package → version pinned in uv.lock (empty outside the repo, e.g. in Colab)."""
    lock = root / "uv.lock" if root else None
    if lock is None or not lock.exists():
        return {}
    with open(lock, "rb") as fh:
        data = tomllib.load(fh)
    return {p["name"]: p.get("version", "") for p in data.get("package", [])}


def environment_report() -> dict:
    """What a results file should carry so a reader can rebuild the environment that made it."""
    root = qn.repo_root()
    locked = locked_versions(root)
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "interpreter": sys.executable,
        "in_repo": root is not None,
        "virtual_env": sys.prefix != sys.base_prefix,
        "packages": {p: (installed_version(p), locked.get(p)) for p in KEY_PACKAGES},
    }


env = environment_report()
print(f"Python {env['python']} ({env['implementation']}) on {env['os']}")
print(f"interpreter: {env['interpreter']}")
print(f"inside the course repo: {env['in_repo']} · inside a virtual environment: {env['virtual_env']}")
print(f"data cache: {qn.data_cache()} · offline mode: {qn.offline()}")
print(f"\n{'package':<12} {'installed':<12} {'uv.lock':<12} status")
for name, (have, want) in env["packages"].items():
    status = ("no lockfile" if want is None else "matches lock" if have == want else "DIFFERS from lock")
    print(f"{name:<12} {have or 'missing':<12} {want or '-':<12} {status}")

assert sys.version_info >= (3, 12), "the course targets Python 3.12+ (see .python-version)"
assert all(v[0] for v in env["packages"].values()), "a core package is missing: run `uv sync`"

# %% [markdown]
# ## 2 · Same seed, same numbers
#
# A pseudo-random generator is a deterministic function of its seed. Create two generators from
# the same seed and they produce the same stream; change the seed by one and everything changes.
# Always create a **`numpy.random.Generator`** and pass it around explicitly — never call the
# legacy global `np.random.seed(...)` and hope nothing else draws from it.

# %%
SEED = 42

a = np.random.default_rng(SEED).standard_normal(3)
b = np.random.default_rng(SEED).standard_normal(3)
c = np.random.default_rng(SEED + 1).standard_normal(3)
print("seed 42, run 1:", a)
print("seed 42, run 2:", b)
print("seed 43       :", c)
assert np.array_equal(a, b), "same seed must give bit-identical draws"
assert not np.allclose(a, c), "a different seed must give different draws"
for i, v in enumerate(a, start=1):
    lab.record(f"seed42_draw{i}", float(v))
lab.record("seed43_draw1", float(c[0]))

# %% [markdown]
# **A fingerprint you can compare across machines.** Hash 100,000 seeded integers. Integer
# generation involves no floating-point maths, so the hash is identical on every machine with the
# same NumPy — the page quotes it, and CI re-checks it whenever the lab changes.

# %%
N_FINGERPRINT = 100_000

# qn.repro.fingerprint = the first 16 hex digits of SHA-256 over the array's int64 bytes.
# Read src/quantnb/repro.py: it is four lines.
fp_run1 = qn.repro.fingerprint(np.random.default_rng(SEED).integers(0, 1_000_000, size=N_FINGERPRINT))
fp_run2 = qn.repro.fingerprint(np.random.default_rng(SEED).integers(0, 1_000_000, size=N_FINGERPRINT))
print(f"fingerprint of {N_FINGERPRINT:,} seeded integers: {fp_run1} (run 2: {fp_run2})")
assert fp_run1 == fp_run2
lab.record("int_fingerprint", fp_run1)
lab.record("n_fingerprint", N_FINGERPRINT)

# %% [markdown]
# ## 3 · A seed is a trial
#
# Same strategy, same code, same synthetic market — 20 different seeds. The strategy's true
# Sharpe ratio is μ/σ = 0.07/0.18 ≈ 0.39 (synthetic excess returns, risk-free rate 0). Five years
# of daily data measure it with a standard error of about 1/√Y ≈ 0.45 (Lo's formula, via
# `qn.stats.sharpe_se`; Session 1.3's rule of thumb), so the seed alone moves the answer a lot.
# Rerunning with new seeds until the result looks good is a hidden trial, exactly like the ones
# counted in Session 1.3.

# %%
MU, SIGMA, YEARS = 0.07, 0.18, 5
SEEDS = list(range(20))

seed_sr = []
for s in SEEDS:
    px = qn.synth.gbm_prices(PERIODS_PER_YEAR * YEARS, mu=MU, sigma=SIGMA,
                             rng=np.random.default_rng(s))["price"]
    seed_sr.append(sharpe_ratio(simple_returns(px), 0.0))
seed_sr = np.array(seed_sr)
true_sr = MU / SIGMA
se_theory = qn.stats.sharpe_se(true_sr, YEARS)   # ≈ 1/√YEARS for daily data
best = int(np.argmax(seed_sr))
print(np.round(seed_sr, 2))
print(f"true Sharpe {true_sr:.2f} · across seeds: min {seed_sr.min():.2f}, max {seed_sr.max():.2f} "
      f"(seed {best}), mean {seed_sr.mean():.2f}, sd {seed_sr.std(ddof=1):.2f} vs theory {se_theory:.2f}")

lab.record("seed_n", len(SEEDS))
lab.record("seed_true_sr", true_sr)
lab.record("seed_se_theory", se_theory)
lab.record("seed_sr_min", float(seed_sr.min()))
lab.record("seed_sr_max", float(seed_sr.max()))
lab.record("seed_sr_mean", float(seed_sr.mean()))
lab.record("seed_sr_sd", float(seed_sr.std(ddof=1)))
lab.record("seed_best", best)
lab.record("seed_max_over_true", float(seed_sr.max() / true_sr))
lab.record("seed_years", YEARS)
assert seed_sr.max() - seed_sr.min() > 1.5 * se_theory, "the seed alone moves a 5-year Sharpe a lot"
assert abs(seed_sr.mean() - true_sr) < 3 * se_theory / math.sqrt(len(SEEDS))

# %% [markdown]
# ## 4 · Shared streams leak: the cache that changed my numbers
#
# Seeding is necessary, not sufficient. If two parts of a pipeline draw from **one** generator,
# anything that changes how many draws the first part takes silently shifts every number after
# it. The classic culprit is a cache: on the first run the synthetic data is built (consuming
# draws); on the second run it is read from disk (consuming none), so the bootstrap that follows
# sees a different stream. Same seed, same data, different answer.
#
# The fix is one independent child stream per component, spawned from the seed.

# %%
BOOT_SEED = 2026
N_DAYS, N_BOOT = 5 * PERIODS_PER_YEAR, 2_000


def build_returns(gen: np.random.Generator) -> np.ndarray:
    """Five years of synthetic daily returns (the 'expensive' data step we cache)."""
    return gen.normal(0.0004, 0.01, N_DAYS)


def bootstrap_low(returns: np.ndarray, gen: np.random.Generator) -> float:
    """5th percentile of the bootstrapped annualised mean return."""
    idx = gen.integers(0, len(returns), size=(N_BOOT, len(returns)))
    return float(np.quantile(returns[idx].mean(axis=1) * PERIODS_PER_YEAR, 0.05))


def run_pipeline(cache: dict, streams: str) -> float:
    """One run of data → bootstrap. `streams` is 'shared' (one generator) or 'spawned' (two)."""
    if streams == "shared":
        data_gen = boot_gen = np.random.default_rng(BOOT_SEED)
    else:
        data_gen, boot_gen = np.random.default_rng(BOOT_SEED).spawn(2)
    if "returns" not in cache:                  # cache miss: build (this consumes draws)
        cache["returns"] = build_returns(data_gen)
    return bootstrap_low(cache["returns"], boot_gen)


shared_cache: dict = {}
shared_run1 = run_pipeline(shared_cache, "shared")    # cold cache: builds the data
shared_run2 = run_pipeline(shared_cache, "shared")    # warm cache: reads the data
spawn_cache: dict = {}
spawn_run1 = run_pipeline(spawn_cache, "spawned")
spawn_run2 = run_pipeline(spawn_cache, "spawned")
print(f"one shared stream : run 1 {shared_run1:.4%} · run 2 {shared_run2:.4%}   ← same seed, different answer")
print(f"spawned streams   : run 1 {spawn_run1:.4%} · run 2 {spawn_run2:.4%}   ← identical")

assert shared_run1 != shared_run2, "a warm cache shifts a shared stream"
assert spawn_run1 == spawn_run2, "independent streams make the cache invisible to the bootstrap"
lab.record("boot_shared_run1", shared_run1)
lab.record("boot_shared_run2", shared_run2)
lab.record("boot_spawned", spawn_run1)
lab.record("boot_n", N_BOOT)

# %% [markdown]
# ## 5 · The hash-seed trap: iterating over a set
#
# Python salts the hash of every string with a random value chosen when the interpreter starts,
# so **the iteration order of a set of strings changes from one run to the next**. Assign random
# draws to tickers in set order and your seeded results change anyway. We start two fresh
# interpreters with different hash seeds and print the same set. The fix is `sorted(...)`.

# %%
TICKERS = {"SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "HYG"}
SNIPPET = "print(list({'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD', 'HYG'}))"


def set_order_in_fresh_interpreter(hash_seed: str) -> str:
    env_vars = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run([sys.executable, "-c", SNIPPET], env=env_vars, capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


order_1, order_2 = set_order_in_fresh_interpreter("1"), set_order_in_fresh_interpreter("2")
print("interpreter A:", order_1)
print("interpreter B:", order_2)
print("sorted, always:", sorted(TICKERS))
assert order_1 != order_2, "set iteration order depends on the per-process hash seed"
lab.record("set_order_differs", order_1 != order_2)

# %% [markdown]
# ## 6 · The data cache: build once, fingerprint, reuse
#
# Real labs download data once into `qn.data_cache()` (git-ignored) and reuse it. The pattern is
# the same for synthetic data, so we practise on a synthetic panel: 50 assets, 10 years of daily
# returns (indexed by elapsed trading day, not calendar date), with known volatilities from 15%
# to 45%.
#
# Three rules make a cache safe:
# 1. **Name the file by everything that decides its content**: the parameters, the seed and the
#    random stream that drew it, and the versions of the code that drew it (NumPy may change
#    `Generator` streams in feature releases). Change any of them and the name changes, so stale
#    data can never be silently reused.
# 2. **Fingerprint the content, not the file.** A manifest next to the data stores a hash of the
#    *values*; a cache hit is accepted only if the values still hash to it.
# 3. **Give the builder its own random stream** (§4), so a hit and a miss leave every later number
#    unchanged.

# %%
CACHE_DIR = qn.data_cache() / "u01s05"
panel_gen, = lab.rng.spawn(1)   # the builder's own stream (spawning never advances lab.rng)
PANEL_PARAMS = {
    # what to build
    "n_assets": 50, "years": 10, "mu": 0.07, "vol_lo": 0.15, "vol_hi": 0.45,
    # which random draws: the lab's seed and this stream's position in the spawn tree
    "seed": lab.seed, "spawn_key": list(panel_gen.bit_generator.seed_seq.spawn_key),
    # which code drew them
    "generator": "quantnb.synth.gbm_prices", "quantnb": qn.__version__, "numpy": np.__version__,
}


def params_key(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def true_vols(params: dict) -> np.ndarray:
    return np.linspace(params["vol_lo"], params["vol_hi"], params["n_assets"])


def build_panel(params: dict, gen: np.random.Generator) -> pd.DataFrame:
    """Long-format synthetic price panel: one row per (trading day, ticker)."""
    n_days = params["years"] * PERIODS_PER_YEAR
    day = np.arange(1, n_days + 1)          # simulated time is elapsed trading days, not dates
    frames = []
    for i, vol in enumerate(true_vols(params)):
        px = qn.synth.gbm_prices(n_days, mu=params["mu"], sigma=float(vol), rng=gen)["price"]
        frames.append(pd.DataFrame({"day": day, "ticker": f"SYN{i:02d}",
                                    "close": px.to_numpy()[1:], "ret": simple_returns(px).to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def content_fingerprint(df: pd.DataFrame) -> str:
    """Hash of the VALUES in canonical (ticker, day) order — independent of format, codec, row order.

    `qn.repro.content_fingerprint` sorts rows and columns, rounds floats to 12 significant digits
    (so harmless last-bit differences don't change it) and hashes the result with SHA-256.
    """
    return qn.repro.content_fingerprint(df.set_index(["ticker", "day"]))


def load_or_build(params: dict, gen: np.random.Generator) -> tuple[pd.DataFrame, str, Path]:
    """Read the panel from the cache if it is there and intact; otherwise build and write it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data_path = CACHE_DIR / f"panel-{params_key(params)}.parquet"
    meta_path = data_path.with_suffix(".json")
    if data_path.exists() and meta_path.exists():
        df = pd.read_parquet(data_path)
        if json.loads(meta_path.read_text(encoding="utf-8"))["fingerprint"] == content_fingerprint(df):
            return df, "hit", data_path
        print("cache entry failed its fingerprint check: rebuilding")
    df = build_panel(params, gen)
    df.to_parquet(data_path, index=False)
    meta = {"params": params, "fingerprint": content_fingerprint(df), "rows": len(df),
            "built_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df, "miss", data_path


t0 = time.perf_counter()
panel, status, PANEL_PATH = load_or_build(PANEL_PARAMS, panel_gen)
t_first = time.perf_counter() - t0
t0 = time.perf_counter()
panel_again, status_again, _ = load_or_build(PANEL_PARAMS, panel_gen)
t_second = time.perf_counter() - t0
print(f"first call: cache {status} in {t_first:.2f}s · second call: cache {status_again} in {t_second:.2f}s")
print(f"{PANEL_PATH.name}: {len(panel):,} rows, {PANEL_PATH.stat().st_size / 1e6:.1f} MB on disk")
print(f"content fingerprint: {content_fingerprint(panel)}")

assert status_again == "hit"
pd.testing.assert_frame_equal(panel.reset_index(drop=True), panel_again.reset_index(drop=True),
                              check_dtype=False)
# Rule 1, checked: a different seed, stream or parameter can never map to this file.
for change in ({"seed": lab.seed + 1}, {"spawn_key": [1]}, {"n_assets": 51}):
    assert params_key({**PANEL_PARAMS, **change}) != params_key(PANEL_PARAMS), change
lab.record("panel_rows", len(panel))
lab.record("panel_assets", PANEL_PARAMS["n_assets"])
lab.record("panel_years", PANEL_PARAMS["years"])
lab.record("panel_vol_lo", PANEL_PARAMS["vol_lo"])
lab.record("panel_vol_hi", PANEL_PARAMS["vol_hi"])

# %% [markdown]
# **Why fingerprint the content?** Write the same data with two compression codecs: the files'
# bytes differ (and they would also differ between two pyarrow versions, which stamp themselves
# into the file), but the values — and so the content fingerprint — are identical.

# %%
scratch = {codec: CACHE_DIR / f"codec-test-{codec}.parquet" for codec in ("snappy", "zstd")}
for codec, path in scratch.items():
    panel.to_parquet(path, index=False, compression=codec)
file_hashes = {c: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for c, p in scratch.items()}
content_hashes = {c: content_fingerprint(pd.read_parquet(p)) for c, p in scratch.items()}
print("file bytes  :", file_hashes)
print("content hash:", content_hashes)
assert file_hashes["snappy"] != file_hashes["zstd"]
assert content_hashes["snappy"] == content_hashes["zstd"] == content_fingerprint(panel)
lab.record("codec_file_hash_differs", file_hashes["snappy"] != file_hashes["zstd"])
lab.record("codec_content_hash_equal", content_hashes["snappy"] == content_hashes["zstd"])
for path in scratch.values():
    path.unlink()

# %% [markdown]
# ## 7 · Query it three ways: pandas, DuckDB SQL, a polars lazy frame
#
# The same question — annualised mean and volatility per asset — asked three ways. DuckDB runs
# SQL directly on the Parquet file; polars builds a lazy query plan and reads only the columns
# the plan needs. Unit 4 goes deep on this stack; here we only check that the engines agree.
#
# They agree to within about 1e-15, but **not necessarily bit for bit**: on this panel DuckDB's
# means differ from pandas' in the last digit or two (polars happens to match exactly), because
# engines may sum in a different order or with a different algorithm, and floating-point addition
# is not associative. That is why results are compared with a tolerance — here and in the
# course's CI.

# %%
ANN = PERIODS_PER_YEAR
TOL = 1e-12


def stats_pandas(path: Path) -> pd.DataFrame:
    g = pd.read_parquet(path, columns=["ticker", "ret"]).groupby("ticker")["ret"]
    return pd.DataFrame({"n_days": g.count(), "ann_mean": g.mean() * ANN,
                         "ann_vol": g.std() * math.sqrt(ANN)}).reset_index()


SQL_STATS = """
SELECT ticker,
       count(*)                        AS n_days,
       avg(ret) * 252                  AS ann_mean,
       stddev_samp(ret) * sqrt(252)    AS ann_vol
FROM read_parquet(?)
GROUP BY ticker
ORDER BY ticker
"""


def stats_duckdb(path: Path) -> pd.DataFrame:
    with duckdb.connect() as con:
        return con.execute(SQL_STATS, [path.as_posix()]).df()


def stats_polars(path: Path) -> pd.DataFrame:
    query = (pl.scan_parquet(path)                         # lazy: nothing is read yet
               .group_by("ticker")
               .agg(n_days=pl.len(),
                    ann_mean=pl.col("ret").mean() * ANN,
                    ann_vol=pl.col("ret").std() * math.sqrt(ANN))
               .sort("ticker"))
    return query.collect().to_pandas()


timings, results = {}, {}
for name, fn in (("pandas", stats_pandas), ("duckdb", stats_duckdb), ("polars", stats_polars)):
    t0 = time.perf_counter()
    results[name] = fn(PANEL_PATH).sort_values("ticker").reset_index(drop=True)
    timings[name] = time.perf_counter() - t0
print("file → answer, this machine (printed only):", {k: f"{v * 1000:.0f} ms" for k, v in timings.items()})
# The optimised plan: polars pushes the column selection into the scan ("PROJECT 2/4 COLUMNS").
print(pl.scan_parquet(PANEL_PATH).group_by("ticker").agg(pl.col("ret").std()).explain())

ref = results["pandas"]
max_diff = 0.0
for name in ("duckdb", "polars"):
    other = results[name]
    assert (other["ticker"].astype(str).to_numpy() == ref["ticker"].astype(str).to_numpy()).all()
    assert (other["n_days"].to_numpy() == ref["n_days"].to_numpy()).all()
    for col in ("ann_mean", "ann_vol"):
        gaps = np.abs(other[col].to_numpy() - ref[col].to_numpy())
        max_diff = max(max_diff, float(gaps.max()))
        assert gaps.max() < TOL, f"{name} disagrees with pandas on {col} by {gaps.max()}"
        print(f"{name} vs pandas, {col}: {int((gaps == 0).sum())}/{len(gaps)} bit-identical, "
              f"max gap {gaps.max():.1e}")   # printed only: engine internals vary by version
print(f"largest disagreement between engines: {max_diff:.1e} (tolerance {TOL:.0e})")
lab.record("engine_tol", TOL)
print(ref.head())

# %% [markdown]
# **Synthetic data lets you check the answer against the truth.** Volatility estimates land close
# to the truth; mean-return estimates, from the same ten years, miss by much more. The mean is the
# number you know least well — Session 2.3 returns to this.

# %%
vol_err = np.abs(ref["ann_vol"].to_numpy() - true_vols(PANEL_PARAMS))
mean_err = np.abs(ref["ann_mean"].to_numpy() - PANEL_PARAMS["mu"])
print(f"max |vol error| {vol_err.max():.2%} · max |mean error| {mean_err.max():.2%}")
assert vol_err.max() < 0.03, "ten years of daily data pin volatility down to a couple of points"
assert mean_err.max() > 5 * vol_err.max(), "…and leave the mean far less certain"
lab.record("vol_err_max_pp", float(vol_err.max() * 100))     # in percentage points
lab.record("mean_err_max_pp", float(mean_err.max() * 100))
lab.record("panel_mu", PANEL_PARAMS["mu"])

# %% [markdown]
# **Something SQL is good at.** The worst day for an equal-weighted portfolio of all 50 assets —
# group by trading day, average, sort. Checked against pandas.

# %%
SQL_WORST_DAY = """
SELECT day, avg(ret) AS ew_ret
FROM read_parquet(?)
GROUP BY day
ORDER BY ew_ret
LIMIT 1
"""
with duckdb.connect() as con:
    worst = con.execute(SQL_WORST_DAY, [PANEL_PATH.as_posix()]).df()
ew = panel.groupby("day")["ret"].mean()
assert int(worst["day"].iloc[0]) == int(ew.idxmin())
assert abs(worst["ew_ret"].iloc[0] - ew.min()) < TOL
print(f"worst equal-weight day: trading day {ew.idxmin()} (year {ew.idxmin() / PERIODS_PER_YEAR:.1f}), "
      f"{ew.min():.2%}")
lab.record("ew_worst_day", int(ew.idxmin()))
lab.record("ew_worst_ret", float(ew.min()))

# %% [markdown]
# **Floating-point addition is not associative.** Add the same million returns forwards and then
# backwards: two different answers. Neither is wrong; both differ from the exact sum in the last
# few digits. Exact equality is the wrong test for floats — compare with a tolerance.

# %%
x = ((np.random.default_rng(SEED).random(1_000_000) - 0.5) * 0.02).tolist()
fwd = 0.0
for v in x:
    fwd += v
bwd = 0.0
for v in reversed(x):
    bwd += v
print(f"forwards {fwd!r}\nbackwards {bwd!r}\nexact     {math.fsum(x)!r}")
assert fwd != bwd and abs(fwd - bwd) < 1e-10
lab.record("sum_fwd_minus_bwd", fwd - bwd)

# %% [markdown]
# ## 8 · The run manifest
#
# Everything needed to reproduce a result, in one small dictionary: the code version, the
# environment, the data fingerprint and the seed. Paste it into your research log (Session 1.3)
# with every result; experiment trackers such as MLflow (Session 6.8) record it for you.

# %%
def git_state(root: Path | None) -> str:
    """Short commit hash, marked '+dirty' when there are uncommitted changes."""
    if root is None:
        return "unknown (not in the repo)"
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


manifest = {
    "session": lab.session,
    "code": git_state(qn.repo_root()),
    "environment": {"python": env["python"],
                    **{p: v[0] for p, v in env["packages"].items() if p in ("numpy", "pandas", "duckdb", "polars")}},
    "data": {"file": PANEL_PATH.name, "fingerprint": content_fingerprint(panel)},
    "randomness": {"seed": lab.seed, "streams": "spawned per component"},
}
print(json.dumps(manifest, indent=2))
lab.record("lab_seed", lab.seed)

# %% [markdown]
# ## 9 · Chart for the session page

# %%
lab.chart("seed_sharpe", charts.column_chart(
    [str(s) for s in SEEDS], seed_sr.tolist(),
    title=f"Sharpe ratio of one strategy under {len(SEEDS)} seeds (true Sharpe {true_sr:.2f}, {YEARS} years)",
    y_fmt=charts.fmt_num(1), value_labels=False,
    roles=["alt1" if i == best else "strategy" for i in range(len(SEEDS))]))

# %%
lab.save()
