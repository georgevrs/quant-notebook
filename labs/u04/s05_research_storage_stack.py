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
# # 4.5 · The Research Storage Stack — companion lab
#
# **Quant Notebook** · Unit 4 · Session 5 · [Read the session](https://georgevrs.github.io/quant-notebook/unit04-data-engineering/session05-research-storage-stack.html)
#
# Parquet, Arrow, DuckDB and polars — a fast, local, columnar research stack — plus when kdb+ and
# managed warehouses earn their cost.
#
# We build one synthetic cross-sectional panel (prices, volume and a few microstructure columns for
# `N_SYMBOLS` symbols over `N_YEARS` years, tagged with a sector), write it to disk three ways —
# a single CSV, a single Parquet file, and a Hive-partitioned Parquet dataset — and then run the
# *same* cross-sectional query against all three with pandas, DuckDB and polars. Every number the
# session page quotes is recorded with `lab.record(...)` and saved to `out/` by the last cell, so
# the page and this notebook can never disagree.
#
# **Benchmark caveat, stated once and honestly:** these are single-machine, single-run wall-clock
# times on whatever hardware runs this notebook (disk cache, core count and background load all
# move the absolute numbers). They are not a formal benchmark suite. What is robust across
# machines — and what the lab actually asserts — is the *shape* of the result: columnar beats
# row-based, and partition + predicate pushdown beats a full columnar scan, by wide, repeatable
# margins. The first query a process ever sends to DuckDB or polars also pays a one-off
# thread-pool warm-up of a second or so; we report the steady-state time after that warm-up, which
# is the number that matters once a research process is actually running.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR
from quantnb.synth import factor_model_returns

lab = qn.Lab("4.5")  # seeds the random generators: every run gives the same numbers

# One RNG stream per experiment (AUTHORING.md §12a): the return-generating process and the
# liquidity/microstructure fields are independent draws, so changing one later cannot silently
# change the other.
rng_returns, rng_liquidity = lab.rng.spawn(2)

# %% [markdown]
# ## 1 · Build a synthetic cross-sectional panel
#
# `N_SYMBOLS` symbols, `N_YEARS` years of daily bars, tagged with one of `N_SECTORS` sectors.
# `factor_model_returns` (Session 3.3's factor model) generates all symbols' daily returns in one
# vectorized call — the only way to reach millions of rows in seconds. On top of the resulting
# prices we synthesise a few extra columns (open/high/low/vwap/trade count/bid/ask) that a real
# bar store would have and this query never touches — that is what makes column pruning matter
# below, not just partition pruning.

# %%
N_SYMBOLS = 1_500
N_YEARS = 5
N_DAYS = N_YEARS * PERIODS_PER_YEAR      # 1,260 trading days
N_SECTORS = 10                            # SEC00 … SEC09, symbols assigned round-robin
QUERY_SECTOR = "SEC03"


def make_synthetic_panel(n_days: int, n_symbols: int, n_sectors: int,
                          rng_returns: np.random.Generator, rng_liquidity: np.random.Generator) -> pd.DataFrame:
    """A long-format synthetic OHLCV + microstructure panel: one row per (date, symbol).

    Prices come from a known factor model (Session 3.3), so any query result can be sanity-checked
    against a hand-computable truth. Volume, spread and the OHLC spread around the close are
    cosmetic — realistic enough to fill out a schema, not a claim about real market microstructure.
    """
    fm = factor_model_returns(n_days, n_symbols, n_factors=4, rng=rng_returns)
    close = 100.0 * (1.0 + fm["returns"]).cumprod()          # T x N price levels
    dates = close.index.to_numpy()
    symbols = np.array([f"SYM{i:04d}" for i in range(n_symbols)])
    sector_of_symbol = np.array([f"SEC{i % n_sectors:02d}" for i in range(n_symbols)])

    c = close.to_numpy()
    noise = lambda scale: rng_liquidity.normal(0, scale, size=c.shape)  # noqa: E731
    open_ = c * (1 + noise(0.003))
    high = np.maximum(open_, c) * (1 + np.abs(noise(0.002)))
    low = np.minimum(open_, c) * (1 - np.abs(noise(0.002)))
    vwap = (open_ + c) / 2 * (1 + noise(0.001))
    half_spread = 0.0005
    base_liquidity = rng_liquidity.lognormal(mean=12.5, sigma=1.0, size=n_symbols)       # shares/day
    daily_mult = rng_liquidity.lognormal(mean=0.0, sigma=0.35, size=c.shape)
    volume = (base_liquidity[None, :] * daily_mult).astype(np.int64)
    trade_count = (volume / rng_liquidity.uniform(80, 200, size=c.shape)).astype(np.int64) + 1

    dates_rep = np.repeat(dates, n_symbols)
    return pd.DataFrame({
        "date": dates_rep,
        "year": pd.DatetimeIndex(dates_rep).year.to_numpy().astype(np.int16),
        "sector": pd.Categorical(np.tile(sector_of_symbol, n_days)),
        "symbol": pd.Categorical(np.tile(symbols, n_days)),
        "open": open_.reshape(-1).astype(np.float32),
        "high": high.reshape(-1).astype(np.float32),
        "low": low.reshape(-1).astype(np.float32),
        "close": c.reshape(-1).astype(np.float32),
        "vwap": vwap.reshape(-1).astype(np.float32),
        "volume": volume.reshape(-1),
        "trade_count": trade_count.reshape(-1),
        "bid": (c * (1 - half_spread)).reshape(-1).astype(np.float32),
        "ask": (c * (1 + half_spread)).reshape(-1).astype(np.float32),
    })


t0 = time.perf_counter()
panel = make_synthetic_panel(N_DAYS, N_SYMBOLS, N_SECTORS, rng_returns, rng_liquidity)
t_build = time.perf_counter() - t0
n_rows, n_cols = panel.shape
symbols_per_sector = N_SYMBOLS // N_SECTORS
assert N_SYMBOLS % N_SECTORS == 0, "keep symbols divisible by sectors so every partition is the same size"
print(f"built {n_rows:,} rows x {n_cols} columns in {t_build:.2f}s "
      f"({panel.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory)")

lab.record("n_symbols", N_SYMBOLS)
lab.record("n_years", N_YEARS)
lab.record("n_days", N_DAYS)
lab.record("n_sectors", N_SECTORS)
lab.record("n_rows", n_rows)
lab.record("n_cols", n_cols)
lab.record("symbols_per_sector", symbols_per_sector)

# %% [markdown]
# ## 2 · Write it three ways: CSV, one Parquet file, partitioned Parquet
#
# Same data, three physical layouts: a row-oriented text format, a columnar file with no
# partitioning, and a **Hive-partitioned** Parquet dataset (`year=…/sector=…/part.parquet`) —
# `pandas.DataFrame.to_parquet(partition_cols=...)` writes exactly this directory layout.

# %%
tmpdir = tempfile.TemporaryDirectory(prefix="qn45_bench_")   # self-cleaning; this is a benchmark, not a cache
TMP = Path(tmpdir.name)
csv_path = TMP / "panel.csv"
mono_parquet_path = TMP / "panel_single_file.parquet"
partitioned_dir = TMP / "panel_partitioned"

t0 = time.perf_counter()
panel.to_csv(csv_path, index=False, lineterminator="\n")  # LF only: deterministic byte count cross-platform
t_write_csv = time.perf_counter() - t0
csv_bytes = csv_path.stat().st_size

t0 = time.perf_counter()
panel.to_parquet(mono_parquet_path, index=False, compression="snappy")
t_write_mono = time.perf_counter() - t0
mono_bytes = mono_parquet_path.stat().st_size

t0 = time.perf_counter()
# partition_cols wants plain strings, not pandas Categoricals, for the directory names
panel.assign(sector=panel["sector"].astype(str)).to_parquet(
    partitioned_dir, index=False, partition_cols=["year", "sector"], compression="snappy")
t_write_part = time.perf_counter() - t0
part_files = sorted(partitioned_dir.rglob("*.parquet"))
part_bytes = sum(f.stat().st_size for f in part_files)

print(f"CSV          {csv_bytes / 1e6:6.1f} MB  write {t_write_csv:5.2f}s")
print(f"Parquet 1-file {mono_bytes / 1e6:6.1f} MB  write {t_write_mono:5.2f}s")
print(f"Parquet part.  {part_bytes / 1e6:6.1f} MB  write {t_write_part:5.2f}s  ({len(part_files)} files)")

n_partitions = N_YEARS * N_SECTORS
assert len(part_files) == n_partitions, "one file per (year, sector) partition"
assert mono_bytes < csv_bytes * 0.6, "columnar + compression should shrink the file well below 60% of the CSV"

lab.record("csv_mb", csv_bytes / 1e6)
lab.record("mono_parquet_mb", mono_bytes / 1e6)
lab.record("partitioned_parquet_mb", part_bytes / 1e6)
lab.record("compression_ratio", csv_bytes / mono_bytes)
lab.record("n_partition_files", len(part_files))
lab.record("t_write_csv", t_write_csv)
lab.record("t_write_mono_parquet", t_write_mono)
lab.record("t_write_partitioned_parquet", t_write_part)

# %% [markdown]
# ## 3 · The same query, four ways
#
# A realistic cross-sectional panel scan: *for one sector, in one year, the average close and
# total dollar volume of every symbol.* It only needs 5 of the panel's 13 columns
# (`year`, `sector`, `symbol`, `close`, `volume`) and only 1 of the 50 partitions.
#
# `best_of` runs a callable a few times and keeps the minimum — the standard way to report a
# timing without the noise of the first call's cache misses and (for DuckDB/polars) thread-pool
# start-up dominating the number.

# %%
def best_of(fn, reps: int):
    times, result = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, min(times)


QUERY_YEAR = 2012   # the 3rd of the N_YEARS simulated years, given the default 2010-01-04 start
assert QUERY_YEAR in panel["year"].unique(), "QUERY_YEAR must be one of the simulated years"


def naive_pandas_csv():
    """The naive way: read the whole file, filter in memory, then aggregate."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    sub = df[(df["year"] == QUERY_YEAR) & (df["sector"] == QUERY_SECTOR)].copy()
    sub["dollar_volume"] = sub["close"] * sub["volume"]
    return sub.groupby("symbol").agg(avg_close=("close", "mean"), dollar_volume=("dollar_volume", "sum"))


def naive_pandas_parquet():
    """A columnar FILE, but still read in full: partitioning, not the format, is what is missing."""
    df = pd.read_parquet(mono_parquet_path)
    sub = df[(df["year"] == QUERY_YEAR) & (df["sector"].astype(str) == QUERY_SECTOR)].copy()
    sub["dollar_volume"] = sub["close"] * sub["volume"]
    return sub.groupby("symbol", observed=True).agg(avg_close=("close", "mean"), dollar_volume=("dollar_volume", "sum"))


con = duckdb.connect()


def duckdb_partitioned():
    """DuckDB reads the Hive layout directly: partition pruning, then column pruning, then a scan."""
    query = f"""
        SELECT symbol, avg(close) AS avg_close, sum(close * volume) AS dollar_volume
        FROM read_parquet('{partitioned_dir.as_posix()}/**/*.parquet', hive_partitioning = true)
        WHERE year = {QUERY_YEAR} AND sector = '{QUERY_SECTOR}'
        GROUP BY symbol
    """
    return con.execute(query).df().set_index("symbol")


def polars_lazy_partitioned():
    """polars' LazyFrame builds a query plan; nothing runs until .collect() — see the assert below."""
    lf = pl.scan_parquet(partitioned_dir, hive_partitioning=True)
    assert isinstance(lf, pl.LazyFrame), "scan_parquet must be lazy: no I/O until .collect()"
    plan = (lf.filter((pl.col("year") == QUERY_YEAR) & (pl.col("sector") == QUERY_SECTOR))
              .group_by("symbol")
              .agg([pl.col("close").mean().alias("avg_close"),
                    (pl.col("close") * pl.col("volume")).sum().alias("dollar_volume")]))
    return plan.collect().to_pandas().set_index("symbol")


agg_csv, t_naive_csv = best_of(naive_pandas_csv, reps=2)
agg_mono, t_naive_mono = best_of(naive_pandas_parquet, reps=2)
agg_duck, t_duckdb = best_of(duckdb_partitioned, reps=3)
agg_pl, t_polars = best_of(polars_lazy_partitioned, reps=3)

for name, secs in (("pandas + CSV (full scan)", t_naive_csv), ("pandas + 1 Parquet file (full scan)", t_naive_mono),
                   ("DuckDB + partitioned Parquet", t_duckdb), ("polars + partitioned Parquet", t_polars)):
    print(f"{name:38s} {secs * 1000:8.1f} ms")

# %% [markdown]
# **Assert what we teach: every engine agrees, and the disciplined ones are dramatically faster.**
# We check agreement against the naive CSV result (nothing in its path can accidentally benefit
# from partitioning) and require the partitioned engines to beat it by at least 5x and the
# columnar-but-unpartitioned file by at least 3x — thresholds chosen with a wide safety margin
# below what we actually measure (often 100-300x), so the assert is about the shape of the result,
# not a specific machine's exact numbers.

# %%
for other, label in ((agg_mono, "1-file Parquet"), (agg_duck, "DuckDB"), (agg_pl, "polars")):
    aligned = other.loc[agg_csv.index]
    assert np.allclose(agg_csv["avg_close"], aligned["avg_close"], rtol=1e-2), f"{label} avg_close disagrees"
    assert np.allclose(agg_csv["dollar_volume"], aligned["dollar_volume"], rtol=1e-2), f"{label} dollar_volume disagrees"
assert len(agg_csv) == symbols_per_sector, "one row per symbol in the queried sector"

assert t_naive_mono < t_naive_csv * 0.5, "a columnar file should at least halve the naive-CSV time"
assert t_duckdb < t_naive_csv / 5, "partition + predicate pushdown should beat naive CSV by 5x+"
assert t_polars < t_naive_csv / 5, "partition + predicate pushdown should beat naive CSV by 5x+"
assert t_duckdb < t_naive_mono / 3, "partitioning should beat a columnar full scan by 3x+ on top"
assert t_polars < t_naive_mono / 3, "partitioning should beat a columnar full scan by 3x+ on top"

lab.record("query_year", QUERY_YEAR)
lab.record("t_naive_csv_ms", t_naive_csv * 1000)
lab.record("t_naive_parquet_ms", t_naive_mono * 1000)
lab.record("t_duckdb_ms", t_duckdb * 1000)
lab.record("t_polars_ms", t_polars * 1000)
lab.record("speedup_duckdb_vs_csv", t_naive_csv / t_duckdb)
lab.record("speedup_polars_vs_csv", t_naive_csv / t_polars)
lab.record("speedup_duckdb_vs_mono", t_naive_mono / t_duckdb)
lab.record("speedup_polars_vs_mono", t_naive_mono / t_polars)
lab.record("n_symbols_in_result", len(agg_csv))

# %% [markdown]
# ## 4 · Quantify the pruning
#
# Partition pruning skips whole files before opening them; column pruning skips columns within the
# files it does open. Multiplying the two fractions is only a back-of-envelope estimate — Parquet
# compresses columns differently, so it is not exact — but it is the right order of magnitude for
# "how much less did the disciplined query touch?"

# %%
COLUMNS_NEEDED = {"year", "sector", "symbol", "close", "volume"}   # what the query actually reads
frac_partitions = 1 / n_partitions
frac_columns = len(COLUMNS_NEEDED) / n_cols
frac_estimated_scanned = frac_partitions * frac_columns

lab.record("n_partitions_total", n_partitions)
lab.record("n_partitions_touched", 1)
lab.record("frac_partitions_touched", frac_partitions)
lab.record("n_columns_total", n_cols)
lab.record("n_columns_touched", len(COLUMNS_NEEDED))
lab.record("frac_columns_touched", frac_columns)
lab.record("frac_estimated_scanned", frac_estimated_scanned)
print(f"touches {frac_partitions:.1%} of partitions x {frac_columns:.1%} of columns "
      f"≈ {frac_estimated_scanned:.2%} of the bytes, in principle")

# %% [markdown]
# ## 5 · The other failure mode: too many small files
#
# Partitioning helps until it doesn't. Take the exact slice this query wants (one year, one
# sector) and write it two ways: as the one file it already is inside `partitioned_dir`, and
# re-partitioned *again* by symbol — turning `symbols_per_sector` rows-worth of data into that many
# tiny files. Reading "the same bytes" now means opening every one of them.

# %%
year_sector_slice = panel[(panel["year"] == QUERY_YEAR) & (panel["sector"].astype(str) == QUERY_SECTOR)]
one_file = TMP / "slice_one_file.parquet"
many_files_dir = TMP / "slice_many_tiny_files"
year_sector_slice.to_parquet(one_file, index=False, compression="snappy")
year_sector_slice.assign(symbol=year_sector_slice["symbol"].astype(str)).to_parquet(
    many_files_dir, index=False, partition_cols=["symbol"], compression="snappy")
n_tiny_files = len(list(many_files_dir.rglob("*.parquet")))


def read_one_file():
    return con.execute(f"SELECT symbol, avg(close) FROM read_parquet('{one_file.as_posix()}') GROUP BY symbol").df()


def read_many_files():
    return con.execute(
        f"SELECT symbol, avg(close) FROM read_parquet('{many_files_dir.as_posix()}/**/*.parquet', "
        f"hive_partitioning = true) GROUP BY symbol").df()


_, t_read_one = best_of(read_one_file, reps=3)
_, t_read_many = best_of(read_many_files, reps=3)
print(f"same data: 1 file {t_read_one * 1000:.1f} ms vs {n_tiny_files} files {t_read_many * 1000:.1f} ms")

assert n_tiny_files == symbols_per_sector
assert t_read_many > t_read_one * 1.3, "many tiny files should be measurably slower to read than one file"

lab.record("n_tiny_files", n_tiny_files)
lab.record("t_read_one_file_ms", t_read_one * 1000)
lab.record("t_read_many_files_ms", t_read_many * 1000)
lab.record("slowdown_many_files", t_read_many / t_read_one)

con.close()
tmpdir.cleanup()

# %% [markdown]
# ## 6 · Charts for the session page

# %%
lab.chart("file_size", charts.column_chart(
    ["CSV", "Parquet (1 file)", "Parquet (partitioned)"],
    [csv_bytes / 1e6, mono_bytes / 1e6, part_bytes / 1e6],
    title=f"Size on disk for the same {n_rows:,} rows, three formats",
    y_fmt=charts.fmt_num(0, suffix=" MB"),
    roles=["loss", "alt1", "strategy"]))

lab.chart("small_files", charts.column_chart(
    ["1 file", f"{n_tiny_files} tiny files"],
    [t_read_one * 1000, t_read_many * 1000],
    title="Reading the identical slice as one file vs one file per symbol",
    y_fmt=charts.fmt_num(1, suffix=" ms"),
    roles=["strategy", "loss"]))

# %%
lab.save()
