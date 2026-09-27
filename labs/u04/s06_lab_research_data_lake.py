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
# # 4.6 · Lab: Build Your Research Data Lake — companion lab
#
# **Quant Notebook** · Unit 4 · Session 6 · [Read the session](https://georgevrs.github.io/quant-notebook/unit04-data-engineering/session06-lab-research-data-lake.html)
#
# Build an idempotent, tested, point-in-time data pipeline with quality checks and lineage — the foundation for every later lab.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# This is Unit 4's capstone: it pulls together corporate-action adjustment (4.3), as-of joins and
# bars (4.4) and the Parquet/DuckDB/polars storage stack (4.5) into one small, real, runnable
# research data lake — three synthetic symbols, a stock split, a ticker rename, a quarterly index
# review, quality gates, a lineage manifest and an idempotent build. Every byte is synthetic and
# every file lives in the git-ignored cache; nothing here is downloaded or committed.

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
import shutil

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_ds

import quantnb as qn
from quantnb.repro import content_fingerprint
from quantnb.returns import in_years, max_drawdown, sharpe_ratio, simple_returns

lab = qn.Lab("4.6")  # seeds the random generators: every run gives the same numbers
rng = lab.rng

# One RNG stream per experiment, so changing one section never changes another's numbers.
rng_prices, rng_volume, rng_actions, rng_quality = rng.spawn(4)

# %% [markdown]
# ## 1 · The raw landing zone — three synthetic "vendor feeds"
#
# Three symbols, three years of daily prices, each a geometric Brownian motion with its own drift
# and volatility (`quantnb.synth.gbm_prices`, one spawned RNG stream per symbol). Daily volume is
# correlated with the size of the day's move — busy days trade more — which later makes dollar
# bars genuinely adapt to activity instead of just resampling calendar time.

# %%
SYMBOLS = ["ALPHA", "BETA", "GAMMA"]
SIM_YEARS = 3
N_DAYS = 252 * SIM_YEARS
MU = {"ALPHA": 0.08, "BETA": 0.11, "GAMMA": 0.06}
SIGMA = {"ALPHA": 0.20, "BETA": 0.28, "GAMMA": 0.16}
BASE_SHARE_VOLUME = 200_000

price_rngs = dict(zip(SYMBOLS, rng_prices.spawn(len(SYMBOLS))))
volume_rngs = dict(zip(SYMBOLS, rng_volume.spawn(len(SYMBOLS))))

raw: dict[str, pd.DataFrame] = {}
for sym in SYMBOLS:
    px = qn.synth.gbm_prices(N_DAYS, mu=MU[sym], sigma=SIGMA[sym], rng=price_rngs[sym])["price"]
    day_move = px.pct_change().fillna(0.0).abs()
    noise = volume_rngs[sym].lognormal(mean=0.0, sigma=0.35, size=len(px))
    volume = BASE_SHARE_VOLUME * (1.0 + 8.0 * day_move) * noise
    raw[sym] = pd.DataFrame({"raw_price": px.to_numpy(), "volume": volume}, index=px.index)

lab.record("n_symbols", len(SYMBOLS))
lab.record("n_days", N_DAYS)
lab.record("sim_years", SIM_YEARS)

# %% [markdown]
# ## 2 · A corporate action lands in the raw feed
#
# ALPHA does a 2-for-1 split partway through the window: the vendor feed simply halves the price
# and doubles the share volume from the ex-date on, exactly as a real feed would. Left alone, this
# is a fake -50% one-day return sitting in the data. The **adjustment factor** is 1.0 after the
# split and the split ratio before it; multiplying pre-split raw prices by that factor is a
# **ratio adjustment**, and applying it across the whole history is a **back-adjustment** — the
# general pipeline for this is <a data-xref="4.3"></a>'s job; here we apply the same idea end to end.

# %%
SPLIT_SYMBOL = "ALPHA"
SPLIT_DAY = 500          # trading-day index within ALPHA's window
SPLIT_RATIO = 2.0        # 2-for-1


def adjustment_factor(n: int, event_day: int, ratio: float) -> np.ndarray:
    """Cumulative back-adjustment factor for ONE corporate action: 1/`ratio` before the event,
    1.0 from the event day on, so the whole series reads in TODAY's share terms. A 2-for-1 split
    (ratio=2) halved the raw price on the ex-date, so pre-event prices must also be halved."""
    factor = np.ones(n)
    factor[:event_day] = 1.0 / ratio
    return factor


def back_adjust(raw_price: pd.Series, factor: np.ndarray) -> pd.Series:
    return raw_price * factor


split_frame = raw[SPLIT_SYMBOL].copy()
split_frame.iloc[SPLIT_DAY:, split_frame.columns.get_loc("raw_price")] /= SPLIT_RATIO
split_frame.iloc[SPLIT_DAY:, split_frame.columns.get_loc("volume")] *= SPLIT_RATIO
raw[SPLIT_SYMBOL] = split_frame

adj_factor = {sym: np.ones(len(raw[sym])) for sym in SYMBOLS}
adj_factor[SPLIT_SYMBOL] = adjustment_factor(len(raw[SPLIT_SYMBOL]), SPLIT_DAY, SPLIT_RATIO)
adjusted_price = {
    sym: back_adjust(raw[sym]["raw_price"], adj_factor[sym]) for sym in SYMBOLS
}

raw_cliff = float(raw[SPLIT_SYMBOL]["raw_price"].iloc[SPLIT_DAY] / raw[SPLIT_SYMBOL]["raw_price"].iloc[SPLIT_DAY - 1] - 1)
adj_cliff = float(adjusted_price[SPLIT_SYMBOL].iloc[SPLIT_DAY] / adjusted_price[SPLIT_SYMBOL].iloc[SPLIT_DAY - 1] - 1)
lab.record("raw_split_return", raw_cliff)
lab.record("adj_split_return", adj_cliff)
assert raw_cliff < -0.35, "the injected 2-for-1 split must show up as a large fake drop in the raw feed"
assert abs(adj_cliff) < 0.15, "back-adjustment must remove the split cliff, leaving an ordinary day's move"

# %% [markdown]
# ## 3 · Stable identifiers through a ticker rename
#
# BETA is renamed BETA2 partway through the window — same company, new ticker. A pipeline that
# joins on the ticker string silently splits one security's history into two. A **security
# master** maps every ticker string to a permanent **security identifier**, so joins downstream
# use the identifier, never the ticker.

# %%
RENAME_DAY = 300
SEC_ID = {"ALPHA": "SEC001", "BETA": "SEC002", "GAMMA": "SEC003"}


def curated_frame(sym: str) -> pd.DataFrame:
    out = raw[sym].reset_index(names="date")
    out["adj_price"] = adjusted_price[sym].to_numpy()
    out["adj_factor"] = adj_factor[sym]
    out["security_id"] = SEC_ID[sym]
    if sym == "BETA":
        out["symbol"] = np.where(np.arange(len(out)) < RENAME_DAY, "BETA", "BETA2")
    else:
        out["symbol"] = sym
    return out[["date", "symbol", "security_id", "adj_price", "adj_factor", "volume"]]


curated_no_membership = pd.concat([curated_frame(s) for s in SYMBOLS], ignore_index=True)
curated_no_membership = curated_no_membership.sort_values("date").reset_index(drop=True)
lab.record("n_distinct_tickers", curated_no_membership["symbol"].nunique())
lab.record("n_distinct_security_ids", curated_no_membership["security_id"].nunique())
assert curated_no_membership["symbol"].nunique() == 4       # ALPHA, BETA, BETA2, GAMMA
assert curated_no_membership["security_id"].nunique() == 3  # but only three real companies

# %% [markdown]
# ## 4 · An as-of join that never peeks
#
# A quarterly index-committee review sets each security's index membership. Joining "the most
# recent review as of this date" is an **as-of join**: `direction="backward"` matches the last
# review on or before the row's date and never a later one. `direction="nearest"` looks like a
# harmless shortcut but sometimes matches a review that has not happened yet — a look-ahead leak.

# %%
review_dates = pd.bdate_range(start=curated_no_membership["date"].min(), periods=SIM_YEARS * 4, freq="63B")
sec_ids = sorted(set(SEC_ID.values()))
reviews = pd.DataFrame({
    "review_date": np.tile(review_dates.to_numpy(), len(sec_ids)),
    "security_id": np.repeat(sec_ids, len(review_dates)),
    "in_index": rng_actions.integers(0, 2, len(review_dates) * len(sec_ids)).astype(bool),
}).sort_values("review_date").reset_index(drop=True)

backward = pd.merge_asof(curated_no_membership, reviews, left_on="date", right_on="review_date",
                         by="security_id", direction="backward")
naive_nearest = pd.merge_asof(curated_no_membership, reviews, left_on="date", right_on="review_date",
                              by="security_id", direction="nearest")

lookahead = naive_nearest["review_date"] > naive_nearest["date"]
wrong_membership = lookahead & (naive_nearest["in_index"] != backward["in_index"])
lab.record("naive_join_lookahead_rows", int(lookahead.sum()))
lab.record("naive_join_wrong_rows", int(wrong_membership.sum()))
assert lookahead.sum() > 0, "on this panel a nearest-match join must peek at a future review at least once"

curated = backward.drop(columns=["review_date"]).rename(columns={"in_index": "in_index"})
lab.record("n_reviews", len(reviews))

# %% [markdown]
# ## 5 · A data contract and its quality checks
#
# A **data contract** is the schema and the rules a curated table promises to satisfy — the columns
# that must exist, which ones may never be null, and which values must make sense — so every later
# lab can query the lake without re-verifying it. A **data quality check** tests the contract
# against real data and reports every violation it finds, not just the first.

# %%
CONTRACT_COLUMNS = ("date", "symbol", "security_id", "adj_price", "adj_factor", "volume")


def check_quality(df: pd.DataFrame) -> list[str]:
    """Validate `df` against the curated-bars data contract. Returns one message per violation."""
    problems = []
    missing = set(CONTRACT_COLUMNS) - set(df.columns)
    if missing:
        return [f"missing columns: {sorted(missing)}"]
    if df["date"].isna().any() or df["symbol"].isna().any():
        problems.append("null key column (date or symbol)")
    if df.duplicated(["date", "symbol"]).any():
        problems.append("duplicate (date, symbol) rows")
    if not df["date"].is_monotonic_increasing:
        problems.append("timestamps not sorted ascending")
    if (df["adj_price"] <= 0).any():
        problems.append("non-positive price")
    if (df["volume"] < 0).any():
        problems.append("negative volume")
    if (df["adj_factor"] <= 0).any():
        problems.append("non-positive adjustment factor")
    return problems


clean_violations = check_quality(curated)
lab.record("clean_violations", len(clean_violations))
assert clean_violations == [], f"the curated table must satisfy its own contract: {clean_violations}"

# Inject four distinct, known violations into a COPY and check the contract catches all of them.
corrupted = curated.copy()
bad_rows = rng_quality.choice(len(corrupted), size=4, replace=False)
corrupted.iloc[bad_rows[0], corrupted.columns.get_loc("adj_price")] = -1.0
corrupted.iloc[bad_rows[1], corrupted.columns.get_loc("volume")] = -100.0
corrupted.iloc[bad_rows[2], corrupted.columns.get_loc("symbol")] = None
dup_row = corrupted.iloc[[bad_rows[3]]]
corrupted = pd.concat([corrupted, dup_row], ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)

caught = check_quality(corrupted)
lab.record("injected_violations", 4)
lab.record("caught_violations", len(caught))
assert len(caught) == 4, f"expected to catch all 4 injected violations, caught {caught}"

# %% [markdown]
# ## 6 · Bars: sampling by information, not by the clock
#
# Time bars (one row per day) sample the clock. **Dollar bars** and **volume bars** instead close a
# new bar every time cumulative traded value (or shares) crosses a threshold, so quiet stretches
# produce fewer, longer bars and busy stretches produce more, shorter ones (López de Prado, 2018,
# ch. 2). Sizing the threshold from the data's own total keeps the bar count comparable across runs.

# %%
def sample_bars(df: pd.DataFrame, metric: pd.Series, threshold: float) -> pd.DataFrame:
    """Close a bar every time `metric`'s running total crosses a multiple of `threshold`."""
    bucket = (metric.cumsum() // threshold).astype(int)
    bars = df.groupby(bucket).agg(
        date=("date", "last"), open=("adj_price", "first"), high=("adj_price", "max"),
        low=("adj_price", "min"), close=("adj_price", "last"), volume=("volume", "sum"),
    ).reset_index(drop=True)
    return bars


alpha_bars_source = curated[curated["symbol"] == "ALPHA"].reset_index(drop=True)
TARGET_BARS = 500
dollar_metric = alpha_bars_source["adj_price"] * alpha_bars_source["volume"]
volume_metric = alpha_bars_source["volume"]
dollar_bars = sample_bars(alpha_bars_source, dollar_metric, dollar_metric.sum() / TARGET_BARS)
volume_bars = sample_bars(alpha_bars_source, volume_metric, volume_metric.sum() / TARGET_BARS)

for name, bars in (("dollar", dollar_bars), ("volume", volume_bars)):
    assert (bars["high"] >= bars[["open", "close", "low"]].max(axis=1)).all(), f"{name} bars: high must dominate"
    assert (bars["low"] <= bars[["open", "close", "high"]].min(axis=1)).all(), f"{name} bars: low must be dominated"

lab.record("n_time_bars", len(alpha_bars_source))
lab.record("n_dollar_bars", len(dollar_bars))
lab.record("n_volume_bars", len(volume_bars))

# how much a dollar bar's calendar span shrinks near the split, when trading is busiest
near_split = dollar_bars["date"].between(
    alpha_bars_source["date"].iloc[max(SPLIT_DAY - 40, 0)], alpha_bars_source["date"].iloc[min(SPLIT_DAY + 40, N_DAYS - 1)])
quiet = ~near_split
def mean_days_between(dates: pd.Series) -> float:
    gaps = pd.to_datetime(dates).diff().dropna()
    return float(gaps.dt.total_seconds().mean() / 86_400) if len(gaps) else float("nan")


days_per_bar_busy = mean_days_between(dollar_bars.loc[near_split, "date"]) if near_split.sum() > 1 else float("nan")
days_per_bar_quiet = mean_days_between(dollar_bars.loc[quiet, "date"]) if quiet.sum() > 1 else float("nan")
lab.record("days_per_dollar_bar_busy", days_per_bar_busy)
lab.record("days_per_dollar_bar_quiet", days_per_bar_quiet)

# %% [markdown]
# ## 7 · Writing the lake: partitioned Parquet, an idempotent build, a lineage manifest
#
# The curated table is written once as **Parquet** partitioned by symbol (Hive-style
# `symbol=ALPHA/…`), the layout DuckDB and polars both read natively. An **idempotent pipeline**
# run twice with the same inputs and the same seed either produces byte-identical output or,
# better, detects that nothing changed and skips the rewrite. **Data lineage** is the manifest that
# records, for every build, which inputs (fingerprinted) and which pipeline version produced it.

# %%
LAKE_DIR = qn.data_cache() / "s4_6_research_lake"
shutil.rmtree(LAKE_DIR, ignore_errors=True)
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = LAKE_DIR / "curated"
MANIFEST_PATH = LAKE_DIR / "_lineage.json"
PIPELINE_VERSION = "1.0"


def pipeline_fingerprint(params: dict, seed: int) -> str:
    payload = json.dumps({**params, "seed": seed, "version": PIPELINE_VERSION}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_pipeline(df: pd.DataFrame, params: dict, seed: int) -> dict:
    """Build (or skip) the curated Parquet dataset. Idempotent: same params + seed + data -> same
    fingerprint, and a second call with nothing changed is a verified no-op, not a rewrite."""
    fp = pipeline_fingerprint(params, seed)
    manifests = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    data_fp = content_fingerprint(df)
    if fp in manifests and manifests[fp]["content_fingerprint"] == data_fp:
        return {**manifests[fp], "status": "skipped (already built)"}
    table = pa.Table.from_pandas(df, preserve_index=False)
    pa_ds.write_dataset(table, DATASET_DIR, format="parquet", partitioning=["symbol"],
                        partitioning_flavor="hive", existing_data_behavior="overwrite_or_ignore")
    manifest = {"pipeline_version": PIPELINE_VERSION, "fingerprint": fp, "content_fingerprint": data_fp,
                "rows": int(len(df)), "params": params, "seed": int(seed), "status": "built"}
    manifests[fp] = manifest
    MANIFEST_PATH.write_text(json.dumps(manifests, indent=2, sort_keys=True))
    return manifest


PARAMS = {"symbols": SYMBOLS, "n_days": N_DAYS, "split_day": SPLIT_DAY, "split_ratio": SPLIT_RATIO}
run1 = run_pipeline(curated, PARAMS, lab.seed)
run2 = run_pipeline(curated, PARAMS, lab.seed)
lab.record("run1_status", run1["status"])
lab.record("run2_status", run2["status"])
lab.record("lake_rows", run1["rows"])
lab.record("lake_fingerprint", run1["content_fingerprint"][:12])
assert run1["status"] == "built"
assert run2["status"] == "skipped (already built)"
assert run1["content_fingerprint"] == run2["content_fingerprint"]

n_partitions = len({p.name for p in DATASET_DIR.iterdir() if p.is_dir()})
lab.record("n_partitions", n_partitions)
assert n_partitions == curated["symbol"].nunique()  # ALPHA, BETA, BETA2, GAMMA — one per TICKER, not per security

# %% [markdown]
# ## 8 · Querying the lake: DuckDB views and polars lazy frames
#
# DuckDB reads the partitioned dataset directly and prunes files whose partition value cannot match
# a filter, without ever opening them. polars' **lazy evaluation** builds the same kind of plan —
# it only touches Parquet bytes once `.collect()` runs, after projections and filters are pushed
# down into the reader. Both engines must agree with the pandas table they were built from.

# %%
glob_path = (DATASET_DIR / "**" / "*.parquet").as_posix()
con = duckdb.connect()
con.execute(f"CREATE VIEW curated_view AS SELECT * FROM read_parquet('{glob_path}', hive_partitioning=true)")
duck_total = con.execute("SELECT COUNT(*) FROM curated_view").fetchone()[0]
duck_one_symbol = con.execute("SELECT COUNT(*) FROM curated_view WHERE symbol = ?", [SPLIT_SYMBOL]).fetchone()[0]
lab.record("duck_rows_total", int(duck_total))
lab.record("duck_rows_one_symbol", int(duck_one_symbol))
assert duck_total == len(curated)
assert duck_one_symbol == int((curated["symbol"] == SPLIT_SYMBOL).sum())

lf = pl.scan_parquet(glob_path, hive_partitioning=True)
mean_pl = lf.filter(pl.col("symbol") == SPLIT_SYMBOL).select(pl.col("adj_price").mean()).collect().item()
mean_pd = float(curated.loc[curated["symbol"] == SPLIT_SYMBOL, "adj_price"].mean())
lab.record("cross_engine_diff", abs(mean_pl - mean_pd))
assert abs(mean_pl - mean_pd) < 1e-6, "polars (lazy, over Parquet) and pandas (in memory) must agree exactly"

# Parquet's columnar, dictionary/RLE-encoded layout compresses this table far below an equivalent CSV.
csv_path = LAKE_DIR / "curated_full.csv"
curated.to_csv(csv_path, index=False)
parquet_bytes = sum(p.stat().st_size for p in DATASET_DIR.rglob("*.parquet"))
csv_bytes = csv_path.stat().st_size
lab.record("parquet_bytes", parquet_bytes)
lab.record("csv_bytes", csv_bytes)
lab.record("compression_ratio", csv_bytes / parquet_bytes)
assert parquet_bytes < csv_bytes

# %% [markdown]
# ## 9 · Proving the lake is usable: a research query
#
# Synthetic excess returns, risk-free rate 0. Computing performance directly on the raw feed vs. on
# the curated, adjusted table shows exactly why Session 4.3's work has to happen before Session 2.3's
# Sharpe ratio and drawdown are trustworthy.

# %%
RF = 0.0
r_raw = simple_returns(raw[SPLIT_SYMBOL]["raw_price"])
r_adj = simple_returns(curated.loc[curated["symbol"] == SPLIT_SYMBOL].set_index("date")["adj_price"])

lab.record("sharpe_raw", sharpe_ratio(r_raw, RF))
lab.record("sharpe_adjusted", sharpe_ratio(r_adj, RF))
lab.record("maxdd_raw", max_drawdown(r_raw))
lab.record("maxdd_adjusted", max_drawdown(r_adj))
assert max_drawdown(r_raw) < -0.35, "the unadjusted split alone must manufacture a deep fake drawdown"
assert max_drawdown(r_adj) > max_drawdown(r_raw) + 0.20, "adjustment must remove most of that fake drawdown"

# %% [markdown]
# ## 10 · Chart for the session page

# %%
from quantnb import charts  # noqa: E402

window = slice(max(SPLIT_DAY - 60, 0), min(SPLIT_DAY + 60, N_DAYS))
raw_years = in_years(raw[SPLIT_SYMBOL]["raw_price"])
adj_years = in_years(adjusted_price[SPLIT_SYMBOL])
lab.chart("split_adjustment", charts.line_chart(
    [charts.Series("Raw price (as delivered)", raw_years.iloc[window], role="loss", label_end=False),
     charts.Series("Adjusted price (back-adjusted)", adj_years.iloc[window], role="strategy", label_end=False)],
    title="ALPHA around its 2-for-1 split: raw feed vs. back-adjusted price",
    y_fmt=charts.fmt_num(0, prefix="$")))

# %%
lab.save()
