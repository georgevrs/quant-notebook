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
# # 4.4 · Time, Calendars, As-of Joins & Bars — companion lab
#
# **Quant Notebook** · Unit 4 · Session 4 · [Read the session](https://georgevrs.github.io/quant-notebook/unit04-data-engineering/session04-time-calendars-bars.html)
#
# Timestamps and time zones, exchange calendars, as-of joins that never peek, and information-driven
# bars (time, volume, dollar, tick imbalance) built from synthetic tick data — with pandas 3, polars
# and DuckDB, this unit's research stack.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.

# %%
# Colab or a fresh environment: install the course package (skipped when it is already installed).
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("quantnb") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "quantnb @ git+https://github.com/georgevrs/quant-notebook@main"], check=True)

# %%
import calendar
import datetime as dt
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import kurtosis, norm

import quantnb as qn
from quantnb import charts

lab = qn.Lab("4.4")  # seeds the random generators: every run gives the same numbers
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# %% [markdown]
# ## 1 · One instant, several clocks
#
# A timestamp with no time zone is a bare number: you cannot compare it, join it, or even say how
# many hours ago it was until you know which clock produced it. `America/New_York` sits at UTC-5
# (Eastern Standard Time) in winter and UTC-4 (Eastern Daylight Time) in summer — the SAME wall-clock
# reading on the exchange floor is a different instant in UTC depending on the date.

# %%
winter = pd.Timestamp("2026-01-15 09:30", tz=NY)
summer = pd.Timestamp("2026-07-15 09:30", tz=NY)
winter_offset_h = winter.utcoffset().total_seconds() / 3600
summer_offset_h = summer.utcoffset().total_seconds() / 3600
lab.record("winter_utc_offset_h", winter_offset_h)
lab.record("summer_utc_offset_h", summer_offset_h)
print(f"09:30 ET in January = {winter.tz_convert(UTC).strftime('%H:%M')} UTC "
      f"({winter_offset_h:g}h); 09:30 ET in July = {summer.tz_convert(UTC).strftime('%H:%M')} UTC "
      f"({summer_offset_h:g}h)")
assert winter_offset_h == -5.0
assert summer_offset_h == -4.0

# %% [markdown]
# **Daylight saving is not a rounding error — it is two calendar days a year where the mapping from
# local time to UTC is not a function.** In 2026 the US clocks spring forward on 8 March (the local
# hour 02:00-03:00 never happens) and fall back on 1 November (the local hour 01:00-02:00 happens
# twice). A pipeline that adds a fixed 5-hour offset to "9:30am New York" is exactly one hour wrong
# for roughly eight months of the year, and outright ambiguous on the two changeover days.

# %%
DST_START_2026 = "2026-03-08"
DST_END_2026 = "2026-11-01"
lab.record("dst_start_2026", DST_START_2026)
lab.record("dst_end_2026", DST_END_2026)

# Spring forward: 2026-03-08 02:30 local does not exist. pandas must be told what to do with it.
# "shift_forward" moves to the first valid instant after the gap — the gap swallows the half hour.
spring_forward = pd.Timestamp("2026-03-08 02:30").tz_localize(NY, nonexistent="shift_forward")
lab.record("spring_forward_result", spring_forward.strftime("%H:%M %Z"))
assert (spring_forward.hour, spring_forward.minute) == (3, 0)

# Fall back: 2026-11-01 01:30 local happens twice — ambiguous without more information.
fall_dst = pd.Timestamp("2026-11-01 01:30").tz_localize(NY, ambiguous=True)     # the earlier (EDT) instant
fall_std = pd.Timestamp("2026-11-01 01:30").tz_localize(NY, ambiguous=False)    # the later (EST) instant
fallback_gap_h = (fall_std.tz_convert(UTC) - fall_dst.tz_convert(UTC)).total_seconds() / 3600
lab.record("fallback_gap_h", fallback_gap_h)
assert fallback_gap_h == 1.0
print(f"'01:30' on 2026-11-01 happens at {fall_dst.tz_convert(UTC).strftime('%H:%M')} UTC "
      f"AND {fall_std.tz_convert(UTC).strftime('%H:%M')} UTC — {fallback_gap_h:g}h apart")

# %% [markdown]
# ## 2 · A (teaching-scale) exchange calendar
#
# A time zone tells you how to convert a reading; an **exchange calendar** tells you which readings
# are inside a trading session at all. Weekends are the easy 2/7 of the problem. The rest — roughly
# nine full closures a year plus a couple of early closes — are exchange-specific rules that change
# slowly and must be looked up, not guessed. This calendar is simplified for teaching; §4.4.7 names
# the packages production systems use instead.

# %%
def nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th (1-indexed) given weekday of a month — e.g. the 3rd Monday of January."""
    d = dt.date(year, month, 1)
    return d + dt.timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> dt.date:
    """The last given weekday of a month — e.g. the last Monday of May."""
    last_day = calendar.monthrange(year, month)[1]
    d = dt.date(year, month, last_day)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher) — needed because Good Friday moves."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return dt.date(year, month, day)


def observed(d: dt.date) -> dt.date:
    """A holiday landing on a weekend is observed the nearest weekday (the NYSE convention)."""
    if d.weekday() == 5:   # Saturday -> observed Friday
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:   # Sunday -> observed Monday
        return d + dt.timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[dt.date]:
    """A simplified NYSE-style calendar: the nine full-day closures observed every year."""
    return {
        observed(dt.date(year, 1, 1)),      # New Year's Day
        nth_weekday(year, 1, 0, 3),          # Martin Luther King Jr. Day
        nth_weekday(year, 2, 0, 3),          # Washington's Birthday
        easter_sunday(year) - dt.timedelta(days=2),   # Good Friday
        last_weekday(year, 5, 0),            # Memorial Day
        observed(dt.date(year, 6, 19)),      # Juneteenth (added 2022)
        observed(dt.date(year, 7, 4)),       # Independence Day
        nth_weekday(year, 9, 0, 1),           # Labor Day
        nth_weekday(year, 11, 3, 4),          # Thanksgiving
        observed(dt.date(year, 12, 25)),     # Christmas
    }


def nyse_half_days(year: int, holidays: set[dt.date]) -> set[dt.date]:
    """1:00pm ET closes: the day after Thanksgiving, and 24 Dec when it is a business day."""
    thanksgiving = nth_weekday(year, 11, 3, 4)
    out = {thanksgiving + dt.timedelta(days=1)}
    christmas_eve = dt.date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in holidays:
        out.add(christmas_eve)
    return out


def trading_days(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Business days minus the toy holiday calendar, for every year in [start_year, end_year]."""
    holidays: set[dt.date] = set()
    for y in range(start_year, end_year + 1):
        holidays |= nyse_holidays(y)
    all_days = pd.bdate_range(f"{start_year}-01-01", f"{end_year}-12-31")
    return all_days.difference(pd.DatetimeIndex(sorted(holidays)))


# %%
holidays_2026 = nyse_holidays(2026)
half_days_2026 = nyse_half_days(2026, holidays_2026)
good_friday_2026 = easter_sunday(2026) - dt.timedelta(days=2)
thanksgiving_2026 = nth_weekday(2026, 11, 3, 4)

lab.record("n_nyse_holidays_2026", len(holidays_2026))
lab.record("n_nyse_half_days_2026", len(half_days_2026))
lab.record("good_friday_2026", str(good_friday_2026))
lab.record("independence_day_observed_2026", str(observed(dt.date(2026, 7, 4))))
lab.record("thanksgiving_2026", str(thanksgiving_2026))
lab.record("day_after_thanksgiving_2026", str(thanksgiving_2026 + dt.timedelta(days=1)))

# Facts checked against the NYSE Group 2024-2026 holiday and early-closing calendar.
assert dt.date(2026, 1, 1) in holidays_2026                 # New Year's Day (a Thursday)
assert dt.date(2026, 4, 3) in holidays_2026                 # Good Friday
assert dt.date(2026, 7, 4) not in holidays_2026             # 4 July 2026 is a Saturday...
assert dt.date(2026, 7, 3) in holidays_2026                 # ...so Friday 3 July is observed instead
assert dt.date(2026, 11, 26) in holidays_2026               # Thanksgiving
assert dt.date(2026, 11, 27) in half_days_2026              # 1:00pm ET close, day after Thanksgiving
assert dt.date(2026, 12, 24) in half_days_2026              # Christmas Eve, a Thursday in 2026
assert len(holidays_2026) == 10

cal_2026 = trading_days(2026, 2026)
lab.record("n_trading_days_2026", int(len(cal_2026)))
print(f"2026: {len(holidays_2026)} full closures, {len(half_days_2026)} early closes, "
      f"{len(cal_2026)} trading days")

# %% [markdown]
# ## 3 · As-of joins that never peek
#
# An **as-of join** attaches to each row of a left table the most recent matching row of a right
# table *as of* that row's timestamp — never a later one. Point-in-time research (§4.4.6) depends on
# it: the join key must be the moment a fact became *knowable*, not the moment it happened.

# %%
rng_prices, rng_fund = lab.rng.spawn(2)

START_YEAR, END_YEAR = 2022, 2025
cal = trading_days(START_YEAR, END_YEAR)
prices = qn.synth.gbm_prices(len(cal) - 1, mu=0.08, sigma=0.22, rng=rng_prices)["price"]
prices.index = cal   # the simulated path lands on real trading days: weekends and holidays are gone
prices_df = prices.rename("price").reset_index().rename(columns={"index": "date"}).sort_values("date")

MEAN_LAG_DAYS = 45
quarter_ends = pd.date_range(f"{START_YEAR}-03-31", f"{END_YEAR}-12-31", freq="QE")
lags = rng_fund.integers(35, 56, size=len(quarter_ends))              # days of reporting delay, jittered
available = pd.DatetimeIndex(quarter_ends) + pd.to_timedelta(lags, unit="D")
eps = 1.00 + np.cumsum(rng_fund.normal(0.03, 0.05, size=len(quarter_ends)))
fundamentals = pd.DataFrame({"period_end": quarter_ends, "available_time": available, "eps": eps})
lab.record("reporting_lag_mean_days", float(lags.mean()))
lab.record("n_quarters", len(quarter_ends))

# The CORRECT join: match on the moment the number became public, never on the period it describes.
correct = pd.merge_asof(prices_df, fundamentals.sort_values("available_time")[["available_time", "eps"]],
                        left_on="date", right_on="available_time",
                        direction="backward", tolerance=pd.Timedelta(days=200))

# The BUGGY join: match on period_end directly — the classic look-ahead a naive join manufactures.
buggy = pd.merge_asof(prices_df, fundamentals.sort_values("period_end")[["period_end", "eps", "available_time"]],
                      left_on="date", right_on="period_end", direction="backward")

leak_days = (buggy["available_time"] - buggy["date"]).dt.days
leaking = leak_days[leak_days > 0]
lab.record("pct_days_leaking", float((leak_days > 0).mean()))
lab.record("mean_leak_days", float(leaking.mean()))
lab.record("max_leak_days", float(leaking.max()))
# On the day a quarter ends, the buggy join is already using a number that will not exist for
# another `lag` days; as trading days pass toward the true release date the leak shrinks back to
# zero, so its average over the whole leaking window is close to half the mean reporting lag — a
# structural property of this construction (463 leaking day-rows), not a noisy estimate.
assert abs(leaking.mean() - lags.mean() / 2) < 5.0
assert leaking.max() <= lags.max()
assert ((correct["available_time"] - correct["date"]).dt.days.dropna() <= 0).all()
print(f"buggy join knows the future on {(leak_days > 0).mean():.0%} of days: "
      f"up to {leaking.max():.0f} days ahead, {leaking.mean():.1f} days ahead on average")

# %% [markdown]
# **Tolerance matters too.** Without it, `merge_asof` carries a stale value forward forever once a
# company stops reporting. Drop two consecutive quarters to simulate a real gap, then compare a join
# with no tolerance against one that refuses to reach back more than 200 days.

# %%
gapped = fundamentals.sort_values("available_time").drop(fundamentals.index[[4, 5]])
no_tolerance = pd.merge_asof(prices_df, gapped[["available_time", "eps"]],
                             left_on="date", right_on="available_time", direction="backward")
with_tolerance = pd.merge_asof(prices_df, gapped[["available_time", "eps"]],
                               left_on="date", right_on="available_time",
                               direction="backward", tolerance=pd.Timedelta(days=200))
n_stale_avoided = int((no_tolerance["eps"].notna() & with_tolerance["eps"].isna()).sum())
lab.record("n_stale_avoided", n_stale_avoided)
assert n_stale_avoided > 0
print(f"tolerance=200D turned {n_stale_avoided} stale-but-silent rows into an honest NaN")

# %% [markdown]
# The same discipline in SQL: DuckDB's native `ASOF JOIN` (this unit's research stack) does exactly
# what `merge_asof(direction="backward")` does, over Arrow-backed data DuckDB can query where it sits.

# %%
con = duckdb.connect()
con.register("px", prices_df)
con.register("fx", fundamentals.sort_values("available_time")[["available_time", "eps"]])
duck_result = con.execute("""
    SELECT px.date, px.price, fx.eps
    FROM px ASOF LEFT JOIN fx
      ON px.date >= fx.available_time
    ORDER BY px.date
""").df()
pd.testing.assert_series_equal(duck_result["eps"].reset_index(drop=True),
                               correct["eps"].reset_index(drop=True), check_names=False)
lab.record("duckdb_matches_pandas", True)

# %% [markdown]
# ## 4 · From ticks to bars: four ways to slice a trading day
#
# Synthetic tick-by-tick trades for one session: arrival times from a thinned Poisson process with a
# mild U-shape (busier at the open and close) plus a 20-minute "news" burst with a strong buy
# imbalance, built with polars. Prices move by one tick per trade (or not at all, 15% of the time —
# real prints do repeat); trade sizes are drawn from a heavy-tailed lognormal.

# %%
TRADING_SECONDS = 6.5 * 3600      # a 09:30-16:00 session, in elapsed seconds — never a calendar date
TICK_SIZE = 0.01
BURST_START_S, BURST_LEN_S = 3.0 * 3600, 20 * 60   # a 20-minute news window at hour 3 of the session
N_REPS = 30


def simulate_tick_day(rng: np.random.Generator) -> pl.DataFrame:
    """One synthetic trading day of tick prints: elapsed seconds, price, volume, in a news burst?"""
    lam_base, lam_burst = 1.2, 9.0                     # ticks / second
    lam_max = lam_burst + lam_base * 3
    n_candidates = int(lam_max * TRADING_SECONDS * 1.3)
    t = np.cumsum(rng.exponential(1.0 / lam_max, n_candidates))
    t = t[t < TRADING_SECONDS]

    u_shape = lam_base * (1 + 2 * np.exp(-t / 1800) + 2 * np.exp(-(TRADING_SECONDS - t) / 1800))
    in_burst = (t >= BURST_START_S) & (t < BURST_START_S + BURST_LEN_S)
    intensity = np.where(in_burst, lam_burst, u_shape)
    keep = rng.uniform(0, lam_max, len(t)) < intensity   # thinning (Lewis-Shedler)
    t, in_burst = t[keep], in_burst[keep]
    n = len(t)

    p_up = np.where(in_burst, 0.85, 0.5)                 # informed buying pressure during the burst
    up = rng.uniform(0, 1, n) < p_up
    zero_tick = rng.uniform(0, 1, n) < 0.15
    delta = np.where(zero_tick, 0.0, np.where(up, TICK_SIZE, -TICK_SIZE))
    price = 100.0 + np.cumsum(delta)
    size = rng.lognormal(mean=np.log(100.0), sigma=0.8, size=n)

    return pl.DataFrame({"t": t, "price": price, "volume": np.round(size).clip(min=1), "in_burst": in_burst})


def tick_rule(price: np.ndarray) -> np.ndarray:
    """b_t = sign(P_t - P_{t-1}); a zero tick carries the previous sign forward (b_0 = +1)."""
    diffs = np.diff(price, prepend=price[0])
    b = np.sign(diffs)
    b[0] = 1.0
    return pd.Series(np.where(b == 0.0, np.nan, b)).ffill().to_numpy()


def time_bars(ticks: pl.DataFrame, dt_s: float = 120.0) -> pd.DataFrame:
    """Fixed-clock-time bars: one bar every dt_s seconds, however much or little happened inside it."""
    out = (ticks.with_columns((pl.col("t") // dt_s).cast(pl.Int64).alias("bar"))
                .group_by("bar", maintain_order=True)
                .agg(open=pl.col("price").first(), close=pl.col("price").last(),
                     volume=pl.col("volume").sum(), n_ticks=pl.len(),
                     t_open=pl.col("t").first(), t_close=pl.col("t").last()))
    return out.to_pandas()


def threshold_bars(t: np.ndarray, price: np.ndarray, volume: np.ndarray,
                   activity: np.ndarray, threshold: float) -> pd.DataFrame:
    """A new bar every time cumulative `activity` (volume, or dollar value) crosses `threshold`."""
    bar_id = (np.cumsum(activity) // threshold).astype(int)
    df = pd.DataFrame({"t": t, "price": price, "volume": volume, "bar": bar_id})
    g = df.groupby("bar")
    return g.agg(open=("price", "first"), close=("price", "last"), volume=("volume", "sum"),
                n_ticks=("price", "size"), t_open=("t", "first"), t_close=("t", "last")).reset_index(drop=True)


def tick_imbalance_bars(t: np.ndarray, price: np.ndarray, warmup: int = 200,
                        ewma_alpha: float = 0.2, min_len: int = 5) -> pd.DataFrame:
    """AFML tick imbalance bars: cut when the signed tick run |theta| exceeds an adaptive threshold.

    Bootstrapped from a `warmup`-tick estimate of the expected run length and imbalance, then the
    threshold adapts (an EWMA) from every bar actually formed — so a run of same-signed ticks
    (informed trading) closes a bar fast, and a symmetric random walk needs far more ticks.
    """
    b = tick_rule(price)
    n = len(b)
    exp_T = float(warmup)
    exp_b = max(abs(float(np.mean(b[:warmup]))), 1e-3)
    threshold = exp_T * exp_b
    starts, ends, lengths, imbalances = [], [], [], []
    theta, start = 0.0, 0
    for i in range(n):
        theta += b[i]
        if abs(theta) >= threshold and (i - start + 1) >= min_len:
            starts.append(start); ends.append(i)
            lengths.append(i - start + 1); imbalances.append(theta)
            exp_T = ewma_alpha * lengths[-1] + (1 - ewma_alpha) * exp_T
            exp_b = ewma_alpha * abs(imbalances[-1] / lengths[-1]) + (1 - ewma_alpha) * exp_b
            threshold = exp_T * max(exp_b, 1e-3)
            theta, start = 0.0, i + 1
    return pd.DataFrame({"start": starts, "end": ends, "length": lengths, "imbalance": imbalances})


def log_ret(closes: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.asarray(closes, dtype=float)))


# %% [markdown]
# One illustrative trading day, used for the charts and tables below; the Monte Carlo comparison that
# follows repeats this on 30 independent days to check the claims are not a one-seed accident.

# %%
illustration_rng, *reps_rng = lab.rng.spawn(N_REPS + 1)

ticks = simulate_tick_day(illustration_rng)
t, price = ticks["t"].to_numpy(), ticks["price"].to_numpy()
volume, in_burst = ticks["volume"].to_numpy(), ticks["in_burst"].to_numpy()
dollar = price * volume

tb = time_bars(ticks)
vb = threshold_bars(t, price, volume, volume, max(volume.sum() / len(tb), 1.0))
db = threshold_bars(t, price, volume, dollar, max(dollar.sum() / len(tb), 1.0))
tib = tick_imbalance_bars(t, price)

lab.record("illustration_n_ticks", int(len(t)))
lab.record("illustration_n_time_bars", int(len(tb)))
lab.record("illustration_n_volume_bars", int(len(vb)))
lab.record("illustration_n_dollar_bars", int(len(db)))
lab.record("illustration_n_imbalance_bars", int(len(tib)))
lab.record("illustration_kurt_time", float(kurtosis(log_ret(tb["close"]), fisher=True)))
lab.record("illustration_kurt_volume", float(kurtosis(log_ret(vb["close"]), fisher=True)))
lab.record("illustration_kurt_dollar", float(kurtosis(log_ret(db["close"]), fisher=True)))
print(f"one session: {len(t)} ticks -> {len(tb)} time bars, {len(vb)} volume bars, "
      f"{len(db)} dollar bars, {len(tib)} tick-imbalance bars")

# %% [markdown]
# ## 5 · Charts for the session page

# %%
elapsed_h = t / 3600.0
lab.chart("activity", charts.line_chart(
    [charts.Series("cumulative dollar volume", pd.Series(np.cumsum(dollar), index=elapsed_h),
                   role="strategy", area=True, label_end=False)],
    title="Cumulative dollar volume through one simulated trading day", y_fmt=charts.fmt_usd_compact(1),
    bands=[(BURST_START_S / 3600, (BURST_START_S + BURST_LEN_S) / 3600, "")]))  # named in the caption:
    # the shaded band is too narrow at this scale to hold its own label without overflowing it

lab.chart("bar_counts", charts.column_chart(
    ["Time (2 min)", "Volume", "Dollar", "Tick imbalance"],
    [len(tb), len(vb), len(db), len(tib)],
    title="Bars produced by four sampling rules — one simulated trading day", y_fmt=charts.fmt_num(0)))


def normal_overlay(sample: np.ndarray):
    mu, sd = float(np.mean(sample)), float(np.std(sample, ddof=1))
    xs = np.linspace(sample.min(), sample.max(), 200)
    return xs, norm.pdf(xs, mu, sd)


tb_ret, vb_ret = log_ret(tb["close"]), log_ret(vb["close"])
lab.chart("time_bar_hist", charts.histogram(
    tb_ret, title="Time-bar log returns (2-minute bars)", bins=24, x_fmt=charts.fmt_pct(2),
    density_overlay=normal_overlay(tb_ret)))
lab.chart("volume_bar_hist", charts.histogram(
    vb_ret, title="Volume-bar log returns (count-matched)", bins=24, x_fmt=charts.fmt_pct(2),
    density_overlay=normal_overlay(vb_ret)))

# %% [markdown]
# ## 6 · Monte Carlo: is "volume bars are closer to normal" a real effect?
#
# One session proves nothing on its own — repeat the whole simulation on 30 independent days (a
# fresh RNG stream per day) and compare the *excess kurtosis* of bar returns (0 for a true Gaussian;
# higher means fatter tails). Assert the direction of the effect at four standard errors, not on a
# single lucky draw.

# %%
def mc_stat(name: str, values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean, se = float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))
    lab.record(f"{name}_mean", mean)
    lab.record(f"{name}_se", se)
    return mean, se


kurt_time, kurt_vol, kurt_dollar, dur_burst, dur_calm, dur_diff = [], [], [], [], [], []
for r in reps_rng:
    tk = simulate_tick_day(r)
    t_r, p_r = tk["t"].to_numpy(), tk["price"].to_numpy()
    v_r, b_r = tk["volume"].to_numpy(), tk["in_burst"].to_numpy()
    d_r = p_r * v_r

    tb_r = time_bars(tk)
    vb_r = threshold_bars(t_r, p_r, v_r, v_r, max(v_r.sum() / len(tb_r), 1.0))
    db_r = threshold_bars(t_r, p_r, v_r, d_r, max(d_r.sum() / len(tb_r), 1.0))
    tib_r = tick_imbalance_bars(t_r, p_r)

    kurt_time.append(kurtosis(log_ret(tb_r["close"]), fisher=True))
    kurt_vol.append(kurtosis(log_ret(vb_r["close"]), fisher=True))
    kurt_dollar.append(kurtosis(log_ret(db_r["close"]), fisher=True))

    durations = t_r[tib_r["end"].to_numpy()] - t_r[tib_r["start"].to_numpy()]
    majority_burst = np.array([b_r[s:e + 1].mean() > 0.5 for s, e in zip(tib_r["start"], tib_r["end"])])
    if majority_burst.any():
        dur_burst.append(durations[majority_burst].mean())
    if (~majority_burst).any():
        dur_calm.append(durations[~majority_burst].mean())
    if majority_burst.any() and (~majority_burst).any():
        dur_diff.append(durations[~majority_burst].mean() - durations[majority_burst].mean())

kt_mean, kt_se = mc_stat("kurt_time", kurt_time)
kv_mean, kv_se = mc_stat("kurt_volume", kurt_vol)
kd_mean, kd_se = mc_stat("kurt_dollar", kurt_dollar)
dtv_mean, dtv_se = mc_stat("kurt_diff_time_volume", np.array(kurt_time) - np.array(kurt_vol))
dtd_mean, dtd_se = mc_stat("kurt_diff_time_dollar", np.array(kurt_time) - np.array(kurt_dollar))
mc_stat("imbalance_dur_burst", dur_burst)
mc_stat("imbalance_dur_calm", dur_calm)
dd_mean, dd_se = mc_stat("imbalance_dur_diff", dur_diff)
lab.record("mc_reps", N_REPS)

print(f"excess kurtosis: time {kt_mean:.2f} · volume {kv_mean:.2f} · dollar {kd_mean:.2f} "
      f"(diff vs volume {dtv_mean:.2f} +/- {4 * dtv_se:.2f} at 4 SE)")
assert dtv_mean > 4 * dtv_se, "time bars should be more leptokurtic than count-matched volume bars"
assert dtd_mean > 4 * dtd_se, "time bars should be more leptokurtic than count-matched dollar bars"
assert dd_mean > 4 * dd_se, "tick-imbalance bars should close faster during the informed-trading burst"

# %%
lab.save()
