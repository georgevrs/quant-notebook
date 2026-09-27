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
# # 6.3 · Build an Event-Driven Backtester — companion lab
#
# **Quant Notebook** · Unit 6 · Session 3 · [Read the session](https://georgevrs.github.io/quant-notebook/unit06-backtesting-discipline/session03-event-driven-backtester.html)
#
# An event loop with orders, fills and portfolio state — slower than the vectorized backtester of
# Session 6.2, but built the same shape as a live trading system.
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
from collections import deque
from dataclasses import dataclass

import numpy as np

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR, in_years

lab = qn.Lab("6.3")  # seeds the random generators: every run gives the same numbers
rng = lab.rng
rng_parity, rng_walkthrough, rng_mc = rng.spawn(3)   # one stream per experiment (§12a of AUTHORING.md)

RISK_FREE_RATE = 0.0  # cash earns nothing while flat, as in Session 2.3's convention
lab.record("risk_free_rate", RISK_FREE_RATE)

# %% [markdown]
# ## 1 · The four things every event-driven engine needs
#
# An **event loop** is a queue plus a `while` loop: pop the oldest event, hand it to whoever
# reacts to that type, and let the reaction enqueue more events. Four event/state types are
# enough to backtest almost anything:
#
# * `MarketEvent` — a new price has arrived.
# * `OrderEvent` — the strategy wants a trade (`MARKET`, fill now; `STOP`, fill only if price
#   crosses a level).
# * `FillEvent` — an order was executed, at a price and a quantity.
# * `Portfolio` — the running state: cash, shares held, and the mark-to-market equity curve.
#
# These are exactly the terms this session owns — the vocabulary Unit 15 reuses when the engine
# stops being a backtest and starts being a live trading system.


# %%
@dataclass
class MarketEvent:
    day: int
    price: float


@dataclass
class OrderEvent:
    day: int
    side: str                      # "BUY" or "SELL"
    order_type: str = "MARKET"     # "MARKET" or "STOP"
    stop_price: float | None = None


@dataclass
class FillEvent:
    day: int
    side: str
    price: float
    quantity: float
    commission: float = 0.0


class Portfolio:
    """Cash + shares + the equity curve — the state an OMS keeps in a real system too."""

    def __init__(self, cash0: float = 1.0):
        self.cash = cash0
        self.shares = 0.0
        self.equity_curve: list[float] = []

    @property
    def invested(self) -> bool:
        return self.shares > 0

    def on_fill(self, fill: FillEvent) -> None:
        if fill.side == "BUY":
            self.shares += fill.quantity
            self.cash -= fill.quantity * fill.price + fill.commission
        else:
            self.shares -= fill.quantity
            self.cash += fill.quantity * fill.price - fill.commission
        self.shares = max(self.shares, 0.0)  # this lab never shorts

    def mark_to_market(self, price: float) -> float:
        equity = self.cash + self.shares * price
        self.equity_curve.append(equity)
        return equity


class ExecutionHandler:
    """Turns OrderEvents into FillEvents. A resting STOP is checked on every MarketEvent."""

    def __init__(self, commission_rate: float = 0.0):
        self.commission_rate = commission_rate
        self.resting_stop: float | None = None

    def submit(self, order: OrderEvent, price: float, portfolio: Portfolio) -> FillEvent:
        qty = (portfolio.cash / price) if order.side == "BUY" else portfolio.shares
        if order.side == "SELL":
            self.resting_stop = None               # flat again: the stop no longer applies
        return FillEvent(order.day, order.side, price, qty,
                         commission=abs(qty * price) * self.commission_rate)

    def rest_stop(self, order: OrderEvent) -> None:
        self.resting_stop = order.stop_price

    def check_resting_stop(self, day: int, price: float, portfolio: Portfolio) -> FillEvent | None:
        if self.resting_stop is not None and portfolio.invested and price <= self.resting_stop:
            self.resting_stop = None
            return FillEvent(day, "SELL", price, portfolio.shares, commission=0.0)
        return None


def run_event_loop(prices: np.ndarray, target_position: np.ndarray,
                    stop_pct: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The event loop: MarketEvent -> strategy decides -> OrderEvent -> FillEvent -> portfolio state.

    `prices[k]` is the price observed at step k (k = 0 .. n). `target_position[k]` is 0/1, decided
    from data available strictly before step k (already lagged — no look-ahead). If `stop_pct` is
    given, every BUY also registers a resting stop `stop_pct` below the fill price.

    The market feed IS the queue: everything one `MarketEvent` causes (a stop check, a decision, an
    `OrderEvent`, the resulting `FillEvent`) is settled before the loop asks the feed for the next
    tick — the same one-message-at-a-time discipline a live system uses, just fed history instead
    of a socket.
    """
    n = len(prices) - 1
    portfolio = Portfolio()
    execution = ExecutionHandler()
    market_events: deque = deque(MarketEvent(k, prices[k]) for k in range(n + 1))
    positions = np.zeros(n + 1)
    stopped_out = False   # a triggered stop takes this simple strategy out for good, no same-day re-entry
    while market_events:
        ev = market_events.popleft()
        portfolio.mark_to_market(ev.price)
        positions[ev.day] = 1.0 if portfolio.invested else 0.0
        stop_fill = execution.check_resting_stop(ev.day, ev.price, portfolio)
        if stop_fill is not None:
            portfolio.on_fill(stop_fill)
            stopped_out = True
        if ev.day == n:
            continue
        want_long = (target_position[ev.day + 1] > 0) and not stopped_out
        order = None
        if want_long and not portfolio.invested:
            order = OrderEvent(ev.day, "BUY")
        elif not want_long and portfolio.invested:
            order = OrderEvent(ev.day, "SELL")
        if order is not None:
            fill = execution.submit(order, ev.price, portfolio)
            portfolio.on_fill(fill)
            if order.side == "BUY" and stop_pct is not None:
                stop_order = OrderEvent(ev.day, "SELL", order_type="STOP",
                                        stop_price=ev.price * (1 - stop_pct))
                execution.rest_stop(stop_order)
    return np.asarray(portfolio.equity_curve), positions


# %% [markdown]
# ## 2 · Same data, same strategy, same answer
#
# Session 6.2 builds a **fair-game universe** — zero-drift GBM, so any Sharpe ratio a signal
# produces is noise, not edge — and a **lookback signal**: the sign of the trailing `k`-day return,
# tested at `k` = 1, 5, 20 and 60. We reuse exactly that construction here: the same fair-game GBM
# (the least volatile of 6.2's eight assets, σ = 12%), the same 60-day lookback, narrowed to one
# asset and to the long side only (position ∈ {0, 1}, not {−1, 0, +1}) so the event loop below only
# needs one order book to read. Nothing about the four event types depends on that narrowing — a
# multi-asset, long-short version is a `Portfolio` keyed by symbol and one `Order` per symbol, not a
# new idea. We run this SAME specification through **two** backtesters on the **same** simulated
# price path: the vectorized `shift`-and-multiply calculation, and the event loop above. If the
# architecture is sound, the two must agree to machine precision — there is only one correct answer
# to "what would this trade have earned".

# %%
MU, SIGMA, N_DAYS, LOOKBACK = 0.0, 0.12, 1_500, 60   # 6.2's fair game: mu=0, its least volatile asset
dt = 1.0 / PERIODS_PER_YEAR
z = rng_parity.standard_normal(N_DAYS)
log_ret = (MU - 0.5 * SIGMA ** 2) * dt + SIGMA * np.sqrt(dt) * z
prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))   # length N_DAYS + 1

# sign of the trailing LOOKBACK-day return, long side only: price[t-1] > price[t-1-LOOKBACK] is
# exactly "the trailing k-day return is positive" — 6.2's lookback_signal(), long-only, one asset
signal = np.zeros(N_DAYS + 1)                       # signal[t]: known once day t's close prints
signal[LOOKBACK + 1:] = (prices[LOOKBACK:-1] > prices[:-LOOKBACK - 1]).astype(float)
target_position = signal                            # trade day t at day t's close, using signal[t]

# --- the vectorized version (Session 6.2's approach): shift, multiply, cumulate ---
ret = prices[1:] / prices[:-1] - 1.0
position_vec = target_position[1:]                  # position held DURING day t = target_position[t]
strategy_ret = position_vec * ret
equity_vec = np.concatenate([[1.0], np.cumprod(1.0 + strategy_ret)])

# --- the event-driven version: the loop above, fed the identical price path ---
equity_evt, position_evt = run_event_loop(prices, target_position)

diff = np.max(np.abs(equity_evt - equity_vec))
lab.record("parity_final_equity_vectorized", float(equity_vec[-1]))
lab.record("parity_final_equity_event", float(equity_evt[-1]))
lab.record("parity_max_abs_diff", float(diff))
lab.record("parity_positions_match", bool(np.array_equal(position_evt[1:], position_vec)))
print(f"vectorized final equity {equity_vec[-1]:.6f} · event-driven {equity_evt[-1]:.6f} · "
      f"max abs diff {diff:.2e}")
# not a statistical claim — both engines do the same floating-point arithmetic on the same data,
# so the two equity curves must match to numerical noise, not to a Monte Carlo tolerance.
assert diff < 1e-9, "the event loop must reproduce the vectorized backtest exactly on this case"
assert np.array_equal(position_evt[1:], position_vec), "the two engines disagree on when they were long"

lab.chart("equity_agree", charts.line_chart(
    [charts.Series("vectorized (6.2)", in_years(equity_vec), role="alt1", end_label="vectorized"),
     charts.Series("event-driven (6.3)", in_years(equity_evt), role="strategy", end_label="event-driven")],
    title="Same data, same strategy — two engines, one answer", y_fmt=charts.fmt_num(2, prefix="$"),
    hline=1.0))

# %% [markdown]
# ## 3 · What a vectorized backtest cannot see: the stop order
#
# A **stop order** only makes sense as a standing instruction that watches every price tick after
# it is placed — a *path-dependent* order. A vectorized backtest sees one row per bar; the only way
# to "check" a stop is to test the bar's close against the stop level once a bar, which means it is
# blind to whatever happened between two closes. The event loop above needs no new machinery for
# this: `ExecutionHandler.check_resting_stop` already runs on every `MarketEvent`, so all we have to
# do is feed it a **finer** stream of price events. First, one path, so you can see the mechanics.

# %%
FINE_PER_DAY = 12          # sub-daily "ticks" the event loop can react to between two closes
STOP_PCT = 0.08


def fine_gbm(n_days: int, per_day: int, mu: float, sigma: float, s0: float,
             rng: np.random.Generator, n_paths: int = 1) -> np.ndarray:
    """Sub-daily GBM path(s), shape (n_days*per_day + 1, n_paths): every `per_day`-th row is a close."""
    steps = n_days * per_day
    dt_fine = 1.0 / (PERIODS_PER_YEAR * per_day)
    z = rng.standard_normal((steps, n_paths))
    log_r = (mu - 0.5 * sigma ** 2) * dt_fine + sigma * np.sqrt(dt_fine) * z
    return s0 * np.exp(np.vstack([np.zeros((1, n_paths)), np.cumsum(log_r, axis=0)]))


H_DAYS = 60
fine_illustrative = fine_gbm(H_DAYS, FINE_PER_DAY, mu=0.0, sigma=0.30, s0=100.0,
                             rng=rng_walkthrough, n_paths=800)
stop_level = 100.0 * (1 - STOP_PCT)
hit_fine = fine_illustrative <= stop_level
triggered = hit_fine.any(axis=0)
# for the walkthrough, pick a triggered path that keeps falling afterwards — the clearest story of
# what the stop bought you (the Monte Carlo below is the general evidence, not this one path)
candidates = np.where(triggered)[0]
worst = candidates[np.argmin(fine_illustrative[-1, candidates])]
one_path = fine_illustrative[:, worst]

always_long = np.ones(H_DAYS + 1)
equity_unprotected, _ = run_event_loop(one_path[::FINE_PER_DAY], always_long)
equity_protected_fine, _ = run_event_loop(one_path, np.ones(H_DAYS * FINE_PER_DAY + 1), stop_pct=STOP_PCT)

lab.record("illustrative_final_unprotected", float(equity_unprotected[-1]))
lab.record("illustrative_final_protected", float(equity_protected_fine[-1]))
print(f"one adverse path: unprotected ends at {equity_unprotected[-1]:.3f}, "
      f"stopped-out at {equity_protected_fine[-1]:.3f}")

lab.chart("stop_value", charts.line_chart(
    [charts.Series("no stop order", in_years(equity_unprotected, PERIODS_PER_YEAR), role="loss"),
     charts.Series(f"stop at −{STOP_PCT:.0%}", in_years(equity_protected_fine, PERIODS_PER_YEAR * FINE_PER_DAY),
                   role="strategy")],
    title="One simulated path — what a resting stop order changes", y_fmt=charts.fmt_num(2, prefix="$"),
    hline=1.0))

# %% [markdown]
# ## 4 · How much worse is "checking only at the close"? (Monte Carlo)
#
# The single path above is intuition, not evidence. To measure the effect, simulate many
# independent paths and compare, on each one, the price at which a stop would actually have
# executed: `fine` checks every sub-daily tick (what the event loop naturally does), `coarse`
# checks only the daily close (the best a vectorized backtest can do without leaving the
# vectorized style). We compute both directly from the price grid — running the object-oriented
# engine over thousands of paths would add runtime without adding anything the walkthrough above
# didn't already show about the *architecture*; what's new here is the *statistic*.

# %%
N_PATHS_MC = 6_000
fine_mc = fine_gbm(H_DAYS, FINE_PER_DAY, mu=0.0, sigma=0.30, s0=100.0, rng=rng_mc, n_paths=N_PATHS_MC)
stop_price = 100.0 * (1 - STOP_PCT)
daily_close = fine_mc[FINE_PER_DAY::FINE_PER_DAY]                 # (H_DAYS, N_PATHS_MC)

hit_fine = fine_mc <= stop_price
fine_idx = np.argmax(hit_fine, axis=0)
fine_triggered = hit_fine.any(axis=0)
fine_exit_price = fine_mc[fine_idx, np.arange(N_PATHS_MC)]

hit_coarse = daily_close <= stop_price
coarse_idx = np.argmax(hit_coarse, axis=0)
coarse_triggered = hit_coarse.any(axis=0)
coarse_exit_price = daily_close[coarse_idx, np.arange(N_PATHS_MC)]

# a coarse trigger is always found by the fine grid too (the close IS one of the fine points),
# so "triggered but coarse misses it" is the whole story of checking too rarely
miss_mask = fine_triggered & ~coarse_triggered
both_mask = fine_triggered & coarse_triggered
assert np.array_equal(coarse_triggered & fine_triggered, coarse_triggered), \
    "a breach visible at the close must also be visible on the finer grid"

n_triggered = int(fine_triggered.sum())
miss_rate = float(miss_mask.sum() / n_triggered)
lab.record("mc_paths", N_PATHS_MC)
lab.record("mc_horizon_days", H_DAYS)
lab.record("mc_stop_pct", STOP_PCT)
lab.record("mc_n_triggered", n_triggered)
lab.record("mc_trigger_rate", n_triggered / N_PATHS_MC)
lab.record("mc_miss_rate", miss_rate)
assert n_triggered > 500, "raise N_PATHS_MC or SIGMA: too few triggered paths for a stable estimate"
assert miss_mask.sum() > 0, "the close-only check should miss at least some intraday breaches"

fine_overshoot_bps = (stop_price - fine_exit_price[both_mask]) / stop_price * 1e4
coarse_overshoot_bps = (stop_price - coarse_exit_price[both_mask]) / stop_price * 1e4
gap_bps = coarse_overshoot_bps - fine_overshoot_bps            # positive: coarse checking cost more
n_pairs = int(both_mask.sum())
mean_gap = float(gap_bps.mean())
se_gap = float(gap_bps.std(ddof=1) / np.sqrt(n_pairs))

lab.record("mc_n_pairs", n_pairs)
lab.record("mean_overshoot_fine_bps", float(fine_overshoot_bps.mean()))
lab.record("mean_overshoot_coarse_bps", float(coarse_overshoot_bps.mean()))
lab.record("mean_overshoot_gap_bps", mean_gap)
lab.record("overshoot_gap_se_bps", se_gap)
lab.record("overshoot_gap_se_multiple", mean_gap / se_gap)
print(f"{n_triggered}/{N_PATHS_MC} paths breached the stop; close-only checking missed "
      f"{miss_rate:.1%} of them entirely; when both catch it, close-only execution is worse by "
      f"{mean_gap:.1f} bps on average (SE {se_gap:.2f}, {mean_gap / se_gap:.1f} SE from zero)")
# Monte Carlo standard: 4 standard errors, never 3, never a seed hunt (§12a of AUTHORING.md)
assert mean_gap > 4 * se_gap, "checking only at the close should be measurably worse, not just noisier"

lab.chart("overshoot_comparison", charts.column_chart(
    ["checks every tick", "checks at the close"],
    [fine_overshoot_bps.mean(), coarse_overshoot_bps.mean()],
    title="Average overshoot past the stop level", y_fmt=charts.fmt_num(1, suffix=" bps"),
    roles=["strategy", "loss"]))

# %%
lab.save()
