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
# # 2.1 · Market Anatomy & the Life of an Order — companion lab
#
# **Quant Notebook** · Unit 2 · Session 1 · [Read the session](https://georgevrs.github.io/quant-notebook/unit02-markets-instruments/session01-market-anatomy.html)
#
# Follow one order from your code through broker, venue, clearing house and custodian — and see where every fee and risk sits.
#
# Four experiments, one per stage of an order's life:
#
# 1. **The book.** Build a small synthetic limit order book and walk a market order through it.
# 2. **The stop.** Simulate stop and stop-limit orders on a price that can gap overnight (Monte Carlo).
# 3. **The clearing house.** Net a day of trades bilaterally and through a central counterparty.
# 4. **The bill.** Add up a round trip's costs, component by component, and turn them into a yearly drag.
#
# Run it top to bottom. Every number the session page quotes is recorded with `lab.record(...)`
# and saved to `out/` by the last cell, so the page and this notebook can never disagree.
#
# All prices, book depths, commissions and venue fees below are **illustrative** round numbers, not
# market data. The two US regulatory fees are the real rates in force in 2026: the SEC's Section 31
# fee ($20.60 per million dollars of sale proceeds, from 4 April 2026) and FINRA's Trading Activity
# Fee ($0.000195 per share sold, capped at $9.79 a trade, from 1 January 2026). The exchange fee is
# set at Regulation NMS's cap on the fee for taking a protected quote, $0.003 a share (as of 2026).
# Simulated time is elapsed trading time (252 days a year), never a calendar date.

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
from scipy.special import zeta

import quantnb as qn
from quantnb import charts

lab = qn.Lab("2.1")  # seeds the random generators: every run gives the same numbers
# One random stream per experiment, so editing one section never changes another's results.
rng_book, rng_thin, rng_stop, rng_net = lab.rng.spawn(4)

# %% [markdown]
# ## 1 · A limit order book, and a market order walking through it
#
# A **limit order book** is the list of resting limit orders on one venue: buyers' **bids** below,
# sellers' **asks** (offers) above, each price level with the total size waiting there. The best bid
# and best ask form the **touch**; the gap between them is the **bid-ask spread**, and halfway
# between them is the **mid price**.
#
# A market order takes whatever is resting, best price first. A small one pays the half-spread (the
# distance from the mid to the touch). A big one also "walks the book": it empties the best level
# and keeps going into worse prices. Its average price is the volume-weighted average of the levels
# it touched (VWAP).

# %%
TICK = 0.01            # US stocks above $1 quote in whole cents
LOT = 100              # shares per round lot
N_LEVELS = 20          # price levels per side in the snapshot


def make_book(rng: np.random.Generator, mid: float, spread_ticks: int, lots_at_touch: float,
              growth: float, level_gap_ticks: int = 1, n_levels: int = N_LEVELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic order-book snapshot: prices on the tick grid, sizes in round lots.

    Level k (0 = the touch) holds on average lots_at_touch × (1 + growth·k) lots — depth grows away
    from the touch, a shape real books often show. Thin books also skip ticks between levels
    (level_gap_ticks > 1). Returns (bids, asks), each sorted best price first.
    """
    mid_ticks = round(mid / TICK)
    k = np.arange(n_levels)
    half = spread_ticks / 2
    bid_ticks = mid_ticks - half - level_gap_ticks * k
    ask_ticks = mid_ticks + half + level_gap_ticks * k
    mean_lots = lots_at_touch * (1 + growth * k)
    bids = pd.DataFrame({"price": np.round(bid_ticks * TICK, 2),
                         "size": LOT * (1 + rng.poisson(mean_lots - 1))})   # at least one lot per level
    asks = pd.DataFrame({"price": np.round(ask_ticks * TICK, 2),
                         "size": LOT * (1 + rng.poisson(mean_lots - 1))})
    return bids, asks


def sweep(levels: pd.DataFrame, qty: int) -> dict:
    """Walk a market order of `qty` shares through one side of the book, best price first.

    Returns the shares filled, the VWAP of the fills, the number of price levels reached, how many
    of them it emptied completely, and the worst price paid. If the displayed book runs out, the
    rest is reported as unfilled.
    """
    size = levels["size"].to_numpy()
    before = np.concatenate([[0], np.cumsum(size)[:-1]])      # shares resting ahead of each level
    take = np.clip(qty - before, 0, size)                     # shares taken at each level
    filled = int(take.sum())
    touched = int((take > 0).sum())
    return {"filled": filled, "unfilled": int(qty - filled),
            "vwap": float((take * levels["price"].to_numpy()).sum() / filled),
            "levels": touched, "emptied": int((take == size).sum()),
            "worst": float(levels["price"].to_numpy()[touched - 1])}


def bps(x: float) -> float:
    """A fraction in basis points (1 bp = 0.01%)."""
    return 1e4 * x


MID = 50.00
bids, asks = make_book(rng_book, MID, spread_ticks=2, lots_at_touch=10, growth=0.25)
best_bid, best_ask = float(bids.price[0]), float(asks.price[0])
mid = (best_bid + best_ask) / 2
spread = best_ask - best_bid
print(pd.concat([bids.head(5).add_prefix("bid_"), asks.head(5).add_prefix("ask_")], axis=1))
print(f"touch {best_bid:.2f} / {best_ask:.2f} · spread ${spread:.2f} = {bps(spread / mid):.1f} bp · mid {mid:.3f}")

for k in range(5):
    lab.record(f"bid_px_{k}", float(bids.price[k]))
    lab.record(f"bid_sz_{k}", int(bids["size"][k]))
    lab.record(f"ask_px_{k}", float(asks.price[k]))
    lab.record(f"ask_sz_{k}", int(asks["size"][k]))
lab.record("mid", mid)
lab.record("spread", spread)
lab.record("spread_bps", bps(spread / mid))
lab.record("half_spread_bps", bps(spread / 2 / mid))
lab.record("book_levels", N_LEVELS)
lab.record("book_depth_shares", int(asks["size"].sum()))
assert np.isclose(mid, MID) and best_ask > best_bid

# %% [markdown]
# **Walk the book.** A market BUY for each order size. The cost against the mid splits exactly into
# the half-spread (mid → best ask) and the depth cost (best ask → VWAP):
#
#     cost = (VWAP − mid)/mid = (ask − mid)/mid + (VWAP − ask)/mid

# %%
ORDER_SIZES = [100, 1_000, 2_500, 5_000, 10_000, 25_000]
walk_rows = []
for q in ORDER_SIZES:
    f = sweep(asks, q)
    if f["unfilled"]:
        print(f"{q:,} shares: the displayed book ran out, {f['unfilled']:,} unfilled — a real order "
              "would keep sweeping hidden or refilled liquidity, or be stopped by a price collar")
    walk_rows.append({"qty": q, **f,
                      "cost_bps": bps((f["vwap"] - mid) / mid),
                      "half_spread_bps": bps((best_ask - mid) / mid),
                      "depth_bps": bps((f["vwap"] - best_ask) / mid),
                      "notional": q * mid})
walk = pd.DataFrame(walk_rows).set_index("qty")
print(walk.round(3))
for q, row in walk.iterrows():
    lab.record(f"walk_{q}_qty", int(q))
    lab.record(f"walk_{q}_vwap", row.vwap)
    lab.record(f"walk_{q}_cost_bps", row.cost_bps)
    lab.record(f"walk_{q}_depth_bps", row.depth_bps)
    lab.record(f"walk_{q}_levels", int(row.levels))
    lab.record(f"walk_{q}_emptied", int(row.emptied))
    lab.record(f"walk_{q}_worst", row.worst)
    lab.record(f"walk_{q}_cost_usd", (row.vwap - mid) * row.filled)   # dollars paid above the mid
    lab.record(f"walk_{q}_notional", row.notional)
# The effective spread of a trade is twice its distance from the mid (Session 14.2 owns the concept).
lab.record("walk_10000_eff_spread_bps", 2 * walk.loc[10_000, "cost_bps"])

# Accounting checks (identities, not statistics):
assert np.allclose(walk.cost_bps, walk.half_spread_bps + walk.depth_bps)
assert (walk.vwap >= best_ask).all()
assert walk.cost_bps.is_monotonic_increasing                         # bigger orders never cost less
small = walk.index <= int(asks["size"][0])                            # fits inside the best level
assert np.allclose(walk.loc[small, "cost_bps"], bps((best_ask - mid) / mid))

lab.chart("book_walk", charts.column_chart(
    [f"{q:,}" for q in ORDER_SIZES], walk.cost_bps.tolist(),
    title="Cost of a market buy against the mid, by order size in shares (synthetic book)",
    y_fmt=charts.fmt_num(1, suffix=" bp")))

# %% [markdown]
# ## 2 · Stop orders and gap risk — Monte Carlo
#
# You buy at 100 and place a **sell stop** at 95: if the price trades at or below 95, the stop
# becomes a market order. Between trades the price moves in small steps, so an intraday trigger
# fills close to 95. But news arrives overnight, and the next morning's first price can be far below
# the stop: the market order fills there. A **stop-limit** (stop 95, limit 93) caps the price you
# accept — and after a big enough gap it cannot sell at the open: it rests at 93 while the stock
# trades below, and may fill later or expire. We measure only whether it could fill at once.
#
# The model: log price with zero drift and 30% annual volatility, checked every 5 minutes (78
# checks in a 6.5-hour session). Overnight, news arrives on average 12 times a year; each news gap is
# up or down with equal odds, with an exponentially distributed size of mean 3% (log terms).
# All parameters are illustrative. Fills ignore the spread, which would add a little more.
#
# **Theory.** Two results let us check the simulation:
#
# * **Gaps.** An exponential gap is memoryless: given that it crossed the stop, the distance it
#   overshot is again exponential with the same mean. So the mean log overshoot is exactly GAP_MEAN,
#   and a stop-limit whose limit sits d below the stop fails to fill with probability exp(−d/GAP_MEAN).
# * **Intraday.** A Gaussian random walk with step s overshoots a distant barrier by about
#   β·s on average, β = −ζ(1/2)/√(2π) ≈ 0.5826 — Siegmund's (1985) overshoot result, which is the
#   Broadie–Glasserman–Kou (1997) discrete-monitoring correction used in Session 2.4.

# %%
SIGMA = 0.30                 # annual volatility of the stock
STEPS_PER_DAY = 78           # 5-minute checks in a 6.5-hour trading session
HORIZON_DAYS = 63            # one quarter of trading days
N_PATHS = 20_000
NEWS_PER_YEAR = 12           # overnight news gaps a year, on average
GAP_MEAN = 0.03              # mean size of a news gap, log terms (exponential)
ENTRY = 100.0
STOP_PCT = 0.05              # sell stop 5% below entry
LIMIT_PCT = 0.07             # the stop-limit variant: limit 7% below entry
MIN_TRIGGERS = 500           # below this many triggers of a kind, skip its statistical checks

BETA_BGK = float(-zeta(0.5) / np.sqrt(2 * np.pi))       # Broadie–Glasserman–Kou constant, ≈ 0.5826
step_sd = SIGMA / np.sqrt(252 * STEPS_PER_DAY)        # sd of one 5-minute log move
b_stop = np.log(1 - STOP_PCT)                         # stop level, log price relative to entry
b_limit = np.log(1 - LIMIT_PCT)
d_limit = b_stop - b_limit                            # log distance from stop down to limit


def simulate_stops(rng: np.random.Generator) -> pd.DataFrame:
    """Simulate N_PATHS price paths until each one's sell stop triggers (or the horizon ends).

    Returns one row per triggered path: the log price at the trigger (where the resulting market
    order fills) and whether it came from an intraday move (kind 1) or an overnight gap (kind 2).
    """
    x = np.zeros(N_PATHS)                   # log price relative to entry
    alive = np.ones(N_PATHS, dtype=bool)
    trig_x = np.full(N_PATHS, np.nan)
    kind = np.zeros(N_PATHS, dtype=int)
    for day in range(HORIZON_DAYS):
        if day > 0:                          # overnight: news gaps up or down
            news = rng.random(N_PATHS) < NEWS_PER_YEAR / 252
            sign = np.where(rng.random(N_PATHS) < 0.5, 1.0, -1.0)
            x = x + news * sign * rng.exponential(GAP_MEAN, N_PATHS)
            hit = alive & (x <= b_stop)      # the stop triggers at the open, at the gapped price
            trig_x[hit], kind[hit] = x[hit], 2
            alive &= ~hit
        path = x + np.cumsum(rng.standard_normal((STEPS_PER_DAY, N_PATHS)) * step_sd, axis=0)
        below = path <= b_stop
        hit = alive & below.any(axis=0)
        first = below.argmax(axis=0)         # the first 5-minute check at or below the stop
        idx = np.flatnonzero(hit)
        trig_x[idx], kind[idx] = path[first[idx], idx], 1
        alive &= ~hit
        x = path[-1]
    out = pd.DataFrame({"x": trig_x, "kind": kind})
    return out[out.kind > 0].reset_index(drop=True)


trig = simulate_stops(rng_stop)
trig["overshoot"] = b_stop - trig.x                         # log distance below the stop
trig["shortfall"] = 1 - np.exp(trig.x - b_stop)             # fill below the stop price, as a fraction
trig["loss"] = 1 - np.exp(trig.x)                           # loss from the entry price
trig["stoplimit_fills"] = trig.x >= b_limit                 # the limit order can execute at once
gap, intra = trig[trig.kind == 2], trig[trig.kind == 1]
n_trig, n_gap, n_intra = len(trig), len(gap), len(intra)
print(f"triggered {n_trig:,} of {N_PATHS:,} paths within {HORIZON_DAYS} days: "
      f"{n_intra:,} intraday, {n_gap:,} by an overnight gap")

lab.record("stop_n_paths", N_PATHS)
lab.record("stop_horizon_days", HORIZON_DAYS)
lab.record("stop_sigma", SIGMA)
lab.record("stop_news_per_year", NEWS_PER_YEAR)
lab.record("stop_gap_mean", GAP_MEAN)
lab.record("stop_entry", ENTRY)
lab.record("stop_price", ENTRY * (1 - STOP_PCT))
lab.record("stop_limit_price", ENTRY * (1 - LIMIT_PCT))
lab.record("stop_pct", STOP_PCT)
lab.record("stop_steps_per_day", STEPS_PER_DAY)
lab.record("stop_p_trigger", n_trig / N_PATHS)
lab.record("stop_share_gap", n_gap / n_trig)
lab.record("stop_n_gap", n_gap)
lab.record("stop_share_shortfall_from_gaps", float(gap.shortfall.sum() / trig.shortfall.sum()))
lab.record("stop_p_nofill_all", float(1 - trig.stoplimit_fills.mean()))
lab.record("stop_beta", BETA_BGK)
lab.record("stop_step_sd", step_sd)

if n_intra >= MIN_TRIGGERS:
    theory_intra = BETA_BGK * step_sd                       # log overshoot, Siegmund's approximation
    se_intra = intra.overshoot.std(ddof=1) / np.sqrt(n_intra)
    lab.record("stop_intra_shortfall", float(intra.shortfall.mean()))
    lab.record("stop_intra_overshoot", float(intra.overshoot.mean()))
    lab.record("stop_intra_theory", theory_intra)
    lab.record("stop_intra_z", float((intra.overshoot.mean() - theory_intra) / se_intra))
    lab.record("stop_intra_shortfall_p95", float(intra.shortfall.quantile(0.95)))
    lab.record("stop_intra_loss", float(intra.loss.mean()))
    lab.record("stop_share_intra", n_intra / n_trig)
    print(f"intraday: mean overshoot {intra.overshoot.mean():.5f} vs β·s = {theory_intra:.5f} "
          f"(z = {(intra.overshoot.mean() - theory_intra) / se_intra:+.2f})")
    # Siegmund's formula is asymptotic (a barrier many steps away), so allow 4 standard errors.
    assert abs(intra.overshoot.mean() - theory_intra) < 4 * se_intra
    assert intra.stoplimit_fills.all(), "an intraday trigger lands a fraction of a percent below the stop"

if n_gap >= MIN_TRIGGERS:
    se_gap = gap.overshoot.std(ddof=1) / np.sqrt(n_gap)
    p_nofill_theory = np.exp(-d_limit / GAP_MEAN)
    p_nofill = float(1 - gap.stoplimit_fills.mean())
    se_nofill = np.sqrt(p_nofill_theory * (1 - p_nofill_theory) / n_gap)
    lab.record("stop_gap_overshoot", float(gap.overshoot.mean()))
    lab.record("stop_gap_shortfall", float(gap.shortfall.mean()))
    lab.record("stop_gap_shortfall_theory", GAP_MEAN / (1 + GAP_MEAN))   # E[1 − e^(−O)], O ~ Exp(mean η)
    lab.record("stop_gap_shortfall_p95", float(gap.shortfall.quantile(0.95)))
    lab.record("stop_gap_shortfall_max", float(gap.shortfall.max()))
    lab.record("stop_gap_loss", float(gap.loss.mean()))
    lab.record("stop_gap_loss_p95", float(gap.loss.quantile(0.95)))
    lab.record("stop_p_nofill_gap", p_nofill)
    lab.record("stop_p_nofill_gap_theory", float(p_nofill_theory))
    lab.record("stop_d_limit", float(d_limit))
    print(f"gaps: mean overshoot {gap.overshoot.mean():.4f} vs {GAP_MEAN} · stop-limit unfilled "
          f"{p_nofill:.1%} vs exp(−d/η) = {p_nofill_theory:.1%}")
    # Memorylessness makes both of these exact; the tolerance is 4 Monte Carlo standard errors.
    assert abs(gap.overshoot.mean() - GAP_MEAN) < 4 * se_gap
    assert abs(p_nofill - p_nofill_theory) < 4 * se_nofill
    # Gap fills are far worse than intraday fills: the point of the experiment.
    if n_intra >= MIN_TRIGGERS:
        assert gap.shortfall.mean() > 10 * intra.shortfall.mean()
        lab.record("stop_gap_vs_intra", float(gap.shortfall.mean() / intra.shortfall.mean()))

    # Chart: where the stop filled after a gap, with the exact theoretical density on top.
    y = -gap.shortfall.to_numpy()                            # fill/stop − 1, negative
    grid = np.linspace(y.min(), 0, 400)
    dens = (1 / GAP_MEAN) * (1 + grid) ** (1 / GAP_MEAN - 1)  # density of e^(−O) − 1, O ~ Exp(mean η)
    lab.chart("stop_gap_fills", charts.histogram(
        y, bins=50, x_fmt=charts.fmt_pct(1),
        title=f"Fill vs stop price after an overnight gap ({n_gap:,} paths) — coral: stop-limit unfilled",
        density_overlay=(grid, dens), overlay_label="theory (memoryless gap)",
        tail_below=float(np.exp(b_limit - b_stop) - 1), tail_label=""))   # label in the title: no collision

# %% [markdown]
# ## 3 · Why clearing houses net: bilateral vs multilateral netting
#
# A day of trading in one stock among 20 clearing members: 50,000 trades, each between a random
# buyer and a different random seller, in round lots with a geometric size distribution (mean 500
# shares). Without netting every trade settles on its own (gross). Netting pair by pair (bilateral)
# leaves one obligation per pair of firms. A central counterparty nets each member against
# everyone at once (multilateral): one number per member per security.
#
# **Theory.** A member's net position is a sum of many ±sizes, so by the central limit theorem its
# expected absolute value is √(2/π) × its standard deviation. Working it through, the share of gross
# volume that still has to move is
#
#     multilateral ≈ √(N·E[s²] / (π·M)) / E[s]          bilateral ≈ √(N(N−1)·E[s²] / (π·M)) / E[s]
#
# for N members, M trades and trade size s. Netted obligations grow like √M while gross grows like M.

# %%
N_MEMBERS = 20
TRADES_PER_DAY = 50_000
N_DAYS = 200                 # independent simulated days, to average over
P_GEOM = 0.2                 # trade size = 100 × Geometric(0.2) shares: mean 500


def net_one_day(rng: np.random.Generator) -> tuple[float, float, float]:
    """Gross shares traded, and shares still to deliver after bilateral and multilateral netting."""
    s = LOT * rng.geometric(P_GEOM, TRADES_PER_DAY)
    buyer = rng.integers(0, N_MEMBERS, TRADES_PER_DAY)
    seller = (buyer + rng.integers(1, N_MEMBERS, TRADES_PER_DAY)) % N_MEMBERS   # never the buyer
    gross = float(s.sum())
    member_net = (np.bincount(buyer, s, N_MEMBERS) - np.bincount(seller, s, N_MEMBERS))
    multilateral = float(np.abs(member_net).sum() / 2)       # every share delivered is also received
    lo, hi = np.minimum(buyer, seller), np.maximum(buyer, seller)
    pair = lo * N_MEMBERS + hi
    signed = np.where(buyer == lo, s, -s)                    # + when the lower-numbered firm buys
    bilateral = float(np.abs(np.bincount(pair, signed, N_MEMBERS * N_MEMBERS)).sum())
    return gross, bilateral, multilateral


days = pd.DataFrame([net_one_day(rng_net) for _ in range(N_DAYS)], columns=["gross", "bilateral", "multilateral"])
ratio_bi = days.bilateral / days.gross
ratio_multi = days.multilateral / days.gross
e_s = LOT / P_GEOM                                           # E[s]
e_s2 = LOT ** 2 * (2 - P_GEOM) / P_GEOM ** 2                 # E[s²] of 100 × Geometric(p)
theory_multi = np.sqrt(N_MEMBERS * e_s2 / (np.pi * TRADES_PER_DAY)) / e_s
theory_bi = np.sqrt(N_MEMBERS * (N_MEMBERS - 1) * e_s2 / (np.pi * TRADES_PER_DAY)) / e_s
print(f"left to settle: bilateral {ratio_bi.mean():.2%} (theory {theory_bi:.2%}), "
      f"multilateral {ratio_multi.mean():.2%} (theory {theory_multi:.2%})")

lab.record("net_members", N_MEMBERS)
lab.record("net_trades", TRADES_PER_DAY)
lab.record("net_days", N_DAYS)
lab.record("net_total_trades", N_DAYS * TRADES_PER_DAY)
lab.record("net_mean_size", e_s)
lab.record("net_pairs", N_MEMBERS * (N_MEMBERS - 1) // 2)
lab.record("net_gross_shares", float(days.gross.mean()))
lab.record("net_bi_ratio", float(ratio_bi.mean()))
lab.record("net_multi_ratio", float(ratio_multi.mean()))
lab.record("net_bi_theory", float(theory_bi))
lab.record("net_multi_theory", float(theory_multi))
lab.record("net_bi_reduction", float(1 - ratio_bi.mean()))
lab.record("net_multi_reduction", float(1 - ratio_multi.mean()))
lab.record("net_bi_over_multi", float(ratio_bi.mean() / ratio_multi.mean()))
# Monte Carlo averages over N_DAYS days against theory, at 4 standard errors. The formulas lean on
# the central limit theorem, so check them only where each netting set has 100+ trades a day.
CLT_MIN_TRADES = 100
if 2 * TRADES_PER_DAY / N_MEMBERS >= CLT_MIN_TRADES:
    assert abs(ratio_multi.mean() - theory_multi) < 4 * ratio_multi.std(ddof=1) / np.sqrt(N_DAYS)
if TRADES_PER_DAY / (N_MEMBERS * (N_MEMBERS - 1) / 2) >= CLT_MIN_TRADES:
    assert abs(ratio_bi.mean() - theory_bi) < 4 * ratio_bi.std(ddof=1) / np.sqrt(N_DAYS)
else:
    print("too few trades per pair for the bilateral formula's normal approximation — compare, don't assert")
assert (days.multilateral <= days.bilateral + 1e-9).all()    # multilateral netting never nets less

# %% [markdown]
# ## 4 · The bill: a round trip, component by component
#
# Buy 2,000 shares with a market order and later sell them the same way, in two stocks at the same
# $50 price: a liquid large cap (the book of section 1) and a thin small cap (wider spread, a
# fifth of the depth, levels three ticks apart). The mid is held fixed, so we measure pure costs.
# Costs are in basis points of the $100,000 notional.
#
# **Yearly drag.** A strategy that is fully invested and pays a round-trip cost c on every one of n
# round trips a year multiplies its wealth by (1 − c)^n, a drag of 1 − (1 − c)^n. For small n·c this
# is about n·c; for the thin stock the linear shortcut overstates it by several percentage points.
#
# | Component | Rate used here | Charged on |
# | --- | --- | --- |
# | Commission | $0.0035 a share (illustrative per-share broker rate) | both sides |
# | Exchange fee for taking liquidity | $0.0030 a share (the Reg NMS access-fee cap, as of 2026) | both sides |
# | Clearing fee | $0.0002 a share (illustrative) | both sides |
# | SEC Section 31 fee | $20.60 per $1 million of proceeds (from 4 April 2026) | sales only |
# | FINRA Trading Activity Fee | $0.000195 a share, max $9.79 a trade (2026) | sales only |
# | Spread and depth | from walking the book | both sides |

# %%
QTY = 2_000
COMMISSION = 0.0035          # $ per share, illustrative
TAKER_FEE = 0.0030           # $ per share, Reg NMS Rule 610 cap for protected quotes ≥ $1 (2026)
CLEARING_FEE = 0.0002        # $ per share, illustrative
SEC_FEE_RATE = 20.60 / 1e6   # per dollar of sale proceeds (SEC fee rate advisory, from 2026-04-04)
FINRA_TAF, FINRA_TAF_CAP = 0.000195, 9.79   # per share sold, capped per trade (2026)
ROUND_TRIPS_PER_YEAR = 50    # a strategy that turns its positions over about weekly

thin_bids, thin_asks = make_book(rng_thin, MID, spread_ticks=10, lots_at_touch=2, growth=0.25, level_gap_ticks=3)


def round_trip(bids: pd.DataFrame, asks: pd.DataFrame, qty: int) -> dict[str, float]:
    """Dollar cost of buying `qty` shares at market and selling them at market, by component."""
    m = (bids.price[0] + asks.price[0]) / 2
    buy, sell = sweep(asks, qty), sweep(bids, qty)
    assert buy["unfilled"] == 0 and sell["unfilled"] == 0, "the round trip must fit in the book"
    proceeds = sell["vwap"] * qty
    return {
        "commission": 2 * COMMISSION * qty,
        "exchange fees": 2 * TAKER_FEE * qty,
        "clearing + regulatory": 2 * CLEARING_FEE * qty + SEC_FEE_RATE * proceeds + min(FINRA_TAF * qty, FINRA_TAF_CAP),
        "spread": (asks.price[0] - bids.price[0]) * qty,                        # half-spread on each side
        "walking the book": (buy["vwap"] - asks.price[0] + bids.price[0] - sell["vwap"]) * qty,
    }


costs = pd.DataFrame({"liquid": round_trip(bids, asks, QTY), "thin": round_trip(thin_bids, thin_asks, QTY)})
notional = QTY * MID
costs_bps = 1e4 * costs / notional
print(costs_bps.round(2))
print(costs_bps.sum().round(2).rename("total bp"))

totals = costs_bps.sum()
explicit = costs_bps.loc[["commission", "exchange fees", "clearing + regulatory"]].sum()
for case in ("liquid", "thin"):
    for comp, key in (("commission", "comm"), ("exchange fees", "exch"), ("clearing + regulatory", "clreg"),
                      ("spread", "spread"), ("walking the book", "depth")):
        lab.record(f"rt_{case}_{key}_bps", float(costs_bps.loc[comp, case]))
    lab.record(f"rt_{case}_total_bps", float(totals[case]))
    lab.record(f"rt_{case}_total_usd", float(costs[case].sum()))
    lab.record(f"rt_{case}_explicit_bps", float(explicit[case]))
    lab.record(f"rt_{case}_implicit_share", float(1 - explicit[case] / totals[case]))
    c_rt = totals[case] / 1e4                                   # round-trip cost as a fraction
    lab.record(f"rt_{case}_annual_drag", float(ROUND_TRIPS_PER_YEAR * c_rt))          # linear, n·c
    lab.record(f"rt_{case}_annual_drag_compound", float(1 - (1 - c_rt) ** ROUND_TRIPS_PER_YEAR))
lab.record("rt_qty", QTY)
lab.record("rt_price", MID)
lab.record("rt_commission_rate", COMMISSION)
lab.record("rt_taker_rate", TAKER_FEE)
lab.record("rt_clearing_rate", CLEARING_FEE)
lab.record("rt_sec_rate_per_million", SEC_FEE_RATE * 1e6)
lab.record("rt_taf_rate", FINRA_TAF)
lab.record("rt_taf_cap", FINRA_TAF_CAP)
lab.record("rt_notional", notional)
lab.record("rt_trips_per_year", ROUND_TRIPS_PER_YEAR)
lab.record("rt_thin_spread_bps", bps((thin_asks.price[0] - thin_bids.price[0]) / MID))
lab.record("rt_thin_depth_shares", int(thin_asks["size"].sum()))
lab.record("rt_thin_over_liquid", float(totals["thin"] / totals["liquid"]))
lab.record("rt_sec_fee_usd", float(SEC_FEE_RATE * notional))
lab.record("rt_taf_usd", float(min(FINRA_TAF * QTY, FINRA_TAF_CAP)))
lab.record("rt_commission_usd", float(COMMISSION * QTY))
lab.record("rt_taker_usd", float(TAKER_FEE * QTY))

# Checks: explicit fees are per share, so identical for both stocks; the spread component is the
# quoted spread in bp; the thin stock costs more in total.
assert np.isclose(explicit["liquid"], explicit["thin"], rtol=1e-3)
assert np.isclose(costs_bps.loc["spread", "liquid"], lab.results["spread_bps"])
assert totals["thin"] > totals["liquid"]

lab.chart("round_trip", charts.grouped_column_chart(
    list(costs_bps.index),
    [("liquid large cap", costs_bps["liquid"].tolist(), "strategy"),
     ("thin small cap", costs_bps["thin"].tolist(), "alt1")],
    title=f"One round trip of {QTY:,} shares at ${MID:,.0f}: cost by component, in basis points",
    y_fmt=charts.fmt_num(1), value_labels=True))

# %%
lab.save()
