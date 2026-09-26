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
# # 1.1 · What Quants Actually Do — companion lab
#
# **Quant Notebook** · Unit 1 · Session 1 · [Read the session](https://georgevrs.github.io/quant-notebook/unit01-quant-landscape/session01-what-quants-do.html)
#
# Two toy quant businesses trade the **same** synthetic market with the **same** capital:
#
# * **A market maker** quotes both sides all day, earns a couple of basis points on thousands of
#   small client trades, and loses to the minority of clients who know where the price is going
#   (adverse selection).
# * **A trend follower** makes a handful of large directional bets a year and holds them for months.
#
# Same market, same money, completely different return shapes — which is the point of the session:
# the way a firm makes money decides what its P&L looks like, what can kill it, and what its quants
# are judged on. Then we do the fee arithmetic that decides who keeps the profits.
#
# **This is a toy.** Real market making (Session 14.5) and trend following (Session 8.1) are far
# richer; we keep only the mechanism that shapes each business's returns. Synthetic excess returns,
# risk-free rate 0 throughout.
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
import numpy as np
import pandas as pd
from scipy.stats import norm, skew

import quantnb as qn
from quantnb import charts
from quantnb.returns import PERIODS_PER_YEAR, sharpe_ratio

lab = qn.Lab("1.1")  # seeds the random generators: every run gives the same numbers
# One independent random stream per experiment, so editing one section never changes another's numbers.
rng_market, rng_flow, rng_worlds = lab.rng.spawn(3)

# %% [markdown]
# ## 1 · One synthetic market, ten years
#
# One liquid instrument, priced once a day for ten years. Each day's return has three parts:
#
# * a slow, hidden **drift** that wanders up and down (an AR(1) process) — the trends a trend
#   follower hopes to catch;
# * ordinary **noise**;
# * a **news jump** whose size depends on how much news arrived that day. News intensity is
#   random: most days are quiet, a few are very loud.
#
# There is no built-in risk premium: the drift averages zero, so holding the market earns nothing
# on average. We know the truth here, which is what makes a synthetic market the right place to
# compare businesses.

# %%
YEARS = 10
DAYS = PERIODS_PER_YEAR * YEARS
NOISE_VOL = 0.008       # ordinary daily noise: 0.8% a day
DRIFT_PERSIST = 0.995   # AR(1) persistence of the hidden drift (half-life ≈ 140 days)
DRIFT_VOL = 0.0010      # typical size of the hidden drift: 10 bp a day (≈ 25% a year)
NEWS_VOL = 0.005        # a day with average news moves the price by about 0.5% on top


def simulate_market(rng: np.random.Generator, days: int = DAYS) -> pd.DataFrame:
    """Daily returns = hidden drift + noise + news jump, indexed by elapsed years (not dates)."""
    shocks = rng.standard_normal(days)
    drift = np.empty(days)
    drift[0] = DRIFT_VOL * shocks[0]
    scale = DRIFT_VOL * np.sqrt(1 - DRIFT_PERSIST ** 2)       # keeps the drift's sd at DRIFT_VOL
    for t in range(1, days):
        drift[t] = DRIFT_PERSIST * drift[t - 1] + scale * shocks[t]
    news = rng.exponential(1.0, days)                        # news intensity: mean 1, long right tail
    jump = NEWS_VOL * news * rng.standard_normal(days)       # loud news days move the price more
    ret = drift + NOISE_VOL * rng.standard_normal(days) + jump
    years = np.arange(1, days + 1) / PERIODS_PER_YEAR        # simulated time is elapsed time
    return pd.DataFrame({"ret": ret, "drift": drift, "news": news, "jump": jump},
                        index=pd.Index(years, name="years"))


market = simulate_market(rng_market)
vol = market["ret"].std() * np.sqrt(PERIODS_PER_YEAR)
multiple = float(np.prod(1 + market["ret"]))
print(f"{DAYS:,} trading days · annualised volatility {vol:.1%} · $1 held all along became ${multiple:.2f}")
lab.record("years", YEARS)
lab.record("market_vol", vol)
lab.record("market_multiple", multiple)   # what $1 held all along became (no risk premium in the toy)

# %% [markdown]
# ## 2 · Business A: an electronic market maker
#
# The market maker (MM) quotes a bid 2 bp below the mid-price and an offer 2 bp above it, and
# collects a small exchange rebate for providing liquidity. Every client trade is $10,000.
#
# * An **uninformed** client trades for their own reasons. The MM holds the position for a moment,
#   the price wiggles randomly, and it lays the risk off. On average it keeps the half-spread.
# * An **informed** client knows where the price is about to go and trades in that direction. The MM
#   loses the move and keeps only the half-spread: P&L = half-spread − |move|.
# * On **news** days, extra informed traders arrive — more of them the louder the news — and pick
#   the MM off before its quotes catch up with the jump.
#
# To compare different shares of informed clients fairly (section 5), we draw the random client
# flow once and compute P&L for any informed share from the same draws (common random numbers).

# %%
CAPITAL = 1_000_000        # both businesses get the same $1m of capital
TRADES_PER_DAY = 400       # everyday client trades (Poisson mean)
TRADE_NOTIONAL = 10_000    # $ per client trade
HALF_SPREAD = 2.0e-4       # 2 bp: buy 2 bp below mid, sell 2 bp above
REBATE = 0.2e-4            # 0.2 bp exchange rebate per trade for providing liquidity
HOLD_MOVE_VOL = 5e-4       # how far the mid wiggles while the MM holds an uninformed trade: 5 bp
INFORMED_MOVE_VOL = 10e-4  # the move an informed client knows is coming: 10 bp typical
INFORMED_SHARE = 0.05      # 5% of everyday clients are informed
NEWS_TRADES = 3            # extra informed trades on a day with average news (more on loud days)


def draw_client_flow(market: pd.DataFrame, rng: np.random.Generator) -> dict:
    """All the randomness of the client flow, drawn once: who trades, which side, what moves."""
    days = len(market)
    n = rng.poisson(TRADES_PER_DAY, days)
    total = int(n.sum())
    return {
        "day": np.repeat(np.arange(days), n),           # which day each trade belongs to
        "u": rng.random(total),                          # compared with the informed share
        "side": rng.choice([-1.0, 1.0], total),          # client buys (+1) or sells (−1)
        "hold_move": HOLD_MOVE_VOL * rng.standard_normal(total),
        "informed_move": np.abs(INFORMED_MOVE_VOL * rng.standard_normal(total)),
        "news_trades": rng.poisson(NEWS_TRADES * market["news"].to_numpy()),
        "days": days,
    }


def market_maker(flow: dict, market: pd.DataFrame, informed_share: float = INFORMED_SHARE) -> dict:
    """Daily P&L of the market maker, plus per-trade P&L for the everyday flow.

    uninformed trade:  (half-spread + rebate) − side × move       (move independent of side)
    informed trade:    (half-spread + rebate) − |move|             (they trade the move's direction)
    news trade:        (half-spread + rebate) − |news jump|        (picked off before quotes adjust)
    """
    edge = HALF_SPREAD + REBATE
    informed = flow["u"] < informed_share
    per_trade = np.where(informed, edge - flow["informed_move"], edge - flow["side"] * flow["hold_move"])
    per_trade = per_trade * TRADE_NOTIONAL
    daily = np.bincount(flow["day"], weights=per_trade, minlength=flow["days"])
    news_loss = flow["news_trades"] * (edge - np.abs(market["jump"].to_numpy())) * TRADE_NOTIONAL
    daily = daily + news_loss
    return {"daily": pd.Series(daily, index=market.index), "per_trade": per_trade,
            "n_trades": len(per_trade) + int(flow["news_trades"].sum())}


flow = draw_client_flow(market, rng_flow)
mm = market_maker(flow, market)
print(f"market maker: {mm['n_trades']:,} trades · mean daily P&L ${mm['daily'].mean():,.0f} · "
      f"per-trade hit rate {(mm['per_trade'] > 0).mean():.1%}")

# %% [markdown]
# ## 3 · Business B: a trend follower
#
# The trend follower is long when a fast moving average of the price is above a slow one and short
# otherwise — the EWMA crossover you will build properly in Session 8.1. It scales the position so
# that its expected volatility is 15% a year of capital (never more than 3× capital), and pays 5 bp
# on every dollar it trades. Decisions use only data up to yesterday's close.

# %%
FAST_SPAN, SLOW_SPAN = 20, 100   # EWMA spans in trading days
VOL_SPAN = 60                    # span of the EWMA volatility estimate
TARGET_VOL = 0.15                # target annual volatility of the P&L, as a share of capital
MAX_LEVERAGE = 3.0
COST = 5e-4                      # 5 bp per dollar traded


def trend_follower(market: pd.DataFrame) -> dict:
    """EWMA-crossover trend follower with volatility targeting. Returns daily P&L and positions."""
    log_price = np.log1p(market["ret"]).cumsum()
    direction = np.sign(log_price.ewm(span=FAST_SPAN).mean() - log_price.ewm(span=SLOW_SPAN).mean())
    vol_est = market["ret"].ewm(span=VOL_SPAN).std() * np.sqrt(PERIODS_PER_YEAR)
    weight = (direction * TARGET_VOL / vol_est).clip(-MAX_LEVERAGE, MAX_LEVERAGE).fillna(0.0)
    held = weight.shift(1).fillna(0.0)                      # decided at yesterday's close
    costs = COST * held.diff().abs().fillna(held.abs()) * CAPITAL
    daily = held * market["ret"] * CAPITAL - costs
    return {"daily": daily, "held": held}


def trend_trades(daily: pd.Series, held: pd.Series) -> pd.Series:
    """Split the P&L into 'trades': runs of days with the same direction (long or short)."""
    side = np.sign(held)
    trade_id = (side != side.shift()).cumsum()
    return daily[side != 0].groupby(trade_id[side != 0]).sum()


tf = trend_follower(market)
tf_trades = trend_trades(tf["daily"], tf["held"])
print(f"trend follower: {len(tf_trades)} trades · mean daily P&L ${tf['daily'].mean():,.0f} · "
      f"trade hit rate {(tf_trades > 0).mean():.0%}")

# %% [markdown]
# ## 4 · Same market, same capital, two return shapes
#
# P&L is not reinvested (each business keeps trading $1m), so we measure everything on daily P&L
# in dollars and as a share of capital. The toy has no interest on cash, so the Sharpe ratio uses a
# risk-free rate of zero. Durations are in trading days.

# %%
MONTH = 21  # trading days in a month


def longest_drawdown_days(daily: pd.Series) -> int:
    """Longest stretch of trading days spent below a previous peak of cumulative P&L."""
    equity = daily.cumsum()
    under = (equity < equity.cummax()).to_numpy()
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def business_stats(daily: pd.Series, trade_pnl: np.ndarray | pd.Series) -> dict:
    """The numbers a manager would look at first: return, risk, shape and hit rates."""
    equity = daily.cumsum()
    monthly = daily.groupby(np.arange(len(daily)) // MONTH).sum()
    trade_pnl = np.asarray(trade_pnl)
    wins, losses = trade_pnl[trade_pnl > 0], trade_pnl[trade_pnl < 0]
    return {
        "annual_pnl": float(daily.mean() * PERIODS_PER_YEAR),
        "annual_return": float(daily.mean() * PERIODS_PER_YEAR / CAPITAL),
        "sharpe": sharpe_ratio(daily / CAPITAL),                   # rf = 0 in the toy world
        "pct_up_days": float((daily > 0).mean()),
        "skew_daily": float(skew(daily)),
        "skew_monthly": float(skew(monthly)),
        "skew_trades": float(skew(trade_pnl)),                     # shape of the P&L per trade
        "max_dd": float((equity - equity.cummax()).min() / CAPITAL),
        "longest_dd_days": longest_drawdown_days(daily),
        "worst_day": float(daily.min()),
        "worst_day_in_days": float(-daily.min() / daily.mean()),   # days of average profit it erased
        "trades_per_year": len(trade_pnl) / YEARS,
        "trade_hit_rate": float((trade_pnl > 0).mean()),
        "win_loss_ratio": float(wins.mean() / -losses.mean()) if len(losses) else float("nan"),
    }


stats = pd.DataFrame({"market maker": business_stats(mm["daily"], mm["per_trade"]),
                      "trend follower": business_stats(tf["daily"], tf_trades)})
print(stats.round(3).to_string())
for key, col in (("mm", "market maker"), ("tf", "trend follower")):
    for name, value in stats[col].items():
        lab.record(f"{key}_{name}", float(value))
lab.record("mm_total_pnl", float(mm["daily"].sum()))
lab.record("tf_total_pnl", float(tf["daily"].sum()))
corr = float(np.corrcoef(mm["daily"], tf["daily"])[0, 1])
lab.record("corr_mm_tf", corr)
lab.record("capital", CAPITAL)
lab.record("mm_trades_per_day", TRADES_PER_DAY)
lab.record("mm_trade_notional", TRADE_NOTIONAL)
lab.record("mm_half_spread_bp", HALF_SPREAD * 1e4)
lab.record("mm_rebate_bp", REBATE * 1e4)
lab.record("mm_edge_bp", (HALF_SPREAD + REBATE) * 1e4)
lab.record("mm_informed_share", INFORMED_SHARE)
lab.record("tf_target_vol", TARGET_VOL)
lab.record("tf_cost_bp", COST * 1e4)

# Top-heavy: what share of the trend follower's gross gains came from its three best trades?
gains = np.sort(tf_trades[tf_trades > 0].to_numpy())[::-1]
top3_share = float(gains[:3].sum() / gains.sum())
lab.record("tf_top3_share", top3_share)
print(f"three best trend trades = {top3_share:.0%} of all gross gains · correlation MM vs TF {corr:.2f}")

# %% [markdown]
# ## 5 · How thin is the market maker's edge?
#
# Keep the client flow fixed and change only the share of everyday clients who are informed.
# In words: each everyday trade earns the half-spread plus rebate (h + b), each informed one gives
# back the move it knew about, E|Δ| on average, and news days cost a roughly fixed amount ℓ per
# everyday trade. So
#
#     E[P&L per trade] = (h + b) − q·E|Δ| − ℓ,        break-even q* = (h + b − ℓ) / E|Δ|

# %%
INFORMED_SHARES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
EDGE = HALF_SPREAD + REBATE
EXP_ABS_INFORMED = INFORMED_MOVE_VOL * np.sqrt(2 / np.pi)     # E|move| of a normal move
sweep = pd.Series({s: market_maker(flow, market, s)["daily"].mean() * PERIODS_PER_YEAR
                   for s in INFORMED_SHARES})
print((sweep / 1e3).round(1).rename("annual P&L, $k"))

trades_per_day = len(flow["u"]) / flow["days"]
news_daily = float((flow["news_trades"] * (EDGE - np.abs(market["jump"].to_numpy())) * TRADE_NOTIONAL).mean())
news_loss_per_trade = -news_daily / (trades_per_day * TRADE_NOTIONAL)     # ℓ, as a return per everyday trade
breakeven_theory = (EDGE - news_loss_per_trade) / EXP_ABS_INFORMED
# measured break-even: linear interpolation of the sweep's zero crossing
s_arr, p_arr = np.array(INFORMED_SHARES), sweep.to_numpy()
k = int(np.argmax(p_arr < 0))
breakeven_sim = float(s_arr[k - 1] + (0 - p_arr[k - 1]) * (s_arr[k] - s_arr[k - 1]) / (p_arr[k] - p_arr[k - 1]))
print(f"ℓ = {news_loss_per_trade * 1e4:.2f} bp · break-even informed share: "
      f"theory {breakeven_theory:.1%} · simulated {breakeven_sim:.1%}")
for s, p in sweep.items():
    lab.record(f"sweep_{int(round(s * 100))}", float(p))
lab.record("sweep_25_loss", -float(sweep.iloc[-1]))                   # the 25% case as a positive loss
lab.record("news_loss_per_trade_bp", news_loss_per_trade * 1e4)
lab.record("breakeven_theory", breakeven_theory)
lab.record("breakeven_no_news", EDGE / EXP_ABS_INFORMED)             # q* with ℓ = 0
lab.record("breakeven_sim", breakeven_sim)
lab.record("exp_abs_informed_bp", EXP_ABS_INFORMED * 1e4)
# ℓ comes from the same path, so the only noise left is the per-trade draws: well under 0.02.
assert abs(breakeven_theory - breakeven_sim) < 0.02, "the formula predicts the simulated break-even"
assert sweep.iloc[0] > 0 > sweep.iloc[-1], "enough informed flow turns the MM's edge negative"

# %% [markdown]
# ## 6 · Skill or luck? 200 synthetic worlds
#
# One ten-year path is one draw. Rerun both businesses on 200 fresh markets with the same rules.
# This is where we check the shapes the session teaches — on averages over many worlds, against
# theory where we have it — rather than on one lucky or unlucky path.
#
# Theory for the market maker, per day: everyday flow earns N·(h + b − q·E|Δ|) on average, and news
# costs NEWS_TRADES·N·(h + b − 2·NEWS_VOL·√(2/π)) (news intensity is exponential, so E[news²] = 2).
# Its per-trade hit rate is (1 − q)·Φ((h + b)/σ_hold) + q·(2Φ((h + b)/σ_informed) − 1).

# %%
N_WORLDS = 200
rows = []
for _ in range(N_WORLDS):
    m = simulate_market(rng_worlds)
    w_mm = market_maker(draw_client_flow(m, rng_worlds), m)
    w_tf = trend_follower(m)
    s_mm = business_stats(w_mm["daily"], w_mm["per_trade"])
    s_tf = business_stats(w_tf["daily"], trend_trades(w_tf["daily"], w_tf["held"]))
    rows.append({**{f"mm_{k}": v for k, v in s_mm.items()}, **{f"tf_{k}": v for k, v in s_tf.items()},
                 "corr": float(np.corrcoef(w_mm["daily"], w_tf["daily"])[0, 1]),
                 "vol": m["ret"].std() * np.sqrt(PERIODS_PER_YEAR),
                 "mm_mean_daily": w_mm["daily"].mean()})
worlds = pd.DataFrame(rows)
med = worlds.median()

mm_theory_daily = (TRADES_PER_DAY * TRADE_NOTIONAL * (EDGE - INFORMED_SHARE * EXP_ABS_INFORMED)
                   + NEWS_TRADES * TRADE_NOTIONAL * (EDGE - 2 * NEWS_VOL * np.sqrt(2 / np.pi)))
mm_theory_hit = ((1 - INFORMED_SHARE) * norm.cdf(EDGE / HOLD_MOVE_VOL)
                 + INFORMED_SHARE * (2 * norm.cdf(EDGE / INFORMED_MOVE_VOL) - 1))
mc_mean, mc_se = worlds["mm_mean_daily"].mean(), worlds["mm_mean_daily"].std() / np.sqrt(N_WORLDS)
print(f"MM mean daily P&L: Monte Carlo ${mc_mean:,.0f} ± {mc_se:,.0f} vs theory ${mm_theory_daily:,.0f}")
print(f"MM per-trade hit rate: median {med['mm_trade_hit_rate']:.3f} vs theory {mm_theory_hit:.3f}")

for key in ("mm", "tf"):
    arr = worlds[f"{key}_sharpe"]
    lab.record(f"worlds_{key}_sr_median", float(arr.median()))
    lab.record(f"worlds_{key}_sr_p10", float(arr.quantile(0.10)))
    lab.record(f"worlds_{key}_sr_p90", float(arr.quantile(0.90)))
    lab.record(f"worlds_{key}_p_negative", float((arr < 0).mean()))
    # where does the path shown on the page sit among the 200 worlds?
    lab.record(f"{key}_path_pctile", int(round(100 * (arr < stats.loc["sharpe", 'market maker' if key == 'mm' else 'trend follower']).mean())))
for col, value in med.items():           # typical values: the page's table quotes these medians
    lab.record(f"worlds_{col}_median", float(value))
lab.record("mm_theory_annual", mm_theory_daily * PERIODS_PER_YEAR)
lab.record("mm_theory_hit", mm_theory_hit)
lab.record("n_worlds", N_WORLDS)
print(worlds[["mm_sharpe", "tf_sharpe", "mm_pct_up_days", "tf_pct_up_days", "mm_skew_daily",
              "tf_skew_trades", "tf_trade_hit_rate", "tf_win_loss_ratio", "corr"]].describe().round(2).T)

# The shapes the session teaches, checked on Monte Carlo averages (never on one path):
assert abs(mc_mean - mm_theory_daily) < 3 * mc_se + 1.0, "MM mean P&L matches theory within 3 standard errors"
assert abs(med["mm_trade_hit_rate"] - mm_theory_hit) < 0.01, "MM per-trade hit rate matches theory"
assert (worlds["mm_sharpe"] > 0).all() and med["mm_sharpe"] > 2 * med["tf_sharpe"], "MM: far more reliable"
assert med["mm_pct_up_days"] > med["tf_pct_up_days"] + 0.1, "MM wins most days; TF about half"
assert med["mm_skew_daily"] < 0 < med["tf_skew_trades"], "negative skew for the MM's days, positive for the TF's trades"
assert med["tf_trade_hit_rate"] < 0.5 < 1.5 < med["tf_win_loss_ratio"], "TF: lose more often, win bigger"
assert med["tf_max_dd"] < med["mm_max_dd"], "TF drawdowns are deeper (both are negative)"
assert abs(med["corr"]) < 0.1, "the two businesses are nearly uncorrelated"
assert 0.3 < med["tf_sharpe"] < 1.0, "a real but modest trend edge"
assert 0.14 < med["vol"] < 0.22, "the toy market looks like a typical liquid asset: ~15–20% vol"

# %% [markdown]
# ## 7 · Who keeps the profits? Fee arithmetic
#
# A prop firm keeps all of its P&L. A fund manages other people's money and is paid in fees, so the
# investor's return is what is left after them. For one year, starting at the high-water mark and
# with no hurdle rate:
#
#     net = gross − management fee − pass-through costs − performance fee × max(0, what is left)
#
# The fee levels are US averages reported by ICI (index equity ETFs and equity mutual funds,
# asset-weighted, 2025) and HFR's industry averages for hedge funds (1Q 2026); the pass-through
# line is an illustrative multi-manager example — real terms vary by firm and are rarely public.

# %%
FEE_MODELS = {  # name: (label, management fee, performance fee, pass-through costs), per year
    "etf": ("US index ETF", 0.0014, 0.0, 0.0),        # US index equity ETFs, asset-weighted 2025 (ICI)
    "mutual": ("US mutual fund", 0.0040, 0.0, 0.0),   # US equity mutual funds, asset-weighted 2025 (ICI)
    "hfr": ("HF average", 0.0132, 0.1578, 0.0),       # industry-average hedge fund, 1Q 2026 (HFR)
    "2and20": ("HF 2 and 20", 0.02, 0.20, 0.0),       # the classic hedge-fund headline
    "passthru": ("pod, illustrative", 0.0, 0.20, 0.06),  # costs passed through + performance fee
}
GROSS_GOOD, GROSS_LEAN = 0.10, 0.04


def investor_net(gross: float, mgmt: float, perf: float, passthrough: float = 0.0) -> float:
    """Investor's net return for one year above the high-water mark, no hurdle."""
    after_costs = gross - mgmt - passthrough
    return after_costs - perf * max(after_costs, 0.0)


fees = pd.DataFrame({name: {"net_good": investor_net(GROSS_GOOD, *terms[1:]),
                            "net_lean": investor_net(GROSS_LEAN, *terms[1:])}
                     for name, terms in FEE_MODELS.items()}).T
fees["fees_good"] = GROSS_GOOD - fees["net_good"]
fees["share_good"] = fees["net_good"] / GROSS_GOOD
print(fees.round(4))
for name, row in fees.iterrows():
    for col, value in row.items():
        lab.record(f"fee_{name}_{col}", float(value))
lab.record("gross_good", GROSS_GOOD)
lab.record("gross_lean", GROSS_LEAN)
assert np.isclose(fees.loc["2and20", "net_good"], 0.064)            # 10 − 2 − 0.2 × 8 = 6.4%
assert fees.loc["passthru", "net_lean"] < 0 < GROSS_LEAN             # the fund made money, the investor lost

# %% [markdown]
# ## 8 · Charts for the session page

# %%
usd = charts.fmt_usd_compact(1)

lab.chart("equity", charts.line_chart(
    [charts.Series("market maker", mm["daily"].cumsum() / CAPITAL, role="strategy", end_label="MM"),
     charts.Series("trend follower", tf["daily"].cumsum() / CAPITAL, role="alt1", end_label="TF")],
    title="Cumulative P&L as a share of capital, by year ($1m each, same market)",
    y_fmt=charts.fmt_pct(0), hline=0.0))

MM_BINS = np.arange(-3_000, 1_201, 100)      # $100 bins; the few days below −$3k go in the first bar
TF_BINS = np.arange(-40_000, 40_001, 2_000)  # $2,000 bins; days beyond ±$40k go in the edge bars
lab.chart("hist_mm", charts.histogram(
    np.clip(mm["daily"], MM_BINS[0], MM_BINS[-1]), bins=MM_BINS, x_fmt=usd, tail_below=0.0,
    tail_label="losing days", title="Market maker: daily P&L (days below −$3k drawn at the left edge)"))
lab.chart("hist_tf", charts.histogram(
    np.clip(tf["daily"], TF_BINS[0], TF_BINS[-1]), bins=TF_BINS, x_fmt=usd, tail_below=0.0,
    tail_label="losing days", title="Trend follower: daily P&L (days beyond ±$40k drawn at the edges)"))

lab.chart("breakeven", charts.column_chart(
    [f"{s:.0%}" for s in INFORMED_SHARES], sweep.tolist(), y_fmt=charts.fmt_usd_compact(0),
    title="Market maker's annual P&L as the share of informed clients rises"))

lab.chart("fees", charts.grouped_column_chart(
    [label for label, *_ in FEE_MODELS.values()],
    [("investor keeps", fees["net_good"].tolist(), "alt2"), ("fees & costs", fees["fees_good"].tolist(), "alt1")],
    title="A +10% gross year: what the investor keeps and what fees and costs take",
    y_fmt=charts.fmt_pct(2), value_labels=True))

# %%
lab.save()
