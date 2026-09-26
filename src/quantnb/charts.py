"""Data → inline SVG charts in the notebook design system (no JavaScript, no images).

Every function returns an `<svg viewBox="0 0 720 H">` string (never width/height, never ids),
ready to be saved with `Lab.chart(name, svg)` and injected into a session page inside
`<figure class="diagram chart">` by tools/build.py.

Colour carries meaning (validated with the dataviz palette checker against the paper surface):
  role        colour     use
  strategy    #3A56C5    the thing the session is about (categorical slot 1)
  alt1        #E8850C    a second compared series (slot 2 — always direct-labelled)
  alt2        #3F8E5A    a third compared series (slot 3)
  benchmark   #4A5A68    de-emphasised reference (grey, thinner)
  loss        #D64B2A    drawdowns, losses, the naive way — never paired with pine by colour alone
Out-of-sample periods are a shaded band with a label, not a line colour.
Text is always ink, never the series colour; lines are 2px; gridlines are solid hairlines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd

ROLE_COLOURS = {
    "strategy": "#3A56C5",
    "alt1": "#E8850C",
    "alt2": "#3F8E5A",
    "benchmark": "#4A5A68",
    "loss": "#D64B2A",
    "ink": "#1F2C38",
}
INK, INK_SOFT, GRID, AXIS, SURFACE = "#1F2C38", "#4A5A68", "#E4E8EC", "#C9CFC6", "#FBFBF8"
FONT = "font-family:'IBM Plex Sans',sans-serif"
W = 720


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _r(v: float) -> str:
    """Round a coordinate to 1 decimal so regenerated charts don't churn diffs."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _esc(s: str) -> str:
    return escape(str(s), quote=True)


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("non-finite axis range")
    if hi == lo:
        hi, lo = hi + 1, lo - 1
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    start = math.floor(lo / step + 1e-9) * step
    ticks, t = [], start
    while True:  # always end on the first tick at or above hi, so data never touches the frame
        ticks.append(round(t, 12))
        if t >= hi - 1e-12:
            break
        t += step
    return ticks


def log_ticks(lo: float, hi: float) -> list[float]:
    out = []
    for k in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** k
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
    return out or [lo, hi]


def lttb(x: np.ndarray, y: np.ndarray, threshold: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets downsampling: keeps the visual shape of long series."""
    n = len(x)
    if threshold >= n or threshold < 3:
        return x, y
    every = (n - 2) / (threshold - 2)
    idx = [0]
    a = 0
    for i in range(threshold - 2):
        lo = int(math.floor((i + 1) * every)) + 1
        hi = min(int(math.floor((i + 2) * every)) + 1, n)
        avg_x, avg_y = x[lo:hi].mean(), y[lo:hi].mean()
        r_lo = int(math.floor(i * every)) + 1
        r_hi = int(math.floor((i + 1) * every)) + 1
        area = np.abs((x[a] - avg_x) * (y[r_lo:r_hi] - y[a]) - (x[a] - x[r_lo:r_hi]) * (avg_y - y[a]))
        a = r_lo + int(np.argmax(area))
        idx.append(a)
    idx.append(n - 1)
    return x[idx], y[idx]


_EPOCH = pd.Timestamp("1970-01-01")
_DAY = pd.Timedelta(days=1)


def _to_num(x) -> tuple[np.ndarray, bool]:
    """Convert an index (datetime or numeric) to floats; report whether it was dates.

    Dates become float days since 1970-01-01 via Timedelta arithmetic — never .asi8, whose unit
    depends on the index resolution (pandas 3 infers s/ms/us/ns).
    """
    if isinstance(x, pd.DatetimeIndex) or (len(x) and isinstance(x[0], (pd.Timestamp, datetime, np.datetime64))):
        idx = pd.DatetimeIndex(x)
        return np.asarray((idx - _EPOCH) / _DAY, dtype=float), True
    return np.asarray(x, dtype=float), False


def _date_ticks(lo: float, hi: float) -> list[tuple[float, str]]:
    start, end = _EPOCH + lo * _DAY, _EPOCH + hi * _DAY
    years = (end - start).days / 365.25
    if years > 12:
        freq, fmt = "5YS", "%Y"
    elif years > 5:
        freq, fmt = "2YS", "%Y"
    elif years > 1.5:
        freq, fmt = "YS", "%Y"
    elif years > 0.5:
        freq, fmt = "QS", "%b %Y"
    else:
        freq, fmt = "MS", "%b %Y"
    ticks = pd.date_range(start.normalize(), end, freq=freq)
    return [(float((t - _EPOCH) / _DAY), t.strftime(fmt)) for t in ticks if start <= t <= end]


@dataclass
class Series:
    name: str
    values: pd.Series
    role: str = "strategy"
    area: bool = False          # 10% wash under the line
    label_end: bool = True      # direct label at the right end
    width: float | None = None
    end_label: str | None = None  # short text for the end label (defaults to the name)


@dataclass
class _Frame:
    height: int
    left: float = 64
    right: float = 118
    top: float = 44
    bottom: float = 36
    parts: list[str] = field(default_factory=list)

    @property
    def x0(self): return self.left
    @property
    def x1(self): return W - self.right
    @property
    def y0(self): return self.height - self.bottom
    @property
    def y1(self): return self.top


def _caps(s: str) -> str:
    """Uppercase ASCII letters only: Greek β/α must not become Β/Α, which read as Latin B/A."""
    return "".join(ch.upper() if ch.isascii() else ch for ch in s)


def _open(height: int, title: str) -> list[str]:
    return [f'<svg viewBox="0 0 {W} {height}" xmlns="http://www.w3.org/2000/svg" style="{FONT}" role="img" aria-label="{_esc(title)}">',
            f'  <text x="26" y="22" font-size="11" font-weight="600" fill="{INK}">{_esc(_caps(title))}</text>']


def _legend(series: list[Series], y: float = 36) -> list[str]:
    out, x = [], 26
    for s in series:
        c = ROLE_COLOURS[s.role]
        out.append(f'  <line x1="{x}" y1="{y - 4}" x2="{x + 16}" y2="{y - 4}" stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>')
        out.append(f'  <text x="{x + 22}" y="{y}" font-size="11" fill="{INK_SOFT}">{_esc(s.name)}</text>')
        x += 22 + 6.4 * len(s.name) + 22
    return out


def _y_axis(f: _Frame, ticks: list[float], scale, fmt) -> list[str]:
    out = []
    for t in ticks:
        y = scale(t)
        if f.y1 - 0.5 <= y <= f.y0 + 0.5:
            out.append(f'  <line x1="{_r(f.x0)}" y1="{_r(y)}" x2="{_r(f.x1)}" y2="{_r(y)}" stroke="{GRID}" stroke-width="1"/>')
            out.append(f'  <text x="{_r(f.x0 - 8)}" y="{_r(y + 4)}" font-size="11" text-anchor="end" fill="{INK_SOFT}" style="font-variant-numeric:tabular-nums">{_esc(fmt(t))}</text>')
    return out


def _x_axis(f: _Frame, ticks: list[tuple[float, str]], scale) -> list[str]:
    out = [f'  <line x1="{_r(f.x0)}" y1="{_r(f.y0)}" x2="{_r(f.x1)}" y2="{_r(f.y0)}" stroke="{AXIS}" stroke-width="1"/>']
    for v, label in ticks:
        x = scale(v)
        if f.x0 - 1 <= x <= f.x1 + 1:
            out.append(f'  <line x1="{_r(x)}" y1="{_r(f.y0)}" x2="{_r(x)}" y2="{_r(f.y0 + 4)}" stroke="{AXIS}" stroke-width="1"/>')
            out.append(f'  <text x="{_r(x)}" y="{_r(f.y0 + 18)}" font-size="11" text-anchor="middle" fill="{INK_SOFT}">{_esc(label)}</text>')
    return out


def _no_neg_zero(s: str) -> str:
    """Drop the sign of a negative zero and use a true minus sign (U+2212) for negatives."""
    if s.startswith("-") and not any(ch in "123456789" for ch in s):
        return s[1:]
    return "−" + s[1:] if s.startswith("-") else s


def fmt_pct(decimals: int = 0):
    return lambda v: _no_neg_zero(f"{v * 100:.{decimals}f}%")


def fmt_num(decimals: int = 0, prefix: str = "", suffix: str = ""):
    return lambda v: _no_neg_zero(f"{prefix}{v:,.{decimals}f}{suffix}")


# ---------------------------------------------------------------------------------------------
# line chart
# ---------------------------------------------------------------------------------------------
def line_chart(series: list[Series], *, title: str, y_fmt=fmt_num(0), height: int = 300,
               logy: bool = False, hline: float | None = None, hline_label: str | None = None,
               bands: list[tuple] | None = None, y_min: float | None = None, y_max: float | None = None,
               max_points: int = 400) -> str:
    """Lines over a shared x (dates or numbers). ≤ 3 highlighted series + optional benchmark.

    bands: [(x_start, x_end, label)] shaded regions, e.g. the out-of-sample period.
    """
    if not series:
        raise ValueError("no series")
    f = _Frame(height=height, top=52 if len(series) > 1 else 40)
    xs, ys = [], []
    for s in series:
        x, is_date = _to_num(s.values.index)
        y = s.values.to_numpy(dtype=float)
        ok = np.isfinite(y)
        x, y = lttb(x[ok], y[ok], max_points)
        xs.append(x)
        ys.append(y)
    all_x, all_y = np.concatenate(xs), np.concatenate(ys)
    xlo, xhi = float(all_x.min()), float(all_x.max())
    ylo = float(np.min(all_y)) if y_min is None else y_min
    yhi = float(np.max(all_y)) if y_max is None else y_max
    if hline is not None:
        ylo, yhi = min(ylo, hline), max(yhi, hline)

    if logy:
        if ylo <= 0:
            raise ValueError("log scale needs positive values")
        yt = log_ticks(ylo, yhi)
        lo_l, hi_l = math.log(min(ylo, yt[0])), math.log(max(yhi, yt[-1]))
        yscale = lambda v: f.y0 - (math.log(v) - lo_l) / (hi_l - lo_l) * (f.y0 - f.y1)  # noqa: E731
    else:
        yt = nice_ticks(ylo, yhi, 5)
        lo_v, hi_v = min(yt[0], ylo), max(yt[-1], yhi)
        yscale = lambda v: f.y0 - (v - lo_v) / (hi_v - lo_v) * (f.y0 - f.y1)  # noqa: E731
    xscale = lambda v: f.x0 + (v - xlo) / (xhi - xlo or 1) * (f.x1 - f.x0)  # noqa: E731

    parts = _open(height, title)
    if len(series) > 1:
        parts += _legend(series)
    for band in bands or []:
        b0, b1, label = band
        bx0 = xscale(float((pd.Timestamp(b0) - _EPOCH) / _DAY) if not isinstance(b0, (int, float)) else b0)
        bx1 = xscale(float((pd.Timestamp(b1) - _EPOCH) / _DAY) if not isinstance(b1, (int, float)) else b1)
        parts.append(f'  <rect x="{_r(bx0)}" y="{_r(f.y1)}" width="{_r(bx1 - bx0)}" height="{_r(f.y0 - f.y1)}" fill="#1F7A55" fill-opacity="0.07"/>')
        parts.append(f'  <text x="{_r(bx0 + 6)}" y="{_r(f.y1 + 13)}" font-size="10.5" fill="{INK_SOFT}" font-style="italic">{_esc(label)}</text>')
    parts += _y_axis(f, yt, yscale, y_fmt)
    x_ticks = _date_ticks(xlo, xhi) if is_date else [(t, f"{t:g}") for t in nice_ticks(xlo, xhi, 6)]
    parts += _x_axis(f, x_ticks, xscale)
    if hline is not None:
        yy = yscale(hline)
        parts.append(f'  <line x1="{_r(f.x0)}" y1="{_r(yy)}" x2="{_r(f.x1)}" y2="{_r(yy)}" stroke="{INK_SOFT}" stroke-width="1"/>')
        if hline_label:
            parts.append(f'  <text x="{_r(f.x0 + 4)}" y="{_r(yy - 5)}" font-size="10.5" fill="{INK_SOFT}">{_esc(hline_label)}</text>')

    end_labels = []
    for s, x, y in zip(series, xs, ys):
        c = ROLE_COLOURS[s.role]
        width = s.width or (1.5 if s.role == "benchmark" else 2)
        pts = " ".join(f"{_r(xscale(a))},{_r(yscale(b))}" for a, b in zip(x, y))
        if s.area:
            base = yscale(max(min(yt[0], 0.0), yt[0]) if not logy else yt[0])
            base = yscale(0.0) if (not logy and yt[0] <= 0 <= yt[-1]) else base
            parts.append(f'  <polygon points="{_r(xscale(x[0]))},{_r(base)} {pts} {_r(xscale(x[-1]))},{_r(base)}" fill="{c}" fill-opacity="0.10"/>')
        parts.append(f'  <polyline points="{pts}" fill="none" stroke="{c}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"><title>{_esc(s.name)}</title></polyline>')
        if s.label_end:
            ex, ey = xscale(x[-1]), yscale(y[-1])
            parts.append(f'  <circle cx="{_r(ex)}" cy="{_r(ey)}" r="4" fill="{c}" stroke="{SURFACE}" stroke-width="2"><title>{_esc(s.name)}: {_esc(y_fmt(y[-1]))}</title></circle>')
            short = s.end_label or s.name
            end_labels.append([ey, f"{short}: {y_fmt(y[-1])}" if len(series) > 1 else y_fmt(y[-1]), ex])
    # end labels: text in ink, de-collided with leader lines if they crowd
    end_labels.sort(key=lambda e: e[0])
    placed = []
    for ey, text, ex in end_labels:
        ty = max(ey, (placed[-1] + 14) if placed else ey)
        placed.append(ty)
        if abs(ty - ey) > 2:
            parts.append(f'  <line x1="{_r(ex + 5)}" y1="{_r(ey)}" x2="{_r(ex + 12)}" y2="{_r(ty - 4)}" stroke="{AXIS}" stroke-width="1"/>')
        parts.append(f'  <text x="{_r(ex + 12)}" y="{_r(ty + 4)}" font-size="11" fill="{INK}">{_esc(text)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def drawdown_chart(dd: pd.Series, *, title: str, height: int = 220) -> str:
    """Underwater curve: fractional drawdown (≤ 0) as a coral wash with a 2px coral line."""
    s = Series("drawdown", dd, role="loss", area=True, label_end=False)
    svg = line_chart([s], title=title, y_fmt=fmt_pct(0), height=height, y_max=0.0)
    worst_i = int(np.argmin(dd.to_numpy()))
    note = f'  <text x="{W - 26}" y="22" font-size="11" text-anchor="end" fill="{INK}">worst {fmt_pct(1)(dd.iloc[worst_i])}</text>'
    return svg.replace("\n</svg>", "\n" + note + "\n</svg>")


# ---------------------------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------------------------
def histogram(values, *, title: str, bins: int = 60, x_fmt=fmt_pct(1), height: int = 280,
              density_overlay: tuple[np.ndarray, np.ndarray] | None = None, overlay_label: str = "normal fit",
              tail_below: float | None = None, tail_label: str = "left tail") -> str:
    """Density histogram in ultramarine; optional overlay curve in ink; optional coral left tail."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(v, bins=bins, density=True)
    f = _Frame(height=height, right=40)
    ymax = float(counts.max())
    if density_overlay is not None:
        ymax = max(ymax, float(np.max(density_overlay[1])))
    xlo, xhi = float(edges[0]), float(edges[-1])
    xt = nice_ticks(xlo, xhi, 6)
    xscale = lambda x: f.x0 + (x - xlo) / (xhi - xlo) * (f.x1 - f.x0)  # noqa: E731
    yscale = lambda y: f.y0 - y / (ymax * 1.08) * (f.y0 - f.y1)  # noqa: E731
    parts = _open(height, title)
    parts += [f'  <line x1="{_r(f.x0)}" y1="{_r(yscale(ymax * k / 4))}" x2="{_r(f.x1)}" y2="{_r(yscale(ymax * k / 4))}" stroke="{GRID}" stroke-width="1"/>' for k in (1, 2, 3, 4)]
    parts += _x_axis(f, [(t, x_fmt(t)) for t in xt], xscale)
    gap = 1.0
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        if c <= 0:
            continue
        x0, x1 = xscale(a) + gap / 2, xscale(b) - gap / 2
        y = yscale(c)
        colour = ROLE_COLOURS["loss"] if (tail_below is not None and b <= tail_below) else ROLE_COLOURS["strategy"]
        parts.append(f'  <rect x="{_r(x0)}" y="{_r(y)}" width="{_r(max(x1 - x0, 0.6))}" height="{_r(f.y0 - y)}" fill="{colour}" fill-opacity="0.55">'
                     f"<title>{_esc(x_fmt(a))} to {_esc(x_fmt(b))}</title></rect>")
    if density_overlay is not None:
        ox, oy = density_overlay
        keep = (ox >= xlo) & (ox <= xhi)
        pts = " ".join(f"{_r(xscale(a))},{_r(yscale(b))}" for a, b in zip(ox[keep], oy[keep]))
        parts.append(f'  <polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="2" stroke-linejoin="round"><title>{_esc(overlay_label)}</title></polyline>')
        parts.append(f'  <line x1="{W - 190}" y1="32" x2="{W - 174}" y2="32" stroke="{INK}" stroke-width="2.5"/>')
        parts.append(f'  <text x="{W - 168}" y="36" font-size="11" fill="{INK_SOFT}">{_esc(overlay_label)}</text>')
    if tail_below is not None:
        tx = xscale(tail_below)
        parts.append(f'  <line x1="{_r(tx)}" y1="{_r(f.y1 - 2)}" x2="{_r(tx)}" y2="{_r(f.y0)}" stroke="{INK_SOFT}" stroke-width="1"/>')
        if tail_label:  # above the plot area, never on top of the bars
            parts.append(f'  <text x="{_r(tx - 5)}" y="{_r(f.y1 - 6)}" font-size="10.5" text-anchor="end" fill="{INK}">{_esc(tail_label)} ←</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------
# column chart
# ---------------------------------------------------------------------------------------------
def column_chart(labels: list[str], values: list[float], *, title: str, y_fmt=fmt_num(1), height: int = 260,
                 roles: list[str] | None = None, value_labels: bool = True,
                 y_min: float | None = None, y_max: float | None = None) -> str:
    """Vertical columns ≤ 24px wide, 4px rounded data-end, square at the baseline; values on caps."""
    vals = np.asarray(values, dtype=float)
    f = _Frame(height=height, right=30, bottom=44)
    # 15% headroom beyond the extremes so value labels never collide with the axis labels
    span = max(float(vals.max()) - min(0.0, float(vals.min())), 1e-12)
    lo_pad = min(0.0, float(vals.min()) - 0.15 * span) if vals.min() < 0 else 0.0
    hi_pad = max(0.0, float(vals.max()) + 0.15 * span) if vals.max() > 0 else 0.0
    if y_min is not None:  # a fixed range lets two charts be compared by eye
        lo_pad = min(lo_pad, y_min)
    if y_max is not None:
        hi_pad = max(hi_pad, y_max)
    yt = nice_ticks(lo_pad, hi_pad, 5)
    lo, hi = yt[0], yt[-1]
    yscale = lambda v: f.y0 - (v - lo) / (hi - lo) * (f.y0 - f.y1)  # noqa: E731
    parts = _open(height, title)
    parts += _y_axis(f, yt, yscale, y_fmt)
    band = (f.x1 - f.x0) / len(vals)
    bw = min(24.0, band * 0.6)
    base = yscale(0.0)
    parts.append(f'  <line x1="{_r(f.x0)}" y1="{_r(base)}" x2="{_r(f.x1)}" y2="{_r(base)}" stroke="{AXIS}" stroke-width="1"/>')
    for i, (lab, v) in enumerate(zip(labels, vals)):
        cx = f.x0 + band * (i + 0.5)
        role = (roles[i] if roles else ("loss" if v < 0 else "strategy"))
        c = ROLE_COLOURS[role]
        top = yscale(v)
        x0, x1 = cx - bw / 2, cx + bw / 2
        rr = min(4.0, abs(base - top) / 2)
        if v >= 0:
            d = (f"M{_r(x0)},{_r(base)} V{_r(top + rr)} Q{_r(x0)},{_r(top)} {_r(x0 + rr)},{_r(top)} "
                 f"H{_r(x1 - rr)} Q{_r(x1)},{_r(top)} {_r(x1)},{_r(top + rr)} V{_r(base)} Z")
        else:
            d = (f"M{_r(x0)},{_r(base)} V{_r(top - rr)} Q{_r(x0)},{_r(top)} {_r(x0 + rr)},{_r(top)} "
                 f"H{_r(x1 - rr)} Q{_r(x1)},{_r(top)} {_r(x1)},{_r(top - rr)} V{_r(base)} Z")
        parts.append(f'  <path d="{d}" fill="{c}"><title>{_esc(lab)}: {_esc(y_fmt(v))}</title></path>')
        if value_labels:
            ty = top - 7 if v >= 0 else top + 15
            parts.append(f'  <text x="{_r(cx)}" y="{_r(ty)}" font-size="11" text-anchor="middle" fill="{INK}" style="font-variant-numeric:tabular-nums">{_esc(y_fmt(v))}</text>')
        parts.append(f'  <text x="{_r(cx)}" y="{_r(f.y0 + 18)}" font-size="11" text-anchor="middle" fill="{INK_SOFT}">{_esc(lab)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def fmt_usd_compact(decimals: int = 1):
    """Dollars with a true minus sign and k/m/bn suffixes: −$3.2k, $9.9k, $1.5m."""
    def f(v: float) -> str:
        a = abs(v)
        for div, suf in ((1e9, "bn"), (1e6, "m"), (1e3, "k")):
            if a >= div:
                s = f"${a / div:.{decimals}f}{suf}"
                break
        else:
            s = f"${a:,.0f}"
        return ("−" if v < 0 and a > 0 else "") + s
    return f


# ---------------------------------------------------------------------------------------------
# grouped columns
# ---------------------------------------------------------------------------------------------
def grouped_column_chart(categories: list[str], series: list[tuple[str, list[float], str]], *, title: str,
                         y_fmt=fmt_pct(0), height: int = 280, value_labels: bool = False,
                         y_min: float | None = None, y_max: float | None = None) -> str:
    """Grouped columns: ≤ 3 series of (name, values, role), one group per category, with a legend.

    Same marks as column_chart (≤ 24px, rounded data-end, 2px surface gap between neighbours).
    """
    if not 1 <= len(series) <= 3:
        raise ValueError("grouped_column_chart takes 1–3 series")
    vals = np.array([v for _, vs, _ in series for v in vs], dtype=float)
    f = _Frame(height=height, top=56, right=30, bottom=44)
    span = max(float(vals.max()) - min(0.0, float(vals.min())), 1e-12)
    lo = min(0.0, float(vals.min()) - 0.12 * span) if vals.min() < 0 else 0.0
    hi = max(0.0, float(vals.max()) + 0.12 * span) if vals.max() > 0 else 0.0
    lo = min(lo, y_min) if y_min is not None else lo
    hi = max(hi, y_max) if y_max is not None else hi
    yt = nice_ticks(lo, hi, 5)
    ylo, yhi = yt[0], yt[-1]
    yscale = lambda v: f.y0 - (v - ylo) / (yhi - ylo) * (f.y0 - f.y1)  # noqa: E731
    parts = _open(height, title)
    parts += _legend([Series(name, pd.Series([0.0]), role=role) for name, _, role in series])
    parts += _y_axis(f, yt, yscale, y_fmt)
    base = yscale(0.0)
    parts.append(f'  <line x1="{_r(f.x0)}" y1="{_r(base)}" x2="{_r(f.x1)}" y2="{_r(base)}" stroke="{AXIS}" stroke-width="1"/>')
    band = (f.x1 - f.x0) / len(categories)
    k = len(series)
    bw = min(24.0, band * 0.7 / k)
    gap = 2.0
    for i, cat in enumerate(categories):
        cx = f.x0 + band * (i + 0.5)
        left = cx - (k * bw + (k - 1) * gap) / 2
        for j, (name, vs, role) in enumerate(series):
            v = float(vs[i])
            x0 = left + j * (bw + gap)
            x1 = x0 + bw
            top = yscale(v)
            rr = min(4.0, abs(base - top) / 2)
            if v >= 0:
                d = (f"M{_r(x0)},{_r(base)} V{_r(top + rr)} Q{_r(x0)},{_r(top)} {_r(x0 + rr)},{_r(top)} "
                     f"H{_r(x1 - rr)} Q{_r(x1)},{_r(top)} {_r(x1)},{_r(top + rr)} V{_r(base)} Z")
            else:
                d = (f"M{_r(x0)},{_r(base)} V{_r(top - rr)} Q{_r(x0)},{_r(top)} {_r(x0 + rr)},{_r(top)} "
                     f"H{_r(x1 - rr)} Q{_r(x1)},{_r(top)} {_r(x1)},{_r(top - rr)} V{_r(base)} Z")
            parts.append(f'  <path d="{d}" fill="{ROLE_COLOURS[role]}"><title>{_esc(cat)} · {_esc(name)}: {_esc(y_fmt(v))}</title></path>')
            if value_labels:
                ty = top - 6 if v >= 0 else top + 13
                parts.append(f'  <text x="{_r((x0 + x1) / 2)}" y="{_r(ty)}" font-size="10" text-anchor="middle" fill="{INK}" style="font-variant-numeric:tabular-nums">{_esc(y_fmt(v))}</text>')
        parts.append(f'  <text x="{_r(cx)}" y="{_r(f.y0 + 18)}" font-size="11" text-anchor="middle" fill="{INK_SOFT}">{_esc(cat)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------------------------
def scatter_chart(x, y, *, title: str, x_fmt=fmt_num(1), y_fmt=fmt_num(1), height: int = 320,
                  role: str = "strategy", diagonal: bool = False, fit_line: bool = False,
                  x_label: str | None = None, y_label: str | None = None, max_points: int = 1500,
                  rng_seed: int = 0) -> str:
    """Dots (r=2.5, semi-transparent) with optional y = x diagonal (QQ plots) or OLS fit line.

    More than `max_points` points are thinned by a seeded random subsample so the SVG stays small.
    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    if len(xv) > max_points:
        keep = np.sort(np.random.default_rng(rng_seed).choice(len(xv), max_points, replace=False))
        xs, ys = xv[keep], yv[keep]
    else:
        xs, ys = xv, yv
    f = _Frame(height=height, right=40, bottom=48 if x_label else 36)
    lo_x, hi_x = float(xv.min()), float(xv.max())
    lo_y, hi_y = float(yv.min()), float(yv.max())
    if diagonal:
        lo_x = lo_y = min(lo_x, lo_y)
        hi_x = hi_y = max(hi_x, hi_y)
    xt, yt = nice_ticks(lo_x, hi_x, 6), nice_ticks(lo_y, hi_y, 5)
    xscale = lambda v: f.x0 + (v - xt[0]) / (xt[-1] - xt[0]) * (f.x1 - f.x0)  # noqa: E731
    yscale = lambda v: f.y0 - (v - yt[0]) / (yt[-1] - yt[0]) * (f.y0 - f.y1)  # noqa: E731
    parts = _open(height, title)
    parts += _y_axis(f, yt, yscale, y_fmt)
    parts += _x_axis(f, [(t, x_fmt(t)) for t in xt], xscale)
    if x_label:
        parts.append(f'  <text x="{_r((f.x0 + f.x1) / 2)}" y="{_r(height - 8)}" font-size="11" text-anchor="middle" fill="{INK_SOFT}">{_esc(x_label)}</text>')
    if y_label:
        parts.append(f'  <text x="{_r(f.x0)}" y="{_r(f.y1 - 8)}" font-size="11" fill="{INK_SOFT}">{_esc(y_label)}</text>')
    if diagonal:
        a, b = max(xt[0], yt[0]), min(xt[-1], yt[-1])
        parts.append(f'  <line x1="{_r(xscale(a))}" y1="{_r(yscale(a))}" x2="{_r(xscale(b))}" y2="{_r(yscale(b))}" stroke="{INK_SOFT}" stroke-width="1.5"><title>y = x</title></line>')
    c = ROLE_COLOURS[role]
    for a, b in zip(xs, ys):
        parts.append(f'  <circle cx="{_r(xscale(a))}" cy="{_r(yscale(b))}" r="2.5" fill="{c}" fill-opacity="0.45"/>')
    if fit_line and len(xv) > 2:
        slope, icept = np.polyfit(xv, yv, 1)
        xa, xb = xt[0], xt[-1]
        ya, yb = icept + slope * xa, icept + slope * xb
        parts.append(f'  <line x1="{_r(xscale(xa))}" y1="{_r(yscale(ya))}" x2="{_r(xscale(xb))}" y2="{_r(yscale(yb))}" stroke="{INK}" stroke-width="2"><title>OLS fit: slope {slope:.3g}</title></line>')
    parts.append("</svg>")
    return "\n".join(parts)
