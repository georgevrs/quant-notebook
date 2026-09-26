import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from quantnb import Lab, charts, returns, synth

SVG_NS = "{http://www.w3.org/2000/svg}"


# ---------------------------------------------------------------- returns
def test_simple_and_log_returns_known_values():
    p = pd.Series([100.0, 110.0, 99.0])
    assert returns.simple_returns(p).round(10).tolist() == [0.1, -0.1]
    lr = returns.log_returns(p)
    assert np.allclose(lr, np.log([1.1, 0.9]))
    # log returns add up; simple returns compound
    assert np.isclose(lr.sum(), np.log(99 / 100))
    assert np.isclose(returns.cumulative_growth(returns.simple_returns(p)).iloc[-1], 0.99)


def test_conversions_round_trip():
    r = np.array([-0.5, 0.0, 0.25, 1.0])
    assert np.allclose(returns.log_to_simple(returns.simple_to_log(r)), r)


def test_sharpe_and_vol_annualisation():
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0.0004, 0.01, 252 * 40))
    assert abs(returns.annualised_vol(daily) - 0.01 * np.sqrt(252)) < 0.003
    assert abs(returns.sharpe_ratio(daily) - 0.0004 / 0.01 * np.sqrt(252)) < 0.1


def test_drawdown_is_nonpositive_and_max_known():
    r = pd.Series([0.10, -0.50, 0.20])  # wealth 1.1, 0.55, 0.66 → max dd −50%
    dd = returns.drawdown(r)
    assert (dd <= 0).all()
    assert np.isclose(returns.max_drawdown(r), -0.5)


def test_leverage_and_drag():
    r = pd.Series([0.01, -0.02])
    assert np.allclose(returns.leveraged_returns(r, 2.0, 0.0001), [0.0199, -0.0401])
    assert returns.volatility_drag(0.2) == pytest.approx(0.02)


# ---------------------------------------------------------------- synth
def test_gbm_moments_match_inputs():
    rng = np.random.default_rng(1)
    px = synth.gbm_prices(252 * 200, mu=0.08, sigma=0.2, rng=rng)
    lr = returns.log_returns(px["price"])
    assert abs(lr.std() * np.sqrt(252) - 0.2) < 0.005
    assert abs(lr.mean() * 252 - (0.08 - 0.02)) < 0.03


def test_garch_has_fat_tails_and_clustering():
    rng = np.random.default_rng(2)
    df = synth.garch_returns(20_000, rng=rng)
    r = df["ret"] - df["ret"].mean()
    kurt = ((r ** 4).mean() / (r ** 2).mean() ** 2)
    assert kurt > 4.0  # normal would be 3
    sq = r ** 2
    assert sq.autocorr(1) > 0.05  # volatility clustering
    assert abs(r.autocorr(1)) < 0.05  # but returns themselves ~unpredictable


def test_factor_model_shapes():
    out = synth.factor_model_returns(500, 20, 3, rng=np.random.default_rng(3))
    assert out["returns"].shape == (500, 20)
    assert out["loadings"].shape == (20, 3)


# ---------------------------------------------------------------- charts
def _check_svg(svg: str):
    root = ET.fromstring(svg)
    assert root.tag == SVG_NS + "svg"
    assert "viewBox" in root.attrib
    assert "width" not in root.attrib and "height" not in root.attrib
    assert not [el for el in root.iter() if "id" in el.attrib], "charts must not use ids"
    return root


def test_line_chart_valid_and_deterministic():
    rng = np.random.default_rng(4)
    px = synth.gbm_prices(2000, rng=rng)["price"]
    bench = px * 0.9
    s = [charts.Series("strategy", px), charts.Series("benchmark", bench, role="benchmark")]
    svg1 = charts.line_chart(s, title="Growth of $100", logy=True)
    svg2 = charts.line_chart(s, title="Growth of $100", logy=True)
    assert svg1 == svg2
    root = _check_svg(svg1)
    polylines = [el for el in root.iter(SVG_NS + "polyline")]
    assert len(polylines) == 2
    assert len(polylines[0].attrib["points"].split()) <= 400  # downsampled


def test_drawdown_histogram_columns_valid():
    rng = np.random.default_rng(5)
    r = synth.garch_returns(3000, rng=rng)["ret"]
    _check_svg(charts.drawdown_chart(returns.drawdown(r), title="Drawdown"))
    x = np.linspace(-0.08, 0.08, 200)
    pdf = np.exp(-x ** 2 / (2 * r.var())) / np.sqrt(2 * np.pi * r.var())
    _check_svg(charts.histogram(r, title="Daily returns", density_overlay=(x, pdf), tail_below=-0.03))
    _check_svg(charts.column_chart(["1x", "2x", "3x"], [0.05, 0.07, -0.02], title="CAGR by leverage",
                                   y_fmt=charts.fmt_pct(0)))


def test_nice_ticks():
    assert charts.nice_ticks(0, 1, 5) == [0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0] or \
        charts.nice_ticks(0, 1, 5)[0] == 0
    t = charts.nice_ticks(-0.43, 0.0, 5)
    assert t[0] <= -0.43 and t[-1] >= 0


# ---------------------------------------------------------------- lab
def test_lab_writes_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # outside the repo → ./out/
    lab = Lab("2.3")
    lab.record("sharpe", np.float64(0.5))
    lab.chart("demo", charts.column_chart(["a"], [1.0], title="t"))
    path = lab.save()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["results"]["sharpe"] == 0.5
    assert (tmp_path / "out" / "s03_demo.svg").exists()
    with pytest.raises(ValueError):
        lab.record("bad", float("nan"))
