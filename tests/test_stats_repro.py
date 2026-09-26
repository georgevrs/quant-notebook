import numpy as np
import pandas as pd
import pytest

from quantnb import charts, repro, stats


def test_sharpe_se_daily_vs_annual():
    # daily data: correction negligible, SE ≈ 1/sqrt(Y)
    assert stats.sharpe_se(0.37, 20) == pytest.approx(1 / np.sqrt(20), rel=1e-3)
    # yearly observations: Lo's full correction
    assert stats.sharpe_se(1.0, 10, periods_per_year=1) == pytest.approx(np.sqrt(1.5 / 10))


def test_years_needed_matches_se():
    y = stats.years_needed(0.5)
    assert 1.96 * stats.sharpe_se(0.5, y) == pytest.approx(0.5, rel=1e-9)


def test_expected_max_sharpe_known_value():
    assert stats.expected_max_sharpe(1, 5) == 0.0
    assert stats.expected_max_sharpe(1000, 5) == pytest.approx(1.455, abs=0.01)


def test_factor_regression_recovers_truth():
    rng = np.random.default_rng(0)
    n = 252 * 30
    mkt = rng.normal(0.0003, 0.01, n)
    y = 0.0002 + 1.5 * mkt + rng.normal(0, 0.005, n)
    out = stats.factor_regression(y, mkt, names=["mkt"])
    assert out["betas"]["mkt"] == pytest.approx(1.5, abs=0.02)
    assert out["alpha_annual"] == pytest.approx(0.0504, abs=0.045)  # SE ≈ 1.45 pp: allow 3 SE
    assert out["alpha_t"] > 2


def test_oos_r2_zero_forecast():
    y = np.array([0.01, -0.02, 0.03])
    assert stats.oos_r2(y, np.zeros(3)) == 0.0
    assert stats.oos_r2(y, y) == 1.0


def test_annual_sharpe_arrays():
    x = np.random.default_rng(1).normal(0.001, 0.01, (5000, 3))
    s = stats.annual_sharpe(x)
    assert s.shape == (3,)


def test_fingerprints():
    assert repro.fingerprint([1, 2, 3]) == repro.fingerprint(np.array([1, 2, 3], dtype=np.int32))
    df = pd.DataFrame({"b": [1.0, 2.0], "a": [3.0, 4.0]}, index=[1, 0])
    same = pd.DataFrame({"a": [4.0, 3.0], "b": [2.0, 1.0]}, index=[0, 1])
    assert repro.content_fingerprint(df) == repro.content_fingerprint(same)
    assert repro.content_fingerprint(df + 1e-15) == repro.content_fingerprint(df)
    assert repro.content_fingerprint(df + 1e-3) != repro.content_fingerprint(df)


def test_new_charts_valid():
    import xml.etree.ElementTree as ET
    svg = charts.grouped_column_chart(["A", "B"], [("x", [0.1, -0.05], "strategy"), ("y", [0.02, 0.03], "alt1")],
                                      title="Grouped β test")
    root = ET.fromstring(svg)
    assert "viewBox" in root.attrib and "β" in svg
    q = np.sort(np.random.default_rng(2).standard_normal(500))
    ET.fromstring(charts.scatter_chart(q, q + 0.1, title="QQ", diagonal=True))
    ET.fromstring(charts.scatter_chart(q, 2 * q, title="fit", fit_line=True))
