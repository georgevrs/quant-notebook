"""Tests for helpers promoted from Wave 2 labs (barrier probability, jump diffusion, options, time axes)."""
import numpy as np
import pytest

from quantnb import options, returns, stats, synth


def test_p_touch_matches_simulation():
    n = 20_000
    px = synth.gbm_prices(252, mu=0.05, sigma=0.3, n_paths=n, rng=np.random.default_rng(7)).to_numpy()
    for ratio in (1.3, 0.8):
        hit = (px.max(axis=0) >= ratio * px[0]).mean() if ratio > 1 else (px.min(axis=0) <= ratio * px[0]).mean()
        theory = stats.p_touch(ratio, 0.05, 0.3, 1.0, dt=1 / 252)
        assert abs(hit - theory) < 4 * np.sqrt(theory * (1 - theory) / n)


def test_p_touch_edge_cases():
    assert stats.p_touch(1.0, 0.05, 0.2, 1.0) == 1.0
    # the continuity correction makes a discretely monitored barrier harder to touch
    assert stats.p_touch(1.3, 0.05, 0.3, 1.0, dt=1 / 12) < stats.p_touch(1.3, 0.05, 0.3, 1.0)


def test_jump_diffusion_mean_is_compensated_and_skewed():
    r = synth.jump_diffusion_returns(252, 20_000, mu=0.07, sigma=0.15, jump_rate=1.0, jump_mean=-0.1,
                                     rng=np.random.default_rng(3))
    growth = np.prod(1 + r, axis=0)
    assert abs(growth.mean() - np.exp(0.07)) < 4 * growth.std() / np.sqrt(growth.size)
    assert stats.skewness(np.log1p(r)) < -1


def test_batch_se_close_to_analytic_for_the_mean():
    x = np.random.default_rng(5).standard_normal(100_000)
    assert stats.batch_se(np.mean, x, 50) == pytest.approx(1 / np.sqrt(100_000), rel=0.3)


def test_options_parity_and_implied_vol():
    c = options.bs_price(100, 105, 0.5, 0.03, 0.2, q=0.01)
    p = options.bs_price(100, 105, 0.5, 0.03, 0.2, q=0.01, kind="put")
    assert abs((c - p) - (100 * np.exp(-0.01 * 0.5) - 105 * np.exp(-0.03 * 0.5))) < 1e-10
    iv, _ = options.implied_vol(float(c), 100, 105, 0.5, 0.03, q=0.01)
    assert abs(iv - 0.2) < 1e-8
    with pytest.raises(ValueError):
        options.implied_vol(200.0, 100, 105, 0.5, 0.03)


def test_in_years_and_paths_drawdown():
    s = returns.in_years([1.0, 2.0, 3.0])
    assert list(s.index) == [1 / 252, 2 / 252, 3 / 252]
    dd = returns.max_drawdown_paths(np.array([[0.1, -0.5], [-0.5, 0.2]]))
    assert np.allclose(dd, [-0.5, -0.5])
