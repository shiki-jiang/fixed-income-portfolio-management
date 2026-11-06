"""Offline deterministic tests: `python -m pytest -q`."""

import numpy as np
import pandas as pd

from portfolio import (
    BENCHMARK, LOOKBACK, TICKERS, optimize_weights, run_backtest, summarize,
)


def synthetic_prices():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2019-01-01", periods=650)
    shocks = rng.normal(0.0001, 0.004, (len(dates), 6))
    shocks[:, 1] += shocks[:, 0] * 0.6
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS + ["AGG"])


def test_optimizer_satisfies_constraints():
    returns = synthetic_prices().pct_change().dropna()
    result = optimize_weights(returns.iloc[:LOOKBACK], BENCHMARK)
    assert abs(result.sum() - 1) < 1e-6
    assert (result >= -1e-6).all()
    assert (result <= 0.45 + 1e-6).all()
    assert result[TICKERS[:3]].sum() >= 0.70 - 1e-6
    assert result["HYG"] <= 0.05 + 1e-6


def test_daily_reconciliation_and_weights():
    daily, weights, attribution, trades = run_backtest(synthetic_prices(), "2020-05-01")
    np.testing.assert_allclose(
        attribution[TICKERS].sum(axis=1) + attribution["trading_cost"],
        daily["net_active_return"], atol=1e-12,
    )
    np.testing.assert_allclose(
        weights.filter(like="portfolio_").sum(axis=1), 1, atol=1e-10,
    )
    assert not trades.empty
    assert (daily["portfolio_nav"] > 0).all()
    assert summarize(daily)["tracking_error"] >= 0


def test_first_trade_does_not_use_future_returns():
    original = synthetic_prices()
    shifted = original.copy()
    decision_date = pd.Timestamp("2020-05-01")
    shifted.loc[shifted.index > decision_date, TICKERS] *= 1.3
    _, _, _, original_trades = run_backtest(original, "2020-05-01")
    _, _, _, shifted_trades = run_backtest(shifted, "2020-05-01")
    first = original_trades.date.min()
    a = original_trades.loc[original_trades.date == first, "new_weight"].to_numpy()
    b = shifted_trades.loc[shifted_trades.date == first, "new_weight"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-10)
