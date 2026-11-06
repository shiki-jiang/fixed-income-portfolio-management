"""Portfolio management analytics: run `python portfolio.py --refresh`."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import Bounds, LinearConstraint, minimize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
TICKERS = ["SHY", "IEF", "TLT", "LQD", "HYG"]
BENCHMARK = pd.Series([0.15, 0.35, 0.10, 0.30, 0.10], index=TICKERS)
START, END, TEST_START = "2018-01-01", "2026-11-01", "2021-01-01"
INITIAL_CAPITAL = 100_000_000
LOOKBACK = 252
COST_BPS = 5  # Per dollar bought or sold
TURNOVER_PENALTY = 0.0002  # Penalizes traded weight in the annualized-variance objective


def get_prices(refresh: bool = False) -> pd.DataFrame:
    path = DATA / "etf_prices.csv"
    if not refresh and path.exists():
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        raw = yf.download(
            TICKERS + ["AGG"], start=START, end=END,
            auto_adjust=True, group_by="ticker", progress=False,
        )
        prices = pd.DataFrame({t: raw[t]["Close"] for t in TICKERS + ["AGG"]})
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.sort_index().dropna()
        DATA.mkdir(exist_ok=True)
        prices.to_csv(path)
    prices = prices[TICKERS + ["AGG"]].sort_index().dropna()
    if len(prices) < 500 or not prices.index.is_unique or (prices <= 0).any().any():
        raise ValueError("ETF prices are incomplete or invalid; check the Yahoo download.")
    return prices.loc[START:pd.Timestamp(END) - pd.Timedelta(days=1)]


def get_treasury(refresh: bool = False) -> pd.DataFrame:
    path = DATA / "treasury_yields.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    frames = []
    for year in range(int(START[:4]), int(END[:4]) + 1):
        url = (
            f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
            f"&field_tdr_date_value={year}&page&_format=csv"
        )
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        required = ["Date", "2 Yr", "5 Yr", "10 Yr", "30 Yr"]
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Unexpected U.S. Treasury columns for {year}: {list(frame.columns)}")
        frames.append(frame[required])

    yields = pd.concat(frames, ignore_index=True)
    yields["Date"] = pd.to_datetime(yields["Date"])
    yields = yields.set_index("Date").sort_index()
    yields = yields.apply(pd.to_numeric, errors="coerce")
    yields = yields.loc[START:pd.Timestamp(END) - pd.Timedelta(days=1)].dropna()
    if yields.empty:
        raise ValueError("No valid Treasury yields were downloaded.")
    yields.to_csv(path)
    return yields


def optimize_weights(history: pd.DataFrame, previous: pd.Series) -> pd.Series:
    """Minimize ex-ante active variance plus L1 turnover, subject to mandate."""
    if len(history) < LOOKBACK:
        raise ValueError("Need at least 252 prior observations for covariance estimation.")

    # A small diagonal shrinkage stabilizes the estimated covariance matrix.
    cov = history[TICKERS].tail(LOOKBACK).cov().to_numpy() * 252
    cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov))
    n = len(TICKERS)
    b, prev = BENCHMARK.to_numpy(), previous[TICKERS].to_numpy()

    def objective(x):
        d = x[:n] - b
        return d @ cov @ d + TURNOVER_PENALTY * x[n:].sum()

    def gradient(x):
        return np.r_[2 * cov @ (x[:n] - b), np.full(n, TURNOVER_PENALTY)]

    # x = [portfolio weights, auxiliary absolute-weight-changes].
    row_sum = np.r_[np.ones(n), np.zeros(n)]
    row_treasury = np.r_[np.array([1, 1, 1, 0, 0]), np.zeros(n)]
    row_high_yield = np.r_[np.array([0, 0, 0, 0, 1]), np.zeros(n)]
    abs_constraints = np.vstack([
        np.c_[-np.eye(n), np.eye(n)],  # u >= w - previous
        np.c_[np.eye(n), np.eye(n)],   # u >= previous - w
    ])
    constraints = [
        LinearConstraint(row_sum, 1, 1),
        LinearConstraint(row_treasury, 0.70, np.inf),
        LinearConstraint(row_high_yield, -np.inf, 0.05),
        LinearConstraint(abs_constraints, np.r_[-prev, prev], np.inf),
    ]
    guess = np.array([0.20, 0.40, 0.15, 0.20, 0.05])
    x0 = np.r_[guess, np.abs(guess - prev)]
    bounds = Bounds(np.zeros(2 * n), np.r_[np.full(n, 0.45), np.full(n, np.inf)])
    result = minimize(
        objective, x0, jac=gradient, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"ftol": 1e-11, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    weights = pd.Series(result.x[:n], index=TICKERS)
    if (abs(weights.sum() - 1) > 1e-6 or weights[TICKERS[:3]].sum() < 0.70 - 1e-6
            or weights["HYG"] > 0.05 + 1e-6 or weights.max() > 0.45 + 1e-6):
        raise RuntimeError("Optimizer returned weights violating the mandate.")
    return weights


def run_backtest(prices: pd.DataFrame, start: str = TEST_START):
    """Trades before each monthly first-day return using only prior data."""
    returns = prices.pct_change(fill_method=None).dropna()
    if returns.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)].shape[0] < LOOKBACK:
        raise ValueError("Not enough observations before the out-of-sample start date.")

    portfolio_w = BENCHMARK.copy()
    benchmark_w = BENCHMARK.copy()
    portfolio_nav = benchmark_nav = INITIAL_CAPITAL
    rows, weight_rows, trades, contributions = [], [], [], []
    current_month = None

    for i, (date, daily_returns) in enumerate(returns.iterrows()):
        if date < pd.Timestamp(start):
            continue
        month = date.to_period("M")
        cost_fraction = turnover = 0.0

        if month != current_month:
            previous = portfolio_w.copy()
            portfolio_w = optimize_weights(returns.iloc[i - LOOKBACK:i], previous)
            benchmark_w = BENCHMARK.copy()  # Frictionless, monthly rebalanced custom index.
            changes = portfolio_w - previous
            turnover = float(changes.abs().sum())  # One-way purchases + sales, counted separately.
            cost_fraction = COST_BPS / 10_000 * turnover
            for ticker in TICKERS:
                trade = float(changes[ticker] * portfolio_nav)
                if abs(trade) > 0.01:
                    trades.append({
                        "date": date, "ticker": ticker,
                        "old_weight": previous[ticker],
                        "new_weight": portfolio_w[ticker],
                        "trade_usd": trade,
                        "estimated_cost_usd": abs(trade) * COST_BPS / 10_000,
                    })
            current_month = month

        r = daily_returns[TICKERS]
        gross = float(portfolio_w @ r)
        benchmark_return = float(benchmark_w @ r)
        net = gross - cost_fraction
        portfolio_nav *= 1 + net
        benchmark_nav *= 1 + benchmark_return
        active_contrib = (portfolio_w - benchmark_w) * r

        contributions.append({"date": date, **active_contrib.to_dict(), "trading_cost": -cost_fraction})
        rows.append({
            "date": date, "portfolio_return": net, "portfolio_gross_return": gross,
            "benchmark_return": benchmark_return, "agg_return": daily_returns["AGG"],
            "gross_active_return": gross - benchmark_return,
            "net_active_return": net - benchmark_return,
            "trading_cost": cost_fraction, "turnover": turnover,
            "portfolio_nav": portfolio_nav, "benchmark_nav": benchmark_nav,
        })

        # Price-induced holding drift; transaction costs are charged proportionally to NAV.
        portfolio_w = portfolio_w * (1 + r) / (1 + gross)
        benchmark_w = benchmark_w * (1 + r) / (1 + benchmark_return)
        weight_rows.append({
            "date": date, **{f"portfolio_{t}": portfolio_w[t] for t in TICKERS},
            **{f"benchmark_{t}": benchmark_w[t] for t in TICKERS},
        })

    daily = pd.DataFrame(rows).set_index("date")
    if daily.empty:
        raise ValueError("There are no backtest dates in the downloaded data.")
    for name, col in (("portfolio", "portfolio_return"),
                      ("benchmark", "benchmark_return"), ("agg", "agg_return")):
        daily[f"{name}_index"] = (1 + daily[col]).cumprod() * 100
        daily[f"{name}_drawdown"] = daily[f"{name}_index"] / daily[f"{name}_index"].cummax() - 1
    weights = pd.DataFrame(weight_rows).set_index("date")
    attribution = pd.DataFrame(contributions).set_index("date")
    blotter = pd.DataFrame(trades)
    return daily, weights, attribution, blotter


def summarize(daily: pd.DataFrame) -> dict:
    def asset_metrics(col):
        r = daily[col]
        wealth = (1 + r).prod()
        index = (1 + r).cumprod()
        return {
            "annualized_return": float(wealth ** (252 / len(r)) - 1),
            "annualized_volatility": float(r.std(ddof=1) * np.sqrt(252)),
            "maximum_drawdown": float((index / index.cummax() - 1).min()),
            "total_return": float(wealth - 1),
        }

    active = daily["net_active_return"]
    te = float(active.std(ddof=1) * np.sqrt(252))
    annualized_active = float(active.mean() * 252)
    return {
        "portfolio": asset_metrics("portfolio_return"),
        "benchmark": asset_metrics("benchmark_return"),
        "agg": asset_metrics("agg_return"),
        "tracking_error": te,
        "annualized_active_return": annualized_active,
        "information_ratio": annualized_active / te if te > 0 else None,
        "end_nav": float(daily["portfolio_nav"].iloc[-1]),
        "total_traded_weight": float(daily["turnover"].sum()),
        "total_transaction_cost_usd": float(
            (daily["trading_cost"] * daily["portfolio_nav"].shift(1)
             .fillna(INITIAL_CAPITAL)).sum()
        ),
        "backtest_start": str(daily.index.min().date()),
        "backtest_end": str(daily.index.max().date()),
    }


def current_risk(weights: pd.DataFrame, summary: dict) -> pd.DataFrame:
    """Scenario estimates only: durations are explicitly illustrative inputs."""
    inputs = pd.read_csv(ROOT / "config" / "duration_assumptions.csv").set_index("ticker")
    if inputs.loc[TICKERS, "source"].ne("ILLUSTRATIVE").any():
        raise ValueError("Update risk input metadata and verification before using live duration data.")
    latest = weights.iloc[-1]
    risk = pd.DataFrame({
        "portfolio_weight": [latest[f"portfolio_{t}"] for t in TICKERS],
        "benchmark_weight": [latest[f"benchmark_{t}"] for t in TICKERS],
        "effective_duration_assumption": inputs.loc[TICKERS, "duration_years"].to_numpy(),
    }, index=TICKERS)
    risk.index.name = "ticker"
    risk["duration_contribution"] = risk.portfolio_weight * risk.effective_duration_assumption
    risk["dv01_usd"] = risk.duration_contribution * summary["end_nav"] / 10_000
    return risk


def save_results(daily, weights, attribution, trades, summary, risk, treasury=None):
    OUTPUT.mkdir(exist_ok=True)
    for filename, frame in [
        ("daily.csv", daily), ("weights.csv", weights),
        ("attribution.csv", attribution), ("trades.csv", trades),
        ("risk.csv", risk),
    ]:
        frame.to_csv(OUTPUT / filename, index=(filename != "trades.csv"))
    if treasury is not None:
        treasury.to_csv(OUTPUT / "treasury_yields.csv")

    duration = float(risk.duration_contribution.sum())
    summary["illustrative_duration_years"] = duration
    summary["illustrative_dv01_usd"] = float(risk.dv01_usd.sum())
    summary["risk_inputs"] = "ILLUSTRATIVE effective durations; not dated issuer observations"
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))

    def pct(x):
        return f"{100 * x:.2f}%"

    report = f"""# Fixed Income Portfolio Review — {summary['backtest_end']}


## Mandate and method

- Initial hypothetical capital: $100 million; test dates: {summary['backtest_start']} to {summary['backtest_end']}.
- Custom benchmark (monthly rebalance): SHY 15%, IEF 35%, TLT 10%, LQD 30%, HYG 10%.
- Optimized portfolio: ≥70% Treasuries, ≤5% HYG, ≤45% any ETF, no shorting.
- Covariance: trailing 252 trading days; optimizer uses only data preceding each trade.
- Monthly rebalancing; assumed 5 bps transaction cost per dollar bought or sold.

## Backtested results (net of modeled portfolio trading costs)

| Metric | Portfolio | Custom benchmark | AGG reference |
|---|---:|---:|---:|
| Total return | {pct(summary['portfolio']['total_return'])} | {pct(summary['benchmark']['total_return'])} | {pct(summary['agg']['total_return'])} |
| Annualized return | {pct(summary['portfolio']['annualized_return'])} | {pct(summary['benchmark']['annualized_return'])} | {pct(summary['agg']['annualized_return'])} |
| Annualized volatility | {pct(summary['portfolio']['annualized_volatility'])} | {pct(summary['benchmark']['annualized_volatility'])} | {pct(summary['agg']['annualized_volatility'])} |
| Maximum drawdown | {pct(summary['portfolio']['maximum_drawdown'])} | {pct(summary['benchmark']['maximum_drawdown'])} | {pct(summary['agg']['maximum_drawdown'])} |

- Net annualized tracking error versus the custom benchmark: {pct(summary['tracking_error'])}.
- Annualized arithmetic active return: {pct(summary['annualized_active_return'])}.
- Information ratio: {summary['information_ratio'] if summary['information_ratio'] is not None else 'N/A'}.
- Modeled portfolio transaction costs across backtest: ${summary['total_transaction_cost_usd']:,.0f}.

## Interpretation

The optimization minimizes benchmark-relative risk subject to
portfolio allocation constraints and a turnover penalty. It does
not seek to maximize returns or generate alpha.

Over the evaluation period, the optimized portfolio earned a
cumulative return of {pct(summary['portfolio']['total_return'])},
compared with {pct(summary['benchmark']['total_return'])}
for the custom benchmark. Annualized volatility was
{pct(summary['portfolio']['annualized_volatility'])}
versus {pct(summary['benchmark']['annualized_volatility'])},
while maximum drawdown was
{pct(summary['portfolio']['maximum_drawdown'])}
versus {pct(summary['benchmark']['maximum_drawdown'])}.

The constrained portfolio experienced slightly lower realized
volatility and a smaller drawdown, but underperformed the benchmark
after modeled transaction costs. These are sample-specific outcomes,
not guaranteed properties of the optimization objective.


## Interest-rate stress test (illustrative inputs only)

- Portfolio value at test end: ${summary['end_nav']:,.0f}.
- Assumed effective duration: {duration:.2f} years.
- Assumed DV01: ${summary['illustrative_dv01_usd']:,.0f} per basis point.
- Parallel +100 bp shock: approximately ${-100 * summary['illustrative_dv01_usd']:,.0f}.
- Parallel −100 bp shock: approximately ${100 * summary['illustrative_dv01_usd']:,.0f}.

**Limitations:** ETF adjusted prices proxy total returns; transaction costs are hypothetical;
current duration assumptions are NOT historical fund attributes; scenario P&L is
first-order duration only, excluding convexity and spread changes. Daily ETF
attribution is additive within a day but is not a geometrically linked multi-period
attribution. ETF sector proxies are not the same as a security-level bond portfolio.
"""
    (OUTPUT / "portfolio_review.md").write_text(report)

    import plotly.graph_objects as go

    figure = go.Figure()
    for name, label in [("portfolio_index", "Optimized portfolio"),
                        ("benchmark_index", "Custom benchmark"),
                        ("agg_index", "AGG reference")]:
        figure.add_scatter(x=daily.index, y=daily[name], name=label)
    figure.update_layout(title="Growth of 100: adjusted ETF returns", yaxis_title="Index")
    figure.write_html(OUTPUT / "performance.html", include_plotlyjs="cdn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Download data again")
    args = parser.parse_args()
    prices = get_prices(args.refresh)
    try:
        treasury = get_treasury(args.refresh)
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Treasury data unavailable ({exc}); ETF backtest will still run.")
        treasury = None
    daily, weights, attribution, trades = run_backtest(prices)
    summary = summarize(daily)
    risk = current_risk(weights, summary)
    save_results(daily, weights, attribution, trades, summary, risk, treasury)
    print("Project completed. Open outputs/portfolio_review.md or run: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
