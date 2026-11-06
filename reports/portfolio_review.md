# Fixed Income Portfolio Review — 2026-10-30


## Mandate and method

- Initial hypothetical capital: $100 million; test dates: 2021-01-04 to 2026-10-30.
- Custom benchmark (monthly rebalance): SHY 15%, IEF 35%, TLT 10%, LQD 30%, HYG 10%.
- Optimized portfolio: ≥70% Treasuries, ≤5% HYG, ≤45% any ETF, no shorting.
- Covariance: trailing 252 trading days; optimizer uses only data preceding each trade.
- Monthly rebalancing; assumed 5 bps transaction cost per dollar bought or sold.

## Backtested results (net of modeled portfolio trading costs)

| Metric | Portfolio | Custom benchmark | AGG reference |
|---|---:|---:|---:|
| Total return | -7.21% | -6.63% | -2.86% |
| Annualized return | -1.28% | -1.17% | -0.50% |
| Annualized volatility | 6.78% | 7.01% | 5.84% |
| Maximum drawdown | -20.05% | -20.74% | -17.98% |

- Net annualized tracking error versus the custom benchmark: 0.52%.
- Annualized arithmetic active return: -0.12%.
- Information ratio: -0.23824800101710597.
- Modeled portfolio transaction costs across backtest: $22,252.

## Interpretation

The optimization minimizes benchmark-relative risk subject to
portfolio allocation constraints and a turnover penalty. It does
not seek to maximize returns or generate alpha.

Over the evaluation period, the optimized portfolio earned a
cumulative return of -7.21%,
compared with -6.63%
for the custom benchmark. Annualized volatility was
6.78%
versus 7.01%,
while maximum drawdown was
-20.05%
versus -20.74%.

The constrained portfolio experienced slightly lower realized
volatility and a smaller drawdown, but underperformed the benchmark
after modeled transaction costs. These are sample-specific outcomes,
not guaranteed properties of the optimization objective.


## Interest-rate stress test (illustrative inputs only)

- Portfolio value at test end: $92,793,354.
- Assumed effective duration: 7.03 years.
- Assumed DV01: $65,273 per basis point.
- Parallel +100 bp shock: approximately $-6,527,332.
- Parallel −100 bp shock: approximately $6,527,332.

**Limitations:** ETF adjusted prices proxy total returns; transaction costs are hypothetical;
current duration assumptions are NOT historical fund attributes; scenario P&L is
first-order duration only, excluding convexity and spread changes. Daily ETF
attribution is additive within a day but is not a geometrically linked multi-period
attribution. ETF sector proxies are not the same as a security-level bond portfolio.
