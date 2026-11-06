"""Run after `python portfolio.py`: `streamlit run dashboard.py`."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TICKERS = ["SHY", "IEF", "TLT", "LQD", "HYG"]

st.set_page_config(page_title="Fixed Income PM Analytics", layout="wide")
st.title("Fixed Income Portfolio Management")
st.caption("Research simulation • ETF proxies • Custom benchmark • $100m starting capital")

if not (OUT / "summary.json").exists():
    st.error("First run: python portfolio.py --refresh")
    st.stop()

summary = json.loads((OUT / "summary.json").read_text())
daily = pd.read_csv(OUT / "daily.csv", index_col=0, parse_dates=True)
weights = pd.read_csv(OUT / "weights.csv", index_col=0, parse_dates=True)
attribution = pd.read_csv(OUT / "attribution.csv", index_col=0, parse_dates=True)
trades = pd.read_csv(OUT / "trades.csv", parse_dates=["date"])
risk = pd.read_csv(OUT / "risk.csv", index_col="ticker")

st.caption(f"Out-of-sample: {summary['backtest_start']} to {summary['backtest_end']}. "
           "Duration inputs are illustrative, not issuer-reported historical observations.")

tab_performance, tab_risk, tab_rebalance, tab_attribution = st.tabs([
    "Performance", "Risk & Scenarios", "Rebalancing", "Attribution"
])

with tab_performance:
    a, b, c, d = st.columns(4)
    a.metric("Annualized portfolio return", f"{summary['portfolio']['annualized_return']:.2%}")
    b.metric("Annualized tracking error", f"{summary['tracking_error']:.2%}")
    c.metric("Annualized active return", f"{summary['annualized_active_return']:.2%}")
    d.metric("Max portfolio drawdown", f"{summary['portfolio']['maximum_drawdown']:.2%}")
    history = daily[["portfolio_index", "benchmark_index", "agg_index"]].rename(columns={
        "portfolio_index": "Optimized portfolio", "benchmark_index": "Custom benchmark",
        "agg_index": "AGG reference",
    })
    st.plotly_chart(px.line(history, title="Growth of 100 (adjusted ETF prices)"),
                    use_container_width=True)
    st.plotly_chart(px.line(daily[["portfolio_drawdown", "benchmark_drawdown"]],
                            title="Historical drawdowns"), use_container_width=True)
    st.caption("Custom benchmark rebalances monthly. Net portfolio performance deducts modeled trading costs.")

with tab_risk:
    a, b, c = st.columns(3)
    a.metric("Portfolio duration (assumed)", f"{summary['illustrative_duration_years']:.2f} years")
    b.metric("DV01 (assumed)", f"${summary['illustrative_dv01_usd']:,.0f}")
    c.metric("Current hypothetical NAV", f"${summary['end_nav']/1e6:,.2f}m")
    shock_bp = st.slider("Parallel rate shock, basis points", -200, 200, 100, 25)
    impact = -summary["illustrative_dv01_usd"] * shock_bp
    st.metric("First-order estimated P&L", f"${impact/1e6:+,.2f}m")
    st.warning("Scenario uses assumed ETF effective durations. It excludes convexity, "
               "credit-spread moves and changing ETF composition. Not a market-value forecast.")
    st.dataframe(risk.style.format({
        "portfolio_weight": "{:.2%}", "benchmark_weight": "{:.2%}",
        "effective_duration_assumption": "{:.2f}",
        "duration_contribution": "{:.2f}", "dv01_usd": "${:,.0f}",
    }), use_container_width=True)
    treasury_path = OUT / "treasury_yields.csv"
    if treasury_path.exists():
        yields = pd.read_csv(treasury_path, index_col=0, parse_dates=True)
        st.plotly_chart(px.line(yields, title="U.S. Treasury par yields (%)"),
                        use_container_width=True)
        st.caption("Treasury par yields are plotted for context, not treated as zero-coupon discount curves.")
    else:
        st.info("Treasury curve data were unavailable. Run `python portfolio.py --refresh` to retry.")

with tab_rebalance:
    current = risk[["portfolio_weight", "benchmark_weight"]].rename(columns={
        "portfolio_weight": "Portfolio", "benchmark_weight": "Drifted benchmark"
    })
    st.plotly_chart(px.bar(current, barmode="group", title="Latest portfolio vs benchmark weights"),
                    use_container_width=True)
    st.metric("Total traded weight over test", f"{summary['total_traded_weight']:.2f}x NAV")
    st.metric("Estimated trading costs across test", f"${summary['total_transaction_cost_usd']:,.0f}")
    st.subheader("Latest rebalance trades")
    if not trades.empty:
        last_date = trades["date"].max()
        st.dataframe(trades.loc[trades["date"] == last_date].sort_values("ticker"),
                     hide_index=True, use_container_width=True)
    st.caption("Mandate limits are imposed at each monthly rebalance; weights subsequently drift with returns.")

with tab_attribution:
    recent = st.selectbox("Attribution window", ["Entire backtest", "Last 252 observations"])
    contrib = attribution if recent == "Entire backtest" else attribution.tail(252)
    chart = (contrib[TICKERS].sum() * 100).rename("Contribution (percentage points)")
    st.plotly_chart(px.bar(chart, title="Sum of daily gross active return contributions"),
                    use_container_width=True)
    reconciliation = pd.DataFrame({
        "Daily ETF contributions": attribution[TICKERS].sum(axis=1),
        "Daily modeled costs": attribution["trading_cost"],
        "Daily net active return": daily["net_active_return"],
    })
    st.dataframe(reconciliation.tail(10).style.format("{:.5%}"), use_container_width=True)
    st.caption("Daily contributions reconcile to net active return after costs. "
               "Totals across dates are unlinked arithmetic contributions, not compounded attribution.")

st.caption("Source: Yahoo Finance adjusted ETF prices and optional U.S. Treasury par yields. "
           "Educational quantitative research portfolio; not a BlackRock system.")
