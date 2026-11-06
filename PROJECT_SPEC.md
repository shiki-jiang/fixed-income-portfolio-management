# Project scope and acceptance criteria — frozen version 1.0

**Purpose:** demonstrate fixed-income portfolio management analytics for an associate-level investment/portfolio-management application. Not employer-affiliated.

**The five required capabilities:**
1. Daily adjusted ETF returns and a custom monthly rebalanced benchmark (SHY/IEF/TLT/LQD/HYG, 15/35/10/30/10).
2. Monthly optimization: track benchmark risk with strictly trailing 252-day return covariance, constrained to ≥70% Treasuries, ≤5% HYG, ≤45% single ETF, long-only, sum 100%, turnover penalty.
3. Net historical walk-forward portfolio with 5 bp each side transaction costs and 2021–2026 out-of-sample window, starting from $100m hypothetical capital.
4. Daily return attribution, current portfolio weights, duration/DV01 and +/− parallel rate stresses; issuer duration data not falsely implied by illustrative inputs.
5. Four-tab Streamlit dashboard, machine-readable results, one-page portfolio review and tests.

**Pass/fail checks:** optimizer constraints; daily attribution plus costs equals daily net active return; weights sum to one; no lookahead in first rebalance; downloadable sources validated; clear label for hypothetical data and duration inputs.

**Out of scope in version 1:** Bloomberg/Aladdin feeds, real index security holdings, individual bond cashflows, historical ETF duration, real trade execution, credit spread duration, key-rate hedging, current market live data, predicting returns. Only add these as clearly labeled future extensions.

If continuing in another chat, upload this file plus `portfolio.py`, `dashboard.py`, `tests/test_portfolio.py`, and describe requested changes. Treat this document as authoritative over generic suggestions.
