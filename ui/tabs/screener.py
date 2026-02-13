"""
Screener and About Tabs

Basic filtering and documentation tabs.

NOTE: This is a modular version of the screener functionality.
The full implementation is in the main ui/app.py file.
"""

import streamlit as st

from ui.components.common import get_stock_screener, get_improved_screener


def render_unified_screener_tab():
    """
    Render the Screener tab for basic stock filtering.

    NOTE: For full functionality, the original app.py should be used.
    """
    st.header("Stock Screener")

    st.markdown("""
    Filter stocks by various criteria. This is a basic screener without
    recommendations - use the Top Picks tab for trading signals.
    """)

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        price_range = st.selectbox(
            "Price Range",
            ["All", "Under Rs50", "Rs50-200", "Rs200-500", "Rs500-1000", "Above Rs1000"]
        )

    with col2:
        market_cap = st.selectbox(
            "Market Cap",
            ["All", "Large Cap", "Mid Cap", "Small Cap"]
        )

    # Get screener
    screener = get_stock_screener()

    if st.button("Screen Stocks", type="primary"):
        if screener is None:
            st.error("Stock Screener not available")
        else:
            st.info("For full screening functionality, please use the main ui/app.py")
            st.markdown("""
            **How to run the full screener:**

            ```bash
            streamlit run ui/app.py
            ```

            The full screener supports:
            - Price range filtering
            - Market cap filtering
            - Momentum screening
            - Volume filtering
            - Sector filtering
            """)

    with st.expander("What This Tab Does"):
        st.markdown("""
        - Filter stocks by price range
        - Filter by market cap
        - Filter by momentum
        - Basic screening without recommendations
        """)


def render_about_tab():
    """
    Render the About tab with documentation.
    """
    st.header("About Stock Prediction Engine")

    st.markdown("""
    ### Architecture

    The Stock Prediction Engine uses a **4-model ensemble** approach:

    | Model | Weight | Components |
    |-------|--------|------------|
    | Base Technical | 35% | RSI, MACD, Bollinger Bands, Moving Averages |
    | Physics Engine | 25% | Momentum conservation, Spring reversion, Energy clustering |
    | Math Engine | 20% | Fourier cycles, Hurst exponent, Entropy analysis |
    | Regime Detection | 20% | HMM-based market regime identification |

    ### Signal Generation

    | Signal | Criteria |
    |--------|----------|
    | STRONG BUY | Score >= 0.70, 3+ models agree |
    | BUY | Score >= 0.60, 2+ models agree |
    | HOLD | Score in neutral zone (0.45-0.60) |
    | SELL | *Disabled* - Backtest showed 28.6% accuracy (inverted) |

    ### Calibration

    Based on backtesting 330 trades:
    - BUY Signal Accuracy: **57.6%**
    - Calibration Error: **1.9%**

    ### Key Files

    | File | Description |
    |------|-------------|
    | `config/thresholds.py` | Centralized thresholds |
    | `src/models/scoring.py` | Basic scoring engine |
    | `src/models/enhanced_scoring.py` | 4-model ensemble |
    | `src/core/advanced_engine.py` | Advanced mathematical models |
    | `src/engines/advanced_predictor.py` | Domain-based prediction |

    ### UI Structure

    The UI is organized into modular components:

    ```
    ui/
    +-- app.py              # Main entry point
    +-- components/
    |   +-- common.py       # Shared components
    +-- tabs/
        +-- top_picks.py    # Top Picks tab
        +-- analysis.py     # Analysis tabs
        +-- screener.py     # Screener tab
    ```

    ### Links

    - [NSE India](https://www.nseindia.com)
    - [MoneyControl](https://www.moneycontrol.com)
    - [TradingView](https://www.tradingview.com)
    """)
