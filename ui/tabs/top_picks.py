"""
Top Picks Tab

Auto-scan for best trading opportunities (Intraday + Swing).
Uses ComprehensiveAnalyzer for consistent results.

NOTE: This is a modular version of the top picks functionality.
The full implementation is in the main ui/app.py file.
This module provides the interface for future modularization.
"""

import streamlit as st

# Import shared components
from ui.components.common import (
    get_comprehensive_analyzer,
    get_unified_prediction,
    generate_stock_insight,
    render_enhanced_signal_box,
)


def render_top_picks_tab():
    """
    Render the Top Picks tab.

    NOTE: For full functionality, the original app.py should be used.
    This is a placeholder for the modular architecture.
    """
    st.header("Top Picks - Auto Scan")

    st.markdown("""
    ### Scan for BUY Signals

    This tab scans 70+ stocks and identifies the best trading opportunities
    using the 4-model ensemble (Base + Physics + Math + Regime).
    """)

    # Get analyzer
    analyzer = get_comprehensive_analyzer()

    if analyzer is None:
        st.error("Comprehensive Analyzer not available. Please check the installation.")
        return

    # Scan button
    if st.button("Scan All Stocks for BUY Signals", type="primary"):
        st.info("For full scanning functionality, please use the main ui/app.py")
        st.markdown("""
        **How to run the full scanner:**

        ```bash
        streamlit run ui/app.py
        ```

        The full scanner will:
        1. Analyze 70+ stocks (NIFTY 50 + High-Liquidity Midcaps)
        2. Apply strict filtering criteria
        3. Use the centralized 4-model ensemble
        4. Show Intraday and Swing picks separately
        """)

    # Documentation
    with st.expander("What This Tab Does"):
        st.markdown("""
        ### Features

        1. **Auto-Scans** 70+ stocks (NIFTY 50 + High-Liquidity Midcaps)
        2. **Uses 4-Model Ensemble** (Base + Physics + Math + Regime)
        3. **Filters** only BUY and STRONG BUY signals
        4. **Shows Two Sections:**
           - **Intraday Picks** - For same-day trading
           - **Swing Picks** - For 10-20 day holding period

        ### Filtering Criteria

        - Score >= 0.60 (60% confidence)
        - 2+ models must agree
        - ATR < 5% (volatility cap)
        - Liquidity > Rs10 Cr daily
        - Not in choppy regime
        """)
