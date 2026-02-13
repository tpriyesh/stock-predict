"""
Analysis Tabs

Deep analysis and sector analysis tabs.
Uses ComprehensiveAnalyzer for consistent results.

NOTE: This is a modular version of the analysis functionality.
The full implementation is in the main ui/app.py file.
"""

import streamlit as st

from ui.components.common import (
    get_comprehensive_analyzer,
    get_unified_prediction,
    render_deep_ai_analysis,
)


def render_comprehensive_analysis_tab():
    """
    Render the Deep Analysis tab for single stock analysis.

    NOTE: For full functionality, the original app.py should be used.
    """
    st.header("Deep Analysis")

    # Stock input
    symbol = st.text_input(
        "Enter Stock Symbol",
        placeholder="e.g., RELIANCE, TCS, INFY",
        help="Enter NSE stock symbol without .NS suffix"
    ).upper().strip()

    if not symbol:
        st.info("Enter a stock symbol above to analyze")
        return

    # Get analyzer
    analyzer = get_comprehensive_analyzer()
    if analyzer is None:
        st.error("Comprehensive Analyzer not available")
        return

    # Analyze button
    if st.button(f"Analyze {symbol}", type="primary"):
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                analysis = analyzer.analyze(symbol, skip_ai=True)
                if analysis:
                    st.success(f"Analysis complete for {symbol}")

                    # Show basic metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Price", f"Rs{analysis.technical.price:,.2f}")
                    with col2:
                        st.metric("RSI", f"{analysis.technical.rsi_14:.0f}")
                    with col3:
                        st.metric("P/E", f"{analysis.fundamental.pe_ratio:.1f}")
                    with col4:
                        st.metric("1M Change", f"{analysis.technical.change_1m:+.1f}%")

                    # Intraday recommendation
                    st.subheader("Intraday Recommendation")
                    rec = analysis.intraday_rec
                    if "BUY" in rec.signal.value:
                        st.success(f"**{rec.signal.value}** - Win Prob: {rec.win_probability:.0%}")
                    else:
                        st.info(f"**{rec.signal.value}** - Win Prob: {rec.win_probability:.0%}")

                    # Trade levels
                    lc1, lc2, lc3 = st.columns(3)
                    with lc1:
                        st.metric("Entry", f"Rs{rec.entry_price:,.2f}")
                    with lc2:
                        st.metric("Stop Loss", f"Rs{rec.stop_loss:,.2f}")
                    with lc3:
                        st.metric("Target", f"Rs{rec.target_1:,.2f}")
                else:
                    st.error(f"Could not analyze {symbol}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    # Documentation
    with st.expander("What This Tab Does"):
        st.markdown("""
        - Single stock comprehensive analysis
        - Technical indicators with interpretation
        - Fundamental analysis (P/E, ROE, Debt)
        - Risk metrics (Beta, Volatility, Drawdown)
        - Intraday and Swing recommendations
        - AI-powered insights (on-demand)
        """)


def render_sector_analysis_tab():
    """
    Render the Sector Analysis tab.

    NOTE: For full functionality, the original app.py should be used.
    """
    st.header("Sector Analysis")

    sectors = [
        "IT", "Banking", "Pharma", "Auto", "FMCG",
        "Metals", "Energy", "Infra", "Realty", "Telecom"
    ]

    selected_sector = st.selectbox("Select Sector", sectors)

    st.info(f"Sector analysis for {selected_sector} - Use main ui/app.py for full functionality")

    with st.expander("What This Tab Does"):
        st.markdown("""
        - Sector-wise stock analysis
        - Compare stocks within sectors
        - Identify sector leaders
        - Sector momentum analysis
        """)
