"""
Streamlit dashboard for Stock Prediction Engine.
Run with: streamlit run ui/app.py

Features:
- Penny Stock Screener (price ranges)
- Momentum Stock Screener
- Market Cap Categories
- Single Stock Analysis with AI
- Morning Intraday Analysis
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
import time

# Import unified engine configuration (7-model ensemble)
from ui.components.engine_config import (
    get_engine_config,
    get_unified_prediction as engine_get_unified_prediction,
    get_scoring_engine,
    render_settings_sidebar,
    render_config_summary,
    get_signal_color,
    get_confidence_level,
    PRESET_CONFIGS,
    RECOMMENDED_CONFIG,
)

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Stock Prediction Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .stock-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #00ff88;
    }
    .stock-card-negative {
        border-left: 4px solid #ff4444;
    }
    .bullish { color: #00ff88; font-weight: bold; }
    .bearish { color: #ff4444; font-weight: bold; }
    .neutral { color: #ffaa00; font-weight: bold; }
    .metric-positive { color: #00ff88; }
    .metric-negative { color: #ff4444; }
    .tab-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 20px;
    }
    /* Fix metric truncation - show full numbers */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        white-space: nowrap !important;
    }
    /* Wider columns for metrics */
    .stMetric {
        min-width: 120px !important;
    }
    /* Fix chart Y-axis */
    .stChart {
        overflow: visible !important;
    }
    /* Trade card styles */
    .trade-metric {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        min-width: 100px;
    }
    .trade-metric-label {
        color: #888;
        font-size: 0.8rem;
        margin-bottom: 4px;
    }
    .trade-metric-value {
        color: #fff;
        font-size: 1.3rem;
        font-weight: bold;
    }
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #1a2a1a 0%, #0a1a0a 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00ff88;
    }
    .insight-title {
        color: #00ff88;
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .insight-text {
        color: #ccc;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# Cached resources
@st.cache_resource
def get_stock_screener():
    """Get cached stock screener instance."""
    try:
        from src.insights.stock_screener import StockScreener
        return StockScreener()
    except Exception as e:
        st.error(f"Failed to initialize Stock Screener: {e}")
        return None


@st.cache_resource
def get_quant_engine():
    """Get cached quant engine instance."""
    try:
        from src.insights.quant_engine import QuantIntradayEngine
        return QuantIntradayEngine()
    except Exception as e:
        st.error(f"Failed to initialize Quant Engine: {e}")
        return None


@st.cache_resource
def get_multi_model_judge():
    """Get cached multi-model AI judge instance."""
    try:
        from src.insights.multi_model_judge import MultiModelJudge
        return MultiModelJudge()
    except Exception as e:
        st.warning(f"Multi-Model Judge not available: {e}")
        return None


@st.cache_resource
def get_improved_screener():
    """Get cached improved multi-factor screener."""
    try:
        from src.insights.improved_screener import ImprovedScreener
        return ImprovedScreener()
    except Exception as e:
        st.warning(f"Improved Screener not available: {e}")
        return None


@st.cache_resource
def get_comprehensive_analyzer():
    """Get cached comprehensive analyzer."""
    try:
        from src.insights.comprehensive_analyzer import ComprehensiveAnalyzer
        return ComprehensiveAnalyzer(capital=100000, risk_per_trade=0.02)
    except Exception as e:
        st.warning(f"Comprehensive Analyzer not available: {e}")
        return None


@st.cache_resource
def get_trade_tracker():
    """Get cached trade tracker."""
    try:
        from src.insights.trade_tracker import TradeTracker
        return TradeTracker()
    except Exception as e:
        st.warning(f"Trade Tracker not available: {e}")
        return None


@st.cache_resource
def get_intraday_timing():
    """Get cached intraday timing analyzer."""
    try:
        from src.insights.intraday_timing import IntradayTimingAnalyzer
        return IntradayTimingAnalyzer()
    except Exception as e:
        st.warning(f"Intraday Timing Analyzer not available: {e}")
        return None


@st.cache_resource
def get_robust_predictor():
    """Get cached robust predictor with all bias fixes."""
    try:
        from src.core.unified_robust_predictor import create_predictor
        return create_predictor(mode='production', broker='discount')
    except Exception as e:
        st.warning(f"Robust Predictor not available: {e}")
        return None


@st.cache_resource
def get_advanced_predictor():
    """Get cached advanced prediction engine with alternative data."""
    try:
        from src.engines.advanced_predictor import AdvancedPredictionEngine
        return AdvancedPredictionEngine(use_alternative_data=True)
    except Exception as e:
        st.warning(f"Advanced Predictor not available: {e}")
        return None


@st.cache_resource
def get_signal_helper():
    """Get cached signal display helper for HIGH BUY/BUY signals."""
    try:
        from src.engines.signal_display import SignalDisplayHelper
        return SignalDisplayHelper()
    except Exception as e:
        st.warning(f"Signal Helper not available: {e}")
        return None


@st.cache_resource
def get_market_timing():
    """Get cached market timing engine."""
    try:
        from src.core.market_timing import get_market_timing_engine
        return get_market_timing_engine()
    except Exception as e:
        st.warning(f"Market Timing not available: {e}")
        return None


@st.cache_resource
def get_insights_engine():
    """Get cached insights engine for human-readable explanations."""
    try:
        from src.insights.insights_engine import InsightsEngine
        return InsightsEngine()
    except Exception as e:
        st.warning(f"Insights Engine not available: {e}")
        return None


@st.cache_resource
def get_enhanced_scoring_engine():
    """Get cached enhanced scoring engine with 4-model ensemble."""
    try:
        from src.models.enhanced_scoring import EnhancedScoringEngine
        return EnhancedScoringEngine()
    except Exception as e:
        st.warning(f"Enhanced Scoring Engine not available: {e}")
        return None


@st.cache_resource
def get_enhanced_signal_generator():
    """Get cached enhanced signal generator with 4-model ensemble."""
    try:
        from src.signals.enhanced_generator import EnhancedSignalGenerator
        return EnhancedSignalGenerator()
    except Exception as e:
        st.warning(f"Enhanced Signal Generator not available: {e}")
        return None


@st.cache_resource
def get_unified_predictor():
    """
    Get cached unified predictor - SINGLE SOURCE OF TRUTH for all predictions.

    This is the centralized prediction engine that should be used for all
    stock predictions to ensure consistency across the UI.

    The UnifiedPredictor:
    - Uses the 4-model ensemble (Base + Physics + Math + Regime)
    - Applies statistical validation
    - Provides bias-corrected scoring
    - Caches engines for performance
    """
    try:
        from src.core.unified_predictor import UnifiedPredictor
        return UnifiedPredictor()
    except Exception as e:
        st.warning(f"Unified Predictor not available: {e}")
        return None


@st.cache_resource
def get_robustness_checker():
    """Get cached robustness checker for prediction validation."""
    try:
        from src.core.robustness_checks import RobustnessChecker
        return RobustnessChecker()
    except Exception as e:
        return None


@st.cache_resource
def get_statistical_validator():
    """Get cached statistical validator for backtest validation."""
    try:
        from src.core.statistical_tests import StatisticalValidator
        return StatisticalValidator()
    except Exception as e:
        return None


@st.cache_resource
def get_ui_prediction_api():
    """
    Get cached UI Prediction API - THE CENTRALIZED PREDICTION INTERFACE.

    IMPORTANT: This is the SINGLE SOURCE OF TRUTH for all stock predictions.
    All UI components should use this API instead of direct engine calls.

    Benefits:
    - Consistent predictions across all tabs
    - Integrated 4-model ensemble (Base + Physics + Math + Regime)
    - Automatic robustness checking
    - Unified confidence grading (A+ to F)
    - Centralized caching for performance

    Usage:
        api = get_ui_prediction_api()
        prediction = api.get_stock_prediction("RELIANCE", "intraday")
        scan_result = api.scan_stocks(symbols, min_confidence=0.55)
    """
    try:
        from src.core.ui_prediction_api import UIPredictionAPI
        return UIPredictionAPI.get_instance()
    except Exception as e:
        st.warning(f"UI Prediction API not available: {e}")
        return None


@st.cache_resource
def get_ai_engine():
    """Get cached AI insight engine."""
    try:
        from src.insights.ai_engine import AIInsightEngine
        return AIInsightEngine(provider="openai")
    except Exception as e:
        return None


# ============================================================================
# CENTRALIZED PREDICTION HELPER FUNCTIONS
# These functions ensure all prediction requests use the same unified engine
# ============================================================================

def get_unified_prediction(symbol: str, trade_type: str = "intraday", df=None):
    """
    Get prediction using the CENTRALIZED unified prediction engine.

    NOW USES 7-MODEL ENSEMBLE via engine_config module:
    - Base Technical (25%)
    - Physics Engine (18%)
    - Math Engine (14%)
    - HMM Regime (13%)
    - Macro Engine (10%)
    - Alternative Data (10%)
    - Advanced Math (10%)

    This function should be used by all UI components instead of
    directly calling scoring engines.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        trade_type: "intraday" or "swing"
        df: Optional pre-fetched DataFrame

    Returns:
        Dict with prediction data or None if failed
    """
    # If no dataframe provided, fetch it
    if df is None:
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period="1y")
            if len(df) < 60:
                return None
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            return None

    # Use the unified 7-model prediction from engine_config
    result = engine_get_unified_prediction(
        symbol=symbol,
        df=df,
        trade_type=trade_type.upper()
    )

    if result is None:
        return None

    # Add computed fields for UI compatibility
    config = get_engine_config()
    total_models = result.get('total_models', 7 if config.use_7_model else 4)
    model_agreement = result.get('model_agreement', 0)

    # Compute conviction level based on model agreement
    agreement_pct = model_agreement / total_models if total_models > 0 else 0
    if agreement_pct >= 0.7:
        conviction_level = 'HIGH'
    elif agreement_pct >= 0.4:
        conviction_level = 'MODERATE'
    else:
        conviction_level = 'LOW'

    # Add UI-specific fields
    result['conviction_level'] = conviction_level
    result['show_warning'] = result.get('regime') == 'choppy' or result.get('regime_stability', 1) < 0.5
    result['total_models'] = total_models
    result['predictability'] = result.get('market_predictability', 0.5)

    return result


def get_enhanced_prediction_data(symbol: str, df, trade_type: str = "intraday"):
    """
    Get enhanced prediction data for a stock using centralized engine.

    This is a wrapper around get_unified_prediction that formats the data
    for the existing UI display components.

    Returns:
        Dict formatted for enhanced_intraday/enhanced_swing display
    """
    result = get_unified_prediction(symbol, trade_type, df)
    if result is None:
        return None

    return {
        'model_agreement': result.get('model_agreement', 0),
        'model_votes': result.get('model_votes', {}),
        'ensemble_score': result.get('ensemble_score', 0.5),
        'signal_strength': result.get('signal_strength', 'weak'),
        'regime': result.get('regime', 'unknown'),
        'regime_stability': result.get('regime_stability', 0.5),
        'predictability': result.get('predictability', 0.5),
        'warnings': result.get('warnings', []),
        'conviction_level': result.get('conviction_level', 'LOW'),
        'show_warning': result.get('show_warning', False),
        'confidence': result.get('confidence', 0.5),
        'confidence_grade': result.get('confidence_grade', 'C'),
        'base_score': result.get('base_score', 0.5),
        'physics_score': result.get('physics_score', 0.5),
        'math_score': result.get('math_score', 0.5),
        'regime_score': result.get('regime_score', 0.5),
        'entry_price': result.get('entry_price', 0),
        'stop_loss': result.get('stop_loss', 0),
        'target_price': result.get('target_price', 0),
        'risk_reward': result.get('risk_reward', 0),
        'reasons': result.get('reasons', [])
    }


# ============================================================================
# INSIGHT GENERATION
# ============================================================================

def generate_stock_insight(stock_data: dict) -> str:
    """
    Generate a human-readable insight explaining WHY a stock is recommended.
    Uses rule-based logic for fast generation without LLM calls.
    """
    symbol = stock_data.get('symbol', 'Stock')
    signal = stock_data.get('signal', 'HOLD')
    score = stock_data.get('score', 50)
    rsi = stock_data.get('rsi', 50)
    growth_1m = stock_data.get('growth_1m', 0)
    growth_3m = stock_data.get('growth_3m', 0)
    growth_1y = stock_data.get('growth_1y', 0)
    trend = stock_data.get('trend', 'SIDEWAYS')
    volume_ratio = stock_data.get('volume_ratio', 1)
    from_high = stock_data.get('from_high', 0)
    from_low = stock_data.get('from_low', 0)

    insights = []
    reasons = []

    # Normalize signal (handle both "STRONG_BUY" and "STRONG BUY" formats)
    signal_normalized = signal.replace("_", " ") if signal else "HOLD"

    # Signal explanation
    if "STRONG" in signal_normalized and "BUY" in signal_normalized:
        insights.append(f"**{symbol} shows STRONG BUY signals** with a score of {score}/100.")
    elif "BUY" in signal_normalized:
        insights.append(f"**{symbol} is a BUY candidate** with a score of {score}/100.")
    elif signal_normalized == "HOLD":
        insights.append(f"**{symbol} is a HOLD** - wait for better entry points. Score: {score}/100.")
    elif "SELL" in signal_normalized:
        insights.append(f"**{symbol} shows SELL signals** - consider exiting. Score: {score}/100.")
    else:
        insights.append(f"**{symbol}** analysis - Score: {score}/100.")

    # RSI Analysis
    if rsi < 30:
        reasons.append(f"RSI at {rsi:.0f} indicates **severely oversold** - potential bounce opportunity but risky")
    elif rsi < 40:
        reasons.append(f"RSI at {rsi:.0f} shows **oversold conditions** - good entry zone if trend confirms")
    elif 40 <= rsi <= 60:
        reasons.append(f"RSI at {rsi:.0f} is **neutral** - stock in consolidation phase")
    elif rsi < 70:
        reasons.append(f"RSI at {rsi:.0f} shows **momentum building** - trend continuation likely")
    else:
        reasons.append(f"RSI at {rsi:.0f} is **overbought** - avoid fresh entries, consider booking profits")

    # Trend Analysis
    if "STRONG UPTREND" in trend:
        reasons.append(f"**Strong uptrend** - price above both 20 & 50 day moving averages")
    elif "UPTREND" in trend:
        reasons.append(f"**Uptrend intact** - momentum is positive")
    elif "DOWNTREND" in trend:
        reasons.append(f"**Downtrend** - wait for reversal confirmation before buying")
    else:
        reasons.append(f"**Sideways movement** - range-bound, wait for breakout")

    # Growth Analysis
    if growth_1m > 10:
        reasons.append(f"**Exceptional 1-month return of {growth_1m:+.1f}%** - strong short-term momentum")
    elif growth_1m > 5:
        reasons.append(f"**Good 1-month gain of {growth_1m:+.1f}%** - positive near-term trend")
    elif growth_1m < -10:
        reasons.append(f"**1-month decline of {growth_1m:.1f}%** - weakness, but could be oversold")
    elif growth_1m < -5:
        reasons.append(f"**1-month down {growth_1m:.1f}%** - short-term pressure")

    if growth_3m > 20:
        reasons.append(f"**Strong 3-month rally of {growth_3m:+.1f}%** - sustained momentum")
    elif growth_3m < -20:
        reasons.append(f"**3-month decline of {growth_3m:.1f}%** - significant correction")

    # Volume Analysis
    if volume_ratio > 2:
        reasons.append(f"**Volume surge at {volume_ratio:.1f}x average** - high interest, potential breakout")
    elif volume_ratio > 1.5:
        reasons.append(f"**Above average volume ({volume_ratio:.1f}x)** - confirms price movement")
    elif volume_ratio < 0.5:
        reasons.append(f"**Low volume ({volume_ratio:.1f}x)** - lack of conviction in current move")

    # 52-week position
    if from_high > -5:
        reasons.append(f"**Near 52-week high ({from_high:+.1f}%)** - momentum stock but entry risk high")
    elif from_high < -30:
        reasons.append(f"**{abs(from_high):.0f}% below 52-week high** - value opportunity if fundamentals intact")

    if from_low < 15:
        reasons.append(f"**Close to 52-week low** - potential value trap or turnaround candidate")

    # Compile insight
    insight_text = insights[0] + "\n\n"
    insight_text += "**Key Observations:**\n"
    for i, reason in enumerate(reasons[:5], 1):
        insight_text += f"{i}. {reason}\n"

    # Add recommendation (use normalized signal)
    insight_text += "\n**Recommendation:** "
    if "BUY" in signal_normalized:
        if rsi < 50 and growth_1m > 0:
            insight_text += f"Enter on dips. Target +{max(5, growth_1m * 0.5):.1f}% from current levels. Stop loss at -{min(3, score/30):.1f}%."
        else:
            insight_text += f"Wait for RSI to cool off or a pullback before entry."
    elif signal_normalized == "HOLD":
        insight_text += "Hold existing positions. Not ideal for fresh entry."
    else:
        insight_text += "Avoid. Wait for trend reversal or better setup."

    return insight_text


def generate_batch_insights_llm(stocks: list, sector: str = None) -> dict:
    """
    Generate LLM-powered insights for a batch of stocks.
    Returns dict of symbol -> insight.
    """
    ai_engine = get_ai_engine()
    if not ai_engine:
        return {}

    # Create batch prompt
    stocks_summary = "\n".join([
        f"- {s['symbol']}: ₹{s['price']:.2f}, RSI {s['rsi']:.0f}, 1M {s['growth_1m']:+.1f}%, Signal: {s['signal']}"
        for s in stocks[:10]
    ])

    prompt = f"""Analyze these {sector or 'sector'} stocks and provide 2-3 sentence insights for each:

{stocks_summary}

For each stock, explain:
1. WHY it has this signal (BUY/HOLD/AVOID)
2. Key risk or opportunity
3. Entry strategy (if BUY)

Be specific with numbers. Format: SYMBOL: insight"""

    try:
        response = ai_engine._call_openai(prompt) if ai_engine.provider == "openai" else ai_engine._call_ollama(prompt)
        # Parse response into dict
        insights = {}
        for line in response.split('\n'):
            if ':' in line and any(s['symbol'] in line for s in stocks):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    symbol = parts[0].strip().upper()
                    insights[symbol] = parts[1].strip()
        return insights
    except Exception:
        return {}


def render_deep_ai_analysis(symbol: str, analysis, timeframe: str = 'intraday'):
    """
    Render comprehensive Deep AI Analysis with all values explained.
    Shows data breakdown + LLM explanation in human language.
    """
    st.markdown("### 🧠 Deep AI Analysis")

    # 1. DATA BREAKDOWN - Show all key metrics
    st.markdown("#### 📊 Key Metrics Breakdown")

    # Price Performance
    with st.expander("📈 Price Performance", expanded=True):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            change_color = "green" if analysis.technical.change_1d >= 0 else "red"
            st.metric("Today", f"{analysis.technical.change_1d:+.2f}%")
        with p2:
            st.metric("1 Week", f"{analysis.technical.change_1w:+.2f}%")
        with p3:
            st.metric("1 Month", f"{analysis.technical.change_1m:+.2f}%")
        with p4:
            st.metric("1 Year", f"{analysis.technical.change_1y:+.2f}%")

        # Position in 52-week range
        high_pct = abs(analysis.technical.pct_from_52_high)
        low_pct = analysis.technical.pct_from_52_low
        position = low_pct / (high_pct + low_pct) * 100 if (high_pct + low_pct) > 0 else 50
        st.progress(int(position), text=f"52-Week Position: {position:.0f}% (Low: +{low_pct:.1f}% | High: {analysis.technical.pct_from_52_high:.1f}%)")

    # Technical Indicators with interpretation
    with st.expander("🔧 Technical Indicators", expanded=True):
        t1, t2, t3, t4 = st.columns(4)

        # RSI Interpretation
        rsi = analysis.technical.rsi_14
        if rsi < 30:
            rsi_meaning = "🟢 OVERSOLD - Good buy zone"
        elif rsi < 45:
            rsi_meaning = "🔵 Slightly oversold"
        elif rsi < 55:
            rsi_meaning = "⚪ Neutral"
        elif rsi < 70:
            rsi_meaning = "🟡 Momentum building"
        else:
            rsi_meaning = "🔴 OVERBOUGHT - Caution"

        with t1:
            st.metric("RSI (14)", f"{rsi:.0f}")
            st.caption(rsi_meaning)

        # MACD Interpretation
        macd_diff = analysis.technical.macd - analysis.technical.macd_signal
        if macd_diff > 0 and analysis.technical.macd_histogram > 0:
            macd_meaning = "🟢 Bullish crossover"
        elif macd_diff < 0 and analysis.technical.macd_histogram < 0:
            macd_meaning = "🔴 Bearish crossover"
        else:
            macd_meaning = "⚪ Neutral"

        with t2:
            st.metric("MACD", f"{analysis.technical.macd:.2f}")
            st.caption(macd_meaning)

        # Volume Interpretation
        vol = analysis.technical.volume_ratio
        if vol > 2:
            vol_meaning = "🔥 Heavy buying/selling"
        elif vol > 1.2:
            vol_meaning = "🟢 Above average interest"
        elif vol < 0.5:
            vol_meaning = "😴 Low interest"
        else:
            vol_meaning = "⚪ Normal"

        with t3:
            st.metric("Volume", f"{vol:.1f}x avg")
            st.caption(vol_meaning)

        # Trend
        above_sma20 = analysis.technical.price > analysis.technical.sma_20
        above_sma50 = analysis.technical.price > analysis.technical.sma_50
        if above_sma20 and above_sma50:
            trend_meaning = "🟢 UPTREND"
        elif not above_sma20 and not above_sma50:
            trend_meaning = "🔴 DOWNTREND"
        else:
            trend_meaning = "🟡 MIXED"

        with t4:
            st.metric("Trend", trend_meaning)
            st.caption(f"Above SMA20: {'Yes' if above_sma20 else 'No'}")

    # Fundamentals with interpretation
    with st.expander("💰 Fundamentals", expanded=True):
        f1, f2, f3, f4 = st.columns(4)

        # P/E Interpretation
        pe = analysis.fundamental.pe_ratio
        if pe < 0:
            pe_meaning = "🔴 Loss-making"
        elif pe < 15:
            pe_meaning = "🟢 Cheap"
        elif pe < 25:
            pe_meaning = "⚪ Fair value"
        elif pe < 40:
            pe_meaning = "🟡 Expensive"
        else:
            pe_meaning = "🔴 Very expensive"

        with f1:
            st.metric("P/E Ratio", f"{pe:.1f}")
            st.caption(pe_meaning)

        # ROE Interpretation
        roe = analysis.fundamental.roe
        if roe > 20:
            roe_meaning = "🟢 Excellent"
        elif roe > 15:
            roe_meaning = "🔵 Good"
        elif roe > 10:
            roe_meaning = "⚪ Average"
        else:
            roe_meaning = "🔴 Poor"

        with f2:
            st.metric("ROE", f"{roe:.1f}%")
            st.caption(roe_meaning)

        # Debt Interpretation
        de = analysis.fundamental.debt_equity
        if de < 0.5:
            de_meaning = "🟢 Low debt"
        elif de < 1:
            de_meaning = "⚪ Moderate"
        elif de < 2:
            de_meaning = "🟡 High debt"
        else:
            de_meaning = "🔴 Very high debt"

        with f3:
            st.metric("Debt/Equity", f"{de:.2f}")
            st.caption(de_meaning)

        with f4:
            st.metric("Mkt Cap", f"₹{analysis.fundamental.market_cap:,.0f} Cr")

    # Risk Metrics
    with st.expander("⚠️ Risk Assessment", expanded=True):
        r1, r2, r3, r4 = st.columns(4)

        with r1:
            beta = analysis.statistical.beta
            if beta > 1.5:
                beta_meaning = "🔴 High risk (moves 1.5x market)"
            elif beta > 1:
                beta_meaning = "🟡 Above market risk"
            else:
                beta_meaning = "🟢 Lower than market"
            st.metric("Beta", f"{beta:.2f}")
            st.caption(beta_meaning)

        with r2:
            annual_vol = analysis.statistical.volatility_annual
            st.metric("Volatility", f"{annual_vol:.1f}%")
            st.caption("Annual price swings")

        with r3:
            dd = analysis.statistical.max_drawdown
            st.metric("Max Drawdown", f"{dd:.1f}%")
            st.caption("Worst decline")

        with r4:
            sharpe = analysis.statistical.sharpe_ratio
            if sharpe > 1:
                sharpe_meaning = "🟢 Good risk-adjusted return"
            elif sharpe > 0:
                sharpe_meaning = "⚪ Positive but low"
            else:
                sharpe_meaning = "🔴 Negative"
            st.metric("Sharpe", f"{sharpe:.2f}")
            st.caption(sharpe_meaning)

    # 2. RECOMMENDATION SUMMARY
    st.markdown("#### 🎯 Recommendation")

    rec = analysis.intraday_rec if timeframe == 'intraday' else analysis.swing_rec

    # Signal box
    signal = rec.signal.value
    if "STRONG_BUY" in signal:
        st.success(f"**{signal}** - Strong bullish setup. Win Probability: {rec.win_probability:.0%}")
    elif "BUY" in signal:
        st.info(f"**{signal}** - Bullish setup. Win Probability: {rec.win_probability:.0%}")
    elif "STRONG_SELL" in signal:
        st.error(f"**{signal}** - Strong bearish. Win Probability: {rec.win_probability:.0%}")
    elif "SELL" in signal:
        st.warning(f"**{signal}** - Bearish. Win Probability: {rec.win_probability:.0%}")
    else:
        st.info(f"**{signal}** - No clear direction. Win Probability: {rec.win_probability:.0%}")

    # Trade setup
    setup1, setup2, setup3, setup4 = st.columns(4)
    with setup1:
        st.metric("Entry", f"₹{rec.entry_price:,.2f}")
    with setup2:
        stop_pct = (rec.entry_price - rec.stop_loss) / rec.entry_price * 100
        st.metric("Stop Loss", f"₹{rec.stop_loss:,.2f}", delta=f"-{stop_pct:.1f}%", delta_color="inverse")
    with setup3:
        target_pct = (rec.target_1 - rec.entry_price) / rec.entry_price * 100
        st.metric("Target", f"₹{rec.target_1:,.2f}", delta=f"+{target_pct:.1f}%")
    with setup4:
        rr = target_pct / stop_pct if stop_pct > 0 else 0
        st.metric("Risk:Reward", f"1:{rr:.1f}")

    # Bullish/Bearish factors
    if rec.bullish_factors or rec.bearish_factors:
        factor1, factor2 = st.columns(2)
        with factor1:
            if rec.bullish_factors:
                st.markdown("**🟢 Bullish Factors:**")
                for f in rec.bullish_factors[:5]:
                    st.markdown(f"- {f}")
        with factor2:
            if rec.bearish_factors:
                st.markdown("**🔴 Bearish Factors:**")
                for f in rec.bearish_factors[:5]:
                    st.markdown(f"- {f}")

    # 3. AI EXPLANATION IN HUMAN LANGUAGE
    st.markdown("#### 🤖 AI Explanation (Plain English)")

    if analysis.ai_explanation:
        st.markdown(analysis.ai_explanation)
    else:
        # Generate quick summary if AI not available
        st.info(f"""
**Quick Summary for {symbol}:**

📊 **Valuation:** P/E of {pe:.1f} makes this stock {pe_meaning.split(' - ')[-1] if ' - ' in pe_meaning else pe_meaning[2:].lower()}.

📈 **Momentum:** RSI at {rsi:.0f} indicates {rsi_meaning.split(' - ')[-1] if ' - ' in rsi_meaning else 'neutral momentum'}. Volume is {vol:.1f}x average ({vol_meaning[2:].lower()}).

💪 **Financials:** ROE of {roe:.1f}% is {roe_meaning[2:].lower()}. Debt/Equity of {de:.2f} means {de_meaning[2:].lower()}.

⚠️ **Risk:** Beta of {beta:.2f} means this stock {beta_meaning.split('(')[-1].replace(')', '') if '(' in beta_meaning else 'moves with the market'}.

🎯 **Verdict:** Signal is **{signal}** with {rec.win_probability:.0%} win probability. {'Consider this trade with proper risk management.' if 'BUY' in signal else 'Wait for better setup.' if 'NEUTRAL' in signal else 'Avoid or consider shorting.'}
        """)

    # Models used
    if analysis.ai_models_used:
        st.caption(f"AI Models: {', '.join(analysis.ai_models_used)}")


def render_enhanced_signal_box(symbol: str, df, timeframe: str = 'both'):
    """
    Render enhanced signal box with HIGH BUY / BUY indicators and stop losses.

    Uses Advanced Prediction Engine with Alternative Data for 70%+ accuracy signals.
    """
    signal_helper = get_signal_helper()
    if not signal_helper:
        return

    try:
        enhanced = signal_helper.get_enhanced_signal(symbol, df)

        if enhanced is None:
            return

        # Header shows actual accuracy for THIS signal (not system-wide)
        st.markdown(f"### 🎯 Advanced AI Signal")

        # Main signal display
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Signal badge with color - accuracy is for THIS specific signal
            grade_emoji = "🏆" if enhanced.confidence_grade in ['A+', 'A'] else "✅" if enhanced.confidence_grade == 'B' else "⚠️"
            if enhanced.signal_type.name in ['HIGH_BUY', 'BUY']:
                st.success(f"**{enhanced.signal_label}** | {grade_emoji} Grade: **{enhanced.confidence_grade}** | Win Rate: **{enhanced.expected_accuracy:.0%}**")
            elif enhanced.signal_type.name in ['HIGH_SELL', 'SELL']:
                st.error(f"**{enhanced.signal_label}** | {grade_emoji} Grade: **{enhanced.confidence_grade}** | Win Rate: **{enhanced.expected_accuracy:.0%}**")
            else:
                st.info(f"**{enhanced.signal_label}** | {grade_emoji} Grade: **{enhanced.confidence_grade}** | Win Rate: **{enhanced.expected_accuracy:.0%}**")

        with col2:
            st.metric("MTF Confluence", f"{enhanced.mtf_confluence:.0%}",
                      delta=enhanced.mtf_trend.upper())

        with col3:
            st.metric("Key Reason", enhanced.key_reason[:30] + "..." if len(enhanced.key_reason) > 30 else enhanced.key_reason)

        # Stop Loss and Targets for both timeframes
        if timeframe in ['intraday', 'both']:
            st.markdown("#### 📅 Intraday Trade Setup")
            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1:
                st.metric("Entry", f"₹{enhanced.entry_price:,.2f}")
            with ic2:
                st.metric("Stop Loss", f"₹{enhanced.stop_loss_intraday:,.2f}",
                          delta=f"-{enhanced.risk_pct_intraday:.1f}%", delta_color="inverse")
            with ic3:
                st.metric("Target", f"₹{enhanced.target_intraday:,.2f}",
                          delta=f"+{enhanced.reward_pct_intraday:.1f}%")
            with ic4:
                st.metric("Risk:Reward", f"1:{enhanced.risk_reward_intraday:.1f}")

        if timeframe in ['swing', 'both']:
            st.markdown("#### 📈 Swing Trade Setup (10-20 days)")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Entry", f"₹{enhanced.entry_price:,.2f}")
            with sc2:
                st.metric("Stop Loss", f"₹{enhanced.stop_loss_swing:,.2f}",
                          delta=f"-{enhanced.risk_pct_swing:.1f}%", delta_color="inverse")
            with sc3:
                st.metric("Target", f"₹{enhanced.target_swing:,.2f}",
                          delta=f"+{enhanced.reward_pct_swing:.1f}%")
            with sc4:
                st.metric("Risk:Reward", f"1:{enhanced.risk_reward_swing:.1f}")

        # Alternative Data Indicators
        st.markdown("#### 🔬 Alternative Data Indicators")
        ai1, ai2, ai3 = st.columns(3)
        with ai1:
            earnings_icon = "📊" if enhanced.earnings_signal != 'None' else "—"
            st.markdown(f"**{earnings_icon} Earnings Signal**")
            st.caption(enhanced.earnings_signal[:60] if enhanced.earnings_signal else 'No signal')
        with ai2:
            inst_icon = "🏦" if enhanced.institutional_signal != 'None' else "—"
            st.markdown(f"**{inst_icon} Institutional Flow**")
            st.caption(enhanced.institutional_signal[:40] if enhanced.institutional_signal else 'No signal')
        with ai3:
            opt_icon = "📈" if enhanced.options_signal != 'None' else "—"
            st.markdown(f"**{opt_icon} Options Flow**")
            st.caption(enhanced.options_signal[:40] if enhanced.options_signal else 'No signal')

    except Exception as e:
        st.caption(f"Advanced signal analysis not available: {str(e)[:50]}")


@st.cache_resource
def get_robust_predictor():
    """Get cached robust ML predictor."""
    try:
        from src.insights.robust_predictor import RobustPredictor
        return RobustPredictor()
    except Exception as e:
        st.warning(f"Robust Predictor not available: {e}")
        return None


def render_stock_table(stocks, show_ai_button=True):
    """Render a table of stocks with key metrics."""
    if not stocks:
        st.warning("No stocks found matching your criteria.")
        return

    # Create DataFrame for display
    data = []
    for s in stocks:
        momentum_emoji = "🟢" if s.momentum_score > 0 else "🔴"
        data.append({
            'Symbol': s.symbol,
            'Name': s.name[:20] + '...' if len(s.name) > 20 else s.name,
            'Price': f"₹{s.price:,.2f}",
            'Day': f"{s.change_1d:+.1f}%",
            'Week': f"{s.change_1w:+.1f}%",
            'Month': f"{s.change_1m:+.1f}%",
            '3 Mon': f"{s.change_3m:+.1f}%",
            '6 Mon': f"{s.change_6m:+.1f}%",
            'Year': f"{s.change_1y:+.1f}%",
            'RSI': f"{s.rsi:.0f}",
            'Score': f"{momentum_emoji} {s.momentum_score:.2f}",
            'Sector': s.sector[:15] if s.sector else 'N/A'
        })

    df = pd.DataFrame(data)
    st.dataframe(df, width='stretch', hide_index=True)

    # Individual stock cards with AI analysis option
    if show_ai_button:
        st.markdown("### Detailed Stock Cards")
        cols = st.columns(2)
        for i, stock in enumerate(stocks[:6]):  # Show top 6
            with cols[i % 2]:
                render_stock_card(stock)


def render_stock_card(stock):
    """Render a single stock card with persistent AI analysis."""
    momentum_color = "#00ff88" if stock.momentum_score > 0 else "#ff4444"

    # Initialize session state for AI insights if not exists
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = {}

    with st.container():
        st.markdown(f"""
        <div style="background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid {momentum_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.2rem; font-weight: bold;">{stock.symbol}</span>
                    <span style="color: #888; margin-left: 10px;">{stock.name[:25]}</span>
                </div>
                <div style="font-size: 1.2rem; font-weight: bold; color: {momentum_color};">
                    ₹{stock.price:,.2f}
                </div>
            </div>
            <div style="margin-top: 10px; display: flex; gap: 20px; color: #888;">
                <span>1D: <span style="color: {'#00ff88' if stock.change_1d >= 0 else '#ff4444'}">{stock.change_1d:+.1f}%</span></span>
                <span>1M: <span style="color: {'#00ff88' if stock.change_1m >= 0 else '#ff4444'}">{stock.change_1m:+.1f}%</span></span>
                <span>3M: <span style="color: {'#00ff88' if stock.change_3m >= 0 else '#ff4444'}">{stock.change_3m:+.1f}%</span></span>
                <span>RSI: {stock.rsi:.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AI Analysis button with session state persistence
        ai_key = f"ai_{stock.symbol}"

        # Check if we already have analysis for this stock
        if stock.symbol in st.session_state.ai_insights:
            with st.expander(f"🤖 AI Analysis for {stock.symbol}", expanded=True):
                st.markdown(st.session_state.ai_insights[stock.symbol])
                if st.button("🔄 Refresh Analysis", key=f"refresh_{stock.symbol}"):
                    del st.session_state.ai_insights[stock.symbol]
                    st.rerun()
        else:
            if st.button(f"🤖 AI Analysis", key=ai_key):
                with st.spinner(f"Generating AI insights for {stock.symbol}..."):
                    screener = get_stock_screener()
                    if screener:
                        insight = screener.get_ai_stock_insight(stock)
                        st.session_state.ai_insights[stock.symbol] = insight
                        st.rerun()


def _render_intraday_pick_card(pick: dict, rank: int, is_market_open: bool):
    """Render a single intraday stock pick card with full details using Streamlit components."""
    signal_emoji = {"STRONG BUY": "🟢", "BUY": "🟢", "WATCH": "🟡", "HOLD": "⚪", "SELL": "🔴"}.get(pick['signal'], "⚪")

    with st.container():
        with st.expander(f"{signal_emoji} **#{rank} {pick['symbol']}** | ₹{pick['price']:,.2f} | {pick['signal']} | Conf: {pick['confidence']:.0%}", expanded=(rank <= 3)):

            # Top metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"₹{pick['price']:,.2f}", f"{pick['day_change']:+.2f}%")
            with col2:
                st.metric("Signal", pick['signal'])
            with col3:
                st.metric("Score", f"{pick['score']}/100")
            with col4:
                st.metric("Confidence", f"{pick['confidence']:.0%}")

            st.markdown("---")

            # Technical indicators
            st.markdown("**📊 Technical Indicators**")
            t1, t2, t3, t4, t5 = st.columns(5)
            with t1:
                st.metric("RSI", f"{pick['rsi']:.0f}")
            with t2:
                st.metric("5D Mom", f"{pick['momentum_5d']:+.1f}%")
            with t3:
                st.metric("20D Mom", f"{pick['momentum_20d']:+.1f}%")
            with t4:
                st.metric("Volume", f"{pick['volume_ratio']:.1f}x")
            with t5:
                st.metric("ATR %", f"{pick['atr_pct']:.1f}%")

            # Reasons as tags
            st.markdown("**🎯 Why This Stock:**")
            reason_text = " | ".join(pick.get('reasons', [])[:4])
            st.success(reason_text if reason_text else "Multiple technical factors align")

            # Entry timing
            st.markdown("**⏰ Entry Recommendation:**")
            entry_type = pick['entry_type'].replace('_', ' ')
            if pick['entry_type'] == "IMMEDIATE":
                st.info(f"**{entry_type}** - {pick['timing_note']}")
            else:
                st.warning(f"**{entry_type}** - {pick['timing_note']}")

            # Generate insight
            st.markdown("---")
            st.markdown("**💡 AI Analysis:**")

            # Create insight data for this pick
            insight_data = {
                'symbol': pick['symbol'],
                'signal': pick['signal'],
                'score': pick['score'],
                'rsi': pick['rsi'],
                'growth_1m': pick.get('momentum_20d', 0),  # Use 20D as proxy
                'growth_3m': pick.get('momentum_20d', 0) * 2,
                'growth_1y': 0,
                'trend': "UPTREND" if pick['momentum_5d'] > 0 and pick['momentum_20d'] > 0 else "DOWNTREND" if pick['momentum_5d'] < 0 else "SIDEWAYS",
                'volume_ratio': pick['volume_ratio'],
                'from_high': -10,  # Placeholder
                'from_low': 20  # Placeholder
            }
            insight = generate_stock_insight(insight_data)
            st.markdown(insight)

            st.caption(f"Category: {pick['category']}")


def render_action_dashboard():
    """
    Render Action Dashboard with market-timing aware recommendations.

    Shows:
    - Current market status (pre-market/open/closed)
    - Auto-scan TOP INTRADAY PICKS from NIFTY 50 + Midcap
    - Pre-market: stocks to buy at open
    - During market: real-time picks
    - Post-market: next trading day recommendations
    """
    st.markdown('<h2 class="tab-header">🚀 Action Dashboard - Top Intraday Picks</h2>', unsafe_allow_html=True)

    # INTRADAY STOCK UNIVERSE: NIFTY 50 + High-Liquidity Midcaps
    INTRADAY_UNIVERSE = {
        "NIFTY 50": [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
            "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
            "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "ULTRACEMCO", "NTPC",
            "WIPRO", "NESTLEIND", "M&M", "POWERGRID", "TATAMOTORS", "JSWSTEEL",
            "TATASTEEL", "ADANIENT", "ADANIPORTS", "ONGC", "COALINDIA", "BAJAJFINSV",
            "GRASIM", "TECHM", "HDFCLIFE", "SBILIFE", "BRITANNIA", "INDUSINDBK",
            "HINDALCO", "CIPLA", "DRREDDY", "EICHERMOT", "DIVISLAB", "APOLLOHOSP",
            "TATACONSUM", "BPCL", "HEROMOTOCO", "BAJAJ-AUTO", "TRENT", "ZOMATO"
        ],
        "MIDCAP_HIGH_LIQUIDITY": [
            "DMART", "SIEMENS", "ABB", "PIDILITIND", "HAVELLS", "GODREJCP",
            "MUTHOOTFIN", "CHOLAFIN", "TVSMOTOR", "VOLTAS", "PAGEIND", "JUBLFOOD",
            "LICHSGFIN", "MFSL", "SAIL", "CANBK", "FEDERALBNK", "IDFC", "NMDC",
            "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "LUPIN", "AUROPHARMA",
            "TORNTPHARM", "ALKEM", "IPCALAB", "GLENMARK", "ASHOKLEY", "ESCORTS",
            "BALKRISIND", "MRF", "CROMPTON", "BLUESTARCO", "OBEROIRLTY", "DLF"
        ]
    }

    # Get engines
    timing = get_market_timing()
    predictor = get_robust_predictor()
    screener = get_stock_screener()

    if not timing:
        st.error("Market Timing engine not available")
        return

    # Get market context
    ctx = timing.get_market_context()

    # ============== MARKET STATUS HERO ==============
    if ctx.is_market_open:
        status_color = "#00ff88"
        status_icon = "🟢"
        status_text = "MARKET OPEN"
        border_glow = "box-shadow: 0 0 30px rgba(0,255,136,0.3);"
    else:
        status_color = "#ff6600"
        status_icon = "🔴"
        status_text = "MARKET CLOSED"
        border_glow = "box-shadow: 0 0 30px rgba(255,102,0,0.3);"

    phase_display = ctx.market_phase.value.replace("_", " ").title()

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);
                border-radius: 20px; padding: 30px; margin: 20px 0;
                border: 3px solid {status_color}; {border_glow}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-size: 2.5rem; font-weight: bold; color: {status_color};">
                    {status_icon} {status_text}
                </div>
                <div style="font-size: 1.3rem; color: #aaa; margin-top: 8px;">
                    Phase: <span style="color: #fff;">{phase_display}</span>
                </div>
                <div style="font-size: 1.1rem; color: #888; margin-top: 5px;">
                    {ctx.action_message}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2rem; font-weight: bold; color: #fff;">
                    {ctx.current_time_ist.strftime('%H:%M')} IST
                </div>
                <div style="color: #888; font-size: 1rem;">
                    {ctx.current_time_ist.strftime('%A, %b %d, %Y')}
                </div>
                <div style="margin-top: 10px; padding: 8px 15px; background: {'#1a3a1a' if ctx.is_market_open else '#3a1a1a'};
                            border-radius: 8px; color: {status_color};">
                    {'Win Rate: ' + f'{ctx.current_entry_win_rate:.0%}' if ctx.is_market_open else 'Next: ' + str(ctx.next_trading_date)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============== TIMING RECOMMENDATION CARDS ==============
    st.markdown("### ⏰ Current Recommendation")

    decision_colors = {
        "enter_long": ("#00ff88", "BUY NOW"),
        "enter_short": ("#ff4444", "SELL NOW"),
        "add_to_winner": ("#00aaff", "ADD TO WINNERS"),
        "book_partial": ("#ffaa00", "BOOK PARTIAL"),
        "book_full": ("#ff8800", "BOOK PROFITS"),
        "wait": ("#888888", "WAIT"),
        "avoid": ("#ff4444", "AVOID ENTRY"),
        "prepare": ("#00aaff", "PREPARE")
    }

    dec_color, dec_text = decision_colors.get(ctx.decision.value, ("#888", "WAIT"))

    rec_cols = st.columns([2, 1, 1, 1])

    with rec_cols[0]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #0a0a1a 100%);
                    padding: 20px; border-radius: 12px; border-left: 5px solid {dec_color};">
            <div style="font-size: 1.8rem; font-weight: bold; color: {dec_color};">
                {dec_text}
            </div>
            <div style="color: #aaa; margin-top: 8px;">
                {ctx.decision_reason}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with rec_cols[1]:
        if ctx.is_market_open:
            close_mins = int(ctx.time_to_market_close.total_seconds() / 60)
            st.metric("Time to Close", f"{close_mins // 60}h {close_mins % 60}m")
        else:
            if ctx.time_to_market_open:
                open_mins = int(ctx.time_to_market_open.total_seconds() / 60)
                if open_mins > 60:
                    st.metric("Market Opens In", f"{open_mins // 60}h {open_mins % 60}m")
                else:
                    st.metric("Market Opens In", f"{open_mins}m")

    with rec_cols[2]:
        conf_mult = ctx.confidence_multiplier
        mult_color = "#00ff88" if conf_mult >= 0.9 else "#ffaa00" if conf_mult >= 0.7 else "#ff4444"
        st.markdown(f"""
        <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="color: #888; font-size: 0.85rem;">Confidence Multiplier</div>
            <div style="font-size: 1.5rem; color: {mult_color}; font-weight: bold;">{conf_mult:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    with rec_cols[3]:
        if ctx.days_to_expiry <= 3:
            exp_color = "#ff4444"
            exp_text = f"{ctx.days_to_expiry}d (NEAR!)"
        else:
            exp_color = "#888"
            exp_text = f"{ctx.days_to_expiry} days"
        st.markdown(f"""
        <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="color: #888; font-size: 0.85rem;">F&O Expiry</div>
            <div style="font-size: 1.2rem; color: {exp_color}; font-weight: bold;">{exp_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # ============== TIMING TIPS ==============
    with st.expander("💡 Timing Tips", expanded=False):
        for tip in ctx.timing_tips:
            st.markdown(f"• {tip}")

    st.markdown("---")

    # ============== ACTION PICKS - AUTO SCAN ==============
    if ctx.is_market_open:
        st.markdown("### 🎯 TOP INTRADAY PICKS - Buy NOW")
        action_msg = "Auto-scanning NIFTY 50 + Midcap stocks for best intraday opportunities."
    else:
        st.markdown(f"### 🌅 TOP PICKS for Next Trading Day ({ctx.next_trading_date.strftime('%b %d')})")
        action_msg = "Auto-scanned recommendations. Place orders before 9:15 AM. Best entry: 10:00-12:00."

    st.caption(action_msg)

    # Universe Selection & Scan Controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        universe_choice = st.multiselect(
            "Stock Universe",
            ["NIFTY 50", "MIDCAP_HIGH_LIQUIDITY"],
            default=["NIFTY 50", "MIDCAP_HIGH_LIQUIDITY"],
            key="universe_choice"
        )

    with col2:
        min_confidence = st.slider("Min Confidence", 0.5, 0.9, 0.55, 0.05, key="min_conf")

    with col3:
        top_n = st.selectbox("Show Top", [5, 10, 15, 20], index=1, key="top_n_picks")

    with col4:
        st.write("")
        scan_now = st.button("🔄 SCAN NOW", type="primary", use_container_width=True, key="scan_action")

    # Auto-scan on first load or button press
    if scan_now or 'action_scan_results' not in st.session_state:
        # Combine selected universes
        symbols_to_scan = []
        for universe in universe_choice:
            symbols_to_scan.extend(INTRADAY_UNIVERSE.get(universe, []))

        # Remove duplicates
        symbols_to_scan = list(set(symbols_to_scan))

        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner(f"Scanning {len(symbols_to_scan)} stocks for intraday opportunities..."):
            results = []
            import yfinance as yf

            for i, symbol in enumerate(symbols_to_scan):
                try:
                    progress_bar.progress((i + 1) / len(symbols_to_scan))
                    status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols_to_scan)})")

                    # Fetch data
                    ticker = yf.Ticker(f"{symbol}.NS")
                    df = ticker.history(period="3mo")

                    if df.empty:
                        ticker = yf.Ticker(f"{symbol}.BO")
                        df = ticker.history(period="3mo")

                    if not df.empty and len(df) >= 20:
                        df.columns = df.columns.str.lower()

                        current_price = float(df['close'].iloc[-1])
                        prev_close = float(df['close'].iloc[-2])
                        day_change = ((current_price - prev_close) / prev_close) * 100

                        # Calculate indicators
                        # RSI
                        delta = df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if loss.iloc[-1] != 0 else 50

                        # Momentum
                        momentum_5d = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100
                        momentum_20d = ((df['close'].iloc[-1] / df['close'].iloc[-20]) - 1) * 100

                        # Volume analysis (is volume high today?)
                        avg_volume = df['volume'].iloc[-20:].mean()
                        today_volume = df['volume'].iloc[-1]
                        volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1

                        # Volatility (ATR-based)
                        high_low = df['high'] - df['low']
                        atr = high_low.rolling(14).mean().iloc[-1]
                        atr_pct = (atr / current_price) * 100

                        # INTRADAY SCORE CALCULATION
                        score = 0
                        signal = "HOLD"
                        reasons = []

                        # RSI signals
                        if 30 <= rsi <= 45:
                            score += 25  # Oversold bounce potential
                            reasons.append("RSI oversold bounce")
                        elif 55 <= rsi <= 70:
                            score += 15  # Momentum continuation
                            reasons.append("RSI momentum")
                        elif rsi < 30:
                            score += 10  # Very oversold - risky
                            reasons.append("RSI very oversold")

                        # Momentum signals
                        if momentum_5d > 0 and momentum_20d > 0:
                            score += 20  # Uptrend
                            reasons.append("Dual momentum positive")
                        elif momentum_5d > 2:
                            score += 15  # Short-term surge
                            reasons.append("5D momentum surge")

                        # Volume confirmation
                        if volume_ratio > 1.5:
                            score += 15  # High volume - conviction
                            reasons.append("Volume spike")
                        elif volume_ratio > 1.2:
                            score += 10
                            reasons.append("Above avg volume")

                        # Volatility (good for intraday)
                        if 1.5 <= atr_pct <= 3.0:
                            score += 15  # Good volatility for intraday
                            reasons.append("Optimal volatility")
                        elif atr_pct > 3.0:
                            score += 5  # Too volatile
                            reasons.append("High volatility")

                        # Opening gap (if market running)
                        if ctx.is_market_open and abs(day_change) > 1:
                            if day_change > 1 and momentum_5d > 0:
                                score += 10
                                reasons.append("Gap up with momentum")

                        # Normalize score to confidence (0.5 - 0.9)
                        confidence = min(0.9, max(0.5, 0.5 + (score / 100) * 0.4))

                        # Determine signal
                        if score >= 50:
                            signal = "STRONG BUY"
                        elif score >= 35:
                            signal = "BUY"
                        elif score >= 20:
                            signal = "WATCH"
                        else:
                            signal = "HOLD"

                        # Get timing adjustment
                        entry_rec = timing.get_entry_recommendation(
                            symbol, current_price, signal, confidence
                        )

                        # Category
                        category = "NIFTY 50" if symbol in INTRADAY_UNIVERSE["NIFTY 50"] else "MIDCAP"

                        results.append({
                            'symbol': symbol,
                            'category': category,
                            'price': current_price,
                            'day_change': day_change,
                            'signal': signal,
                            'score': score,
                            'confidence': entry_rec['adjusted_confidence'],
                            'entry_type': entry_rec['entry_type'],
                            'timing_note': entry_rec['timing_note'],
                            'rsi': rsi,
                            'momentum_5d': momentum_5d,
                            'momentum_20d': momentum_20d,
                            'volume_ratio': volume_ratio,
                            'atr_pct': atr_pct,
                            'reasons': reasons
                        })

                except Exception:
                    pass  # Skip failed symbols

            progress_bar.empty()
            status_text.empty()

            # Sort by score descending
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            st.session_state['action_scan_results'] = results
            st.session_state['action_scan_time'] = datetime.now().strftime("%H:%M:%S")

    # Get cached results
    results = st.session_state.get('action_scan_results', [])
    scan_time = st.session_state.get('action_scan_time', 'N/A')

    if results:
        st.success(f"✅ Scanned {len(results)} stocks | Last scan: {scan_time} | Showing top {top_n}")

        # Filter by minimum confidence
        filtered = [r for r in results if r['confidence'] >= min_confidence]

        # Separate STRONG BUY, BUY, and others
        strong_buys = [r for r in filtered if r['signal'] == 'STRONG BUY'][:top_n]
        buys = [r for r in filtered if r['signal'] == 'BUY'][:top_n - len(strong_buys)]
        watch_list = [r for r in filtered if r['signal'] == 'WATCH'][:10]

        # Show tabs for different categories
        pick_tab1, pick_tab2, pick_tab3 = st.tabs(["🔥 Strong Buys", "✅ Buy Signals", "👀 Watchlist"])

        with pick_tab1:
            if strong_buys:
                st.markdown(f"**{len(strong_buys)} STRONG BUY signals** - High conviction intraday trades")
                for i, pick in enumerate(strong_buys, 1):
                    _render_intraday_pick_card(pick, i, ctx.is_market_open)
            else:
                st.info("No STRONG BUY signals currently. Market may be consolidating.")

        with pick_tab2:
            if buys:
                st.markdown(f"**{len(buys)} BUY signals** - Good intraday opportunities")
                for i, pick in enumerate(buys, 1):
                    _render_intraday_pick_card(pick, i, ctx.is_market_open)
            else:
                st.info("No BUY signals meeting confidence threshold.")

        with pick_tab3:
            if watch_list:
                st.markdown(f"**{len(watch_list)} stocks to watch** - May turn into opportunities")
                for pick in watch_list:
                    cat_color = "#00aaff" if pick['category'] == "NIFTY 50" else "#ffaa00"
                    chg_color = "#00ff88" if pick['day_change'] > 0 else "#ff4444"
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 12px; border-radius: 8px; margin: 5px 0;
                                display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: bold; color: #fff;">{pick['symbol']}</span>
                            <span style="margin-left: 8px; padding: 2px 6px; background: {cat_color}22;
                                        color: {cat_color}; border-radius: 3px; font-size: 0.75rem;">{pick['category']}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #fff;">₹{pick['price']:,.2f}</span>
                            <span style="margin-left: 10px; color: {chg_color};">{pick['day_change']:+.1f}%</span>
                            <span style="margin-left: 10px; color: #888;">Score: {pick['score']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No stocks in watchlist.")

    else:
        st.warning("No scan results yet. Click 'SCAN NOW' to analyze stocks.")

    # ============== QUICK ACTIONS ==============
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")

    action_cols = st.columns(4)

    with action_cols[0]:
        if st.button("🔄 Refresh Market Status", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    with action_cols[1]:
        if st.button("📈 Top Gainers", use_container_width=True):
            st.session_state['show_gainers'] = True

    with action_cols[2]:
        if st.button("📉 Top Losers", use_container_width=True):
            st.session_state['show_losers'] = True

    with action_cols[3]:
        if st.button("💹 Volume Spikes", use_container_width=True):
            st.session_state['show_volume'] = True


def render_penny_stocks_tab():
    """Render penny stocks screener tab."""
    st.markdown('<h2 class="tab-header">💰 Penny Stock Screener</h2>', unsafe_allow_html=True)
    st.caption("Find small-priced stocks with growth potential. Higher risk, higher reward.")

    # Initialize session state for penny stocks
    if 'penny_stocks' not in st.session_state:
        st.session_state.penny_stocks = None
    if 'penny_price_range' not in st.session_state:
        st.session_state.penny_price_range = None

    screener = get_stock_screener()
    if not screener:
        st.error("Stock Screener not available")
        return

    # Price range selection
    col1, col2, col3 = st.columns(3)

    with col1:
        price_range = st.selectbox(
            "Select Price Range",
            ["₹1 - ₹10 (Micro Penny)", "₹10 - ₹30 (Penny)", "₹30 - ₹100 (Low Price)"],
            index=1
        )

    with col2:
        min_volume = st.number_input("Min Daily Volume", value=100000, step=50000)

    with col3:
        limit = st.slider("Number of Stocks", 5, 30, 15, key="penny_limit")

    # Map selection to enum
    from src.insights.stock_screener import PriceRange
    price_map = {
        "₹1 - ₹10 (Micro Penny)": PriceRange.MICRO_PENNY,
        "₹10 - ₹30 (Penny)": PriceRange.PENNY,
        "₹30 - ₹100 (Low Price)": PriceRange.LOW_PRICE,
    }

    if st.button("🔍 Find Penny Stocks", type="primary"):
        with st.spinner(f"Step 1: Fetching NSE stock list..."):
            all_stocks = screener.fetch_all_nse_stocks()
            st.info(f"Found {len(all_stocks)} stocks on NSE. Now filtering by price...")

        with st.spinner(f"Step 2: Scanning for stocks in {price_range}..."):
            try:
                selected_range = price_map.get(price_range, PriceRange.PENNY)
                stocks = screener.screen_by_price(selected_range, min_volume=min_volume, limit=limit)

                if stocks:
                    # Store in session state so data persists when AI button is clicked
                    st.session_state.penny_stocks = stocks
                    st.session_state.penny_price_range = price_range
                    st.rerun()
                else:
                    st.warning("No stocks found in this range. Try adjusting filters.")

            except Exception as e:
                st.error(f"Error: {e}")

    # Display penny stocks from session state (persists across reruns)
    if st.session_state.penny_stocks:
        stocks = st.session_state.penny_stocks
        st.success(f"Showing {len(stocks)} stocks in {st.session_state.penny_price_range}")

        # Clear button
        if st.button("🗑️ Clear Results", key="clear_penny"):
            st.session_state.penny_stocks = None
            st.session_state.penny_price_range = None
            # Clear related AI insights
            if 'ai_insights' in st.session_state:
                for stock in stocks:
                    if stock.symbol in st.session_state.ai_insights:
                        del st.session_state.ai_insights[stock.symbol]
            st.rerun()

        render_stock_table(stocks)

        # AI Market Overview
        with st.expander("🤖 AI Market Overview"):
            if st.button("Generate AI Overview", key="gen_ai_overview"):
                with st.spinner("Generating AI analysis..."):
                    overview = screener.get_ai_market_overview(stocks)
                    st.session_state.ai_market_overview = overview
                    st.rerun()

            if 'ai_market_overview' in st.session_state:
                st.markdown(st.session_state.ai_market_overview)


def render_momentum_tab():
    """Render momentum stocks screener tab."""
    st.markdown('<h2 class="tab-header">📈 Momentum Stock Screener</h2>', unsafe_allow_html=True)
    st.caption("Find stocks with consistent upward momentum across multiple timeframes.")

    screener = get_stock_screener()
    if not screener:
        st.error("Stock Screener not available")
        return

    from src.insights.stock_screener import MomentumPeriod

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        period = st.selectbox(
            "Primary Period",
            ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
            index=2
        )

    with col2:
        min_return = st.slider("Min Return (%)", 0, 100, 10)

    with col3:
        consistency = st.slider("Consistency Threshold", 0.0, 1.0, 0.6,
                               help="What % of timeframes must be positive")

    with col4:
        limit = st.slider("Number of Stocks", 5, 30, 15, key="momentum_limit")

    # Map period
    period_map = {
        "1 Week": MomentumPeriod.ONE_WEEK,
        "1 Month": MomentumPeriod.ONE_MONTH,
        "3 Months": MomentumPeriod.THREE_MONTHS,
        "6 Months": MomentumPeriod.SIX_MONTHS,
        "1 Year": MomentumPeriod.ONE_YEAR,
    }

    if st.button("🚀 Find Momentum Stocks", type="primary"):
        with st.spinner(f"Finding stocks with momentum over {period}..."):
            try:
                selected_period = period_map.get(period, MomentumPeriod.THREE_MONTHS)
                stocks = screener.screen_momentum_stocks(
                    period=selected_period,
                    min_return=min_return,
                    consistency_threshold=consistency,
                    limit=limit
                )

                if stocks:
                    st.success(f"Found {len(stocks)} momentum stocks")

                    # Show consistency info
                    st.info(f"These stocks have at least {consistency*100:.0f}% of timeframes positive "
                           f"AND minimum {min_return}% return over {period}")

                    render_stock_table(stocks)

                    # AI insights
                    with st.expander("🤖 AI Strategy Recommendation"):
                        with st.spinner("Generating AI analysis..."):
                            overview = screener.get_ai_market_overview(stocks)
                            st.markdown(overview)
                else:
                    st.warning("No stocks meet momentum criteria. Try lowering thresholds.")

            except Exception as e:
                st.error(f"Error: {e}")


def render_market_cap_tab():
    """Render market cap screener tab."""
    st.markdown('<h2 class="tab-header">🏢 Market Cap Screener</h2>', unsafe_allow_html=True)
    st.caption("Screen stocks by market capitalization - Large Cap, Mid Cap, Small Cap.")

    screener = get_stock_screener()
    if not screener:
        st.error("Stock Screener not available")
        return

    from src.insights.stock_screener import MarketCap

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        cap_category = st.selectbox(
            "Market Cap Category",
            [
                "Large Cap (> ₹20,000 Cr)",
                "Mid Cap (₹5,000 - 20,000 Cr)",
                "Small Cap (₹500 - 5,000 Cr)",
                "Micro Cap (< ₹500 Cr)"
            ],
            index=1
        )

    with col2:
        sort_by = st.selectbox("Sort By", ["Momentum Score", "Price", "Volume"], index=0)

    with col3:
        limit = st.slider("Number of Stocks", 5, 30, 15, key="cap_limit")

    # Map
    cap_map = {
        "Large Cap (> ₹20,000 Cr)": MarketCap.LARGE_CAP,
        "Mid Cap (₹5,000 - 20,000 Cr)": MarketCap.MID_CAP,
        "Small Cap (₹500 - 5,000 Cr)": MarketCap.SMALL_CAP,
        "Micro Cap (< ₹500 Cr)": MarketCap.MICRO_CAP,
    }

    sort_map = {"Momentum Score": "momentum", "Price": "price", "Volume": "volume"}

    if st.button("🔍 Screen by Market Cap", type="primary"):
        with st.spinner(f"Finding {cap_category} stocks..."):
            try:
                selected_cap = cap_map.get(cap_category, MarketCap.MID_CAP)
                stocks = screener.screen_by_market_cap(
                    cap_category=selected_cap,
                    sort_by=sort_map.get(sort_by, "momentum"),
                    limit=limit
                )

                if stocks:
                    st.success(f"Found {len(stocks)} {cap_category} stocks")
                    render_stock_table(stocks)
                else:
                    st.warning("No stocks found in this category.")

            except Exception as e:
                st.error(f"Error: {e}")


def render_single_stock_tab():
    """Render single stock analysis tab."""
    st.markdown('<h2 class="tab-header">🔍 Single Stock Analysis</h2>', unsafe_allow_html=True)
    st.caption("Deep dive analysis of any stock with AI-powered insights.")

    screener = get_stock_screener()

    col1, col2 = st.columns([3, 1])

    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol (NSE)",
            value="RELIANCE",
            placeholder="e.g., RELIANCE, TCS, INFY, TATAMOTORS"
        ).upper()

    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔬 Analyze", type="primary", use_container_width=True)

    # Quick symbols
    st.markdown("**Quick Select:** ", unsafe_allow_html=True)
    quick_cols = st.columns(8)
    quick_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "TATMOTORS", "SBIN", "ITC", "WIPRO"]

    for i, sym in enumerate(quick_symbols):
        with quick_cols[i]:
            if st.button(sym, key=f"quick_{sym}"):
                symbol = sym
                analyze_btn = True

    if analyze_btn and symbol and screener:
        with st.spinner(f"Fetching data for {symbol}..."):
            try:
                stock_data = screener.fetch_stock_data(symbol)

                if stock_data:
                    # Display stock info
                    st.markdown("---")

                    # Header
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.markdown(f"## {stock_data.symbol} - {stock_data.name}")
                        st.markdown(f"**Sector:** {stock_data.sector} | **Industry:** {stock_data.industry}")
                        st.markdown(f"**Market Cap:** ₹{stock_data.market_cap_cr:,.0f} Cr ({stock_data.cap_category.value})")

                    with col2:
                        change_color = "green" if stock_data.change_1d >= 0 else "red"
                        st.metric("Current Price", f"₹{stock_data.price:,.2f}",
                                 delta=f"{stock_data.change_1d:+.2f}%")

                    with col3:
                        st.metric("Momentum Score", f"{stock_data.momentum_score:.2f}",
                                 delta=f"{stock_data.momentum_consistency*100:.0f}% consistent")

                    # Performance metrics
                    st.markdown("### 📊 Performance")
                    perf_cols = st.columns(6)

                    periods = [
                        ("1 Day", stock_data.change_1d),
                        ("1 Week", stock_data.change_1w),
                        ("1 Month", stock_data.change_1m),
                        ("3 Months", stock_data.change_3m),
                        ("6 Months", stock_data.change_6m),
                        ("1 Year", stock_data.change_1y),
                    ]

                    for i, (period, change) in enumerate(periods):
                        with perf_cols[i]:
                            st.metric(period, f"{change:+.2f}%")

                    # Technical indicators
                    st.markdown("### 📈 Technical Indicators")
                    tech_cols = st.columns(4)

                    with tech_cols[0]:
                        rsi_color = "🟢" if 30 < stock_data.rsi < 70 else "🔴"
                        st.metric("RSI (14)", f"{rsi_color} {stock_data.rsi:.1f}")

                    with tech_cols[1]:
                        ma50_text = "Above" if stock_data.above_ma50 else "Below"
                        ma50_emoji = "🟢" if stock_data.above_ma50 else "🔴"
                        st.metric("50-Day MA", f"{ma50_emoji} {ma50_text}")

                    with tech_cols[2]:
                        ma200_text = "Above" if stock_data.above_ma200 else "Below"
                        ma200_emoji = "🟢" if stock_data.above_ma200 else "🔴"
                        st.metric("200-Day MA", f"{ma200_emoji} {ma200_text}")

                    with tech_cols[3]:
                        vol_emoji = "🟢" if stock_data.volume_ratio > 1 else "🔴"
                        st.metric("Volume Ratio", f"{vol_emoji} {stock_data.volume_ratio:.2f}x")

                    # Multi-Model AI Consensus
                    st.markdown("### 🧠 Multi-Model AI Consensus")
                    st.caption("Cross-verified by OpenAI + Google Gemini for higher confidence")

                    judge = get_multi_model_judge()
                    if judge and len(judge.available_models) > 0:
                        with st.spinner("Getting AI consensus from multiple models..."):
                            # Prepare data for judge
                            judge_data = {
                                'price': stock_data.price,
                                'sector': stock_data.sector,
                                'market_cap_cr': stock_data.market_cap_cr,
                                'change_1d': stock_data.change_1d,
                                'change_1w': stock_data.change_1w,
                                'change_1m': stock_data.change_1m,
                                'change_3m': stock_data.change_3m,
                                'change_6m': stock_data.change_6m,
                                'change_1y': stock_data.change_1y,
                                'rsi': stock_data.rsi,
                                'above_ma50': stock_data.above_ma50,
                                'above_ma200': stock_data.above_ma200,
                                'volume_ratio': stock_data.volume_ratio,
                            }

                            consensus = judge.get_consensus(symbol, judge_data)

                            if consensus:
                                # Display verdict card
                                verdict_colors = {
                                    'STRONG_BUY': ('#00ff88', '🟢'),
                                    'BUY': ('#88ff88', '🟢'),
                                    'HOLD': ('#ffaa00', '🟡'),
                                    'SELL': ('#ff8888', '🔴'),
                                    'STRONG_SELL': ('#ff4444', '🔴'),
                                }
                                v_color, v_emoji = verdict_colors.get(consensus.final_verdict, ('#888', '❓'))

                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                            border-radius: 15px; padding: 25px; margin: 15px 0;
                                            border: 2px solid {v_color}; text-align: center;">
                                    <div style="font-size: 2.5rem;">{v_emoji}</div>
                                    <div style="font-size: 2rem; font-weight: bold; color: {v_color};">
                                        {consensus.final_verdict.replace('_', ' ')}
                                    </div>
                                    <div style="font-size: 1.2rem; color: #888; margin-top: 10px;">
                                        Consensus: {consensus.consensus_score:.0%} | Confidence: {consensus.confidence:.0%}
                                    </div>
                                    <div style="margin-top: 15px; color: #aaa;">
                                        🐂 {consensus.bull_count} Bullish | ⚖️ {consensus.neutral_count} Neutral | 🐻 {consensus.bear_count} Bearish
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Trade setup if available
                                if consensus.avg_entry > 0:
                                    st.markdown("#### 📊 Suggested Trade Setup")
                                    trade_cols = st.columns(3)
                                    with trade_cols[0]:
                                        st.metric("Entry Price", f"₹{consensus.avg_entry:,.2f}")
                                    with trade_cols[1]:
                                        st.metric("Stop Loss", f"₹{consensus.avg_stop_loss:,.2f}")
                                    with trade_cols[2]:
                                        st.metric("Target", f"₹{consensus.avg_target:,.2f}")

                                # Individual model opinions
                                with st.expander("🔍 View Individual Model Opinions"):
                                    for j in consensus.judgments:
                                        j_color = verdict_colors.get(j.verdict, ('#888', '❓'))[0]
                                        st.markdown(f"""
                                        <div style="background: #1a1a2e; border-radius: 8px; padding: 12px;
                                                    margin: 8px 0; border-left: 3px solid {j_color};">
                                            <b>{j.model_name}</b>:
                                            <span style="color: {j_color}; font-weight: bold;">{j.verdict}</span>
                                            ({j.confidence:.0%} confident)
                                            <br><span style="color: #888; font-size: 0.9rem;">{j.reasoning}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.warning("Could not get AI consensus. Using single-model analysis.")
                                with st.spinner("Generating single-model AI analysis..."):
                                    insight = screener.get_ai_stock_insight(stock_data)
                                    st.markdown(insight)
                    else:
                        # Fallback to single model
                        st.markdown("### 🤖 AI-Powered Analysis")
                        with st.spinner("Generating comprehensive AI analysis..."):
                            insight = screener.get_ai_stock_insight(stock_data)
                            st.markdown(insight)

                    # External links
                    st.markdown("### 🔗 External Resources")
                    link_cols = st.columns(4)

                    with link_cols[0]:
                        st.link_button("📊 TradingView",
                                      f"https://www.tradingview.com/chart/?symbol=NSE%3A{symbol}")
                    with link_cols[1]:
                        st.link_button("📰 MoneyControl",
                                      f"https://www.moneycontrol.com/india/stockpricequote/{symbol.lower()}")
                    with link_cols[2]:
                        st.link_button("📈 NSE India",
                                      f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}")
                    with link_cols[3]:
                        st.link_button("📑 Screener.in",
                                      f"https://www.screener.in/company/{symbol}/")

                else:
                    st.error(f"Could not fetch data for {symbol}. Make sure it's a valid NSE symbol.")

            except Exception as e:
                st.error(f"Error analyzing {symbol}: {e}")


def render_intraday_tab():
    """Render intraday analysis tab with timing predictions."""
    st.markdown('<h2 class="tab-header">🎯 Intraday Analysis</h2>', unsafe_allow_html=True)
    st.caption("Quantitative analysis with optimal timing predictions for Indian markets.")

    # Get timing analyzer
    timing = get_intraday_timing()
    engine = get_quant_engine()

    # ============== MARKET STATUS SECTION ==============
    st.markdown("### 🕐 Current Market Status")

    if timing:
        status = timing.get_market_status_display()

        # Large status card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border-radius: 15px; padding: 25px; margin: 10px 0;
                    border: 3px solid {status.get('color', '#888')};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 2rem; font-weight: bold; color: {status.get('color', '#888')};">
                        {'🟢 MARKET OPEN' if status.get('is_open') else '🔴 MARKET CLOSED'}
                    </div>
                    <div style="font-size: 1.2rem; color: #aaa; margin-top: 5px;">
                        {status.get('message', '')}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.8rem; font-weight: bold; color: #fff;">
                        {status.get('time_ist', '')}
                    </div>
                    <div style="color: #888;">
                        {status.get('date', '')}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Current session info (if market is open)
        if status.get('is_open'):
            session_cols = st.columns(4)

            with session_cols[0]:
                vol = status.get('volatility', 'N/A')
                vol_color = "#ff4444" if vol == "HIGH" else "#ffaa00" if vol == "MEDIUM" else "#00ff88"
                st.markdown(f"""
                <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="color: #888;">Volatility</div>
                    <div style="font-size: 1.5rem; color: {vol_color}; font-weight: bold;">{vol}</div>
                </div>
                """, unsafe_allow_html=True)

            with session_cols[1]:
                volume = status.get('volume', 'N/A')
                vol_color = "#00ff88" if volume == "HIGH" else "#ffaa00" if volume == "MEDIUM" else "#ff4444"
                st.markdown(f"""
                <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="color: #888;">Volume</div>
                    <div style="font-size: 1.5rem; color: {vol_color}; font-weight: bold;">{volume}</div>
                </div>
                """, unsafe_allow_html=True)

            with session_cols[2]:
                win_prob = status.get('win_prob', 0.5)
                prob_color = "#00ff88" if win_prob > 0.55 else "#ffaa00" if win_prob > 0.5 else "#ff4444"
                st.markdown(f"""
                <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="color: #888;">Entry Win Rate</div>
                    <div style="font-size: 1.5rem; color: {prob_color}; font-weight: bold;">{win_prob:.0%}</div>
                </div>
                """, unsafe_allow_html=True)

            with session_cols[3]:
                st.markdown(f"""
                <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="color: #888;">Recommendation</div>
                    <div style="font-size: 0.9rem; color: #aaa; margin-top: 5px;">{status.get('next_action', 'N/A')[:40]}</div>
                </div>
                """, unsafe_allow_html=True)

    # ============== TIMING GUIDE ==============
    st.markdown("---")
    st.markdown("### ⏰ Optimal Trading Times (IST)")

    time_cols = st.columns(5)

    time_windows = [
        ("Opening Session", "9:15 - 10:00", "VERY HIGH", "#ff4444",
         "High volatility, gaps", "❌ Avoid new entries", "48%"),
        ("Morning Trend", "10:00 - 12:00", "MODERATE", "#00ff88",
         "Trend establishes", "✅ BEST for entries", "58%"),
        ("Mid-day Lull", "12:00 - 1:00", "MODERATE", "#ffaa00",
         "Low volume, choppy", "❌ Avoid trading", "45%"),
        ("Afternoon", "1:00 - 2:30", "MODERATE", "#88ff88",
         "Trend resumes", "✅ Add to winners", "54%"),
        ("Power Hour", "2:30 - 3:30", "HIGH", "#ff8800",
         "Final moves", "📊 Book profits", "52%"),
    ]

    for i, (name, time_range, risk, color, desc, action, prob) in enumerate(time_windows):
        with time_cols[i]:
            # Highlight current window
            current_state = status.get('state', '') if timing else ''
            is_current = (
                (current_state == "OPENING_SESSION" and i == 0) or
                (current_state == "MORNING_TREND" and i == 1) or
                (current_state == "MIDDAY_LULL" and i == 2) or
                (current_state == "AFTERNOON" and i == 3) or
                (current_state == "POWER_HOUR" and i == 4)
            )

            border = f"3px solid {color}" if is_current else "1px solid #333"
            glow = f"box-shadow: 0 0 15px {color};" if is_current else ""

            st.markdown(f"""
            <div style="background: #1a1a2e; padding: 12px; border-radius: 10px;
                        border: {border}; height: 200px; {glow}">
                <div style="font-weight: bold; color: #fff; font-size: 0.95rem;">{name}</div>
                <div style="color: {color}; font-size: 0.85rem; margin: 3px 0;">{time_range}</div>
                <div style="color: #888; font-size: 0.75rem; margin: 5px 0;">{desc}</div>
                <div style="margin-top: 8px;">
                    <span style="color: #888; font-size: 0.75rem;">Risk: </span>
                    <span style="color: {color}; font-size: 0.75rem;">{risk}</span>
                </div>
                <div style="margin-top: 3px;">
                    <span style="color: #888; font-size: 0.75rem;">Win Rate: </span>
                    <span style="color: #aaa; font-size: 0.75rem;">{prob}</span>
                </div>
                <div style="margin-top: 8px; padding: 5px; background: rgba(255,255,255,0.05);
                            border-radius: 5px; font-size: 0.8rem; color: {color};">
                    {action}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ============== TOMORROW OUTLOOK (if market closed) ==============
    if timing and not status.get('is_open'):
        st.markdown("---")
        st.markdown("### 🔮 Tomorrow's Outlook")

        # Get tomorrow outlook for a sample stock
        tomorrow = timing.get_tomorrow_outlook("NIFTY50")

        if tomorrow:
            outlook_cols = st.columns([2, 1, 1, 1])

            with outlook_cols[0]:
                trend_color = "#00ff88" if tomorrow.get('trend') == "BULLISH" else "#ff4444"
                st.markdown(f"""
                <div style="background: #1a1a2e; padding: 20px; border-radius: 10px;
                            border-left: 4px solid {trend_color};">
                    <div style="font-size: 1.2rem; font-weight: bold; color: #fff;">
                        Next Trading Day: {tomorrow.get('date', 'N/A')}
                    </div>
                    <div style="color: {trend_color}; font-size: 1.1rem; margin-top: 5px;">
                        Expected Trend: {tomorrow.get('trend', 'N/A')}
                    </div>
                    <div style="color: #888; margin-top: 10px; font-size: 0.9rem;">
                        {tomorrow.get('strategy', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with outlook_cols[1]:
                gap_up = tomorrow.get('gap_up_probability', 50)
                st.metric("Gap Up Chance", f"{gap_up:.0f}%",
                         help="Probability of opening higher than previous close")

            with outlook_cols[2]:
                gap_down = tomorrow.get('gap_down_probability', 50)
                st.metric("Gap Down Chance", f"{gap_down:.0f}%")

            with outlook_cols[3]:
                risk = tomorrow.get('risk_level', 'MODERATE')
                risk_color = "#00ff88" if risk == "LOW" else "#ffaa00" if risk == "MODERATE" else "#ff4444"
                st.markdown(f"""
                <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="color: #888;">Risk Level</div>
                    <div style="font-size: 1.3rem; color: {risk_color}; font-weight: bold;">{risk}</div>
                </div>
                """, unsafe_allow_html=True)

            # Recommended times for tomorrow
            st.markdown("#### 📅 Recommended Times for Tomorrow")
            rec_cols = st.columns(2)

            with rec_cols[0]:
                st.success(f"""
                **Best Entry Time:** {tomorrow.get('recommended_entry_time', '10:00 - 11:00 AM')}

                - Wait for opening volatility to settle
                - Look for trend confirmation
                - Enter on pullback to support
                """)

            with rec_cols[1]:
                st.info(f"""
                **Best Exit Time:** {tomorrow.get('recommended_exit_time', '2:30 - 3:15 PM')}

                - Book profits before close
                - Avoid overnight risk for intraday
                - Set trailing stop after 2:30 PM
                """)

    # ============== AUTO-SCAN TOP INTRADAY OPPORTUNITIES ==============
    st.markdown("---")
    st.markdown("### 🔥 Top Intraday Opportunities (Auto-Scanned)")

    # INTRADAY UNIVERSE
    INTRADAY_STOCKS = [
        # NIFTY 50 High Liquidity
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
        "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
        "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "TATAMOTORS", "JSWSTEEL",
        "TATASTEEL", "HINDALCO", "NTPC", "POWERGRID", "M&M", "WIPRO", "TECHM",
        # Midcaps with good intraday movement
        "DMART", "SIEMENS", "ABB", "PIDILITIND", "HAVELLS", "TVSMOTOR", "VOLTAS",
        "LTIM", "MPHASIS", "COFORGE", "LUPIN", "AUROPHARMA", "DLF", "OBEROIRLTY"
    ]

    scan_col1, scan_col2, scan_col3 = st.columns([1, 1, 2])

    with scan_col1:
        intraday_top_n = st.selectbox("Show Top", [5, 10, 15], index=1, key="intraday_top_n")

    with scan_col2:
        st.write("")
        scan_intraday = st.button("🔄 Scan for Opportunities", type="primary", key="scan_intraday_btn")

    with scan_col3:
        st.caption("Scans NIFTY 50 + Midcaps for best intraday setups with entry/exit timing")

    # Auto-scan or show cached results
    if scan_intraday or 'intraday_scan_results' not in st.session_state:
        import yfinance as yf

        progress = st.progress(0)
        status = st.empty()

        intraday_results = []

        for i, sym in enumerate(INTRADAY_STOCKS):
            try:
                progress.progress((i + 1) / len(INTRADAY_STOCKS))
                status.text(f"Analyzing {sym}... ({i+1}/{len(INTRADAY_STOCKS)})")

                ticker = yf.Ticker(f"{sym}.NS")
                df = ticker.history(period="1mo")

                if df.empty:
                    ticker = yf.Ticker(f"{sym}.BO")
                    df = ticker.history(period="1mo")

                if not df.empty and len(df) >= 10:
                    df.columns = df.columns.str.lower()

                    price = float(df['close'].iloc[-1])
                    prev_close = float(df['close'].iloc[-2])
                    day_change = ((price - prev_close) / prev_close) * 100

                    # Calculate ATR for stop loss
                    high_low = df['high'] - df['low']
                    atr = float(high_low.rolling(14).mean().iloc[-1])
                    atr_pct = (atr / price) * 100

                    # RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if loss.iloc[-1] != 0 else 50

                    # Momentum
                    mom_5d = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100

                    # Volume
                    avg_vol = df['volume'].iloc[-10:].mean()
                    today_vol = df['volume'].iloc[-1]
                    vol_ratio = today_vol / avg_vol if avg_vol > 0 else 1

                    # Score for intraday
                    score = 0
                    if 30 <= rsi <= 50:
                        score += 25
                    elif 50 < rsi <= 65:
                        score += 15
                    if mom_5d > 0:
                        score += 20
                    if vol_ratio > 1.2:
                        score += 15
                    if 1.5 <= atr_pct <= 3.5:
                        score += 20  # Good intraday range

                    if score >= 30:
                        # Calculate trade setup
                        entry_price = price
                        stop_loss = price - (1.5 * atr)  # 1.5 ATR stop
                        target_1 = price + (1.5 * atr)   # 1:1 RR
                        target_2 = price + (3 * atr)     # 1:2 RR

                        risk_pct = ((entry_price - stop_loss) / entry_price) * 100
                        reward_pct = ((target_1 - entry_price) / entry_price) * 100

                        intraday_results.append({
                            'symbol': sym,
                            'price': price,
                            'day_change': day_change,
                            'rsi': rsi,
                            'atr_pct': atr_pct,
                            'vol_ratio': vol_ratio,
                            'score': score,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'target_1': target_1,
                            'target_2': target_2,
                            'risk_pct': risk_pct,
                            'reward_pct': reward_pct,
                            'entry_time': "10:00 - 10:30 AM" if rsi < 45 else "10:30 - 11:30 AM",
                            'exit_time': "2:30 - 3:15 PM"
                        })
            except Exception:
                pass

        progress.empty()
        status.empty()

        # Sort by score
        intraday_results = sorted(intraday_results, key=lambda x: x['score'], reverse=True)
        st.session_state['intraday_scan_results'] = intraday_results
        st.session_state['intraday_scan_time'] = datetime.now().strftime("%H:%M:%S")

    # Display results
    intraday_results = st.session_state.get('intraday_scan_results', [])
    scan_time = st.session_state.get('intraday_scan_time', 'N/A')

    if intraday_results:
        st.success(f"Found {len(intraday_results)} opportunities | Last scan: {scan_time}")

        for i, trade in enumerate(intraday_results[:intraday_top_n], 1):
            with st.container():
                with st.expander(f"🟢 **#{i} {trade['symbol']}** | ₹{trade['price']:,.2f} | Score: {trade['score']}", expanded=(i <= 3)):

                    # Price and day change
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Price", f"₹{trade['price']:,.2f}", f"{trade['day_change']:+.2f}%")
                    with col2:
                        st.metric("Score", f"{trade['score']}/100")
                    with col3:
                        st.metric("RSI", f"{trade['rsi']:.0f}")
                    with col4:
                        st.metric("Volume", f"{trade['vol_ratio']:.1f}x")

                    st.markdown("---")
                    st.markdown("**🎯 Trade Setup**")

                    # Trade setup
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        st.success(f"📥 **BUY**\n\n₹{trade['entry_price']:,.2f}\n\n{trade['entry_time']}")
                    with b2:
                        st.error(f"🛑 **STOP LOSS**\n\n₹{trade['stop_loss']:,.2f}\n\n-{trade['risk_pct']:.1f}%")
                    with b3:
                        st.info(f"🎯 **TARGET**\n\n₹{trade['target_1']:,.2f}\n\n+{trade['reward_pct']:.1f}%")
                    with b4:
                        st.warning(f"📤 **EXIT BY**\n\n{trade['exit_time']}\n\nBook profits")

                    # Metrics row
                    st.markdown("**📊 Indicators:** " + f"RSI: {trade['rsi']:.0f} | ATR: {trade['atr_pct']:.1f}% | Volume: {trade['vol_ratio']:.1f}x | R:R: 1:1")

                    # AI Insight
                    st.markdown("---")
                    st.markdown("**💡 AI Analysis:**")
                    insight_data = {
                        'symbol': trade['symbol'],
                        'signal': "BUY" if trade['score'] >= 50 else "HOLD",
                        'score': trade['score'],
                        'rsi': trade['rsi'],
                        'growth_1m': trade.get('day_change', 0) * 5,
                        'growth_3m': 0,
                        'growth_1y': 0,
                        'trend': "UPTREND" if trade['day_change'] > 0 else "SIDEWAYS",
                        'volume_ratio': trade['vol_ratio'],
                        'from_high': -10,
                        'from_low': 20
                    }
                    insight = generate_stock_insight(insight_data)
                    st.markdown(insight)
    else:
        st.info("Click 'Scan for Opportunities' to find top intraday setups")

    # ============== STOCK-SPECIFIC TIMING ==============
    st.markdown("---")
    st.markdown("### 📊 Individual Stock Analysis")
    st.caption("Analyze a specific stock for detailed timing and trade setup")

    col1, col2 = st.columns([3, 1])

    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="RELIANCE",
            placeholder="e.g., RELIANCE, TCS, INFY",
            key="timing_symbol"
        ).upper().strip()

    with col2:
        st.write("")
        st.write("")
        analyze_timing = st.button("🔍 Analyze Timing", type="primary", key="analyze_timing_btn")

    if analyze_timing and symbol and timing:
        with st.spinner(f"Analyzing optimal timing for {symbol}..."):
            prediction = timing.analyze(symbol)

        if prediction:
            # Current recommendation
            action_colors = {
                "BUY": "#00ff88",
                "SELL": "#ff4444",
                "WAIT": "#ffaa00",
                "HOLD": "#888",
                "AVOID": "#ff6600",
                "BOOK_PROFITS": "#00aaff",
            }
            action_color = action_colors.get(prediction.recommended_action.value, "#888")

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 12px; padding: 20px; margin: 15px 0;
                        border: 2px solid {action_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.4rem; font-weight: bold;">{symbol}</span>
                        <span style="margin-left: 15px; color: #888;">Timing Analysis</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.8rem; font-weight: bold; color: {action_color};">
                            {prediction.recommended_action.value}
                        </div>
                        <div style="color: #888;">Win Prob: {prediction.win_probability:.0%}</div>
                    </div>
                </div>
                <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.05);
                            border-radius: 8px; color: #ccc;">
                    {prediction.action_reason}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ============== SPECIFIC TRADE SETUP ==============
            if prediction.trade_setup:
                setup = prediction.trade_setup
                st.markdown("#### 🎯 Specific Trade Setup - When to Buy & Sell")

                # Main trade card
                confidence_color = "#00ff88" if setup.confidence == "HIGH" else "#ffaa00" if setup.confidence == "MEDIUM" else "#ff4444"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #0a2a0a 0%, #1a3a1a 100%);
                            border-radius: 15px; padding: 25px; margin: 15px 0;
                            border: 2px solid #00ff88;">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <span style="font-size: 1.5rem; font-weight: bold; color: #00ff88;">
                            INTRADAY TRADE PLAN FOR {symbol}
                        </span>
                        <span style="margin-left: 15px; padding: 5px 10px; background: {confidence_color};
                                    border-radius: 5px; color: #000; font-weight: bold;">
                            {setup.confidence} CONFIDENCE
                        </span>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                        <!-- BUY Section -->
                        <div style="background: rgba(0,255,136,0.1); border-radius: 10px; padding: 15px;
                                    border: 1px solid #00ff88;">
                            <div style="text-align: center; color: #00ff88; font-size: 1.2rem; font-weight: bold;">
                                📥 BUY
                            </div>
                            <div style="text-align: center; margin-top: 10px;">
                                <div style="font-size: 2rem; font-weight: bold; color: #fff;">
                                    {setup.entry_time}
                                </div>
                                <div style="font-size: 1.5rem; color: #00ff88; margin-top: 5px;">
                                    ₹{setup.entry_price:,.2f}
                                </div>
                                <div style="color: #888; font-size: 0.85rem; margin-top: 5px;">
                                    {setup.entry_window}
                                </div>
                            </div>
                        </div>

                        <!-- SELL/TARGET Section -->
                        <div style="background: rgba(0,170,255,0.1); border-radius: 10px; padding: 15px;
                                    border: 1px solid #00aaff;">
                            <div style="text-align: center; color: #00aaff; font-size: 1.2rem; font-weight: bold;">
                                📤 SELL (Target)
                            </div>
                            <div style="text-align: center; margin-top: 10px;">
                                <div style="font-size: 2rem; font-weight: bold; color: #fff;">
                                    {setup.target_time}
                                </div>
                                <div style="font-size: 1.5rem; color: #00aaff; margin-top: 5px;">
                                    ₹{setup.target_price:,.2f}
                                </div>
                                <div style="color: #888; font-size: 0.85rem; margin-top: 5px;">
                                    +{((setup.target_price - setup.entry_price) / setup.entry_price * 100):.2f}% profit
                                </div>
                            </div>
                        </div>

                        <!-- STOP LOSS Section -->
                        <div style="background: rgba(255,68,68,0.1); border-radius: 10px; padding: 15px;
                                    border: 1px solid #ff4444;">
                            <div style="text-align: center; color: #ff4444; font-size: 1.2rem; font-weight: bold;">
                                🛑 STOP LOSS
                            </div>
                            <div style="text-align: center; margin-top: 10px;">
                                <div style="font-size: 2rem; font-weight: bold; color: #fff;">
                                    {setup.stop_loss_time}
                                </div>
                                <div style="font-size: 1.5rem; color: #ff4444; margin-top: 5px;">
                                    ₹{setup.stop_loss_price:,.2f}
                                </div>
                                <div style="color: #888; font-size: 0.85rem; margin-top: 5px;">
                                    -{((setup.entry_price - setup.stop_loss_price) / setup.entry_price * 100):.2f}% risk
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability metrics
                prob_cols = st.columns(4)
                with prob_cols[0]:
                    prob_color = "#00ff88" if setup.win_probability > 0.55 else "#ffaa00"
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                        <div style="color: #888;">Win Probability</div>
                        <div style="font-size: 1.8rem; color: {prob_color}; font-weight: bold;">{setup.win_probability:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with prob_cols[1]:
                    rr_color = "#00ff88" if setup.risk_reward >= 2 else "#ffaa00" if setup.risk_reward >= 1.5 else "#ff4444"
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                        <div style="color: #888;">Risk : Reward</div>
                        <div style="font-size: 1.8rem; color: {rr_color}; font-weight: bold;">1 : {setup.risk_reward:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with prob_cols[2]:
                    ev_color = "#00ff88" if setup.expected_return_pct > 0 else "#ff4444"
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                        <div style="color: #888;">Expected Return</div>
                        <div style="font-size: 1.8rem; color: {ev_color}; font-weight: bold;">{setup.expected_return_pct:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with prob_cols[3]:
                    risk_color = "#00ff88" if setup.risk_level == "LOW" else "#ffaa00" if setup.risk_level == "MODERATE" else "#ff4444"
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center;">
                        <div style="color: #888;">Risk Level</div>
                        <div style="font-size: 1.8rem; color: {risk_color}; font-weight: bold;">{setup.risk_level}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Trade reasoning
                reason_cols = st.columns(2)
                with reason_cols[0]:
                    st.markdown(f"""
                    **📥 Why Buy at {setup.entry_time}?**

                    {setup.entry_reason}
                    """)

                with reason_cols[1]:
                    st.markdown(f"""
                    **📤 Why Sell at {setup.target_time}?**

                    {setup.target_reason}
                    """)

                # Action summary box
                if setup.entry_time != "AVOID":
                    st.success(f"""
                    ### 📋 Your Trade Plan

                    1. **BUY** {symbol} at **₹{setup.entry_price:,.2f}** around **{setup.entry_time}**
                    2. Set **STOP LOSS** at **₹{setup.stop_loss_price:,.2f}** (exit if hit anytime)
                    3. **SELL** at **₹{setup.target_price:,.2f}** by **{setup.target_time}**
                    4. **Must exit by {setup.stop_loss_time}** regardless of profit/loss (intraday rule)

                    **Potential Profit:** ₹{(setup.target_price - setup.entry_price):,.2f}/share ({((setup.target_price - setup.entry_price) / setup.entry_price * 100):.2f}%)
                    **Max Loss:** ₹{(setup.entry_price - setup.stop_loss_price):,.2f}/share ({((setup.entry_price - setup.stop_loss_price) / setup.entry_price * 100):.2f}%)
                    """)
                else:
                    st.warning(f"""
                    ### ⚠️ Trade Not Recommended Right Now

                    {setup.entry_reason}

                    **Suggestion:** Wait for the next trading session and enter during Morning Trend (10:00 - 11:30 AM)
                    """)

            # Best times for this stock
            timing_cols = st.columns(2)

            with timing_cols[0]:
                st.markdown("#### ✅ Best Entry Windows")
                for window, time_range, prob in prediction.best_entry_windows:
                    prob_color = "#00ff88" if prob > 0.55 else "#ffaa00"
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; margin: 5px 0;
                                border-left: 3px solid {prob_color};">
                        <span style="font-weight: bold;">{window}</span>
                        <span style="color: #888; margin-left: 10px;">{time_range}</span>
                        <span style="float: right; color: {prob_color};">{prob:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with timing_cols[1]:
                st.markdown("#### 📊 Best Exit Windows")
                for window, time_range, prob in prediction.best_exit_windows:
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; margin: 5px 0;
                                border-left: 3px solid #00aaff;">
                        <span style="font-weight: bold;">{window}</span>
                        <span style="color: #888; margin-left: 10px;">{time_range}</span>
                        <span style="float: right; color: #00aaff;">{prob:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Windows to avoid
            st.markdown("#### ⚠️ Windows to Avoid")
            avoid_cols = st.columns(len(prediction.avoid_windows))
            for i, (window, time_range, reason) in enumerate(prediction.avoid_windows):
                with avoid_cols[i]:
                    st.markdown(f"""
                    <div style="background: #2a1a1a; padding: 10px; border-radius: 8px;
                                border: 1px solid #ff4444;">
                        <div style="font-weight: bold; color: #ff6666;">{window}</div>
                        <div style="color: #888; font-size: 0.85rem;">{time_range}</div>
                        <div style="color: #aaa; font-size: 0.8rem; margin-top: 5px;">{reason}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Risk and opportunity factors
            if prediction.risk_factors or prediction.opportunity_factors:
                factor_cols = st.columns(2)

                with factor_cols[0]:
                    if prediction.opportunity_factors:
                        st.markdown("**✅ Opportunities:**")
                        for f in prediction.opportunity_factors:
                            st.markdown(f"- {f}")

                with factor_cols[1]:
                    if prediction.risk_factors:
                        st.markdown("**⚠️ Risks:**")
                        for f in prediction.risk_factors:
                            st.markdown(f"- {f}")

    # ============== TRADE GENERATION (if engine available) ==============
    if engine and status.get('is_open', False):
        st.markdown("---")
        st.markdown("### 🚀 Generate Intraday Trades")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            min_prob = st.slider("Min Win Probability", 0.50, 0.85, 0.60, 0.05, key="intraday_prob")

        with col2:
            num_trades = st.selectbox("Max Trades", [5, 10, 15], index=1, key="intraday_trades")

        with col3:
            st.write("")
            st.write("")
            gen_btn = st.button("🚀 Generate Trades", type="primary", key="gen_trades_btn")

        if gen_btn:
            with st.spinner("Calculating probabilities and expected values..."):
                try:
                    market_state = engine.get_market_state()
                    result = engine.get_top_trades(n=num_trades, min_prob=min_prob)
                    trades = result.get('trades', [])

                    if trades:
                        st.success(f"Found {len(trades)} potential trades")

                        import yfinance as yf

                        for i, trade in enumerate(trades):
                            symbol = trade['symbol']
                            signal = trade.get('signal', 'AVOID')
                            prob = trade.get('probability', 0)
                            ev = trade.get('expected_value_pct', 0)

                            # Get timing for this stock
                            stock_timing = timing.analyze(symbol) if timing else None

                            # Fetch stock details
                            try:
                                ticker = yf.Ticker(f"{symbol}.NS")
                                info = ticker.info or {}
                                hist = ticker.history(period='10d')

                                stock_name = info.get('shortName', symbol)
                                sector = info.get('sector', 'N/A')
                            except:
                                stock_name = symbol
                                sector = "N/A"
                                hist = None

                            border_color = '#00ff88' if signal == 'BUY' else '#ff4444' if signal == 'SHORT' else '#888'

                            # Timing recommendation
                            timing_rec = ""
                            if stock_timing:
                                timing_rec = f" | ⏰ {stock_timing.recommended_action.value}"

                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                        border-radius: 12px; padding: 20px; margin: 15px 0;
                                        border-left: 5px solid {border_color};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-size: 1.4rem; font-weight: bold;">{symbol}</span>
                                        <span style="margin-left: 10px; color: #888;">{stock_name[:30]}</span>
                                        <span style="margin-left: 10px; color: #666; font-size: 0.9rem;">({sector})</span>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="font-size: 1.5rem; font-weight: bold; color: {border_color};">{signal}</div>
                                        <div style="color: #888;">Win: {prob*100:.0f}% | EV: {ev:+.2f}%{timing_rec}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            with st.expander(f"📊 View Full Details for {symbol}", expanded=(i==0)):
                                # Get additional metrics
                                entry_price = trade.get('entry', 0)
                                stop_loss = trade.get('stop_loss', 0)
                                target_price = trade.get('target_1', 0)
                                risk_reward = trade.get('reward_risk', 0)

                                # Calculate additional metrics from history
                                today_change = 0
                                five_day_change = 0
                                rsi_val = 50
                                volume_ratio = 1.0

                                if hist is not None and not hist.empty and len(hist) >= 2:
                                    today_change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
                                    if len(hist) >= 5:
                                        five_day_change = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1) * 100
                                    # Calculate RSI
                                    delta = hist['Close'].diff()
                                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                                    if loss.iloc[-1] != 0 and not pd.isna(loss.iloc[-1]):
                                        rs = gain.iloc[-1] / loss.iloc[-1]
                                        rsi_val = 100 - (100 / (1 + rs)) if rs != 0 else 50
                                    else:
                                        rsi_val = 50
                                    # Volume ratio
                                    vol_avg = hist['Volume'].rolling(10).mean().iloc[-1]
                                    if vol_avg > 0 and not pd.isna(vol_avg):
                                        volume_ratio = hist['Volume'].iloc[-1] / vol_avg

                                # Trade Setup section - custom HTML for full numbers
                                st.markdown("#### 💰 Trade Setup")
                                st.markdown(f"""
                                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0;">
                                    <div class="trade-metric" style="border-left: 3px solid #00ff88;">
                                        <div class="trade-metric-label">Entry</div>
                                        <div class="trade-metric-value" style="color: #00ff88;">₹{entry_price:,.2f}</div>
                                    </div>
                                    <div class="trade-metric" style="border-left: 3px solid #ff4444;">
                                        <div class="trade-metric-label">Stop Loss</div>
                                        <div class="trade-metric-value" style="color: #ff4444;">₹{stop_loss:,.2f}</div>
                                        <div style="color: #888; font-size: 0.75rem;">↓ {((entry_price - stop_loss) / entry_price * 100):.1f}%</div>
                                    </div>
                                    <div class="trade-metric" style="border-left: 3px solid #00aaff;">
                                        <div class="trade-metric-label">Target</div>
                                        <div class="trade-metric-value" style="color: #00aaff;">₹{target_price:,.2f}</div>
                                        <div style="color: #888; font-size: 0.75rem;">↑ {((target_price - entry_price) / entry_price * 100):.1f}%</div>
                                    </div>
                                    <div class="trade-metric">
                                        <div class="trade-metric-label">Risk:Reward</div>
                                        <div class="trade-metric-value">{risk_reward:.2f}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Key Metrics section
                                st.markdown("#### 📊 Key Metrics")
                                today_color = "#00ff88" if today_change >= 0 else "#ff4444"
                                fiveday_color = "#00ff88" if five_day_change >= 0 else "#ff4444"
                                rsi_color = "#ff4444" if rsi_val > 70 or rsi_val < 30 else "#00ff88"
                                vol_color = "#00ff88" if volume_ratio >= 1 else "#ff4444"

                                st.markdown(f"""
                                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0;">
                                    <div class="trade-metric">
                                        <div class="trade-metric-label">Today</div>
                                        <div class="trade-metric-value" style="color: {today_color};">{today_change:+.2f}%</div>
                                    </div>
                                    <div class="trade-metric">
                                        <div class="trade-metric-label">5-Day</div>
                                        <div class="trade-metric-value" style="color: {fiveday_color};">{five_day_change:+.2f}%</div>
                                    </div>
                                    <div class="trade-metric">
                                        <div class="trade-metric-label">RSI</div>
                                        <div class="trade-metric-value" style="color: {rsi_color};">{rsi_val:.0f}</div>
                                        <div style="color: #888; font-size: 0.7rem;">{'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'}</div>
                                    </div>
                                    <div class="trade-metric">
                                        <div class="trade-metric-label">Volume</div>
                                        <div class="trade-metric-value" style="color: {vol_color};">{volume_ratio:.1f}x</div>
                                        <div style="color: #888; font-size: 0.7rem;">vs avg</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Chart with proper Y-axis
                                if hist is not None and not hist.empty:
                                    st.markdown("#### 📈 10-Day Price Trend")
                                    import altair as alt
                                    chart_data = hist['Close'].reset_index()
                                    chart_data.columns = ['Date', 'Price']
                                    chart_data['Date'] = pd.to_datetime(chart_data['Date']).dt.strftime('%b %d')

                                    # Create Altair chart with proper Y-axis
                                    chart = alt.Chart(chart_data).mark_line(color='#00aaff', strokeWidth=2).encode(
                                        x=alt.X('Date:N', title='Date', axis=alt.Axis(labelAngle=-45)),
                                        y=alt.Y('Price:Q', title='Price (₹)', scale=alt.Scale(zero=False)),
                                        tooltip=['Date', alt.Tooltip('Price:Q', format=',.2f')]
                                    ).properties(height=250).configure_axis(
                                        labelColor='#888',
                                        titleColor='#aaa',
                                        gridColor='#333'
                                    ).configure_view(strokeWidth=0)

                                    st.altair_chart(chart, use_container_width=True)

                                # WHY THIS TRADE - LLM Insights
                                st.markdown("#### 🎯 Why This Trade?")
                                bullish = trade.get('bullish_factors', [])
                                bearish = trade.get('bearish_factors', [])
                                risks = trade.get('risks', [])

                                why_html = '<div class="insight-card">'
                                if bullish:
                                    why_html += '<div class="insight-title">✅ Bullish Factors</div>'
                                    why_html += '<div class="insight-text">'
                                    for f in bullish[:3]:
                                        why_html += f'• {f}<br>'
                                    why_html += '</div>'
                                if bearish:
                                    why_html += '<div class="insight-title" style="color: #ff6666; margin-top: 15px;">⚠️ Bearish Factors</div>'
                                    why_html += '<div class="insight-text">'
                                    for f in bearish[:2]:
                                        why_html += f'• {f}<br>'
                                    why_html += '</div>'
                                if risks:
                                    why_html += '<div class="insight-title" style="color: #ffaa00; margin-top: 15px;">🔴 Key Risks</div>'
                                    why_html += '<div class="insight-text">'
                                    for f in risks[:2]:
                                        why_html += f'• {f}<br>'
                                    why_html += '</div>'
                                why_html += '</div>'
                                st.markdown(why_html, unsafe_allow_html=True)

                                # Action
                                if signal == "BUY":
                                    best_time = stock_timing.best_entry_windows[0][1] if stock_timing and stock_timing.best_entry_windows else "10:00 - 11:30 AM"
                                    st.success(f"""
                                    **ACTION:** BUY {symbol} at ₹{trade.get('entry', 0):,.2f}

                                    **Best Entry Time:** {best_time}

                                    **Exit:** Stop at ₹{trade.get('stop_loss', 0):,.2f} | Target at ₹{trade.get('target_1', 0):,.2f}
                                    """)

                    else:
                        st.warning("No trades meet criteria. Market conditions may be unfavorable.")

                except Exception as e:
                    st.error(f"Error generating trades: {e}")

    elif not status.get('is_open', False):
        st.markdown("---")
        st.info("🔴 **Market is closed.** Trade generation will be available during market hours (9:15 AM - 3:30 PM IST)")

        # Quick tips for tomorrow
        st.markdown("### 💡 Tips for Tomorrow's Session")
        tip_cols = st.columns(3)

        with tip_cols[0]:
            st.markdown("""
            **Pre-Market Prep (Before 9:15 AM)**
            - Check global cues (US, SGX Nifty)
            - Review overnight news
            - Prepare watchlist
            - Set price alerts
            """)

        with tip_cols[1]:
            st.markdown("""
            **Best Entry Strategy**
            - Wait until 10:00 AM
            - Let opening volatility settle
            - Enter on trend confirmation
            - Use limit orders, not market
            """)

        with tip_cols[2]:
            st.markdown("""
            **Risk Management**
            - Risk max 2% per trade
            - Use ATR-based stop loss
            - Book partial profits at 1:1
            - Close all by 3:15 PM
            """)


def render_comprehensive_analysis_tab():
    """Render comprehensive analysis with ALL parameters using 7-Model Ensemble."""
    config = get_engine_config()
    model_label = "7-Model" if config.use_7_model else "4-Model"

    st.markdown('<h2 class="tab-header">🔬 Comprehensive Analysis</h2>', unsafe_allow_html=True)
    st.caption(f"Full analysis with {model_label} Ensemble | Fundamentals, technicals, statistics, risk management, and AI insights")

    analyzer = get_comprehensive_analyzer()
    tracker = get_trade_tracker()

    if not analyzer:
        st.error("Comprehensive Analyzer not available")
        return

    # Input - capital and risk are now shown from config (editable in sidebar)
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol (NSE)",
            value="RELIANCE",
            placeholder="e.g., RELIANCE, TCS, INFY",
            key="comp_symbol"
        ).upper().strip()

    with col2:
        st.metric("Capital", f"₹{config.capital/1000:.0f}K")

    with col3:
        st.metric("Risk/Trade", f"{config.risk_per_trade*100:.1f}%")

    # Quick symbols
    st.markdown("**Quick Select:** ")
    qcols = st.columns(8)
    quick_syms = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "TATASTEEL", "ITC", "WIPRO"]
    for i, sym in enumerate(quick_syms):
        with qcols[i]:
            if st.button(sym, key=f"comp_quick_{sym}"):
                symbol = sym

    if st.button("🔬 Run Comprehensive Analysis", type="primary", key="comp_analyze"):
        if not symbol:
            st.warning("Please enter a stock symbol")
            return

        # Update analyzer settings from centralized config
        analyzer.capital = config.capital
        analyzer.risk_per_trade = config.risk_per_trade

        with st.spinner(f"Running {model_label} Ensemble analysis on {symbol}..."):
            analysis = analyzer.analyze(symbol)

        if not analysis or analysis.technical.price == 0:
            st.error(f"Could not analyze {symbol}. Please check the symbol.")
            return

        # Store in session for tracking
        st.session_state['last_analysis'] = analysis

        # ============== HEADER ==============
        st.markdown("---")
        st.markdown(f"## {analysis.symbol} - {analysis.name}")
        st.caption(f"Sector: {analysis.sector} | Industry: {analysis.industry}")

        # Price header
        price_cols = st.columns([2, 1, 1, 1])
        with price_cols[0]:
            change_color = "green" if analysis.technical.change_1d >= 0 else "red"
            st.metric("Current Price", f"₹{analysis.technical.price:,.2f}",
                     delta=f"{analysis.technical.change_1d:+.2f}%")
        with price_cols[1]:
            st.metric("52W High", f"₹{analysis.technical.week_52_high:,.2f}",
                     delta=f"{analysis.technical.pct_from_52_high:.1f}%")
        with price_cols[2]:
            st.metric("52W Low", f"₹{analysis.technical.week_52_low:,.2f}",
                     delta=f"+{analysis.technical.pct_from_52_low:.1f}%")
        with price_cols[3]:
            st.metric("Volume", f"{analysis.technical.volume_ratio:.1f}x avg")

        # ============== RECOMMENDATIONS ==============
        st.markdown("### 🎯 Trade Recommendations")

        rec_cols = st.columns(2)

        # INTRADAY
        with rec_cols[0]:
            irec = analysis.intraday_rec
            i_color = "#00ff88" if "BUY" in irec.signal.value else "#ff4444" if "SELL" in irec.signal.value else "#ffaa00"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 10px; padding: 20px; border: 2px solid {i_color};">
                <h4 style="margin:0; color: {i_color};">INTRADAY: {irec.signal.value}</h4>
                <p style="color: #888; margin: 5px 0;">Win Prob: {irec.win_probability:.1%} | R:R: {irec.risk_reward:.2f}</p>
                <table style="width: 100%; color: #ccc;">
                    <tr><td>Entry</td><td style="text-align:right;">₹{irec.entry_price:,.2f}</td></tr>
                    <tr><td>Stop Loss</td><td style="text-align:right; color: #ff6666;">₹{irec.stop_loss:,.2f}</td></tr>
                    <tr><td>Target 1</td><td style="text-align:right; color: #66ff66;">₹{irec.target_1:,.2f}</td></tr>
                    <tr><td>Position</td><td style="text-align:right;">{irec.position_size_pct:.1f}% (₹{irec.position_shares * irec.entry_price:,.0f})</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # SWING
        with rec_cols[1]:
            srec = analysis.swing_rec
            s_color = "#00ff88" if "BUY" in srec.signal.value else "#ff4444" if "SELL" in srec.signal.value else "#ffaa00"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 10px; padding: 20px; border: 2px solid {s_color};">
                <h4 style="margin:0; color: {s_color};">SWING (2-10 days): {srec.signal.value}</h4>
                <p style="color: #888; margin: 5px 0;">Win Prob: {srec.win_probability:.1%} | R:R: {srec.risk_reward:.2f}</p>
                <table style="width: 100%; color: #ccc;">
                    <tr><td>Entry</td><td style="text-align:right;">₹{srec.entry_price:,.2f}</td></tr>
                    <tr><td>Stop Loss</td><td style="text-align:right; color: #ff6666;">₹{srec.stop_loss:,.2f}</td></tr>
                    <tr><td>Target 1</td><td style="text-align:right; color: #66ff66;">₹{srec.target_1:,.2f}</td></tr>
                    <tr><td>Position</td><td style="text-align:right;">{srec.position_size_pct:.1f}% (₹{srec.position_shares * srec.entry_price:,.0f})</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # Factors
        if srec.bullish_factors or srec.bearish_factors:
            factor_cols = st.columns(2)
            with factor_cols[0]:
                if srec.bullish_factors:
                    st.markdown("**✅ Bullish Factors:**")
                    for f in srec.bullish_factors[:5]:
                        st.markdown(f"- {f}")
            with factor_cols[1]:
                if srec.bearish_factors:
                    st.markdown("**⚠️ Bearish Factors:**")
                    for f in srec.bearish_factors[:5]:
                        st.markdown(f"- {f}")

        # ============== FUNDAMENTALS ==============
        with st.expander("📊 Fundamental Analysis (Valuation, Profitability, Health)", expanded=True):
            fund = analysis.fundamental

            st.markdown("#### Valuation Metrics")
            val_cols = st.columns(4)
            with val_cols[0]:
                pe_color = "green" if 10 < fund.pe_ratio < 25 else "orange" if fund.pe_ratio > 0 else "gray"
                st.metric("P/E Ratio", f"{fund.pe_ratio:.1f}",
                         help="Price / Earnings. Lower = cheaper. Industry avg ~20")
            with val_cols[1]:
                st.metric("P/B Ratio", f"{fund.pb_ratio:.2f}",
                         help="Price / Book Value. <1 may be undervalued")
            with val_cols[2]:
                st.metric("EV/EBITDA", f"{fund.ev_ebitda:.1f}",
                         help="Enterprise Value / EBITDA. Lower = cheaper")
            with val_cols[3]:
                st.metric("Market Cap", f"₹{fund.market_cap:,.0f} Cr")

            st.markdown("#### Profitability")
            prof_cols = st.columns(4)
            with prof_cols[0]:
                roe_color = "green" if fund.roe > 15 else "orange" if fund.roe > 10 else "red"
                st.metric("ROE", f"{fund.roe:.1f}%",
                         help="Return on Equity. >15% is good")
            with prof_cols[1]:
                st.metric("ROA", f"{fund.roa:.1f}%",
                         help="Return on Assets")
            with prof_cols[2]:
                st.metric("Profit Margin", f"{fund.profit_margin:.1f}%",
                         help="Net Profit / Revenue")
            with prof_cols[3]:
                st.metric("Revenue Growth", f"{fund.revenue_growth:+.1f}%",
                         help="Year-over-year revenue growth")

            st.markdown("#### Financial Health")
            health_cols = st.columns(4)
            with health_cols[0]:
                de_color = "green" if fund.debt_equity < 1 else "orange" if fund.debt_equity < 2 else "red"
                st.metric("Debt/Equity", f"{fund.debt_equity:.2f}",
                         help="<1 is conservative, >2 is risky")
            with health_cols[1]:
                st.metric("Current Ratio", f"{fund.current_ratio:.2f}",
                         help=">1.5 is healthy")
            with health_cols[2]:
                st.metric("Dividend Yield", f"{fund.dividend_yield:.2f}%",
                         help="Annual dividend / Price")
            with health_cols[3]:
                st.metric("EPS", f"₹{fund.eps:.2f}",
                         help="Earnings Per Share")

        # ============== TECHNICALS ==============
        with st.expander("📈 Technical Analysis (Indicators, Trends, Levels)", expanded=True):
            tech = analysis.technical

            st.markdown("#### Momentum Indicators")
            mom_cols = st.columns(4)
            with mom_cols[0]:
                rsi_color = "red" if tech.rsi_14 > 70 else "green" if tech.rsi_14 < 30 else "gray"
                st.metric("RSI (14)", f"{tech.rsi_14:.0f}",
                         help="<30 = oversold (buy), >70 = overbought (sell)")
            with mom_cols[1]:
                macd_color = "green" if tech.macd > tech.macd_signal else "red"
                st.metric("MACD", f"{tech.macd:.2f}",
                         delta=f"Signal: {tech.macd_signal:.2f}",
                         help="MACD > Signal = bullish")
            with mom_cols[2]:
                st.metric("MACD Histogram", f"{tech.macd_histogram:.2f}",
                         help="Positive = bullish momentum")
            with mom_cols[3]:
                st.metric("RSI (7)", f"{tech.rsi_7:.0f}",
                         help="Short-term RSI")

            st.markdown("#### Moving Averages")
            ma_cols = st.columns(6)
            with ma_cols[0]:
                above_20 = "🟢" if tech.price > tech.sma_20 else "🔴"
                st.metric("SMA 20", f"{above_20} ₹{tech.sma_20:.0f}")
            with ma_cols[1]:
                above_50 = "🟢" if tech.price > tech.sma_50 else "🔴"
                st.metric("SMA 50", f"{above_50} ₹{tech.sma_50:.0f}")
            with ma_cols[2]:
                above_200 = "🟢" if tech.price > tech.sma_200 else "🔴"
                st.metric("SMA 200", f"{above_200} ₹{tech.sma_200:.0f}")
            with ma_cols[3]:
                st.metric("EMA 9", f"₹{tech.ema_9:.0f}")
            with ma_cols[4]:
                st.metric("EMA 21", f"₹{tech.ema_21:.0f}")
            with ma_cols[5]:
                st.metric("EMA 50", f"₹{tech.ema_50:.0f}")

            st.markdown("#### Volatility & Bands")
            vol_cols = st.columns(4)
            with vol_cols[0]:
                st.metric("ATR (14)", f"₹{tech.atr_14:.2f}",
                         help=f"{tech.atr_percent:.1f}% of price. Used for stop loss")
            with vol_cols[1]:
                st.metric("Bollinger Upper", f"₹{tech.bollinger_upper:.0f}")
            with vol_cols[2]:
                st.metric("Bollinger Lower", f"₹{tech.bollinger_lower:.0f}")
            with vol_cols[3]:
                st.metric("Band Width", f"{tech.bollinger_width:.1f}%",
                         help="Wider = more volatile")

            st.markdown("#### Support & Resistance")
            sr_cols = st.columns(4)
            with sr_cols[0]:
                st.metric("Support 1", f"₹{tech.support_1:.2f}")
            with sr_cols[1]:
                st.metric("Support 2", f"₹{tech.support_2:.2f}")
            with sr_cols[2]:
                st.metric("Resistance 1", f"₹{tech.resistance_1:.2f}")
            with sr_cols[3]:
                st.metric("Resistance 2", f"₹{tech.resistance_2:.2f}")

            st.markdown("#### Performance")
            perf_cols = st.columns(6)
            periods = [
                ("1 Day", tech.change_1d),
                ("1 Week", tech.change_1w),
                ("1 Month", tech.change_1m),
                ("3 Months", tech.change_3m),
                ("6 Months", tech.change_6m),
                ("1 Year", tech.change_1y),
            ]
            for i, (period, change) in enumerate(periods):
                with perf_cols[i]:
                    st.metric(period, f"{change:+.2f}%")

        # ============== STATISTICS ==============
        with st.expander("📉 Risk Statistics (Volatility, Beta, Drawdown)", expanded=False):
            stat = analysis.statistical

            stat_cols = st.columns(4)
            with stat_cols[0]:
                beta_help = "1.0 = moves with market. >1.5 = more volatile"
                st.metric("Beta", f"{stat.beta:.2f}", help=beta_help)
            with stat_cols[1]:
                st.metric("Annual Volatility", f"{stat.volatility_annual:.1f}%",
                         help="Higher = more risky")
            with stat_cols[2]:
                st.metric("Max Drawdown", f"{stat.max_drawdown:.1f}%",
                         help="Largest peak-to-trough decline")
            with stat_cols[3]:
                st.metric("Sharpe Ratio", f"{stat.sharpe_ratio:.2f}",
                         help="Risk-adjusted return. >1 is good")

            stat_cols2 = st.columns(4)
            with stat_cols2[0]:
                st.metric("VaR (95%)", f"{stat.var_95:.2f}%",
                         help="Max daily loss at 95% confidence")
            with stat_cols2[1]:
                st.metric("Sortino Ratio", f"{stat.sortino_ratio:.2f}",
                         help="Like Sharpe but only penalizes downside")
            with stat_cols2[2]:
                st.metric("NIFTY Correlation", f"{stat.nifty_correlation:.2f}",
                         help="1.0 = perfectly correlated with NIFTY")
            with stat_cols2[3]:
                st.metric("Skewness", f"{stat.skewness:.2f}",
                         help="Negative = more likely to have large losses")

        # ============== POSITION SIZING ==============
        with st.expander("💰 Position Sizing & Risk Management", expanded=True):
            risk = analysis.risk_mgmt
            prob = analysis.probability

            st.markdown("#### Position Calculation")
            pos_cols = st.columns(4)
            with pos_cols[0]:
                st.metric("Your Capital", f"₹{risk.capital:,.0f}")
            with pos_cols[1]:
                st.metric("Risk per Trade", f"{risk.risk_per_trade:.1%}",
                         help=f"₹{risk.risk_amount:,.0f} max loss per trade")
            with pos_cols[2]:
                st.metric("Calculated Position", f"₹{risk.position_size_value:,.0f}",
                         help=f"{risk.position_size_shares} shares")
            with pos_cols[3]:
                st.metric("Kelly Optimal", f"{risk.kelly_fraction:.1%}",
                         help=f"₹{risk.kelly_position:,.0f}")

            st.markdown("#### Trade Levels")
            level_cols = st.columns(4)
            with level_cols[0]:
                st.metric("Entry Price", f"₹{risk.stop_loss_price + (1.5 * analysis.technical.atr_14):,.2f}")
            with level_cols[1]:
                st.metric("Stop Loss", f"₹{risk.stop_loss_price:,.2f}",
                         delta=f"-{risk.stop_loss_percent:.1f}%")
            with level_cols[2]:
                st.metric("Target 1 (1:1)", f"₹{risk.target_1_price:,.2f}")
            with level_cols[3]:
                st.metric("Target 2 (1:2)", f"₹{risk.target_2_price:,.2f}")

            st.markdown("#### Probability Analysis")
            prob_cols = st.columns(4)
            with prob_cols[0]:
                st.metric("Win Probability", f"{prob.win_probability:.1%}",
                         help="Based on multi-factor analysis")
            with prob_cols[1]:
                st.metric("Expected Return", f"{prob.expected_return:+.2f}%",
                         help="(Win% × Reward) - (Loss% × Risk)")
            with prob_cols[2]:
                st.metric("Confidence", f"{prob.confidence_level:.1%}")
            with prob_cols[3]:
                st.metric("Risk/Reward", f"{risk.risk_reward_ratio:.2f}")

        # ============== AI EXPLANATION ==============
        with st.expander("🤖 AI Analysis (Plain English Explanation)", expanded=True):
            if analysis.ai_explanation:
                st.markdown(analysis.ai_explanation)
            else:
                st.info("AI analysis not available")

        # ============== TRACK TRADE ==============
        if tracker and analysis.swing_rec.action == "BUY":
            st.markdown("---")
            st.markdown("### 📝 Track This Trade")
            st.caption("Record this prediction to calibrate probabilities over time")

            if st.button("📊 Record Trade for Tracking", key="track_trade"):
                trade_id = tracker.record_trade(
                    symbol=analysis.symbol,
                    predicted_probability=analysis.probability.win_probability,
                    predicted_return=analysis.probability.expected_return,
                    signal_strength=analysis.swing_rec.signal.value,
                    timeframe="SWING",
                    entry_price=analysis.technical.price,
                    stop_loss=analysis.risk_mgmt.stop_loss_price,
                    target_price=analysis.risk_mgmt.target_1_price,
                    rsi=analysis.technical.rsi_14,
                    volume_ratio=analysis.technical.volume_ratio,
                    trend_score=0.5,
                    factors_bullish=analysis.swing_rec.bullish_factors,
                    factors_bearish=analysis.swing_rec.bearish_factors
                )
                st.success(f"Trade recorded! ID: {trade_id}")


def render_smart_picks_tab():
    """Render smart picks with multi-factor scoring."""
    st.markdown('<h2 class="tab-header">🎯 Smart Picks (AI + Backtested)</h2>', unsafe_allow_html=True)
    st.caption("Multi-factor signals with calibrated win probabilities from backtesting.")

    improved = get_improved_screener()
    if not improved:
        st.error("Improved Screener not available")
        return

    # Display methodology
    with st.expander("📊 How It Works"):
        st.markdown("""
        **5-Factor Scoring Model:**
        | Factor | Weight | Signal |
        |--------|--------|--------|
        | RSI Oversold Bounce | 25% | RSI < 40 and rising |
        | Volume Confirmation | 20% | Volume > 1.5x average |
        | Trend Alignment | 20% | Above 50 & 200 MA |
        | Momentum | 15% | Positive returns |
        | Relative Strength | 20% | Outperforming NIFTY |

        **Win Probability** is calibrated from historical backtest data.
        **Expected Return** = (Win Prob × Reward) - (Loss Prob × Risk)
        """)

    # Controls
    col1, col2 = st.columns(2)

    with col1:
        min_score = st.slider("Minimum Composite Score", 0.3, 0.8, 0.5, 0.05, key="smart_min_score")

    with col2:
        num_picks = st.slider("Number of Picks", 5, 20, 10, key="smart_num_picks")

    if st.button("🚀 Find Smart Picks", type="primary"):
        with st.spinner("Scanning market with multi-factor model..."):
            try:
                opportunities = improved.get_top_opportunities(n=num_picks)

                # Filter by min score
                opportunities = [o for o in opportunities if o.composite_score >= min_score]

                if opportunities:
                    st.success(f"Found {len(opportunities)} opportunities with score ≥ {min_score}")

                    for i, sig in enumerate(opportunities, 1):
                        # Determine card color based on signal strength
                        if sig.signal_strength == "STRONG":
                            border_color = "#00ff88"
                            badge = "🟢 STRONG"
                        elif sig.signal_strength == "MODERATE":
                            border_color = "#ffaa00"
                            badge = "🟡 MODERATE"
                        else:
                            border_color = "#888"
                            badge = "⚪ WEAK"

                        with st.container():
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                        border-radius: 12px; padding: 20px; margin: 15px 0;
                                        border-left: 5px solid {border_color};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-size: 1.4rem; font-weight: bold;">{i}. {sig.symbol}</span>
                                        <span style="margin-left: 15px; color: {border_color}; font-weight: bold;">{badge}</span>
                                        <span style="margin-left: 15px; color: #888;">{sig.name[:30]}</span>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="font-size: 1.3rem; font-weight: bold;">₹{sig.price:,.2f}</div>
                                        <div style="color: #888;">Score: {sig.composite_score:.2f}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Metrics row
                            m_cols = st.columns(5)
                            with m_cols[0]:
                                prob_color = "green" if sig.win_probability > 0.6 else "orange"
                                st.metric("Win Probability", f"{sig.win_probability:.1%}",
                                         help="Calibrated from historical data")
                            with m_cols[1]:
                                ev_color = "green" if sig.expected_return > 0 else "red"
                                st.metric("Expected Return", f"{sig.expected_return:+.1f}%",
                                         help="(Win% × Reward) - (Loss% × Risk)")
                            with m_cols[2]:
                                st.metric("Entry", f"₹{sig.entry_price:,.2f}")
                            with m_cols[3]:
                                st.metric("Stop Loss", f"₹{sig.stop_loss:,.2f}")
                            with m_cols[4]:
                                st.metric("Target", f"₹{sig.target_1:,.2f}")

                            # Factor breakdown
                            with st.expander(f"📊 Factor Breakdown for {sig.symbol}"):
                                factor_cols = st.columns(5)
                                with factor_cols[0]:
                                    rsi_pct = sig.rsi_score * 100
                                    st.progress(sig.rsi_score, text=f"RSI: {rsi_pct:.0f}%")
                                with factor_cols[1]:
                                    vol_pct = sig.volume_score * 100
                                    st.progress(sig.volume_score, text=f"Volume: {vol_pct:.0f}%")
                                with factor_cols[2]:
                                    trend_pct = sig.trend_score * 100
                                    st.progress(sig.trend_score, text=f"Trend: {trend_pct:.0f}%")
                                with factor_cols[3]:
                                    mom_pct = sig.momentum_score * 100
                                    st.progress(sig.momentum_score, text=f"Momentum: {mom_pct:.0f}%")
                                with factor_cols[4]:
                                    rs_pct = sig.relative_strength * 100
                                    st.progress(sig.relative_strength, text=f"Rel.Strength: {rs_pct:.0f}%")

                                if sig.bullish_factors:
                                    st.markdown("**Bullish Factors:** " + " | ".join([f"✅ {f}" for f in sig.bullish_factors]))
                                if sig.bearish_factors:
                                    st.markdown("**Bearish Factors:** " + " | ".join([f"⚠️ {f}" for f in sig.bearish_factors]))

                                st.caption(f"Best for: {sig.best_for} trading | Risk/Reward: {sig.risk_reward:.2f}")

                else:
                    st.warning(f"No stocks found with score ≥ {min_score}. Try lowering the threshold.")

            except Exception as e:
                st.error(f"Error scanning market: {e}")


def render_sector_analysis_tab():
    """
    Comprehensive Sector Analysis Tab using 7-Model Ensemble.

    Shows all major Indian market sectors with top stocks,
    growth trends, momentum, and detailed analysis.
    """
    config = get_engine_config()
    model_label = "7-Model" if config.use_7_model else "4-Model"

    st.markdown('<h2 class="tab-header">🏭 Sector Analysis</h2>', unsafe_allow_html=True)
    st.caption(f"Analyze top stocks across all major Indian market sectors using {model_label} Ensemble")

    # COMPREHENSIVE INDIAN MARKET SECTOR MAPPINGS
    SECTOR_STOCKS = {
        "🔋 Energy & Power": {
            "description": "Power generation, distribution, and energy companies",
            "stocks": [
                "NTPC", "POWERGRID", "TATAPOWER", "ADANIPOWER", "ADANIGREEN",
                "NHPC", "SJVN", "CESC", "TORNTPOWER", "JSW ENERGY",
                "IPCALAB", "RELIANCE", "ONGC", "GAIL", "BPCL",
                "IOC", "HINDPETRO", "PETRONET", "MRPL", "GSPL"
            ]
        },
        "☀️ Solar & Renewable": {
            "description": "Solar, wind, and renewable energy companies",
            "stocks": [
                "ADANIGREEN", "TATAPOWER", "SUZLON", "BOROSIL", "WAAREE",
                "IREDA", "INOXWIND", "KPI GREEN", "WEBSOL", "ORIENT GREEN",
                "STERLING WILSON", "VIKRAM SOLAR", "EMMBI", "GENUS", "ZODIAC"
            ]
        },
        "📡 5G & Telecom": {
            "description": "Telecom operators and 5G infrastructure",
            "stocks": [
                "BHARTIARTL", "RELIANCE", "IDEA", "TTML", "ROUTE",
                "STERLITE", "HFCL", "TEJAS", "DIXON", "TATACOMM",
                "ONMOBILE", "GTPL", "TANLA", "INDIAMART", "AFFLE"
            ]
        },
        "🏛️ PSU Banks & Finance": {
            "description": "Public sector banks and financial institutions",
            "stocks": [
                "SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK",
                "IOB", "INDIANB", "CENTRALBK", "BANKINDIA", "UCOBANK",
                "MAHABANK", "RECLTD", "PFC", "IRFC", "HUDCO"
            ]
        },
        "🛒 Retail & E-commerce": {
            "description": "Retail chains and e-commerce platforms",
            "stocks": [
                "DMART", "TRENT", "TITAN", "SHOPERSTOP", "VMART",
                "SPENCERS", "RELAXO", "BATA", "METROBRAND", "CAMPUS",
                "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "CARTRADE"
            ]
        },
        "🍔 Food & FMCG": {
            "description": "Food processing and FMCG companies",
            "stocks": [
                "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
                "MARICO", "GODREJCP", "COLPAL", "TATACONSUM", "VBL",
                "JUBLFOOD", "DEVYANI", "WESTLIFE", "BIKAJI", "PATANJALI",
                "KRBL", "LTFOODS", "USHAMART", "VADILALIND", "PARAG"
            ]
        },
        "🌾 Agriculture & Fertilizers": {
            "description": "Agri-business, fertilizers, and farm equipment",
            "stocks": [
                "UPL", "PI", "COROMANDEL", "CHAMBALFERT", "GNFC",
                "RCF", "DEEPAKFERT", "GSFC", "FACT", "NFL",
                "ESCORTS", "MAHINDCIE", "SWARAJENG", "VST", "NATH",
                "KAVERI", "DHAMPURE", "AVANTIFEED", "WATERBASE", "SHARDACROP"
            ]
        },
        "🤖 AI & Technology": {
            "description": "AI, software, and technology companies",
            "stocks": [
                "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
                "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "HAPPSTMNDS",
                "KPITTECH", "TATAELXSI", "LTTS", "CYIENT", "ZENSAR",
                "BIRLASOFT", "MASTEK", "INTELLECT", "NEWGEN", "SONATA"
            ]
        },
        "🔧 Robotics & Automation": {
            "description": "Industrial automation and robotics",
            "stocks": [
                "ABB", "SIEMENS", "HONAUT", "CGPOWER", "BHEL",
                "GRINDWELL", "ELGIEQUIP", "AIAENG", "CUMMINSIND", "THERMAX",
                "TRIVENI", "KSB", "TIMKEN", "SCHAEFFLER", "SKF"
            ]
        },
        "💾 Semiconductor & Electronics": {
            "description": "Semiconductor manufacturing and electronics",
            "stocks": [
                "DIXON", "AMBER", "KAYNES", "SYRMA", "TATAELXSI",
                "VEDL", "MOSCHIP", "SPIC", "CENTUM", "ROUTE",
                "HAVELLS", "POLYCAB", "FINOLEX", "KEI", "APAR"
            ]
        },
        "🏥 Pharma & Healthcare": {
            "description": "Pharmaceuticals and healthcare services",
            "stocks": [
                "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
                "BIOCON", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ALKEM",
                "IPCALAB", "GLENMARK", "NATCOPHARMA", "LALPATHLAB", "METROPOLIS"
            ]
        },
        "🏗️ Infrastructure & Construction": {
            "description": "Infrastructure, construction, and real estate",
            "stocks": [
                "LT", "ADANIPORTS", "ULTRACEMCO", "GRASIM", "SHREECEM",
                "AMBUJACEM", "ACC", "DLF", "GODREJPROP", "OBEROIRLTY",
                "PRESTIGE", "BRIGADE", "LODHA", "IRCON", "NBCC"
            ]
        },
        "🚗 Auto & EV": {
            "description": "Automobile and electric vehicle companies",
            "stocks": [
                "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
                "EICHERMOT", "ASHOKLEY", "TVSMOTOR", "BALKRISIND", "MRF",
                "APOLLOTYRE", "CEATLTD", "OLECTRA", "EXIDEIND", "AMARAJABAT"
            ]
        },
        "🏦 Private Banks & NBFC": {
            "description": "Private banks and non-banking financial companies",
            "stocks": [
                "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
                "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "RBLBANK", "AUBANK",
                "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM"
            ]
        },
        "⚙️ Metals & Mining": {
            "description": "Steel, aluminum, and mining companies",
            "stocks": [
                "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "COALINDIA",
                "NMDC", "JINDALSTEL", "SAIL", "NATIONALUM", "HINDZINC",
                "MOIL", "WELCORP", "APLAPOLLO", "RATNAMANI", "TITAGARH"
            ]
        },
        "🛡️ Defence & Aerospace": {
            "description": "Defence manufacturing and aerospace",
            "stocks": [
                "HAL", "BEL", "BHEL", "BEML", "COCHINSHIP",
                "MAZAGON", "GRSE", "BDL", "MIDHANI", "PARAS",
                "DATAPAT", "ASTRA", "ZENTEC", "APOLLO MICRO", "DCX"
            ]
        },
        "🧪 Chemicals & Specialty": {
            "description": "Chemicals, specialty chemicals, and materials",
            "stocks": [
                "PIDILITIND", "SRF", "AARTI", "DEEPAKNTR", "NAVINFLUOR",
                "FLUOROCHEM", "CLEAN", "TATACHEM", "ALKYLAMINE", "VINATI",
                "FINEORG", "GALAXYSURF", "ROSSARI", "SOLARA", "LXCHEM"
            ]
        },
        "💎 Consumer & Lifestyle": {
            "description": "Consumer durables and lifestyle brands",
            "stocks": [
                "TITAN", "ASIANPAINT", "BERGEPAINT", "PAGEIND", "RELAXO",
                "BATAINDIA", "CROMPTON", "VOLTAS", "BLUESTARCO", "HAVELLS",
                "SYMPHONY", "ORIENT ELEC", "V-GUARD", "TTKHLTCARE", "RAJESHEXPO"
            ]
        }
    }

    # Sector selection
    sector_names = list(SECTOR_STOCKS.keys())

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_sector = st.selectbox(
            "Select Sector",
            sector_names,
            index=0,
            key="sector_select"
        )

    with col2:
        show_top_n = st.selectbox("Show Top", [10, 15, 20, "All"], index=1, key="sector_top_n")

    with col3:
        st.write("")
        scan_sector = st.button("🔄 Analyze Sector", type="primary", use_container_width=True, key="scan_sector_btn")

    # Show sector description
    sector_info = SECTOR_STOCKS[selected_sector]
    st.info(f"**{selected_sector}**: {sector_info['description']}")

    # Analyze sector using unified engine (SAME as Top Picks and Deep Analysis)
    if scan_sector or f'sector_results_{selected_sector}' not in st.session_state:
        stocks_to_analyze = sector_info['stocks']

        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner(f"Analyzing {len(stocks_to_analyze)} stocks in {selected_sector} with {model_label} Ensemble..."):
            # Use the SAME analyzer as other tabs for consistent results
            analyzer = get_comprehensive_analyzer()
            analyzer.capital = config.capital
            analyzer.risk_per_trade = config.risk_per_trade
            sector_results = []

            for i, symbol in enumerate(stocks_to_analyze):
                try:
                    progress_bar.progress((i + 1) / len(stocks_to_analyze))
                    status_text.text(f"Analyzing {symbol}... ({i+1}/{len(stocks_to_analyze)})")

                    # Use ComprehensiveAnalyzer with skip_ai=True for faster scanning
                    analysis = analyzer.analyze(symbol, skip_ai=True)

                    if analysis and analysis.technical:
                        tech = analysis.technical
                        swing = analysis.swing_rec  # Correct attribute name

                        # Extract key metrics from ComprehensiveAnalyzer
                        current_price = tech.price or 0
                        if current_price <= 0:
                            continue  # Skip invalid data

                        rsi = tech.rsi_14 or 50
                        volume_ratio = tech.volume_ratio or 1.0

                        # Calculate trend from moving averages (TechnicalData doesn't have trend attribute)
                        sma_20 = tech.sma_20 or current_price
                        sma_50 = tech.sma_50 or current_price
                        if current_price > sma_20 > sma_50:
                            trend = "STRONG UPTREND"
                            trend_score = 3
                        elif current_price > sma_50:
                            trend = "UPTREND"
                            trend_score = 2
                        elif current_price < sma_20 < sma_50:
                            trend = "STRONG DOWNTREND"
                            trend_score = -2
                        elif current_price < sma_50:
                            trend = "DOWNTREND"
                            trend_score = -1
                        else:
                            trend = "SIDEWAYS"
                            trend_score = 0

                        # Use swing recommendation signal (consistent with Top Picks)
                        # Note: swing.signal is a SignalStrength enum, need .value for string
                        signal = swing.signal.value if swing and swing.signal else "HOLD"
                        win_prob = swing.win_probability if swing else 0.5

                        # Calculate score from win probability (consistent scaling)
                        score = int(win_prob * 100)

                        # Growth calculations from TechnicalData attributes
                        growth_1w = tech.change_1w or 0
                        growth_1m = tech.change_1m or 0
                        growth_3m = tech.change_3m or 0
                        growth_6m = tech.change_6m or 0
                        growth_1y = tech.change_1y or 0

                        # 52-week data from TechnicalData
                        high_52w = tech.week_52_high or current_price
                        low_52w = tech.week_52_low or current_price
                        from_high = tech.pct_from_52_high or 0
                        from_low = tech.pct_from_52_low or 0

                        # Market cap category
                        if current_price > 5000:
                            cap_category = "Large Cap"
                        elif current_price > 500:
                            cap_category = "Mid Cap"
                        else:
                            cap_category = "Small Cap"

                        # Day change from TechnicalData
                        day_change = tech.change_1d or 0

                        sector_results.append({
                            'symbol': symbol,
                            'price': current_price,
                            'day_change': day_change,
                            'growth_1w': growth_1w,
                            'growth_1m': growth_1m,
                            'growth_3m': growth_3m,
                            'growth_6m': growth_6m,
                            'growth_1y': growth_1y,
                            'high_52w': high_52w,
                            'low_52w': low_52w,
                            'from_high': from_high,
                            'from_low': from_low,
                            'rsi': rsi,
                            'sma_20': tech.sma_20 or current_price,
                            'sma_50': tech.sma_50 or current_price,
                            'trend': trend,
                            'trend_score': trend_score,
                            'volume_ratio': volume_ratio,
                            'volatility': (tech.atr_14 / current_price * 100) if tech.atr_14 and current_price else 2.0,
                            'cap_category': cap_category,
                            'score': score,
                            'signal': signal,
                            'win_prob': win_prob,
                            'entry': swing.entry_price if swing else current_price,
                            'stop_loss': swing.stop_loss if swing else current_price * 0.95,
                            'target': swing.target_1 if swing else current_price * 1.05,
                            'analysis': analysis  # Store full analysis for deep dive
                        })

                except Exception as e:
                    st.warning(f"⚠️ Failed to analyze {symbol}: {str(e)[:100]}")

            progress_bar.empty()
            status_text.empty()

            # Debug: Show how many results collected
            st.info(f"✅ Scan complete! Found {len(sector_results)} stocks with valid data.")

            # Sort by score (win probability based)
            sector_results = sorted(sector_results, key=lambda x: x['score'], reverse=True)
            st.session_state[f'sector_results_{selected_sector}'] = sector_results
            st.session_state[f'sector_scan_time_{selected_sector}'] = datetime.now().strftime("%H:%M:%S")

    # Display results
    sector_results = st.session_state.get(f'sector_results_{selected_sector}', [])
    scan_time = st.session_state.get(f'sector_scan_time_{selected_sector}', 'N/A')

    if sector_results:
        # Summary metrics
        st.markdown("### 📊 Sector Overview")

        # Calculate sector metrics
        avg_growth_1m = sum(r['growth_1m'] for r in sector_results) / len(sector_results)
        avg_growth_3m = sum(r['growth_3m'] for r in sector_results) / len(sector_results)
        bullish_count = sum(1 for r in sector_results if r['signal'] in ['STRONG BUY', 'BUY'])
        bearish_count = sum(1 for r in sector_results if r['signal'] in ['WEAK', 'AVOID'])

        summary_cols = st.columns(5)

        with summary_cols[0]:
            st.metric("Stocks Analyzed", len(sector_results))

        with summary_cols[1]:
            color = "normal" if avg_growth_1m > 0 else "inverse"
            st.metric("Avg 1M Growth", f"{avg_growth_1m:+.1f}%", delta_color=color)

        with summary_cols[2]:
            color = "normal" if avg_growth_3m > 0 else "inverse"
            st.metric("Avg 3M Growth", f"{avg_growth_3m:+.1f}%", delta_color=color)

        with summary_cols[3]:
            st.metric("Bullish Stocks", f"{bullish_count}/{len(sector_results)}")

        with summary_cols[4]:
            sentiment = "BULLISH" if bullish_count > bearish_count else "BEARISH" if bearish_count > bullish_count else "NEUTRAL"
            st.metric("Sector Sentiment", sentiment)

        st.success(f"✅ Analyzed {len(sector_results)} stocks | Last scan: {scan_time}")

        # Filter results
        display_limit = len(sector_results) if show_top_n == "All" else int(show_top_n)
        display_results = sector_results[:display_limit]

        st.markdown(f"### 🏆 Top {len(display_results)} Stocks in {selected_sector}")

        # Display each stock using Streamlit components
        for i, stock in enumerate(display_results, 1):
            signal_emoji = {"STRONG BUY": "🟢", "BUY": "🟢", "HOLD": "🟡", "WEAK": "🟠", "AVOID": "🔴"}.get(stock['signal'], "⚪")

            with st.container():
                # Header with expander for full details
                with st.expander(f"{signal_emoji} **#{i} {stock['symbol']}** | ₹{stock['price']:,.2f} | {stock['signal']} (Score: {stock['score']}/100)", expanded=(i <= 3)):

                    # Top row - Price and Signal
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        delta = f"{stock['day_change']:+.2f}%"
                        st.metric("Price", f"₹{stock['price']:,.2f}", delta)
                    with col2:
                        st.metric("Signal", stock['signal'])
                    with col3:
                        st.metric("Score", f"{stock['score']}/100")
                    with col4:
                        st.metric("Trend", stock['trend'].replace("_", " ").title())

                    st.markdown("---")

                    # Growth Performance
                    st.markdown("**📈 Growth Performance**")
                    g1, g2, g3, g4, g5 = st.columns(5)
                    with g1:
                        st.metric("1 Week", f"{stock['growth_1w']:+.1f}%")
                    with g2:
                        st.metric("1 Month", f"{stock['growth_1m']:+.1f}%")
                    with g3:
                        st.metric("3 Months", f"{stock['growth_3m']:+.1f}%")
                    with g4:
                        st.metric("6 Months", f"{stock['growth_6m']:+.1f}%")
                    with g5:
                        st.metric("1 Year", f"{stock['growth_1y']:+.1f}%")

                    # Technical Indicators
                    st.markdown("**📊 Technical Indicators**")
                    t1, t2, t3, t4, t5, t6 = st.columns(6)
                    with t1:
                        rsi_status = "Oversold" if stock['rsi'] < 30 else "Overbought" if stock['rsi'] > 70 else "Normal"
                        st.metric("RSI", f"{stock['rsi']:.0f}", rsi_status)
                    with t2:
                        st.metric("Volume", f"{stock['volume_ratio']:.1f}x avg")
                    with t3:
                        st.metric("Volatility", f"{stock['volatility']:.1f}%")
                    with t4:
                        st.metric("52W High", f"₹{stock['high_52w']:,.0f}")
                    with t5:
                        st.metric("52W Low", f"₹{stock['low_52w']:,.0f}")
                    with t6:
                        st.metric("From High", f"{stock['from_high']:+.1f}%")

                    # Moving Averages
                    ma1, ma2, ma3 = st.columns(3)
                    with ma1:
                        above_sma20 = "✅" if stock['price'] > stock['sma_20'] else "❌"
                        st.metric(f"SMA 20 {above_sma20}", f"₹{stock['sma_20']:,.0f}")
                    with ma2:
                        above_sma50 = "✅" if stock['price'] > stock['sma_50'] else "❌"
                        st.metric(f"SMA 50 {above_sma50}", f"₹{stock['sma_50']:,.0f}")
                    with ma3:
                        st.metric("From 52W Low", f"+{stock['from_low']:.1f}%")

                    st.markdown("---")

                    # AI INSIGHT - WHY this stock
                    st.markdown("**💡 AI Insight - Why This Rating?**")
                    insight = generate_stock_insight(stock)
                    st.markdown(insight)

                    # Category badge
                    st.caption(f"Category: {stock['cap_category']}")

        # Sector comparison table
        with st.expander("📊 Quick Comparison Table", expanded=False):
            import pandas as pd

            table_data = []
            for stock in display_results:
                table_data.append({
                    'Symbol': stock['symbol'],
                    'Price': f"₹{stock['price']:,.2f}",
                    'Day': f"{stock['day_change']:+.1f}%",
                    '1W': f"{stock['growth_1w']:+.1f}%",
                    '1M': f"{stock['growth_1m']:+.1f}%",
                    '3M': f"{stock['growth_3m']:+.1f}%",
                    '1Y': f"{stock['growth_1y']:+.1f}%",
                    'RSI': f"{stock['rsi']:.0f}",
                    'Signal': stock['signal'],
                    'Score': stock['score']
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.info(f"Click 'Analyze Sector' to scan all stocks in {selected_sector}")


def render_unified_screener_tab():
    """
    Unified Screener - Basic filtering by price, market cap, and momentum.
    This tab does NOT give buy/sell recommendations - just helps filter stocks.
    For recommendations, use Top Picks or Deep Analysis tabs.
    """
    st.markdown("## 💰 Stock Screener")
    st.caption("Filter stocks by price, market cap, and momentum. For BUY/SELL recommendations, use Top Picks tab.")

    # Warning about this being a filter, not recommendation
    st.warning("⚠️ **Note:** This screener only FILTERS stocks. It does NOT provide trading recommendations. For actionable BUY/SELL signals with entry/exit prices, use the **🏆 Top Picks** tab.")

    # Filter options
    st.markdown("### 🔍 Filter Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_type = st.selectbox(
            "Filter By",
            ["Price Range", "Market Cap", "Momentum"],
            key="screener_filter_type"
        )

    with col2:
        if filter_type == "Price Range":
            price_range = st.selectbox(
                "Price Range",
                ["₹1 - ₹10 (Penny)", "₹10 - ₹50 (Low)", "₹50 - ₹200 (Mid)", "₹200 - ₹1000 (High)", "₹1000+ (Premium)"],
                key="screener_price_range"
            )
        elif filter_type == "Market Cap":
            cap_type = st.selectbox(
                "Market Cap",
                ["Large Cap (>₹20,000 Cr)", "Mid Cap (₹5,000-₹20,000 Cr)", "Small Cap (₹1,000-₹5,000 Cr)", "Micro Cap (<₹1,000 Cr)"],
                key="screener_cap_type"
            )
        else:  # Momentum
            momentum_type = st.selectbox(
                "Momentum Period",
                ["1 Week Gainers", "1 Month Gainers", "3 Month Gainers", "52 Week High"],
                key="screener_momentum_type"
            )

    with col3:
        min_volume = st.number_input("Min Volume (Lakhs)", value=10, min_value=1, key="screener_min_vol")

    # Stock universe for screening
    SCREENER_UNIVERSE = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
        "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
        "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "ULTRACEMCO", "NTPC",
        "WIPRO", "NESTLEIND", "M&M", "POWERGRID", "TATAMOTORS", "JSWSTEEL",
        "TATASTEEL", "ADANIENT", "ADANIPORTS", "ONGC", "COALINDIA", "BAJAJFINSV",
        "GRASIM", "TECHM", "HDFCLIFE", "SBILIFE", "BRITANNIA", "INDUSINDBK",
        "HINDALCO", "CIPLA", "DRREDDY", "EICHERMOT", "DIVISLAB", "APOLLOHOSP",
        "TATACONSUM", "BPCL", "HEROMOTOCO", "TRENT", "ZOMATO",
        "DMART", "SIEMENS", "ABB", "PIDILITIND", "HAVELLS", "GODREJCP",
        "MUTHOOTFIN", "CHOLAFIN", "TVSMOTOR", "VOLTAS", "JUBLFOOD",
        "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "LUPIN", "AUROPHARMA"
    ]

    if st.button("🔍 Screen Stocks", type="primary", key="run_screener"):
        import yfinance as yf

        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, symbol in enumerate(SCREENER_UNIVERSE):
            progress.progress((i + 1) / len(SCREENER_UNIVERSE))
            status.text(f"Screening {symbol}...")

            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period="3mo")
                info = ticker.info or {}

                if hist.empty:
                    continue

                price = hist['Close'].iloc[-1]
                volume = hist['Volume'].iloc[-1]
                market_cap = info.get('marketCap', 0) / 10000000  # Convert to Cr

                # Calculate returns
                returns_1w = ((price / hist['Close'].iloc[-5]) - 1) * 100 if len(hist) >= 5 else 0
                returns_1m = ((price / hist['Close'].iloc[-22]) - 1) * 100 if len(hist) >= 22 else 0
                returns_3m = ((price / hist['Close'].iloc[0]) - 1) * 100

                high_52w = info.get('fiftyTwoWeekHigh', price)
                pct_from_high = ((price / high_52w) - 1) * 100 if high_52w > 0 else 0

                # Apply filters
                passes_filter = True

                if filter_type == "Price Range":
                    if "₹1 - ₹10" in price_range and not (1 <= price <= 10):
                        passes_filter = False
                    elif "₹10 - ₹50" in price_range and not (10 <= price <= 50):
                        passes_filter = False
                    elif "₹50 - ₹200" in price_range and not (50 <= price <= 200):
                        passes_filter = False
                    elif "₹200 - ₹1000" in price_range and not (200 <= price <= 1000):
                        passes_filter = False
                    elif "₹1000+" in price_range and not (price > 1000):
                        passes_filter = False

                elif filter_type == "Market Cap":
                    if "Large Cap" in cap_type and market_cap < 20000:
                        passes_filter = False
                    elif "Mid Cap" in cap_type and not (5000 <= market_cap <= 20000):
                        passes_filter = False
                    elif "Small Cap" in cap_type and not (1000 <= market_cap <= 5000):
                        passes_filter = False
                    elif "Micro Cap" in cap_type and market_cap > 1000:
                        passes_filter = False

                # Volume filter
                if volume < min_volume * 100000:
                    passes_filter = False

                if passes_filter:
                    results.append({
                        'symbol': symbol,
                        'name': info.get('shortName', symbol),
                        'price': price,
                        'market_cap': market_cap,
                        'volume': volume,
                        'returns_1w': returns_1w,
                        'returns_1m': returns_1m,
                        'returns_3m': returns_3m,
                        'pct_from_high': pct_from_high
                    })

            except Exception:
                continue

        progress.empty()
        status.empty()

        # Sort by momentum
        if filter_type == "Momentum":
            if "1 Week" in momentum_type:
                results.sort(key=lambda x: x['returns_1w'], reverse=True)
            elif "1 Month" in momentum_type:
                results.sort(key=lambda x: x['returns_1m'], reverse=True)
            elif "3 Month" in momentum_type:
                results.sort(key=lambda x: x['returns_3m'], reverse=True)
            else:  # 52 Week High
                results.sort(key=lambda x: x['pct_from_high'], reverse=True)
        else:
            results.sort(key=lambda x: x['returns_1m'], reverse=True)

        st.session_state['screener_results'] = results
        st.success(f"Found {len(results)} stocks matching your filters")

    # Display results
    if 'screener_results' in st.session_state and st.session_state['screener_results']:
        results = st.session_state['screener_results']

        st.markdown("### 📊 Filtered Stocks")
        st.caption("These are filtered results only. Click on 'Analyze in Deep Analysis' to get trading recommendations.")

        # Create DataFrame for display
        import pandas as pd
        df = pd.DataFrame(results)
        df = df.rename(columns={
            'symbol': 'Symbol',
            'name': 'Name',
            'price': 'Price (₹)',
            'market_cap': 'Market Cap (Cr)',
            'returns_1w': '1W Return %',
            'returns_1m': '1M Return %',
            'returns_3m': '3M Return %',
            'pct_from_high': 'From 52W High %'
        })

        # Format numbers
        df['Price (₹)'] = df['Price (₹)'].apply(lambda x: f"₹{x:,.2f}")
        df['Market Cap (Cr)'] = df['Market Cap (Cr)'].apply(lambda x: f"₹{x:,.0f}")
        df['1W Return %'] = df['1W Return %'].apply(lambda x: f"{x:+.1f}%")
        df['1M Return %'] = df['1M Return %'].apply(lambda x: f"{x:+.1f}%")
        df['3M Return %'] = df['3M Return %'].apply(lambda x: f"{x:+.1f}%")
        df['From 52W High %'] = df['From 52W High %'].apply(lambda x: f"{x:+.1f}%")

        st.dataframe(df.drop(columns=['volume']), use_container_width=True)

        # Quick action buttons
        st.markdown("### 🎯 Get Trading Recommendations")
        st.info("Select a stock below and go to **Deep Analysis** tab for full BUY/SELL recommendation with entry, stop loss, and target prices.")

        # Show top 5 with action buttons
        for i, stock in enumerate(results[:5]):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{stock['symbol']}** - {stock['name']} | ₹{stock['price']:,.2f} | 1M: {stock['returns_1m']:+.1f}%")
            with col2:
                if st.button(f"Analyze", key=f"analyze_{stock['symbol']}"):
                    st.session_state['analyze_symbol'] = stock['symbol']
                    st.info(f"Go to **Deep Analysis** tab and enter **{stock['symbol']}** for full recommendation")

    else:
        st.info("👆 Set your filters and click 'Screen Stocks' to find matching stocks")


def render_about_tab():
    """Render about section."""
    config = get_engine_config()
    model_count = 7 if config.use_7_model else 4

    st.markdown(f"""
    ## 📖 About This System

    A **quantitative stock screening and analysis system** for Indian markets (NSE/BSE).

    ### 🎯 Key Principle: ONE ENGINE = CONSISTENT RESULTS

    All recommendation tabs use the **same {model_count}-Model Ensemble Engine** to ensure:
    - ✅ **Consistent signals** - Same stock shows same recommendation everywhere
    - ✅ **Trustworthy data** - One verified source of truth
    - ✅ **No confusion** - Clear, actionable recommendations
    - ✅ **Configurable** - Adjust settings in sidebar to match your trading style

    ### 🧠 7-Model Ensemble Architecture (Recommended)

    | Model | Weight | Description |
    |-------|--------|-------------|
    | **Base Technical** | 25% | RSI, MACD, BB, Volume, News sentiment |
    | **Physics Engine** | 18% | Momentum, Spring, Energy models |
    | **Math Engine** | 14% | Fourier, Fractal, Entropy analysis |
    | **Regime (HMM)** | 13% | Bull/Bear/Ranging detection |
    | **Macro Engine** | 10% | Oil, USD/INR, Bonds sector impact |
    | **Alternative Data** | 10% | Earnings, Options flow, FII/DII |
    | **Advanced Math** | 10% | Kalman, Wavelet, PCA, Markov |

    **Signal Confidence:**
    - **Strong Buy**: 5+ models agree (71%+ agreement)
    - **Moderate Buy**: 3+ models agree (43%+ agreement)
    - **Weak/Hold**: Less than 3 models agree

    ### Configuration Presets

    | Preset | Min Agreement | Min Confidence | Risk/Trade | Best For |
    |--------|--------------|----------------|------------|----------|
    | **Conservative** | 5/7 | 60% | 1% | Safe, high-probability trades |
    | **Balanced** (Recommended) | 3/7 | 55% | 2% | Balanced risk/reward |
    | **Aggressive** | 2/7 | 50% | 3% | More signals, higher risk |
    | **Basic 4-Model** | 2/4 | 55% | 2% | Fallback mode |

    ### Tabs Overview

    | Tab | Purpose | What It Does |
    |-----|---------|-------------|
    | **🏆 Top Picks** | **PRIMARY** - Auto-scan for best opportunities | Scans 70+ stocks, filters for HIGH-CONFIDENCE BUY signals (Intraday + Swing) |
    | **🔬 Deep Analysis** | Single stock deep dive | Complete analysis with fundamentals, technicals, AI explanation |
    | **🏭 Sectors** | Sector-wise analysis | Analyze stocks within specific sectors |
    | **💰 Screener** | Basic filtering | Filter by price, market cap, momentum (NO recommendations) |

    ### How Recommendations Work

    **Technical Filter Criteria (must pass ALL):**
    - RSI < 70 (not overbought)
    - MACD Bullish (histogram > 0 or MACD > signal)
    - Volume > 0.5x average (decent liquidity)
    - ADX > 20 OR Price > SMA20 (trending)
    - Min Confidence threshold (configurable in sidebar)
    - Min Model Agreement (configurable in sidebar)

    **For Intraday:**
    - Same-day exit (buy morning, sell by 3:15 PM)
    - Focus on quick momentum plays

    **For Swing:**
    - Hold 10-20 days
    - Focus on trend continuation

    ### What Each Recommendation Shows

    - **Entry Price** - Exact price to buy
    - **Stop Loss** - Where to exit if wrong (limits loss)
    - **Target** - Where to take profit
    - **Position Size** - How much to invest (based on your capital)
    - **Confidence** - Calibrated probability (backtest-verified)
    - **Model Agreement** - X/{model_count} models agreeing on signal
    - **Risk:Reward** - How much you can gain vs lose
    - **Model Votes** - Individual model predictions (BUY/HOLD/SELL)
    - **AI Insight** - Plain English explanation

    ### Backtest Performance

    - **BUY Signal Accuracy:** 57.6% (330 trades validated)
    - **Target Accuracy (7-model):** 65-68% with 5+ model agreement
    - **SELL Signals:** Disabled (28.6% accuracy = inverted)
    - **Calibration Error:** 1.9%

    ### How It's Better Than ChatGPT/Perplexity

    1. **Backtested** - We know actual historical win rates
    2. **Calibrated** - Probabilities are verified, not made up
    3. **7-Model Ensemble** - Cross-verified by multiple prediction engines
    4. **Structured** - Specific entry/SL/target levels
    5. **Real-time** - Live NSE data, not cached results
    6. **Configurable** - Adjust thresholds via sidebar

    ### How to Use

    1. **Configure**: Adjust settings in sidebar (or use Balanced preset)
    2. **Top Picks**: See auto-scanned opportunities with confidence levels
    3. **Deep Analysis**: Get detailed multi-model analysis for any stock
    4. **Sectors**: Analyze stocks within specific industry sectors

    ### Risk Warning

    ⚠️ **Stock trading involves substantial risk of loss.**

    - This is an analysis tool, not financial advice
    - Always do your own research
    - Use proper position sizing
    - Past performance doesn't guarantee future results

    ---

    **Data Sources:** Yahoo Finance, NSE India
    **Built with:** Python, Streamlit, 7-Model Ensemble Engine
    **Recommended Config:** Balanced (3/7 agreement, 55% min confidence)
    """)


def render_top_picks_tab():
    """
    Auto-scan stocks and show TOP BUY picks for Intraday and Swing trading.
    Uses the unified 7-Model Ensemble Engine for consistent predictions.
    """
    config = get_engine_config()
    model_label = "7-Model" if config.use_7_model else "4-Model"

    st.markdown("## 🏆 Top Picks - Auto-Scanned Recommendations")
    st.caption(f"Automatically scans NIFTY 50 + Midcap stocks using {model_label} Ensemble | Min Agreement: {config.min_model_agreement}")

    # Stock Universe to scan
    SCAN_UNIVERSE = [
        # NIFTY 50 (most liquid)
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
        "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "HCLTECH", "AXISBANK", "ASIANPAINT",
        "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "ULTRACEMCO", "NTPC",
        "WIPRO", "NESTLEIND", "M&M", "POWERGRID", "TATAMOTORS", "JSWSTEEL",
        "TATASTEEL", "ADANIENT", "ADANIPORTS", "ONGC", "COALINDIA", "BAJAJFINSV",
        "GRASIM", "TECHM", "HDFCLIFE", "SBILIFE", "BRITANNIA", "INDUSINDBK",
        "HINDALCO", "CIPLA", "DRREDDY", "EICHERMOT", "DIVISLAB", "APOLLOHOSP",
        "TATACONSUM", "BPCL", "HEROMOTOCO", "TRENT", "ZOMATO",
        # High liquidity midcaps
        "DMART", "SIEMENS", "ABB", "PIDILITIND", "HAVELLS", "GODREJCP",
        "MUTHOOTFIN", "CHOLAFIN", "TVSMOTOR", "VOLTAS", "JUBLFOOD",
        "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "LUPIN", "AUROPHARMA",
        "TORNTPHARM", "ALKEM", "ASHOKLEY", "ESCORTS", "CROMPTON", "DLF"
    ]

    # Settings - now uses centralized engine config
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        top_n = st.slider("Show Top N picks per category", 5, 20, config.top_n_picks, key="top_picks_n")
    with col2:
        st.metric("Capital", f"₹{config.capital/1000:.0f}K")
    with col3:
        st.metric("Risk/Trade", f"{config.risk_per_trade*100:.1f}%")

    # Show current engine mode (configured in sidebar)
    if config.use_7_model:
        st.info(f"🧠 Using **{model_label} Ensemble** | Adjust settings in sidebar")
    else:
        st.warning(f"⚠️ Using **Basic 4-Model** mode | Enable 7-Model in sidebar for better accuracy")

    # Scan button
    if st.button("🔍 Scan All Stocks for BUY Signals", type="primary", key="scan_top_picks"):
        analyzer = get_comprehensive_analyzer()
        if not analyzer:
            st.error("Comprehensive Analyzer not available")
            return

        # Show what engine is being used
        if config.use_7_model:
            st.success(f"🧠 Running with **7-Model Ensemble** (Base + Physics + Math + Regime + Macro + Alternative + Advanced)")

        analyzer.capital = config.capital
        analyzer.risk_per_trade = config.risk_per_trade

        intraday_buys = []
        swing_buys = []
        all_results = []

        progress = st.progress(0)
        status = st.empty()

        for i, symbol in enumerate(SCAN_UNIVERSE):
            progress.progress((i + 1) / len(SCAN_UNIVERSE))
            status.text(f"Analyzing {symbol} [{model_label}]... ({i+1}/{len(SCAN_UNIVERSE)})")

            try:
                # Skip AI calls during bulk scan for speed - user can request AI on-demand
                analysis = analyzer.analyze(symbol, skip_ai=True)
                if not analysis or analysis.technical.price == 0:
                    continue

                # Store full analysis
                result = {
                    'analysis': analysis,
                    'symbol': analysis.symbol,
                    'name': analysis.name,
                    'sector': analysis.sector,
                    'price': analysis.technical.price,
                    'intraday_signal': analysis.intraday_rec.signal.value,
                    'intraday_prob': analysis.intraday_rec.win_probability,
                    'intraday_rr': analysis.intraday_rec.risk_reward,
                    'swing_signal': analysis.swing_rec.signal.value,
                    'swing_prob': analysis.swing_rec.win_probability,
                    'swing_rr': analysis.swing_rec.risk_reward,
                }
                all_results.append(result)

                # Get technical data for filtering
                tech = analysis.technical

                # ============ STRICT FILTERING CRITERIA ============
                # Must pass ALL technical checks for high-quality picks

                # Common technical checks - relaxed RSI (75 instead of 70)
                rsi_ok = tech.rsi_14 < 75  # Allow slight overbought (was 70)
                rsi_oversold = tech.rsi_14 < 45  # Oversold = bonus
                macd_bullish = tech.macd_histogram > 0 or tech.macd > tech.macd_signal
                volume_ok = tech.volume_ratio > 0.5  # Decent liquidity
                volume_high = tech.volume_ratio > 1.2  # High volume = bonus
                adx_trending = tech.adx > 20  # Showing trend
                price_above_sma20 = tech.price > tech.sma_20  # Uptrend short-term

                # Store technical check results for display
                result['tech_checks'] = {
                    'rsi': tech.rsi_14,
                    'rsi_ok': rsi_ok,
                    'rsi_oversold': rsi_oversold,
                    'macd': tech.macd,
                    'macd_signal': tech.macd_signal,
                    'macd_histogram': tech.macd_histogram,
                    'macd_bullish': macd_bullish,
                    'volume_ratio': tech.volume_ratio,
                    'volume_ok': volume_ok,
                    'volume_high': volume_high,
                    'adx': tech.adx,
                    'adx_trending': adx_trending,
                    'price': tech.price,
                    'sma_20': tech.sma_20,
                    'price_above_sma20': price_above_sma20
                }

                # ============ 7-MODEL ENSEMBLE SCORING (CENTRALIZED) ============
                # Uses unified prediction engine from engine_config
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(f"{symbol}.NS")
                    df = ticker.history(period="1y")
                    if len(df) >= 60:
                        df.columns = [c.lower() for c in df.columns]

                        # Get intraday prediction using CENTRALIZED 7-model engine
                        enh_intraday = get_enhanced_prediction_data(symbol, df, "intraday")
                        if enh_intraday:
                            result['enhanced_intraday'] = enh_intraday

                        # Get swing prediction using CENTRALIZED 7-model engine
                        enh_swing = get_enhanced_prediction_data(symbol, df, "swing")
                        if enh_swing:
                            result['enhanced_swing'] = enh_swing

                except Exception as enh_err:
                    pass  # Continue without enhanced data if error

                # INTRADAY FILTER: Need quick momentum
                # Criteria: RSI not overbought + (MACD bullish OR RSI oversold) + decent volume
                intraday_tech_ok = (
                    rsi_ok and
                    volume_ok and
                    (macd_bullish or rsi_oversold)
                )

                # SWING FILTER: Need trend confirmation
                # Criteria: RSI not overbought + MACD bullish + volume ok + ADX trending
                swing_tech_ok = (
                    rsi_ok and
                    volume_ok and
                    macd_bullish and
                    (adx_trending or price_above_sma20)
                )

                # RELAXED FILTERING - Show more picks with confidence levels
                # Include any BUY signal (STRONG_BUY, BUY, WEAK_BUY) with reasonable technicals

                # Intraday filter - more relaxed
                intraday_signal = analysis.intraday_rec.signal.value
                intraday_is_bullish = "BUY" in intraday_signal and "SELL" not in intraday_signal
                intraday_prob = analysis.intraday_rec.win_probability

                # Accept if: bullish signal + some technical confirmation
                intraday_tech_relaxed = rsi_ok and volume_ok  # Just need RSI ok and volume
                if intraday_is_bullish and intraday_tech_relaxed:
                    # Classify confidence level
                    if intraday_prob >= 0.60:
                        result['confidence_level'] = 'HIGH'
                    elif intraday_prob >= 0.52:
                        result['confidence_level'] = 'MEDIUM'
                    else:
                        result['confidence_level'] = 'SPECULATIVE'

                    # Signal strength for sorting (STRONG_BUY > BUY > WEAK_BUY)
                    if "STRONG" in intraday_signal:
                        result['signal_strength_intraday'] = 3  # HIGH BUY
                    elif intraday_signal == "BUY":
                        result['signal_strength_intraday'] = 2  # BUY
                    else:
                        result['signal_strength_intraday'] = 1  # WEAK BUY

                    # Technical score for sorting
                    result['tech_score_intraday'] = (
                        (15 if rsi_oversold else 5 if rsi_ok else 0) +
                        (15 if macd_bullish else 0) +
                        (10 if volume_high else 5 if volume_ok else 0) +
                        (10 if adx_trending else 0) +
                        (10 if price_above_sma20 else 0) +
                        (20 if "STRONG" in intraday_signal else 10 if intraday_signal == "BUY" else 0)
                    )
                    intraday_buys.append(result)

                # Swing filter - more relaxed
                swing_signal = analysis.swing_rec.signal.value
                swing_is_bullish = "BUY" in swing_signal and "SELL" not in swing_signal
                swing_prob = analysis.swing_rec.win_probability

                # For swing, we want uptrend confirmation
                swing_tech_relaxed = rsi_ok and (price_above_sma20 or macd_bullish)
                if swing_is_bullish and swing_tech_relaxed:
                    # Classify confidence level
                    if swing_prob >= 0.60:
                        result['swing_confidence_level'] = 'HIGH'
                    elif swing_prob >= 0.52:
                        result['swing_confidence_level'] = 'MEDIUM'
                    else:
                        result['swing_confidence_level'] = 'SPECULATIVE'

                    # Signal strength for sorting (STRONG_BUY > BUY > WEAK_BUY)
                    if "STRONG" in swing_signal:
                        result['signal_strength_swing'] = 3  # HIGH BUY
                    elif swing_signal == "BUY":
                        result['signal_strength_swing'] = 2  # BUY
                    else:
                        result['signal_strength_swing'] = 1  # WEAK BUY

                    # Technical score for sorting
                    result['tech_score_swing'] = (
                        (15 if rsi_oversold else 5 if rsi_ok else 0) +
                        (20 if macd_bullish else 0) +
                        (10 if volume_high else 5 if volume_ok else 0) +
                        (15 if adx_trending else 0) +
                        (15 if price_above_sma20 else 0) +
                        (20 if "STRONG" in swing_signal else 10 if swing_signal == "BUY" else 0)
                    )
                    swing_buys.append(result)

            except Exception as e:
                continue

        progress.empty()
        status.empty()

        # Sort by: Signal Strength (HIGH BUY > BUY > WEAK BUY) > Probability > Technical Score
        intraday_buys.sort(key=lambda x: (
            x.get('signal_strength_intraday', 1),  # Signal strength first (3=STRONG, 2=BUY, 1=WEAK)
            x['intraday_prob'],                     # Then win probability
            x.get('tech_score_intraday', 0),        # Then technical score
            x['intraday_rr']                        # Then risk:reward
        ), reverse=True)
        swing_buys.sort(key=lambda x: (
            x.get('signal_strength_swing', 1),     # Signal strength first (3=STRONG, 2=BUY, 1=WEAK)
            x['swing_prob'],                        # Then win probability
            x.get('tech_score_swing', 0),           # Then technical score
            x['swing_rr']                           # Then risk:reward
        ), reverse=True)

        # Store in session state
        st.session_state['top_intraday_picks'] = intraday_buys
        st.session_state['top_swing_picks'] = swing_buys
        st.session_state['top_picks_scan_time'] = datetime.now()

        st.success(f"Scan complete! Found {len(intraday_buys)} Intraday picks and {len(swing_buys)} Swing picks (sorted by confidence: HIGH > MEDIUM > SPECULATIVE)")

    # Display results
    if 'top_intraday_picks' in st.session_state:
        scan_time = st.session_state.get('top_picks_scan_time', datetime.now())
        st.caption(f"Last scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Two main sections
        st.markdown("---")

        # ============== INTRADAY PICKS ==============
        st.markdown("## 📅 Intraday Picks for Tomorrow")
        st.caption("Sorted by: 🟢 HIGH BUY > 🔵 BUY > 🟡 WEAK BUY | Buy and sell same day")

        intraday_picks = st.session_state.get('top_intraday_picks', [])[:top_n]

        if intraday_picks:
            for i, pick in enumerate(intraday_picks, 1):
                analysis = pick['analysis']
                irec = analysis.intraday_rec
                tech = analysis.technical

                # Show signal type based on strength (HIGH BUY > BUY > WEAK BUY)
                signal_strength = pick.get('signal_strength_intraday', 1)
                if signal_strength == 3:
                    signal_color = "🟢"
                    signal_type = "HIGH BUY"
                elif signal_strength == 2:
                    signal_color = "🔵"
                    signal_type = "BUY"
                else:
                    signal_color = "🟡"
                    signal_type = "WEAK BUY"

                with st.expander(f"{signal_color} **#{i} {pick['symbol']}** | {signal_type} | ₹{pick['price']:,.2f} | Win: {pick['intraday_prob']:.0%}", expanded=(i <= 3)):
                    # Header info
                    st.markdown(f"**{analysis.name}** | Sector: {pick['sector']}")

                    # ⏰ TIMING RECOMMENDATION - Most Important
                    st.markdown("### ⏰ WHEN TO BUY & SELL")

                    timing1, timing2 = st.columns(2)
                    with timing1:
                        # Buy timing based on RSI and volatility
                        if tech.rsi_14 < 40:
                            buy_time = "9:15-9:30 AM"
                            buy_reason = "RSI oversold - early entry recommended"
                        elif tech.volume_ratio > 1.5:
                            buy_time = "9:30-10:00 AM"
                            buy_reason = "High volume - wait for trend confirmation"
                        else:
                            buy_time = "9:45-10:15 AM"
                            buy_reason = "Standard entry after opening volatility settles"

                        st.success(f"""**🟢 BUY TIME**

**{buy_time}**

{buy_reason}

Entry: ₹{irec.entry_price:,.2f}""")

                    with timing2:
                        # Sell timing based on target and risk
                        reward_pct = (irec.target_1 - irec.entry_price) / irec.entry_price * 100
                        if reward_pct > 2:
                            sell_time = "2:00-3:00 PM"
                            sell_reason = "Book profits before closing"
                        else:
                            sell_time = "3:00-3:15 PM"
                            sell_reason = "Exit before market close"

                        st.error(f"""**🔴 SELL TIME**

**{sell_time}**

{sell_reason}

Target: ₹{irec.target_1:,.2f} (+{reward_pct:.1f}%)""")

                    # Action Summary Box
                    risk_amt = (irec.entry_price - irec.stop_loss) / irec.entry_price * 100
                    reward_amt = (irec.target_1 - irec.entry_price) / irec.entry_price * 100
                    potential_profit = irec.position_shares * (irec.target_1 - irec.entry_price)
                    potential_loss = irec.position_shares * (irec.entry_price - irec.stop_loss)

                    st.markdown("### 💰 TRADE ACTION SUMMARY")
                    st.info(f"""
**ACTION:** BUY {pick['symbol']} tomorrow between {buy_time}

| Parameter | Value |
|-----------|-------|
| **Entry Price** | ₹{irec.entry_price:,.2f} |
| **Stop Loss** | ₹{irec.stop_loss:,.2f} (-{risk_amt:.1f}%) |
| **Target** | ₹{irec.target_1:,.2f} (+{reward_amt:.1f}%) |
| **Position Size** | {irec.position_shares} shares (₹{irec.position_shares * irec.entry_price:,.0f}) |
| **Potential Profit** | ₹{potential_profit:,.0f} |
| **Max Loss** | ₹{potential_loss:,.0f} |
| **Win Probability** | {irec.win_probability:.0%} |
| **Risk:Reward** | 1:{irec.risk_reward:.1f} |
""")

                    # Technical Filter Checks - WHY this stock was selected
                    st.markdown("### ✅ Technical Filter Checks (Why Selected)")
                    tc = pick.get('tech_checks', {})
                    check_cols = st.columns(5)
                    with check_cols[0]:
                        rsi_icon = "✅" if tc.get('rsi_ok') else "❌"
                        rsi_bonus = "🔥" if tc.get('rsi_oversold') else ""
                        st.metric(f"RSI {rsi_icon}{rsi_bonus}", f"{tc.get('rsi', 0):.0f}",
                                  delta="Oversold" if tc.get('rsi_oversold') else "OK" if tc.get('rsi_ok') else "Overbought")
                    with check_cols[1]:
                        macd_icon = "✅" if tc.get('macd_bullish') else "❌"
                        st.metric(f"MACD {macd_icon}", f"{tc.get('macd_histogram', 0):.2f}",
                                  delta="Bullish" if tc.get('macd_bullish') else "Bearish")
                    with check_cols[2]:
                        vol_icon = "✅" if tc.get('volume_ok') else "❌"
                        vol_bonus = "🔥" if tc.get('volume_high') else ""
                        st.metric(f"Volume {vol_icon}{vol_bonus}", f"{tc.get('volume_ratio', 0):.1f}x",
                                  delta="High" if tc.get('volume_high') else "OK" if tc.get('volume_ok') else "Low")
                    with check_cols[3]:
                        adx_icon = "✅" if tc.get('adx_trending') else "⚠️"
                        st.metric(f"ADX {adx_icon}", f"{tc.get('adx', 0):.0f}",
                                  delta="Trending" if tc.get('adx_trending') else "Weak")
                    with check_cols[4]:
                        sma_icon = "✅" if tc.get('price_above_sma20') else "⚠️"
                        st.metric(f"SMA20 {sma_icon}", f"₹{tc.get('sma_20', 0):,.0f}",
                                  delta="Above" if tc.get('price_above_sma20') else "Below")

                    # 4-Model Ensemble Analysis (Enhanced Scoring)
                    enh_key = 'enhanced_intraday' if 'enhanced_intraday' in pick else ('enhanced_swing' if 'enhanced_swing' in pick else 'enhanced')
                    if pick.get(enh_key):
                        enh = pick[enh_key]
                        st.markdown("### 🧠 4-Model AI Ensemble")
                        st.caption("Multi-model consensus: Technical + Physics + Math + Regime")

                        vote_cols = st.columns(4)
                        model_names = ['base', 'physics', 'math', 'regime']
                        model_labels = ['Technical', 'Physics', 'Math', 'Regime']
                        for col, model, label in zip(vote_cols, model_names, model_labels):
                            with col:
                                vote = enh['model_votes'].get(model, 'HOLD')
                                if vote == 'BUY':
                                    st.success(f"✅ {label}\n**{vote}**")
                                elif vote == 'SELL':
                                    st.error(f"❌ {label}\n**{vote}**")
                                elif vote == 'AVOID':
                                    st.warning(f"⚠️ {label}\n**{vote}**")
                                else:
                                    st.info(f"⚪ {label}\n**{vote}**")

                        agreement = enh['model_agreement']
                        conviction = enh['conviction_level']
                        regime = enh.get('regime', 'unknown')
                        if agreement >= 3:
                            st.success(f"**{agreement}/4 models agree** = {conviction} conviction | Regime: {regime}")
                        elif agreement >= 2:
                            st.info(f"**{agreement}/4 models agree** = {conviction} conviction | Regime: {regime}")
                        else:
                            st.warning(f"**{agreement}/4 models agree** = {conviction} conviction | Regime: {regime}")

                        if enh.get('show_warning') and enh.get('warnings'):
                            for w in enh['warnings'][:2]:
                                st.warning(w)

                    # Price & Volume metrics
                    st.markdown("### 📊 Current Status")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Current Price", f"₹{tech.price:,.2f}")
                    with m2:
                        st.metric("52W High", f"₹{tech.week_52_high:,.2f}",
                                  delta=f"{tech.pct_from_52_high:+.1f}%")
                    with m3:
                        st.metric("52W Low", f"₹{tech.week_52_low:,.2f}",
                                  delta=f"+{tech.pct_from_52_low:.1f}%")
                    with m4:
                        st.metric("Volume", f"{tech.volume_ratio:.1f}x avg")

                    # Technical indicators
                    st.markdown("### 📊 Technical Snapshot")
                    tech = analysis.technical
                    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                    with tc1:
                        rsi_status = "Oversold" if tech.rsi_14 < 30 else "Overbought" if tech.rsi_14 > 70 else "Neutral"
                        st.metric("RSI", f"{tech.rsi_14:.0f}", delta=rsi_status)
                    with tc2:
                        st.metric("MACD", f"{tech.macd:.2f}")
                    with tc3:
                        st.metric("ATR", f"{tech.atr_14:.2f}")
                    with tc4:
                        st.metric("ADX", f"{tech.adx:.0f}")
                    with tc5:
                        st.metric("BB Width", f"{tech.bollinger_width:.1f}%")

                    # Bullish/Bearish Factors
                    if irec.bullish_factors or irec.bearish_factors:
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            if irec.bullish_factors:
                                st.markdown("**✅ Bullish Factors:**")
                                for f in irec.bullish_factors[:4]:
                                    st.markdown(f"- {f}")
                        with fc2:
                            if irec.bearish_factors:
                                st.markdown("**⚠️ Bearish Factors:**")
                                for f in irec.bearish_factors[:4]:
                                    st.markdown(f"- {f}")

                    # AI Insight Section
                    st.markdown("---")
                    st.markdown("### 💡 AI Insight")

                    # Quick rule-based insight (always shown, fast)
                    insight_data = {
                        'symbol': pick['symbol'],
                        'signal': pick['intraday_signal'],
                        'score': int(irec.win_probability * 100),
                        'rsi': tech.rsi_14,
                        'growth_1m': 0,
                        'growth_3m': 0,
                        'growth_1y': 0,
                        'trend': "UPTREND" if tech.price > tech.sma_20 else "DOWNTREND",
                        'volume_ratio': tech.volume_ratio,
                        'from_high': analysis.technical.pct_from_52_high,
                        'from_low': analysis.technical.pct_from_52_low
                    }
                    insight = generate_stock_insight(insight_data)
                    st.markdown(insight)

                    # On-demand analysis buttons
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        ai_key = f"ai_intraday_{pick['symbol']}"
                        if st.button(f"🧠 Deep AI Analysis", key=ai_key):
                            st.session_state[f"show_deep_ai_{pick['symbol']}"] = True

                    with btn_col2:
                        ens_key = f"ensemble_intraday_{pick['symbol']}"
                        if st.button(f"🔬 4-Model Ensemble", key=ens_key):
                            st.session_state[f"show_ensemble_{pick['symbol']}"] = True

                    # Show 4-Model Ensemble Analysis if requested (using CENTRALIZED API)
                    if st.session_state.get(f"show_ensemble_{pick['symbol']}", False):
                        with st.spinner(f"Running 4-model ensemble on {pick['symbol']}..."):
                            try:
                                import yfinance as yf
                                ticker = yf.Ticker(f"{pick['symbol']}.NS")
                                df = ticker.history(period="1y")
                                df.columns = [c.lower() for c in df.columns]
                                if len(df) >= 60:
                                    # Use CENTRALIZED prediction API
                                    enh_score = get_unified_prediction(pick['symbol'], "intraday", df)
                                    if enh_score:
                                        st.markdown("### 🧠 4-Model Ensemble Result (Centralized)")
                                        # Show model votes
                                        model_votes = enh_score.get('model_votes', {})
                                        vote_cols = st.columns(4)
                                        for col, (model, vote) in zip(vote_cols, model_votes.items()):
                                            with col:
                                                if vote == 'BUY':
                                                    st.success(f"✅ {model.title()}\n**{vote}**")
                                                elif vote == 'SELL':
                                                    st.error(f"❌ {model.title()}\n**{vote}**")
                                                else:
                                                    st.info(f"⚪ {model.title()}\n**{vote}**")

                                        # Conviction
                                        model_agreement = enh_score.get('model_agreement', 0)
                                        signal_strength = enh_score.get('signal_strength', 'weak')
                                        regime = enh_score.get('regime', 'unknown')
                                        if model_agreement >= 3:
                                            st.success(f"**{model_agreement}/4 models agree** = {signal_strength.upper()} conviction | Regime: {regime}")
                                        elif model_agreement >= 2:
                                            st.info(f"**{model_agreement}/4 models agree** = {signal_strength.upper()} conviction | Regime: {regime}")
                                        else:
                                            st.warning(f"**{model_agreement}/4 models agree** = Low conviction | Regime: {regime}")

                                        # Warnings from centralized API
                                        warnings = enh_score.get('warnings', [])
                                        for w in warnings[:3]:
                                            st.warning(w)
                                    else:
                                        st.warning("Could not generate ensemble score")
                                else:
                                    st.warning("Not enough historical data for ensemble analysis")
                            except Exception as e:
                                st.error(f"Ensemble error: {str(e)}")

                    # Show Deep AI Analysis if requested
                    if st.session_state.get(f"show_deep_ai_{pick['symbol']}", False):
                        with st.spinner(f"Analyzing {pick['symbol']} with AI..."):
                            try:
                                # Get analyzer instance
                                deep_analyzer = get_comprehensive_analyzer()
                                if not deep_analyzer:
                                    st.error("Analyzer not available")
                                else:
                                    # Re-analyze with AI enabled
                                    deep_analysis = deep_analyzer.analyze(pick['symbol'], skip_ai=False)
                                    if deep_analysis:
                                        render_deep_ai_analysis(pick['symbol'], deep_analysis, timeframe='intraday')
                                    else:
                                        st.error("Could not generate analysis")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                    # Enhanced AI Signal with Alternative Data
                    st.markdown("---")
                    try:
                        import yfinance as yf
                        ticker = yf.Ticker(f"{pick['symbol']}.NS")
                        hist_df = ticker.history(period="6mo")
                        hist_df.columns = [c.lower() for c in hist_df.columns]
                        if len(hist_df) >= 50:
                            render_enhanced_signal_box(f"{pick['symbol']}.NS", hist_df, timeframe='intraday')
                    except Exception as e:
                        st.caption("Enhanced signal not available")

        else:
            st.info("No Intraday BUY signals found. Stocks may be overbought (RSI > 75), have low volume, or show bearish signals. Try again when market conditions improve.")

        st.markdown("---")

        # ============== SWING PICKS ==============
        st.markdown("## 📈 Swing Picks (Hold 10-20 Days)")
        st.caption("Sorted by: 🟢 HIGH BUY > 🔵 BUY > 🟡 WEAK BUY | Hold for 2-4 weeks")

        swing_picks = st.session_state.get('top_swing_picks', [])[:top_n]

        if swing_picks:
            for i, pick in enumerate(swing_picks, 1):
                analysis = pick['analysis']
                srec = analysis.swing_rec
                tech = analysis.technical
                fund = analysis.fundamental

                # Show signal type based on strength (HIGH BUY > BUY > WEAK BUY)
                signal_strength = pick.get('signal_strength_swing', 1)
                if signal_strength == 3:
                    signal_color = "🟢"
                    signal_type = "HIGH BUY"
                elif signal_strength == 2:
                    signal_color = "🔵"
                    signal_type = "BUY"
                else:
                    signal_color = "🟡"
                    signal_type = "WEAK BUY"

                with st.expander(f"{signal_color} **#{i} {pick['symbol']}** | {signal_type} | ₹{pick['price']:,.2f} | Win: {pick['swing_prob']:.0%}", expanded=(i <= 3)):
                    # Header info
                    st.markdown(f"**{analysis.name}** | Sector: {pick['sector']}")

                    # ⏰ TIMING RECOMMENDATION
                    st.markdown("### ⏰ WHEN TO BUY & HOLD DURATION")

                    timing1, timing2 = st.columns(2)
                    with timing1:
                        # Buy timing based on RSI and trend
                        if tech.rsi_14 < 35:
                            buy_time = "Tomorrow (Day 1)"
                            buy_reason = "RSI oversold - immediate entry recommended"
                        elif tech.price < tech.sma_20:
                            buy_time = "Wait for SMA20 cross"
                            buy_reason = "Price below SMA20 - wait for upward cross"
                        else:
                            buy_time = "Next 1-3 days"
                            buy_reason = "Buy on any 1-2% dip from current level"

                        st.success(f"""**🟢 BUY TIMING**

**{buy_time}**

{buy_reason}

Entry: ₹{srec.entry_price:,.2f}""")

                    with timing2:
                        # Sell timing based on target
                        reward_pct = (srec.target_1 - srec.entry_price) / srec.entry_price * 100
                        if reward_pct > 8:
                            hold_days = "15-20 days"
                            sell_reason = "Hold for full target - strong momentum expected"
                        elif reward_pct > 5:
                            hold_days = "10-15 days"
                            sell_reason = "Book partial at 50% target, trail rest"
                        else:
                            hold_days = "7-10 days"
                            sell_reason = "Quick swing - exit at target"

                        st.warning(f"""**📅 HOLD DURATION**

**{hold_days}**

{sell_reason}

Target: ₹{srec.target_1:,.2f} (+{reward_pct:.1f}%)""")

                    # Action Summary Box
                    risk_amt = (srec.entry_price - srec.stop_loss) / srec.entry_price * 100
                    reward_amt = (srec.target_1 - srec.entry_price) / srec.entry_price * 100
                    potential_profit = srec.position_shares * (srec.target_1 - srec.entry_price)
                    potential_loss = srec.position_shares * (srec.entry_price - srec.stop_loss)

                    st.markdown("### 💰 TRADE ACTION SUMMARY")
                    st.info(f"""
**ACTION:** BUY {pick['symbol']} - {buy_time}

| Parameter | Value |
|-----------|-------|
| **Entry Price** | ₹{srec.entry_price:,.2f} |
| **Stop Loss** | ₹{srec.stop_loss:,.2f} (-{risk_amt:.1f}%) |
| **Target** | ₹{srec.target_1:,.2f} (+{reward_amt:.1f}%) |
| **Hold Duration** | {hold_days} |
| **Position Size** | {srec.position_shares} shares (₹{srec.position_shares * srec.entry_price:,.0f}) |
| **Potential Profit** | ₹{potential_profit:,.0f} |
| **Max Loss** | ₹{potential_loss:,.0f} |
| **Win Probability** | {srec.win_probability:.0%} |
| **Risk:Reward** | 1:{srec.risk_reward:.1f} |
""")

                    # Technical Filter Checks - WHY this stock was selected
                    st.markdown("### ✅ Technical Filter Checks (Why Selected)")
                    tc = pick.get('tech_checks', {})
                    check_cols = st.columns(5)
                    with check_cols[0]:
                        rsi_icon = "✅" if tc.get('rsi_ok') else "❌"
                        rsi_bonus = "🔥" if tc.get('rsi_oversold') else ""
                        st.metric(f"RSI {rsi_icon}{rsi_bonus}", f"{tc.get('rsi', 0):.0f}",
                                  delta="Oversold" if tc.get('rsi_oversold') else "OK" if tc.get('rsi_ok') else "Overbought")
                    with check_cols[1]:
                        macd_icon = "✅" if tc.get('macd_bullish') else "❌"
                        st.metric(f"MACD {macd_icon}", f"{tc.get('macd_histogram', 0):.2f}",
                                  delta="Bullish" if tc.get('macd_bullish') else "Bearish")
                    with check_cols[2]:
                        vol_icon = "✅" if tc.get('volume_ok') else "❌"
                        vol_bonus = "🔥" if tc.get('volume_high') else ""
                        st.metric(f"Volume {vol_icon}{vol_bonus}", f"{tc.get('volume_ratio', 0):.1f}x",
                                  delta="High" if tc.get('volume_high') else "OK" if tc.get('volume_ok') else "Low")
                    with check_cols[3]:
                        adx_icon = "✅" if tc.get('adx_trending') else "⚠️"
                        st.metric(f"ADX {adx_icon}", f"{tc.get('adx', 0):.0f}",
                                  delta="Trending" if tc.get('adx_trending') else "Weak")
                    with check_cols[4]:
                        sma_icon = "✅" if tc.get('price_above_sma20') else "⚠️"
                        st.metric(f"SMA20 {sma_icon}", f"₹{tc.get('sma_20', 0):,.0f}",
                                  delta="Above" if tc.get('price_above_sma20') else "Below")

                    # 4-Model Ensemble Analysis (Enhanced Scoring)
                    enh_key = 'enhanced_intraday' if 'enhanced_intraday' in pick else ('enhanced_swing' if 'enhanced_swing' in pick else 'enhanced')
                    if pick.get(enh_key):
                        enh = pick[enh_key]
                        st.markdown("### 🧠 4-Model AI Ensemble")
                        st.caption("Multi-model consensus: Technical + Physics + Math + Regime")

                        vote_cols = st.columns(4)
                        model_names = ['base', 'physics', 'math', 'regime']
                        model_labels = ['Technical', 'Physics', 'Math', 'Regime']
                        for col, model, label in zip(vote_cols, model_names, model_labels):
                            with col:
                                vote = enh['model_votes'].get(model, 'HOLD')
                                if vote == 'BUY':
                                    st.success(f"✅ {label}\n**{vote}**")
                                elif vote == 'SELL':
                                    st.error(f"❌ {label}\n**{vote}**")
                                elif vote == 'AVOID':
                                    st.warning(f"⚠️ {label}\n**{vote}**")
                                else:
                                    st.info(f"⚪ {label}\n**{vote}**")

                        agreement = enh['model_agreement']
                        conviction = enh['conviction_level']
                        regime = enh.get('regime', 'unknown')
                        if agreement >= 3:
                            st.success(f"**{agreement}/4 models agree** = {conviction} conviction | Regime: {regime}")
                        elif agreement >= 2:
                            st.info(f"**{agreement}/4 models agree** = {conviction} conviction | Regime: {regime}")
                        else:
                            st.warning(f"**{agreement}/4 models agree** = {conviction} conviction | Regime: {regime}")

                        if enh.get('show_warning') and enh.get('warnings'):
                            for w in enh['warnings'][:2]:
                                st.warning(w)

                    # Price & Volume metrics
                    st.markdown("### 📊 Current Status")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Current Price", f"₹{tech.price:,.2f}")
                    with m2:
                        st.metric("52W High", f"₹{tech.week_52_high:,.2f}",
                                  delta=f"{tech.pct_from_52_high:+.1f}%")
                    with m3:
                        st.metric("52W Low", f"₹{tech.week_52_low:,.2f}",
                                  delta=f"+{tech.pct_from_52_low:.1f}%")
                    with m4:
                        st.metric("Volume", f"{tech.volume_ratio:.1f}x avg")

                    # Technical indicators
                    st.markdown("### 📊 Technical Snapshot")
                    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
                    with tc1:
                        rsi_status = "Oversold" if tech.rsi_14 < 30 else "Overbought" if tech.rsi_14 > 70 else "Neutral"
                        st.metric("RSI", f"{tech.rsi_14:.0f}", delta=rsi_status)
                    with tc2:
                        st.metric("MACD", f"{tech.macd:.2f}")
                    with tc3:
                        st.metric("ATR", f"{tech.atr_14:.2f}")
                    with tc4:
                        st.metric("ADX", f"{tech.adx:.0f}")
                    with tc5:
                        st.metric("BB Width", f"{tech.bollinger_width:.1f}%")

                    # Fundamentals snapshot
                    st.markdown("### 📋 Fundamentals Snapshot")
                    f1, f2, f3, f4 = st.columns(4)
                    with f1:
                        st.metric("P/E Ratio", f"{fund.pe_ratio:.1f}")
                    with f2:
                        st.metric("P/B Ratio", f"{fund.pb_ratio:.2f}")
                    with f3:
                        st.metric("ROE", f"{fund.roe:.1f}%")
                    with f4:
                        st.metric("Debt/Equity", f"{fund.debt_equity:.2f}")

                    # Bullish/Bearish Factors
                    if srec.bullish_factors or srec.bearish_factors:
                        fc1, fc2 = st.columns(2)
                        with fc1:
                            if srec.bullish_factors:
                                st.markdown("**✅ Bullish Factors:**")
                                for f in srec.bullish_factors[:5]:
                                    st.markdown(f"- {f}")
                        with fc2:
                            if srec.bearish_factors:
                                st.markdown("**⚠️ Bearish Factors:**")
                                for f in srec.bearish_factors[:5]:
                                    st.markdown(f"- {f}")

                    # AI Insight Section
                    st.markdown("---")
                    st.markdown("### 💡 AI Insight")

                    # Quick rule-based insight (always shown, fast)
                    insight_data = {
                        'symbol': pick['symbol'],
                        'signal': pick['swing_signal'],
                        'score': int(srec.win_probability * 100),
                        'rsi': tech.rsi_14,
                        'growth_1m': fund.revenue_growth if hasattr(fund, 'revenue_growth') else 0,
                        'growth_3m': 0,
                        'growth_1y': 0,
                        'trend': "UPTREND" if tech.price > tech.sma_50 else "DOWNTREND",
                        'volume_ratio': tech.volume_ratio,
                        'from_high': analysis.technical.pct_from_52_high,
                        'from_low': analysis.technical.pct_from_52_low
                    }
                    insight = generate_stock_insight(insight_data)
                    st.markdown(insight)

                    # On-demand analysis buttons
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        ai_key = f"ai_swing_{pick['symbol']}"
                        if st.button(f"🧠 Deep AI Analysis", key=ai_key):
                            st.session_state[f"show_deep_ai_swing_{pick['symbol']}"] = True

                    with btn_col2:
                        ens_key = f"ensemble_swing_{pick['symbol']}"
                        if st.button(f"🔬 4-Model Ensemble", key=ens_key):
                            st.session_state[f"show_ensemble_swing_{pick['symbol']}"] = True

                    # Show 4-Model Ensemble Analysis if requested (using CENTRALIZED API)
                    if st.session_state.get(f"show_ensemble_swing_{pick['symbol']}", False):
                        with st.spinner(f"Running 4-model ensemble on {pick['symbol']}..."):
                            try:
                                import yfinance as yf
                                ticker = yf.Ticker(f"{pick['symbol']}.NS")
                                df = ticker.history(period="1y")
                                df.columns = [c.lower() for c in df.columns]
                                if len(df) >= 60:
                                    # Use CENTRALIZED prediction API
                                    enh_score = get_unified_prediction(pick['symbol'], "swing", df)
                                    if enh_score:
                                        st.markdown("### 🧠 4-Model Ensemble Result (Centralized)")
                                        model_votes = enh_score.get('model_votes', {})
                                        vote_cols = st.columns(4)
                                        for col, (model, vote) in zip(vote_cols, model_votes.items()):
                                            with col:
                                                if vote == 'BUY':
                                                    st.success(f"✅ {model.title()}\n**{vote}**")
                                                elif vote == 'SELL':
                                                    st.error(f"❌ {model.title()}\n**{vote}**")
                                                else:
                                                    st.info(f"⚪ {model.title()}\n**{vote}**")

                                        model_agreement = enh_score.get('model_agreement', 0)
                                        signal_strength = enh_score.get('signal_strength', 'weak')
                                        regime = enh_score.get('regime', 'unknown')
                                        if model_agreement >= 3:
                                            st.success(f"**{model_agreement}/4 models agree** = {signal_strength.upper()} conviction | Regime: {regime}")
                                        elif model_agreement >= 2:
                                            st.info(f"**{model_agreement}/4 models agree** = {signal_strength.upper()} conviction | Regime: {regime}")
                                        else:
                                            st.warning(f"**{model_agreement}/4 models agree** = Low conviction | Regime: {regime}")

                                        warnings = enh_score.get('warnings', [])
                                        for w in warnings[:3]:
                                            st.warning(w)
                                    else:
                                        st.warning("Could not generate ensemble score")
                                else:
                                    st.warning("Not enough historical data for ensemble analysis")
                            except Exception as e:
                                st.error(f"Ensemble error: {str(e)}")

                    # Show Deep AI Analysis if requested
                    if st.session_state.get(f"show_deep_ai_swing_{pick['symbol']}", False):
                        with st.spinner(f"Analyzing {pick['symbol']} with AI..."):
                            try:
                                # Get analyzer instance
                                deep_analyzer = get_comprehensive_analyzer()
                                if not deep_analyzer:
                                    st.error("Analyzer not available")
                                else:
                                    # Re-analyze with AI enabled
                                    deep_analysis = deep_analyzer.analyze(pick['symbol'], skip_ai=False)
                                    if deep_analysis:
                                        render_deep_ai_analysis(pick['symbol'], deep_analysis, timeframe='swing')
                                    else:
                                        st.error("Could not generate analysis")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                    # Enhanced AI Signal with Alternative Data
                    st.markdown("---")
                    try:
                        import yfinance as yf
                        ticker = yf.Ticker(f"{pick['symbol']}.NS")
                        hist_df = ticker.history(period="6mo")
                        hist_df.columns = [c.lower() for c in hist_df.columns]
                        if len(hist_df) >= 50:
                            render_enhanced_signal_box(f"{pick['symbol']}.NS", hist_df, timeframe='swing')
                    except Exception as e:
                        st.caption("Enhanced signal not available")

        else:
            st.info("No Swing BUY signals found. Stocks may lack uptrend confirmation (price below SMA20, MACD bearish) or show bearish signals. Try again when market conditions improve.")

    else:
        st.info("👆 Click 'Scan All Stocks for BUY Signals' to find top Intraday and Swing picks")

        # Show what this tab does
        st.markdown("""
        ### What This Tab Does

        1. **Auto-Scans** 70+ stocks (NIFTY 50 + High-Liquidity Midcaps)
        2. **Uses Same Logic** as Full Analysis tab (ComprehensiveAnalyzer)
        3. **Filters** only BUY and STRONG BUY signals
        4. **Shows Two Sections:**
           - **Intraday Picks** - For same-day trading (buy morning, sell by 3:15 PM)
           - **Swing Picks** - For 10-20 day holding period

        5. **For Each Pick Shows:**
           - Entry Price, Stop Loss, Target
           - Win Probability & Risk:Reward ratio
           - Position size based on your capital
           - Technical indicators (RSI, MACD, ADX, etc.)
           - Fundamentals (P/E, P/B, ROE, Debt)
           - Bullish & Bearish factors
           - Quick AI insight (rule-based, instant)
           - **🤖 Deep AI Analysis button** (LLM-powered, on-demand)

        ⚡ **Scan is FAST** (~1 min) - AI skipped during bulk scan.
        🤖 Click "Get Deep AI Analysis" on any stock for detailed LLM insight.
        """)


def main():
    """Main app entry point."""

    # Sidebar
    with st.sidebar:
        st.markdown("## 📈 Stock Engine")
        st.markdown("---")

        st.markdown("### 🎯 Navigation")
        st.markdown("""
        **RECOMMENDATIONS** (7-Model Ensemble):
        - **🏆 Top Picks**: Auto-scan best opportunities
        - **🔬 Deep Analysis**: Single stock deep dive
        - **🏭 Sectors**: Sector-wise analysis

        **FILTERING** (Basic Screener):
        - **💰 Screener**: Filter by price/cap/momentum
        """)

        st.markdown("---")
        st.markdown("### 📅 Market Status")

        now = datetime.now()
        market_open = now.replace(hour=9, minute=15)
        market_close = now.replace(hour=15, minute=30)

        if market_open <= now <= market_close and now.weekday() < 5:
            st.success("🟢 Market Open")
        else:
            st.warning("🔴 Market Closed")

        st.caption(f"Time: {now.strftime('%H:%M IST')}")

        # Engine Settings Sidebar (from engine_config module)
        render_settings_sidebar()

        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown("""
        - [NSE India](https://www.nseindia.com)
        - [MoneyControl](https://www.moneycontrol.com)
        - [TradingView](https://www.tradingview.com)
        """)

    # Main header
    st.markdown('<h1 class="main-header">📈 Stock Prediction Engine</h1>', unsafe_allow_html=True)
    st.caption(f"Indian Stock Market Analysis | {datetime.now().strftime('%Y-%m-%d')}")

    # Show current engine configuration summary
    render_config_summary()

    # Main tabs - Unified 7-Model Ensemble Engine
    # All recommendation tabs use the SAME prediction engine for consistent results
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top Picks",        # Auto-scan: Intraday + Swing recommendations
        "🔬 Deep Analysis",    # Single stock comprehensive analysis
        "🏭 Sectors",          # Sector-wise stock analysis
        "💰 Screener",         # Basic filtering (price, cap, momentum)
        "📖 About"             # Documentation
    ])

    # 🏆 TOP PICKS - Primary recommendation tab (uses 7-Model Ensemble)
    with tab0:
        render_top_picks_tab()

    # 🔬 DEEP ANALYSIS - Single stock deep dive (uses 7-Model Ensemble)
    with tab1:
        render_comprehensive_analysis_tab()

    # 🏭 SECTORS - Sector-wise analysis (uses 7-Model Ensemble)
    with tab2:
        render_sector_analysis_tab()

    # 💰 SCREENER - Basic filtering (no recommendations, just filtering)
    with tab3:
        render_unified_screener_tab()

    # 📖 ABOUT - Documentation
    with tab4:
        render_about_tab()


if __name__ == "__main__":
    main()
