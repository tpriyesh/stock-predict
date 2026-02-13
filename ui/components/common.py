"""
Common UI Components and Cached Resources

This module contains all shared components, cached resources, and utility
functions used across the Streamlit dashboard tabs.

Split from the main app.py for maintainability.
"""

import streamlit as st
import pandas as pd
from datetime import datetime


# ============================================================================
# CUSTOM CSS
# ============================================================================

def inject_custom_css():
    """Inject custom CSS styles for the dashboard."""
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
        .stMetric {
            min-width: 120px !important;
        }
        .stChart {
            overflow: visible !important;
        }
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


# ============================================================================
# CACHED RESOURCES
# ============================================================================

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
        from src.engines.advanced_predictor import DomainPredictionEngine
        return DomainPredictionEngine(use_alternative_data=True)
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
# PREDICTION HELPER FUNCTIONS
# ============================================================================

def get_unified_prediction(symbol: str, trade_type: str = "intraday", df=None):
    """
    Get prediction using the CENTRALIZED unified prediction engine.

    This function should be used by all UI components instead of
    directly calling scoring engines.
    """
    api = get_ui_prediction_api()
    if api is None:
        # Fallback to enhanced scoring engine
        engine = get_enhanced_scoring_engine()
        if engine is None or df is None:
            return None

        try:
            from src.storage.models import TradeType
            from src.features.technical import TechnicalIndicators

            if 'rsi' not in df.columns:
                df = TechnicalIndicators.calculate_all(df)

            trade_type_enum = TradeType.INTRADAY if trade_type == "intraday" else TradeType.SWING
            score = engine.score_stock(symbol, df, trade_type_enum)

            if score is None:
                return None

            return {
                'symbol': symbol,
                'signal': score.signal.value,
                'confidence': score.confidence,
                'model_agreement': score.model_agreement,
                'model_votes': score.model_votes,
                'ensemble_score': score.ensemble_score,
                'signal_strength': score.signal_strength,
                'regime': score.regime,
                'regime_stability': score.regime_stability,
                'predictability': score.market_predictability,
                'warnings': score.warnings,
                'entry_price': score.entry_price,
                'stop_loss': score.stop_loss,
                'target_price': score.target_price,
                'risk_reward': score.risk_reward,
                'current_price': score.current_price,
                'base_score': score.base_score,
                'physics_score': score.physics_score,
                'math_score': score.math_score,
                'regime_score': score.regime_score,
                'reasons': score.reasons,
                'conviction_level': 'HIGH' if score.model_agreement >= 3 else ('MODERATE' if score.model_agreement >= 2 else 'LOW'),
                'show_warning': score.regime == 'choppy' or score.regime_stability < 0.5
            }
        except Exception as e:
            return None

    # Use centralized API
    prediction = api.get_stock_prediction(symbol, trade_type, df)
    if prediction is None:
        return None

    return {
        'symbol': prediction.symbol,
        'signal': prediction.signal,
        'confidence': prediction.confidence,
        'confidence_grade': prediction.confidence_grade,
        'model_agreement': prediction.model_agreement,
        'model_votes': prediction.model_votes,
        'ensemble_score': prediction.ensemble_score,
        'signal_strength': prediction.signal_strength,
        'regime': prediction.regime,
        'regime_stability': prediction.regime_stability,
        'predictability': prediction.market_predictability,
        'warnings': prediction.warnings,
        'entry_price': prediction.entry_price,
        'stop_loss': prediction.stop_loss,
        'target_price': prediction.target_price,
        'risk_reward': prediction.risk_reward,
        'current_price': prediction.current_price,
        'base_score': prediction.base_score,
        'physics_score': prediction.physics_score,
        'math_score': prediction.math_score,
        'regime_score': prediction.regime_score,
        'reasons': prediction.reasons,
        'conviction_level': 'HIGH' if prediction.model_agreement >= 3 else ('MODERATE' if prediction.model_agreement >= 2 else 'LOW'),
        'show_warning': prediction.regime == 'choppy' or prediction.regime_stability < 0.5
    }


def get_enhanced_prediction_data(symbol: str, df, trade_type: str = "intraday"):
    """
    Get enhanced prediction data for a stock using centralized engine.
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

    # Normalize signal
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

    # Add recommendation
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

    stocks_summary = "\n".join([
        f"- {s['symbol']}: Rs{s['price']:.2f}, RSI {s['rsi']:.0f}, 1M {s['growth_1m']:+.1f}%, Signal: {s['signal']}"
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


# ============================================================================
# RENDER COMPONENTS
# ============================================================================

def render_deep_ai_analysis(symbol: str, analysis, timeframe: str = 'intraday'):
    """Render comprehensive Deep AI Analysis with all values explained."""
    st.markdown("### Deep AI Analysis")

    # Price Performance
    with st.expander("Price Performance", expanded=True):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Today", f"{analysis.technical.change_1d:+.2f}%")
        with p2:
            st.metric("1 Week", f"{analysis.technical.change_1w:+.2f}%")
        with p3:
            st.metric("1 Month", f"{analysis.technical.change_1m:+.2f}%")
        with p4:
            st.metric("1 Year", f"{analysis.technical.change_1y:+.2f}%")

    # Technical Indicators
    with st.expander("Technical Indicators", expanded=True):
        t1, t2, t3, t4 = st.columns(4)
        rsi = analysis.technical.rsi_14

        with t1:
            st.metric("RSI (14)", f"{rsi:.0f}")
            if rsi < 30:
                st.caption("OVERSOLD - Good buy zone")
            elif rsi > 70:
                st.caption("OVERBOUGHT - Caution")
            else:
                st.caption("Neutral")

        with t2:
            st.metric("MACD", f"{analysis.technical.macd:.2f}")

        with t3:
            vol = analysis.technical.volume_ratio
            st.metric("Volume", f"{vol:.1f}x avg")

        with t4:
            above_sma20 = analysis.technical.price > analysis.technical.sma_20
            above_sma50 = analysis.technical.price > analysis.technical.sma_50
            if above_sma20 and above_sma50:
                trend = "UPTREND"
            elif not above_sma20 and not above_sma50:
                trend = "DOWNTREND"
            else:
                trend = "MIXED"
            st.metric("Trend", trend)

    # Recommendation
    st.markdown("#### Recommendation")
    rec = analysis.intraday_rec if timeframe == 'intraday' else analysis.swing_rec
    signal = rec.signal.value

    if "BUY" in signal:
        st.success(f"**{signal}** - Win Probability: {rec.win_probability:.0%}")
    elif "SELL" in signal:
        st.error(f"**{signal}** - Win Probability: {rec.win_probability:.0%}")
    else:
        st.info(f"**{signal}** - Win Probability: {rec.win_probability:.0%}")

    # Trade setup
    setup1, setup2, setup3 = st.columns(3)
    with setup1:
        st.metric("Entry", f"Rs{rec.entry_price:,.2f}")
    with setup2:
        st.metric("Stop Loss", f"Rs{rec.stop_loss:,.2f}")
    with setup3:
        st.metric("Target", f"Rs{rec.target_1:,.2f}")


def render_enhanced_signal_box(symbol: str, df, timeframe: str = 'both'):
    """Render enhanced signal box with model agreement."""
    enh_data = get_enhanced_prediction_data(symbol.replace('.NS', ''), df, timeframe)
    if enh_data is None:
        st.info("Enhanced signal not available")
        return

    model_votes = enh_data.get('model_votes', {})
    model_agreement = enh_data.get('model_agreement', 0)
    signal_strength = enh_data.get('signal_strength', 'weak')

    st.markdown("#### 4-Model Ensemble")

    cols = st.columns(4)
    for col, (model, vote) in zip(cols, model_votes.items()):
        with col:
            if vote == 'BUY':
                st.success(f"{model.title()}: **{vote}**")
            elif vote == 'SELL':
                st.error(f"{model.title()}: **{vote}**")
            else:
                st.info(f"{model.title()}: **{vote}**")

    if model_agreement >= 3:
        st.success(f"**{model_agreement}/4 models agree** - {signal_strength.upper()} conviction")
    elif model_agreement >= 2:
        st.info(f"**{model_agreement}/4 models agree** - {signal_strength.upper()} conviction")
    else:
        st.warning(f"**{model_agreement}/4 models agree** - Low conviction")


def render_stock_table(stocks, show_ai_button=True):
    """Render a table of stocks with key metrics."""
    if not stocks:
        st.info("No stocks to display")
        return

    df_data = []
    for stock in stocks:
        df_data.append({
            'Symbol': stock.get('symbol', ''),
            'Price': f"Rs{stock.get('price', 0):.2f}",
            'Change': f"{stock.get('change', 0):+.2f}%",
            'RSI': f"{stock.get('rsi', 0):.0f}",
            'Signal': stock.get('signal', 'HOLD'),
            'Score': stock.get('score', 0)
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)


def render_stock_card(stock):
    """Render a stock card with key information."""
    symbol = stock.get('symbol', '')
    price = stock.get('price', 0)
    change = stock.get('change', 0)
    signal = stock.get('signal', 'HOLD')

    card_class = "stock-card" if change >= 0 else "stock-card stock-card-negative"

    st.markdown(f"""
    <div class="{card_class}">
        <h4>{symbol}</h4>
        <p>Price: Rs{price:.2f} ({change:+.2f}%)</p>
        <p>Signal: <strong>{signal}</strong></p>
    </div>
    """, unsafe_allow_html=True)
