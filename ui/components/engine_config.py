"""
Unified Engine Configuration for Stock Prediction UI

This module provides:
1. Centralized configuration for the 7-model ensemble
2. Session state management for user settings
3. UI components for settings sidebar
4. Single source of truth for prediction engine access

ALL TABS SHOULD USE THIS MODULE for consistent predictions.
"""

import streamlit as st
from dataclasses import dataclass
from typing import Optional, Dict, Any
from functools import lru_cache
import pandas as pd
from loguru import logger


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class EngineConfig:
    """Configuration for the prediction engine."""
    # Model settings
    use_7_model: bool = True           # Use 7-model ensemble (vs 4-model fallback)
    min_model_agreement: int = 3       # Minimum models that must agree (1-7)

    # Trading settings
    capital: float = 100000.0          # Trading capital in INR
    risk_per_trade: float = 0.02       # Risk per trade (2% default)

    # Filter thresholds
    min_confidence: float = 0.55       # Minimum confidence to show (55%)
    min_liquidity_cr: float = 5.0      # Minimum liquidity in Crores
    max_atr_pct: float = 6.0           # Maximum ATR% (volatility filter)

    # Signal thresholds
    strong_buy_agreement: int = 5      # Models needed for strong buy (5/7)
    moderate_buy_agreement: int = 3    # Models needed for moderate buy (3/7)

    # Display settings
    top_n_picks: int = 10              # Number of top picks to show
    show_warnings: bool = True         # Show risk warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'use_7_model': self.use_7_model,
            'min_model_agreement': self.min_model_agreement,
            'capital': self.capital,
            'risk_per_trade': self.risk_per_trade,
            'min_confidence': self.min_confidence,
            'min_liquidity_cr': self.min_liquidity_cr,
            'max_atr_pct': self.max_atr_pct,
            'strong_buy_agreement': self.strong_buy_agreement,
            'moderate_buy_agreement': self.moderate_buy_agreement,
            'top_n_picks': self.top_n_picks,
            'show_warnings': self.show_warnings
        }


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESET_CONFIGS = {
    'conservative': EngineConfig(
        use_7_model=True,
        min_model_agreement=5,          # Require 5/7 agreement
        min_confidence=0.60,
        risk_per_trade=0.01,            # 1% risk
        min_liquidity_cr=10.0,          # Higher liquidity requirement
        max_atr_pct=4.0,                # Lower volatility tolerance
        strong_buy_agreement=6,
        moderate_buy_agreement=4,
    ),
    'balanced': EngineConfig(
        use_7_model=True,
        min_model_agreement=3,          # Default 3/7 agreement
        min_confidence=0.55,
        risk_per_trade=0.02,            # 2% risk
        min_liquidity_cr=5.0,
        max_atr_pct=6.0,
        strong_buy_agreement=5,
        moderate_buy_agreement=3,
    ),
    'aggressive': EngineConfig(
        use_7_model=True,
        min_model_agreement=2,          # Lower agreement threshold
        min_confidence=0.50,
        risk_per_trade=0.03,            # 3% risk
        min_liquidity_cr=3.0,           # Accept lower liquidity
        max_atr_pct=8.0,                # Accept higher volatility
        strong_buy_agreement=4,
        moderate_buy_agreement=2,
    ),
    'basic_4_model': EngineConfig(
        use_7_model=False,              # Use 4-model fallback
        min_model_agreement=2,
        min_confidence=0.55,
        risk_per_trade=0.02,
        min_liquidity_cr=5.0,
        max_atr_pct=6.0,
        strong_buy_agreement=3,
        moderate_buy_agreement=2,
    ),
}

# Recommended configuration
RECOMMENDED_CONFIG = 'balanced'


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def get_engine_config() -> EngineConfig:
    """Get current engine configuration from session state."""
    if 'engine_config' not in st.session_state:
        st.session_state.engine_config = PRESET_CONFIGS[RECOMMENDED_CONFIG]
    return st.session_state.engine_config


def set_engine_config(config: EngineConfig):
    """Set engine configuration in session state."""
    st.session_state.engine_config = config
    # Clear cached engine to force reload with new config
    if 'scoring_engine' in st.session_state:
        del st.session_state.scoring_engine


def apply_preset(preset_name: str):
    """Apply a preset configuration."""
    if preset_name in PRESET_CONFIGS:
        set_engine_config(PRESET_CONFIGS[preset_name])


# ============================================================================
# CACHED ENGINE ACCESS
# ============================================================================

@st.cache_resource(show_spinner=False)
def _create_scoring_engine(use_7_model: bool):
    """Create and cache the scoring engine."""
    try:
        from src.models.enhanced_scoring import EnhancedScoringEngine
        engine = EnhancedScoringEngine(use_7_model_ensemble=use_7_model)
        logger.info(f"Created scoring engine (7-model: {use_7_model})")
        return engine
    except Exception as e:
        logger.error(f"Failed to create scoring engine: {e}")
        return None


def get_scoring_engine():
    """Get the scoring engine with current configuration."""
    config = get_engine_config()
    return _create_scoring_engine(config.use_7_model)


@st.cache_resource(show_spinner=False)
def _create_signal_generator():
    """Create and cache the signal generator."""
    try:
        from src.signals.enhanced_generator import EnhancedSignalGenerator
        generator = EnhancedSignalGenerator()
        logger.info("Created enhanced signal generator")
        return generator
    except Exception as e:
        logger.error(f"Failed to create signal generator: {e}")
        return None


def get_signal_generator():
    """Get the enhanced signal generator."""
    return _create_signal_generator()


# ============================================================================
# UNIFIED PREDICTION FUNCTION
# ============================================================================

def get_unified_prediction(
    symbol: str,
    df: pd.DataFrame,
    trade_type: str = 'INTRADAY',
    news_score: float = 0.5
) -> Optional[Dict[str, Any]]:
    """
    SINGLE SOURCE OF TRUTH for predictions.

    All tabs should use this function for consistent results.

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        df: DataFrame with OHLCV data
        trade_type: 'INTRADAY' or 'SWING'
        news_score: News sentiment score (0-1)

    Returns:
        Dictionary with prediction results or None if failed
    """
    config = get_engine_config()
    engine = get_scoring_engine()

    if engine is None:
        logger.error("Scoring engine not available")
        return None

    if df is None or df.empty or len(df) < 60:
        logger.warning(f"{symbol}: Insufficient data for prediction")
        return None

    try:
        from src.storage.models import TradeType

        tt = TradeType.INTRADAY if trade_type.upper() == 'INTRADAY' else TradeType.SWING

        score = engine.score_stock(
            symbol=symbol,
            df=df,
            trade_type=tt,
            news_score=news_score,
            news_reasons=[]
        )

        if score is None:
            return None

        # Apply config filters
        if score.confidence < config.min_confidence:
            return None
        if score.liquidity_cr < config.min_liquidity_cr:
            return None
        if score.atr_pct > config.max_atr_pct:
            return None

        return {
            'symbol': score.symbol,
            'signal': score.signal.value,
            'confidence': score.confidence,
            'signal_strength': score.signal_strength,
            'model_agreement': score.model_agreement,
            'total_models': score.total_models,
            'model_votes': score.model_votes,
            'entry_price': score.entry_price,
            'stop_loss': score.stop_loss,
            'target_price': score.target_price,
            'risk_reward': score.risk_reward,
            'current_price': score.current_price,
            'atr_pct': score.atr_pct,
            'liquidity_cr': score.liquidity_cr,
            'regime': score.regime,
            'regime_stability': score.regime_stability,
            'sector': score.sector,
            'reasons': score.reasons,
            'warnings': score.warnings if config.show_warnings else [],
            'recommended_strategy': score.recommended_strategy,
            'market_predictability': score.market_predictability,
            # Individual model scores
            'base_score': score.base_score,
            'physics_score': score.physics_score,
            'math_score': score.math_score,
            'regime_score': score.regime_score,
            'macro_score': score.macro_score,
            'alternative_score': score.alternative_score,
            'advanced_score': score.advanced_score,
            'ensemble_score': score.ensemble_score,
        }

    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {e}")
        return None


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_settings_sidebar():
    """
    Render the settings sidebar component.

    Call this from the main app.py in the sidebar section.
    """
    config = get_engine_config()

    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Engine Settings")

        # Preset selector
        preset_names = list(PRESET_CONFIGS.keys())
        current_preset = 'custom'
        for name, preset in PRESET_CONFIGS.items():
            if config.to_dict() == preset.to_dict():
                current_preset = name
                break

        selected_preset = st.selectbox(
            "Configuration Preset",
            options=['custom'] + preset_names,
            index=0 if current_preset == 'custom' else preset_names.index(current_preset) + 1,
            help="Choose a preset or customize below"
        )

        if selected_preset != 'custom' and selected_preset != current_preset:
            apply_preset(selected_preset)
            st.rerun()

        # Show recommended badge
        if selected_preset == RECOMMENDED_CONFIG:
            st.success(f"✓ {RECOMMENDED_CONFIG.title()} (Recommended)")

        # Model settings
        st.markdown("##### Model Settings")

        use_7_model = st.toggle(
            "Use 7-Model Ensemble",
            value=config.use_7_model,
            help="Enable all 7 models including Macro, Alternative Data, and Advanced Math"
        )

        max_agreement = 7 if use_7_model else 4
        min_agreement = st.slider(
            "Min Model Agreement",
            min_value=1,
            max_value=max_agreement,
            value=min(config.min_model_agreement, max_agreement),
            help=f"Minimum models that must agree for a signal (out of {max_agreement})"
        )

        # Filter settings
        st.markdown("##### Filter Settings")

        min_confidence = st.slider(
            "Min Confidence %",
            min_value=40,
            max_value=80,
            value=int(config.min_confidence * 100),
            help="Only show stocks above this confidence level"
        ) / 100

        min_liquidity = st.slider(
            "Min Liquidity (Cr)",
            min_value=1.0,
            max_value=50.0,
            value=config.min_liquidity_cr,
            step=1.0,
            help="Minimum daily trading volume in Crores"
        )

        max_atr = st.slider(
            "Max Volatility (ATR%)",
            min_value=2.0,
            max_value=10.0,
            value=config.max_atr_pct,
            step=0.5,
            help="Maximum allowed volatility"
        )

        # Trading settings
        st.markdown("##### Trading Settings")

        capital = st.number_input(
            "Capital (₹)",
            min_value=10000,
            max_value=10000000,
            value=int(config.capital),
            step=10000,
            help="Your trading capital"
        )

        risk_pct = st.slider(
            "Risk per Trade %",
            min_value=0.5,
            max_value=5.0,
            value=config.risk_per_trade * 100,
            step=0.5,
            help="Maximum risk per trade"
        ) / 100

        # Display settings
        st.markdown("##### Display Settings")

        top_n = st.slider(
            "Top N Picks",
            min_value=5,
            max_value=30,
            value=config.top_n_picks,
            help="Number of top picks to display"
        )

        show_warnings = st.toggle(
            "Show Risk Warnings",
            value=config.show_warnings,
            help="Display risk warnings with predictions"
        )

        # Apply changes
        new_config = EngineConfig(
            use_7_model=use_7_model,
            min_model_agreement=min_agreement,
            capital=float(capital),
            risk_per_trade=risk_pct,
            min_confidence=min_confidence,
            min_liquidity_cr=min_liquidity,
            max_atr_pct=max_atr,
            strong_buy_agreement=5 if use_7_model else 3,
            moderate_buy_agreement=min_agreement,
            top_n_picks=top_n,
            show_warnings=show_warnings,
        )

        # Check if config changed
        if new_config.to_dict() != config.to_dict():
            set_engine_config(new_config)

        # Model info
        st.markdown("---")
        st.caption(f"**Active Models:** {7 if use_7_model else 4}")
        if use_7_model:
            st.caption("Base + Physics + Math + Regime + Macro + Alternative + Advanced")
        else:
            st.caption("Base + Physics + Math + Regime")


def render_config_summary():
    """Render a compact config summary (for tab headers)."""
    config = get_engine_config()

    model_text = "7-Model" if config.use_7_model else "4-Model"
    cols = st.columns(4)

    with cols[0]:
        st.metric("Engine", model_text)
    with cols[1]:
        st.metric("Min Agreement", f"{config.min_model_agreement}/{7 if config.use_7_model else 4}")
    with cols[2]:
        st.metric("Min Confidence", f"{config.min_confidence:.0%}")
    with cols[3]:
        st.metric("Capital", f"₹{config.capital/1000:.0f}K")


def get_signal_color(signal: str) -> str:
    """Get color for signal display."""
    if signal in ['BUY', 'STRONG_BUY']:
        return '#00ff88'
    elif signal in ['SELL', 'STRONG_SELL']:
        return '#ff4444'
    else:
        return '#ffaa00'


def get_confidence_level(confidence: float, model_agreement: int, total_models: int) -> str:
    """Get confidence level description."""
    agreement_pct = model_agreement / total_models if total_models > 0 else 0

    if confidence >= 0.65 and agreement_pct >= 0.7:
        return 'HIGH'
    elif confidence >= 0.55 and agreement_pct >= 0.4:
        return 'MEDIUM'
    else:
        return 'LOW'
