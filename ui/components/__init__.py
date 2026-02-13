"""
UI Components Package

Shared components and utilities for the Streamlit dashboard.
"""

from .common import (
    # Cached resources
    get_stock_screener,
    get_quant_engine,
    get_multi_model_judge,
    get_improved_screener,
    get_comprehensive_analyzer,
    get_trade_tracker,
    get_intraday_timing,
    get_robust_predictor,
    get_advanced_predictor,
    get_signal_helper,
    get_market_timing,
    get_insights_engine,
    get_enhanced_scoring_engine,
    get_enhanced_signal_generator,
    get_unified_predictor,
    get_robustness_checker,
    get_statistical_validator,
    get_ui_prediction_api,
    get_ai_engine,
    # Prediction helpers
    get_unified_prediction,
    get_enhanced_prediction_data,
    # Insight generation
    generate_stock_insight,
    generate_batch_insights_llm,
    # UI components
    render_deep_ai_analysis,
    render_enhanced_signal_box,
    render_stock_table,
    render_stock_card,
    # CSS
    inject_custom_css,
)

__all__ = [
    'get_stock_screener',
    'get_quant_engine',
    'get_multi_model_judge',
    'get_improved_screener',
    'get_comprehensive_analyzer',
    'get_trade_tracker',
    'get_intraday_timing',
    'get_robust_predictor',
    'get_advanced_predictor',
    'get_signal_helper',
    'get_market_timing',
    'get_insights_engine',
    'get_enhanced_scoring_engine',
    'get_enhanced_signal_generator',
    'get_unified_predictor',
    'get_robustness_checker',
    'get_statistical_validator',
    'get_ui_prediction_api',
    'get_ai_engine',
    'get_unified_prediction',
    'get_enhanced_prediction_data',
    'generate_stock_insight',
    'generate_batch_insights_llm',
    'render_deep_ai_analysis',
    'render_enhanced_signal_box',
    'render_stock_table',
    'render_stock_card',
    'inject_custom_css',
]
