"""
UI Tabs Package

Individual tab renderers for the Streamlit dashboard.
Each tab is a separate module for maintainability.
"""

from .top_picks import render_top_picks_tab
from .analysis import render_comprehensive_analysis_tab, render_sector_analysis_tab
from .screener import render_unified_screener_tab, render_about_tab

__all__ = [
    'render_top_picks_tab',
    'render_comprehensive_analysis_tab',
    'render_sector_analysis_tab',
    'render_unified_screener_tab',
    'render_about_tab',
]
