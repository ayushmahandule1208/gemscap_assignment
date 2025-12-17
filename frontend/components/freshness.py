"""
Data Freshness Indicator Component
Shows real-time data age with color coding.

Quants always ask: "How fresh is this?"
"""

import streamlit as st
from datetime import datetime
from typing import Optional

from models import DataFreshness, FreshnessLevel


def render_freshness_badge(
    freshness: DataFreshness,
    show_time: bool = True,
    compact: bool = False,
) -> None:
    """
    Render a data freshness badge.
    
    Color coding:
    - Green: < 5 seconds (FRESH)
    - Yellow: 5-30 seconds (STALE)
    - Red: > 30 seconds (OLD)
    - Gray: Disconnected
    """
    
    color = freshness.level.color
    
    labels = {
        FreshnessLevel.FRESH: "LIVE",
        FreshnessLevel.STALE: "DELAYED",
        FreshnessLevel.OLD: "STALE",
        FreshnessLevel.DISCONNECTED: "OFFLINE",
    }
    label = labels.get(freshness.level, "UNKNOWN")
    
    # Pulse animation for fresh data
    animation = "animation: pulse 1.5s infinite;" if freshness.level == FreshnessLevel.FRESH else ""
    
    if compact:
        st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; background: {color}22; border: 1px solid {color}44; border-radius: 12px;">
            <div style="width: 6px; height: 6px; background: {color}; border-radius: 50%; {animation}"></div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {color}; font-weight: 600;">{label}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        time_display = freshness.display if show_time else ""
        last_update = ""
        if freshness.last_update and show_time:
            last_update = freshness.last_update.strftime('%H:%M:%S')
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: {color}15; border: 1px solid {color}30; border-radius: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; background: {color}; border-radius: 50%; box-shadow: 0 0 8px {color}; {animation}"></div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {color}; font-weight: 600;">{label}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #f0f4f8;">{time_display}</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #64748b;">{last_update}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_freshness_header(
    freshness: DataFreshness,
    title: str = "Data Status",
) -> None:
    """
    Render a full freshness header with warning if stale.
    """
    
    render_freshness_badge(freshness, show_time=True, compact=False)
    
    # Warning banner for stale/old data
    if freshness.level == FreshnessLevel.STALE:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; padding: 8px 12px; border-radius: 0 6px 6px 0; margin-top: 8px;">
            <span style="font-family: 'Inter', sans-serif; font-size: 12px; color: #fbbf24;">
                ⚠️ Data is slightly delayed. Signals may not reflect current market.
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    elif freshness.level == FreshnessLevel.OLD:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; padding: 8px 12px; border-radius: 0 6px 6px 0; margin-top: 8px;">
            <span style="font-family: 'Inter', sans-serif; font-size: 12px; color: #f87171;">
                🚨 Data is stale (>30s). Consider refreshing or checking connection.
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    elif freshness.level == FreshnessLevel.DISCONNECTED:
        st.markdown("""
        <div style="background: rgba(100, 116, 139, 0.1); border-left: 3px solid #64748b; padding: 8px 12px; border-radius: 0 6px 6px 0; margin-top: 8px;">
            <span style="font-family: 'Inter', sans-serif; font-size: 12px; color: #94a3b8;">
                📡 No connection to live feed. Showing cached data if available.
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_inline_freshness(age_seconds: float) -> str:
    """
    Return inline HTML for freshness indicator (for use in larger components).
    """
    if age_seconds < 5:
        color = "#10b981"
        label = "LIVE"
    elif age_seconds < 30:
        color = "#f59e0b"
        label = "DELAYED"
    else:
        color = "#ef4444"
        label = "STALE"
    
    return f"""
    <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; background: {color}22; border-radius: 10px; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: {color}; font-weight: 600;">
        <span style="width: 4px; height: 4px; background: {color}; border-radius: 50%;"></span>
        {label} ({age_seconds:.1f}s)
    </span>
    """

