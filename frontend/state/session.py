"""
Session State Management
Explicit separation of UI state, data state, analytics state, and system state.

This prevents:
- Stale analytics from being displayed
- UI state affecting computations
- Cache invalidation bugs

Structure:
    session_state = {
        "ui": {           # User interface state
            "selected_tab": 0,
            "auto_refresh": False,
            ...
        },
        "data": {         # Raw data cache
            "symbols": [],
            "ohlc": {},
            "availability": {},
            ...
        },
        "analytics": {    # Computed analytics
            "pair_analysis": None,
            "regime": None,
            ...
        },
        "system": {       # System/connection state
            "connected": False,
            "backend_url": "...",
            "live_feed": {},
            ...
        }
    }
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

from models import (
    PairAnalysisResult,
    RegimeAnalysis,
    DataFreshness,
    LiveFeedStatus,
    FreshnessLevel,
)


# =============================================================================
# State Schemas
# =============================================================================

@dataclass
class UIState:
    """User interface state"""
    selected_symbols: tuple = ("BTCUSDT", "ETHUSDT")
    timeframe: str = "1m"
    rolling_window: int = 20
    z_threshold: float = 2.0
    regression_type: str = "OLS"
    auto_refresh: bool = False
    show_advanced: bool = False


@dataclass
class DataState:
    """Raw data cache"""
    symbols: list = field(default_factory=list)
    ohlc_cache: dict = field(default_factory=dict)  # symbol -> DataFrame
    availability: dict = field(default_factory=dict)
    features_status: dict = field(default_factory=dict)
    last_fetch: Optional[datetime] = None


@dataclass
class AnalyticsState:
    """Computed analytics state"""
    pair_analysis: Optional[Dict] = None  # Will store PairAnalysisResult as dict
    regime: Optional[Dict] = None
    last_computed: Optional[datetime] = None
    is_stale: bool = True
    computation_hash: str = ""  # Hash of inputs to detect changes


@dataclass
class SystemState:
    """System/connection state"""
    backend_url: str = "http://localhost:8000"
    connected: bool = False
    live_feed: dict = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    data_freshness: Optional[Dict] = None
    errors: list = field(default_factory=list)


# =============================================================================
# Session Manager
# =============================================================================

class SessionManager:
    """
    Centralized session state management.
    
    Usage:
        manager = SessionManager()
        manager.init()
        
        # Access state
        ui = manager.ui
        ui.auto_refresh = True
        
        # Update analytics
        manager.update_analytics(result)
    """
    
    def __init__(self):
        self._initialized = False
    
    def init(self) -> None:
        """Initialize all session state sections"""
        if "ui" not in st.session_state:
            st.session_state.ui = asdict(UIState())
        
        if "data" not in st.session_state:
            st.session_state.data = asdict(DataState())
        
        if "analytics" not in st.session_state:
            st.session_state.analytics = asdict(AnalyticsState())
        
        if "system" not in st.session_state:
            st.session_state.system = asdict(SystemState())
        
        self._initialized = True
    
    # =========================================================================
    # Property Accessors
    # =========================================================================
    
    @property
    def ui(self) -> Dict[str, Any]:
        return st.session_state.get("ui", {})
    
    @property
    def data(self) -> Dict[str, Any]:
        return st.session_state.get("data", {})
    
    @property
    def analytics(self) -> Dict[str, Any]:
        return st.session_state.get("analytics", {})
    
    @property
    def system(self) -> Dict[str, Any]:
        return st.session_state.get("system", {})
    
    # =========================================================================
    # State Updates
    # =========================================================================
    
    def update_ui(self, **kwargs) -> None:
        """Update UI state"""
        for key, value in kwargs.items():
            st.session_state.ui[key] = value
    
    def update_data(self, **kwargs) -> None:
        """Update data state"""
        for key, value in kwargs.items():
            st.session_state.data[key] = value
        st.session_state.data["last_fetch"] = datetime.now().isoformat()
    
    def update_analytics(self, pair_analysis: Optional[PairAnalysisResult] = None) -> None:
        """Update analytics state with new computation"""
        now = datetime.now()
        
        if pair_analysis:
            st.session_state.analytics["pair_analysis"] = {
                "symbol_a": pair_analysis.symbol_a,
                "symbol_b": pair_analysis.symbol_b,
                "hedge_ratio": pair_analysis.hedge_ratio,
                "current_zscore": pair_analysis.current_zscore,
                "correlation": pair_analysis.correlation,
                "is_stationary": pair_analysis.is_stationary,
                "half_life": pair_analysis.half_life,
                "confidence": pair_analysis.confidence.value,
                "signal": pair_analysis.signal,
            }
            
            if pair_analysis.regime:
                st.session_state.analytics["regime"] = {
                    "state": pair_analysis.regime.state.value,
                    "confidence": pair_analysis.regime.confidence.value,
                    "adf_pvalue": pair_analysis.regime.adf_pvalue,
                    "half_life": pair_analysis.regime.half_life,
                }
        
        st.session_state.analytics["last_computed"] = now.isoformat()
        st.session_state.analytics["is_stale"] = False
    
    def update_system(self, **kwargs) -> None:
        """Update system state"""
        for key, value in kwargs.items():
            st.session_state.system[key] = value
    
    def update_freshness(self, last_tick_time: Optional[datetime] = None) -> DataFreshness:
        """Update and return data freshness"""
        freshness = DataFreshness.from_timestamp(last_tick_time)
        st.session_state.system["data_freshness"] = {
            "age_seconds": freshness.age_seconds,
            "level": freshness.level.value,
            "last_update": freshness.last_update.isoformat() if freshness.last_update else None,
        }
        return freshness
    
    def mark_analytics_stale(self) -> None:
        """Mark analytics as stale (needs recomputation)"""
        st.session_state.analytics["is_stale"] = True
    
    def add_error(self, error: str) -> None:
        """Add error to system state"""
        errors = st.session_state.system.get("errors", [])
        errors.append({
            "message": error,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 10 errors
        st.session_state.system["errors"] = errors[-10:]
    
    def clear_errors(self) -> None:
        """Clear all errors"""
        st.session_state.system["errors"] = []
    
    # =========================================================================
    # Computed Properties
    # =========================================================================
    
    def get_computation_hash(self, symbol1: str, symbol2: str, timeframe: str) -> str:
        """Generate hash for computation inputs to detect changes"""
        return f"{symbol1}:{symbol2}:{timeframe}"
    
    def should_recompute(self, symbol1: str, symbol2: str, timeframe: str) -> bool:
        """Check if analytics should be recomputed"""
        current_hash = self.get_computation_hash(symbol1, symbol2, timeframe)
        stored_hash = st.session_state.analytics.get("computation_hash", "")
        
        if current_hash != stored_hash:
            st.session_state.analytics["computation_hash"] = current_hash
            return True
        
        return st.session_state.analytics.get("is_stale", True)
    
    @property
    def is_connected(self) -> bool:
        return st.session_state.system.get("connected", False)
    
    @property
    def symbols(self) -> list:
        return st.session_state.data.get("symbols", [])
    
    @property
    def live_feed_running(self) -> bool:
        return st.session_state.system.get("live_feed", {}).get("is_running", False)


# =============================================================================
# Module-Level Functions (for backwards compatibility)
# =============================================================================

_manager: Optional[SessionManager] = None


def get_manager() -> SessionManager:
    """Get or create session manager singleton"""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


def init_session() -> SessionManager:
    """Initialize session state and return manager"""
    manager = get_manager()
    manager.init()
    return manager


def get_ui_state() -> Dict[str, Any]:
    """Get UI state"""
    return get_manager().ui


def get_data_state() -> Dict[str, Any]:
    """Get data state"""
    return get_manager().data


def get_analytics_state() -> Dict[str, Any]:
    """Get analytics state"""
    return get_manager().analytics


def get_system_state() -> Dict[str, Any]:
    """Get system state"""
    return get_manager().system


def update_analytics(pair_analysis: Optional[PairAnalysisResult] = None) -> None:
    """Update analytics state"""
    get_manager().update_analytics(pair_analysis)


def update_data_freshness(last_tick_time: Optional[datetime] = None) -> DataFreshness:
    """Update data freshness"""
    return get_manager().update_freshness(last_tick_time)

