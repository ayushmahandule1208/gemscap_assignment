"""
Typed Analytics Models
Lightweight dataclasses for frontend analytics state.

These prevent the dict → DataFrame → dict anti-pattern
and make analytics bugs visible at development time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
import pandas as pd


# =============================================================================
# Enums
# =============================================================================

class RegimeState(Enum):
    """Market regime classification"""
    MEAN_REVERTING = "mean_reverting"
    TRENDING = "trending"
    TRANSITION = "transition"
    UNKNOWN = "unknown"
    
    @property
    def label(self) -> str:
        labels = {
            "mean_reverting": "🟢 Mean-Reverting",
            "trending": "🔴 Trending",
            "transition": "🟡 Transition",
            "unknown": "⚪ Unknown",
        }
        return labels.get(self.value, "⚪ Unknown")
    
    @property
    def color(self) -> str:
        colors = {
            "mean_reverting": "#10b981",
            "trending": "#ef4444",
            "transition": "#f59e0b",
            "unknown": "#64748b",
        }
        return colors.get(self.value, "#64748b")


class ConfidenceLevel(Enum):
    """Confidence in analytics result"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT_DATA = "insufficient_data"
    
    @property
    def label(self) -> str:
        labels = {
            "high": "✓ HIGH",
            "medium": "~ MEDIUM",
            "low": "⚠ LOW",
            "insufficient_data": "✗ INSUFFICIENT DATA",
        }
        return labels.get(self.value, "?")
    
    @property
    def color(self) -> str:
        colors = {
            "high": "#10b981",
            "medium": "#f59e0b",
            "low": "#ef4444",
            "insufficient_data": "#64748b",
        }
        return colors.get(self.value, "#64748b")


class FreshnessLevel(Enum):
    """Data freshness classification"""
    FRESH = "fresh"        # < 5s
    STALE = "stale"        # 5-30s
    OLD = "old"            # > 30s
    DISCONNECTED = "disconnected"
    
    @property
    def color(self) -> str:
        colors = {
            "fresh": "#10b981",
            "stale": "#f59e0b",
            "old": "#ef4444",
            "disconnected": "#64748b",
        }
        return colors.get(self.value, "#64748b")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class DataFreshness:
    """Tracks data freshness and staleness"""
    last_update: Optional[datetime] = None
    age_seconds: float = 0.0
    level: FreshnessLevel = FreshnessLevel.DISCONNECTED
    
    @classmethod
    def from_timestamp(cls, ts: Optional[datetime]) -> "DataFreshness":
        if ts is None:
            return cls(level=FreshnessLevel.DISCONNECTED)
        
        age = (datetime.now() - ts).total_seconds()
        
        if age < 5:
            level = FreshnessLevel.FRESH
        elif age < 30:
            level = FreshnessLevel.STALE
        else:
            level = FreshnessLevel.OLD
        
        return cls(last_update=ts, age_seconds=age, level=level)
    
    @property
    def display(self) -> str:
        if self.level == FreshnessLevel.DISCONNECTED:
            return "Disconnected"
        return f"{self.age_seconds:.1f}s ago"


@dataclass
class FeatureStatus:
    """Feature unlock status"""
    name: str
    unlocked: bool
    required_bars: int
    current_bars: int
    progress: int  # 0-100
    
    @property
    def progress_pct(self) -> str:
        return f"{self.progress}%"


@dataclass
class LiveFeedStatus:
    """Live feed connection status"""
    is_running: bool = False
    symbols: List[str] = field(default_factory=list)
    ticks_received: int = 0
    ticks_per_second: float = 0.0
    uptime_seconds: float = 0.0
    last_tick_time: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "LiveFeedStatus":
        return cls(
            is_running=data.get("is_running", False),
            symbols=data.get("symbols", []),
            ticks_received=data.get("ticks_received", 0),
            ticks_per_second=data.get("ticks_per_second", 0.0),
            uptime_seconds=data.get("uptime_seconds", 0.0),
        )


@dataclass
class RegimeAnalysis:
    """Regime detection result"""
    state: RegimeState
    confidence: ConfidenceLevel
    adf_pvalue: Optional[float] = None
    half_life: Optional[float] = None
    zscore_persistence: Optional[float] = None  # How long z-score stays extreme
    volatility_ratio: Optional[float] = None    # Current vol / historical vol
    
    @classmethod
    def detect(
        cls,
        adf_pvalue: Optional[float],
        half_life: Optional[float],
        current_zscore: Optional[float],
        zscore_series: Optional[pd.Series] = None,
        min_bars: int = 50
    ) -> "RegimeAnalysis":
        """
        Detect market regime from analytics.
        
        Regime Logic:
        - Mean-Reverting: ADF p < 0.05 AND half_life < 50
        - Trending: ADF p > 0.1 OR half_life > 100
        - Transition: In between
        """
        # Insufficient data
        if adf_pvalue is None or half_life is None:
            return cls(
                state=RegimeState.UNKNOWN,
                confidence=ConfidenceLevel.INSUFFICIENT_DATA
            )
        
        # Calculate z-score persistence if series provided
        zscore_persistence = None
        if zscore_series is not None and len(zscore_series) > 10:
            # Count how many of last 10 bars had |z| > 1.5
            recent = zscore_series.tail(10).abs()
            zscore_persistence = (recent > 1.5).mean()
        
        # Regime detection
        is_stationary = adf_pvalue < 0.05
        fast_reversion = half_life is not None and half_life < 50
        slow_reversion = half_life is not None and half_life > 100
        
        if is_stationary and fast_reversion:
            state = RegimeState.MEAN_REVERTING
            # High confidence if very significant
            if adf_pvalue < 0.01 and half_life < 30:
                confidence = ConfidenceLevel.HIGH
            else:
                confidence = ConfidenceLevel.MEDIUM
        elif not is_stationary or slow_reversion:
            state = RegimeState.TRENDING
            confidence = ConfidenceLevel.MEDIUM if adf_pvalue > 0.1 else ConfidenceLevel.LOW
        else:
            state = RegimeState.TRANSITION
            confidence = ConfidenceLevel.LOW
        
        return cls(
            state=state,
            confidence=confidence,
            adf_pvalue=adf_pvalue,
            half_life=half_life,
            zscore_persistence=zscore_persistence
        )


@dataclass
class PairAnalysisResult:
    """
    Complete pair analysis result with typed fields.
    
    This replaces the dict-based approach and makes
    analytics bugs visible at development time.
    """
    # Identifiers
    symbol_a: str
    symbol_b: str
    timeframe: str
    bars_used: int
    
    # Core Analytics
    hedge_ratio: float
    hedge_ratio_r2: float
    spread_mean: float
    spread_std: float
    current_spread: float
    current_zscore: float
    correlation: float
    
    # Stationarity
    adf_statistic: Optional[float] = None
    adf_pvalue: Optional[float] = None
    is_stationary: bool = False
    
    # Mean Reversion
    half_life: Optional[float] = None
    is_mean_reverting: bool = False
    
    # Regime
    regime: Optional[RegimeAnalysis] = None
    
    # Freshness
    freshness: Optional[DataFreshness] = None
    computed_at: datetime = field(default_factory=datetime.now)
    
    # Confidence
    @property
    def confidence(self) -> ConfidenceLevel:
        """Overall confidence in the analysis"""
        if self.bars_used < 20:
            return ConfidenceLevel.INSUFFICIENT_DATA
        if self.bars_used < 50:
            return ConfidenceLevel.LOW
        if self.is_stationary and self.hedge_ratio_r2 > 0.7:
            return ConfidenceLevel.HIGH
        if self.hedge_ratio_r2 > 0.5:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    
    @property
    def signal(self) -> str:
        """Trading signal based on z-score"""
        if abs(self.current_zscore) < 0.5:
            return "NEUTRAL"
        elif self.current_zscore > 2.0:
            return "SHORT_SPREAD"
        elif self.current_zscore < -2.0:
            return "LONG_SPREAD"
        elif self.current_zscore > 1.0:
            return "WEAK_SHORT"
        elif self.current_zscore < -1.0:
            return "WEAK_LONG"
        return "NEUTRAL"
    
    @classmethod
    def from_api_response(cls, data: dict) -> "PairAnalysisResult":
        """Parse API response into typed model"""
        hr = data.get("hedge_ratio", {})
        spread = data.get("spread", {})
        z = data.get("z_score", {})
        corr = data.get("correlation", {})
        stat = data.get("stationarity", {})
        mr = data.get("mean_reversion", {})
        
        result = cls(
            symbol_a=data.get("symbol_a", ""),
            symbol_b=data.get("symbol_b", ""),
            timeframe=data.get("timeframe", "1m"),
            bars_used=data.get("bars_used", 0),
            hedge_ratio=hr.get("beta", 1.0),
            hedge_ratio_r2=hr.get("r_squared", 0.0),
            spread_mean=spread.get("mean", 0.0),
            spread_std=spread.get("std", 1.0),
            current_spread=spread.get("current", 0.0),
            current_zscore=z.get("current", 0.0),
            correlation=corr.get("current", 0.0),
            adf_statistic=stat.get("statistic"),
            adf_pvalue=stat.get("p_value"),
            is_stationary=stat.get("is_stationary", False),
            half_life=mr.get("half_life"),
            is_mean_reverting=mr.get("is_mean_reverting", False),
        )
        
        # Detect regime
        result.regime = RegimeAnalysis.detect(
            adf_pvalue=result.adf_pvalue,
            half_life=result.half_life,
            current_zscore=result.current_zscore
        )
        
        return result


@dataclass  
class BacktestResult:
    """Typed backtest result"""
    total_return: float
    win_rate: float
    num_trades: int
    sharpe: float
    max_drawdown: float
    avg_trade_return: float
    trades: List[Dict] = field(default_factory=list)
    
    @classmethod
    def from_api_response(cls, data: dict) -> "BacktestResult":
        bt = data.get("backtest", {})
        return cls(
            total_return=bt.get("total_return", 0.0),
            win_rate=bt.get("win_rate", 0.0),
            num_trades=bt.get("num_trades", 0),
            sharpe=bt.get("sharpe", 0.0),
            max_drawdown=bt.get("max_drawdown", 0.0),
            avg_trade_return=bt.get("avg_trade_return", 0.0),
            trades=bt.get("trades", []),
        )

