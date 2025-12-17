"""
Analytics Output Types
Dataclasses for analytics results.
"""

from dataclasses import dataclass
from typing import Dict, Optional


# =============================================================================
# LIVE ANALYTICS OUTPUT TYPES
# =============================================================================

@dataclass
class LivePriceStats:
    """
    Live price statistics - updated every tick.
    
    Purpose: Regime detection, volatility awareness, liquidity intuition
    """
    last_price: float
    rolling_mean: float
    rolling_std: float
    min_price: float
    max_price: float
    vwap: Optional[float] = None


@dataclass
class LiveSpread:
    """
    Live spread - updated every tick.
    
    THE SINGLE MOST IMPORTANT LIVE METRIC for stat-arb.
    """
    spread: float
    z_score: float
    hedge_ratio: float  # From batch
    spread_mean: float
    spread_std: float


# =============================================================================
# BATCH ANALYTICS OUTPUT TYPES
# =============================================================================

@dataclass
class BatchHedgeRatio:
    """
    Hedge ratio from OLS regression.
    
    price_A = α + β * price_B + ε
    """
    beta: float          # Hedge ratio
    alpha: float         # Intercept
    r_squared: float     # Goodness of fit
    std_error: float     # Standard error


@dataclass
class BatchCorrelation:
    """Rolling correlation result."""
    current: float
    mean: float
    std: float
    is_stable: bool      # std < 0.2


@dataclass
class BatchADF:
    """
    ADF test result for stationarity.
    
    Purpose: Validate mean-reversion assumption
    """
    statistic: float
    p_value: float
    is_stationary: bool
    critical_values: Dict[str, float]


@dataclass
class BatchHalfLife:
    """
    Half-life of mean reversion.
    
    Lower half-life = faster mean reversion = better for trading
    """
    half_life: float
    lambda_coef: float   # Mean reversion speed
    is_mean_reverting: bool
    r_squared: float

