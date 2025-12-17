"""
Live Analytics
Real-time analytics with <500ms latency requirement.

Update: Every tick
Use: Monitoring, alerts, dashboards

These analytics run continuously as new ticks arrive.
"""

import numpy as np
from typing import Optional

from .models import LivePriceStats, LiveSpread


def price_stats(
    prices: np.ndarray,
    volumes: np.ndarray = None,
    window: int = 20
) -> LivePriceStats:
    """
    Compute live price statistics.
    
    Purpose: Regime detection, volatility awareness, liquidity intuition
    
    Args:
        prices: Price array (recent N prices)
        volumes: Volume array (optional, for VWAP)
        window: Rolling window for mean/std
    
    Returns:
        LivePriceStats with last, mean, std, min, max, vwap
    """
    if len(prices) == 0:
        return LivePriceStats(0, 0, 0, 0, 0, None)
    
    last_price = float(prices[-1])
    
    # Rolling stats (use available data if less than window)
    effective_window = min(window, len(prices))
    recent = prices[-effective_window:]
    
    rolling_mean = float(np.mean(recent))
    rolling_std = float(np.std(recent)) if len(recent) > 1 else 0.0
    
    min_price = float(np.min(prices))
    max_price = float(np.max(prices))
    
    # VWAP if volumes provided
    vwap = None
    if volumes is not None and len(volumes) == len(prices):
        total_volume = np.sum(volumes)
        if total_volume > 0:
            vwap = float(np.sum(prices * volumes) / total_volume)
    
    return LivePriceStats(
        last_price=last_price,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        min_price=min_price,
        max_price=max_price,
        vwap=vwap
    )


def spread(
    price_a: float,
    price_b: float,
    hedge_ratio: float,
    spread_mean: float,
    spread_std: float
) -> LiveSpread:
    """
    Compute live spread and z-score.
    
    THE SINGLE MOST IMPORTANT LIVE METRIC.
    
    Formula:
        spread = price_A - β * price_B
        z_score = (spread - μ) / σ
    
    Args:
        price_a: Current price of asset A
        price_b: Current price of asset B
        hedge_ratio: β from batch OLS (not recomputed live)
        spread_mean: μ from batch computation
        spread_std: σ from batch computation
    
    Returns:
        LiveSpread with spread, z_score
    """
    spread_value = price_a - hedge_ratio * price_b
    
    if spread_std > 0:
        z = (spread_value - spread_mean) / spread_std
    else:
        z = 0.0
    
    return LiveSpread(
        spread=spread_value,
        z_score=z,
        hedge_ratio=hedge_ratio,
        spread_mean=spread_mean,
        spread_std=spread_std
    )


def z_score(
    current_value: float,
    rolling_mean: float,
    rolling_std: float
) -> float:
    """
    Compute z-score for any value.
    
    z = (x - μ) / σ
    
    Used for: Alerting (z > 2, z < -2), trade intuition, visual emphasis
    """
    if rolling_std <= 0:
        return 0.0
    return (current_value - rolling_mean) / rolling_std


def rolling_z_scores(
    values: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Compute rolling z-scores for an array.
    
    Args:
        values: Input array
        window: Rolling window
    
    Returns:
        Array of z-scores
    """
    import pandas as pd
    
    series = pd.Series(values)
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    z_scores = (series - rolling_mean) / rolling_std
    return z_scores.fillna(0).values

