"""
Pair Analysis
Complete analysis for stat-arb pairs.

Combines batch analytics into comprehensive results.
"""

import numpy as np
from typing import Dict, Any

from . import batch


def full_analysis(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    symbol_a: str = "A",
    symbol_b: str = "B",
    window: int = 20,
    z_threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Complete pair analysis for stat-arb.
    
    Combines all batch analytics into single result.
    
    Args:
        prices_a: Asset A price series
        prices_b: Asset B price series
        symbol_a: Asset A symbol
        symbol_b: Asset B symbol
        window: Rolling window
        z_threshold: Z-score threshold for signals
    
    Returns:
        Complete analysis dictionary
    """
    # Hedge ratio
    hr = batch.hedge_ratio(prices_a, prices_b)
    
    # Spread stats
    spread = batch.spread_stats(prices_a, prices_b, hr.beta, window)
    
    # Current z-score
    current_z = spread['z_scores'][-1] if len(spread['z_scores']) > 0 else 0
    
    # Signal
    if current_z > z_threshold:
        signal = "SHORT_SPREAD"
    elif current_z < -z_threshold:
        signal = "LONG_SPREAD"
    else:
        signal = "NEUTRAL"
    
    # Correlation
    corr = batch.rolling_correlation(prices_a, prices_b, window)
    
    # Stationarity
    adf = batch.adf_test(spread['spread'])
    
    # Half-life
    hl = batch.half_life(spread['spread'])
    
    return {
        'pair': f"{symbol_a}/{symbol_b}",
        'hedge_ratio': {
            'beta': round(float(hr.beta), 4),
            'alpha': round(float(hr.alpha), 4),
            'r_squared': round(float(hr.r_squared), 4),
        },
        'spread': {
            'current': float(spread['spread'][-1]),
            'mean': round(float(spread['mean']), 4),
            'std': round(float(spread['std']), 4),
        },
        'z_score': {
            'current': round(float(current_z), 4),
            'threshold': float(z_threshold),
        },
        'correlation': {
            'current': round(float(corr.current), 4),
            'mean': round(float(corr.mean), 4),
            'is_stable': bool(corr.is_stable),
        },
        'stationarity': {
            'adf_statistic': float(adf.statistic),
            'p_value': float(adf.p_value),
            'is_stationary': bool(adf.is_stationary),
        },
        'mean_reversion': {
            'half_life': round(float(hl.half_life), 2) if hl.half_life != float('inf') else None,
            'is_mean_reverting': bool(hl.is_mean_reverting),
        },
        'signal': signal,
        'window': int(window),
    }


def quick_stats(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    hedge_ratio_value: float = None
) -> Dict[str, Any]:
    """
    Quick spread statistics for dashboard.
    
    Lighter than full_analysis, for frequent updates.
    """
    if hedge_ratio_value is None:
        hr = batch.hedge_ratio(prices_a, prices_b)
        hedge_ratio_value = hr.beta
    
    spread = prices_a - hedge_ratio_value * prices_b
    spread_mean = float(np.nanmean(spread))
    spread_std = float(np.nanstd(spread))
    
    current_spread = float(spread[-1])
    current_z = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
    
    return {
        'hedge_ratio': hedge_ratio_value,
        'spread': current_spread,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'z_score': current_z
    }

