"""
Analytics Module
MFT / Stat-Arb focused analytics with clear Live vs Batch separation.

Structure:
    analytics/
    ├── models.py    → Output types (dataclasses)
    ├── live.py      → Live analytics (<500ms)
    ├── batch.py     → Batch analytics (bar close)
    └── pair.py      → Full pair analysis

Usage:
    from analytics import live, batch, pair
    
    # Live (every tick)
    stats = live.price_stats(prices)
    spread = live.spread(price_a, price_b, hedge_ratio, mean, std)
    
    # Batch (on bar close)
    hr = batch.hedge_ratio(prices_a, prices_b)
    adf = batch.adf_test(spread_series)
    
    # Full analysis
    result = pair.full_analysis(prices_a, prices_b)

Design Principles:
    ✓ ALL functions are PURE (inputs → computation → outputs)
    ✓ NO database access
    ✓ NO state management
    ✓ NO WebSocket handling
"""

from . import live
from . import batch
from . import pair

from .models import (
    # Live types
    LivePriceStats,
    LiveSpread,
    # Batch types
    BatchHedgeRatio,
    BatchCorrelation,
    BatchADF,
    BatchHalfLife,
)

__all__ = [
    # Modules
    "live",
    "batch", 
    "pair",
    # Types
    "LivePriceStats",
    "LiveSpread",
    "BatchHedgeRatio",
    "BatchCorrelation",
    "BatchADF",
    "BatchHalfLife",
]

