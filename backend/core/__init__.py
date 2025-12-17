"""
Core Module
Business logic for data ingestion and analytics.

Exports:
    Models: TickEvent, OHLCBar, DataSource, IngestionResult
    Engine: get_engine, IngestionEngine
    Buffer: LiveBuffer, OHLCBuffer
    Converters: to_tick_event, to_ohlc_bar
"""

from .models import (
    TickEvent,
    OHLCBar,
    DataSource,
    IngestionResult,
    TickBatch,
    to_tick_event,
    to_ohlc_bar,
)

from .engine import get_engine, IngestionEngine
from .buffer import LiveBuffer, OHLCBuffer
from .resampler import OHLCResampler, resample_ticks

# Re-export from db for convenience
from db import get_storage, SQLiteStorage

__all__ = [
    # Models
    "TickEvent",
    "OHLCBar", 
    "DataSource",
    "IngestionResult",
    "TickBatch",
    "to_tick_event",
    "to_ohlc_bar",
    # Engine
    "get_engine",
    "IngestionEngine",
    # Buffer
    "LiveBuffer",
    "OHLCBuffer",
    # Resampler
    "OHLCResampler",
    "resample_ticks",
    # Storage (from db)
    "get_storage",
    "SQLiteStorage",
]
