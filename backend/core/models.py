"""
Domain Models
The SINGLE SOURCE OF TRUTH for data formats.

After normalization, the system only sees these types.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# =============================================================================
# Data Source
# =============================================================================

class DataSource(str, Enum):
    """Where data came from — tagged at entry, never changes"""
    WEBSOCKET = "websocket"
    UPLOAD = "upload"
    API = "api"


# =============================================================================
# TickEvent — The Core Data Contract
# =============================================================================

class TickEvent(BaseModel):
    """
    A single trade/tick event.
    
    This is THE internal representation. Everything converts to this.
    The engine never sees JSON, CSV rows, or WebSocket payloads.
    It sees ONLY TickEvents.
    
    Fields:
        symbol: Uppercase symbol (BTCUSDT)
        ts: Parsed datetime
        price: Float price
        size: Trade size/quantity
        source: Where it came from
        side: Optional buy/sell
        trade_id: Optional exchange trade ID
    """
    symbol: str = Field(..., min_length=1, max_length=20)
    ts: datetime
    price: float = Field(..., gt=0)
    size: float = Field(default=0.0, ge=0)
    source: DataSource = DataSource.UPLOAD
    side: Optional[Literal["buy", "sell"]] = None
    trade_id: Optional[str] = None
    
    @field_validator('symbol', mode='before')
    @classmethod
    def uppercase_symbol(cls, v):
        """Always uppercase symbols"""
        return v.upper() if isinstance(v, str) else v
    
    @field_validator('ts', mode='before')
    @classmethod
    def parse_timestamp(cls, v):
        """Handle various timestamp formats"""
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # ISO format
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        if isinstance(v, (int, float)):
            # Unix timestamp (seconds or milliseconds)
            if v > 1e12:
                return datetime.fromtimestamp(v / 1000)
            return datetime.fromtimestamp(v)
        return v


# =============================================================================
# OHLCBar — Aggregated Price Data
# =============================================================================

class OHLCBar(BaseModel):
    """
    A single OHLC bar (candlestick).
    
    Either:
    - Created by resampling ticks
    - Uploaded directly from OHLC CSV
    """
    symbol: str
    ts: datetime  # Bar open time
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trade_count: int = 0
    source: DataSource = DataSource.UPLOAD
    
    @field_validator('symbol', mode='before')
    @classmethod
    def uppercase_symbol(cls, v):
        return v.upper() if isinstance(v, str) else v
    
    def update(self, price: float, size: float = 0.0):
        """Update bar with new tick (mutates in place)"""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += size
        self.trade_count += 1


# =============================================================================
# API Response Models
# =============================================================================

class IngestionResult(BaseModel):
    """Result of data ingestion"""
    success: bool = True
    count: int = 0
    errors: int = 0
    symbols: List[str] = []
    message: str = ""


class TickBatch(BaseModel):
    """Batch of ticks from external source"""
    ticks: List[dict]


# =============================================================================
# Converters — External → Internal
# =============================================================================

def to_tick_event(data: dict, source: DataSource = DataSource.UPLOAD) -> TickEvent:
    """
    Convert external data format to TickEvent.
    
    This is the NORMALIZATION POINT.
    All external formats go through here.
    
    Handles:
    - timestamp/ts field variants
    - size/qty/volume field variants
    - type coercion
    """
    # Timestamp variants
    ts = data.get('timestamp') or data.get('ts') or data.get('time')
    
    # Size variants
    size = data.get('size') or data.get('qty') or data.get('volume') or 0.0
    
    return TickEvent(
        symbol=data['symbol'],
        ts=ts,
        price=float(data['price']),
        size=float(size),
        source=source,
        side=data.get('side'),
        trade_id=str(data.get('trade_id')) if data.get('trade_id') else None
    )


def to_ohlc_bar(data: dict, source: DataSource = DataSource.UPLOAD) -> OHLCBar:
    """Convert external OHLC data to OHLCBar"""
    ts = data.get('timestamp') or data.get('ts') or data.get('time')
    
    return OHLCBar(
        symbol=data.get('symbol', 'UNKNOWN'),
        ts=ts,
        open=float(data['open']),
        high=float(data['high']),
        low=float(data['low']),
        close=float(data['close']),
        volume=float(data.get('volume', 0)),
        source=source
    )
