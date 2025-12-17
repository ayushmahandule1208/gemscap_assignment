"""
In-Memory Buffers
Fast, bounded, symbol-keyed storage for real-time access.

Purpose:
- Analytics need fast access
- Dashboards need recent data
- No disk I/O allowed here

This is READ-OPTIMIZED, NOT DURABLE.
"""

from collections import deque
from typing import Dict, List, Optional
from datetime import datetime

from .models import TickEvent, OHLCBar


# =============================================================================
# Live Buffer (Ticks)
# =============================================================================

class LiveBuffer:
    """
    In-memory buffer for live tick data.
    
    - Per-symbol deques with automatic eviction
    - O(1) append, O(1) latest access
    - Thread-safe not required for single-threaded FastAPI
    
    Usage:
        buffer = LiveBuffer(maxlen=10000)
        buffer.append(tick)
        recent = buffer.get("BTCUSDT", limit=100)
    """
    
    def __init__(self, maxlen: int = 10000):
        self.maxlen = maxlen
        self._data: Dict[str, deque] = {}
        self._count: int = 0
    
    def append(self, tick: TickEvent) -> None:
        """Add single tick to buffer"""
        symbol = tick.symbol
        
        if symbol not in self._data:
            self._data[symbol] = deque(maxlen=self.maxlen)
        
        self._data[symbol].append(tick)
        self._count += 1
    
    def extend(self, ticks: List[TickEvent]) -> int:
        """Add multiple ticks. Returns count added."""
        for tick in ticks:
            self.append(tick)
        return len(ticks)
    
    def get(self, symbol: str, limit: int = None) -> List[TickEvent]:
        """Get ticks for symbol (most recent last)"""
        symbol = symbol.upper()
        if symbol not in self._data:
            return []
        
        data = list(self._data[symbol])
        if limit:
            return data[-limit:]
        return data
    
    def get_latest(self, symbol: str) -> Optional[TickEvent]:
        """Get most recent tick"""
        symbol = symbol.upper()
        if symbol not in self._data or len(self._data[symbol]) == 0:
            return None
        return self._data[symbol][-1]
    
    def get_prices(self, symbol: str, limit: int = None) -> List[float]:
        """Get price array for analytics"""
        ticks = self.get(symbol, limit)
        return [t.price for t in ticks]
    
    def symbols(self) -> List[str]:
        """List all symbols in buffer"""
        return list(self._data.keys())
    
    def count(self, symbol: str = None) -> int:
        """Get tick count"""
        if symbol:
            return len(self._data.get(symbol.upper(), []))
        return self._count
    
    def clear(self, symbol: str = None) -> None:
        """Clear buffer"""
        if symbol:
            self._data.pop(symbol.upper(), None)
        else:
            self._data.clear()
            self._count = 0
    
    def stats(self) -> dict:
        """Buffer statistics"""
        return {
            "total_ticks": self._count,
            "symbols": len(self._data),
            "per_symbol": {sym: len(d) for sym, d in self._data.items()}
        }


# =============================================================================
# OHLC Buffer (Bars)
# =============================================================================

class OHLCBuffer:
    """
    In-memory buffer for OHLC bars.
    
    Structure: symbol → timeframe → deque[OHLCBar]
    
    Usage:
        buffer = OHLCBuffer(maxlen=1000)
        buffer.append("BTCUSDT", "1m", bar)
        bars = buffer.get("BTCUSDT", "1m", limit=100)
    """
    
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        # symbol -> timeframe -> deque[OHLCBar]
        self._data: Dict[str, Dict[str, deque]] = {}
    
    def append(self, symbol: str, timeframe: str, bar: OHLCBar) -> None:
        """Add or update a bar"""
        symbol = symbol.upper()
        
        if symbol not in self._data:
            self._data[symbol] = {}
        if timeframe not in self._data[symbol]:
            self._data[symbol][timeframe] = deque(maxlen=self.maxlen)
        
        bars = self._data[symbol][timeframe]
        
        # Update last bar if same timestamp, else append
        if bars and bars[-1].ts == bar.ts:
            bars[-1] = bar
        else:
            bars.append(bar)
    
    def extend(self, symbol: str, timeframe: str, bars: List[OHLCBar]) -> int:
        """Add multiple bars"""
        for bar in bars:
            self.append(symbol, timeframe, bar)
        return len(bars)
    
    def get(self, symbol: str, timeframe: str, limit: int = None) -> List[OHLCBar]:
        """Get bars for symbol and timeframe"""
        symbol = symbol.upper()
        
        if symbol not in self._data or timeframe not in self._data[symbol]:
            return []
        
        data = list(self._data[symbol][timeframe])
        if limit:
            return data[-limit:]
        return data
    
    def get_latest(self, symbol: str, timeframe: str) -> Optional[OHLCBar]:
        """Get most recent bar"""
        bars = self.get(symbol, timeframe, limit=1)
        return bars[0] if bars else None
    
    def symbols(self) -> List[str]:
        """List all symbols"""
        return list(self._data.keys())
    
    def timeframes(self, symbol: str) -> List[str]:
        """List timeframes for symbol"""
        symbol = symbol.upper()
        if symbol not in self._data:
            return []
        return list(self._data[symbol].keys())
    
    def clear(self, symbol: str = None) -> None:
        """Clear buffer"""
        if symbol:
            self._data.pop(symbol.upper(), None)
        else:
            self._data.clear()
    
    def stats(self) -> dict:
        """Buffer statistics"""
        result = {}
        for symbol, timeframes in self._data.items():
            result[symbol] = {tf: len(bars) for tf, bars in timeframes.items()}
        return result

