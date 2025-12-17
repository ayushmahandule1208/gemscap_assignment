"""
OHLC Resampler
Converts tick stream to OHLC bars in real-time.

Flow:
1. Tick arrives
2. Find correct time bucket
3. Update building bar (open/high/low/close/volume)
4. If time boundary crossed → emit completed bar
"""

from typing import Dict, Optional, List
from datetime import datetime

from .models import TickEvent, OHLCBar


# =============================================================================
# Timeframe Configuration
# =============================================================================

TIMEFRAME_SECONDS = {
    "1s": 1,
    "5s": 5,
    "10s": 10,
    "30s": 30,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


# =============================================================================
# Resampler
# =============================================================================

class OHLCResampler:
    """
    Real-time OHLC bar builder from tick stream.
    
    Maintains current "building" bar per symbol.
    Emits completed bars when time boundary is crossed.
    
    Usage:
        resampler = OHLCResampler("1m")
        
        for tick in tick_stream:
            completed = resampler.process(tick)
            if completed:
                # Bar finished, do something with it
                save_bar(completed)
    """
    
    def __init__(self, timeframe: str = "1m"):
        self.timeframe = timeframe
        self.interval = TIMEFRAME_SECONDS.get(timeframe, 60)
        
        # Current building bars: symbol -> OHLCBar
        self._building: Dict[str, OHLCBar] = {}
    
    def _get_bar_timestamp(self, ts: datetime) -> datetime:
        """
        Get bar open time for a given timestamp.
        
        Example (1m bars):
            10:00:23 → 10:00:00
            10:01:59 → 10:01:00
        """
        epoch = ts.timestamp()
        bar_epoch = (epoch // self.interval) * self.interval
        return datetime.fromtimestamp(bar_epoch)
    
    def process(self, tick: TickEvent) -> Optional[OHLCBar]:
        """
        Process a single tick.
        
        Returns:
            - Completed bar if time boundary crossed
            - None if bar still building
        """
        symbol = tick.symbol
        bar_ts = self._get_bar_timestamp(tick.ts)
        
        current = self._building.get(symbol)
        
        # =====================================================================
        # Case 1: No current bar OR new time period
        # =====================================================================
        if current is None or current.ts != bar_ts:
            completed = None
            
            # If we had a bar, it's now complete
            if current is not None:
                completed = current
            
            # Start new bar
            self._building[symbol] = OHLCBar(
                symbol=symbol,
                ts=bar_ts,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.size,
                trade_count=1,
                source=tick.source
            )
            
            return completed
        
        # =====================================================================
        # Case 2: Same time period — update current bar
        # =====================================================================
        current.high = max(current.high, tick.price)
        current.low = min(current.low, tick.price)
        current.close = tick.price
        current.volume += tick.size
        current.trade_count += 1
        
        return None
    
    def process_batch(self, ticks: List[TickEvent]) -> List[OHLCBar]:
        """
        Process batch of ticks.
        Returns all completed bars.
        """
        completed = []
        for tick in ticks:
            bar = self.process(tick)
            if bar:
                completed.append(bar)
        return completed
    
    def get_building(self, symbol: str) -> Optional[OHLCBar]:
        """Get current building bar (incomplete)"""
        return self._building.get(symbol.upper())
    
    def flush(self, symbol: str = None) -> List[OHLCBar]:
        """
        Force-complete building bars.
        Useful at end of session or file.
        
        Returns flushed bars.
        """
        if symbol:
            symbol = symbol.upper()
            if symbol in self._building:
                bar = self._building.pop(symbol)
                return [bar]
            return []
        
        # Flush all
        bars = list(self._building.values())
        self._building.clear()
        return bars
    
    def clear(self, symbol: str = None) -> None:
        """Clear resampler state"""
        if symbol:
            self._building.pop(symbol.upper(), None)
        else:
            self._building.clear()


# =============================================================================
# Batch Resampling (for file uploads)
# =============================================================================

def resample_ticks(
    ticks: List[TickEvent],
    timeframe: str = "1m"
) -> List[OHLCBar]:
    """
    Batch resample ticks to OHLC bars.
    
    Stateless utility function for file uploads.
    Sorts ticks by timestamp first.
    """
    if not ticks:
        return []
    
    # Sort by timestamp
    sorted_ticks = sorted(ticks, key=lambda t: t.ts)
    
    # Process through resampler
    resampler = OHLCResampler(timeframe)
    completed = resampler.process_batch(sorted_ticks)
    
    # Flush final building bar
    final = resampler.flush()
    
    return completed + final

