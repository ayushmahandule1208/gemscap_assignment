from typing import List, Dict, Optional, Callable, Any
from datetime import datetime

from .models import TickEvent, OHLCBar, DataSource, IngestionResult
from .buffer import LiveBuffer, OHLCBuffer
from .resampler import OHLCResampler
from db import get_storage

OnTickCallback = Callable[[TickEvent], None]
OnBarCallback = Callable[[OHLCBar, str], None]


class IngestionEngine:
    def __init__(
        self,
        buffer_size: int = 10000,
        ohlc_buffer_size: int = 1000,
        timeframes: List[str] = None,
        persist: bool = True
    ):
        self._tick_buffer = LiveBuffer(maxlen=buffer_size)
        self._ohlc_buffer = OHLCBuffer(maxlen=ohlc_buffer_size)
        self._timeframes = timeframes or ["1m"]
        self._resamplers: Dict[str, OHLCResampler] = {
            tf: OHLCResampler(tf) for tf in self._timeframes
        }
        self._persist = persist
        self._storage = get_storage() if persist else None
        self._on_tick: List[OnTickCallback] = []
        self._on_bar: List[OnBarCallback] = []
        self._stats = {
            "ticks_ingested": 0,
            "bars_created": 0,
            "bars_uploaded": 0,
            "errors": 0,
            "start_time": datetime.now()
        }
    
    def ingest(self, tick: TickEvent) -> None:
        self._process_tick(tick)
    
    def ingest_batch(self, ticks: List[TickEvent]) -> IngestionResult:
        if not ticks:
            return IngestionResult(success=True, count=0, message="No ticks")
        
        errors = 0
        for tick in ticks:
            try:
                self._process_tick(tick)
            except Exception:
                errors += 1
                self._stats["errors"] += 1
        
        symbols = list(set(t.symbol for t in ticks))
        return IngestionResult(
            success=errors == 0,
            count=len(ticks) - errors,
            errors=errors,
            symbols=symbols,
            message=f"Ingested {len(ticks) - errors} ticks"
        )
    
    def ingest_ohlc(self, bars: List[OHLCBar], timeframe: str = "1m") -> IngestionResult:
        if not bars:
            return IngestionResult(success=True, count=0, message="No bars")
        
        for bar in bars:
            self._ohlc_buffer.append(bar.symbol, timeframe, bar)
        
        if self._storage:
            self._storage.save_ohlc_bars(bars, timeframe)
        
        self._stats["bars_uploaded"] += len(bars)
        symbols = list(set(b.symbol for b in bars))
        
        return IngestionResult(
            success=True,
            count=len(bars),
            symbols=symbols,
            message=f"Ingested {len(bars)} OHLC bars"
        )
    
    def get_ticks(self, symbol: str, limit: int = 1000) -> List[TickEvent]:
        return self._tick_buffer.get(symbol, limit)
    
    def get_prices(self, symbol: str, limit: int = 1000) -> List[float]:
        return self._tick_buffer.get_prices(symbol, limit)
    
    def get_ohlc(self, symbol: str, timeframe: str = "1m", limit: int = 500) -> List[OHLCBar]:
        bars = self._ohlc_buffer.get(symbol, timeframe, limit)
        if not bars and self._storage:
            bars = self._storage.get_ohlc_bars(symbol, timeframe, limit)
        return bars
    
    def get_symbols(self) -> List[str]:
        buffer_symbols = set(self._tick_buffer.symbols())
        ohlc_symbols = set(self._ohlc_buffer.symbols())
        if self._storage:
            storage_symbols = set(self._storage.get_symbols())
            return sorted(buffer_symbols | ohlc_symbols | storage_symbols)
        return sorted(buffer_symbols | ohlc_symbols)
    
    def on_tick(self, callback: OnTickCallback) -> None:
        self._on_tick.append(callback)
    
    def on_bar(self, callback: OnBarCallback) -> None:
        self._on_bar.append(callback)
    
    def clear(self, symbol: str = None) -> None:
        self._tick_buffer.clear(symbol)
        self._ohlc_buffer.clear(symbol)
        for resampler in self._resamplers.values():
            resampler.clear(symbol)
        if self._storage:
            self._storage.clear(symbol)
        if not symbol:
            self._stats = {
                "ticks_ingested": 0,
                "bars_created": 0,
                "bars_uploaded": 0,
                "errors": 0,
                "start_time": datetime.now()
            }
    
    def stats(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self._stats["start_time"]).total_seconds()
        return {
            **self._stats,
            "uptime_seconds": uptime,
            "tick_buffer": self._tick_buffer.stats(),
            "ohlc_buffer": self._ohlc_buffer.stats(),
            "symbols": self.get_symbols(),
            "timeframes": self._timeframes,
        }
    
    def get_data_availability(self) -> Dict[str, Any]:
        symbols = self.get_symbols()
        availability = {}
        
        for symbol in symbols:
            symbol_data = {
                "ticks": len(self._tick_buffer.get(symbol, 10000)),
                "bars": {}
            }
            for tf in self._timeframes:
                bars = self.get_ohlc(symbol, tf, 10000)
                symbol_data["bars"][tf] = len(bars)
            availability[symbol] = symbol_data
        
        feature_requirements = {
            "price_chart": {"min_bars": 1, "description": "Price visualization"},
            "basic_stats": {"min_bars": 2, "description": "Rolling mean, std, min, max"},
            "spread_analysis": {"min_bars": 5, "description": "Spread and Z-score"},
            "hedge_ratio": {"min_bars": 10, "description": "OLS/Robust hedge ratio"},
            "adf_test": {"min_bars": 20, "description": "ADF stationarity test"},
            "half_life": {"min_bars": 20, "description": "Mean reversion half-life"},
            "correlation": {"min_bars": 5, "description": "Correlation analysis"},
            "backtest": {"min_bars": 30, "description": "Strategy backtesting"},
            "alerts": {"min_bars": 1, "description": "Alert monitoring"},
        }
        
        features_status = {}
        if len(symbols) >= 2:
            sym1, sym2 = symbols[0], symbols[1]
            bars1 = availability.get(sym1, {}).get("bars", {}).get("1m", 0)
            bars2 = availability.get(sym2, {}).get("bars", {}).get("1m", 0)
            min_bars = min(bars1, bars2)
            
            for feature, req in feature_requirements.items():
                features_status[feature] = {
                    "unlocked": min_bars >= req["min_bars"],
                    "required": req["min_bars"],
                    "current": min_bars,
                    "description": req["description"],
                    "progress": min(100, int(min_bars / req["min_bars"] * 100)) if req["min_bars"] > 0 else 100
                }
        else:
            bars = availability.get(symbols[0], {}).get("bars", {}).get("1m", 0) if symbols else 0
            for feature, req in feature_requirements.items():
                features_status[feature] = {
                    "unlocked": bars >= req["min_bars"],
                    "required": req["min_bars"],
                    "current": bars,
                    "description": req["description"],
                    "progress": min(100, int(bars / req["min_bars"] * 100)) if req["min_bars"] > 0 else 100
                }
        
        return {
            "symbols": availability,
            "features": features_status,
            "requirements": feature_requirements,
            "total_symbols": len(symbols),
            "has_pair_data": len(symbols) >= 2
        }
    
    def _process_tick(self, tick: TickEvent) -> None:
        self._tick_buffer.append(tick)
        self._stats["ticks_ingested"] += 1
        
        for timeframe, resampler in self._resamplers.items():
            completed_bar = resampler.process(tick)
            if completed_bar:
                self._ohlc_buffer.append(tick.symbol, timeframe, completed_bar)
                self._stats["bars_created"] += 1
                if self._storage:
                    self._storage.save_ohlc_bars([completed_bar], timeframe)
                for callback in self._on_bar:
                    try:
                        callback(completed_bar, timeframe)
                    except Exception:
                        pass
        
        for callback in self._on_tick:
            try:
                callback(tick)
            except Exception:
                pass


_engine: Optional[IngestionEngine] = None


def get_engine() -> IngestionEngine:
    global _engine
    if _engine is None:
        _engine = IngestionEngine(
            buffer_size=10000,
            ohlc_buffer_size=1000,
            timeframes=["1s", "1m", "5m"],
            persist=True
        )
    return _engine
