"""
Live Feed Service
Connects to Binance WebSocket and ingests ticks automatically.

Usage:
    from services import get_live_feed
    
    feed = get_live_feed()
    feed.start(["btcusdt", "ethusdt"])
    # Ticks automatically flow into the ingestion engine
    feed.stop()
"""

import asyncio
import json
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import websockets

from core import get_engine, TickEvent, DataSource


@dataclass
class FeedStats:
    """Live feed statistics"""
    is_running: bool = False
    symbols: List[str] = None
    ticks_received: int = 0
    ticks_per_second: float = 0.0
    last_tick_time: Optional[datetime] = None
    connected_at: Optional[datetime] = None
    errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "symbols": self.symbols or [],
            "ticks_received": self.ticks_received,
            "ticks_per_second": round(self.ticks_per_second, 1),
            "last_tick_time": self.last_tick_time.isoformat() if self.last_tick_time else None,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "uptime_seconds": (datetime.now() - self.connected_at).total_seconds() if self.connected_at else 0,
            "errors": self.errors
        }


class LiveFeedService:
    """
    Binance WebSocket live feed service.
    
    Connects to Binance Futures WebSocket and automatically
    ingests ticks into the IngestionEngine.
    """
    
    BINANCE_WS_URL = "wss://fstream.binance.com/ws"
    
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._websockets: Dict[str, Any] = {}
        self._stats = FeedStats()
        self._engine = None
        self._tick_times: List[float] = []
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def stats(self) -> FeedStats:
        return self._stats
    
    def start(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Start live feed for given symbols.
        
        Args:
            symbols: List of symbols (e.g., ["btcusdt", "ethusdt"])
        
        Returns:
            Status dict
        """
        if self._running:
            return {"status": "already_running", "symbols": self._stats.symbols}
        
        # Normalize symbols
        symbols = [s.lower().strip() for s in symbols if s.strip()]
        if not symbols:
            return {"status": "error", "message": "No symbols provided"}
        
        # Get engine
        self._engine = get_engine()
        
        # Reset stats
        self._stats = FeedStats(
            is_running=True,
            symbols=symbols,
            ticks_received=0,
            connected_at=datetime.now()
        )
        
        # Start background thread
        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop, args=(symbols,), daemon=True)
        self._thread.start()
        
        return {"status": "started", "symbols": symbols}
    
    def stop(self) -> Dict[str, Any]:
        """Stop live feed"""
        if not self._running:
            return {"status": "not_running"}
        
        self._running = False
        self._stats.is_running = False
        
        # Close websockets
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._close_all(), self._loop)
        
        return {
            "status": "stopped",
            "total_ticks": self._stats.ticks_received
        }
    
    def _run_async_loop(self, symbols: List[str]):
        """Run async event loop in background thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connect_all(symbols))
        except Exception as e:
            self._stats.errors += 1
        finally:
            self._loop.close()
            self._running = False
            self._stats.is_running = False
    
    async def _connect_all(self, symbols: List[str]):
        """Connect to all symbol streams"""
        tasks = [self._connect_symbol(sym) for sym in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _connect_symbol(self, symbol: str):
        """Connect to single symbol stream"""
        url = f"{self.BINANCE_WS_URL}/{symbol}@trade"
        
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._websockets[symbol] = ws
                    
                    while self._running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            await self._process_message(message, symbol)
                        except asyncio.TimeoutError:
                            # Send ping to keep alive
                            await ws.ping()
                        except websockets.ConnectionClosed:
                            break
                            
            except Exception as e:
                self._stats.errors += 1
                if self._running:
                    await asyncio.sleep(5)  # Reconnect delay
    
    async def _process_message(self, message: str, symbol: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            if data.get('e') == 'trade':
                # Convert to TickEvent
                tick = TickEvent(
                    symbol=data['s'].upper(),
                    ts=datetime.fromtimestamp(data['T'] / 1000),
                    price=float(data['p']),
                    size=float(data['q']),
                    side='sell' if data['m'] else 'buy',
                    source=DataSource.WEBSOCKET
                )
                
                # Ingest into engine
                if self._engine:
                    self._engine.ingest(tick)
                
                # Update stats
                self._stats.ticks_received += 1
                self._stats.last_tick_time = tick.ts
                
                # Calculate ticks per second
                now = datetime.now().timestamp()
                self._tick_times.append(now)
                self._tick_times = [t for t in self._tick_times if now - t < 1.0]
                self._stats.ticks_per_second = len(self._tick_times)
                
        except Exception as e:
            self._stats.errors += 1
    
    async def _close_all(self):
        """Close all websocket connections"""
        for symbol, ws in self._websockets.items():
            try:
                await ws.close()
            except:
                pass
        self._websockets.clear()


# Singleton
_live_feed: Optional[LiveFeedService] = None


def get_live_feed() -> LiveFeedService:
    """Get or create live feed service singleton"""
    global _live_feed
    if _live_feed is None:
        _live_feed = LiveFeedService()
    return _live_feed

