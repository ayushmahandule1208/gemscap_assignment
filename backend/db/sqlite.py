"""
SQLite Storage
Persistent storage layer.

Responsibilities:
- Write ticks and OHLC bars to database
- Read historical data
- Handle schema

NOT responsible for:
- Validation (done upstream)
- Analytics (done elsewhere)
- Business logic (engine handles this)
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from core.models import TickEvent, OHLCBar, DataSource


class SQLiteStorage:
    """
    SQLite persistence for market data.
    
    Tables:
        - ticks: Raw tick data
        - ohlc: Aggregated OHLC bars
    """
    
    def __init__(self, db_path: str = "data/quant.db"):
        self.db_path = db_path
        self._ensure_directory()
        self._init_schema()
    
    def _ensure_directory(self):
        """Create data directory"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_schema(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL DEFAULT 0,
                    source TEXT DEFAULT 'upload',
                    side TEXT,
                    trade_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts 
                ON ticks(symbol, timestamp);
                
                CREATE TABLE IF NOT EXISTS ohlc (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    trade_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'upload',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_tf_ts 
                ON ohlc(symbol, timeframe, timestamp);
            """)
    
    # =========================================================================
    # Write Operations
    # =========================================================================
    
    def save_ticks(self, ticks: List[TickEvent]) -> int:
        """Save tick events to database"""
        if not ticks:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO ticks 
                   (symbol, timestamp, price, size, source, side, trade_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (t.symbol, t.ts.isoformat(), t.price, t.size,
                     t.source.value, t.side, t.trade_id)
                    for t in ticks
                ]
            )
            return len(ticks)
    
    def save_ohlc_bars(self, bars: List[OHLCBar], timeframe: str) -> int:
        """Save OHLC bars with upsert"""
        if not bars:
            return 0
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO ohlc 
                   (symbol, timeframe, timestamp, open, high, low, close, volume, trade_count, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (b.symbol, timeframe, b.ts.isoformat(),
                     b.open, b.high, b.low, b.close,
                     b.volume, b.trade_count, b.source.value)
                    for b in bars
                ]
            )
            return len(bars)
    
    # =========================================================================
    # Read Operations
    # =========================================================================
    
    def get_ticks(self, symbol: str, limit: int = 10000) -> List[TickEvent]:
        """Read ticks from storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT * FROM ticks 
                   WHERE symbol = ? 
                   ORDER BY timestamp DESC LIMIT ?""",
                [symbol.upper(), limit]
            )
            rows = cursor.fetchall()
        
        return [
            TickEvent(
                symbol=row["symbol"],
                ts=datetime.fromisoformat(row["timestamp"]),
                price=row["price"],
                size=row["size"],
                source=DataSource(row["source"]) if row["source"] else DataSource.UPLOAD,
                side=row["side"],
                trade_id=row["trade_id"]
            )
            for row in reversed(rows)  # Chronological order
        ]
    
    def get_ohlc_bars(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 500
    ) -> List[OHLCBar]:
        """Read OHLC bars from storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT * FROM ohlc 
                   WHERE symbol = ? AND timeframe = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                [symbol.upper(), timeframe, limit]
            )
            rows = cursor.fetchall()
        
        return [
            OHLCBar(
                symbol=row["symbol"],
                ts=datetime.fromisoformat(row["timestamp"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                trade_count=row["trade_count"],
                source=DataSource(row["source"]) if row["source"] else DataSource.UPLOAD
            )
            for row in reversed(rows)
        ]
    
    def get_ticks_df(self, symbol: str, limit: int = 10000) -> pd.DataFrame:
        """Read ticks as DataFrame (for analytics)"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """SELECT timestamp, price, size FROM ticks 
                   WHERE symbol = ? 
                   ORDER BY timestamp DESC LIMIT ?""",
                conn,
                params=[symbol.upper(), limit]
            )
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_ohlc_df(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 500
    ) -> pd.DataFrame:
        """Read OHLC as DataFrame (for analytics)"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """SELECT timestamp, open, high, low, close, volume 
                   FROM ohlc 
                   WHERE symbol = ? AND timeframe = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                conn,
                params=[symbol.upper(), timeframe, limit]
            )
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_symbols(self) -> List[str]:
        """Get all symbols with data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT DISTINCT symbol FROM ticks 
                   UNION 
                   SELECT DISTINCT symbol FROM ohlc 
                   ORDER BY symbol"""
            )
            return [row[0] for row in cursor.fetchall()]
    
    def get_stats(self) -> dict:
        """Get storage statistics"""
        with sqlite3.connect(self.db_path) as conn:
            tick_count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            ohlc_count = conn.execute("SELECT COUNT(*) FROM ohlc").fetchone()[0]
        
        return {
            "tick_count": tick_count,
            "ohlc_count": ohlc_count,
            "symbols": self.get_symbols(),
            "db_path": self.db_path
        }
    
    # =========================================================================
    # Management
    # =========================================================================
    
    def clear(self, symbol: str = None):
        """Clear data"""
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                conn.execute("DELETE FROM ticks WHERE symbol = ?", [symbol.upper()])
                conn.execute("DELETE FROM ohlc WHERE symbol = ?", [symbol.upper()])
            else:
                conn.execute("DELETE FROM ticks")
                conn.execute("DELETE FROM ohlc")


# =============================================================================
# Singleton
# =============================================================================

_storage: Optional[SQLiteStorage] = None


def get_storage() -> SQLiteStorage:
    """Get singleton storage instance"""
    global _storage
    if _storage is None:
        _storage = SQLiteStorage()
    return _storage
