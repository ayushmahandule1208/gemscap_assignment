from fastapi import APIRouter, HTTPException, Query
from typing import List

from core import get_engine, OHLCBar

router = APIRouter(prefix="/data", tags=["Data"])


@router.get("/symbols")
async def list_symbols():
    engine = get_engine()
    return {
        "symbols": engine.get_symbols(),
        "stats": engine.stats()
    }


@router.get("/stats")
async def get_stats():
    engine = get_engine()
    return engine.stats()


@router.get("/availability")
async def get_data_availability():
    engine = get_engine()
    return engine.get_data_availability()


@router.get("/availability/{symbol}")
async def get_symbol_availability(
    symbol: str,
    timeframe: str = Query(default="1m")
):
    engine = get_engine()
    bars = engine.get_ohlc(symbol, timeframe, 10000)
    ticks = engine.get_ticks(symbol, 10000)
    
    bar_count = len(bars)
    tick_count = len(ticks)
    
    features = {
        "price_chart": {"required": 1, "unlocked": bar_count >= 1},
        "basic_stats": {"required": 5, "unlocked": bar_count >= 5},
        "spread_analysis": {"required": 20, "unlocked": bar_count >= 20},
        "hedge_ratio": {"required": 30, "unlocked": bar_count >= 30},
        "adf_test": {"required": 50, "unlocked": bar_count >= 50},
        "half_life": {"required": 50, "unlocked": bar_count >= 50},
        "backtest": {"required": 100, "unlocked": bar_count >= 100},
    }
    
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "bars": bar_count,
        "ticks": tick_count,
        "features": features
    }


@router.get("/{symbol}/ticks")
async def get_ticks(
    symbol: str,
    limit: int = Query(default=1000)
):
    engine = get_engine()
    ticks = engine.get_ticks(symbol, limit)
    
    if not ticks:
        raise HTTPException(404, f"No tick data for {symbol}")
    
    return {
        "symbol": symbol.upper(),
        "count": len(ticks),
        "data": [
            {
                "timestamp": t.ts.isoformat(),
                "price": t.price,
                "size": t.size,
                "side": t.side
            }
            for t in ticks
        ]
    }


@router.get("/{symbol}/prices")
async def get_prices(
    symbol: str,
    limit: int = Query(default=1000)
):
    engine = get_engine()
    prices = engine.get_prices(symbol, limit)
    
    if not prices:
        raise HTTPException(404, f"No price data for {symbol}")
    
    return {
        "symbol": symbol.upper(),
        "count": len(prices),
        "prices": prices
    }


@router.get("/{symbol}/ohlc")
async def get_ohlc(
    symbol: str,
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=500)
):
    engine = get_engine()
    bars = engine.get_ohlc(symbol, timeframe, limit)
    
    if not bars:
        raise HTTPException(404, f"No OHLC data for {symbol}")
    
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "count": len(bars),
        "data": [
            {
                "timestamp": b.ts.isoformat(),
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume
            }
            for b in bars
        ]
    }


@router.post("/clear")
async def clear_data(symbol: str = Query(default=None)):
    engine = get_engine()
    engine.clear(symbol)
    return {"message": f"Cleared {'all data' if not symbol else f'data for {symbol}'}"}
