"""
Data Export API
Download endpoints for processed data and analytics outputs.

Formats:
    - CSV (default) — Excel/pandas compatible
    - JSON — For programmatic access

Exports:
    - Tick data
    - OHLC data  
    - Analytics results
    - Alert history
"""

import io
import csv
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import numpy as np

from core import get_engine
from analytics import batch, pair
from alerts import get_alert_engine

router = APIRouter(prefix="/export", tags=["Export"])


# =============================================================================
# Tick Data Export
# =============================================================================

@router.get("/ticks/{symbol}")
async def export_ticks(
    symbol: str,
    format: str = Query(default="csv", description="csv or json"),
    limit: int = Query(default=10000, le=100000)
):
    """
    Export tick data for a symbol.
    
    Returns:
        CSV or JSON file download
    """
    engine = get_engine()
    ticks = engine.get_ticks(symbol.upper(), limit)
    
    if not ticks:
        raise HTTPException(404, f"No tick data for {symbol}")
    
    filename = f"ticks_{symbol.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if format == "json":
        data = [
            {
                "timestamp": t.ts.isoformat(),
                "symbol": t.symbol,
                "price": t.price,
                "size": t.size,
                "side": t.side
            }
            for t in ticks
        ]
        content = json.dumps(data, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}.json"}
        )
    
    # CSV format
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "symbol", "price", "size", "side"])
    
    for t in ticks:
        writer.writerow([t.ts.isoformat(), t.symbol, t.price, t.size, t.side or ""])
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
    )


# =============================================================================
# OHLC Data Export
# =============================================================================

@router.get("/ohlc/{symbol}")
async def export_ohlc(
    symbol: str,
    timeframe: str = Query(default="1m", description="1s, 1m, or 5m"),
    format: str = Query(default="csv", description="csv or json"),
    limit: int = Query(default=5000, le=50000)
):
    """
    Export OHLC data for a symbol.
    
    Returns:
        CSV or JSON file with OHLC bars
    """
    engine = get_engine()
    bars = engine.get_ohlc(symbol.upper(), timeframe, limit)
    
    if not bars:
        raise HTTPException(404, f"No OHLC data for {symbol}")
    
    filename = f"ohlc_{symbol.upper()}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if format == "json":
        data = [
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
        content = json.dumps(data, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}.json"}
        )
    
    # CSV format
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
    
    for b in bars:
        writer.writerow([b.ts.isoformat(), b.open, b.high, b.low, b.close, b.volume])
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
    )


# =============================================================================
# Analytics Export
# =============================================================================

@router.get("/analytics/pair")
async def export_pair_analytics(
    symbol_a: str = Query(..., description="First symbol"),
    symbol_b: str = Query(..., description="Second symbol"),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=500),
    format: str = Query(default="json", description="json or csv")
):
    """
    Export full pair analytics results.
    
    Includes: hedge ratio, spread stats, ADF, half-life, correlation
    """
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a.upper(), timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b.upper(), timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    # Run analysis
    result = pair.full_analysis(
        prices_a, prices_b,
        symbol_a.upper(), symbol_b.upper()
    )
    result['timeframe'] = timeframe
    result['bars_used'] = min_len
    result['export_time'] = datetime.now().isoformat()
    
    filename = f"analytics_{symbol_a.upper()}_{symbol_b.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if format == "json":
        content = json.dumps(result, indent=2, default=str)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}.json"}
        )
    
    # CSV format (flattened)
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Flatten nested dict
    rows = []
    def flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flatten(v, f"{key}_")
            else:
                rows.append((key, v))
    
    flatten(result)
    writer.writerow(["metric", "value"])
    for key, value in rows:
        writer.writerow([key, value])
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
    )


@router.get("/analytics/spread")
async def export_spread_series(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=500),
    format: str = Query(default="csv")
):
    """
    Export spread and z-score time series.
    
    Useful for plotting and further analysis.
    """
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a.upper(), timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b.upper(), timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data")
    
    # Align timestamps
    times_a = {b.ts: b for b in bars_a}
    times_b = {b.ts: b for b in bars_b}
    common_times = sorted(set(times_a.keys()) & set(times_b.keys()))
    
    if len(common_times) < 20:
        raise HTTPException(400, "Insufficient overlapping data")
    
    prices_a = np.array([times_a[t].close for t in common_times])
    prices_b = np.array([times_b[t].close for t in common_times])
    
    # Compute spread stats
    spread_data = batch.spread_stats(prices_a, prices_b)
    
    filename = f"spread_{symbol_a.upper()}_{symbol_b.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if format == "json":
        data = [
            {
                "timestamp": common_times[i].isoformat(),
                "price_a": float(prices_a[i]),
                "price_b": float(prices_b[i]),
                "spread": float(spread_data['spread'][i]),
                "z_score": float(spread_data['z_scores'][i]),
                "rolling_mean": float(spread_data['rolling_mean'][i]),
                "rolling_std": float(spread_data['rolling_std'][i])
            }
            for i in range(len(common_times))
        ]
        content = json.dumps(data, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}.json"}
        )
    
    # CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "price_a", "price_b", "spread", "z_score", "rolling_mean", "rolling_std"])
    
    for i in range(len(common_times)):
        writer.writerow([
            common_times[i].isoformat(),
            prices_a[i],
            prices_b[i],
            spread_data['spread'][i],
            spread_data['z_scores'][i],
            spread_data['rolling_mean'][i],
            spread_data['rolling_std'][i]
        ])
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
    )


# =============================================================================
# Alert History Export
# =============================================================================

@router.get("/alerts")
async def export_alerts(
    format: str = Query(default="csv", description="csv or json"),
    limit: int = Query(default=200)
):
    """
    Export alert history.
    
    Includes all triggered alerts with timestamps and values.
    """
    alert_engine = get_alert_engine()
    history = alert_engine.get_history(limit)
    
    if not history:
        raise HTTPException(404, "No alert history")
    
    filename = f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if format == "json":
        data = [e.to_dict() for e in history]
        content = json.dumps(data, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}.json"}
        )
    
    # CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "rule_id", "metric", "value", "threshold", "operator", "symbols", "message", "severity"])
    
    for e in history:
        writer.writerow([
            e.timestamp.isoformat(),
            e.rule_id,
            e.metric,
            e.value,
            e.threshold,
            e.operator,
            "|".join(e.symbols),
            e.message,
            e.severity.value
        ])
    
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
    )


# =============================================================================
# Backtest Export
# =============================================================================

@router.get("/backtest")
async def export_backtest(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=500),
    entry_z: float = Query(default=2.0),
    exit_z: float = Query(default=0.0),
    format: str = Query(default="json")
):
    """
    Export backtest results.
    
    Includes trade list, equity curve, and performance metrics.
    """
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a.upper(), timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b.upper(), timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    result = batch.mini_backtest(
        prices_a, prices_b,
        entry_z=entry_z,
        exit_z=exit_z
    )
    
    result['pair'] = f"{symbol_a.upper()}/{symbol_b.upper()}"
    result['timeframe'] = timeframe
    result['bars_used'] = min_len
    result['export_time'] = datetime.now().isoformat()
    
    filename = f"backtest_{symbol_a.upper()}_{symbol_b.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    content = json.dumps(result, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}.json"}
    )

