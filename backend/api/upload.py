from fastapi import APIRouter, HTTPException, UploadFile, File, Query
import pandas as pd
from io import StringIO
import json
from typing import List

from core import (
    TickEvent, OHLCBar, DataSource,
    IngestionResult, TickBatch,
    to_tick_event, to_ohlc_bar,
    get_engine
)

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/csv", response_model=IngestionResult)
async def upload_csv(
    file: UploadFile = File(...),
    symbol: str = Query(..., description="Symbol (e.g., BTCUSDT)")
):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')))
    df.columns = df.columns.str.lower().str.strip()
    
    engine = get_engine()
    
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        bars = _parse_ohlc_csv(df, symbol)
        return engine.ingest_ohlc(bars, timeframe="1m")
    else:
        ticks = _parse_tick_csv(df, symbol)
        return engine.ingest_batch(ticks)


def _parse_tick_csv(df: pd.DataFrame, symbol: str) -> List[TickEvent]:
    if 'ts' in df.columns and 'timestamp' not in df.columns:
        df = df.rename(columns={'ts': 'timestamp'})
    if 'qty' in df.columns and 'size' not in df.columns:
        df = df.rename(columns={'qty': 'size'})
    
    if 'timestamp' not in df.columns or 'price' not in df.columns:
        raise HTTPException(400, "CSV must have 'timestamp' and 'price' columns")
    
    ticks = []
    for _, row in df.iterrows():
        try:
            tick = to_tick_event({
                'symbol': symbol,
                'timestamp': str(row['timestamp']),
                'price': row['price'],
                'size': row.get('size', 0),
                'side': row.get('side')
            }, source=DataSource.UPLOAD)
            ticks.append(tick)
        except Exception:
            continue
    
    return ticks


def _parse_ohlc_csv(df: pd.DataFrame, symbol: str) -> List[OHLCBar]:
    ts_col = next(
        (c for c in ['timestamp', 'ts', 'time', 'datetime', 'date'] if c in df.columns),
        None
    )
    if not ts_col:
        raise HTTPException(400, "OHLC CSV must have timestamp column")
    
    bars = []
    for _, row in df.iterrows():
        try:
            bar = to_ohlc_bar({
                'symbol': symbol,
                'timestamp': str(row[ts_col]),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row.get('volume', 0)
            }, source=DataSource.UPLOAD)
            bars.append(bar)
        except Exception:
            continue
    
    return bars


@router.post("/ndjson", response_model=IngestionResult)
async def upload_ndjson(
    file: UploadFile = File(...),
    symbol: str = Query(default=None, description="Override symbol")
):
    content = await file.read()
    lines = content.decode('utf-8').strip().split('\n')
    
    ticks = []
    errors = 0
    
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if symbol:
                data['symbol'] = symbol
            tick = to_tick_event(data, source=DataSource.UPLOAD)
            ticks.append(tick)
        except Exception:
            errors += 1
    
    if not ticks:
        raise HTTPException(400, "No valid records in file")
    
    engine = get_engine()
    result = engine.ingest_batch(ticks)
    result.errors += errors
    
    return result


@router.post("/ticks", response_model=IngestionResult)
async def upload_tick_batch(payload: TickBatch):
    ticks = []
    errors = 0
    
    for data in payload.ticks:
        try:
            tick = to_tick_event(data, source=DataSource.API)
            ticks.append(tick)
        except Exception:
            errors += 1
    
    if not ticks:
        raise HTTPException(400, "No valid ticks")
    
    engine = get_engine()
    result = engine.ingest_batch(ticks)
    result.errors += errors
    
    return result


@router.post("/tick", response_model=IngestionResult)
async def upload_single_tick(
    symbol: str = Query(...),
    price: float = Query(...),
    timestamp: str = Query(default=None),
    size: float = Query(default=0.0),
    side: str = Query(default=None)
):
    tick = to_tick_event({
        'symbol': symbol,
        'timestamp': timestamp,
        'price': price,
        'size': size,
        'side': side
    }, source=DataSource.WEBSOCKET)
    
    engine = get_engine()
    engine.ingest(tick)
    
    return IngestionResult(
        success=True,
        count=1,
        symbols=[tick.symbol],
        message="Tick ingested"
    )
