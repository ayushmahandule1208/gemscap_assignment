from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import numpy as np

from core import get_engine
from analytics import live, batch, pair
from alerts import get_alert_engine

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/{symbol}/stats")
async def get_price_stats(
    symbol: str,
    window: int = Query(default=20)
):
    engine = get_engine()
    prices = engine.get_prices(symbol, limit=window * 2)
    
    if not prices or len(prices) < 2:
        raise HTTPException(404, f"Insufficient data for {symbol}")
    
    prices_arr = np.array(prices)
    stats = live.price_stats(prices_arr, window=window)
    
    return {
        "symbol": symbol.upper(),
        "window": window,
        "stats": {
            "last_price": stats.last_price,
            "rolling_mean": round(stats.rolling_mean, 4),
            "rolling_std": round(stats.rolling_std, 4),
            "min_price": stats.min_price,
            "max_price": stats.max_price,
            "vwap": stats.vwap
        }
    }


@router.get("/spread")
async def get_live_spread(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    hedge_ratio: Optional[float] = Query(None),
    window: int = Query(default=20)
):
    engine = get_engine()
    
    prices_a = engine.get_prices(symbol_a, limit=window * 5)
    prices_b = engine.get_prices(symbol_b, limit=window * 5)
    
    if not prices_a or not prices_b:
        raise HTTPException(404, "Insufficient data for one or both symbols")
    
    arr_a = np.array(prices_a)
    arr_b = np.array(prices_b)
    
    min_len = min(len(arr_a), len(arr_b))
    arr_a = arr_a[-min_len:]
    arr_b = arr_b[-min_len:]
    
    if hedge_ratio is None:
        hr = batch.hedge_ratio(arr_a, arr_b)
        hedge_ratio = hr.beta
    
    spread_data = batch.spread_stats(arr_a, arr_b, hedge_ratio, window)
    
    result = live.spread(
        price_a=float(arr_a[-1]),
        price_b=float(arr_b[-1]),
        hedge_ratio=hedge_ratio,
        spread_mean=spread_data['mean'],
        spread_std=spread_data['std']
    )
    
    alert_engine = get_alert_engine()
    analytics_data = {"z_score": result.z_score, "spread": result.spread}
    alert_engine.evaluate(analytics_data, symbols=[symbol_a.upper(), symbol_b.upper()])
    
    return {
        "pair": f"{symbol_a.upper()}/{symbol_b.upper()}",
        "live": {
            "spread": round(result.spread, 4),
            "z_score": round(result.z_score, 4),
        },
        "params": {
            "hedge_ratio": round(hedge_ratio, 4),
            "spread_mean": round(spread_data['mean'], 4),
            "spread_std": round(spread_data['std'], 4),
        }
    }


@router.get("/hedge-ratio")
async def compute_hedge_ratio(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=100),
    method: str = Query(default="ols")
):
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a, timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b, timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data for one or both symbols")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    hr = batch.hedge_ratio(prices_a, prices_b, method=method)
    
    return {
        "pair": f"{symbol_a.upper()}/{symbol_b.upper()}",
        "timeframe": timeframe,
        "bars_used": min_len,
        "method": method,
        "hedge_ratio": {
            "beta": round(hr.beta, 6),
            "alpha": round(hr.alpha, 4),
            "r_squared": round(hr.r_squared, 4),
            "std_error": round(hr.std_error, 6)
        }
    }


@router.get("/adf")
async def run_adf_test(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=100)
):
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a, timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b, timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    hr = batch.hedge_ratio(prices_a, prices_b)
    spread = prices_a - hr.beta * prices_b
    adf_result = batch.adf_test(spread)
    
    return {
        "pair": f"{symbol_a.upper()}/{symbol_b.upper()}",
        "timeframe": timeframe,
        "bars_used": min_len,
        "adf": {
            "statistic": adf_result.statistic,
            "p_value": adf_result.p_value,
            "is_stationary": adf_result.is_stationary,
            "critical_values": adf_result.critical_values
        },
        "interpretation": "Spread IS stationary (mean-reverting)" if adf_result.is_stationary else "Spread is NOT stationary"
    }


@router.get("/half-life")
async def compute_half_life(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=100)
):
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a, timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b, timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    hr = batch.hedge_ratio(prices_a, prices_b)
    spread = prices_a - hr.beta * prices_b
    hl = batch.half_life(spread)
    
    return {
        "pair": f"{symbol_a.upper()}/{symbol_b.upper()}",
        "timeframe": timeframe,
        "bars_used": min_len,
        "half_life": {
            "periods": round(hl.half_life, 2) if hl.half_life != float('inf') else None,
            "lambda": round(hl.lambda_coef, 6),
            "is_mean_reverting": hl.is_mean_reverting,
            "r_squared": round(hl.r_squared, 4)
        }
    }


@router.get("/pair")
async def full_pair_analysis(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=100),
    window: int = Query(default=20),
    z_threshold: float = Query(default=2.0)
):
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a, timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b, timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data for pair analysis")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    if min_len < 5:
        raise HTTPException(400, f"Need at least 5 bars, got {min_len}")
    
    result = pair.full_analysis(
        prices_a=prices_a,
        prices_b=prices_b,
        symbol_a=symbol_a.upper(),
        symbol_b=symbol_b.upper(),
        window=window,
        z_threshold=z_threshold
    )
    
    result['timeframe'] = timeframe
    result['bars_used'] = min_len
    
    return result


@router.get("/kalman-hedge")
async def kalman_hedge_ratio(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=200),
    delta: float = Query(default=1e-4)
):
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a, timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b, timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    result = batch.kalman_hedge_ratio(prices_a, prices_b, delta=delta)
    
    return {
        "pair": f"{symbol_a.upper()}/{symbol_b.upper()}",
        "timeframe": timeframe,
        "bars_used": min_len,
        "kalman": {
            "current_hedge_ratio": result['current'],
            "mean_hedge_ratio": result['mean'],
            "std": result['std'],
            "is_stable": result['is_stable'],
            "last_10": [round(x, 4) for x in result['hedge_ratios'][-10:].tolist()]
        }
    }


@router.get("/robust-hedge")
async def robust_hedge_ratio(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=100),
    method: str = Query(default="huber")
):
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a, timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b, timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    result = batch.robust_hedge_ratio(prices_a, prices_b, method=method)
    
    return {
        "pair": f"{symbol_a.upper()}/{symbol_b.upper()}",
        "timeframe": timeframe,
        "bars_used": min_len,
        "robust": result
    }


@router.get("/backtest")
async def run_backtest(
    symbol_a: str = Query(...),
    symbol_b: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=500),
    entry_z: float = Query(default=2.0),
    exit_z: float = Query(default=0.0),
    window: int = Query(default=20)
):
    engine = get_engine()
    
    bars_a = engine.get_ohlc(symbol_a, timeframe, limit)
    bars_b = engine.get_ohlc(symbol_b, timeframe, limit)
    
    if not bars_a or not bars_b:
        raise HTTPException(404, "Insufficient OHLC data")
    
    prices_a = np.array([b.close for b in bars_a])
    prices_b = np.array([b.close for b in bars_b])
    
    min_len = min(len(prices_a), len(prices_b))
    prices_a = prices_a[-min_len:]
    prices_b = prices_b[-min_len:]
    
    if min_len < 30:
        raise HTTPException(400, f"Need at least 30 bars for backtest, got {min_len}")
    
    result = batch.mini_backtest(
        prices_a, prices_b,
        entry_z=entry_z,
        exit_z=exit_z,
        window=window
    )
    
    return {
        "pair": f"{symbol_a.upper()}/{symbol_b.upper()}",
        "timeframe": timeframe,
        "bars_used": min_len,
        "backtest": result
    }


@router.get("/correlation-matrix")
async def correlation_matrix(
    symbols: str = Query(...),
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=100)
):
    engine = get_engine()
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    if len(symbol_list) < 2:
        raise HTTPException(400, "Need at least 2 symbols")
    
    price_dict = {}
    for sym in symbol_list:
        bars = engine.get_ohlc(sym, timeframe, limit)
        if bars:
            price_dict[sym] = np.array([b.close for b in bars])
    
    if len(price_dict) < 2:
        raise HTTPException(404, "Insufficient data for correlation matrix")
    
    result = batch.cross_correlation_matrix(price_dict)
    result['timeframe'] = timeframe
    
    return result


@router.get("/{symbol}/liquidity")
async def check_liquidity(
    symbol: str,
    timeframe: str = Query(default="1m"),
    limit: int = Query(default=100),
    min_volume: float = Query(default=None),
    min_dollar_volume: float = Query(default=None),
    max_spread_pct: float = Query(default=None)
):
    engine = get_engine()
    bars = engine.get_ohlc(symbol, timeframe, limit)
    
    if not bars:
        raise HTTPException(404, f"No OHLC data for {symbol}")
    
    volumes = np.array([b.volume for b in bars])
    prices = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    result = batch.liquidity_filter(
        volumes=volumes,
        prices=prices,
        highs=highs,
        lows=lows,
        min_avg_volume=min_volume,
        min_dollar_volume=min_dollar_volume,
        max_spread_pct=max_spread_pct
    )
    
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "bars_used": len(bars),
        "liquidity": result
    }
