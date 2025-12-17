"""
Live Feed API
Endpoints to control live Binance data streaming.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services import get_live_feed


router = APIRouter(prefix="/live", tags=["Live Feed"])


class StartFeedRequest(BaseModel):
    """Request to start live feed"""
    symbols: List[str] = ["btcusdt", "ethusdt"]


class FeedResponse(BaseModel):
    """Response for feed operations"""
    status: str
    symbols: Optional[List[str]] = None
    message: Optional[str] = None
    total_ticks: Optional[int] = None


@router.post("/start", response_model=FeedResponse)
async def start_live_feed(request: StartFeedRequest):
    """
    Start live data feed from Binance.
    
    Connects to Binance WebSocket and automatically ingests
    tick data into the system.
    """
    feed = get_live_feed()
    result = feed.start(request.symbols)
    return result


@router.post("/stop", response_model=FeedResponse)
async def stop_live_feed():
    """Stop live data feed"""
    feed = get_live_feed()
    result = feed.stop()
    return result


@router.get("/status")
async def get_feed_status():
    """
    Get current status of live feed.
    
    Returns:
        Feed statistics including tick count, rate, uptime
    """
    feed = get_live_feed()
    return {
        "status": "running" if feed.is_running else "stopped",
        **feed.stats.to_dict()
    }


@router.get("/symbols")
async def get_available_symbols():
    """
    Get list of commonly used trading symbols.
    """
    return {
        "popular": [
            "btcusdt",
            "ethusdt",
            "bnbusdt",
            "solusdt",
            "xrpusdt",
            "dogeusdt",
            "adausdt",
            "avaxusdt",
            "dotusdt",
            "linkusdt"
        ],
        "pairs": [
            ["btcusdt", "ethusdt"],
            ["btcusdt", "bnbusdt"],
            ["ethusdt", "bnbusdt"],
            ["solusdt", "ethusdt"]
        ]
    }

