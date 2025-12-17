"""
Alerts API
Endpoints for managing alert rules and streaming events.

Endpoints:
    POST   /api/alerts/rules        → Create alert rule
    GET    /api/alerts/rules        → List all rules
    GET    /api/alerts/rules/{id}   → Get rule by ID
    DELETE /api/alerts/rules/{id}   → Delete rule
    POST   /api/alerts/rules/{id}/enable   → Enable rule
    POST   /api/alerts/rules/{id}/disable  → Disable rule
    GET    /api/alerts/history      → Get alert history
    GET    /api/alerts/stream       → SSE stream for real-time alerts
    POST   /api/alerts/test         → Test evaluate with mock data
"""

import asyncio
import json
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from alerts import (
    get_alert_engine,
    AlertRule,
    AlertMetric,
    AlertOperator,
)

router = APIRouter(prefix="/alerts", tags=["Alerts"])


# =============================================================================
# Request Models
# =============================================================================

class CreateAlertRequest(BaseModel):
    """Request body for creating an alert"""
    symbols: List[str]
    metric: str  # z_score, spread, correlation, volume, price
    operator: str  # >, <, >=, <=, abs>
    threshold: float
    cooldown_sec: int = 60
    name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "metric": "z_score",
                "operator": "abs>",
                "threshold": 2.0,
                "cooldown_sec": 60,
                "name": "Z-score threshold alert"
            }
        }


class TestAlertRequest(BaseModel):
    """Request body for testing alerts"""
    z_score: Optional[float] = None
    spread: Optional[float] = None
    correlation: Optional[float] = None
    volume: Optional[float] = None
    price: Optional[float] = None
    symbols: List[str] = ["TEST"]


# =============================================================================
# Rule Management
# =============================================================================

@router.post("/rules")
async def create_rule(request: CreateAlertRequest):
    """
    Create a new alert rule.
    
    Metrics: z_score, spread, correlation, volume, price, half_life
    Operators: > (gt), < (lt), >= (gte), <= (lte), abs> (absolute value)
    """
    engine = get_alert_engine()
    
    # Validate metric
    try:
        metric = AlertMetric(request.metric)
    except ValueError:
        raise HTTPException(400, f"Invalid metric: {request.metric}. Use: z_score, spread, correlation, volume, price, half_life")
    
    # Validate operator
    try:
        operator = AlertOperator(request.operator)
    except ValueError:
        raise HTTPException(400, f"Invalid operator: {request.operator}. Use: >, <, >=, <=, abs>")
    
    # Create rule
    rule = AlertRule(
        id="",
        symbols=[s.upper() for s in request.symbols],
        metric=metric,
        operator=operator,
        threshold=request.threshold,
        cooldown_sec=request.cooldown_sec,
        name=request.name or "",
        enabled=True
    )
    
    engine.add_rule(rule)
    
    return {
        "message": "Alert rule created",
        "rule": rule.to_dict()
    }


@router.get("/rules")
async def list_rules():
    """Get all alert rules"""
    engine = get_alert_engine()
    rules = engine.get_rules()
    
    return {
        "count": len(rules),
        "rules": [r.to_dict() for r in rules]
    }


@router.get("/rules/{rule_id}")
async def get_rule(rule_id: str):
    """Get a specific alert rule"""
    engine = get_alert_engine()
    rule = engine.get_rule(rule_id)
    
    if not rule:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    
    # Include state info
    state = engine._states.get(rule_id)
    state_info = {
        "is_active": state.is_active if state else False,
        "trigger_count": state.trigger_count if state else 0,
        "last_triggered": state.last_triggered_at.isoformat() if state and state.last_triggered_at else None
    }
    
    return {
        "rule": rule.to_dict(),
        "state": state_info
    }


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """Delete an alert rule"""
    engine = get_alert_engine()
    
    if not engine.remove_rule(rule_id):
        raise HTTPException(404, f"Rule not found: {rule_id}")
    
    return {"message": f"Rule {rule_id} deleted"}


@router.post("/rules/{rule_id}/enable")
async def enable_rule(rule_id: str):
    """Enable an alert rule"""
    engine = get_alert_engine()
    
    if not engine.enable_rule(rule_id):
        raise HTTPException(404, f"Rule not found: {rule_id}")
    
    return {"message": f"Rule {rule_id} enabled"}


@router.post("/rules/{rule_id}/disable")
async def disable_rule(rule_id: str):
    """Disable an alert rule"""
    engine = get_alert_engine()
    
    if not engine.disable_rule(rule_id):
        raise HTTPException(404, f"Rule not found: {rule_id}")
    
    return {"message": f"Rule {rule_id} disabled"}


# =============================================================================
# Alert History
# =============================================================================

@router.get("/history")
async def get_history(limit: int = Query(default=50, le=200)):
    """Get recent alert history"""
    engine = get_alert_engine()
    history = engine.get_history(limit)
    
    return {
        "count": len(history),
        "alerts": [e.to_dict() for e in history]
    }


@router.delete("/history")
async def clear_history():
    """Clear alert history"""
    engine = get_alert_engine()
    engine.clear_history()
    
    return {"message": "Alert history cleared"}


# =============================================================================
# SSE Stream
# =============================================================================

@router.get("/stream")
async def stream_alerts():
    """
    Server-Sent Events stream for real-time alerts.
    
    Connect via EventSource in browser:
        const es = new EventSource('/api/alerts/stream');
        es.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    engine = get_alert_engine()
    
    async def event_generator():
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Alert stream connected'})}\n\n"
        
        while True:
            try:
                # Wait for event with timeout (for keepalive)
                event = await engine.get_event(timeout=30.0)
                
                if event:
                    yield f"data: {json.dumps(event.to_dict())}\n\n"
                else:
                    # Keepalive ping
                    yield f": keepalive\n\n"
                    
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# =============================================================================
# Testing & Management
# =============================================================================

@router.post("/test")
async def test_alerts(request: TestAlertRequest):
    """
    Test alert evaluation with mock analytics data.
    
    Useful for testing rules without live data.
    """
    engine = get_alert_engine()
    
    # Build analytics dict from request
    analytics = {}
    if request.z_score is not None:
        analytics["z_score"] = request.z_score
    if request.spread is not None:
        analytics["spread"] = request.spread
    if request.correlation is not None:
        analytics["correlation"] = request.correlation
    if request.volume is not None:
        analytics["volume"] = request.volume
    if request.price is not None:
        analytics["price"] = request.price
    
    if not analytics:
        raise HTTPException(400, "Provide at least one metric value")
    
    # Evaluate
    triggered = engine.evaluate(analytics, symbols=request.symbols)
    
    return {
        "input": analytics,
        "symbols": request.symbols,
        "triggered_count": len(triggered),
        "triggered": [e.to_dict() for e in triggered]
    }


@router.get("/stats")
async def get_stats():
    """Get alert engine statistics"""
    engine = get_alert_engine()
    return engine.stats()


@router.post("/reset")
async def reset_states():
    """Reset all alert states (clear cooldowns)"""
    engine = get_alert_engine()
    engine.reset_states()
    
    return {"message": "Alert states reset"}

