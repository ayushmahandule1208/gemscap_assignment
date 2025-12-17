"""
Alert System
Rule-based alerts that monitor live analytics.

Structure:
    alerts/
    ├── models.py    → AlertRule, AlertState, AlertEvent
    └── engine.py    → AlertEngine (evaluation + state)

Usage:
    from alerts import get_alert_engine, AlertRule, AlertMetric, AlertOperator
    
    # Get engine
    engine = get_alert_engine()
    
    # Add rule
    rule = AlertRule(
        id="",
        symbols=["BTCUSDT", "ETHUSDT"],
        metric=AlertMetric.Z_SCORE,
        operator=AlertOperator.ABS_GT,
        threshold=2.0,
        cooldown_sec=60
    )
    engine.add_rule(rule)
    
    # Evaluate (called when analytics update)
    triggered = engine.evaluate({"z_score": 2.5}, symbols=["BTCUSDT", "ETHUSDT"])
    
    # Get history
    history = engine.get_history(limit=20)
"""

from .models import (
    AlertRule,
    AlertState,
    AlertEvent,
    AlertMetric,
    AlertOperator,
    AlertSeverity,
)

from .engine import (
    AlertEngine,
    get_alert_engine,
)

__all__ = [
    # Models
    "AlertRule",
    "AlertState",
    "AlertEvent",
    "AlertMetric",
    "AlertOperator",
    "AlertSeverity",
    # Engine
    "AlertEngine",
    "get_alert_engine",
]

