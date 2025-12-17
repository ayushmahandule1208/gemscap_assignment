"""
Alert Models
Data structures for alert rules, state, and events.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class AlertMetric(str, Enum):
    """Supported alert metrics"""
    Z_SCORE = "z_score"
    SPREAD = "spread"
    CORRELATION = "correlation"
    VOLUME = "volume"
    PRICE = "price"
    HALF_LIFE = "half_life"


class AlertOperator(str, Enum):
    """Alert condition operators"""
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    ABS_GT = "abs>"  # |value| > threshold


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """
    User-defined alert rule.
    
    Example:
        "Alert me when z_score > 2.0 for BTCUSDT/ETHUSDT"
    """
    id: str
    symbols: List[str]
    metric: AlertMetric
    operator: AlertOperator
    threshold: float
    cooldown_sec: int = 60
    enabled: bool = True
    name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"alert_{uuid.uuid4().hex[:8]}"
        if not self.name:
            self.name = f"{self.metric.value} {self.operator.value} {self.threshold}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbols": self.symbols,
            "metric": self.metric.value,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "cooldown_sec": self.cooldown_sec,
            "enabled": self.enabled,
            "name": self.name,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertRule":
        return cls(
            id=data.get("id", ""),
            symbols=data.get("symbols", []),
            metric=AlertMetric(data["metric"]),
            operator=AlertOperator(data["operator"]),
            threshold=float(data["threshold"]),
            cooldown_sec=int(data.get("cooldown_sec", 60)),
            enabled=data.get("enabled", True),
            name=data.get("name", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )


@dataclass
class AlertState:
    """
    Runtime state for an alert rule.
    
    Tracks when alert was last triggered to implement cooldown.
    """
    rule_id: str
    last_triggered_at: Optional[datetime] = None
    is_active: bool = False
    trigger_count: int = 0
    last_value: Optional[float] = None
    
    def can_trigger(self, cooldown_sec: int) -> bool:
        """Check if cooldown has elapsed"""
        if self.last_triggered_at is None:
            return True
        elapsed = (datetime.now() - self.last_triggered_at).total_seconds()
        return elapsed >= cooldown_sec
    
    def record_trigger(self, value: float) -> None:
        """Record that alert was triggered"""
        self.last_triggered_at = datetime.now()
        self.is_active = True
        self.trigger_count += 1
        self.last_value = value
    
    def reset(self) -> None:
        """Reset when condition becomes false"""
        self.is_active = False


@dataclass
class AlertEvent:
    """
    A triggered alert event.
    
    This is what gets sent to the frontend and persisted to history.
    """
    id: str
    rule_id: str
    timestamp: datetime
    metric: str
    value: float
    threshold: float
    operator: str
    symbols: List[str]
    message: str
    severity: AlertSeverity = AlertSeverity.WARNING
    
    def __post_init__(self):
        if not self.id:
            self.id = f"evt_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "timestamp": self.timestamp.isoformat(),
            "metric": self.metric,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "operator": self.operator,
            "symbols": self.symbols,
            "message": self.message,
            "severity": self.severity.value
        }
    
    @classmethod
    def from_rule(cls, rule: AlertRule, value: float) -> "AlertEvent":
        """Create event from triggered rule"""
        # Determine severity
        if rule.metric == AlertMetric.Z_SCORE:
            severity = AlertSeverity.CRITICAL if abs(value) > 2.5 else AlertSeverity.WARNING
        else:
            severity = AlertSeverity.WARNING
        
        # Generate message
        pair = "/".join(rule.symbols) if len(rule.symbols) > 1 else rule.symbols[0]
        message = f"{rule.metric.value} {rule.operator.value} {rule.threshold} ({pair})"
        
        if rule.metric == AlertMetric.Z_SCORE:
            if value > 0:
                message = f"Z-score crossed +{rule.threshold} (value: {value:.2f})"
            else:
                message = f"Z-score crossed -{rule.threshold} (value: {value:.2f})"
        
        return cls(
            id="",
            rule_id=rule.id,
            timestamp=datetime.now(),
            metric=rule.metric.value,
            value=value,
            threshold=rule.threshold,
            operator=rule.operator.value,
            symbols=rule.symbols,
            message=message,
            severity=severity
        )

