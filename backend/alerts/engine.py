import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from collections import deque

from .models import (
    AlertRule, 
    AlertState, 
    AlertEvent, 
    AlertMetric, 
    AlertOperator,
    AlertSeverity
)


class AlertEngine:
    def __init__(self, history_size: int = 100):
        self._rules: Dict[str, AlertRule] = {}
        self._states: Dict[str, AlertState] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._history: deque = deque(maxlen=history_size)
        self._callbacks: List[Callable[[AlertEvent], None]] = []
        self._stats = {
            "evaluations": 0,
            "triggers": 0,
            "suppressed": 0,
            "start_time": datetime.now()
        }
    
    def add_rule(self, rule: AlertRule) -> AlertRule:
        self._rules[rule.id] = rule
        if rule.id not in self._states:
            self._states[rule.id] = AlertState(rule_id=rule.id)
        return rule
    
    def remove_rule(self, rule_id: str) -> bool:
        if rule_id in self._rules:
            del self._rules[rule_id]
            if rule_id in self._states:
                del self._states[rule_id]
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        return self._rules.get(rule_id)
    
    def get_rules(self) -> List[AlertRule]:
        return list(self._rules.values())
    
    def enable_rule(self, rule_id: str) -> bool:
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            return True
        return False
    
    def evaluate(self, analytics: Dict[str, Any], symbols: List[str] = None) -> List[AlertEvent]:
        triggered = []
        self._stats["evaluations"] += 1
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            if symbols and not self._symbols_match(rule.symbols, symbols):
                continue
            
            value = self._get_metric_value(analytics, rule.metric)
            if value is None:
                continue
            
            condition_met = self._evaluate_condition(value, rule.operator, rule.threshold)
            
            state = self._states.get(rule.id)
            if state is None:
                state = AlertState(rule_id=rule.id)
                self._states[rule.id] = state
            
            if condition_met:
                if not state.is_active and state.can_trigger(rule.cooldown_sec):
                    event = AlertEvent.from_rule(rule, value)
                    state.record_trigger(value)
                    triggered.append(event)
                    self._history.append(event)
                    self._stats["triggers"] += 1
                    
                    try:
                        self._event_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass
                    
                    for callback in self._callbacks:
                        try:
                            callback(event)
                        except Exception:
                            pass
                else:
                    self._stats["suppressed"] += 1
            else:
                if state.is_active:
                    state.reset()
        
        return triggered
    
    def _symbols_match(self, rule_symbols: List[str], data_symbols: List[str]) -> bool:
        rule_set = set(s.upper() for s in rule_symbols)
        data_set = set(s.upper() for s in data_symbols)
        return bool(rule_set & data_set)
    
    def _get_metric_value(self, analytics: Dict[str, Any], metric: AlertMetric) -> Optional[float]:
        key_map = {
            AlertMetric.Z_SCORE: ["z_score", "z", "zscore"],
            AlertMetric.SPREAD: ["spread", "spread_value"],
            AlertMetric.CORRELATION: ["correlation", "corr", "rolling_corr"],
            AlertMetric.VOLUME: ["volume", "vol"],
            AlertMetric.PRICE: ["price", "last_price", "close"],
            AlertMetric.HALF_LIFE: ["half_life", "halflife"],
        }
        
        for key in key_map.get(metric, []):
            if key in analytics:
                val = analytics[key]
                if isinstance(val, (int, float)):
                    return float(val)
        return None
    
    def _evaluate_condition(self, value: float, operator: AlertOperator, threshold: float) -> bool:
        if operator == AlertOperator.GT:
            return value > threshold
        elif operator == AlertOperator.LT:
            return value < threshold
        elif operator == AlertOperator.GTE:
            return value >= threshold
        elif operator == AlertOperator.LTE:
            return value <= threshold
        elif operator == AlertOperator.ABS_GT:
            return abs(value) > threshold
        return False
    
    async def get_event(self, timeout: float = None) -> Optional[AlertEvent]:
        try:
            if timeout:
                return await asyncio.wait_for(self._event_queue.get(), timeout)
            return await self._event_queue.get()
        except asyncio.TimeoutError:
            return None
    
    def get_history(self, limit: int = 50) -> List[AlertEvent]:
        history = list(self._history)
        history.reverse()
        return history[:limit]
    
    def on_alert(self, callback: Callable[[AlertEvent], None]) -> None:
        self._callbacks.append(callback)
    
    def clear_history(self) -> None:
        self._history.clear()
    
    def reset_states(self) -> None:
        for state in self._states.values():
            state.reset()
            state.last_triggered_at = None
            state.trigger_count = 0
    
    def stats(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self._stats["start_time"]).total_seconds()
        return {
            **self._stats,
            "uptime_seconds": round(uptime, 2),
            "rules_count": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "history_size": len(self._history),
            "queue_size": self._event_queue.qsize()
        }


_alert_engine: Optional[AlertEngine] = None


def get_alert_engine() -> AlertEngine:
    global _alert_engine
    if _alert_engine is None:
        _alert_engine = AlertEngine()
    return _alert_engine
