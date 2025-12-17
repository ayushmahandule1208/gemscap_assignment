"""
Panel Components for Quant Analytics Dashboard
Sidebar panels and information displays
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional


class PanelBuilder:
    """Build dashboard panels and UI components"""
    
    @staticmethod
    def render_data_health_panel(health_data: Dict) -> None:
        """Render data health status panel"""
        
        st.markdown("""
        <div class="panel-header">
            <span class="panel-icon">🔬</span>
            <span>Data Health</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate overall health score
        clean_rate = health_data.get('clean_rate', 99.9)
        
        if clean_rate >= 99:
            status_class = "status-good"
            status_icon = "✓"
        elif clean_rate >= 95:
            status_class = "status-warning"
            status_icon = "⚠"
        else:
            status_class = "status-bad"
            status_icon = "✗"
        
        st.markdown(f"""
        <div class="health-score {status_class}">
            <span class="score-value">{clean_rate:.2f}%</span>
            <span class="score-label">Clean Rate</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual metrics
        metrics = [
            ("Out of Order", health_data.get('out_of_order', 0), 'out_of_order'),
            ("Duplicates", health_data.get('duplicates', 0), 'duplicates'),
            ("Price Spikes", health_data.get('price_spikes', 0), 'price_spikes'),
            ("Data Gaps", health_data.get('gaps', 0), 'gaps'),
        ]
        
        for label, value, key in metrics:
            icon = "✓" if value == 0 else "⚠"
            cls = "metric-ok" if value == 0 else "metric-warn"
            st.markdown(f"""
            <div class="health-metric {cls}">
                <span class="metric-icon">{icon}</span>
                <span class="metric-label">{label}</span>
                <span class="metric-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="health-total">
            Total Ticks: <span class="highlight">{health_data.get('total_ticks', 0):,}</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_system_metrics_panel(metrics: Dict) -> None:
        """Render system performance metrics panel"""
        
        st.markdown("""
        <div class="panel-header">
            <span class="panel-icon">⚡</span>
            <span>System Metrics</span>
        </div>
        """, unsafe_allow_html=True)
        
        # WebSocket status
        ws_status = metrics.get('websocket_status', 'disconnected')
        ws_class = "status-connected" if ws_status == 'connected' else "status-disconnected"
        st.markdown(f"""
        <div class="ws-status {ws_class}">
            <span class="ws-dot"></span>
            <span>WebSocket: {ws_status}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Latency metrics
        latency_metrics = [
            ("Ingestion", metrics.get('ingestion_latency_ms', 0), "ms"),
            ("Analytics", metrics.get('analytics_compute_ms', 0), "ms"),
            ("Alert Trigger", metrics.get('alert_trigger_ms', 0), "ms"),
        ]
        
        st.markdown('<div class="latency-grid">', unsafe_allow_html=True)
        for label, value, unit in latency_metrics:
            # Color based on value
            if value < 50:
                cls = "latency-good"
            elif value < 100:
                cls = "latency-ok"
            else:
                cls = "latency-bad"
            
            st.markdown(f"""
            <div class="latency-item {cls}">
                <span class="latency-value">{value:.0f}</span>
                <span class="latency-unit">{unit}</span>
                <span class="latency-label">{label}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # System resources
        st.markdown(f"""
        <div class="system-resources">
            <div class="resource-item">
                <span class="resource-label">CPU</span>
                <div class="resource-bar">
                    <div class="resource-fill" style="width: {metrics.get('cpu_usage_pct', 0)}%"></div>
                </div>
                <span class="resource-value">{metrics.get('cpu_usage_pct', 0):.1f}%</span>
            </div>
            <div class="resource-item">
                <span class="resource-label">MEM</span>
                <div class="resource-bar">
                    <div class="resource-fill" style="width: {min(metrics.get('memory_usage_mb', 0) / 10.24, 100)}%"></div>
                </div>
                <span class="resource-value">{metrics.get('memory_usage_mb', 0):.0f} MB</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Throughput
        st.markdown(f"""
        <div class="throughput">
            <span class="throughput-value">{metrics.get('ticks_per_second', 0):.0f}</span>
            <span class="throughput-label">ticks/sec</span>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_alerts_panel(alerts: List[Dict]) -> None:
        """Render alerts panel"""
        
        st.markdown("""
        <div class="panel-header">
            <span class="panel-icon">🔔</span>
            <span>Alerts</span>
        </div>
        """, unsafe_allow_html=True)
        
        if not alerts:
            st.markdown("""
            <div class="no-alerts">
                <span class="no-alerts-icon">✓</span>
                <span>No active alerts</span>
            </div>
            """, unsafe_allow_html=True)
            return
        
        for alert in alerts[:5]:  # Show last 5 alerts
            severity = alert.get('severity', 'info')
            timestamp = alert.get('timestamp', datetime.now())
            
            if isinstance(timestamp, datetime):
                time_str = timestamp.strftime('%H:%M:%S')
            else:
                time_str = str(timestamp)
            
            severity_icons = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'critical': '🚨'
            }
            
            st.markdown(f"""
            <div class="alert-item alert-{severity}">
                <div class="alert-header">
                    <span class="alert-icon">{severity_icons.get(severity, 'ℹ️')}</span>
                    <span class="alert-time">{time_str}</span>
                </div>
                <div class="alert-symbol">{alert.get('symbol', '')}</div>
                <div class="alert-message">{alert.get('message', '')}</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_analytics_summary(analytics: Dict) -> None:
        """Render analytics summary panel"""
        
        st.markdown("""
        <div class="panel-header">
            <span class="panel-icon">📊</span>
            <span>Analytics</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        metrics = [
            ("Current Spread", analytics.get('current_spread', 0), ".4f"),
            ("Rolling Mean", analytics.get('rolling_mean', 0), ".4f"),
            ("Rolling Std", analytics.get('rolling_std', 0), ".4f"),
            ("Z-Score", analytics.get('z_score', 0), ".2f"),
            ("Hedge Ratio", analytics.get('hedge_ratio', 1.0), ".4f"),
        ]
        
        for label, value, fmt in metrics:
            # Highlight z-score if significant
            highlight = ""
            if label == "Z-Score" and abs(value) > 2:
                highlight = "highlight-alert"
            
            st.markdown(f"""
            <div class="analytics-metric {highlight}">
                <span class="analytics-label">{label}</span>
                <span class="analytics-value">{value:{fmt}}</span>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_adf_test_results(results: Dict) -> None:
        """Render ADF test results"""
        
        st.markdown("""
        <div class="test-header">
            <span>ADF Stationarity Test</span>
        </div>
        """, unsafe_allow_html=True)
        
        if results.get('statistic') is None:
            st.markdown(f"""
            <div class="test-message">{results.get('message', 'Test not available')}</div>
            """, unsafe_allow_html=True)
            return
        
        is_stationary = results.get('is_stationary', False)
        status_class = "test-pass" if is_stationary else "test-fail"
        
        st.markdown(f"""
        <div class="test-result {status_class}">
            <div class="test-stat">
                <span class="test-label">Test Statistic</span>
                <span class="test-value">{results.get('statistic', 0):.4f}</span>
            </div>
            <div class="test-stat">
                <span class="test-label">P-Value</span>
                <span class="test-value">{results.get('p_value', 0):.4f}</span>
            </div>
            <div class="test-conclusion">
                {results.get('message', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Critical values
        if results.get('critical_values'):
            st.markdown('<div class="critical-values">', unsafe_allow_html=True)
            for level, value in results['critical_values'].items():
                st.markdown(f"""
                <div class="cv-item">
                    <span class="cv-level">{level}</span>
                    <span class="cv-value">{value:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

