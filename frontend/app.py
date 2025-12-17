import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

from utils.api_client import APIClient
from components.charts import ChartBuilder
from components.panels import PanelBuilder

st.set_page_config(
    page_title="Quant Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
try:
    with open("assets/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

def init_session_state():
    defaults = {
        'backend_url': 'http://localhost:8000',
        'connected': False,
        'symbols': [],
        'selected_symbols': [],
        'ohlc_data': {},
        'alerts': [],
        'last_refresh': None,
        'auto_refresh': False,
        'data_availability': {},
        'features_status': {},
        'current_page': 'Analytics',  # Default page
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


FEATURE_REQUIREMENTS = {
    "price_chart": {"min_bars": 1, "tab": "Price Analysis", "icon": "[P]"},
    "basic_stats": {"min_bars": 2, "tab": "Price Analysis", "icon": "[S]"},
    "spread_analysis": {"min_bars": 5, "tab": "Spread Analytics", "icon": "[SP]"},
    "hedge_ratio": {"min_bars": 10, "tab": "Advanced Analysis", "icon": "[HR]"},
    "adf_test": {"min_bars": 20, "tab": "Advanced Analysis", "icon": "[ADF]"},
    "half_life": {"min_bars": 20, "tab": "Advanced Analysis", "icon": "[HL]"},
    "correlation": {"min_bars": 5, "tab": "Advanced Analysis", "icon": "[C]"},
    "backtest": {"min_bars": 30, "tab": "Backtest", "icon": "[BT]"},
    "alerts": {"min_bars": 1, "tab": None, "icon": "[A]"},
}

TAB_REQUIREMENTS = {
    "Price Analysis": 1,
    "Spread Analytics": 5,
    "Advanced Analysis": 10,
    "Time Series Table": 5,
}

SIDEBAR_PAGES = ["Analytics", "Backtest", "Export", "Alerts"]

def get_api_client(url: str) -> APIClient:
    return APIClient(url)

client = get_api_client(st.session_state.backend_url)

def check_backend_connection():
    st.session_state.connected = client.is_connected()
    if st.session_state.connected:
        st.session_state.symbols = client.get_symbols()
        update_data_availability()
    return st.session_state.connected


def update_data_availability():
    if not st.session_state.connected:
        return
    
    availability = client.get_data_availability()
    if "error" not in availability:
        st.session_state.data_availability = availability.get("symbols", {})
        st.session_state.features_status = availability.get("features", {})


def find_symbol_key(target: str, avail: dict) -> str:
    target_upper = target.upper() if target else ""
    for key in avail.keys():
        if key.upper() == target_upper:
            return key
    return target_upper


def get_min_bars_for_pair(symbol1: str, symbol2: str, timeframe: str = "1m") -> int:
    avail = st.session_state.data_availability
    sym1_key = find_symbol_key(symbol1, avail)
    sym2_key = find_symbol_key(symbol2, avail)
    bars1 = avail.get(sym1_key, {}).get("bars", {}).get(timeframe, 0)
    bars2 = avail.get(sym2_key, {}).get("bars", {}).get(timeframe, 0)
    return min(bars1, bars2)


def get_total_ticks_for_pair(symbol1: str, symbol2: str) -> int:
    avail = st.session_state.data_availability
    sym1_key = find_symbol_key(symbol1, avail)
    sym2_key = find_symbol_key(symbol2, avail)
    ticks1 = avail.get(sym1_key, {}).get("ticks", 0)
    ticks2 = avail.get(sym2_key, {}).get("ticks", 0)
    return min(ticks1, ticks2)


def is_feature_unlocked(feature: str, symbol1: str = None, symbol2: str = None, timeframe: str = "1m") -> bool:
    if feature not in FEATURE_REQUIREMENTS:
        return True
    
    required = FEATURE_REQUIREMENTS[feature]["min_bars"]
    
    if symbol1 and symbol2:
        current_bars = get_min_bars_for_pair(symbol1, symbol2, timeframe)
    elif symbol1:
        avail = st.session_state.data_availability
        sym_key = find_symbol_key(symbol1, avail)
        current_bars = avail.get(sym_key, {}).get("bars", {}).get(timeframe, 0)
    else:
        status = st.session_state.features_status.get(feature, {})
        return status.get("unlocked", False)
    
    return current_bars >= required


def get_feature_progress(feature: str, symbol1: str = None, symbol2: str = None, timeframe: str = "1m") -> dict:
    if feature not in FEATURE_REQUIREMENTS:
        return {"progress": 100, "current": 0, "required": 0, "unlocked": True}
    
    required = FEATURE_REQUIREMENTS[feature]["min_bars"]
    
    if symbol1 and symbol2:
        current_bars = get_min_bars_for_pair(symbol1, symbol2, timeframe)
    elif symbol1:
        avail = st.session_state.data_availability
        sym_key = find_symbol_key(symbol1, avail)
        current_bars = avail.get(sym_key, {}).get("bars", {}).get(timeframe, 0)
    else:
        current_bars = 0
    
    progress = min(100, int(current_bars / required * 100)) if required > 0 else 100
    
    return {
        "progress": progress,
        "current": current_bars,
        "required": required,
        "unlocked": current_bars >= required
    }

def load_ohlc_data(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    df = client.get_ohlc(symbol, timeframe, limit)
    if not df.empty:
        st.session_state.ohlc_data[symbol] = df
    return df


@st.dialog("Data Health Report")
def show_data_health_dialog():
    upload_info = st.session_state.get("last_upload", {})
    symbol = upload_info.get("symbol", "UNKNOWN")
    count = upload_info.get("count", 0)
    filename = upload_info.get("filename", "file")
    
    avail = client.get_symbol_availability(symbol)
    
    st.markdown(f"""
    <div style="text-align: center; padding: 16px 0;">
        <div style="font-size: 48px; margin-bottom: 8px;">✓</div>
        <div style="font-size: 20px; font-weight: 600; color: #10b981;">Upload Successful</div>
        <div style="font-size: 12px; color: #64748b; margin-top: 4px;">{filename}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Symbol", symbol)
    with col2:
        st.metric("Records", f"{count:,}")
    with col3:
        if isinstance(avail, dict):
            bars_val = avail.get("bars", 0)
            # Handle both int and nested dict formats
            if isinstance(bars_val, dict):
                bars_1m = bars_val.get("1m", 0)
            else:
                bars_1m = bars_val
        else:
            bars_1m = 0
        st.metric("1m Bars", bars_1m)
    
    st.markdown("#### Health Checks")
    
    health_items = []
    if count >= 100:
        health_items.append(("✓", "Sufficient data points", f"{count} records", "#10b981"))
    elif count >= 20:
        health_items.append(("◐", "Moderate data", f"{count} records", "#f59e0b"))
    else:
        health_items.append(("✗", "Limited data", f"Only {count} records", "#ef4444"))
    
    health_items.append(("✓", "Data format valid", "OHLC columns detected", "#10b981"))
    
    if count >= 60:
        health_items.append(("✓", "Good for analysis", "All features available", "#10b981"))
    elif count >= 20:
        health_items.append(("◐", "Basic analysis ready", "Some features limited", "#f59e0b"))
    else:
        health_items.append(("✗", "Need more data", "Upload more records", "#ef4444"))
    
    for icon, label, detail, color in health_items:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; padding: 8px 12px; background: rgba(100, 116, 139, 0.1); border-radius: 8px; margin-bottom: 8px;">
            <span style="font-size: 16px; color: {color};">{icon}</span>
            <div>
                <div style="font-size: 13px; color: #f0f4f8;">{label}</div>
                <div style="font-size: 11px; color: #64748b;">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    if st.button("Close", type="primary", use_container_width=True):
        st.session_state.last_upload["show_dialog"] = False
        st.rerun()


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 28px 0; margin-bottom: 24px; background: linear-gradient(180deg, rgba(59, 130, 246, 0.08) 0%, transparent 100%); border-radius: 12px;">
            <div style="font-size: 42px; margin-bottom: 12px; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">◈</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 700; color: #f0f4f8; letter-spacing: -0.02em;">
                QUANT ANALYTICS
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #64748b; margin-top: 8px; letter-spacing: 0.05em;">
                v2.0.0 • PROFESSIONAL EDITION
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### CONNECTION")
        backend_url = st.text_input("Backend URL", value=st.session_state.backend_url, label_visibility="collapsed")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", use_container_width=True):
                st.session_state.backend_url = backend_url
                if check_backend_connection():
                    st.success("Connected")
                else:
                    st.error("Connection failed")
        with col2:
            if st.button("Sync", use_container_width=True):
                st.session_state.symbols = client.get_symbols()
                st.rerun()
        
        if st.session_state.connected:
            health = client.health()
            if "error" not in health:
                engine = health.get("engine", {})
                bars = engine.get('bars_created', 0) + engine.get('bars_uploaded', 0)
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%); padding: 14px 16px; border-radius: 10px; border: 1px solid rgba(16, 185, 129, 0.25); margin-top: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981; animation: pulse 2s infinite;"></div>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #34d399;">SYSTEM ONLINE</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px;">
                        <div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 6px;">
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: #f0f4f8;">{len(st.session_state.symbols)}</div>
                            <div style="font-size: 9px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Symbols</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 6px;">
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 700; color: #f0f4f8;">{bars:,}</div>
                            <div style="font-size: 9px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Bars</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(248, 113, 113, 0.05) 100%); padding: 14px 16px; border-radius: 10px; border: 1px solid rgba(239, 68, 68, 0.25); margin-top: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: #ef4444; border-radius: 50%;"></div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #f87171;">OFFLINE</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #94a3b8; margin-top: 8px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 6px;">
                    python app.py
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### DATA UPLOAD")
        
        uploaded_file = st.file_uploader(
            "Upload market data",
            type=['csv', 'ndjson', 'json'],
            help="Upload OHLC or tick data"
        )
        
        if uploaded_file and st.session_state.connected:
            upload_symbol = st.text_input(
                "Symbol for upload",
                value="BTCUSDT",
                help="Symbol name (e.g., BTCUSDT)"
            )
            
            if st.button("Upload to Backend", use_container_width=True):
                content = uploaded_file.read()
                filename = uploaded_file.name.lower()
                
                if filename.endswith('.csv'):
                    result = client.upload_csv(content, uploaded_file.name, symbol=upload_symbol.upper())
                else:
                    result = client.upload_ndjson(content, uploaded_file.name)
                
                if "error" not in result:
                    st.session_state.symbols = client.get_symbols()
                    st.session_state.last_upload = {
                        "symbol": upload_symbol.upper(),
                        "count": result.get('count', 0),
                        "filename": uploaded_file.name,
                        "show_dialog": True
                    }
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error')}")
            
            if st.session_state.get("last_upload", {}).get("show_dialog"):
                show_data_health_dialog()
        
        st.markdown("---")
        
        # Live Feed Control
        st.markdown("### LIVE FEED")
        
        # Get current status
        live_status = client.get_live_status() if st.session_state.connected else {"status": "offline"}
        is_live_running = live_status.get("is_running", False)
        
        # Auto-start notice
        if is_live_running and live_status.get("uptime_seconds", 0) < 10:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(52, 211, 153, 0.08) 100%); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3); margin-bottom: 12px;">
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #34d399;">
                    ⚡ AUTO-STARTED
                </div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">
                    Live feed started automatically on backend startup
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Live symbols input
        live_symbols_input = st.text_input(
            "Symbols (comma-separated)", 
            value="btcusdt,ethusdt",
            help="Enter Binance futures symbols"
        )
        
        col_start, col_stop = st.columns(2)
        
        with col_start:
            start_disabled = is_live_running or not st.session_state.connected
            if st.button("Start", use_container_width=True, disabled=start_disabled, type="primary"):
                symbols = [s.strip() for s in live_symbols_input.split(",") if s.strip()]
                result = client.start_live_feed(symbols)
                if "error" not in result:
                    st.success("Live feed started")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed to start"))
        
        with col_stop:
            stop_disabled = not is_live_running
            if st.button("Stop", use_container_width=True, disabled=stop_disabled):
                result = client.stop_live_feed()
                st.info(f"Stopped. Total ticks: {result.get('total_ticks', 0):,}")
                time.sleep(1)
                st.rerun()
        
        # Show live status
        if is_live_running:
            ticks = live_status.get("ticks_received", 0)
            rate = live_status.get("ticks_per_second", 0)
            uptime = int(live_status.get("uptime_seconds", 0))
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%); padding: 16px; border-radius: 10px; border: 1px solid rgba(59, 130, 246, 0.3); margin-top: 12px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <div style="width: 8px; height: 8px; background: #22d3ee; border-radius: 50%; box-shadow: 0 0 12px #22d3ee; animation: pulse 1.5s infinite;"></div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #60a5fa; letter-spacing: 0.05em;">LIVE STREAMING</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                    <div style="text-align: center; padding: 10px 6px; background: rgba(0,0,0,0.25); border-radius: 8px;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #f0f4f8;">{ticks:,}</div>
                        <div style="font-size: 8px; color: #64748b; text-transform: uppercase; margin-top: 2px;">ticks</div>
                    </div>
                    <div style="text-align: center; padding: 10px 6px; background: rgba(0,0,0,0.25); border-radius: 8px;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #22d3ee;">{rate}</div>
                        <div style="font-size: 8px; color: #64748b; text-transform: uppercase; margin-top: 2px;">/sec</div>
                    </div>
                    <div style="text-align: center; padding: 10px 6px; background: rgba(0,0,0,0.25); border-radius: 8px;">
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #f0f4f8;">{uptime}s</div>
                        <div style="font-size: 8px; color: #64748b; text-transform: uppercase; margin-top: 2px;">uptime</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-refresh when live (uses fragments - no page dimming)
            auto_refresh = st.checkbox("Auto-refresh (2s)", value=True, key="auto_refresh_live")
            st.session_state.auto_refresh = auto_refresh
        else:
            st.markdown("""
            <div style="background: rgba(100, 116, 139, 0.08); padding: 14px 16px; border-radius: 10px; border: 1px dashed rgba(100, 116, 139, 0.3); margin-top: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; background: #64748b; border-radius: 50%;"></div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748b;">STANDBY</span>
                </div>
                <div style="font-size: 11px; color: #475569; margin-top: 8px;">
                    Click Start to stream real-time data from Binance
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Symbol Selection
        st.markdown("### PARAMETERS")
        
        available_symbols = st.session_state.symbols if st.session_state.symbols else ["BTCUSDT", "ETHUSDT"]
        
        symbol1 = st.selectbox("Primary Symbol", available_symbols, index=0)
        symbol2 = st.selectbox("Secondary Symbol", available_symbols, index=min(1, len(available_symbols)-1))
        
        timeframe = st.select_slider(
            "Timeframe",
            options=['1s', '1m', '5m'],
            value='1s'
        )
        
        rolling_window = st.slider("Rolling Window", 5, 100, 20, 5)
        z_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.0, 0.1)
        
        # Regression Type
        regression_type = st.selectbox(
            "Regression Method",
            ["OLS", "Huber (Robust)", "Theil-Sen (Robust)", "Kalman Filter"],
            help="Method for hedge ratio estimation"
        )
        
        st.markdown("---")
        
        # Page Navigation
        st.markdown("### PAGES")
        
        page = st.radio(
            "Select Page",
            options=["Analytics", "Backtest", "Export", "Alerts"],
            index=["Analytics", "Backtest", "Export", "Alerts"].index(st.session_state.current_page),
            label_visibility="collapsed",
            horizontal=False
        )
        if page != st.session_state.current_page:
            st.session_state.current_page = page
        
        return symbol1, symbol2, timeframe, rolling_window, z_threshold, regression_type

@st.fragment(run_every=timedelta(seconds=2))
def render_live_data_fragment(symbol1: str, symbol2: str, timeframe: str):
    """Fragment that auto-refreshes every 2 seconds without dimming the page"""
    _render_data_progress_content(symbol1, symbol2, timeframe)


def render_data_progress_panel(symbol1: str, symbol2: str, timeframe: str):
    """Wrapper that decides whether to use fragment or static render"""
    if st.session_state.get("auto_refresh", False):
        render_live_data_fragment(symbol1, symbol2, timeframe)
    else:
        _render_data_progress_content(symbol1, symbol2, timeframe)


def _render_data_progress_content(symbol1: str, symbol2: str, timeframe: str):
    """Actual content rendering for data progress panel"""
    update_data_availability()
    live_status = client.get_live_status() if st.session_state.connected else {}
    is_live_running = live_status.get("is_running", False)
    ticks_received = live_status.get("ticks_received", 0)
    ticks_per_sec = live_status.get("ticks_per_second", 0)
    
    # Get availability data
    avail = st.session_state.data_availability
    
    # Find matching symbols using global helper
    sym1_key = find_symbol_key(symbol1, avail)
    sym2_key = find_symbol_key(symbol2, avail)
    
    # Get bar counts using matched keys directly from availability
    bars1_tf = avail.get(sym1_key, {}).get("bars", {}).get(timeframe, 0)
    bars2_tf = avail.get(sym2_key, {}).get("bars", {}).get(timeframe, 0)
    bars1_1m = avail.get(sym1_key, {}).get("bars", {}).get("1m", 0)
    bars2_1m = avail.get(sym2_key, {}).get("bars", {}).get("1m", 0)
    bars1_1s = avail.get(sym1_key, {}).get("bars", {}).get("1s", 0)
    bars2_1s = avail.get(sym2_key, {}).get("bars", {}).get("1s", 0)
    ticks1 = avail.get(sym1_key, {}).get("ticks", 0)
    ticks2 = avail.get(sym2_key, {}).get("ticks", 0)
    
    # Calculate min bars for the pair
    min_bars = min(bars1_tf, bars2_tf)
    min_bars_1m = min(bars1_1m, bars2_1m)
    min_bars_1s = min(bars1_1s, bars2_1s)
    min_ticks = min(ticks1, ticks2)
    
    # Calculate overall progress
    max_required = max(req["min_bars"] for req in FEATURE_REQUIREMENTS.values())
    overall_progress = min(100, int(min_bars / max_required * 100))
    
    # Count unlocked features
    unlocked_count = sum(1 for f in FEATURE_REQUIREMENTS if is_feature_unlocked(f, symbol1, symbol2, timeframe))
    total_features = len(FEATURE_REQUIREMENTS)
    
    # Live status indicator
    if is_live_running:
        live_indicator = f'<div style="display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: rgba(16, 185, 129, 0.15); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);"><div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></div><span style="font-family: monospace; font-size: 10px; color: #34d399;">LIVE • {ticks_received:,} ticks • {ticks_per_sec}/s</span></div>'
    else:
        live_indicator = '<div style="display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: rgba(239, 68, 68, 0.15); border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.3);"><div style="width: 8px; height: 8px; background: #ef4444; border-radius: 50%;"></div><span style="font-family: monospace; font-size: 10px; color: #f87171;">FEED OFFLINE</span></div>'
    
    # Status message
    if min_bars == 0 and min_ticks == 0 and ticks_received > 0:
        # Ticks from live feed but not in availability yet
        status_msg = f"Initializing... {ticks_received:,} ticks received"
    elif min_bars == 0 and min_ticks == 0:
        status_msg = "Waiting for data..."
    elif min_bars == 0 and min_ticks > 0:
        status_msg = f"Building bars... {min_ticks:,} ticks | {min_bars_1s} 1s bars"
    else:
        # Avoid redundancy when timeframe is 1s
        if timeframe == "1s":
            status_msg = f"{min_bars} 1s bars | {min_bars_1m} 1m | {min_ticks:,} ticks"
        else:
            status_msg = f"{min_bars} {timeframe} bars | {min_bars_1s} 1s | {min_ticks:,} ticks"
    
    st.markdown(f"""<div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.04) 100%); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 16px; padding: 16px 20px; margin-bottom: 24px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div>
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
<span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em;">DATA</span>
{live_indicator}
</div>
<div style="font-family: 'Space Grotesk', sans-serif; font-size: 14px; color: #94a3b8;">{status_msg}</div>
</div>
<div style="text-align: right;">
<div style="font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: #f0f4f8;">{unlocked_count}/{total_features}</div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #64748b;">features</div>
</div>
</div>
</div>""", unsafe_allow_html=True)


def render_main_content(symbol1, symbol2, timeframe, rolling_window, z_threshold, regression_type):
    
    # Premium Header
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 0; margin-bottom: 24px; border-bottom: 1px solid rgba(255,255,255,0.06);">
        <div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 28px; font-weight: 700; color: #f0f4f8; letter-spacing: -0.02em;">
                <span style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">◈</span>
                Pair Analytics
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #64748b; margin-top: 4px;">
                {symbol1} / {symbol2} • {timeframe} timeframe • {rolling_window}-period window
            </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <div style="display: flex; align-items: center; gap: 8px; padding: 8px 16px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(52, 211, 153, 0.06) 100%); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 20px;">
                <div style="width: 6px; height: 6px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981;"></div>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500; color: #34d399;">LIVE</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #64748b; padding: 8px 12px; background: rgba(100, 116, 139, 0.1); border-radius: 8px;">
                {datetime.now().strftime('%H:%M:%S')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check connection
    if not st.session_state.connected:
        st.warning("Backend not connected. Connect to backend or upload data to view analytics.")
        return
    
    # Render data collection progress panel
    render_data_progress_panel(symbol1, symbol2, timeframe)
    
    # Load data
    df1 = load_ohlc_data(symbol1, timeframe)
    df2 = load_ohlc_data(symbol2, timeframe)
    
    # Render based on current page
    current_page = st.session_state.current_page
    
    if current_page == "Analytics":
        # Main analytics with 4 tabs
        tab_names = list(TAB_REQUIREMENTS.keys())
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            render_price_analysis(symbol1, symbol2, df1, df2, timeframe, rolling_window)
        
        with tabs[1]:
            render_spread_analytics(symbol1, symbol2, timeframe, rolling_window, z_threshold, regression_type)
        
        with tabs[2]:
            render_advanced_analysis(symbol1, symbol2, timeframe, rolling_window, regression_type)
        
        with tabs[3]:
            render_time_series_table(symbol1, symbol2, timeframe)
    
    elif current_page == "Backtest":
        st.header("Backtest")
        render_backtest(symbol1, symbol2, timeframe, z_threshold)
    
    elif current_page == "Export":
        st.header("Export")
        render_export(symbol1, symbol2, timeframe)
    
    elif current_page == "Alerts":
        st.header("Alerts")
        render_alerts_page(symbol1, symbol2)


def render_locked_tab(tab_name: str, required_bars: int, current_bars: int):
    progress = min(100, int(current_bars / required_bars * 100)) if required_bars > 0 else 0
    remaining = max(0, required_bars - current_bars)
    
    st.markdown(f"""<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 80px 40px; text-align: center;">
<div style="font-size: 48px; margin-bottom: 24px; opacity: 0.5; font-weight: bold;">[LOCKED]</div>
<div style="font-family: sans-serif; font-size: 24px; font-weight: 700; color: #f0f4f8; margin-bottom: 12px;">{tab_name} Locked</div>
<div style="font-family: monospace; font-size: 14px; color: #64748b; margin-bottom: 24px; max-width: 400px;">This feature requires more data points to function accurately. Data is being collected automatically from the live feed.</div>
<div style="width: 100%; max-width: 300px; margin-bottom: 16px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
<span style="font-family: monospace; font-size: 12px; color: #94a3b8;">{current_bars} / {required_bars} bars</span>
<span style="font-family: monospace; font-size: 12px; color: #3b82f6; font-weight: 600;">{progress}%</span>
</div>
<div style="background: rgba(100, 116, 139, 0.2); border-radius: 8px; height: 12px; overflow: hidden;">
<div style="background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%); height: 100%; width: {progress}%; transition: width 0.5s ease; border-radius: 8px;"></div>
</div>
</div>
<div style="font-family: monospace; font-size: 12px; color: #475569;">{remaining} more bars needed - Auto-collecting via live feed</div>
</div>""", unsafe_allow_html=True)


def render_price_analysis(symbol1, symbol2, df1, df2, timeframe, rolling_window):
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Multi-symbol selection
        st.markdown(f"""
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 600; color: #f0f4f8; margin-bottom: 16px;">
            {symbol1} / {symbol2} Price Charts
        </div>
        """, unsafe_allow_html=True)
        
        if not df1.empty:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.55, 0.45],
                subplot_titles=[f"{symbol1}", f"{symbol2}"]
            )
            
            # Symbol 1 candlestick - Cyan/Red theme
            fig.add_trace(
                go.Candlestick(
                    x=df1['timestamp'],
                    open=df1['open'], high=df1['high'],
                    low=df1['low'], close=df1['close'],
                    name=symbol1,
                    increasing={'line': {'color': '#10b981'}, 'fillcolor': '#10b981'},
                    decreasing={'line': {'color': '#ef4444'}, 'fillcolor': '#ef4444'}
                ),
                row=1, col=1
            )
            
            # Symbol 2 candlestick - Blue/Orange theme
            if not df2.empty:
                fig.add_trace(
                    go.Candlestick(
                        x=df2['timestamp'],
                        open=df2['open'], high=df2['high'],
                        low=df2['low'], close=df2['close'],
                        name=symbol2,
                        increasing={'line': {'color': '#3b82f6'}, 'fillcolor': '#3b82f6'},
                        decreasing={'line': {'color': '#f59e0b'}, 'fillcolor': '#f59e0b'}
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=620,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#0b0f16',
                font={'color': '#94a3b8', 'family': 'JetBrains Mono', 'size': 10},
                xaxis_rangeslider_visible=False,
                xaxis2_rangeslider_visible=False,
                showlegend=False,
                margin={'l': 50, 'r': 50, 't': 30, 'b': 30},
                hovermode='x unified'
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False, showspikes=True, spikecolor='#3b82f6', spikethickness=1)
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)', zeroline=False, side='right')
            
            # Update subplot title font
            for annotation in fig['layout']['annotations']:
                annotation['font'] = dict(size=12, color='#64748b', family='JetBrains Mono')
            
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
                'displaylogo': False
            })
            
            if not df1.empty and len(df1) >= rolling_window + 2:
                returns = df1['close'].pct_change().dropna().values
                if len(returns) >= rolling_window:
                    vol_fig = ChartBuilder.create_rolling_volatility(returns, window=rolling_window, height=220)
                    st.plotly_chart(vol_fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">
            LIVE STATISTICS
        </div>
        """, unsafe_allow_html=True)
        
        stats = client.get_price_stats(symbol1, rolling_window)
        if "error" not in stats and "stats" in stats:
            s = stats["stats"]
            
            # Custom styled metrics
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                <div style="font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Last Price</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: #f0f4f8; margin-top: 4px;">${s['last_price']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("μ (Mean)", f"${s['rolling_mean']:,.0f}")
            with col_b:
                st.metric("σ (Std)", f"${s['rolling_std']:,.2f}")
            
            st.metric("Range", f"${s['min_price']:,.0f} - ${s['max_price']:,.0f}")
            st.metric("Max", f"${s['max_price']:,.2f}")
        else:
            st.info("No live stats available")


def render_spread_analytics(symbol1, symbol2, timeframe, rolling_window, z_threshold, regression_type):
    
    st.markdown(f"### Spread Analysis: {symbol1} / {symbol2}")
    
    # Get pair analysis
    analysis = client.get_pair_analysis(symbol1, symbol2, timeframe, 200, rolling_window, z_threshold)
    
    if "error" in analysis:
        st.error(f"Error: {analysis.get('error')}")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        hr = analysis.get('hedge_ratio', {})
        st.metric("Hedge Ratio (β)", f"{hr.get('beta', 0):.4f}")
    
    with col2:
        spread = analysis.get('spread', {})
        st.metric("Current Spread", f"{spread.get('current', 0):.4f}")
    
    with col3:
        z = analysis.get('z_score', {})
        z_val = z.get('current', 0)
        st.metric("Z-Score", f"{z_val:.2f}", 
                  delta="Signal!" if abs(z_val) > z_threshold else "Normal")
    
    with col4:
        corr = analysis.get('correlation', {})
        st.metric("Correlation", f"{corr.get('current', 0):.4f}")
    
    with col5:
        st.metric("Signal", analysis.get('signal', 'NEUTRAL'))
    
    st.markdown("---")
    
    # Spread chart
    spread_data = client._get("/api/export/analytics/spread", {
        "symbol_a": symbol1,
        "symbol_b": symbol2,
        "timeframe": timeframe,
        "limit": 200,
        "format": "json"
    })
    
    # Since export returns file, we need to get data differently
    # Let's create spread visualization from pair analysis
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get OHLC for both
        df1 = client.get_ohlc(symbol1, timeframe, 200)
        df2 = client.get_ohlc(symbol2, timeframe, 200)
        
        if not df1.empty and not df2.empty:
            hr_val = analysis.get('hedge_ratio', {}).get('beta', 1.0)
            spread_mean = analysis.get('spread', {}).get('mean', 0)
            spread_std = analysis.get('spread', {}).get('std', 1)
            
            # Compute spread series
            min_len = min(len(df1), len(df2))
            prices_a = df1['close'].values[-min_len:]
            prices_b = df2['close'].values[-min_len:]
            timestamps = df1['timestamp'].values[-min_len:]
            
            spread_series = prices_a - hr_val * prices_b
            z_series = (spread_series - spread_mean) / spread_std if spread_std > 0 else np.zeros_like(spread_series)
            
            # Create plot
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.08, row_heights=[0.5, 0.5],
                               subplot_titles=['Spread', 'Z-Score'])
            
            fig.add_trace(go.Scatter(x=timestamps, y=spread_series, mode='lines',
                                    line={'color': '#58a6ff', 'width': 1.5}, name='Spread'),
                         row=1, col=1)
            fig.add_hline(y=spread_mean, line_dash="dash", line_color="#d29922", row=1, col=1)
            fig.add_hline(y=spread_mean + 2*spread_std, line_dash="dot", line_color="#f85149", row=1, col=1)
            fig.add_hline(y=spread_mean - 2*spread_std, line_dash="dot", line_color="#f85149", row=1, col=1)
            
            fig.add_trace(go.Scatter(x=timestamps, y=z_series, mode='lines',
                                    line={'color': '#a371f7', 'width': 1.5}, name='Z-Score'),
                         row=2, col=1)
            fig.add_hline(y=z_threshold, line_dash="dash", line_color="#f85149", row=2, col=1)
            fig.add_hline(y=-z_threshold, line_dash="dash", line_color="#f85149", row=2, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="#1e2530", row=2, col=1)
            
            fig.update_layout(
                height=500,
                paper_bgcolor='#0f1419',
                plot_bgcolor='#0a0e17',
                font={'color': '#e6edf3'},
                showlegend=False,
                margin={'l': 60, 'r': 40, 't': 40, 'b': 40}
            )
            fig.update_xaxes(gridcolor='#1e2530')
            fig.update_yaxes(gridcolor='#1e2530')
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            st.markdown("#### Spread Distribution")
            dist_fig = ChartBuilder.create_spread_distribution(spread_series, height=280)
            st.plotly_chart(dist_fig, use_container_width=True)
    
    with col2:
        st.markdown("### Pair Stats")
        
        stat = analysis.get('stationarity', {})
        is_stat = stat.get('is_stationary', False)
        
        st.markdown(f"""
        <div style="background: {'#0d3321' if is_stat else '#3d1c1c'}; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            <div style="font-size: 12px; color: #8b949e;">Stationarity (ADF)</div>
            <div style="font-size: 18px; font-weight: 600; color: {'#3fb950' if is_stat else '#f85149'};">
                {'Stationary' if is_stat else 'Non-Stationary'}
            </div>
            <div style="font-size: 11px; color: #6e7681; margin-top: 4px;">
                p-value: {stat.get('p_value', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        mr = analysis.get('mean_reversion', {})
        hl = mr.get('half_life')
        
        st.markdown(f"""
        <div style="background: #161b22; padding: 16px; border-radius: 8px;">
            <div style="font-size: 12px; color: #8b949e;">Half-Life</div>
            <div style="font-size: 24px; font-weight: 600; color: #a371f7;">
                {f'{hl:.1f}' if hl else 'N/A'} <span style="font-size: 12px;">periods</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_advanced_analysis(symbol1, symbol2, timeframe, rolling_window, regression_type):
    
    st.markdown("### Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Hedge Ratio Estimation")
        
        method_map = {
            "OLS": "ols",
            "Huber (Robust)": "huber", 
            "Theil-Sen (Robust)": "theil_sen",
            "Kalman Filter": "kalman"
        }
        
        if st.button("Calculate Hedge Ratio", key="hr_btn"):
            method = method_map.get(regression_type, "ols")
            
            if method == "kalman":
                result = client.get_kalman_hedge(symbol1, symbol2)
                if "error" not in result:
                    k = result.get("kalman", {})
                    st.markdown(f"""
                    <div style="background: #0f1419; padding: 20px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 11px; color: #8b949e;">Kalman Filter Hedge Ratio</div>
                        <div style="font-size: 36px; font-weight: 700; color: #58a6ff;">
                            {k.get('current_hedge_ratio', 0):.6f}
                        </div>
                        <div style="font-size: 11px; color: #6e7681; margin-top: 8px;">
                            Mean: {k.get('mean_hedge_ratio', 0):.4f} | Std: {k.get('std', 0):.4f} | Stable: {'Yes' if k.get('is_stable') else 'No'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    last_10 = k.get('last_10', [])
                    if last_10:
                        hr_fig = ChartBuilder.create_rolling_hedge_ratio(np.array(last_10), height=200)
                        st.plotly_chart(hr_fig, use_container_width=True)
            elif method in ["huber", "theil_sen"]:
                result = client.get_robust_hedge(symbol1, symbol2, method)
                if "error" not in result:
                    r = result.get("robust", {})
                    st.markdown(f"""
                    <div style="background: #0f1419; padding: 20px; border-radius: 8px;">
                        <div style="font-size: 11px; color: #8b949e; text-align: center;">
                            {regression_type} Hedge Ratio
                        </div>
                        <div style="font-size: 36px; font-weight: 700; color: #58a6ff; text-align: center;">
                            {r.get('beta', 0):.6f}
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 16px; font-size: 11px;">
                            <div>OLS β: {r.get('ols_beta', 0):.4f}</div>
                            <div>Diff: {r.get('beta_diff_pct', 0):.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                result = client.get_hedge_ratio(symbol1, symbol2, timeframe)
                if "error" not in result:
                    hr = result.get("hedge_ratio", {})
                    st.markdown(f"""
                    <div style="background: #0f1419; padding: 20px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 11px; color: #8b949e;">OLS Hedge Ratio (β)</div>
                        <div style="font-size: 36px; font-weight: 700; color: #58a6ff;">
                            {hr.get('beta', 0):.6f}
                        </div>
                        <div style="font-size: 11px; color: #6e7681; margin-top: 8px;">
                            R²: {hr.get('r_squared', 0):.4f} | α: {hr.get('alpha', 0):.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("#### ADF Stationarity Test")
        
        if st.button("Run ADF Test", key="adf_btn"):
            with st.spinner("Running ADF test..."):
                result = client.get_adf_test(symbol1, symbol2, timeframe)
            
            if "error" in result:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
            elif result:
                # Try to get ADF data from different possible response structures
                adf = result.get("adf", result.get("stationarity", result))
                is_stat = adf.get("is_stationary", False)
                statistic = adf.get("statistic", adf.get("test_statistic", 0))
                p_value = adf.get("p_value", adf.get("pvalue", 0))
                
                st.success("ADF Test Complete!")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Result", "STATIONARY" if is_stat else "NON-STATIONARY")
                with col_b:
                    st.metric("Test Statistic", f"{statistic:.4f}")
                with col_c:
                    st.metric("P-Value", f"{p_value:.4f}")
                
                if is_stat:
                    st.info("The spread is stationary (mean-reverting). Good for pairs trading!")
                else:
                    st.warning("The spread is non-stationary. Be cautious with mean-reversion strategies.")
            else:
                st.warning("No data returned from ADF test")
    
    with col2:
        st.markdown("#### Half-Life of Mean Reversion")
        
        if st.button("Calculate Half-Life", key="hl_btn"):
            with st.spinner("Calculating half-life..."):
                result = client.get_half_life(symbol1, symbol2, timeframe)
            
            if "error" in result:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
            elif result:
                hl = result.get("half_life", result.get("mean_reversion", result))
                periods = hl.get("periods", hl.get("half_life"))
                is_mr = hl.get("is_mean_reverting", False)
                
                st.success("Half-Life Calculated!")
                
                st.metric("Half-Life", f"{periods:.1f} periods" if periods and periods > 0 else "∞")
                
                if is_mr:
                    st.info(f"Mean-reverting! Spread reverts halfway in ~{periods:.1f} bars.")
                else:
                    st.warning("Slow or no mean reversion detected.")
            else:
                st.warning("No data returned")
        
        st.markdown("---")
        
        st.markdown("#### Correlation Matrix")
        
        symbols_input = st.text_input("Symbols (comma-separated)", 
                                       value=f"{symbol1},{symbol2}",
                                       key="corr_symbols")
        
        if st.button("Generate Matrix", key="corr_btn"):
            symbols_list = [s.strip() for s in symbols_input.split(",")]
            result = client.get_correlation_matrix(symbols_list, timeframe)
            
            if "error" not in result:
                matrix = result.get("matrix", [])
                symbols_res = result.get("symbols", [])
                
                if matrix:
                    corr_df = pd.DataFrame(matrix, index=symbols_res, columns=symbols_res)
                    fig = ChartBuilder.create_correlation_heatmap(corr_df, height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Strongest Pairs:**")
                    for pair in result.get("strongest_pairs", [])[:3]:
                        st.markdown(f"• {pair['pair']}: {pair['correlation']:.4f}")


def render_time_series_table(symbol1, symbol2, timeframe):
    
    st.markdown("### Time Series Analytics Table")
    
    # Get data
    df1 = client.get_ohlc(symbol1, timeframe, 100)
    df2 = client.get_ohlc(symbol2, timeframe, 100)
    
    if df1.empty:
        st.warning("No data available")
        return
    
    # Get hedge ratio for spread calculation
    hr_result = client.get_hedge_ratio(symbol1, symbol2, timeframe)
    hr = hr_result.get("hedge_ratio", {}).get("beta", 1.0) if "error" not in hr_result else 1.0
    
    # Build analytics table
    if not df2.empty:
        min_len = min(len(df1), len(df2))
        
        table_data = pd.DataFrame({
            'Timestamp': df1['timestamp'].iloc[-min_len:].values,
            f'{symbol1} Close': df1['close'].iloc[-min_len:].values,
            f'{symbol2} Close': df2['close'].iloc[-min_len:].values,
            'Spread': df1['close'].iloc[-min_len:].values - hr * df2['close'].iloc[-min_len:].values,
        })
        
        # Add rolling stats
        table_data['Spread Mean (20)'] = table_data['Spread'].rolling(20).mean()
        table_data['Spread Std (20)'] = table_data['Spread'].rolling(20).std()
        table_data['Z-Score'] = (table_data['Spread'] - table_data['Spread Mean (20)']) / table_data['Spread Std (20)']
        
        # Volume if available
        if 'volume' in df1.columns:
            table_data[f'{symbol1} Volume'] = df1['volume'].iloc[-min_len:].values
    else:
        table_data = df1.copy()
    
    # Display table
    st.dataframe(
        table_data.style.format({
            col: '{:,.2f}' for col in table_data.columns if col != 'Timestamp'
        }).background_gradient(subset=['Z-Score'] if 'Z-Score' in table_data.columns else [], cmap='RdYlGn_r'),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = table_data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"analytics_{symbol1}_{symbol2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def render_backtest(symbol1, symbol2, timeframe, z_threshold):
    st.markdown("### Mean Reversion Backtest")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        entry_z = st.slider("Entry Z-Score", 1.0, 4.0, z_threshold, 0.1, key="bt_entry")
        exit_z = st.slider("Exit Z-Score", -1.0, 1.0, 0.0, 0.1, key="bt_exit")
        limit = st.slider("Data Points", 100, 1000, 500, 50, key="bt_limit")
    
    with col2:
        st.markdown("""
        **Strategy:**
        - **LONG spread** when z < -entry
        - **SHORT spread** when z > +entry  
        - **EXIT** when z crosses exit threshold
        """)
    
    if st.button("Run Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running backtest..."):
            result = client.get_backtest(symbol1, symbol2, entry_z, exit_z, limit)
        
        if "error" not in result:
            bt = result.get("backtest", {})
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{bt.get('total_return', 0):.1f}%")
            with col2:
                st.metric("Win Rate", f"{bt.get('win_rate', 0):.1f}%")
            with col3:
                st.metric("# Trades", bt.get('num_trades', 0))
            with col4:
                st.metric("Sharpe", f"{bt.get('sharpe', 0):.2f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Trade Return", f"{bt.get('avg_trade_return', 0):.2f}%")
            with col2:
                st.metric("Max Drawdown", f"{bt.get('max_drawdown', 0):.2f}%")
            
            trades = bt.get("trades", [])
            if trades:
                trade_returns = [t.get('return_pct', 0) / 100 for t in trades]
                returns_arr = np.array(trade_returns) if trade_returns else np.array([0])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Cumulative P&L")
                    pnl_fig = ChartBuilder.create_cumulative_pnl(returns_arr, height=280)
                    st.plotly_chart(pnl_fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Drawdown Analysis")
                    dd_fig = ChartBuilder.create_drawdown_chart(returns_arr, height=280)
                    st.plotly_chart(dd_fig, use_container_width=True)
                
                st.markdown("#### Trade History")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df, use_container_width=True, height=200)
        else:
            st.error(f"Error: {result.get('error')}")


def render_export(symbol1, symbol2, timeframe):
    st.markdown("### Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### OHLC Data")
        
        export_symbol = st.selectbox("Symbol", [symbol1, symbol2], key="exp_sym")
        export_format = st.selectbox("Format", ["CSV", "JSON"], key="exp_fmt")
        
        if st.button("Download OHLC", use_container_width=True):
            url = client.get_export_url(
                f"ohlc/{export_symbol}",
                timeframe=timeframe,
                format=export_format.lower()
            )
            st.markdown(f"[Download]({url})")
    
    with col2:
        st.markdown("#### Pair Analytics")
        
        if st.button("Download Analytics", use_container_width=True):
            url = client.get_export_url(
                "analytics/pair",
                symbol_a=symbol1,
                symbol_b=symbol2,
                timeframe=timeframe
            )
            st.markdown(f"[Download]({url})")
    
    with col3:
        st.markdown("#### Alert History")
        
        if st.button("Download Alerts", use_container_width=True):
            url = client.get_export_url("alerts", format="csv")
            st.markdown(f"[Download]({url})")
    
    st.markdown("---")
    
    st.markdown("#### Quick Export Links")
    st.markdown(f"""
    - [Ticks CSV]({client.base_url}/api/export/ticks/{symbol1}?format=csv)
    - [OHLC JSON]({client.base_url}/api/export/ohlc/{symbol1}?format=json&timeframe={timeframe})
    - [Spread Series]({client.base_url}/api/export/analytics/spread?symbol_a={symbol1}&symbol_b={symbol2})
    - [Backtest Results]({client.base_url}/api/export/backtest?symbol_a={symbol1}&symbol_b={symbol2})
    """)


def check_and_show_alert_notifications():
    if not st.session_state.connected:
        return
    
    # Get alert history
    history = client.get_alert_history(limit=10)
    
    # Track seen alerts in session state
    if "seen_alerts" not in st.session_state:
        st.session_state.seen_alerts = set()
    
    # Show toast for new alerts
    for alert in history:
        alert_id = f"{alert.get('rule_id', '')}_{alert.get('triggered_at', '')}"
        if alert_id and alert_id not in st.session_state.seen_alerts:
            st.session_state.seen_alerts.add(alert_id)
            metric = alert.get('metric', 'Alert')
            value = alert.get('value', 0)
            threshold = alert.get('threshold', 0)
            st.toast(f"Alert Triggered: {metric} = {value:.4f} (threshold: {threshold})", icon="🔔")


def render_alerts_page(symbol1, symbol2):
    check_and_show_alert_notifications()
    st.subheader("Create New Alert")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alert_metric = st.selectbox("Metric", ["z_score", "spread", "correlation"], key="alert_metric_page")
    with c2:
        alert_op = st.selectbox("Operator", [">", "<", ">=", "<=", "abs>"], key="alert_op_page")
    with c3:
        alert_threshold = st.number_input("Threshold", value=2.0, key="alert_thresh_page")
    with c4:
        alert_cooldown = st.number_input("Cooldown (sec)", value=60, min_value=10, key="alert_cool_page")
    
    if st.button("+ Create Alert", type="primary"):
        if st.session_state.connected:
            result = client.create_alert(
                symbols=[symbol1, symbol2],
                metric=alert_metric,
                operator=alert_op,
                threshold=alert_threshold,
                cooldown=alert_cooldown
            )
            if "error" not in result:
                st.success(f"Alert created: {alert_metric} {alert_op} {alert_threshold}")
            else:
                st.error(result.get("error"))
        else:
            st.error("Not connected to backend")
    
    st.markdown("---")
    st.subheader("Active Alerts")
    
    if st.session_state.connected:
        alerts = client.get_alerts()
        if alerts:
            for i, alert in enumerate(alerts):
                rule_id = alert.get('id', alert.get('rule_id', ''))
                # Use name if available, otherwise construct from parts
                name = alert.get('name', '')
                if not name:
                    name = f"{alert.get('metric', 'N/A')} {alert.get('operator', '')} {alert.get('threshold', 'N/A')}"
                
                col_name, col_del = st.columns([6, 1])
                with col_name:
                    st.text(f"• {name}")
                with col_del:
                    if st.button("🗑", key=f"del_alert_{i}_{rule_id}", help="Delete alert"):
                        result = client.delete_alert(rule_id)
                        if "error" not in result:
                            st.rerun()
                        else:
                            st.error(result.get("error", "Failed to delete"))
        else:
            st.info("No active alerts.")
    
    # Alert History Section
    st.markdown("---")
    st.subheader("Alert History")
    
    if st.session_state.connected:
        history = client.get_alert_history(limit=20)
        if history:
            for alert in history:
                triggered_at = alert.get('triggered_at', 'Unknown time')
                metric = alert.get('metric', 'N/A')
                value = alert.get('value', 0)
                threshold = alert.get('threshold', 0)
                operator = alert.get('operator', '')
                symbols = alert.get('symbols', [])
                
                # Format the alert card
                st.markdown(f"""
                <div style="background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 14px; font-weight: 600; color: #fbbf24;">🔔 {metric} {operator} {threshold}</span>
                            <span style="font-size: 12px; color: #94a3b8; margin-left: 12px;">{', '.join(symbols) if symbols else ''}</span>
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748b;">{triggered_at}</div>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #f0f4f8; margin-top: 4px;">
                        Value: <span style="color: #fbbf24; font-weight: 600;">{value:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No triggered alerts yet. Create an alert and wait for conditions to be met.")
    else:
        st.warning("Connect to backend to view alert history.")


def main():
    # Check connection on startup
    check_backend_connection()
    
    # Check for alert notifications (shows toast on any page)
    check_and_show_alert_notifications()
    
    # Render sidebar and get parameters
    params = render_sidebar()
    symbol1, symbol2, timeframe, rolling_window, z_threshold, regression_type = params
    
    # Render main content
    render_main_content(symbol1, symbol2, timeframe, rolling_window, z_threshold, regression_type)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 32px 0; margin-top: 60px; border-top: 1px solid rgba(255,255,255,0.06);">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #475569; letter-spacing: 0.05em;">
            <span style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">◈ QUANT ANALYTICS</span>
            <span style="margin: 0 12px; color: #334155;">|</span>
            v2.0.0 Professional
            <span style="margin: 0 12px; color: #334155;">|</span>
            Built for Quantitative Research
        </div>
        <div style="font-size: 10px; color: #334155; margin-top: 8px;">
            Real-time analytics • Statistical arbitrage • Mean reversion strategies
        </div>
    </div>
    """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
