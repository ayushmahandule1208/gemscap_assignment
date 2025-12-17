"""
Backend API Client
Connects Streamlit frontend to FastAPI backend.
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class APIClient:
    """Client for backend API communication"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make GET request"""
        try:
            resp = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Backend not connected. Start backend with: uvicorn main:app --reload"}
        except Exception as e:
            return {"error": str(e)}
    
    def _post(self, endpoint: str, data: dict = None, files: dict = None) -> dict:
        """Make POST request"""
        try:
            if files:
                resp = self.session.post(f"{self.base_url}{endpoint}", files=files, timeout=30)
            else:
                resp = self.session.post(f"{self.base_url}{endpoint}", json=data, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Backend not connected"}
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # Health & Status
    # =========================================================================
    
    def health(self) -> dict:
        """Check backend health"""
        return self._get("/health")
    
    def is_connected(self) -> bool:
        """Check if backend is reachable"""
        result = self.health()
        return "error" not in result
    
    # =========================================================================
    # Data Upload
    # =========================================================================
    
    def upload_csv(self, file_content: bytes, filename: str, symbol: str = None) -> dict:
        """Upload CSV file"""
        # Extract symbol from filename if not provided
        if symbol is None:
            # Try to extract from filename like "btcusdt_demo.csv" -> "BTCUSDT"
            base_name = filename.lower().replace('.csv', '').replace('_demo', '').replace('_', '')
            symbol = base_name.upper() if base_name else "UNKNOWN"
        
        files = {"file": (filename, file_content, "text/csv")}
        try:
            resp = self.session.post(
                f"{self.base_url}/api/upload/csv?symbol={symbol}",
                files=files,
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return {"error": "Backend not connected"}
        except Exception as e:
            return {"error": str(e)}
    
    def upload_ndjson(self, file_content: bytes, filename: str) -> dict:
        """Upload NDJSON file"""
        files = {"file": (filename, file_content, "application/x-ndjson")}
        return self._post("/api/upload/ndjson", files=files)
    
    def upload_ticks(self, ticks: List[dict]) -> dict:
        """Upload tick batch"""
        return self._post("/api/upload/ticks", data={"ticks": ticks})
    
    # =========================================================================
    # Data Retrieval
    # =========================================================================
    
    def get_symbols(self) -> List[str]:
        """Get available symbols"""
        result = self._get("/api/data/symbols")
        return result.get("symbols", [])
    
    def get_ticks(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get tick data"""
        result = self._get(f"/api/data/{symbol}/ticks", {"limit": limit})
        if "error" in result or "data" not in result:
            return pd.DataFrame()
        return pd.DataFrame(result["data"])
    
    def get_ohlc(self, symbol: str, timeframe: str = "1m", limit: int = 500) -> pd.DataFrame:
        """Get OHLC data"""
        result = self._get(f"/api/data/{symbol}/ohlc", {"timeframe": timeframe, "limit": limit})
        if "error" in result or "data" not in result:
            return pd.DataFrame()
        df = pd.DataFrame(result["data"])
        if "timestamp" in df.columns:
            # Use ISO8601 format to handle timestamps with/without microseconds
            df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601')
        return df
    
    def get_prices(self, symbol: str, limit: int = 1000) -> List[float]:
        """Get price array"""
        result = self._get(f"/api/data/{symbol}/prices", {"limit": limit})
        return result.get("prices", [])
    
    # =========================================================================
    # Analytics
    # =========================================================================
    
    def get_price_stats(self, symbol: str, window: int = 20) -> dict:
        """Get live price statistics"""
        return self._get(f"/api/analytics/{symbol}/stats", {"window": window})
    
    def get_spread(self, symbol_a: str, symbol_b: str, window: int = 20) -> dict:
        """Get live spread and z-score"""
        return self._get("/api/analytics/spread", {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "window": window
        })
    
    def get_hedge_ratio(self, symbol_a: str, symbol_b: str, timeframe: str = "1m", 
                        limit: int = 100, method: str = "ols") -> dict:
        """Get hedge ratio"""
        return self._get("/api/analytics/hedge-ratio", {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "timeframe": timeframe,
            "limit": limit,
            "method": method
        })
    
    def get_robust_hedge(self, symbol_a: str, symbol_b: str, method: str = "huber") -> dict:
        """Get robust hedge ratio (Huber/Theil-Sen)"""
        return self._get("/api/analytics/robust-hedge", {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "method": method
        })
    
    def get_kalman_hedge(self, symbol_a: str, symbol_b: str, limit: int = 200) -> dict:
        """Get Kalman filter hedge ratio"""
        return self._get("/api/analytics/kalman-hedge", {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "limit": limit
        })
    
    def get_adf_test(self, symbol_a: str, symbol_b: str, timeframe: str = "1m") -> dict:
        """Run ADF stationarity test"""
        return self._get("/api/analytics/adf", {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "timeframe": timeframe
        })
    
    def get_half_life(self, symbol_a: str, symbol_b: str, timeframe: str = "1m") -> dict:
        """Get half-life of mean reversion"""
        return self._get("/api/analytics/half-life", {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "timeframe": timeframe
        })
    
    def get_pair_analysis(self, symbol_a: str, symbol_b: str, timeframe: str = "1m",
                          limit: int = 100, window: int = 20, z_threshold: float = 2.0) -> dict:
        """Get full pair analysis"""
        return self._get("/api/analytics/pair", {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "timeframe": timeframe,
            "limit": limit,
            "window": window,
            "z_threshold": z_threshold
        })
    
    def get_backtest(self, symbol_a: str, symbol_b: str, entry_z: float = 2.0,
                     exit_z: float = 0.0, limit: int = 500) -> dict:
        """Run mini backtest"""
        return self._get("/api/analytics/backtest", {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "limit": limit
        })
    
    def get_correlation_matrix(self, symbols: List[str], timeframe: str = "1m") -> dict:
        """Get cross-correlation matrix"""
        return self._get("/api/analytics/correlation-matrix", {
            "symbols": ",".join(symbols),
            "timeframe": timeframe
        })
    
    def get_liquidity(self, symbol: str, timeframe: str = "1m") -> dict:
        """Get liquidity metrics"""
        return self._get(f"/api/analytics/{symbol}/liquidity", {"timeframe": timeframe})
    
    # =========================================================================
    # Alerts
    # =========================================================================
    
    def create_alert(self, symbols: List[str], metric: str, operator: str, 
                     threshold: float, cooldown: int = 60, name: str = None) -> dict:
        """Create alert rule"""
        return self._post("/api/alerts/rules", {
            "symbols": symbols,
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "cooldown_sec": cooldown,
            "name": name
        })
    
    def get_alerts(self) -> List[dict]:
        """Get all alert rules"""
        result = self._get("/api/alerts/rules")
        return result.get("rules", [])
    
    def delete_alert(self, rule_id: str) -> dict:
        """Delete alert rule"""
        try:
            resp = self.session.delete(f"{self.base_url}/api/alerts/rules/{rule_id}", timeout=10)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_alert_history(self, limit: int = 50) -> List[dict]:
        """Get alert history"""
        result = self._get("/api/alerts/history", {"limit": limit})
        return result.get("alerts", [])
    
    def test_alert(self, z_score: float = None, spread: float = None, symbols: List[str] = None) -> dict:
        """Test alerts with mock data"""
        data = {"symbols": symbols or ["TEST"]}
        if z_score is not None:
            data["z_score"] = z_score
        if spread is not None:
            data["spread"] = spread
        return self._post("/api/alerts/test", data)
    
    # =========================================================================
    # Live Feed
    # =========================================================================
    
    def start_live_feed(self, symbols: List[str] = None) -> dict:
        """Start live Binance WebSocket feed"""
        if symbols is None:
            symbols = ["btcusdt", "ethusdt"]
        return self._post("/api/live/start", {"symbols": symbols})
    
    def stop_live_feed(self) -> dict:
        """Stop live feed"""
        return self._post("/api/live/stop")
    
    def get_live_status(self) -> dict:
        """Get live feed status"""
        return self._get("/api/live/status")
    
    def get_live_symbols(self) -> dict:
        """Get popular symbols for live feed"""
        return self._get("/api/live/symbols")
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def get_export_url(self, export_type: str, **params) -> str:
        """Get export download URL"""
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.base_url}/api/export/{export_type}?{param_str}"
    
    # =========================================================================
    # Data Availability (Progressive Feature Enabling)
    # =========================================================================
    
    def get_data_availability(self) -> dict:
        """
        Get data availability and feature unlock status.
        
        Returns:
            - Per-symbol bar/tick counts
            - Feature requirements and unlock status
            - Progress toward unlocking features
        """
        return self._get("/api/data/availability")
    
    def get_symbol_availability(self, symbol: str, timeframe: str = "1m") -> dict:
        """Get data availability for a specific symbol"""
        return self._get(f"/api/data/availability/{symbol}", {"timeframe": timeframe})


# Global client instance
_client: Optional[APIClient] = None


def get_client(base_url: str = "http://localhost:8000") -> APIClient:
    """Get or create API client singleton"""
    global _client
    if _client is None:
        _client = APIClient(base_url)
    return _client

