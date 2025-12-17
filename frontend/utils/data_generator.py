"""
Dummy Data Generator for Quant Analytics Dashboard
Generates realistic-looking market data for demonstration purposes
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import random


class DataGenerator:
    """Generate realistic dummy market data for demonstration"""
    
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT']
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
    
    @staticmethod
    def generate_tick_data(
        symbol: str = 'BTCUSDT',
        n_ticks: int = 5000,
        start_price: float = 42000.0,
        volatility: float = 0.0002,
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate realistic tick-level data with bid/ask spreads"""
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=2)
        
        # Generate timestamps with realistic irregular intervals
        intervals = np.random.exponential(0.5, n_ticks)  # seconds between ticks
        timestamps = [start_time]
        for interval in intervals[:-1]:
            timestamps.append(timestamps[-1] + timedelta(seconds=interval))
        
        # Generate price path using geometric Brownian motion
        returns = np.random.normal(0, volatility, n_ticks)
        price_path = start_price * np.exp(np.cumsum(returns))
        
        # Add mean reversion component
        mean_price = start_price
        reversion_strength = 0.001
        for i in range(1, len(price_path)):
            price_path[i] += reversion_strength * (mean_price - price_path[i-1])
        
        # Generate bid/ask with realistic spread
        base_spread = start_price * 0.0001  # 1 basis point spread
        spread_variation = np.random.uniform(0.8, 1.5, n_ticks) * base_spread
        
        bid_prices = price_path - spread_variation / 2
        ask_prices = price_path + spread_variation / 2
        
        # Generate volumes
        volumes = np.random.lognormal(mean=2, sigma=1, size=n_ticks) * 0.1
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'bid': np.round(bid_prices, 2),
            'ask': np.round(ask_prices, 2),
            'mid': np.round(price_path, 2),
            'spread': np.round(ask_prices - bid_prices, 4),
            'volume': np.round(volumes, 4),
            'side': np.random.choice(['buy', 'sell'], n_ticks, p=[0.52, 0.48])
        })
        
        return df
    
    @staticmethod
    def generate_ohlc_data(
        symbol: str = 'BTCUSDT',
        n_bars: int = 500,
        timeframe: str = '1m',
        start_price: float = 42000.0,
        volatility: float = 0.001,
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate OHLC candlestick data"""
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=n_bars)
        
        # Timeframe to timedelta mapping
        tf_map = {
            '1s': timedelta(seconds=1),
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1)
        }
        delta = tf_map.get(timeframe, timedelta(minutes=1))
        
        timestamps = [start_time + delta * i for i in range(n_bars)]
        
        # Generate price path
        returns = np.random.normal(0, volatility, n_bars)
        close_prices = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close prices
        opens = np.roll(close_prices, 1)
        opens[0] = start_price
        
        # High and low within each bar
        bar_volatility = volatility * np.random.uniform(0.5, 2, n_bars)
        highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, bar_volatility)))
        lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, bar_volatility)))
        
        # Volume with intraday pattern
        hour_of_day = np.array([t.hour for t in timestamps])
        volume_pattern = 1 + 0.5 * np.sin((hour_of_day - 9) * np.pi / 12)  # Peak at market hours
        volumes = np.random.lognormal(mean=8, sigma=0.5, size=n_bars) * volume_pattern
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'open': np.round(opens, 2),
            'high': np.round(highs, 2),
            'low': np.round(lows, 2),
            'close': np.round(close_prices, 2),
            'volume': np.round(volumes, 2)
        })
        
        return df
    
    @staticmethod
    def generate_pair_data(
        symbol1: str = 'BTCUSDT',
        symbol2: str = 'ETHUSDT',
        n_bars: int = 500,
        timeframe: str = '1m',
        correlation: float = 0.85
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate correlated pair data for spread analysis"""
        
        # Generate base returns
        base_returns = np.random.normal(0, 0.001, n_bars)
        
        # Generate correlated returns
        noise1 = np.random.normal(0, 0.0005, n_bars)
        noise2 = np.random.normal(0, 0.0005, n_bars)
        
        returns1 = correlation * base_returns + np.sqrt(1 - correlation**2) * noise1
        returns2 = correlation * base_returns + np.sqrt(1 - correlation**2) * noise2
        
        start_time = datetime.now() - timedelta(minutes=n_bars)
        
        # Different base prices for different assets
        price_map = {
            'BTCUSDT': 42000.0,
            'ETHUSDT': 2200.0,
            'BNBUSDT': 320.0,
            'SOLUSDT': 95.0,
            'XRPUSDT': 0.62,
            'ADAUSDT': 0.45
        }
        
        price1 = price_map.get(symbol1, 100.0)
        price2 = price_map.get(symbol2, 100.0)
        
        df1 = DataGenerator.generate_ohlc_data(symbol1, n_bars, timeframe, price1, start_time=start_time)
        df2 = DataGenerator.generate_ohlc_data(symbol2, n_bars, timeframe, price2, start_time=start_time)
        
        # Inject correlation into close prices
        df1['close'] = price1 * np.exp(np.cumsum(returns1))
        df2['close'] = price2 * np.exp(np.cumsum(returns2))
        
        return df1, df2
    
    @staticmethod
    def generate_data_quality_issues(df: pd.DataFrame, issue_rate: float = 0.003) -> Tuple[pd.DataFrame, dict]:
        """Inject realistic data quality issues for demonstration"""
        
        n = len(df)
        n_issues = int(n * issue_rate)
        
        issues = {
            'out_of_order': 0,
            'duplicates': 0,
            'price_spikes': 0,
            'gaps': 0,
            'total_ticks': n,
            'clean_rate': 0.0
        }
        
        df_with_issues = df.copy()
        
        # Out of order timestamps
        if n_issues > 0:
            swap_indices = random.sample(range(1, n-1), min(n_issues, n-2))
            for idx in swap_indices[:n_issues//3]:
                if idx + 1 < len(df_with_issues):
                    df_with_issues.loc[idx, 'timestamp'], df_with_issues.loc[idx+1, 'timestamp'] = \
                        df_with_issues.loc[idx+1, 'timestamp'], df_with_issues.loc[idx, 'timestamp']
                    issues['out_of_order'] += 1
        
        # Duplicates
        dup_count = n_issues // 4
        if dup_count > 0:
            issues['duplicates'] = dup_count
        
        # Price spikes (will be flagged but not injected into data)
        issues['price_spikes'] = max(0, n_issues // 5)
        
        issues['clean_rate'] = round((n - sum([issues['out_of_order'], issues['duplicates'], issues['price_spikes']])) / n * 100, 2)
        
        return df_with_issues, issues
    
    @staticmethod
    def generate_live_tick() -> dict:
        """Generate a single live tick for real-time simulation"""
        
        symbols = DataGenerator.SYMBOLS
        symbol = random.choice(symbols)
        
        base_prices = {
            'BTCUSDT': 42000 + random.uniform(-500, 500),
            'ETHUSDT': 2200 + random.uniform(-50, 50),
            'BNBUSDT': 320 + random.uniform(-10, 10),
            'SOLUSDT': 95 + random.uniform(-5, 5),
            'XRPUSDT': 0.62 + random.uniform(-0.02, 0.02),
            'ADAUSDT': 0.45 + random.uniform(-0.02, 0.02)
        }
        
        price = base_prices[symbol]
        spread = price * 0.0001 * random.uniform(0.8, 1.5)
        
        return {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'bid': round(price - spread/2, 4),
            'ask': round(price + spread/2, 4),
            'mid': round(price, 4),
            'spread': round(spread, 6),
            'volume': round(random.lognormal(2, 1) * 0.1, 4),
            'side': random.choice(['buy', 'sell'])
        }
    
    @staticmethod
    def generate_alerts_history(n_alerts: int = 10) -> List[dict]:
        """Generate sample alert history"""
        
        alert_types = [
            ('z_score_breach', 'Z-score crossed threshold'),
            ('spread_anomaly', 'Spread widened abnormally'),
            ('volume_spike', 'Volume spike detected'),
            ('correlation_break', 'Correlation breakdown detected'),
            ('data_gap', 'Data gap detected')
        ]
        
        alerts = []
        base_time = datetime.now()
        
        for i in range(n_alerts):
            alert_type, message = random.choice(alert_types)
            z_val = round(random.uniform(1.8, 3.5), 2)
            
            alerts.append({
                'timestamp': base_time - timedelta(minutes=random.randint(1, 120)),
                'type': alert_type,
                'message': f"{message} ({z_val:.2f})",
                'severity': random.choice(['warning', 'critical']) if z_val > 2.5 else 'info',
                'symbol': random.choice(DataGenerator.SYMBOLS)
            })
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    @staticmethod
    def generate_system_metrics() -> dict:
        """Generate realistic system performance metrics"""
        
        return {
            'ingestion_latency_ms': round(random.uniform(15, 65), 1),
            'analytics_compute_ms': round(random.uniform(8, 35), 1),
            'alert_trigger_ms': round(random.uniform(40, 120), 1),
            'memory_usage_mb': round(random.uniform(128, 512), 0),
            'cpu_usage_pct': round(random.uniform(5, 45), 1),
            'ticks_per_second': round(random.uniform(50, 500), 0),
            'uptime_hours': round(random.uniform(1, 72), 1),
            'websocket_status': 'connected' if random.random() > 0.05 else 'reconnecting'
        }

