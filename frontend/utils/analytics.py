"""
Analytics Engine for Quant Analytics Dashboard
Provides rolling statistics, z-scores, and statistical tests
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats


class AnalyticsEngine:
    """Compute quantitative analytics on market data"""
    
    @staticmethod
    def compute_rolling_stats(
        prices: pd.Series,
        window: int = 20,
        enforce_no_lookahead: bool = True
    ) -> pd.DataFrame:
        """Compute rolling mean, std, and z-score"""
        
        if enforce_no_lookahead:
            # Use only past data (shift by 1 to exclude current observation)
            rolling_mean = prices.rolling(window=window, min_periods=window).mean().shift(1)
            rolling_std = prices.rolling(window=window, min_periods=window).std().shift(1)
        else:
            rolling_mean = prices.rolling(window=window, min_periods=window).mean()
            rolling_std = prices.rolling(window=window, min_periods=window).std()
        
        z_score = (prices - rolling_mean) / rolling_std
        
        return pd.DataFrame({
            'price': prices,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'z_score': z_score,
            'upper_band': rolling_mean + 2 * rolling_std,
            'lower_band': rolling_mean - 2 * rolling_std
        })
    
    @staticmethod
    def compute_spread_analytics(
        prices1: pd.Series,
        prices2: pd.Series,
        window: int = 20,
        hedge_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """Compute spread analytics between two price series"""
        
        if hedge_ratio is None:
            # Calculate hedge ratio using OLS
            hedge_ratio = AnalyticsEngine.compute_hedge_ratio(prices1, prices2)
        
        # Calculate spread
        spread = prices1 - hedge_ratio * prices2
        
        # Rolling statistics on spread
        spread_stats = AnalyticsEngine.compute_rolling_stats(spread, window)
        spread_stats['hedge_ratio'] = hedge_ratio
        spread_stats['spread'] = spread
        
        return spread_stats
    
    @staticmethod
    def compute_hedge_ratio(
        prices1: pd.Series,
        prices2: pd.Series,
        method: str = 'ols'
    ) -> float:
        """Calculate hedge ratio between two price series"""
        
        # Remove NaN values
        valid_mask = ~(prices1.isna() | prices2.isna())
        p1 = prices1[valid_mask].values
        p2 = prices2[valid_mask].values
        
        if len(p1) < 2:
            return 1.0
        
        if method == 'ols':
            slope, intercept, r_value, p_value, std_err = stats.linregress(p2, p1)
            return slope
        elif method == 'tls':
            # Total Least Squares
            mean1, mean2 = np.mean(p1), np.mean(p2)
            cov_matrix = np.cov(p1 - mean1, p2 - mean2)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            idx = np.argmin(eigenvalues)
            return -eigenvectors[0, idx] / eigenvectors[1, idx]
        else:
            return 1.0
    
    @staticmethod
    def compute_rolling_correlation(
        prices1: pd.Series,
        prices2: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Compute rolling correlation between two price series"""
        
        returns1 = prices1.pct_change()
        returns2 = prices2.pct_change()
        
        rolling_corr = returns1.rolling(window=window).corr(returns2)
        
        return rolling_corr
    
    @staticmethod
    def adf_test(series: pd.Series) -> dict:
        """Augmented Dickey-Fuller test for stationarity"""
        
        clean_series = series.dropna()
        
        if len(clean_series) < 20:
            return {
                'statistic': None,
                'p_value': None,
                'is_stationary': None,
                'critical_values': {},
                'message': 'Insufficient data for ADF test'
            }
        
        try:
            from scipy.stats import norm
            
            # Simplified ADF-like test using first differences
            diff = clean_series.diff().dropna()
            lagged = clean_series.shift(1).dropna()
            
            # Align series
            min_len = min(len(diff), len(lagged))
            diff = diff.iloc[:min_len]
            lagged = lagged.iloc[:min_len]
            
            # Simple regression coefficient
            if lagged.std() > 0:
                rho = np.corrcoef(diff.values, lagged.values[:-1] if len(lagged) > len(diff) else lagged.values[:len(diff)])[0, 1]
                t_stat = rho * np.sqrt(len(diff) - 2) / np.sqrt(1 - rho**2) if abs(rho) < 1 else 0
            else:
                t_stat = 0
            
            # Approximate p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            
            return {
                'statistic': round(t_stat, 4),
                'p_value': round(p_value, 4),
                'is_stationary': p_value < 0.05,
                'critical_values': {
                    '1%': -3.43,
                    '5%': -2.86,
                    '10%': -2.57
                },
                'message': 'Stationary (reject H0)' if p_value < 0.05 else 'Non-stationary (fail to reject H0)'
            }
        except Exception as e:
            return {
                'statistic': None,
                'p_value': None,
                'is_stationary': None,
                'critical_values': {},
                'message': f'Test failed: {str(e)}'
            }
    
    @staticmethod
    def half_life(series: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return float('inf')
        
        lagged = clean_series.shift(1)
        delta = clean_series - lagged
        
        # Remove NaN
        valid_mask = ~(lagged.isna() | delta.isna())
        lagged_clean = lagged[valid_mask].values
        delta_clean = delta[valid_mask].values
        
        if len(lagged_clean) < 2:
            return float('inf')
        
        # Linear regression: delta = lambda * lagged + epsilon
        slope, _, _, _, _ = stats.linregress(lagged_clean, delta_clean)
        
        if slope >= 0:
            return float('inf')
        
        half_life = -np.log(2) / slope
        return half_life
    
    @staticmethod
    def compute_volatility_metrics(
        prices: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """Compute various volatility metrics"""
        
        returns = prices.pct_change()
        
        # Rolling volatility (annualized assuming 252 trading days)
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Parkinson volatility (if OHLC available, simplified here)
        log_returns = np.log(prices / prices.shift(1))
        parkinson_vol = log_returns.rolling(window=window).std() * np.sqrt(252)
        
        # EWMA volatility
        ewma_vol = returns.ewm(span=window).std() * np.sqrt(252)
        
        return pd.DataFrame({
            'returns': returns,
            'rolling_vol': rolling_vol,
            'parkinson_vol': parkinson_vol,
            'ewma_vol': ewma_vol
        })
    
    @staticmethod
    def detect_regime(
        prices: pd.Series,
        short_window: int = 10,
        long_window: int = 50
    ) -> pd.DataFrame:
        """Simple regime detection using moving average crossover"""
        
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        
        regime = pd.Series(index=prices.index, dtype='object')
        regime[short_ma > long_ma] = 'bullish'
        regime[short_ma < long_ma] = 'bearish'
        regime[short_ma == long_ma] = 'neutral'
        
        return pd.DataFrame({
            'price': prices,
            'short_ma': short_ma,
            'long_ma': long_ma,
            'regime': regime
        })
    
    @staticmethod
    def resample_ticks_to_ohlc(
        tick_data: pd.DataFrame,
        timeframe: str = '1m'
    ) -> pd.DataFrame:
        """Resample tick data to OHLC bars"""
        
        if 'timestamp' not in tick_data.columns:
            return tick_data
        
        df = tick_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Determine price column
        price_col = 'mid' if 'mid' in df.columns else 'close' if 'close' in df.columns else df.columns[0]
        
        # Resample
        ohlc = df[price_col].resample(timeframe).ohlc()
        
        if 'volume' in df.columns:
            ohlc['volume'] = df['volume'].resample(timeframe).sum()
        
        return ohlc.dropna().reset_index()

