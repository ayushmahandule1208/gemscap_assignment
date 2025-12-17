"""
Batch Analytics
Analytics computed on bar close or user trigger.

Update: On bar close (1s/1m/5m) or user request
Use: Model estimation, diagnostics, research

These analytics run on resampled data for statistical stability.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats

from .models import (
    BatchHedgeRatio,
    BatchCorrelation,
    BatchADF,
    BatchHalfLife
)


def hedge_ratio(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    method: str = "ols"
) -> BatchHedgeRatio:
    """
    Compute hedge ratio using OLS regression.
    
    price_A = α + β * price_B + ε
    
    Why batch: OLS is sensitive to microstructure noise.
               Stable estimation requires resampled data.
    
    Args:
        prices_a: Price series for asset A (dependent)
        prices_b: Price series for asset B (independent)
        method: 'ols' or 'tls' (total least squares)
    
    Returns:
        BatchHedgeRatio with beta, alpha, r_squared
    """
    # Remove NaN
    valid = ~(np.isnan(prices_a) | np.isnan(prices_b))
    p_a = prices_a[valid]
    p_b = prices_b[valid]
    
    if len(p_a) < 10:
        return BatchHedgeRatio(beta=1.0, alpha=0.0, r_squared=0.0, std_error=0.0)
    
    if method == "ols":
        slope, intercept, r_value, _, std_err = stats.linregress(p_b, p_a)
        return BatchHedgeRatio(
            beta=float(slope),
            alpha=float(intercept),
            r_squared=float(r_value ** 2),
            std_error=float(std_err)
        )
    
    elif method == "tls":
        # Total Least Squares (orthogonal regression)
        mean_a, mean_b = np.mean(p_a), np.mean(p_b)
        cov_matrix = np.cov(p_a - mean_a, p_b - mean_b)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = np.argmin(eigenvalues)
        beta = float(-eigenvectors[0, idx] / eigenvectors[1, idx])
        alpha = float(mean_a - beta * mean_b)
        
        # Approximate R²
        spread = p_a - beta * p_b
        ss_res = np.sum((spread - np.mean(spread)) ** 2)
        ss_tot = np.sum((p_a - np.mean(p_a)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return BatchHedgeRatio(
            beta=beta,
            alpha=alpha,
            r_squared=float(r_squared),
            std_error=0.0
        )
    
    return BatchHedgeRatio(beta=1.0, alpha=0.0, r_squared=0.0, std_error=0.0)


def spread_stats(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    hedge_ratio_value: float = None,
    window: int = 20
) -> Dict[str, Any]:
    """
    Compute batch spread statistics.
    
    Returns μ and σ for live z-score computation.
    
    Args:
        prices_a: Asset A prices
        prices_b: Asset B prices
        hedge_ratio_value: Pre-computed hedge ratio (or computed here)
        window: Rolling window
    
    Returns:
        Dict with spread array, mean, std, z_scores
    """
    if hedge_ratio_value is None:
        hr = hedge_ratio(prices_a, prices_b)
        hedge_ratio_value = hr.beta
    
    spread = prices_a - hedge_ratio_value * prices_b
    
    series = pd.Series(spread)
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    z_scores = (series - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0)
    
    return {
        'spread': spread,
        'mean': float(np.nanmean(spread)),
        'std': float(np.nanstd(spread)),
        'rolling_mean': rolling_mean.values,
        'rolling_std': rolling_std.values,
        'z_scores': z_scores.values,
        'hedge_ratio': hedge_ratio_value
    }


def rolling_correlation(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    window: int = 20
) -> BatchCorrelation:
    """
    Compute rolling correlation between two price series.
    
    Purpose: Detect correlation breakdown, invalidate pairs dynamically
    
    Args:
        prices_a: First price series
        prices_b: Second price series
        window: Rolling window
    
    Returns:
        BatchCorrelation with current, mean, std, stability flag
    """
    returns_a = pd.Series(prices_a).pct_change().dropna()
    returns_b = pd.Series(prices_b).pct_change().dropna()
    
    # Align
    min_len = min(len(returns_a), len(returns_b))
    returns_a = returns_a.iloc[:min_len]
    returns_b = returns_b.iloc[:min_len]
    
    if len(returns_a) < window:
        return BatchCorrelation(current=0, mean=0, std=0, is_stable=False)
    
    rolling_corr = returns_a.rolling(window=window).corr(returns_b)
    rolling_corr = rolling_corr.dropna()
    
    if len(rolling_corr) == 0:
        return BatchCorrelation(current=0, mean=0, std=0, is_stable=False)
    
    current = float(rolling_corr.iloc[-1])
    mean = float(rolling_corr.mean())
    std = float(rolling_corr.std())
    
    return BatchCorrelation(
        current=current,
        mean=mean,
        std=std,
        is_stable=std < 0.2
    )


def adf_test(series: np.ndarray) -> BatchADF:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    Purpose: Validate mean-reversion assumption
    
    Why batch only:
        - Requires sufficient data
        - Computationally expensive
        - Not meaningful tick-by-tick
    
    Args:
        series: Spread or price series
    
    Returns:
        BatchADF with statistic, p-value, stationarity flag
    """
    clean = pd.Series(series).dropna()
    
    critical_values = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    
    if len(clean) < 20:
        return BatchADF(
            statistic=0.0,
            p_value=1.0,
            is_stationary=False,
            critical_values=critical_values
        )
    
    try:
        # Simplified ADF implementation
        diff = clean.diff().dropna()
        lagged = clean.shift(1).dropna()
        
        min_len = min(len(diff), len(lagged))
        diff = diff.iloc[:min_len]
        lagged = lagged.iloc[:min_len]
        
        if lagged.std() > 0 and len(diff) > 2:
            rho = np.corrcoef(diff.values, lagged.values[:len(diff)])[0, 1]
            t_stat = rho * np.sqrt(len(diff) - 2) / np.sqrt(1 - rho**2) if abs(rho) < 1 else 0
        else:
            t_stat = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return BatchADF(
            statistic=round(t_stat, 4),
            p_value=round(p_value, 4),
            is_stationary=p_value < 0.05,
            critical_values=critical_values
        )
    except Exception:
        return BatchADF(
            statistic=0.0,
            p_value=1.0,
            is_stationary=False,
            critical_values=critical_values
        )


def half_life(series: np.ndarray) -> BatchHalfLife:
    """
    Calculate half-life of mean reversion.
    
    Lower half-life = faster mean reversion = better for trading
    
    Args:
        series: Spread series
    
    Returns:
        BatchHalfLife with half_life in periods
    """
    clean = pd.Series(series).dropna()
    
    if len(clean) < 10:
        return BatchHalfLife(
            half_life=float('inf'),
            lambda_coef=0.0,
            is_mean_reverting=False,
            r_squared=0.0
        )
    
    lagged = clean.shift(1)
    delta = clean - lagged
    
    valid = ~(lagged.isna() | delta.isna())
    lagged_clean = lagged[valid].values
    delta_clean = delta[valid].values
    
    if len(lagged_clean) < 2:
        return BatchHalfLife(
            half_life=float('inf'),
            lambda_coef=0.0,
            is_mean_reverting=False,
            r_squared=0.0
        )
    
    slope, _, r_value, _, _ = stats.linregress(lagged_clean, delta_clean)
    
    if slope >= 0:
        return BatchHalfLife(
            half_life=float('inf'),
            lambda_coef=float(slope),
            is_mean_reverting=False,
            r_squared=float(r_value ** 2)
        )
    
    half_life_value = -np.log(2) / slope
    
    return BatchHalfLife(
        half_life=float(half_life_value),
        lambda_coef=float(slope),
        is_mean_reverting=True,
        r_squared=float(r_value ** 2)
    )


def volatility(
    prices: np.ndarray,
    window: int = 20
) -> Dict[str, Optional[float]]:
    """
    Compute volatility metrics.
    
    Args:
        prices: Price array
        window: Rolling window
    
    Returns:
        Dict with volatility measures
    """
    series = pd.Series(prices)
    returns = series.pct_change().dropna()
    
    if len(returns) < 2:
        return {
            'rolling_vol': None,
            'ewma_vol': None,
            'sharpe': None,
            'max_drawdown': None
        }
    
    ann_factor = np.sqrt(252)
    
    rolling_vol = returns.rolling(window=window).std() * ann_factor
    ewma_vol = returns.ewm(span=window).std() * ann_factor
    
    # Sharpe
    sharpe = None
    if returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * ann_factor)
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min()) if not drawdown.isna().all() else None
    
    return {
        'rolling_vol': float(rolling_vol.iloc[-1]) if not pd.isna(rolling_vol.iloc[-1]) else None,
        'ewma_vol': float(ewma_vol.iloc[-1]) if not pd.isna(ewma_vol.iloc[-1]) else None,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }


# =============================================================================
# ADVANCED ANALYTICS
# =============================================================================

def kalman_hedge_ratio(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    delta: float = 1e-4,
    ve: float = 1e-3
) -> Dict[str, Any]:
    """
    Dynamic hedge ratio estimation using Kalman Filter.
    
    Unlike static OLS, the hedge ratio evolves over time.
    This captures regime changes and structural breaks.
    
    State: β_t (hedge ratio at time t)
    Observation: price_A_t = β_t * price_B_t + noise
    
    Args:
        prices_a: Dependent asset prices
        prices_b: Independent asset prices
        delta: State transition variance (how fast β changes)
        ve: Observation noise variance
    
    Returns:
        Dict with:
            - hedge_ratios: Array of time-varying hedge ratios
            - current: Latest hedge ratio
            - std: Standard deviation of hedge ratio over time
            - is_stable: True if std < 0.1
    """
    n = len(prices_a)
    if n < 10:
        return {
            'hedge_ratios': np.array([1.0]),
            'current': 1.0,
            'std': 0.0,
            'is_stable': False
        }
    
    # Initialize
    beta = np.zeros(n)
    P = np.zeros(n)  # Estimation error variance
    
    beta[0] = 1.0  # Initial hedge ratio guess
    P[0] = 1.0     # Initial uncertainty
    
    # Kalman Filter iteration
    for t in range(1, n):
        # Prediction step
        beta_pred = beta[t-1]
        P_pred = P[t-1] + delta
        
        # Update step
        x = prices_b[t]
        y = prices_a[t]
        
        # Kalman gain
        K = P_pred * x / (x * P_pred * x + ve)
        
        # Update estimate
        beta[t] = beta_pred + K * (y - beta_pred * x)
        P[t] = (1 - K * x) * P_pred
    
    current = float(beta[-1])
    beta_std = float(np.std(beta[10:]))  # Exclude warmup
    
    return {
        'hedge_ratios': beta,
        'current': round(current, 6),
        'mean': round(float(np.mean(beta[10:])), 6),
        'std': round(beta_std, 6),
        'is_stable': beta_std < 0.1
    }


def robust_hedge_ratio(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    method: str = "huber"
) -> Dict[str, Any]:
    """
    Robust regression for hedge ratio estimation.
    
    OLS is sensitive to outliers. Robust methods downweight outliers.
    
    Methods:
        - huber: Huber regression (M-estimator)
        - theil_sen: Theil-Sen estimator (median of slopes)
    
    Args:
        prices_a: Dependent asset prices
        prices_b: Independent asset prices
        method: 'huber' or 'theil_sen'
    
    Returns:
        Dict with beta, alpha, method used
    """
    valid = ~(np.isnan(prices_a) | np.isnan(prices_b))
    p_a = prices_a[valid]
    p_b = prices_b[valid]
    
    if len(p_a) < 10:
        return {'beta': 1.0, 'alpha': 0.0, 'method': method}
    
    if method == "huber":
        # Huber regression using iteratively reweighted least squares
        # Simplified implementation
        beta, alpha = _huber_regression(p_b, p_a)
        
    elif method == "theil_sen":
        # Theil-Sen: median of all pairwise slopes
        beta, alpha = _theil_sen_regression(p_b, p_a)
    
    else:
        # Fallback to OLS
        slope, intercept, _, _, _ = stats.linregress(p_b, p_a)
        beta, alpha = slope, intercept
    
    # Compare with OLS
    ols_slope, ols_intercept, _, _, _ = stats.linregress(p_b, p_a)
    
    return {
        'beta': round(float(beta), 6),
        'alpha': round(float(alpha), 4),
        'method': method,
        'ols_beta': round(float(ols_slope), 6),
        'ols_alpha': round(float(ols_intercept), 4),
        'beta_diff_pct': round(abs(beta - ols_slope) / abs(ols_slope) * 100, 2) if ols_slope != 0 else 0
    }


def _huber_regression(x: np.ndarray, y: np.ndarray, k: float = 1.345, max_iter: int = 50) -> tuple:
    """
    Huber M-estimator regression.
    
    Iteratively reweighted least squares with Huber weights.
    k=1.345 gives 95% efficiency for normal data.
    """
    # Initial OLS estimate
    slope, intercept, _, _, _ = stats.linregress(x, y)
    
    for _ in range(max_iter):
        # Residuals
        residuals = y - (slope * x + intercept)
        
        # MAD scale estimate
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = mad / 0.6745 if mad > 0 else 1.0
        
        # Standardized residuals
        u = residuals / scale
        
        # Huber weights
        weights = np.where(np.abs(u) <= k, 1.0, k / np.abs(u))
        
        # Weighted least squares
        w_sum = np.sum(weights)
        wx_sum = np.sum(weights * x)
        wy_sum = np.sum(weights * y)
        wxx_sum = np.sum(weights * x * x)
        wxy_sum = np.sum(weights * x * y)
        
        denom = w_sum * wxx_sum - wx_sum * wx_sum
        if abs(denom) < 1e-10:
            break
        
        new_slope = (w_sum * wxy_sum - wx_sum * wy_sum) / denom
        new_intercept = (wy_sum - new_slope * wx_sum) / w_sum
        
        # Check convergence
        if abs(new_slope - slope) < 1e-6:
            break
        
        slope, intercept = new_slope, new_intercept
    
    return slope, intercept


def _theil_sen_regression(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Theil-Sen estimator: median of all pairwise slopes.
    
    Very robust to outliers (breakdown point = 29%).
    """
    n = len(x)
    
    # For large datasets, sample pairs
    if n > 500:
        indices = np.random.choice(n, size=500, replace=False)
        x = x[indices]
        y = y[indices]
        n = 500
    
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    
    if not slopes:
        return 1.0, 0.0
    
    slope = np.median(slopes)
    intercept = np.median(y - slope * x)
    
    return slope, intercept


def mini_backtest(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    hedge_ratio_value: float = None,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    window: int = 20
) -> Dict[str, Any]:
    """
    Mini mean-reversion backtest.
    
    Strategy:
        - LONG spread when z < -entry_z (spread too low)
        - SHORT spread when z > entry_z (spread too high)
        - EXIT when z crosses exit_z
    
    Args:
        prices_a: Asset A prices
        prices_b: Asset B prices
        hedge_ratio_value: Pre-computed hedge ratio
        entry_z: Z-score threshold for entry (default: 2.0)
        exit_z: Z-score threshold for exit (default: 0.0)
        window: Rolling window for z-score
    
    Returns:
        Dict with backtest results
    """
    if hedge_ratio_value is None:
        hr = hedge_ratio(prices_a, prices_b)
        hedge_ratio_value = hr.beta
    
    n = len(prices_a)
    if n < window + 10:
        return {
            'total_return': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_trade_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'trades': []
        }
    
    # Compute spread and z-scores
    spread = prices_a - hedge_ratio_value * prices_b
    series = pd.Series(spread)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    z_scores = ((series - rolling_mean) / rolling_std).fillna(0).values
    
    # Backtest
    position = 0  # 1 = long spread, -1 = short spread, 0 = flat
    entry_price = 0.0
    trades = []
    equity = [1.0]
    
    for i in range(window, n):
        z = z_scores[i]
        current_spread = spread[i]
        
        # Entry logic
        if position == 0:
            if z < -entry_z:
                # Spread too low, expect reversion up → LONG spread
                position = 1
                entry_price = current_spread
                trades.append({'type': 'LONG', 'entry_idx': i, 'entry_z': z})
            elif z > entry_z:
                # Spread too high, expect reversion down → SHORT spread
                position = -1
                entry_price = current_spread
                trades.append({'type': 'SHORT', 'entry_idx': i, 'entry_z': z})
        
        # Exit logic
        elif position == 1:  # Long spread
            if z >= exit_z:
                # Exit long
                pnl = current_spread - entry_price
                trades[-1]['exit_idx'] = i
                trades[-1]['exit_z'] = z
                trades[-1]['pnl'] = pnl
                trades[-1]['return_pct'] = pnl / abs(entry_price) * 100 if entry_price != 0 else 0
                position = 0
        
        elif position == -1:  # Short spread
            if z <= exit_z:
                # Exit short
                pnl = entry_price - current_spread
                trades[-1]['exit_idx'] = i
                trades[-1]['exit_z'] = z
                trades[-1]['pnl'] = pnl
                trades[-1]['return_pct'] = pnl / abs(entry_price) * 100 if entry_price != 0 else 0
                position = 0
        
        # Track equity (simplified)
        if position != 0 and len(trades) > 0:
            if position == 1:
                unrealized = (current_spread - entry_price) / abs(entry_price) if entry_price != 0 else 0
            else:
                unrealized = (entry_price - current_spread) / abs(entry_price) if entry_price != 0 else 0
            equity.append(equity[-1] * (1 + unrealized * 0.01))  # Scale down
        else:
            equity.append(equity[-1])
    
    # Close any open position
    if position != 0 and trades:
        trades[-1]['exit_idx'] = n - 1
        trades[-1]['exit_z'] = z_scores[-1]
        trades[-1]['pnl'] = (spread[-1] - entry_price) * position
        trades[-1]['return_pct'] = trades[-1]['pnl'] / abs(entry_price) * 100 if entry_price != 0 else 0
    
    # Calculate metrics
    completed_trades = [t for t in trades if 'pnl' in t]
    num_trades = len(completed_trades)
    
    if num_trades == 0:
        return {
            'total_return': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_trade_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'trades': []
        }
    
    wins = sum(1 for t in completed_trades if t['pnl'] > 0)
    returns = [t['return_pct'] for t in completed_trades]
    
    # Max drawdown from equity curve
    equity_arr = np.array(equity)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    
    # Sharpe (annualized, assuming daily)
    returns_arr = np.array(returns)
    sharpe = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)) if np.std(returns_arr) > 0 else 0
    
    return {
        'total_return': round(sum(returns), 2),
        'num_trades': num_trades,
        'win_rate': round(wins / num_trades * 100, 1),
        'avg_trade_return': round(np.mean(returns), 2),
        'max_drawdown': round(max_dd * 100, 2),
        'sharpe': round(sharpe, 2),
        'entry_z': entry_z,
        'exit_z': exit_z,
        'trades': [
            {
                'type': t['type'],
                'entry_idx': t['entry_idx'],
                'exit_idx': t.get('exit_idx'),
                'return_pct': round(t.get('return_pct', 0), 2)
            }
            for t in completed_trades[-10:]  # Last 10 trades
        ]
    }


def cross_correlation_matrix(
    price_dict: Dict[str, np.ndarray],
    window: int = 20
) -> Dict[str, Any]:
    """
    Compute cross-correlation matrix for multiple assets.
    
    Args:
        price_dict: Dict of symbol → price array
        window: Rolling window for correlation
    
    Returns:
        Dict with correlation matrix and heatmap data
    """
    symbols = list(price_dict.keys())
    n = len(symbols)
    
    if n < 2:
        return {'matrix': [], 'symbols': symbols, 'avg_correlation': 0.0}
    
    # Compute returns
    returns_dict = {}
    for sym, prices in price_dict.items():
        returns_dict[sym] = pd.Series(prices).pct_change().dropna().values
    
    # Align lengths
    min_len = min(len(r) for r in returns_dict.values())
    for sym in symbols:
        returns_dict[sym] = returns_dict[sym][-min_len:]
    
    # Build correlation matrix
    corr_matrix = np.zeros((n, n))
    
    for i, sym_i in enumerate(symbols):
        for j, sym_j in enumerate(symbols):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                corr = np.corrcoef(returns_dict[sym_i], returns_dict[sym_j])[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    
    # Find strongest pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'pair': f"{symbols[i]}/{symbols[j]}",
                'correlation': round(corr_matrix[i, j], 4)
            })
    
    pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Average correlation (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    avg_corr = float(np.mean(upper_tri))
    
    return {
        'symbols': symbols,
        'matrix': [[round(corr_matrix[i, j], 4) for j in range(n)] for i in range(n)],
        'strongest_pairs': pairs[:5],
        'avg_correlation': round(avg_corr, 4),
        'data_points': min_len
    }


def liquidity_filter(
    volumes: np.ndarray,
    prices: np.ndarray,
    min_avg_volume: float = None,
    min_dollar_volume: float = None,
    max_spread_pct: float = None,
    highs: np.ndarray = None,
    lows: np.ndarray = None
) -> Dict[str, Any]:
    """
    Filter assets by liquidity metrics.
    
    Args:
        volumes: Volume array
        prices: Price array
        min_avg_volume: Minimum average volume
        min_dollar_volume: Minimum dollar volume (price * volume)
        max_spread_pct: Maximum bid-ask spread proxy (using high-low)
        highs: High prices (for spread calculation)
        lows: Low prices (for spread calculation)
    
    Returns:
        Dict with liquidity metrics and pass/fail flags
    """
    n = len(volumes)
    if n < 5:
        return {
            'avg_volume': 0,
            'avg_dollar_volume': 0,
            'avg_spread_pct': None,
            'passes_filter': False
        }
    
    avg_volume = float(np.mean(volumes))
    avg_dollar_volume = float(np.mean(volumes * prices))
    
    # Spread proxy using high-low range
    avg_spread_pct = None
    if highs is not None and lows is not None:
        mid_prices = (highs + lows) / 2
        spreads = (highs - lows) / mid_prices * 100
        avg_spread_pct = float(np.mean(spreads))
    
    # Check filters
    passes = True
    reasons = []
    
    if min_avg_volume is not None and avg_volume < min_avg_volume:
        passes = False
        reasons.append(f"Volume {avg_volume:.0f} < {min_avg_volume:.0f}")
    
    if min_dollar_volume is not None and avg_dollar_volume < min_dollar_volume:
        passes = False
        reasons.append(f"Dollar volume {avg_dollar_volume:.0f} < {min_dollar_volume:.0f}")
    
    if max_spread_pct is not None and avg_spread_pct is not None and avg_spread_pct > max_spread_pct:
        passes = False
        reasons.append(f"Spread {avg_spread_pct:.2f}% > {max_spread_pct:.2f}%")
    
    return {
        'avg_volume': round(avg_volume, 0),
        'avg_dollar_volume': round(avg_dollar_volume, 0),
        'avg_spread_pct': round(avg_spread_pct, 4) if avg_spread_pct else None,
        'passes_filter': passes,
        'fail_reasons': reasons if not passes else []
    }

