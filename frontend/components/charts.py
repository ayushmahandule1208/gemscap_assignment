"""
Chart Components for Quant Analytics Dashboard
Professional-grade financial charts using Plotly
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class ChartBuilder:
    """Build professional quant-style charts"""
    
    # Color scheme - Terminal/Bloomberg inspired
    COLORS = {
        'bg': '#0a0e17',
        'paper': '#0f1419',
        'grid': '#1e2530',
        'text': '#e6edf3',
        'text_muted': '#7d8590',
        'up': '#3fb950',
        'down': '#f85149',
        'accent': '#58a6ff',
        'accent2': '#a371f7',
        'accent3': '#f0883e',
        'warning': '#d29922',
        'spread': '#39d353',
        'band': 'rgba(88, 166, 255, 0.1)',
        'band_line': 'rgba(88, 166, 255, 0.4)'
    }
    
    @staticmethod
    def get_layout_template() -> dict:
        """Get consistent layout template for all charts"""
        return {
            'paper_bgcolor': ChartBuilder.COLORS['paper'],
            'plot_bgcolor': ChartBuilder.COLORS['bg'],
            'font': {
                'family': 'JetBrains Mono, SF Mono, Consolas, monospace',
                'color': ChartBuilder.COLORS['text'],
                'size': 11
            },
            'margin': {'l': 60, 'r': 40, 't': 40, 'b': 40},
            'xaxis': {
                'gridcolor': ChartBuilder.COLORS['grid'],
                'zerolinecolor': ChartBuilder.COLORS['grid'],
                'tickfont': {'size': 10},
                'showgrid': True,
                'gridwidth': 1
            },
            'yaxis': {
                'gridcolor': ChartBuilder.COLORS['grid'],
                'zerolinecolor': ChartBuilder.COLORS['grid'],
                'tickfont': {'size': 10},
                'showgrid': True,
                'gridwidth': 1,
                'side': 'right'
            },
            'legend': {
                'bgcolor': 'rgba(15, 20, 25, 0.8)',
                'bordercolor': ChartBuilder.COLORS['grid'],
                'borderwidth': 1,
                'font': {'size': 10}
            },
            'hovermode': 'x unified',
            'hoverlabel': {
                'bgcolor': ChartBuilder.COLORS['paper'],
                'bordercolor': ChartBuilder.COLORS['accent'],
                'font': {'family': 'JetBrains Mono, monospace', 'size': 11}
            }
        }
    
    @staticmethod
    def create_candlestick_chart(
        df: pd.DataFrame,
        title: str = "Price",
        show_volume: bool = True,
        height: int = 500
    ) -> go.Figure:
        """Create a professional candlestick chart with optional volume"""
        
        if show_volume and 'volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25]
            )
        else:
            fig = go.Figure()
        
        # Determine column names
        time_col = 'timestamp' if 'timestamp' in df.columns else df.index
        open_col = df['open'] if 'open' in df.columns else df['close']
        high_col = df['high'] if 'high' in df.columns else df['close']
        low_col = df['low'] if 'low' in df.columns else df['close']
        close_col = df['close']
        
        # Candlestick
        candlestick = go.Candlestick(
            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
            open=open_col,
            high=high_col,
            low=low_col,
            close=close_col,
            increasing={'line': {'color': ChartBuilder.COLORS['up']}, 'fillcolor': ChartBuilder.COLORS['up']},
            decreasing={'line': {'color': ChartBuilder.COLORS['down']}, 'fillcolor': ChartBuilder.COLORS['down']},
            name='Price',
            showlegend=False
        )
        
        if show_volume and 'volume' in df.columns:
            fig.add_trace(candlestick, row=1, col=1)
            
            # Color volume bars based on price direction
            colors = [ChartBuilder.COLORS['up'] if c >= o else ChartBuilder.COLORS['down'] 
                     for o, c in zip(open_col, close_col)]
            
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    y=df['volume'],
                    marker_color=colors,
                    opacity=0.5,
                    name='Volume',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Price", row=1, col=1, **ChartBuilder.get_layout_template()['yaxis'])
            fig.update_yaxes(title_text="Vol", row=2, col=1, **ChartBuilder.get_layout_template()['yaxis'])
        else:
            fig.add_trace(candlestick)
        
        # Apply layout
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': title, 'x': 0.02, 'font': {'size': 14}}
        layout['xaxis_rangeslider_visible'] = False
        
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def create_spread_chart(
        spread_data: pd.DataFrame,
        z_threshold: float = 2.0,
        height: int = 350
    ) -> go.Figure:
        """Create spread and z-score chart"""
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.5, 0.5],
            subplot_titles=['Spread', 'Z-Score']
        )
        
        x = spread_data.index if 'timestamp' not in spread_data.columns else spread_data['timestamp']
        
        # Spread line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=spread_data['spread'] if 'spread' in spread_data.columns else spread_data['price'],
                mode='lines',
                line={'color': ChartBuilder.COLORS['accent'], 'width': 1.5},
                name='Spread',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Bollinger bands
        if 'upper_band' in spread_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=spread_data['upper_band'],
                    mode='lines',
                    line={'color': ChartBuilder.COLORS['band_line'], 'width': 1, 'dash': 'dot'},
                    name='+2σ',
                    showlegend=True
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=spread_data['lower_band'],
                    mode='lines',
                    line={'color': ChartBuilder.COLORS['band_line'], 'width': 1, 'dash': 'dot'},
                    name='-2σ',
                    fill='tonexty',
                    fillcolor=ChartBuilder.COLORS['band'],
                    showlegend=True
                ),
                row=1, col=1
            )
        
        if 'rolling_mean' in spread_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=spread_data['rolling_mean'],
                    mode='lines',
                    line={'color': ChartBuilder.COLORS['warning'], 'width': 1},
                    name='Mean',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Z-score
        if 'z_score' in spread_data.columns:
            z_score = spread_data['z_score']
            
            # Color z-score based on threshold
            colors = [ChartBuilder.COLORS['down'] if abs(z) > z_threshold else 
                     ChartBuilder.COLORS['accent'] for z in z_score]
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=z_score,
                    mode='lines',
                    line={'color': ChartBuilder.COLORS['accent2'], 'width': 1.5},
                    name='Z-Score',
                    showlegend=True
                ),
                row=2, col=1
            )
            
            # Threshold lines
            fig.add_hline(y=z_threshold, line_dash="dash", line_color=ChartBuilder.COLORS['down'], 
                         row=2, col=1, opacity=0.7)
            fig.add_hline(y=-z_threshold, line_dash="dash", line_color=ChartBuilder.COLORS['down'], 
                         row=2, col=1, opacity=0.7)
            fig.add_hline(y=0, line_dash="solid", line_color=ChartBuilder.COLORS['grid'], 
                         row=2, col=1, opacity=0.5)
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['showlegend'] = True
        layout['legend'] = {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1}
        
        fig.update_layout(**layout)
        fig.update_annotations(font_size=11, font_color=ChartBuilder.COLORS['text_muted'])
        
        return fig
    
    @staticmethod
    def create_correlation_chart(
        correlation: pd.Series,
        height: int = 250
    ) -> go.Figure:
        """Create rolling correlation chart"""
        
        fig = go.Figure()
        
        x = correlation.index
        
        # Correlation line with gradient fill
        fig.add_trace(
            go.Scatter(
                x=x,
                y=correlation,
                mode='lines',
                line={'color': ChartBuilder.COLORS['accent3'], 'width': 2},
                fill='tozeroy',
                fillcolor='rgba(240, 136, 62, 0.2)',
                name='Correlation'
            )
        )
        
        # Reference lines
        fig.add_hline(y=0, line_dash="solid", line_color=ChartBuilder.COLORS['grid'], opacity=0.5)
        fig.add_hline(y=0.5, line_dash="dot", line_color=ChartBuilder.COLORS['text_muted'], opacity=0.3)
        fig.add_hline(y=-0.5, line_dash="dot", line_color=ChartBuilder.COLORS['text_muted'], opacity=0.3)
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': 'Rolling Correlation', 'x': 0.02, 'font': {'size': 12}}
        layout['yaxis']['range'] = [-1.1, 1.1]
        
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def create_tick_scatter(
        tick_data: pd.DataFrame,
        height: int = 300
    ) -> go.Figure:
        """Create tick-level price scatter plot"""
        
        fig = go.Figure()
        
        x = tick_data['timestamp'] if 'timestamp' in tick_data.columns else tick_data.index
        
        # Bid/Ask spread visualization
        if 'bid' in tick_data.columns and 'ask' in tick_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=tick_data['ask'],
                    mode='markers',
                    marker={'color': ChartBuilder.COLORS['down'], 'size': 3, 'opacity': 0.6},
                    name='Ask'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=tick_data['bid'],
                    mode='markers',
                    marker={'color': ChartBuilder.COLORS['up'], 'size': 3, 'opacity': 0.6},
                    name='Bid'
                )
            )
        else:
            price_col = 'mid' if 'mid' in tick_data.columns else 'close'
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=tick_data[price_col],
                    mode='markers',
                    marker={'color': ChartBuilder.COLORS['accent'], 'size': 3},
                    name='Price'
                )
            )
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': 'Tick Data', 'x': 0.02, 'font': {'size': 12}}
        
        fig.update_layout(**layout)
        
        return fig
    
    @staticmethod
    def create_latency_gauge(
        latency_ms: float,
        title: str = "Latency",
        max_val: float = 200
    ) -> go.Figure:
        """Create a gauge chart for latency metrics"""
        
        # Determine color based on latency
        if latency_ms < 50:
            color = ChartBuilder.COLORS['up']
        elif latency_ms < 100:
            color = ChartBuilder.COLORS['warning']
        else:
            color = ChartBuilder.COLORS['down']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latency_ms,
            number={'suffix': ' ms', 'font': {'size': 24, 'color': ChartBuilder.COLORS['text']}},
            title={'text': title, 'font': {'size': 12, 'color': ChartBuilder.COLORS['text_muted']}},
            gauge={
                'axis': {'range': [0, max_val], 'tickcolor': ChartBuilder.COLORS['text_muted']},
                'bar': {'color': color},
                'bgcolor': ChartBuilder.COLORS['bg'],
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(63, 185, 80, 0.2)'},
                    {'range': [50, 100], 'color': 'rgba(210, 153, 34, 0.2)'},
                    {'range': [100, max_val], 'color': 'rgba(248, 81, 73, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': ChartBuilder.COLORS['text'], 'width': 2},
                    'thickness': 0.75,
                    'value': latency_ms
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor=ChartBuilder.COLORS['paper'],
            font={'color': ChartBuilder.COLORS['text'], 'family': 'JetBrains Mono, monospace'},
            height=180,
            margin={'l': 20, 'r': 20, 't': 40, 'b': 20}
        )
        
        return fig
    
    @staticmethod
    def create_mini_sparkline(
        values: List[float],
        color: str = None,
        height: int = 60
    ) -> go.Figure:
        if color is None:
            color = ChartBuilder.COLORS['accent']
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=values,
                mode='lines',
                line={'color': color, 'width': 1.5},
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(bytes.fromhex(color[1:])) + [0.1])}' if color.startswith('#') else color,
                showlegend=False
            )
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=height,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    
    @staticmethod
    def create_spread_distribution(
        spread: np.ndarray,
        height: int = 300
    ) -> go.Figure:
        fig = go.Figure()
        
        mean_val = np.mean(spread)
        std_val = np.std(spread)
        
        fig.add_trace(
            go.Histogram(
                x=spread,
                nbinsx=50,
                marker_color=ChartBuilder.COLORS['accent'],
                opacity=0.7,
                name='Spread Distribution'
            )
        )
        
        x_range = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 100)
        normal_curve = (1/(std_val * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range - mean_val)/std_val)**2)
        normal_curve = normal_curve * len(spread) * (spread.max() - spread.min()) / 50
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_curve,
                mode='lines',
                line={'color': ChartBuilder.COLORS['warning'], 'width': 2, 'dash': 'dash'},
                name='Normal Fit'
            )
        )
        
        fig.add_vline(x=mean_val, line_dash="solid", line_color=ChartBuilder.COLORS['up'], 
                      annotation_text=f"μ={mean_val:.2f}", annotation_position="top")
        fig.add_vline(x=mean_val + 2*std_val, line_dash="dot", line_color=ChartBuilder.COLORS['down'], opacity=0.7)
        fig.add_vline(x=mean_val - 2*std_val, line_dash="dot", line_color=ChartBuilder.COLORS['down'], opacity=0.7)
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': 'Spread Distribution', 'x': 0.02, 'font': {'size': 12}}
        layout['showlegend'] = True
        layout['legend'] = {'orientation': 'h', 'y': 1.1}
        
        fig.update_layout(**layout)
        return fig
    
    @staticmethod
    def create_rolling_hedge_ratio(
        hedge_ratios: np.ndarray,
        timestamps: List = None,
        height: int = 280
    ) -> go.Figure:
        fig = go.Figure()
        
        x = timestamps if timestamps is not None else list(range(len(hedge_ratios)))
        mean_hr = np.mean(hedge_ratios)
        std_hr = np.std(hedge_ratios)
        
        upper = np.full(len(hedge_ratios), mean_hr + 2*std_hr)
        lower = np.full(len(hedge_ratios), mean_hr - 2*std_hr)
        
        fig.add_trace(
            go.Scatter(x=x, y=upper, mode='lines', line={'width': 0}, showlegend=False, hoverinfo='skip')
        )
        fig.add_trace(
            go.Scatter(x=x, y=lower, mode='lines', line={'width': 0}, fill='tonexty',
                      fillcolor='rgba(88, 166, 255, 0.1)', showlegend=False, hoverinfo='skip')
        )
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=hedge_ratios,
                mode='lines',
                line={'color': ChartBuilder.COLORS['accent'], 'width': 2},
                name='Hedge Ratio'
            )
        )
        
        fig.add_hline(y=mean_hr, line_dash="dash", line_color=ChartBuilder.COLORS['warning'],
                     annotation_text=f"Mean: {mean_hr:.4f}")
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': 'Rolling Hedge Ratio Stability', 'x': 0.02, 'font': {'size': 12}}
        
        fig.update_layout(**layout)
        return fig
    
    @staticmethod
    def create_correlation_heatmap(
        correlation_matrix: pd.DataFrame,
        height: int = 400
    ) -> go.Figure:
        symbols = correlation_matrix.columns.tolist()
        values = correlation_matrix.values
        
        text_values = [[f'{v:.2f}' for v in row] for row in values]
        
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=symbols,
            y=symbols,
            colorscale=[
                [0, '#f85149'],
                [0.5, ChartBuilder.COLORS['bg']],
                [1, '#3fb950']
            ],
            zmin=-1,
            zmax=1,
            text=text_values,
            texttemplate='%{text}',
            textfont={'size': 11, 'color': ChartBuilder.COLORS['text']},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar={
                'title': 'Corr',
                'titleside': 'right',
                'tickfont': {'color': ChartBuilder.COLORS['text']},
                'titlefont': {'color': ChartBuilder.COLORS['text']}
            }
        ))
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': 'Correlation Matrix', 'x': 0.02, 'font': {'size': 12}}
        layout['xaxis'] = {'side': 'bottom', 'tickangle': 45}
        layout['yaxis'] = {'side': 'left', 'autorange': 'reversed'}
        
        fig.update_layout(**layout)
        return fig
    
    @staticmethod
    def create_cumulative_pnl(
        returns: np.ndarray,
        timestamps: List = None,
        benchmark_returns: np.ndarray = None,
        height: int = 320
    ) -> go.Figure:
        fig = go.Figure()
        
        cumulative = np.cumprod(1 + returns) - 1
        x = timestamps if timestamps is not None else list(range(len(returns)))
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cumulative * 100,
                mode='lines',
                line={'color': ChartBuilder.COLORS['up'], 'width': 2},
                fill='tozeroy',
                fillcolor='rgba(63, 185, 80, 0.15)',
                name='Strategy'
            )
        )
        
        if benchmark_returns is not None:
            bench_cumulative = np.cumprod(1 + benchmark_returns) - 1
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=bench_cumulative * 100,
                    mode='lines',
                    line={'color': ChartBuilder.COLORS['text_muted'], 'width': 1.5, 'dash': 'dot'},
                    name='Benchmark'
                )
            )
        
        fig.add_hline(y=0, line_dash="solid", line_color=ChartBuilder.COLORS['grid'])
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': 'Cumulative P&L', 'x': 0.02, 'font': {'size': 12}}
        layout['yaxis']['ticksuffix'] = '%'
        layout['showlegend'] = True
        layout['legend'] = {'orientation': 'h', 'y': 1.1}
        
        fig.update_layout(**layout)
        return fig
    
    @staticmethod
    def create_drawdown_chart(
        returns: np.ndarray,
        timestamps: List = None,
        height: int = 250
    ) -> go.Figure:
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        
        x = timestamps if timestamps is not None else list(range(len(returns)))
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=drawdown,
                mode='lines',
                line={'color': ChartBuilder.COLORS['down'], 'width': 1.5},
                fill='tozeroy',
                fillcolor='rgba(248, 81, 73, 0.3)',
                name='Drawdown'
            )
        )
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        fig.add_annotation(
            x=x[max_dd_idx],
            y=max_dd,
            text=f'Max DD: {max_dd:.1f}%',
            showarrow=True,
            arrowhead=2,
            arrowcolor=ChartBuilder.COLORS['down'],
            font={'color': ChartBuilder.COLORS['down'], 'size': 10}
        )
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': 'Drawdown', 'x': 0.02, 'font': {'size': 12}}
        layout['yaxis']['ticksuffix'] = '%'
        
        fig.update_layout(**layout)
        return fig
    
    @staticmethod
    def create_rolling_volatility(
        returns: np.ndarray,
        window: int = 20,
        timestamps: List = None,
        annualization_factor: float = 252,
        height: int = 280
    ) -> go.Figure:
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(annualization_factor) * 100
        
        x = timestamps if timestamps is not None else list(range(len(returns)))
        
        fig = go.Figure()
        
        mean_vol = rolling_vol.mean()
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rolling_vol,
                mode='lines',
                line={'color': ChartBuilder.COLORS['accent3'], 'width': 2},
                fill='tozeroy',
                fillcolor='rgba(240, 136, 62, 0.2)',
                name='Rolling Volatility'
            )
        )
        
        fig.add_hline(y=mean_vol, line_dash="dash", line_color=ChartBuilder.COLORS['warning'],
                     annotation_text=f"Avg: {mean_vol:.1f}%")
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': f'Rolling {window}-Period Volatility (Annualized)', 'x': 0.02, 'font': {'size': 12}}
        layout['yaxis']['ticksuffix'] = '%'
        
        fig.update_layout(**layout)
        return fig
    
    @staticmethod
    def create_regime_chart(
        spread: np.ndarray,
        regimes: np.ndarray,
        timestamps: List = None,
        height: int = 300
    ) -> go.Figure:
        x = timestamps if timestamps is not None else list(range(len(spread)))
        
        fig = go.Figure()
        
        regime_colors = {
            0: ('rgba(63, 185, 80, 0.2)', 'Mean Reverting'),
            1: ('rgba(210, 153, 34, 0.2)', 'Transition'),
            2: ('rgba(248, 81, 73, 0.2)', 'Trending')
        }
        
        for regime_id, (color, label) in regime_colors.items():
            mask = regimes == regime_id
            if mask.any():
                regime_spread = np.where(mask, spread, np.nan)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=regime_spread,
                        mode='lines',
                        line={'width': 0},
                        fill='tozeroy',
                        fillcolor=color,
                        name=label,
                        showlegend=True
                    )
                )
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=spread,
                mode='lines',
                line={'color': ChartBuilder.COLORS['text'], 'width': 1.5},
                name='Spread'
            )
        )
        
        layout = ChartBuilder.get_layout_template()
        layout['height'] = height
        layout['title'] = {'text': 'Spread with Regime Overlay', 'x': 0.02, 'font': {'size': 12}}
        layout['showlegend'] = True
        layout['legend'] = {'orientation': 'h', 'y': 1.1}
        
        fig.update_layout(**layout)
        return fig

