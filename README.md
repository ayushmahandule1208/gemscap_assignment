# Quant Analytics Platform

A lightweight, end-to-end quantitative analytics system designed to demonstrate real-world data ingestion, processing, analytics, and visualization, similar to how professional quant research tools are structured.

The platform supports live market data and historical datasets, performs both real-time monitoring and batch statistical analysis, and exposes results through a clean, interactive dashboard.

---

## Submissions

Two submissions were made to ensure compatibility with different evaluation workflows:

1. GitHub Repository Link — complete source code, documentation, and architecture
2. ZIP Archive Link — packaged version of the same project for offline review

Both submissions contain identical functionality and documentation.

---

## Setup and Execution

### Prerequisites

* Python 3.9+
* pip
* Stable internet connection (required for live Binance WebSocket feed)

---

### Installation

```bash
git clone https://github.com/ayushmahandule1208/gemscap_assignment.git
cd gemscap_assignment
pip install -r requirements.txt
```

---

### Run (Single Command)

```bash
python app.py
```

This single command starts:

* FastAPI backend server
* In-memory buffers and SQLite persistence
* Streamlit-based analytics dashboard

Local service URLs are printed directly in the terminal.

---

## Dependencies

| Library              | Purpose                        |
| -------------------- | ------------------------------ |
| FastAPI              | Backend API layer              |
| Streamlit            | Interactive frontend dashboard |
| NumPy / Pandas       | Time-series manipulation       |
| SciPy / Statsmodels  | Statistical analysis           |
| scikit-learn         | Robust regression models       |
| websockets / asyncio | Live data ingestion            |
| SQLite3              | Lightweight persistence        |
| matplotlib / plotly  | Visualization                  |

All dependencies are explicitly listed in requirements.txt.

---

## Methodology

### Design Principles

* Single normalization point
  All incoming data is converted into standard internal models (TickEvent, OHLCBar).

* Source-agnostic ingestion
  The analytics engine is unaware of whether data is live or file-based.

* Separation of concerns
  Clear boundaries between ingestion, analytics, alerts, storage, and presentation.

* Pure analytics functions
  All statistical computations are stateless and side-effect free.

* Dual storage model
  In-memory buffers for low-latency access, SQLite for durability and replay.

---

### Data Ingestion

The system supports two independent ingestion paths:

#### 1. Live Market Data

* Binance Futures WebSocket feed
* Trade messages parsed into tick-level events
* Continuous ingestion into the engine

#### 2. Historical Data

* CSV or NDJSON uploads
* Tick-level or OHLC formats
* Normalized before ingestion

Both paths converge into a single unified processing pipeline.

---

### Processing Pipeline

* Tick data is:

  * Stored in a rolling in-memory buffer
  * Resampled into OHLC bars (1s, 1m, 5m)
  * Persisted to SQLite

* Uploaded OHLC data bypasses resampling and is stored directly

* Completed bars trigger:

  * Batch analytics
  * Alert rule evaluation

---

## Analytics Overview

The analytics layer is split into low-latency live analytics and computational batch analytics.

---

### Live Analytics (Low Latency)

Updated on every tick or short interval:

* Rolling mean and standard deviation
* Pair spread calculation
* Z-score of the spread
* Rolling correlation

These metrics are designed for real-time monitoring, diagnostics, and alerts.

---

### Batch Analytics

Triggered on bar close or user request:

#### Hedge Ratio Estimation

* Ordinary Least Squares (OLS)
* Kalman Filter (dynamic hedge ratio)
* Robust regression (Huber, Theil–Sen)

#### Statistical Testing

* Augmented Dickey–Fuller (ADF)
* P-value computation and stationarity flag

#### Mean Reversion Metrics

* Half-life estimation

#### Backtesting

* Simple mean-reversion strategy

  * Entry: |z| > 2
  * Exit: z < 0

#### Cross-Asset Analysis

* Correlation matrix
* Liquidity-based filtering

All analytics functions operate purely on data passed from the engine, without direct database access or shared state.

---

## Frontend

The frontend is implemented using Streamlit, prioritizing analytical clarity over UI complexity.

Features include:

* Symbol and timeframe selection
* Live price and OHLC visualizations
* Spread and Z-score charts
* Advanced analytics outputs
* Alert configuration and history
* Data export utilities

The frontend communicates exclusively via backend APIs, maintaining a clean separation from analytics logic.

---

## Architecture Diagram

A complete system architecture diagram is included and documented using eraser.io:

[https://app.eraser.io/workspace/szDupfpFPJPeZbsyvghZ?origin=share&elements=15rMVRjD0cjeAcv471mOuQ](https://app.eraser.io/workspace/szDupfpFPJPeZbsyvghZ?origin=share&elements=15rMVRjD0cjeAcv471mOuQ)

---

## AI Usage Transparency

ChatGPT was used selectively to:

* Validate architectural decisions
* Clarify system design trade-offs
* Verify statistical formulas
* Explain quantitative concepts
* Improve documentation clarity
* Assist with Streamlit structuring

All analytics logic, system design, and integration decisions were independently reasoned and implemented.

---

## Trade-offs

This project intentionally prioritizes:

* Correctness and analytical clarity
* End-to-end completeness
* Realistic quant-system design

Over:

* Production-scale infrastructure
* Authentication and deployment
* Extensive UI polish

These trade-offs align with the one-day project constraint.

---

## Future Scope

* Additional Data Feeds
  Support CME futures, equities, FX, and REST-based historical APIs via pluggable adapters.

* Scalable Storage
  Migrate from SQLite to a time-series database such as TimescaleDB or ClickHouse.

* Cross-Asset Analytics
  Liquidity filters, rolling cross-correlation heatmaps, and regime detection.

* Distributed Architecture
  Message queues and worker pools for horizontally scalable ingestion and analytics.
