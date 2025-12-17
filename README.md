# Quant Analytics Platform

A lightweight, end-to-end **quantitative analytics application** that ingests live and historical market data, performs real-time and batch analytics, and presents results through a simple interactive frontend.

This project was intentionally scoped to be completed within **one day**, prioritizing **clarity of design, correctness of analytics, and clean data flow** over feature sprawl.

---

## Setup & Execution

### Prerequisites

- Python 3.9+
- pip
- Internet connection (for live Binance feed)

### Installation

```bash
https://github.com/ayushmahandule1208/gemscap_assignment.git
pip install -r requirements.txt
```

### Run (Single Command)

```bash
python app.py
```

This starts:

* The FastAPI backend
* In-memory buffers and SQLite storage
* The Streamlit frontend dashboard

Local URLs are printed in the terminal.

---

## Dependencies

| Library              | Purpose                  |
| -------------------- | ------------------------ |
| FastAPI              | Backend API layer        |
| Streamlit            | Frontend dashboard       |
| NumPy / Pandas       | Time-series manipulation |
| SciPy / Statsmodels  | Statistical analysis     |
| scikit-learn         | Robust regression        |
| websockets / asyncio | Live data ingestion      |
| SQLite3              | Lightweight persistence  |
| matplotlib / plotly  | Visualization            |

All dependencies are listed in `requirements.txt`.

---

## Methodology

### Design Principles

* **Single normalization point**: All incoming data is converted to standard models (`TickEvent`, `OHLCBar`)
* **Source-agnostic ingestion**: The core engine is unaware of data origin (live or file-based)
* **Separation of concerns**: Clear boundaries between ingestion, analytics, alerts, and storage
* **Pure analytics functions**: Stateless, side-effect-free computations
* **Dual storage model**: In-memory buffers for speed, SQLite for durability

---

### Data Ingestion

The system supports two ingestion paths:

1. **Live data**

   * Binance Futures WebSocket
   * Trade messages parsed into tick events
   * Continuous ingestion into the engine

2. **Historical data**

   * CSV / NDJSON uploads
   * Tick-level or OHLC formats
   * Normalized before ingestion

Both paths converge into a **single processing pipeline**.

---

### Processing Pipeline

* Tick data is:

  * Stored in a rolling in-memory buffer
  * Resampled into OHLC bars (1s / 1m / 5m)
  * Persisted to SQLite

* Uploaded OHLC data bypasses resampling and is stored directly

Completed bars trigger batch analytics and alert evaluation.

---

## Analytics Explanation

The analytics layer is divided into **live** and **batch** computations.

---

### Live Analytics (Low Latency)

Updated on each tick or short interval:

* Rolling mean and standard deviation
* Spread between asset pairs
* Z-score of the spread
* Rolling correlation

These metrics are designed for **real-time monitoring and alerting**.

---

### Batch Analytics

Triggered on bar close or user request:

* **Hedge ratio estimation**

  * Ordinary Least Squares (OLS)
  * Kalman Filter (dynamic hedge)
  * Robust regression (Huber, Theil–Sen)

* **Stationarity testing**

  * Augmented Dickey–Fuller (ADF)
  * P-value and stationarity flag

* **Mean reversion metrics**

  * Half-life estimation

* **Backtesting**

  * Simple mean-reversion strategy
    (entry: |z| > 2, exit: z < 0)

* **Cross-asset analysis**

  * Correlation matrix
  * Liquidity-based filtering

All analytics functions operate purely on data passed from the engine, without database access or shared state.

---

## Frontend

The frontend is implemented using **Streamlit**, keeping the focus on analytics rather than UI complexity.

Features include:

* Symbol and timeframe selection
* Live price and OHLC charts
* Spread and z-score visualization
* Advanced analytics outputs
* Alert management and history
* Data export

The frontend communicates exclusively via backend APIs.

---

## Architecture Diagram

The system architecture is documented using **eraser.io** and included in the repository:

link: https://app.eraser.io/workspace/szDupfpFPJPeZbsyvghZ?origin=share&elements=15rMVRjD0cjeAcv471mOuQ


## AI Usage Transparency

ChatGPT was used selectively to:

* Validate architectural decisions
* Clear doubts regarding tradeoffs
* Statistical calculation formulas
* Clarify quantitative concepts
* Improve documentation clarity
* Assist with frontend structuring as I was not well versed with streamlit

All analytics logic, system design, and integration decisions were independently reasoned and implemented.

---

## Scope & Trade-offs

This project prioritizes:

* Correctness and clarity
* End-to-end completeness
* Realistic system design

Over:

* Production-scale infrastructure
* Authentication and deployment
* UI polish

These trade-offs were made intentionally to align with the **one-day project constraint**.

