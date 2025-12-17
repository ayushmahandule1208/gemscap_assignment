from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.upload import router as upload_router
from api.data import router as data_router
from api.analytics import router as analytics_router
from api.alerts import router as alerts_router
from api.export import router as export_router
from api.live import router as live_router
from services import get_live_feed

AUTO_START_SYMBOLS = ["btcusdt", "ethusdt"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    feed = get_live_feed()
    feed.start(AUTO_START_SYMBOLS)
    yield
    if feed.is_running:
        feed.stop()

app = FastAPI(
    title="Quant Analytics API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")
app.include_router(data_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")
app.include_router(export_router, prefix="/api")
app.include_router(live_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "name": "Quant Analytics API",
        "version": "2.0.0",
        "docs": "/docs",
    }

@app.get("/health")
async def health():
    from core import get_engine
    from services import get_live_feed
    
    engine = get_engine()
    stats = engine.stats()
    feed = get_live_feed()
    
    return {
        "status": "healthy",
        "engine": {
            "ticks_ingested": stats["ticks_ingested"],
            "bars_created": stats["bars_created"],
            "bars_uploaded": stats.get("bars_uploaded", 0),
            "symbols": stats["symbols"],
            "uptime_seconds": round(stats["uptime_seconds"], 2)
        },
        "live_feed": {
            "is_running": feed.is_running,
            "symbols": feed.stats.symbols or [],
            "ticks_received": feed.stats.ticks_received
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
