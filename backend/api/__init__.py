"""
API Routers
"""
from .upload import router as upload_router
from .data import router as data_router
from .analytics import router as analytics_router
from .live import router as live_router

__all__ = ["upload_router", "data_router", "analytics_router", "live_router"]

