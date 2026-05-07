"""
Smart Expense Tracker — FastAPI Backend
Provides AI-powered predictions, anomaly detection, and insights.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

from routers.predictions import router as predictions_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Smart Expense Tracker AI Backend starting...")
    yield
    logger.info("🛑 Backend shutting down.")


app = FastAPI(
    title="Smart Expense Tracker API",
    description="AI-powered expense prediction, anomaly detection, and insights",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow all origins by default so any deployed frontend can connect.
# Set ALLOWED_ORIGINS env var to a comma-separated list to restrict (e.g. in production).
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")

if allowed_origins_str.strip() == "*":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    allowed_origins = [o.strip() for o in allowed_origins_str.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(predictions_router)


@app.get("/")
async def root():
    return {
        "message": "Smart Expense Tracker AI Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }
