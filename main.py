"""
Smart Expense Tracker — FastAPI Backend
Provides AI-powered predictions, anomaly detection, insights, upload, and chat.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

from routers.predictions import router as predictions_router
from routers.upload import router as upload_router
from routers.chat import router as chat_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Smart Expense Tracker AI Backend starting...")
    yield
    logger.info("🛑 Backend shutting down.")


app = FastAPI(
    title="Smart Expense Tracker API",
    description="AI-powered expense prediction, anomaly detection, Excel upload, and Gemini chat",
    version="2.0.0",
    lifespan=lifespan
)

# CORS — allow all origins by default so any deployed frontend can connect.
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
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "message": "Smart Expense Tracker AI Backend v2.0",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": [
            "POST /api/predict",
            "POST /api/anomalies",
            "POST /api/insights",
            "POST /api/subscriptions",
            "POST /api/goals",
            "POST /api/forecast",
            "POST /api/categorize",
            "POST /api/upload/parse",
            "GET  /api/upload/template",
            "POST /api/chat",
        ]
    }
