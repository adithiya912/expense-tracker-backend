"""
Predictions router — /api/predict, /api/anomalies, /api/insights,
                     /api/subscriptions, /api/goals, /api/forecast, /api/categorize
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from schemas.models import (
    PredictionRequest, PredictionResponse, CategoryPrediction,
    AnomalyRequest, AnomalyResponse, AnomalyResult,
    InsightResponse, BudgetSuggestion
)
from models.predictor import ExpensePredictor
from models.anomaly import AnomalyDetector
from models.subscriptions import detect_subscriptions
from models.goals import generate_savings_plan
import logging
import numpy as np

router = APIRouter(prefix="/api", tags=["AI Predictions"])
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
async def predict_expenses(request: PredictionRequest):
    if len(request.expenses) == 0:
        raise HTTPException(status_code=400, detail="No expense data provided.")

    expenses_dict = [
        {"id": e.id, "user_id": e.user_id, "amount": e.amount,
         "category": e.category.value, "description": e.description, "date": str(e.date)}
        for e in request.expenses
    ]

    predictor = ExpensePredictor()
    trained = predictor.train(expenses_dict)

    if not trained:
        raise HTTPException(status_code=422, detail="Not enough data. Add at least 3 expenses.")

    predicted_total, category_preds, confidence = predictor.predict_next_month()

    now = datetime.now()
    next_month_str = f"{now.year + 1}-01" if now.month == 12 else f"{now.year}-{now.month + 1:02d}"

    cat_list = [
        CategoryPrediction(category=cat, predicted_amount=info["predicted_amount"],
                           trend=info["trend"], confidence=info["confidence"])
        for cat, info in category_preds.items() if info["predicted_amount"] > 0
    ]

    return PredictionResponse(
        user_id=request.user_id, predicted_month=next_month_str,
        predicted_total=predicted_total, category_predictions=cat_list,
        confidence_score=confidence, model_used="LinearRegression + TimeSeriesFeatures"
    )


@router.post("/anomalies", response_model=AnomalyResponse)
async def detect_anomalies(request: AnomalyRequest):
    if len(request.expenses) == 0:
        raise HTTPException(status_code=400, detail="No expense data provided.")

    expenses_dict = [
        {"id": e.id, "amount": e.amount, "category": e.category.value, "date": str(e.date)}
        for e in request.expenses
    ]

    detector = AnomalyDetector(contamination=0.08)
    results = detector.detect(expenses_dict)

    anomaly_results = [
        AnomalyResult(expense_id=r["expense_id"], date=r["date"], amount=r["amount"],
                      category=r["category"], is_anomaly=r["is_anomaly"],
                      anomaly_score=r["anomaly_score"], severity=r["severity"], reason=r["reason"])
        for r in results
    ]

    return AnomalyResponse(
        user_id=request.user_id,
        anomalies_found=sum(1 for r in anomaly_results if r.is_anomaly),
        results=anomaly_results
    )


@router.post("/insights", response_model=InsightResponse)
async def get_insights(request: PredictionRequest):
    if len(request.expenses) == 0:
        raise HTTPException(status_code=400, detail="No expense data provided.")

    expenses_dict = [
        {"id": e.id, "amount": e.amount, "category": e.category.value, "date": str(e.date), "day_of_week": 0}
        for e in request.expenses
    ]

    predictor = ExpensePredictor()
    predictor.train(expenses_dict)
    insights = predictor.get_insights(expenses_dict)

    suggestions = [
        BudgetSuggestion(category=s["category"], current_avg=s["current_avg"],
                         suggested_budget=s["suggested_budget"], potential_savings=s["potential_savings"], tip=s["tip"])
        for s in insights["budget_suggestions"]
    ]

    return InsightResponse(
        user_id=request.user_id,
        total_spent_last_30d=insights["total_spent_last_30d"],
        avg_daily_spend=insights["avg_daily_spend"],
        top_category=insights["top_category"],
        spending_trend=insights["spending_trend"],
        budget_suggestions=suggestions,
        habit_analysis=insights["habit_analysis"],
        overspending_categories=insights["overspending_categories"],
        generated_at=datetime.now()
    )


# ─── NEW ENDPOINTS ───────────────────────────────────────────────────────────

class SubscriptionRequest(BaseModel):
    user_id: str
    expenses: List[dict]


class GoalRequest(BaseModel):
    user_id: str
    expenses: List[dict]
    target_amount: float
    months: int
    monthly_income: Optional[float] = 0.0


class ForecastRequest(BaseModel):
    user_id: str
    expenses: List[dict]
    horizon_months: Optional[int] = 6


class CategorizeRequest(BaseModel):
    descriptions: List[str]


@router.post("/subscriptions")
async def detect_subs(request: SubscriptionRequest):
    """Detect recurring subscriptions and periodic expenses."""
    if not request.expenses:
        return {"subscriptions": [], "total_monthly": 0.0, "annual_total": 0.0, "savings_potential": 0.0}
    return detect_subscriptions(request.expenses)


@router.post("/goals")
async def savings_goal(request: GoalRequest):
    """Generate a personalized savings plan for a target amount."""
    if request.target_amount <= 0 or request.months <= 0:
        raise HTTPException(400, "target_amount and months must be positive.")
    return generate_savings_plan(
        expenses=request.expenses,
        target_amount=request.target_amount,
        months=request.months,
        monthly_income=request.monthly_income or 0.0,
    )


@router.post("/forecast")
async def forecast_expenses(request: ForecastRequest):
    """Generate N-month spending forecast with confidence bands."""
    if not request.expenses:
        raise HTTPException(400, "No expense data provided.")

    import pandas as pd
    from sklearn.linear_model import Ridge

    df = pd.DataFrame(request.expenses)
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().reset_index()
    monthly["month_ordinal"] = monthly["date"].apply(lambda x: (x.year - 2020) * 12 + x.month)
    monthly["amount"] = monthly["amount"].astype(float)

    horizon = min(request.horizon_months or 6, 12)
    now = datetime.now()

    if len(monthly) < 2:
        avg = float(df["amount"].sum()) / max(1, len(df)) * 30
        forecast = []
        for i in range(1, horizon + 1):
            m = now.month + i
            y = now.year + (m - 1) // 12
            m = ((m - 1) % 12) + 1
            forecast.append({"month": f"{y}-{m:02d}", "predicted": round(avg, 2),
                             "lower": round(avg * 0.85, 2), "upper": round(avg * 1.15, 2)})
        return {"forecast": forecast, "confidence": 0.4, "model": "average_fallback"}

    X = monthly[["month_ordinal"]].values
    y = monthly["amount"].values
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    std = float(np.std(y - model.predict(X)))

    last_ordinal = int(monthly["month_ordinal"].max())
    forecast = []
    for i in range(1, horizon + 1):
        ordinal = last_ordinal + i
        year = 2020 + (ordinal - 1) // 12
        month = ((ordinal - 1) % 12) + 1
        pred = max(0, float(model.predict([[ordinal]])[0]))
        forecast.append({"month": f"{year}-{month:02d}", "predicted": round(pred, 2),
                         "lower": round(max(0, pred - 1.5 * std), 2), "upper": round(pred + 1.5 * std, 2)})

    confidence = min(0.92, 0.45 + len(monthly) * 0.06)
    return {"forecast": forecast, "confidence": round(confidence, 2), "model": "Ridge Regression"}


@router.post("/categorize")
async def categorize_descriptions(request: CategorizeRequest):
    """Auto-categorize transaction descriptions using NLP."""
    from models.categorizer import categorize_batch
    return {"results": categorize_batch(request.descriptions)}


@router.get("/health")
async def health():
    return {"status": "ok", "service": "Smart Expense Tracker AI Backend"}
