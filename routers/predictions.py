"""
Predictions router — /api/predict, /api/anomalies, /api/insights
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from schemas.models import (
    PredictionRequest, PredictionResponse, CategoryPrediction,
    AnomalyRequest, AnomalyResponse, AnomalyResult,
    InsightResponse, BudgetSuggestion
)
from models.predictor import ExpensePredictor
from models.anomaly import AnomalyDetector
import logging

router = APIRouter(prefix="/api", tags=["AI Predictions"])
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
async def predict_expenses(request: PredictionRequest):
    """Predict next month's total and category-wise expenses."""
    if len(request.expenses) == 0:
        raise HTTPException(status_code=400, detail="No expense data provided for prediction.")

    expenses_dict = [
        {
            "id": e.id,
            "user_id": e.user_id,
            "amount": e.amount,
            "category": e.category.value,
            "description": e.description,
            "date": str(e.date)
        }
        for e in request.expenses
    ]

    predictor = ExpensePredictor()
    trained = predictor.train(expenses_dict)

    if not trained:
        raise HTTPException(
            status_code=422,
            detail="Not enough data to generate predictions. Please add at least 3 expenses."
        )

    predicted_total, category_preds, confidence = predictor.predict_next_month()

    # Build next month label
    now = datetime.now()
    if now.month == 12:
        next_month_str = f"{now.year + 1}-01"
    else:
        next_month_str = f"{now.year}-{now.month + 1:02d}"

    cat_list = [
        CategoryPrediction(
            category=cat,
            predicted_amount=info["predicted_amount"],
            trend=info["trend"],
            confidence=info["confidence"]
        )
        for cat, info in category_preds.items()
        if info["predicted_amount"] > 0
    ]

    return PredictionResponse(
        user_id=request.user_id,
        predicted_month=next_month_str,
        predicted_total=predicted_total,
        category_predictions=cat_list,
        confidence_score=confidence,
        model_used="LinearRegression + TimeSeriesFeatures"
    )


@router.post("/anomalies", response_model=AnomalyResponse)
async def detect_anomalies(request: AnomalyRequest):
    """Detect unusual spending patterns using Isolation Forest."""
    if len(request.expenses) == 0:
        raise HTTPException(status_code=400, detail="No expense data provided.")

    expenses_dict = [
        {
            "id": e.id,
            "amount": e.amount,
            "category": e.category.value,
            "date": str(e.date)
        }
        for e in request.expenses
    ]

    detector = AnomalyDetector(contamination=0.08)
    results = detector.detect(expenses_dict)

    anomaly_results = [
        AnomalyResult(
            expense_id=r["expense_id"],
            date=r["date"],
            amount=r["amount"],
            category=r["category"],
            is_anomaly=r["is_anomaly"],
            anomaly_score=r["anomaly_score"],
            severity=r["severity"],
            reason=r["reason"]
        )
        for r in results
    ]

    anomalies_found = sum(1 for r in anomaly_results if r.is_anomaly)

    return AnomalyResponse(
        user_id=request.user_id,
        anomalies_found=anomalies_found,
        results=anomaly_results
    )


@router.post("/insights", response_model=InsightResponse)
async def get_insights(request: PredictionRequest):
    """Generate AI-powered spending insights and budget suggestions."""
    if len(request.expenses) == 0:
        raise HTTPException(status_code=400, detail="No expense data provided.")

    expenses_dict = [
        {
            "id": e.id,
            "amount": e.amount,
            "category": e.category.value,
            "date": str(e.date),
            "day_of_week": 0  # Will be computed in predictor
        }
        for e in request.expenses
    ]

    predictor = ExpensePredictor()
    predictor.train(expenses_dict)
    insights = predictor.get_insights(expenses_dict)

    suggestions = [
        BudgetSuggestion(
            category=s["category"],
            current_avg=s["current_avg"],
            suggested_budget=s["suggested_budget"],
            potential_savings=s["potential_savings"],
            tip=s["tip"]
        )
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


@router.get("/health")
async def health():
    return {"status": "ok", "service": "Smart Expense Tracker AI Backend"}
