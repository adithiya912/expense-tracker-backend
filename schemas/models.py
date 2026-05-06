from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import date, datetime
from enum import Enum


class Category(str, Enum):
    food = "Food"
    travel = "Travel"
    bills = "Bills"
    entertainment = "Entertainment"
    health = "Health"
    shopping = "Shopping"
    education = "Education"
    other = "Other"


class ExpenseRecord(BaseModel):
    id: Optional[str] = None
    user_id: str
    amount: float = Field(gt=0)
    category: Category
    description: Optional[str] = None
    date: date


class PredictionRequest(BaseModel):
    user_id: str
    expenses: List[ExpenseRecord]


class CategoryPrediction(BaseModel):
    category: str
    predicted_amount: float
    trend: str  # "increasing", "decreasing", "stable"
    confidence: float  # 0-1


class PredictionResponse(BaseModel):
    user_id: str
    predicted_month: str  # "YYYY-MM"
    predicted_total: float
    category_predictions: List[CategoryPrediction]
    confidence_score: float
    model_used: str


class AnomalyRequest(BaseModel):
    user_id: str
    expenses: List[ExpenseRecord]


class AnomalyResult(BaseModel):
    expense_id: Optional[str]
    date: date
    amount: float
    category: str
    is_anomaly: bool
    anomaly_score: float
    severity: str  # "low", "medium", "high", "normal"
    reason: Optional[str]


class AnomalyResponse(BaseModel):
    user_id: str
    anomalies_found: int
    results: List[AnomalyResult]


class BudgetSuggestion(BaseModel):
    category: str
    current_avg: float
    suggested_budget: float
    potential_savings: float
    tip: str


class InsightResponse(BaseModel):
    user_id: str
    total_spent_last_30d: float
    avg_daily_spend: float
    top_category: str
    spending_trend: str  # "increasing", "decreasing", "stable"
    budget_suggestions: List[BudgetSuggestion]
    habit_analysis: List[str]
    overspending_categories: List[str]
    generated_at: datetime
