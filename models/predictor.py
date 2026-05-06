"""
Expense Predictor — uses LinearRegression + time-series features
to predict next month's total and category-wise spending.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


CATEGORIES = ["Food", "Travel", "Bills", "Entertainment", "Health", "Shopping", "Education", "Other"]


class ExpensePredictor:
    def __init__(self):
        self.models: Dict[str, LinearRegression] = {}
        self.total_model = LinearRegression()
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-series features from expense dataframe."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month_num"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["day_of_week"] = df["date"].dt.dayofweek
        df["quarter"] = df["date"].dt.quarter
        df["month_ordinal"] = (df["year"] - df["year"].min()) * 12 + df["month_num"]
        return df

    def _aggregate_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate expenses by month and category."""
        df["month_period"] = df["date"].dt.to_period("M")
        monthly = df.groupby(["month_period", "category"])["amount"].sum().reset_index()
        monthly["month_ordinal"] = monthly["month_period"].dt.to_timestamp().apply(
            lambda x: (x.year - 2020) * 12 + x.month
        )
        return monthly

    def train(self, expenses: List[dict]) -> bool:
        """Train prediction models on historical expense data."""
        if len(expenses) < 3:
            return False

        df = pd.DataFrame(expenses)
        df = self._build_features(df)
        monthly = self._aggregate_monthly(df)

        if monthly.empty:
            return False

        # Train total monthly model
        total_by_month = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().reset_index()
        total_by_month["month_ordinal"] = total_by_month["date"].dt.to_timestamp().apply(
            lambda x: (x.year - 2020) * 12 + x.month
        )

        if len(total_by_month) >= 2:
            X_total = total_by_month[["month_ordinal"]].values
            y_total = total_by_month["amount"].values
            self.total_model.fit(X_total, y_total)

        # Train per-category models
        for category in CATEGORIES:
            cat_data = monthly[monthly["category"] == category].copy()
            if len(cat_data) >= 2:
                X = cat_data[["month_ordinal"]].values
                y = cat_data["amount"].values
                model = LinearRegression()
                model.fit(X, y)
                self.models[category] = model

        self.is_trained = True
        self.monthly_data = monthly
        self.total_data = total_by_month
        return True

    def predict_next_month(self) -> Tuple[float, Dict[str, dict]]:
        """Predict next month's total and category spending."""
        import datetime
        now = datetime.datetime.now()
        next_month = (now.year - 2020) * 12 + now.month + 1

        # Predict total
        predicted_total = max(0, float(self.total_model.predict([[next_month]])[0]))

        # Predict per category
        category_preds = {}
        for category in CATEGORIES:
            if category in self.models:
                pred = max(0, float(self.models[category].predict([[next_month]])[0]))

                # Compute trend using last 3 months
                cat_data = self.monthly_data[self.monthly_data["category"] == category]
                if len(cat_data) >= 3:
                    recent = cat_data.sort_values("month_ordinal").tail(3)["amount"].values
                    if recent[-1] > recent[0] * 1.05:
                        trend = "increasing"
                    elif recent[-1] < recent[0] * 0.95:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"

                # Confidence based on data points
                n_points = len(self.models[category].coef_) if hasattr(self.models[category], 'coef_') else 0
                cat_count = len(self.monthly_data[self.monthly_data["category"] == category])
                confidence = min(0.95, 0.5 + cat_count * 0.08)

                category_preds[category] = {
                    "predicted_amount": round(pred, 2),
                    "trend": trend,
                    "confidence": round(confidence, 2)
                }
            else:
                category_preds[category] = {
                    "predicted_amount": 0.0,
                    "trend": "stable",
                    "confidence": 0.3
                }

        n_months = len(self.total_data)
        overall_confidence = min(0.92, 0.4 + n_months * 0.07)
        return round(predicted_total, 2), category_preds, round(overall_confidence, 2)

    def get_insights(self, expenses: List[dict]) -> dict:
        """Generate smart spending insights."""
        df = pd.DataFrame(expenses)
        df = self._build_features(df)
        df["date"] = pd.to_datetime(df["date"])

        # Last 30 days
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        recent = df[df["date"] >= cutoff]

        total_last_30 = float(recent["amount"].sum()) if not recent.empty else 0
        avg_daily = round(total_last_30 / 30, 2)

        # Top category
        if not df.empty:
            top_cat = df.groupby("category")["amount"].sum().idxmax()
        else:
            top_cat = "N/A"

        # Trend
        monthly_totals = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
        if len(monthly_totals) >= 2:
            vals = monthly_totals.values
            if vals[-1] > vals[-2] * 1.1:
                trend = "increasing"
            elif vals[-1] < vals[-2] * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Budget suggestions per category
        cat_avgs = df.groupby("category")["amount"].sum() / max(1, df["date"].dt.to_period("M").nunique())
        suggestions = []
        for cat, avg in cat_avgs.items():
            suggested = round(avg * 0.85, 2)
            savings = round(avg - suggested, 2)
            tips = {
                "Food": "Cook at home more often and meal-prep on weekends.",
                "Travel": "Use public transport or carpool to reduce costs.",
                "Entertainment": "Look for free events and use streaming bundles.",
                "Shopping": "Apply the 24-hour rule before buying non-essentials.",
                "Health": "Compare prices and use generic medications.",
                "Bills": "Review subscriptions monthly and cancel unused ones.",
                "Education": "Use free platforms like Coursera or YouTube first.",
                "Other": "Track these miscellaneous expenses more specifically.",
            }
            suggestions.append({
                "category": cat,
                "current_avg": round(float(avg), 2),
                "suggested_budget": float(suggested),
                "potential_savings": float(savings),
                "tip": tips.get(cat, "Review and optimize spending in this category.")
            })

        # Habit analysis
        habits = []
        if not df.empty:
            weekday_avg = df[df["day_of_week"] < 5]["amount"].mean() if not df[df["day_of_week"] < 5].empty else 0
            weekend_avg = df[df["day_of_week"] >= 5]["amount"].mean() if not df[df["day_of_week"] >= 5].empty else 0
            if weekend_avg > weekday_avg * 1.3:
                habits.append("You spend significantly more on weekends — consider a weekend budget cap.")
            if weekday_avg > weekend_avg * 1.3:
                habits.append("Most spending happens on weekdays — likely driven by work-related expenses.")

            if trend == "increasing":
                habits.append("Your spending has been growing month-over-month. Consider reviewing your budget.")
            elif trend == "decreasing":
                habits.append("Great job! Your spending is trending downward.")

            # Check food dominance
            if not df.empty:
                food_pct = df[df["category"] == "Food"]["amount"].sum() / df["amount"].sum() * 100
                if food_pct > 40:
                    habits.append(f"Food accounts for {food_pct:.0f}% of expenses — try meal prepping to save.")

        overspending = []
        if len(monthly_totals) >= 2:
            for cat in CATEGORIES:
                cat_monthly = df[df["category"] == cat].groupby(df["date"].dt.to_period("M"))["amount"].sum()
                if len(cat_monthly) >= 2 and cat_monthly.iloc[-1] > cat_monthly.mean() * 1.4:
                    overspending.append(cat)

        return {
            "total_spent_last_30d": round(total_last_30, 2),
            "avg_daily_spend": avg_daily,
            "top_category": top_cat,
            "spending_trend": trend,
            "budget_suggestions": suggestions,
            "habit_analysis": habits if habits else ["Keep tracking your expenses to get personalized insights!"],
            "overspending_categories": overspending,
        }
