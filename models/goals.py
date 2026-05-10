"""
Goal-based budget recommendation engine.
Given a savings target and deadline, generates a personalized savings plan.
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np


# Benchmark spending ratios (% of income/total spend) from financial planning guidelines
BENCHMARK_PCT = {
    "Food": 0.25,
    "Travel": 0.10,
    "Bills": 0.20,
    "Entertainment": 0.05,
    "Health": 0.08,
    "Shopping": 0.10,
    "Education": 0.05,
    "Other": 0.07,
}

CATEGORY_TIPS = {
    "Food": [
        "Meal-prep on Sundays to reduce daily food delivery costs.",
        "Cook at home for at least 4 weekday dinners.",
        "Use grocery lists to avoid impulse buys.",
        "Switch from restaurant coffee to home-brewed — saves ₹2000–3000/month.",
    ],
    "Travel": [
        "Use public transport for daily commute instead of cab.",
        "Carpool with colleagues to split fuel costs.",
        "Walk or cycle for distances under 2 km.",
        "Book transport in advance to get lower fares.",
    ],
    "Entertainment": [
        "Share streaming subscriptions with family members.",
        "Look for free community events and open-air shows.",
        "Use library memberships instead of buying books.",
        "Limit dining out to once a week.",
    ],
    "Shopping": [
        "Apply the 48-hour rule before non-essential purchases.",
        "Unsubscribe from promotional emails to reduce impulse buying.",
        "Use cashback apps and credit card rewards strategically.",
        "Buy seasonal items during sale periods only.",
    ],
    "Bills": [
        "Review all subscriptions and cancel unused ones.",
        "Negotiate better rates for internet and mobile plans.",
        "Switch to energy-efficient appliances to cut electricity bills.",
        "Bundle services where possible (internet + streaming).",
    ],
    "Health": [
        "Use generic medicines instead of branded ones (same composition, lower cost).",
        "Prefer preventive checkups over emergency visits.",
        "Use government hospitals for routine consultations.",
        "Join community sports groups instead of gym memberships.",
    ],
    "Education": [
        "Use free resources (Coursera audit, YouTube, Khan Academy) before paying.",
        "Apply for scholarships or employer training budgets.",
        "Buy used textbooks or share with classmates.",
    ],
    "Other": [
        "Break down 'Other' expenses further to identify hidden spending.",
        "Set a weekly cash envelope for miscellaneous needs.",
    ],
}


def generate_savings_plan(
    expenses: List[dict],
    target_amount: float,
    months: int,
    monthly_income: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate a personalized savings plan.

    Args:
        expenses: List of expense records
        target_amount: How much the user wants to save total
        months: How many months to achieve the goal
        monthly_income: Optional — user's monthly income

    Returns:
        Savings plan with per-category cuts and actionable tips
    """
    if not expenses or months <= 0 or target_amount <= 0:
        return {"error": "Invalid inputs for savings plan."}

    df = pd.DataFrame(expenses)
    df["date"] = pd.to_datetime(df["date"])

    # Average monthly spending per category
    n_months = max(1, df["date"].dt.to_period("M").nunique())
    cat_monthly = (
        df.groupby("category")["amount"].sum() / n_months
    ).to_dict()

    total_monthly_spend = sum(cat_monthly.values())
    required_monthly_savings = round(target_amount / months, 2)

    # Calculate how much can be cut per category
    cuts = []
    total_cuttable = 0.0

    for cat, current_avg in sorted(cat_monthly.items(), key=lambda x: -x[1]):
        benchmark_ratio = BENCHMARK_PCT.get(cat, 0.07)
        # Benchmark: what they "should" spend if total is their income
        reference = total_monthly_spend * benchmark_ratio
        excess = max(0, current_avg - reference)
        # Suggest cutting 20–40% of excess or 15% of total, whichever is lower
        suggested_cut = min(excess * 0.5, current_avg * 0.20)
        suggested_budget = max(0, current_avg - suggested_cut)

        cuts.append({
            "category": cat,
            "current_avg": round(current_avg, 2),
            "suggested_budget": round(suggested_budget, 2),
            "monthly_cut": round(suggested_cut, 2),
            "annual_savings": round(suggested_cut * 12, 2),
            "tips": CATEGORY_TIPS.get(cat, ["Review spending in this category."])[:2],
            "priority": "high" if suggested_cut >= required_monthly_savings * 0.3 else
                        "medium" if suggested_cut >= required_monthly_savings * 0.1 else "low",
        })
        total_cuttable += suggested_cut

    # Feasibility check
    feasibility_pct = min(100, (total_cuttable / required_monthly_savings) * 100) if required_monthly_savings > 0 else 100
    if feasibility_pct >= 90:
        feasibility = "achievable"
    elif feasibility_pct >= 60:
        feasibility = "challenging"
    else:
        feasibility = "difficult"

    # Sort by priority (high first)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    cuts.sort(key=lambda x: priority_order[x["priority"]])

    # Projected savings month by month
    projected = []
    cumulative = 0.0
    for m in range(1, months + 1):
        cumulative += total_cuttable
        projected.append({
            "month": m,
            "cumulative_savings": round(min(cumulative, target_amount), 2),
            "target": round((target_amount / months) * m, 2),
        })

    return {
        "target_amount": target_amount,
        "months": months,
        "required_monthly_savings": required_monthly_savings,
        "achievable_monthly_savings": round(total_cuttable, 2),
        "feasibility": feasibility,
        "feasibility_pct": round(feasibility_pct, 1),
        "category_cuts": cuts,
        "projected_savings": projected,
        "summary": _build_summary(required_monthly_savings, total_cuttable, months, target_amount, feasibility),
    }


def _build_summary(required: float, achievable: float, months: int, target: float, feasibility: str) -> str:
    if feasibility == "achievable":
        return (
            f"Great news! By making targeted cuts, you can save ₹{achievable:.0f}/month "
            f"and reach your ₹{target:.0f} goal in {months} months. "
            f"Focus on the High-priority categories first."
        )
    elif feasibility == "challenging":
        adj_months = int(target / achievable) + 1 if achievable > 0 else months * 2
        return (
            f"Your goal is achievable but will take longer. "
            f"At ₹{achievable:.0f}/month in savings, you'd reach ₹{target:.0f} in ~{adj_months} months. "
            f"Consider increasing income or reducing the target."
        )
    else:
        return (
            f"This goal requires ₹{required:.0f}/month but your realistic savings potential is "
            f"₹{achievable:.0f}/month. Try extending the deadline or reducing the target amount."
        )
