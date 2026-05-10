"""
AI Chat router — powered by Google Gemini 1.5 Flash.
Answers natural language questions about the user's spending.
"""
import os
import json
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np

router = APIRouter(prefix="/api", tags=["chat"])

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


class ChatRequest(BaseModel):
    question: str
    expenses: List[dict]
    currency: Optional[str] = "INR"


class ChatResponse(BaseModel):
    answer: str
    suggestions: List[str]
    used_ai: bool


def _build_financial_context(expenses: List[dict], currency: str) -> str:
    """Build a rich financial summary from expense data for the AI prompt."""
    if not expenses:
        return "No expense data available."

    df = pd.DataFrame(expenses)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    symbol = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$"}.get(currency, "₹")
    n_months = max(1, df["date"].dt.to_period("M").nunique())

    total = df["amount"].sum()
    monthly_avg = total / n_months
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_cat = cat_totals.index[0] if not cat_totals.empty else "N/A"
    top_cat_amt = cat_totals.iloc[0] if not cat_totals.empty else 0

    # Monthly trend
    monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
    trend = "stable"
    if len(monthly) >= 2:
        if monthly.iloc[-1] > monthly.iloc[-2] * 1.1:
            trend = "increasing"
        elif monthly.iloc[-1] < monthly.iloc[-2] * 0.9:
            trend = "decreasing"

    # Last month details
    last_month = monthly.iloc[-1] if not monthly.empty else 0

    # Category breakdown
    cat_breakdown = "\n".join(
        f"  - {cat}: {symbol}{amt:.0f} ({amt/total*100:.1f}%)"
        for cat, amt in cat_totals.items()
    )

    # Avg daily
    avg_daily = monthly_avg / 30

    # Recent high-value transactions
    top_txns = df.nlargest(3, "amount")[["date", "amount", "category", "description"]].to_dict("records")
    top_txns_str = "\n".join(
        f"  - {symbol}{t['amount']:.0f} on {str(t['date'])[:10]} ({t['category']}): {t.get('description', '')}"
        for t in top_txns
    )

    return f"""
FINANCIAL CONTEXT:
- Total transactions: {len(df)} expenses over {n_months} months
- Total spent: {symbol}{total:.0f}
- Monthly average: {symbol}{monthly_avg:.0f}
- Daily average: {symbol}{avg_daily:.0f}
- Last month: {symbol}{last_month:.0f}
- Spending trend: {trend}
- Top category: {top_cat} ({symbol}{top_cat_amt:.0f} total)

CATEGORY BREAKDOWN:
{cat_breakdown}

TOP 3 LARGEST TRANSACTIONS:
{top_txns_str}

MONTHLY HISTORY:
{chr(10).join(f"  {str(period)}: {symbol}{amt:.0f}" for period, amt in monthly.items())}
"""


def _rule_based_answer(question: str, expenses: List[dict], currency: str) -> ChatResponse:
    """Fallback rule-based responses when Gemini API is unavailable."""
    q = question.lower()
    df = pd.DataFrame(expenses) if expenses else pd.DataFrame()
    symbol = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "₹")

    if df.empty:
        return ChatResponse(
            answer="I don't have any expense data to analyze yet. Please add some expenses first!",
            suggestions=["Add your first expense", "Load sample data to test AI features"],
            used_ai=False,
        )

    df["date"] = pd.to_datetime(df["date"])
    total = df["amount"].sum()
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_cat = cat_totals.index[0]
    top_amt = cat_totals.iloc[0]

    if any(w in q for w in ["top", "most", "highest", "where", "biggest"]):
        breakdown = "\n".join(f"• {c}: {symbol}{a:.0f} ({a/total*100:.1f}%)" for c, a in cat_totals.head(3).items())
        answer = f"Your top spending categories are:\n\n{breakdown}\n\n**{top_cat}** takes the largest share at {symbol}{top_amt:.0f}."
        suggestions = [f"How can I reduce {top_cat} spending?", "What's my monthly average?"]

    elif any(w in q for w in ["save", "saving", "cut", "reduce", "less"]):
        potential = total * 0.15
        answer = f"Based on your spending of {symbol}{total:.0f}, you could potentially save **{symbol}{potential:.0f}** (15%) by:\n\n• Reducing {top_cat} spending\n• Reviewing subscriptions\n• Setting category budgets"
        suggestions = ["Show me my subscriptions", "Create a savings goal", f"How to cut {top_cat} costs?"]

    elif any(w in q for w in ["average", "monthly", "per month"]):
        n_months = max(1, df["date"].dt.to_period("M").nunique())
        monthly_avg = total / n_months
        answer = f"Your monthly average spend is **{symbol}{monthly_avg:.0f}** across {n_months} months, totaling {symbol}{total:.0f}."
        suggestions = ["How does this compare to last month?", "What should my budget be?"]

    elif any(w in q for w in ["food", "travel", "bills", "entertainment", "health", "shopping"]):
        for cat in cat_totals.index:
            if cat.lower() in q:
                amt = cat_totals[cat]
                pct = amt / total * 100
                answer = f"You've spent **{symbol}{amt:.0f}** on {cat} ({pct:.1f}% of total). " + \
                         ("This is quite high — consider reviewing it." if pct > 30 else "This seems reasonable.")
                suggestions = [f"How to reduce {cat} spending?", "Show all my categories"]
                return ChatResponse(answer=answer, suggestions=suggestions, used_ai=False)
        answer = f"Your total spending is {symbol}{total:.0f}."
        suggestions = ["What are my top categories?"]

    else:
        answer = f"Your total spending is **{symbol}{total:.0f}** across {len(df)} transactions. Your biggest expense category is **{top_cat}** at {symbol}{top_amt:.0f}."
        suggestions = ["Where am I spending the most?", "How much can I save?", "What's my monthly average?"]

    return ChatResponse(answer=answer, suggestions=suggestions, used_ai=False)


SYSTEM_PROMPT = """You are FinanceAI, a personal financial advisor integrated into a Smart Expense Tracker app.
Your job is to analyze the user's real spending data and provide helpful, specific, actionable advice.

Rules:
- Always reference actual numbers from their data (amounts, categories, percentages)
- Be encouraging but honest about overspending
- Give 2-3 specific, actionable steps the user can take
- Keep responses concise but insightful (2-4 paragraphs max)
- Use the currency symbol from the context
- Format key numbers in **bold**
- End with a specific recommendation
"""


@router.post("/chat", response_model=ChatResponse)
async def ask_ai(request: ChatRequest):
    """Ask AI a natural language question about spending."""
    context = _build_financial_context(request.expenses, request.currency or "INR")

    if not GEMINI_API_KEY:
        return _rule_based_answer(request.question, request.expenses, request.currency or "INR")

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        full_prompt = f"""{SYSTEM_PROMPT}

{context}

USER QUESTION: {request.question}

Respond with JSON in this exact format:
{{
  "answer": "your detailed answer here",
  "suggestions": ["follow-up question 1", "follow-up question 2", "follow-up question 3"]
}}"""

        response = model.generate_content(full_prompt)
        text = response.text.strip()

        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)
        return ChatResponse(
            answer=parsed.get("answer", text),
            suggestions=parsed.get("suggestions", []),
            used_ai=True,
        )

    except json.JSONDecodeError:
        # Gemini responded but not in JSON — use raw text
        return ChatResponse(
            answer=response.text if 'response' in dir() else "Unable to parse AI response.",
            suggestions=["What are my top spending categories?", "How much can I save?"],
            used_ai=True,
        )
    except Exception as e:
        # Fallback to rule-based
        return _rule_based_answer(request.question, request.expenses, request.currency or "INR")
