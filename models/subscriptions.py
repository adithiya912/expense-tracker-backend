"""
Subscription & recurring expense detector.
Groups transactions by merchant similarity and detects periodic patterns.
"""
import re
from typing import List, Dict, Any
from rapidfuzz import fuzz
import pandas as pd
import numpy as np


KNOWN_SUBSCRIPTIONS = {
    "netflix": {"name": "Netflix", "category": "Entertainment", "type": "streaming"},
    "spotify": {"name": "Spotify", "category": "Entertainment", "type": "streaming"},
    "amazon prime": {"name": "Amazon Prime", "category": "Entertainment", "type": "streaming"},
    "hotstar": {"name": "Disney+ Hotstar", "category": "Entertainment", "type": "streaming"},
    "youtube premium": {"name": "YouTube Premium", "category": "Entertainment", "type": "streaming"},
    "apple music": {"name": "Apple Music", "category": "Entertainment", "type": "streaming"},
    "jio": {"name": "Jio Recharge", "category": "Bills", "type": "telecom"},
    "airtel": {"name": "Airtel", "category": "Bills", "type": "telecom"},
    "bsnl": {"name": "BSNL", "category": "Bills", "type": "telecom"},
    "gym": {"name": "Gym Membership", "category": "Health", "type": "fitness"},
    "icloud": {"name": "iCloud", "category": "Bills", "type": "cloud"},
    "google one": {"name": "Google One", "category": "Bills", "type": "cloud"},
    "dropbox": {"name": "Dropbox", "category": "Bills", "type": "cloud"},
    "github": {"name": "GitHub", "category": "Education", "type": "saas"},
    "notion": {"name": "Notion", "category": "Education", "type": "saas"},
    "zoom": {"name": "Zoom", "category": "Bills", "type": "saas"},
    "adobe": {"name": "Adobe", "category": "Shopping", "type": "saas"},
}


def _clean_merchant(description: str) -> str:
    """Normalize merchant name from description."""
    desc = description.lower()
    # Remove common noise patterns
    desc = re.sub(r'\b(payment|transfer|upi|neft|imps|ref|order|purchase|pos|atm)\b', '', desc)
    desc = re.sub(r'[0-9]{4,}', '', desc)   # Remove long numbers
    desc = re.sub(r'\s+', ' ', desc).strip()
    return desc[:40]


def _find_known_subscription(description: str) -> dict | None:
    """Check if description matches a known subscription service."""
    desc_lower = description.lower()
    for key, info in KNOWN_SUBSCRIPTIONS.items():
        if key in desc_lower:
            return info
    return None


def detect_subscriptions(expenses: List[dict]) -> Dict[str, Any]:
    """
    Detect recurring subscriptions and periodic expenses.
    Returns list of detected subscriptions with monthly cost estimates.
    """
    if len(expenses) < 3:
        return {"subscriptions": [], "total_monthly": 0.0, "savings_potential": 0.0}

    df = pd.DataFrame(expenses)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    subscriptions = []
    seen_merchants: set[str] = set()

    # Step 1: Check for known subscription services
    for _, row in df.iterrows():
        desc = str(row.get("description", "") or row.get("category", ""))
        match = _find_known_subscription(desc)
        if match:
            merchant_key = match["name"]
            if merchant_key not in seen_merchants:
                seen_merchants.add(merchant_key)
                # Find all transactions for this service
                txns = df[df["description"].str.lower().str.contains(
                    list(KNOWN_SUBSCRIPTIONS.keys())[
                        list(v["name"] for v in KNOWN_SUBSCRIPTIONS.values()).index(match["name"])
                    ], na=False
                )]
                avg_amount = float(txns["amount"].mean()) if not txns.empty else float(row["amount"])
                last_date = txns["date"].max() if not txns.empty else row["date"]
                subscriptions.append({
                    "name": match["name"],
                    "category": match["category"],
                    "type": match["type"],
                    "monthly_cost": round(avg_amount, 2),
                    "last_charged": str(last_date.date()),
                    "occurrences": len(txns),
                    "is_known": True,
                    "cancel_suggestion": _get_cancel_tip(match["name"], avg_amount),
                })

    # Step 2: Detect unknown recurring patterns by merchant clustering
    merchants: Dict[str, list] = {}
    for _, row in df.iterrows():
        merchant = _clean_merchant(str(row.get("description", "")))
        if not merchant:
            continue
        # Try to merge into existing merchant bucket
        merged = False
        for existing_key in list(merchants.keys()):
            if fuzz.ratio(merchant, existing_key) >= 75:
                merchants[existing_key].append(row)
                merged = True
                break
        if not merged:
            merchants[merchant] = [row]

    for merchant_name, txns in merchants.items():
        if len(txns) < 2:
            continue
        if merchant_name in seen_merchants:
            continue

        txn_df = pd.DataFrame(txns).sort_values("date")
        amounts = txn_df["amount"].values
        dates = pd.to_datetime(txn_df["date"].values)

        # Check if amounts are similar (within 20%)
        if len(amounts) >= 2:
            amount_cv = np.std(amounts) / (np.mean(amounts) + 1e-9)
            if amount_cv > 0.25:  # Too variable → not a subscription
                continue

        # Check if dates are periodic (within ±7 days of 30-day interval)
        if len(dates) >= 2:
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_gap = np.mean(gaps)
            if 20 <= avg_gap <= 45:  # Monthly-ish
                seen_merchants.add(merchant_name)
                subscriptions.append({
                    "name": merchant_name.title(),
                    "category": txns[0].get("category", "Other"),
                    "type": "recurring",
                    "monthly_cost": round(float(np.mean(amounts)), 2),
                    "last_charged": str(dates[-1].date()),
                    "occurrences": len(txns),
                    "is_known": False,
                    "cancel_suggestion": f"This appears to recur monthly (~every {int(avg_gap)} days). Review if still needed.",
                })

    total_monthly = sum(s["monthly_cost"] for s in subscriptions)
    # Estimate savings if user cancelled 2 least-used subscriptions
    savings_potential = sum(
        s["monthly_cost"] for s in sorted(subscriptions, key=lambda x: x["occurrences"])[:2]
    )

    return {
        "subscriptions": subscriptions,
        "total_monthly": round(total_monthly, 2),
        "annual_total": round(total_monthly * 12, 2),
        "savings_potential": round(savings_potential, 2),
    }


def _get_cancel_tip(name: str, amount: float) -> str:
    tips = {
        "Netflix": "Consider sharing a plan or switching to the Basic tier.",
        "Spotify": "Switch to free tier with ads, or use YouTube Music free.",
        "Amazon Prime": "Check if you use delivery + video — if not, consider cancelling.",
        "Disney+ Hotstar": "Subscribe only during cricket season if that's the main use.",
        "YouTube Premium": "Use YouTube with an ad blocker instead.",
    }
    return tips.get(name, f"Review if you actively use this service (₹{amount:.0f}/month).")
