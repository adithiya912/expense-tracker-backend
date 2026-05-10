"""
NLP-based expense categorizer.
Uses keyword matching (primary) + fuzzy scoring (secondary).
Works entirely offline — no external API needed.
"""
from rapidfuzz import fuzz
from typing import Tuple

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Food": [
        "swiggy", "zomato", "restaurant", "food", "cafe", "coffee", "pizza", "burger",
        "grocery", "supermarket", "bigbasket", "dunzo", "blinkit", "instamart", "dining",
        "lunch", "dinner", "breakfast", "bakery", "dhaba", "canteen", "mess", "hotel",
        "eatery", "snack", "beverage", "juice", "chai", "tea", "milk", "vegetables",
        "fruits", "kirana", "provision", "dmart", "reliance fresh", "more supermarket",
        "dominos", "kfc", "mcdonalds", "subway", "starbucks", "ccd", "barista",
    ],
    "Travel": [
        "uber", "ola", "rapido", "taxi", "auto", "metro", "bus", "train", "irctc",
        "flight", "airline", "airport", "petrol", "fuel", "diesel", "toll", "parking",
        "cab", "indigo", "spicejet", "airindia", "vistara", "makemytrip", "goibibo",
        "cleartrip", "yatra", "redbus", "abhibus", "railway", "transport", "commute",
        "travel", "trip", "journey", "fare", "ticket",
    ],
    "Bills": [
        "electricity", "water", "gas", "internet", "broadband", "wifi", "mobile",
        "recharge", "postpaid", "prepaid", "dth", "cable", "insurance", "lic", "rent",
        "maintenance", "society", "tata sky", "airtel", "jio", "bsnl", "vi", "vodafone",
        "idea", "utility", "bill", "payment", "msedcl", "bescom", "tneb", "bescom",
        "subscription", "emi", "loan", "housing",
    ],
    "Entertainment": [
        "netflix", "amazon prime", "hotstar", "disney", "spotify", "youtube", "gaming",
        "movie", "cinema", "multiplex", "pvr", "inox", "concert", "event", "show",
        "theatre", "play", "game", "steam", "playstation", "xbox", "fun", "leisure",
        "recreation", "amusement", "park", "zoo",
    ],
    "Health": [
        "pharmacy", "medical", "medicine", "doctor", "hospital", "clinic", "lab",
        "diagnostic", "apollo", "1mg", "netmeds", "pharmeasy", "gym", "fitness",
        "yoga", "health", "wellness", "dental", "dentist", "eye", "optician",
        "physiotherapy", "consultation", "test", "checkup", "scan", "xray",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho", "mall", "shop",
        "store", "clothes", "clothing", "fashion", "electronics", "gadget", "mobile",
        "laptop", "furniture", "home", "decor", "kitchen", "appliance", "jewellery",
        "accessories", "gift", "purchase", "order",
    ],
    "Education": [
        "course", "udemy", "coursera", "edx", "byju", "unacademy", "tuition", "school",
        "college", "university", "books", "stationery", "library", "fees", "exam",
        "coaching", "training", "workshop", "seminar", "certification",
    ],
}

# Weight boost for exact substring match vs fuzzy
EXACT_WEIGHT = 1.0
FUZZY_THRESHOLD = 70  # rapidfuzz score out of 100


def categorize(description: str) -> Tuple[str, float]:
    """
    Categorize a transaction description.
    Returns (category, confidence) where confidence is 0.0–1.0.
    """
    if not description or not description.strip():
        return "Other", 0.3

    desc_lower = description.lower().strip()
    best_category = "Other"
    best_score = 0.0

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            # Exact substring match — high confidence
            if keyword in desc_lower:
                score = EXACT_WEIGHT
                if score > best_score:
                    best_score = score
                    best_category = category
                break

            # Fuzzy match — lower confidence
            ratio = fuzz.partial_ratio(keyword, desc_lower) / 100.0
            if ratio >= FUZZY_THRESHOLD / 100.0 and ratio > best_score:
                best_score = ratio
                best_category = category

    # Confidence: exact=0.92, fuzzy scales 0.5–0.85
    if best_score >= EXACT_WEIGHT:
        confidence = 0.92
    elif best_score > 0:
        confidence = round(0.5 + best_score * 0.35, 2)
    else:
        confidence = 0.30

    return best_category, confidence


def categorize_batch(descriptions: list[str]) -> list[dict]:
    """Categorize a list of descriptions, return list of {category, confidence}."""
    return [
        {"category": cat, "confidence": conf}
        for cat, conf in (categorize(d) for d in descriptions)
    ]
