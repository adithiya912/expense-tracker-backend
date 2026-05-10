"""
Excel/CSV upload router.
Parses bank statements, auto-categorizes transactions, returns preview for user confirmation.
"""
import io
import re
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from models.categorizer import categorize_batch

router = APIRouter(prefix="/api/upload", tags=["upload"])

VALID_EXTENSIONS = {".xlsx", ".xls", ".csv"}

# Common column name patterns for auto-detection
DATE_PATTERNS = ["date", "txn date", "transaction date", "value date", "posting date", "trans date"]
AMOUNT_PATTERNS = ["amount", "debit", "withdrawal", "spent", "credit", "deposit", "transaction amount"]
DESC_PATTERNS = ["description", "narration", "particulars", "details", "remarks", "transaction details",
                 "payee", "merchant", "memo", "reference"]
DEBIT_PATTERNS = ["debit", "withdrawal", "dr", "debit amount", "withdrawal amount"]
CREDIT_PATTERNS = ["credit", "deposit", "cr", "credit amount", "deposit amount"]


def _find_column(columns: list[str], patterns: list[str]) -> Optional[str]:
    """Find best matching column name from a list of patterns."""
    cols_lower = {c.lower().strip(): c for c in columns}
    for pat in patterns:
        if pat in cols_lower:
            return cols_lower[pat]
    # Partial match fallback
    for pat in patterns:
        for col_low, col_orig in cols_lower.items():
            if pat in col_low or col_low in pat:
                return col_orig
    return None


def _clean_amount(val) -> float:
    """Clean currency strings like '₹1,234.56' or '(500.00)' to float."""
    if pd.isna(val):
        return 0.0
    s = str(val).replace("₹", "").replace("$", "").replace(",", "").strip()
    # Negative in brackets: (500.00)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_dataframe(df: pd.DataFrame) -> dict:
    """
    Auto-detect columns and extract date/amount/description from a DataFrame.
    Returns parsed rows and column mapping used.
    """
    # Skip empty leading rows (some bank statements have headers offset)
    # Find header row by looking for 'date' keyword
    df = df.dropna(how="all")
    columns = [str(c) for c in df.columns]

    # Try to detect key columns
    date_col = _find_column(columns, DATE_PATTERNS)
    desc_col = _find_column(columns, DESC_PATTERNS)
    amount_col = _find_column(columns, AMOUNT_PATTERNS)
    debit_col = _find_column(columns, DEBIT_PATTERNS)
    credit_col = _find_column(columns, CREDIT_PATTERNS)

    if not date_col:
        raise ValueError("Could not detect a 'Date' column. Please ensure your file has a date column.")

    rows = []
    for _, row in df.iterrows():
        try:
            date_val = pd.to_datetime(row[date_col], dayfirst=True, errors="coerce")
            if pd.isna(date_val):
                continue

            # Determine amount
            if debit_col and credit_col:
                debit = _clean_amount(row.get(debit_col, 0))
                credit = _clean_amount(row.get(credit_col, 0))
                amount = debit if debit > 0 else (-credit if credit > 0 else 0.0)
            elif amount_col:
                amount = _clean_amount(row.get(amount_col, 0))
            else:
                continue  # Can't determine amount

            if amount <= 0:  # Only import expenses (debits)
                continue

            description = str(row.get(desc_col, "")) if desc_col else ""
            description = re.sub(r'\s+', ' ', description).strip()[:200]

            rows.append({
                "date": date_val.strftime("%Y-%m-%d"),
                "amount": round(amount, 2),
                "description": description,
                "raw_description": description,
            })
        except Exception:
            continue

    return {
        "rows": rows,
        "detected_columns": {
            "date": date_col,
            "amount": amount_col or f"{debit_col}/{credit_col}",
            "description": desc_col,
        },
    }


@router.post("/parse")
async def parse_file(file: UploadFile = File(...)):
    """
    Parse an Excel or CSV file and return auto-categorized transaction preview.
    """
    filename = file.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in VALID_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use .xlsx, .xls, or .csv")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(413, "File too large. Maximum size is 10MB.")

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8", errors="replace")
        elif ext == ".xls":
            df = pd.read_excel(io.BytesIO(contents), engine="xlrd")
        else:
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
    except Exception as e:
        raise HTTPException(422, f"Failed to read file: {str(e)}")

    try:
        parsed = _parse_dataframe(df)
    except ValueError as e:
        raise HTTPException(422, str(e))

    rows = parsed["rows"]
    if not rows:
        raise HTTPException(422, "No valid expense rows found. Check that your file has date, amount, and description columns.")

    # Auto-categorize descriptions
    descriptions = [r["description"] for r in rows]
    categories = categorize_batch(descriptions)

    for i, row in enumerate(rows):
        row["category"] = categories[i]["category"]
        row["confidence"] = categories[i]["confidence"]
        row["id"] = i  # Temp ID for frontend to track edits

    return {
        "total_rows": len(rows),
        "preview": rows[:5],   # First 5 for preview
        "all_rows": rows,       # Full dataset
        "detected_columns": parsed["detected_columns"],
        "filename": filename,
    }


@router.get("/template")
async def download_template():
    """Return a sample Excel template with correct column headers."""
    template_data = {
        "Date": ["2025-01-15", "2025-01-18", "2025-01-22"],
        "Description": ["Swiggy Order", "Uber Ride", "Netflix Subscription"],
        "Amount": [350.00, 120.00, 649.00],
        "Category": ["Food", "Travel", "Entertainment"],
    }
    df = pd.DataFrame(template_data)
    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=expense_template.xlsx"}
    )
