"""
Anomaly Detector — uses Isolation Forest to flag unusual spending patterns.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")


CATEGORY_MAP = {
    "Food": 0, "Travel": 1, "Bills": 2, "Entertainment": 3,
    "Health": 4, "Shopping": 5, "Education": 6, "Other": 7
}


class AnomalyDetector:
    def __init__(self, contamination: float = 0.08):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.is_fitted = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix for anomaly detection."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["category_encoded"] = df["category"].map(CATEGORY_MAP).fillna(7)

        # Rolling statistics
        df = df.sort_values("date")
        df["rolling_mean"] = df["amount"].rolling(window=5, min_periods=1).mean()
        df["rolling_std"] = df["amount"].rolling(window=5, min_periods=1).std().fillna(0)

        features = df[["amount", "category_encoded", "day_of_week",
                       "day_of_month", "month", "rolling_mean", "rolling_std"]].values
        return features, df

    def detect(self, expenses: List[dict]) -> List[dict]:
        """Detect anomalies in expense list."""
        if len(expenses) < 5:
            # Not enough data — use simple statistical approach
            return self._simple_detection(expenses)

        df = pd.DataFrame(expenses)
        features, df_processed = self._build_features(df)

        # Fit and predict
        self.model.fit(features)
        predictions = self.model.predict(features)
        scores = self.model.decision_function(features)

        # Normalize scores to 0-1 (lower = more anomalous)
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            normalized = (scores - score_min) / (score_max - score_min)
        else:
            normalized = np.ones(len(scores)) * 0.5

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            is_anomaly = predictions[i] == -1
            anomaly_score = float(1 - normalized[i])  # higher = more anomalous

            severity = "normal"
            reason = None
            if is_anomaly:
                if anomaly_score > 0.85:
                    severity = "high"
                    reason = f"Extremely unusual amount (${row['amount']:.2f}) for {row['category']}"
                elif anomaly_score > 0.65:
                    severity = "medium"
                    reason = f"Amount is significantly above your typical {row['category']} spending"
                else:
                    severity = "low"
                    reason = f"Slightly unusual spending pattern detected"

            results.append({
                "expense_id": row.get("id"),
                "date": row["date"] if isinstance(row["date"], str) else str(row["date"]),
                "amount": float(row["amount"]),
                "category": row["category"],
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": round(anomaly_score, 3),
                "severity": severity,
                "reason": reason
            })

        return results

    def _simple_detection(self, expenses: List[dict]) -> List[dict]:
        """Simple Z-score based detection for small datasets."""
        amounts = [e["amount"] for e in expenses]
        mean = np.mean(amounts)
        std = np.std(amounts) if np.std(amounts) > 0 else 1

        results = []
        for e in expenses:
            z_score = abs(e["amount"] - mean) / std
            is_anomaly = z_score > 2.0
            anomaly_score = min(1.0, z_score / 3.0)

            severity = "normal"
            reason = None
            if is_anomaly:
                severity = "high" if z_score > 3 else "medium"
                reason = f"Amount ${e['amount']:.2f} is {z_score:.1f} standard deviations from your average"

            results.append({
                "expense_id": e.get("id"),
                "date": str(e["date"]),
                "amount": float(e["amount"]),
                "category": e["category"],
                "is_anomaly": is_anomaly,
                "anomaly_score": round(anomaly_score, 3),
                "severity": severity,
                "reason": reason
            })

        return results
