# Smart Expense Tracker — AI Backend

Python FastAPI backend providing AI-powered expense predictions, anomaly detection, and smart insights.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root info |
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Next-month spending prediction |
| POST | `/api/anomalies` | Anomaly detection |
| POST | `/api/insights` | Budget suggestions & habit analysis |
| GET | `/docs` | Swagger UI |

## Local Development

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env      # fill in your values
uvicorn main:app --reload --port 8000
```

## Environment Variables

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service role key |
| `ALLOWED_ORIGINS` | Comma-separated allowed CORS origins |
