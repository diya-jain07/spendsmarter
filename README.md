# BUDGET_RISK_PREDICTOR

> ML system that predicts per-category overspending risk before the month begins.

```
sys@risk:~$ predict --categories all --horizon next_month
```

## Architecture

```
[USER INPUT: spend + budget]
        ↓
[FEATURE VECTOR dim=10]
        ↓
[NEURAL NET: Linear(10→8) + ReLU → Linear(8→5) + Sigmoid]
        ↓
[OUTPUT: 5 probabilities, one per category]
```

## Stack

- **Frontend**: React + Vite, zero UI libraries, terminal aesthetic
- **Backend**: FastAPI + PyTorch neural network
- **Model**: Binary classifier trained on synthetic behavioral data

---

## Quick Start

### Frontend only (browser-side inference)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

> The frontend runs the model directly in-browser via a JS port of the neural net.
> No backend required for basic usage.

### Full stack (with FastAPI backend)

**Terminal 1 — Backend:**
```bash
cd backend
pip install -r requirements.txt
python run.py
# Trains model on first run (~5 seconds)
# API: http://localhost:8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### API Endpoints

```
GET  /api/health    — model status
POST /api/predict   — run inference
POST /api/train     — retrain on fresh synthetic data
```

**Example request:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "last_month_spend": [480, 1200, 140, 320, 180],
    "next_budget":      [420, 1200, 130, 200, 150]
  }'
```

**Example response:**
```json
{
  "predictions": {
    "groceries":     0.6821,
    "rent":          0.0412,
    "utilities":     0.2955,
    "entertainment": 0.8134,
    "misc":          0.5270
  },
  "highest_risk": { "category": "entertainment", "probability": 0.8134 },
  "safest":        { "category": "rent",         "probability": 0.0412 }
}
```

---

## Project Structure

```
budget-risk-predictor/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx         # Boot sequence + status
│   │   │   ├── InputPanel.jsx     # Spend/budget data entry
│   │   │   ├── RiskDashboard.jsx  # Predictions + visualizations
│   │   │   └── DataTicker.jsx     # Scrolling system stats
│   │   ├── utils/
│   │   │   └── model.js           # In-browser neural net inference
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css              # Terminal dark theme
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── model.py               # PyTorch model + training + inference
│   │   └── main.py                # FastAPI routes
│   ├── run.py
│   └── requirements.txt
│
└── README.md
```

---

## Model Details

| Parameter       | Value                        |
|----------------|------------------------------|
| Architecture    | MLP: 10 → 8 → 5              |
| Activation      | ReLU (hidden), Sigmoid (out) |
| Loss function   | Binary Cross-Entropy          |
| Optimizer       | Adam (lr=0.001)               |
| Training data   | 10,000 synthetic users        |
| Epochs          | 200                           |
| Input features  | [spend×5, budget×5]          |
| Output          | P(overspend) per category    |

### Behavioral patterns in synthetic data

- **Groceries** — slight upward drift, moderate variance
- **Rent** — near-fixed, very low overspend rate
- **Utilities** — seasonal variance ±22%
- **Entertainment** — highest overspend tendency
- **Misc** — high volatility

---

## Extending the System

- Add 6–12 months of history as features (time-series signals)
- Swap MLP for LSTM/Transformer to capture temporal trends
- Add income, pay cycle, subscription flags as features
- Retrain periodically on real user data for personalization
