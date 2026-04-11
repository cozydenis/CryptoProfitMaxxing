# MLOps Project Plan: Crypto Price Trend Predictor

## Overview

**Team size:** 3
**Goal:** Build an end-to-end operationalized ML system that predicts short-term cryptocurrency price trends (up/down), with a focus on the MLOps pipeline rather than model accuracy.

**Why this project works for grading:**
- Crypto data is free, real-time, and constantly shifting — a natural fit for showcasing data versioning, drift detection, and retraining
- Visually impressive demo (charts, live predictions, dashboards)
- Easy to justify tool choices because every MLOps concept maps cleanly to a real need
- Model can be simple (the pipeline is what matters)

---

## Architecture Overview

```
[CoinGecko / Binance API]
        │
        ▼
[Data Ingestion Script] ──► [DVC - Data Versioning]
        │
        ▼
[Feature Engineering]
  (technical indicators:
   RSI, MACD, moving averages)
        │
        ▼
[Model Training] ◄──── [Ray Tune - Hyperparameter Tuning]
        │
        ├──► [MLflow - Experiment Tracking & Model Registry]
        │
        ▼
[Model Evaluation & Testing] ◄── [PyTest + GitHub Actions CI/CD]
        │
        ▼
[Streamlit Dashboard]
  - Live predictions
  - Historical accuracy
  - Model comparison
  - (Optional) Alibi Detect drift monitoring
```

---

## Tools (minimum 3 required, we use 5-6)

| Tool | Purpose | Why this tool? |
|------|---------|---------------|
| **DVC** | Data versioning | Crypto data changes daily; need to track which dataset trained which model |
| **MLflow** | Experiment tracking + model registry | Compare LSTM vs XGBoost vs Random Forest, log metrics, register best model |
| **GitHub Actions** | CI/CD pipeline + automated testing | Run tests on every push, lint code, validate data schema |
| **Streamlit** | Web UI / deployment | Interactive dashboard for predictions — great for demo |
| **Ray Tune** | Hyperparameter tuning | Automate search over learning rates, architectures, etc. |
| **Alibi Detect** (bonus) | Data drift monitoring | Detect when crypto market regime shifts and model may need retraining |

---

## Data

- **Source:** CoinGecko API (free, no auth needed) or Binance public API
- **Assets:** Bitcoin (BTC), Ethereum (ETH), optionally 1-2 more
- **Features:** OHLCV (Open, High, Low, Close, Volume) + derived technical indicators
- **Target:** Binary classification — will price go UP or DOWN in the next 24h?
- **History:** 1-2 years of daily data is plenty
- **Storage:** CSV files versioned with DVC, remote storage on Google Drive or local

---

## Model

Keep it simple — remember, model performance is explicitly **not** the grading focus.

**Baseline:** Logistic Regression or Random Forest on technical indicators
**Stretch:** LSTM or simple Transformer on time series sequences

The key is to have **at least 2 models** tracked in MLflow so you can show experiment comparison in the demo.

---

## Team Work Split (3 people)

### Person A: Data Pipeline
- Write data ingestion script (CoinGecko API → CSV)
- Feature engineering (RSI, MACD, moving averages, etc.)
- Set up DVC for data versioning
- Write data validation tests

### Person B: Model Training & Tracking
- Implement baseline model (sklearn) + LSTM model (PyTorch)
- Set up MLflow tracking server (local)
- Integrate Ray Tune for hyperparameter search
- Log experiments, register best model in MLflow model registry

### Person C: Deployment & CI/CD
- Build Streamlit dashboard (prediction display, charts, model comparison)
- Set up GitHub Actions workflows (linting, tests, model validation)
- (Optional) Integrate Alibi Detect for drift monitoring
- Write integration tests

---

## Timeline

| Week | Milestone |
|------|-----------|
| **8** (Apr 8) | Form team, agree on project, initial repo setup |
| **9** (Apr 15) | Data pipeline working, DVC set up, first baseline model trained |
| **10** (Apr 22) | **PITCH PRESENTATION** — pipeline demo-ready, 2+ models in MLflow |
| **11** (Apr 29) | Streamlit dashboard functional, GitHub Actions CI running |
| **12** (May 6) | Polish: drift detection, improved UI, edge cases |
| **13** (May 13) | Final testing, rehearse presentation |
| **14** (May 20) | **FINAL PRESENTATION** — full demo of end-to-end system |

---

## Pitch Structure (5 min, Week 10)

1. **Problem** (30s): Crypto markets are volatile — can we predict short-term trends and operationalize the system?
2. **Architecture** (1.5min): Show the system diagram, explain data flow
3. **Tool choices** (1.5min): List each tool and *why* it's the right choice for this project
4. **Current progress** (1min): Quick demo of what's working
5. **Next steps** (30s): What remains for the final presentation

---

## Final Presentation Structure (10 min, Week 14)

1. **Motivation & problem** (1min)
2. **System architecture** (2min)
3. **Live demo** (4min): Walk through the full pipeline — ingest data, show DVC versioning, train model, show MLflow dashboard, show Streamlit predictions
4. **Lessons learned** (2min): What worked, what didn't, what you'd do differently
5. **Q&A** (1min)

---

## Grading Checklist

- [ ] At least 3 appropriately selected tools → we have 5-6
- [ ] Complete system (end-to-end) → data ingestion → training → deployment
- [ ] Usable system → Streamlit dashboard anyone can interact with
- [ ] Convincing pitch → clear problem statement, justified tool choices
- [ ] Final demo → live walkthrough of the full pipeline
- [ ] Code quality → clean repo, README, requirements.txt, tests

---

## Quick Start Commands

```bash
# Clone repo
git clone <your-repo-url>
cd crypto-trend-predictor

# Set up environment
conda create -n mlops-project python=3.10
conda activate mlops-project
pip install -r requirements.txt

# Pull data with DVC
dvc pull

# Train model with MLflow tracking
python train.py --model baseline

# Launch Streamlit dashboard
streamlit run app.py

# Run tests
pytest tests/
```

---

## Useful Links

- CoinGecko API docs: https://www.coingecko.com/en/api
- DVC docs: https://dvc.org/doc
- MLflow docs: https://mlflow.org/docs/latest/index.html
- Streamlit docs: https://docs.streamlit.io
- Ray Tune docs: https://docs.ray.io/en/latest/tune/index.html
- Alibi Detect docs: https://docs.seldon.io/projects/alibi-detect/
