# CryptoProfitMaxxing

End-to-end MLOps pipeline that predicts next-day Bitcoin price direction (UP/DOWN). Built as a team project for an MLOps course — the pipeline is the deliverable, not the model accuracy.

**Team:** Djaferi Denis, Martinez Adrian, Nguyen Jason

## Architecture

```
CoinGecko API
  → scripts/ingest.py              DVC: ingest stage
  → data/raw/btc_usd.csv
  → scripts/featurize.py           DVC: featurize stage
  → data/processed/features.csv
  → train.py --model {logreg,rf,lstm}    MLflow tracking + registry
  → tune.py --model {logreg,rf}         Ray Tune hyperparameter search
  → scripts/check_drift.py              Alibi Detect drift monitoring
  → app.py                              Streamlit dashboard
```

## MLOps Tool Stack (6 tools)

| Tool | Purpose |
|---|---|
| **DVC** | Data versioning — tracks datasets as new crypto data arrives daily |
| **MLflow** | Experiment tracking, model comparison, model registry |
| **GitHub Actions** | CI (lint + test on every push) + automated daily retraining |
| **Ray Tune** | Hyperparameter search across model configurations |
| **Streamlit** | Interactive dashboard for predictions and model monitoring |
| **Alibi Detect** | KS-test data drift detection when market regimes shift |

## Quick Start

```bash
# 1. Environment
conda create -n mlops-project python=3.10 -y
conda activate mlops-project
pip install -r requirements.txt

# 2. Run the pipeline
dvc repro                                # ingest CoinGecko data + compute features

# 3. Train models
python train.py --model logreg           # Logistic Regression
python train.py --model rf               # Random Forest
python train.py --model lstm             # PyTorch LSTM

# 4. Hyperparameter tuning (optional)
python tune.py --model rf --num-samples 20

# 5. Check for data drift
python scripts/check_drift.py

# 6. Inspect experiments
mlflow ui                                # http://localhost:5000

# 7. Launch dashboard
streamlit run app.py                     # http://localhost:8501

# 8. Run tests
pytest tests/ --cov=src --cov-report=term-missing
```

## Repository Layout

```
.
├── app.py                        # Streamlit dashboard
├── train.py                      # MLflow-tracked training (logreg, rf, lstm)
├── tune.py                       # Ray Tune hyperparameter search
├── dvc.yaml                      # DVC pipeline definition
├── params.yaml                   # Pipeline parameters
├── requirements.txt
├── .github/workflows/
│   ├── ci.yml                    # Lint + test on push/PR
│   └── retrain.yml               # Daily automated retraining (cron)
├── src/
│   ├── config.py                 # Paths + constants
│   ├── ingestion/coingecko.py    # CoinGecko API client
│   ├── features/
│   │   ├── indicators.py         # TA indicators (RSI, MACD, SMA, EMA)
│   │   └── labels.py             # Leak-free UP/DOWN target
│   ├── models/
│   │   ├── baseline.py           # Chronological split + sklearn fit/eval
│   │   ├── lstm.py               # PyTorch LSTM with sklearn-compatible wrapper
│   │   └── diagnostics.py        # ROC curves, confusion matrices, feature importance
│   ├── tuning/
│   │   ├── search_spaces.py      # Ray Tune search space definitions
│   │   └── runner.py             # Tune orchestration + MLflow integration
│   ├── drift/
│   │   └── detector.py           # Alibi Detect KS-test drift detector
│   └── mlflow_store.py           # MLflow query helpers for the dashboard
├── scripts/
│   ├── ingest.py                 # CoinGecko → raw CSV
│   ├── featurize.py              # Raw CSV → features CSV
│   └── check_drift.py            # CLI drift check (exit 0/1)
├── tests/                        # 99 pytest tests across 12 files
└── docs/
    └── LosMatadores-CryptoTrend-Pitch.pdf
```

## Models

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | sklearn | L1/L2 regularization, baseline |
| Random Forest | sklearn | Feature importance, best overall |
| LSTM | PyTorch | 1-layer LSTM, sklearn-compatible wrapper with context buffer |

All models use the same 11 technical indicator features, chronological 80/20 train/test split (no shuffle — no future leakage), and log to the same MLflow experiment for side-by-side comparison.

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Push to main, PRs | Ruff lint + pytest with coverage |
| `retrain.yml` | Daily cron (06:00 UTC) + manual | Ingest → featurize → train all models → drift check → commit results |

The retrain workflow commits updated `dvc.lock` and `models/metrics.json` back to main with `[skip ci]` to prevent infinite loops.

## Dashboard

The Streamlit dashboard (`app.py`) provides:

- BTC price chart with SMA/EMA overlays
- Side-by-side model comparison (best run per model from MLflow)
- Next-day UP/DOWN prediction with confidence and disagreement banner
- Metric bar chart, ROC curves, confusion matrices
- Random Forest feature importance
- Ray Tune hyperparameter tuning results
- Alibi Detect data drift status with per-feature p-value chart
- All-runs audit trail expander

## Grading Checklist

- [x] 3+ MLOps tools justified (we use 6)
- [x] End-to-end system: ingest → train → register → serve
- [x] Streamlit dashboard is interactive and reproducible
- [x] 3 models compared in MLflow (LogReg, RF, LSTM)
- [x] GitHub Actions green on main
- [x] Automated retraining via scheduled workflow
- [x] Drift monitoring with Alibi Detect
- [x] README, requirements.txt, tests all present
