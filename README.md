# CryptoProfitMaxxing

End-to-end MLOps pipeline that predicts next-day Bitcoin price direction (UP/DOWN). Built as a Week 9 slice of an MLOps course project — the pipeline is the deliverable, not the model accuracy.

## Architecture

```
CoinGecko API
  → scripts/ingest.py         (DVC: ingest stage)
  → data/raw/btc_usd.csv
  → scripts/featurize.py      (DVC: featurize stage)
  → data/processed/features.csv
  → train.py                  (MLflow tracking + registry)
  → models/baseline.pkl
  → app.py                    (Streamlit dashboard)
```

## Stack

| Concern | Tool |
|---|---|
| Data source | CoinGecko (free, no auth) via `pycoingecko` |
| Data versioning | DVC (local remote) |
| Features | `pandas-ta` — RSI, MACD, SMA, EMA, returns |
| Model | scikit-learn — Logistic Regression / Random Forest |
| Experiment tracking + registry | MLflow (local file store) |
| Dashboard | Streamlit + Plotly |
| Tests | pytest + pytest-cov |

## Quick Start

```bash
# 1. Environment
conda create -n mlops-project python=3.10 -y
conda activate mlops-project
pip install -r requirements.txt

# 2. Run the pipeline
dvc repro                                # ingests + featurizes (DVC-tracked)
python train.py --model logreg           # trains, logs to MLflow, registers model
python train.py --model rf               # second model for comparison

# 3. Inspect experiments
mlflow ui                                # http://localhost:5000

# 4. Launch dashboard
streamlit run app.py                     # http://localhost:8501

# 5. Tests
pytest tests/ --cov=src --cov-report=term-missing
```

## Repository Layout

```
.
├── app.py                    # Streamlit dashboard
├── train.py                  # MLflow-tracked training entrypoint
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # Pipeline parameters
├── requirements.txt
├── src/
│   ├── config.py             # paths + constants
│   ├── ingestion/coingecko.py
│   ├── features/
│   │   ├── indicators.py     # TA indicators (pandas-ta)
│   │   └── labels.py         # leak-free UP/DOWN target
│   └── models/baseline.py    # chronological split + fit/eval
├── scripts/
│   ├── ingest.py             # CoinGecko → raw CSV
│   └── featurize.py          # raw CSV → features CSV
├── tests/
│   ├── conftest.py           # synthetic OHLCV fixtures
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_labels.py
│   └── test_baseline.py
└── docs/
    └── project-plan-crypto-predictor.md
```

## Week 9 Scope

Delivered:
- [x] CoinGecko BTC ingestion, DVC-versioned
- [x] Technical indicators + leak-free target labeling
- [x] Logistic Regression + Random Forest baselines
- [x] MLflow experiment tracking + model registry
- [x] Streamlit dashboard skeleton (predictions + price chart)
- [x] pytest coverage on critical paths

Known limitations:
- 365 days of history (CoinGecko free-tier cap)
- Local DVC remote (Google Drive migration in Week 10)
- BTC only (ETH in Week 10)
- No CI (Person C, Week 11)
- No drift monitoring (stretch goal, Week 12)
- No hyperparameter tuning (Ray Tune, Week 12)
