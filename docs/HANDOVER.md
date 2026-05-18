# CryptoProfitMaxxing — Project Handover

**Team:** Djaferi Denis, Martinez Adrian, Nguyen Jason
**Course:** MLOps
**Final presentation:** May 20, 2026

---

## 1. What We Built

An end-to-end MLOps pipeline that predicts next-day Bitcoin price direction (UP/DOWN). The pipeline runs autonomously — fresh data is ingested, models are retrained, and drift is monitored every day without human intervention.

### By the Numbers

| Metric | Value |
|---|---|
| Python source code | ~3,700 lines across 25 files |
| Tests | 112 across 12 test files |
| Test coverage | 87% on `src/` |
| MLOps tools | 6 (DVC, MLflow, GitHub Actions, Ray Tune, Streamlit, Alibi Detect) |
| Models | 3 (Logistic Regression, Random Forest, PyTorch LSTM) |
| Automated retrain runs | 13 consecutive daily runs (and counting) |
| CI status | Green on every push |

---

## 2. Tool Stack — Why Each Tool

| Tool | What It Does | Why We Chose It |
|---|---|---|
| **DVC** | Versions datasets, defines the ingest → featurize pipeline | Crypto data changes daily; need to track which dataset trained which model |
| **MLflow** | Experiment tracking, model comparison, model registry | Compare 3 models side-by-side, log every training run's params + metrics |
| **GitHub Actions** | CI (ruff + pytest on every push) + daily automated retraining | Runs tests on every push, retrains models daily without human intervention |
| **Ray Tune** | Hyperparameter search (20 trials per model) | Automates search over regularization, tree depth, etc. |
| **Streamlit** | Interactive dashboard with predictions, charts, drift status | Great for demo — shows the full pipeline output in one page |
| **Alibi Detect** | KS-test drift detection on feature distributions | Detects when market regime shifts and models may need attention |

---

## 3. Architecture

```
CoinGecko API (free, no auth)
    │
    ▼
scripts/ingest.py ──► data/raw/btc_usd.csv (DVC-tracked)
    │
    ▼
scripts/featurize.py ──► data/processed/features.csv
    │                     (RSI, MACD, SMA, EMA, returns — 11 features)
    │
    ├──► train.py --model logreg    ──► MLflow experiment + model registry
    ├──► train.py --model rf        ──► MLflow experiment + model registry
    ├──► train.py --model lstm      ──► MLflow experiment + model registry
    ├──► tune.py --model {logreg,rf} ──► Ray Tune → MLflow (tagged runs)
    │
    ▼
scripts/check_drift.py ──► Alibi Detect KS-test (per-feature p-values)
    │
    ▼
app.py ──► Streamlit dashboard (http://localhost:8501)
           - Price chart with SMA/EMA overlays
           - 3-model comparison (best run per model from MLflow)
           - Next-day UP/DOWN predictions with disagreement banner
           - Metric bar chart, ROC curves, confusion matrices
           - RF feature importance
           - Ray Tune trial counts
           - Data drift status with p-value bar chart
```

### Automated Pipeline (GitHub Actions)

```
.github/workflows/ci.yml        ──► Runs on every push: ruff lint + pytest
.github/workflows/retrain.yml   ──► Daily at 06:00 UTC:
                                     dvc repro → train 3 models → drift check
                                     → commit dvc.lock + metrics.json back to main
```

---

## 4. How to Run Locally

```bash
# Setup (one time)
conda create -n mlops-project python=3.10 -y
conda activate mlops-project
pip install -r requirements.txt

# Run the pipeline
dvc repro                              # ingest + featurize
python train.py --model logreg         # train Logistic Regression
python train.py --model rf             # train Random Forest
python train.py --model lstm           # train LSTM

# Optional: hyperparameter tuning
python tune.py --model rf --num-samples 20
python tune.py --model logreg --num-samples 20

# Check drift
python scripts/check_drift.py

# View experiments
mlflow ui --port 5001                  # http://localhost:5001

# Launch dashboard
streamlit run app.py                   # http://localhost:8501

# Run tests
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 5. Repository Structure

```
.
├── app.py                            # Streamlit dashboard
├── train.py                          # MLflow-tracked training (logreg, rf, lstm)
├── tune.py                           # Ray Tune hyperparameter search
├── dvc.yaml / dvc.lock               # DVC pipeline definition + lock
├── params.yaml                       # All pipeline parameters
├── requirements.txt                  # Python dependencies
├── .github/workflows/
│   ├── ci.yml                        # CI: lint + test
│   └── retrain.yml                   # Daily automated retraining
├── src/
│   ├── config.py                     # Paths, constants, feature columns
│   ├── ingestion/coingecko.py        # CoinGecko API client
│   ├── features/
│   │   ├── indicators.py             # RSI, MACD, SMA, EMA, returns
│   │   └── labels.py                 # Leak-free UP/DOWN target
│   ├── models/
│   │   ├── baseline.py               # Chronological split, sklearn fit/eval
│   │   ├── lstm.py                   # PyTorch LSTM with sklearn wrapper
│   │   └── diagnostics.py            # ROC, confusion matrix, feature importance
│   ├── tuning/
│   │   ├── search_spaces.py          # Ray Tune search spaces per model
│   │   └── runner.py                 # Tune orchestration + MLflow logging
│   ├── drift/
│   │   └── detector.py               # Alibi Detect KS-test drift detector
│   └── mlflow_store.py               # MLflow query helpers for dashboard
├── scripts/
│   ├── ingest.py                     # CoinGecko → raw CSV
│   ├── featurize.py                  # Raw CSV → features CSV
│   └── check_drift.py               # CLI drift check (exit 0/1)
├── tests/                            # 112 tests across 12 files
├── models/
│   └── metrics.json                  # Latest model metrics (committed by CI)
└── docs/
    ├── LosMatadores-CryptoTrend-Pitch.pdf
    └── HANDOVER.md                   # This file
```

---

## 6. Models — Performance & Why It's OK

| Model | Accuracy | ROC AUC | Notes |
|---|---|---|---|
| Logistic Regression | ~43% | ~0.54 | Baseline, linear decision boundary |
| Random Forest | ~51% | ~0.49 | Best overall, feature importance available |
| LSTM | ~52% | ~0.48 | PyTorch, captures sequential patterns |

**Why accuracy is ~50%:** Short-term crypto price direction is essentially random — academic research consistently shows daily UP/DOWN prediction on BTC is near coin-flip territory. This validates our thesis: *models degrade fast in crypto, which is exactly why you need MLOps tooling (drift detection, automated retraining, experiment tracking).*

The pipeline is the deliverable, not the model accuracy.

---

## 7. Key Design Decisions

| Decision | Why |
|---|---|
| **Chronological split (no shuffle)** | Standard train/test split leaks future data. We split 80/20 in time order. |
| **sklearn-compatible LSTM wrapper** | The LSTM exposes `predict()` / `predict_proba()` so it plugs into `evaluate()` and the dashboard with zero changes. Uses a context buffer for single-row predictions. |
| **Ray Tune trials as top-level MLflow runs** | Each trial is a standalone MLflow run tagged with `tuning_source=ray-tune`. The dashboard's `best_run_per_model()` picks tuned runs automatically if they're best. |
| **KS-test for drift** | Non-parametric, no model training needed, gives per-feature p-values. Perfect for demo. |
| **`[skip ci]` in retrain commits** | Prevents the retrain → push → CI → retrain infinite loop. |
| **Drift always detected** | Expected and correct — price-based features (SMA, EMA, MACD) naturally shift as BTC price changes. Great demo talking point. |

---

## 8. Demo Script (Suggested Flow for May 20)

### Part 1: The Problem (1 min)
- "Crypto is volatile, trades 24/7, models degrade fast"
- Show pitch slide 2

### Part 2: Architecture & Tools (2 min)
- Walk through the 6-tool stack slide
- Emphasize: pipeline is the deliverable

### Part 3: Live Demo (5-7 min)

1. **Show GitHub Actions** — retrain workflow has run 13+ consecutive days
   - Open Actions tab, show green runs, show the commit history
   - "Our pipeline retrains automatically every day"

2. **Show the dashboard** (`streamlit run app.py`)
   - Price chart: "365 days of BTC from CoinGecko, DVC-versioned"
   - Model comparison: "3 models logged to MLflow, best run per model"
   - Predictions: "LSTM says UP, RF and LogReg say DOWN — models disagree"
   - Drift section: "Alibi Detect flags that the market has shifted"
   - "This is why you need automated retraining"

3. **Show MLflow UI** (`mlflow ui --port 5001`)
   - Experiment list, 45+ runs, filter by model
   - Show tuned runs (Ray Tune tag)
   - Compare metrics across models

4. **Run drift check live** (terminal)
   ```bash
   python scripts/check_drift.py
   ```
   - Show per-feature p-values, drift detected

5. **Show CI** — open ci.yml run, show ruff + pytest passing

### Part 4: Wrap Up (1 min)
- "6 MLOps tools, 3 models, automated daily retraining, drift monitoring"
- "112 tests, 87% coverage, CI green on every push"
- Questions?

---

## 9. Known Limitations

| Limitation | Mitigation |
|---|---|
| MLflow data from CI is ephemeral (not visible in local dashboard) | Demo locally; show CI logs for automation proof |
| 365-day history cap (CoinGecko free tier) | Sufficient for demo; could use Binance API for more |
| BTC only (no ETH/altcoins) | Architecture supports adding coins via params.yaml |
| No cloud deployment of dashboard | Run locally for demo; Streamlit Cloud is a 5-minute deploy if needed |
| Model accuracy ~50% | Expected for daily crypto prediction; validates the MLOps thesis |

---

## 10. If Someone Continues This Project

Potential improvements, in order of impact:

1. **Dagshub integration** — free hosted MLflow so CI runs are visible in the dashboard
2. **Multi-coin support** — add ETH, SOL via params.yaml (ingestion already parameterized)
3. **Streamlit Cloud deployment** — `streamlit deploy` for a public demo URL
4. **More features** — order book depth, social sentiment, on-chain metrics
5. **Better models** — XGBoost, transformer-based, ensemble methods
6. **Alerting** — Slack webhook on drift detection or model degradation

---

*Generated: May 18, 2026*
*Repository: https://github.com/cozydenis/CryptoProfitMaxxing*
