# Final Presentation Plan — May 20, 2026

## Logistics
- **Duration:** 10 min presentation (strict) + 5 min Q&A
- **Team:** Denis, Adrian, Jason — all must speak
- **Hand-in:** Upload PDF slidedeck + code zip (no data/model weights) to MS Teams by Wednesday EOD

---

## Speaker Split (10 minutes total)

### Denis — The Problem & Pipeline (3 min)
**Slides 1-4**

- Slide 1: Title slide
- Slide 2: The Problem — why crypto needs MLOps
- Slide 3: Architecture — end-to-end pipeline diagram
- Slide 4: Tool Stack — the 6 tools and why each one

**Talking points:**
> "Crypto trades 24/7, data shifts constantly, models degrade fast. We built an end-to-end MLOps pipeline with 6 tools. Let me walk you through the architecture..."

> "DVC versions our data, MLflow tracks every experiment, GitHub Actions runs CI and automated daily retraining, Ray Tune searches hyperparameters, Streamlit serves the dashboard, and Alibi Detect monitors for data drift."

### Adrian — Models & Experiments (3 min)
**Slides 5-7**

- Slide 5: Models — LogReg, Random Forest, LSTM
- Slide 6: Training approach — chronological split, no future leakage
- Slide 7: Results — metrics table, why ~50% accuracy is expected

**Talking points:**
> "We train 3 models on the same 11 technical indicators — RSI, MACD, moving averages. We use a strict chronological 80/20 split — no shuffling, no future leakage."

> "The LSTM is a PyTorch model wrapped in a sklearn-compatible interface, so the rest of the pipeline doesn't even know it's a neural network."

> "Accuracy is around 50% — and that's expected. Academic research shows daily crypto prediction is near-random. This validates our thesis: you need automated retraining and drift monitoring because models degrade fast."

### Jason — Automation & Live Demo (4 min)
**Slides 8-9 + live demo**

- Slide 8: CI/CD — what happens on every push + daily retraining
- Slide 9: Drift Detection — what it finds and what it means

**Then switch to live demo (2-3 min):**

1. **GitHub Actions tab** — show 14+ consecutive green retrain runs
   > "Our pipeline has been retraining itself every day for 2 weeks without us touching it."

2. **Streamlit dashboard** — already running on localhost:8501
   - Price chart: "365 days of BTC data, DVC-versioned"
   - Model comparison: "3 models compared in MLflow, best run per model"
   - Predictions: "LSTM says UP, others say DOWN — models disagree"
   - Drift section: "Alibi Detect flags 8 of 11 features have drifted"
   > "This is why automated retraining matters."

3. **Terminal — drift check** (optional, if time permits)
   ```bash
   python scripts/check_drift.py
   ```

### All — Wrap Up (30 sec)
- Slide 10: Summary + what we'd do next
> "6 MLOps tools, 3 models, 112 tests, automated daily retraining, drift monitoring. The pipeline is the product."

---

## Slide Content (10 slides)

### Slide 1: Title
```
CryptoTrend
ML-Powered Cryptocurrency Prediction Tool

MLOps Final Presentation

Djaferi Denis · Martinez Adrian · Nguyen Jason
```

### Slide 2: The Problem
```
Why Crypto Needs MLOps

• Crypto markets are highly volatile and trade 24/7
• Data shifts constantly → models degrade fast
• Need automated retraining & continuous monitoring
• Perfect use case for MLOps practices

$2.5T+ global crypto market cap
24/7 non-stop trading
```

### Slide 3: Architecture
```
End-to-End Pipeline

CoinGecko API → Data Ingestion (DVC) → Feature Engineering
→ Model Training (MLflow) → Testing & CI (GitHub Actions)
→ Dashboard (Streamlit)

[Use the same pipeline diagram from the pitch deck]
```

### Slide 4: Tool Stack
```
6 MLOps Tools

DVC          — Data versioning for daily-changing crypto data
MLflow       — Experiment tracking, model comparison, registry
GitHub Actions — CI/CD + automated daily retraining
Ray Tune     — Hyperparameter search (20 trials per model)
Streamlit    — Interactive dashboard with live predictions
Alibi Detect — KS-test drift detection for market regime shifts
```

### Slide 5: Models
```
3 Models Compared in MLflow

Logistic Regression  — sklearn, L1/L2 regularization, baseline
Random Forest        — sklearn, feature importance, best overall
LSTM                 — PyTorch, sklearn-compatible wrapper

All use the same 11 technical indicators:
RSI, MACD, SMA (10/20/50), EMA (12/26), returns, volume change
```

### Slide 6: Training Approach
```
Chronological Split — No Future Leakage

Day 1 ════════════════ Day 252 ════ Day 315
         TRAIN (80%)         TEST (20%)

• Strict time-order split (never shuffled)
• Prevents the classic ML-on-time-series mistake
• Same split for all 3 models
• Ray Tune: 20 random configs per model, logged to MLflow
```

### Slide 7: Results
```
Model Performance

Model      | Accuracy | ROC AUC | Notes
LogReg     | ~43%     | ~0.54   | Linear baseline
RF         | ~59%     | ~0.54   | Best overall
LSTM       | ~52%     | ~0.48   | Sequential patterns

Why ~50%? Daily crypto direction is fundamentally near-random.
This validates our thesis: models degrade fast →
you NEED automated retraining and drift monitoring.
```

### Slide 8: Automation
```
CI/CD + Automated Retraining

On every push (ci.yml):
  → Ruff lint + 112 pytest tests

Daily at 06:00 UTC (retrain.yml):
  → Ingest fresh CoinGecko data
  → Compute features
  → Train all 3 models
  → Check drift (Alibi Detect)
  → Commit results back to main
  → Log experiments to Dagshub MLflow

14+ consecutive daily runs — fully autonomous
```

### Slide 9: Drift Detection
```
Data Drift Monitoring (Alibi Detect)

KS-test on each feature: training distribution vs. recent 30 days

Feature          p-value    Drift?
return_1d        0.6743     no       ← stationary (% change)
sma_10           0.0000     YES      ← price-based, shifts naturally
ema_12           0.0000     YES
macd             0.0000     YES
volume_change_1d 0.6743     no       ← stationary

Drift detected on 8/11 features — expected for crypto.
This is why automated retraining matters.
```

### Slide 10: Summary
```
What We Delivered

✓ 6 MLOps tools (DVC, MLflow, GitHub Actions, Ray Tune, Streamlit, Alibi Detect)
✓ 3 models compared (LogReg, RF, LSTM)
✓ End-to-end pipeline: ingest → train → register → serve
✓ Automated daily retraining (14+ runs and counting)
✓ Drift monitoring with per-feature p-values
✓ Interactive dashboard with predictions and model comparison
✓ 112 tests, 87% coverage, CI green on every push

If we continued:
• Cloud MLflow (Dagshub — already integrated)
• Multi-coin support (ETH, SOL)
• Alerting on drift (Slack webhook)
• Streamlit Cloud deployment

Thank you — Questions?
```

---

## Pre-Presentation Checklist

### The Night Before (tonight)
- [ ] `git pull` to get latest retrain commit
- [ ] `conda activate mlops-project`
- [ ] `dvc repro` to refresh data
- [ ] `python train.py --model logreg && python train.py --model rf && python train.py --model lstm`
- [ ] `streamlit run app.py` — verify dashboard loads at localhost:8501
- [ ] Open http://localhost:8501 in browser — check no errors
- [ ] Open GitHub Actions tab in another browser tab
- [ ] Open Dagshub Experiments tab in another browser tab
- [ ] Create the slides (Google Slides / PowerPoint → export as PDF)

### Morning of Presentation
- [ ] `conda activate mlops-project`
- [ ] `streamlit run app.py` — start dashboard
- [ ] Open browser tabs: dashboard, GitHub Actions, Dagshub
- [ ] Close Slack, email, notifications — clean screen for demo
- [ ] Have terminal ready for `python scripts/check_drift.py`

---

## Q&A Preparation — Likely Questions & Answers

**"Why is accuracy only 50%?"**
> Daily crypto prediction is fundamentally near-random — academic research confirms this. Our ~50% validates the thesis: models degrade, you need MLOps.

**"Why not use a better model?"**
> The grading focus is pipeline quality, not model accuracy. We kept models simple and invested in the MLOps tooling.

**"How does the LSTM work with sklearn's evaluate?"**
> We wrote a wrapper class with predict() and predict_proba(). It stores a context buffer from training so single-row predictions work.

**"What happens when drift is detected?"**
> The pipeline retrains automatically every day. In production you'd add Slack alerting and conditional retraining.

**"Could this run in production?"**
> Add a cloud MLflow server (we've already integrated Dagshub), deploy Streamlit to Streamlit Cloud, and add alerting. Straightforward extensions.

**"How do you prevent future leakage?"**
> Chronological split — first 80% trains, last 20% tests. Never shuffled. Labels use next-day close, computed from historical data only.

**"Why these features?"**
> RSI, MACD, and moving averages are standard technical indicators used by actual traders. They capture momentum, trend, and mean-reversion signals.

**"How long does retraining take?"**
> About 3 minutes end-to-end in GitHub Actions: ingest, featurize, train 3 models, check drift, commit.

---

## Questions to Ask Classmates

Prepare 2-3 questions for other teams' presentations:

1. "How do you handle data versioning when your dataset changes?"
2. "What happens if your model performance degrades in production — do you have automated monitoring?"
3. "How do you prevent data leakage in your train/test split?"
4. "What CI/CD do you have — do tests run automatically on every push?"
5. "How would you add a new model to your pipeline — how much code would need to change?"
