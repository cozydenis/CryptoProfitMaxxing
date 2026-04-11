# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**CryptoProfitMaxxing** — an end-to-end MLOps pipeline that predicts short-term crypto price trends (binary up/down classification). Team project for an MLOps course; grading rewards pipeline quality, not model accuracy.

Full plan: `docs/project-plan-crypto-predictor.md`.

**Key constraint:** the MLOps pipeline is the deliverable. Keep models simple (Logistic Regression, Random Forest, optionally LSTM). Do not over-invest in model tuning.

## Stack

| Concern | Tool |
|---|---|
| Language / runtime | Python 3.10 (conda env `mlops-project`) |
| Data source | CoinGecko API (free, no auth) — Binance public API as fallback |
| Data versioning | DVC (remote: Google Drive or local) |
| Experiment tracking + model registry | MLflow (local tracking server) |
| Hyperparameter tuning | Ray Tune |
| ML libs | scikit-learn (baseline), PyTorch (LSTM stretch) |
| Feature engineering | pandas, TA indicators (RSI, MACD, MAs) |
| Dashboard / deployment | Streamlit |
| CI/CD | GitHub Actions |
| Testing | pytest |
| Drift monitoring (bonus) | Alibi Detect |

## Architecture

```
CoinGecko API
   → data ingestion (scripts/ingest.py)
   → DVC-versioned CSVs (data/)
   → feature engineering (src/features/)
   → training (src/models/) → MLflow tracking + Ray Tune
   → model registry (MLflow)
   → Streamlit app (app.py) serves predictions
   → GitHub Actions runs pytest + data-schema validation on push
```

## Expected Repo Layout

```
.
├── app.py                    # Streamlit dashboard
├── train.py                  # MLflow-tracked training entry point
├── requirements.txt
├── dvc.yaml / .dvc/          # DVC pipeline + config
├── data/                     # DVC-tracked (NOT in git)
├── src/
│   ├── ingestion/            # API clients, data fetchers
│   ├── features/             # Technical indicators, transforms
│   ├── models/               # Baseline + LSTM
│   └── evaluation/
├── tests/                    # pytest — unit + integration
├── .github/workflows/        # CI: lint, test, validate
└── docs/
```

Create directories/files only when you start implementing the relevant slice — do not scaffold empty modules ahead of time.

## Team Split (for context when reviewing PRs)

- **Person A** — Data pipeline: ingestion, feature engineering, DVC, data tests.
- **Person B** — Modeling: sklearn baseline + PyTorch LSTM, MLflow integration, Ray Tune.
- **Person C** — Deployment: Streamlit dashboard, GitHub Actions, integration tests, optional Alibi Detect.

## Timeline (absolute dates, 2026 semester)

| Week | Milestone |
|---|---|
| Apr 15 | Data pipeline + DVC + baseline model |
| Apr 22 | **Pitch presentation** — 2+ models in MLflow |
| Apr 29 | Streamlit functional + CI running |
| May 6  | Drift detection, polish |
| May 20 | **Final presentation** |

Flag any suggestion that jeopardizes these dates.

## Working Rules

- **Planning first.** For any non-trivial change, use the `planner` agent or `/plan` skill before coding.
- **TDD.** Use `/tdd` or `tdd-guide` agent. Write pytest tests first, 80%+ coverage target.
- **Code review.** After writing code, invoke `python-reviewer` agent (or `/python-review`).
- **Immutability and small files.** Follow `~/.claude/rules/common/coding-style.md` — pure functions where possible, files <400 lines.
- **No hardcoded secrets.** CoinGecko needs no auth; anything else goes in `.env` (gitignored).
- **DVC over git for data.** Never commit CSVs directly — use `dvc add`.
- **MLflow for every training run.** No untracked experiments.

## Reference: Everything Claude Code (ECC)

Implementation leverages ECC at `/Users/denis/Strata/everything-claude-code/`. Relevant assets:

**Skills** (invoke via `Skill` tool or `/<name>`):
- `python-patterns` — Pythonic idioms, type hints, PEP 8
- `python-testing` — pytest, fixtures, coverage, TDD
- `tdd-workflow` / `/tdd` — red-green-refactor enforcement
- `backend-patterns` — API design if we expose a prediction endpoint
- `e2e-testing` — Playwright for Streamlit UI smoke tests
- `verification-loop` — pre-commit verification gate
- `plan` / `multi-plan` — implementation planning
- `code-review` / `/python-review` — quality review
- `docs` — Context7 lookups for DVC/MLflow/Ray Tune/Streamlit APIs

**Agents**:
- `planner`, `tdd-guide`, `python-reviewer`, `code-reviewer`, `security-reviewer`, `architect`, `doc-updater`

**Mandatory research step** (from `~/.claude/rules/common/development-workflow.md`): before writing new utility code, check GitHub (`gh search code`) and Context7 for existing implementations. Prefer battle-tested libraries (e.g., `ta` or `ta-lib` for indicators, `ccxt` for exchange APIs) over hand-rolled code.

## Quick Start

```bash
conda create -n mlops-project python=3.10
conda activate mlops-project
pip install -r requirements.txt

dvc pull                       # fetch versioned data
python train.py --model baseline
streamlit run app.py
pytest tests/ --cov=src --cov-report=term-missing
```

## Grading Checklist (keep visible)

- [ ] 3+ MLOps tools justified in pitch (we use 5–6)
- [ ] End-to-end system: ingest → train → register → serve
- [ ] Streamlit dashboard is interactive and reproducible
- [ ] 2+ models compared in MLflow
- [ ] GitHub Actions green on main
- [ ] README, requirements.txt, tests all present
