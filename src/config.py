"""Project-wide constants and path helpers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"
DEFAULT_EXPERIMENT_NAME = "crypto-baseline"
DEFAULT_REGISTERED_MODEL_NAME = "crypto_trend_baseline"

PRICE_COLUMNS = ["timestamp", "close", "volume"]
FEATURE_COLUMNS = [
    "return_1d",
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volume_change_1d",
]
TARGET_COLUMN = "target"
