"""Central place for paths and constants so no script hard-codes machine-specific paths."""
from pathlib import Path

# Project root = the folder that contains `src/`. Works on any machine / OS.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PATH = PROCESSED_DIR / "churn_clean.csv"
CLEANING_LOG_PATH = PROCESSED_DIR / "cleaning_log.md"

ID_COL = "customerID"
TARGET_COL = "Churn"

# Modelling constants. One seed everywhere so every split, CV fold and model is reproducible.
RANDOM_SEED = 42
TEST_SIZE = 0.20      # stratified hold-out set, touched only for final evaluation
CV_FOLDS = 5

# Model artefacts and reports
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_FIGURES_DIR = PROJECT_ROOT / "images" / "model"

# Risk segmentation. These are practical project assumptions, not universal truths; src.predict checks
# that the probability tiers are honest (observed churn rate inside each tier's band) before using them.
RISK_LOW_THRESHOLD = 0.30     # probability < 0.30  -> Low
RISK_HIGH_THRESHOLD = 0.70    # probability > 0.70  -> High; in between -> Medium
HIGH_VALUE_QUANTILE = 0.75    # "High value" = monthly charges in the top quartile of all customers (revenue is the only value signal in the data)
PREDICTIONS_PATH = PROJECT_ROOT / "output" / "customer_churn_predictions.csv"
