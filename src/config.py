"""Central configuration: paths, dataset location and default hyperparameters."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
MODEL_PATH = MODELS_DIR / "recommender.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"

DATASET_NAME = "ml-latest-small"
MOVIELENS_URL = f"https://files.grouplens.org/datasets/movielens/{DATASET_NAME}.zip"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Matrix factorization defaults
MF_PARAMS = dict(n_factors=50, n_epochs=40, lr=0.01, reg=0.05, batch_size=256)

# EASE (implicit-feedback ranker) defaults
EASE_PARAMS = dict(lambda_=300.0, min_item_ratings=5, like_threshold=0.0)

# Hybrid weights (overwritten by tuning in train.py)
# alpha: collaborative (EASE) vs content-based;  beta: popularity prior
HYBRID_ALPHA = 0.9
HYBRID_BETA = 0.0

# Evaluation
TOP_K = 10
RELEVANCE_THRESHOLD = 4.0
# Ignore movies with fewer ratings than this when ranking (evaluation + app default)
MIN_RATINGS = 10
