"""Paths, column schema and experiment constants shared by training and the app."""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
RAW_DATA_PATH: Final = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
TEST_DATA_PATH: Final = PROJECT_ROOT / "test_data.csv"
MODEL_DIR: Final = PROJECT_ROOT / "model"
METRICS_PATH: Final = MODEL_DIR / "metrics.json"

DATASET_URL: Final = (
    "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
)
DATASET_ARCHIVE_MEMBER: Final = "online_shoppers_intention.csv"

TARGET_COLUMN: Final = "Revenue"

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
)

# OperatingSystems, Browser, Region and TrafficType are integer-encoded *labels*.
# Their numeric order carries no meaning, so they are one-hot encoded with the rest.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
)

FEATURE_COLUMNS: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE: Final = 42
TEST_SIZE: Final = 0.20
CV_FOLDS: Final = 5
