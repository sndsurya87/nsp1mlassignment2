"""Dataset acquisition, schema validation and the train/test split.

Run ``python -m shopper_intent.data`` to download the raw data (if absent) and
write the hold-out set to ``test_data.csv``.
"""

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shopper_intent.config import (
    CATEGORICAL_FEATURES,
    DATASET_ARCHIVE_MEMBER,
    DATASET_URL,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_SECONDS = 60
_TARGET_LABELS = {
    "true": 1,
    "1": 1,
    "1.0": 1,
    "false": 0,
    "0": 0,
    "0.0": 0,
}


class SchemaError(ValueError):
    """Raised when a data frame cannot be interpreted using the expected schema."""


def download_raw_dataset(destination: Path = RAW_DATA_PATH) -> Path:
    """Fetch the UCI archive and store its CSV at ``destination``; no-op if already present."""
    if destination.exists():
        return destination

    logger.info("Downloading dataset from %s", DATASET_URL)
    with urllib.request.urlopen(DATASET_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        archive = response.read()

    # Only the named member is read, so a hostile archive path can never be written to disk.
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        content = zipped.read(DATASET_ARCHIVE_MEMBER)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def load_raw_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and drop exact duplicates.

    Duplicates must go *before* splitting: a row present on both sides of the split
    would leak into the test set and inflate every metric, KNN's most of all.
    """
    return pd.read_csv(path).drop_duplicates(ignore_index=True)


def split_train_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a reproducible stratified (train, test) split of raw, unmodified rows."""
    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        stratify=frame[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _category_label(value: Any) -> Any:
    """Normalise a category so ``3``, ``3.0`` and ``"3"`` all map to the label ``"3"``.

    A single blank cell makes pandas read an integer column as float. Without this,
    every value in that column would become an unseen category at prediction time.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the feature columns and coerce them to the dtypes the pipelines expect.

    Row-level cleaning lives here rather than in the pipeline so uploaded CSVs and
    training data follow one code path. Extra columns are ignored.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {', '.join(missing)}")

    features = pd.DataFrame(index=frame.index)
    for column in NUMERIC_FEATURES:
        try:
            features[column] = pd.to_numeric(frame[column]).astype(float)
        except (TypeError, ValueError) as error:
            raise SchemaError(f"Column '{column}' contains non-numeric values.") from error
    for column in CATEGORICAL_FEATURES:
        features[column] = frame[column].map(_category_label).astype(object)
    return features


def prepare_target(frame: pd.DataFrame) -> pd.Series:
    """Encode the target as 0/1, accepting boolean, string and numeric spellings."""
    if TARGET_COLUMN not in frame.columns:
        raise SchemaError(f"Missing required target column: {TARGET_COLUMN}")

    encoded = frame[TARGET_COLUMN].astype(str).str.strip().str.lower().map(_TARGET_LABELS)
    if encoded.isna().any():
        invalid = frame.loc[encoded.isna(), TARGET_COLUMN].unique()[:3].tolist()
        raise SchemaError(f"Target column '{TARGET_COLUMN}' has invalid values, e.g. {invalid}.")
    return encoded.astype(int).rename(TARGET_COLUMN)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_raw_dataset()

    raw_row_count = len(pd.read_csv(RAW_DATA_PATH))
    frame = load_raw_dataset()
    train, test = split_train_test(frame)
    test.to_csv(TEST_DATA_PATH, index=False)

    logger.info("Raw rows: %d (duplicates removed: %d)", raw_row_count, raw_row_count - len(frame))
    logger.info("Train rows: %d | Test rows: %d", len(train), len(test))
    logger.info("Positive rate: %.4f", prepare_target(frame).mean())
    logger.info("Wrote %s", TEST_DATA_PATH.name)


if __name__ == "__main__":
    main()
