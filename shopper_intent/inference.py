"""Score saved models against a user-supplied CSV. Nothing here ever fits a model."""

import io
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from shopper_intent.config import MODEL_DIR
from shopper_intent.data import SchemaError, prepare_features, prepare_target
from shopper_intent.evaluation import ClassificationMetrics, evaluate_model
from shopper_intent.models import MODEL_SPECS


@dataclass(frozen=True)
class ModelResult:
    """One saved model's performance on an evaluation set."""

    name: str
    metrics: ClassificationMetrics
    predictions: np.ndarray


@dataclass(frozen=True)
class EvaluationRun:
    """Every saved model's performance on one evaluation set."""

    target: np.ndarray
    results: dict[str, ModelResult]

    @property
    def comparison_table(self) -> pd.DataFrame:
        """Return one row per model, using the column order the assignment specifies."""
        table = pd.DataFrame(
            {name: result.metrics.as_dict() for name, result in self.results.items()}
        ).transpose()
        table = table.rename(
            columns={
                "accuracy": "Accuracy",
                "auc": "AUC",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "mcc": "MCC",
            }
        )
        table.index.name = "ML Model Name"
        return table


def load_models() -> dict[str, Pipeline]:
    """Load every serialised pipeline, keyed by display name."""
    return {spec.name: joblib.load(MODEL_DIR / f"{spec.slug}.joblib") for spec in MODEL_SPECS}


def parse_csv(content: bytes) -> pd.DataFrame:
    """Parse CSV bytes, reporting malformed input as a :class:`SchemaError`."""
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise SchemaError(f"The file could not be read as a CSV: {error}") from error
    if frame.empty:
        raise SchemaError("The CSV has a header but no data rows.")
    return frame


def evaluate_models(models: dict[str, Pipeline], frame: pd.DataFrame) -> EvaluationRun:
    """Score each saved model on ``frame``, which must hold the features and the target."""
    features = prepare_features(frame)
    target = prepare_target(frame)

    results = {}
    for name, model in models.items():
        metrics, predictions = evaluate_model(model, features, target)
        results[name] = ModelResult(name, metrics, predictions)
    return EvaluationRun(target=target.to_numpy(), results=results)
