"""Evaluation metrics, shared by the training script and the Streamlit app."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

CLASS_LABELS = (0, 1)
CLASS_NAMES = ("No purchase", "Purchase")


@dataclass(frozen=True)
class ClassificationMetrics:
    """The six required metrics. Precision, recall and F1 refer to the positive class."""

    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def positive_class_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return a continuous score for ranking samples by likelihood of the positive class.

    SVC is trained without probability estimates, so it falls back to its signed margin.
    AUC depends only on rank order, which makes the two sources interchangeable.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]
) -> ClassificationMetrics:
    """Compute all metrics. AUC is NaN when ``y_true`` holds only one class (it is undefined)."""
    has_both_classes = len(np.unique(y_true)) == len(CLASS_LABELS)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc=float(roc_auc_score(y_true, y_score)) if has_both_classes else float("nan"),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
    )


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> tuple[ClassificationMetrics, np.ndarray]:
    """Score a fitted model, returning its metrics and hard predictions."""
    predictions = model.predict(features)
    scores = positive_class_scores(model, features)
    return compute_metrics(target, predictions, scores), predictions


def confusion_matrix_frame(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """Return the confusion matrix with labelled axes. Rows are actual, columns predicted."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_LABELS))
    return pd.DataFrame(matrix, index=list(CLASS_NAMES), columns=list(CLASS_NAMES))


def classification_report_frame(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """Return scikit-learn's classification report as a data frame.

    The report stores accuracy as a bare scalar, which a data frame would copy into every
    column. The row is rebuilt to match the text report: only F1 and total support are set.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(CLASS_LABELS),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    accuracy = report.pop("accuracy")
    frame = pd.DataFrame(report).transpose()
    frame.loc["accuracy"] = {
        "precision": float("nan"),
        "recall": float("nan"),
        "f1-score": accuracy,
        "support": float(len(y_true)),
    }
    return frame.loc[[*CLASS_NAMES, "accuracy", "macro avg", "weighted avg"]]
