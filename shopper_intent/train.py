"""Tune, fit, evaluate and persist all five classifiers.

Run ``python -m shopper_intent.train``. Each model is tuned by stratified cross-validation
on the training split only. The hold-out is scored once, after tuning, and never influences
model selection.
"""

import json
import logging
import platform
import time
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from shopper_intent.config import (
    CV_FOLDS,
    METRICS_PATH,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_DATA_PATH,
)
from shopper_intent.data import (
    download_raw_dataset,
    load_raw_dataset,
    prepare_features,
    prepare_target,
    split_train_test,
)
from shopper_intent.evaluation import compute_metrics, evaluate_model
from shopper_intent.models import MODEL_SPECS, ModelSpec

logger = logging.getLogger(__name__)

_JOBLIB_COMPRESSION = 3


def _tune(spec: ModelSpec, features: pd.DataFrame, target: pd.Series) -> GridSearchCV:
    """Grid-search a model, selecting on MCC.

    MCC is used because it stays informative under class imbalance and, unlike AUC, it
    rewards the hard decisions the model actually makes at its default threshold.
    """
    search = GridSearchCV(
        spec.build_pipeline(),
        spec.pipeline_param_grid,
        scoring=make_scorer(matthews_corrcoef),
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    return search.fit(features, target)


def _baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """Score a majority-class predictor, the reference point any real model must beat."""
    placeholder = np.zeros((len(y_train), 1))
    dummy = DummyClassifier(strategy="most_frequent").fit(placeholder, y_train)
    placeholder = np.zeros((len(y_test), 1))
    predictions = dummy.predict(placeholder)
    scores = dummy.predict_proba(placeholder)[:, 1]
    return compute_metrics(y_test, predictions, scores).as_dict()


def _train_one(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Tune and persist one model, returning its hold-out results."""
    started = time.perf_counter()
    search = _tune(spec, x_train, y_train)
    elapsed = time.perf_counter() - started

    model = search.best_estimator_
    metrics, _ = evaluate_model(model, x_test, y_test)
    joblib.dump(model, MODEL_DIR / f"{spec.slug}.joblib", compress=_JOBLIB_COMPRESSION)

    logger.info(
        "%-24s cv_mcc=%.4f test_mcc=%.4f  (%.1fs, %s)",
        spec.name,
        search.best_score_,
        metrics.mcc,
        elapsed,
        search.best_params_,
    )
    return {
        "name": spec.name,
        "slug": spec.slug,
        "test_metrics": metrics.as_dict(),
        "cv_mcc": float(search.best_score_),
        "best_params": {
            key.split("__", maxsplit=1)[1]: value for key, value in search.best_params_.items()
        },
        "search_seconds": round(elapsed, 1),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_raw_dataset()
    MODEL_DIR.mkdir(exist_ok=True)

    train, test = split_train_test(load_raw_dataset())
    test.to_csv(TEST_DATA_PATH, index=False)
    x_train, y_train = prepare_features(train), prepare_target(train)
    x_test, y_test = prepare_features(test), prepare_target(test)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "split": {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_positive_rate": float(y_test.mean()),
        },
        "baseline_metrics": _baseline_metrics(y_train, y_test),
        "models": [_train_one(spec, x_train, y_train, x_test, y_test) for spec in MODEL_SPECS],
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote models and %s to %s/", METRICS_PATH.name, MODEL_DIR.name)


if __name__ == "__main__":
    main()
