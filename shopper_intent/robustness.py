"""Check how far the headline comparison can be trusted.

Run ``python -m shopper_intent.robustness`` after training. It answers two questions the
single-split metrics table cannot, and writes the answers to ``model/robustness.json``:

1. Are differences between models larger than sampling noise? (paired bootstrap on the
   hold-out, so every model is compared on the same resampled rows)
2. How much of the performance depends on ``PageValues``? It is by far the strongest
   feature, and is derived from the value of pages visited before a transaction.
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef

from shopper_intent.config import MODEL_DIR, RANDOM_STATE
from shopper_intent.data import (
    load_raw_dataset,
    prepare_features,
    prepare_target,
    split_train_test,
)
from shopper_intent.evaluation import evaluate_model
from shopper_intent.models import MODEL_SPECS

logger = logging.getLogger(__name__)

ROBUSTNESS_PATH = MODEL_DIR / "robustness.json"
_BOOTSTRAP_ROUNDS = 2000
_NEUTRALISED_FEATURE = "PageValues"


def _bootstrap_mcc(predictions: dict[str, np.ndarray], target: np.ndarray) -> dict[str, np.ndarray]:
    """Return the MCC of every model on the same bootstrap resamples of the hold-out."""
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.integers(0, len(target), size=(_BOOTSTRAP_ROUNDS, len(target)))
    return {
        name: np.array([matthews_corrcoef(target[row], y_pred[row]) for row in indices])
        for name, y_pred in predictions.items()
    }


def _interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train, test = split_train_test(load_raw_dataset())
    x_train, y_train = prepare_features(train), prepare_target(train)
    x_test, y_test = prepare_features(test), prepare_target(test)

    models = {spec.name: joblib.load(MODEL_DIR / f"{spec.slug}.joblib") for spec in MODEL_SPECS}
    predictions = {name: model.predict(x_test) for name, model in models.items()}

    samples = _bootstrap_mcc(predictions, y_test.to_numpy())
    observed = {
        name: float(matthews_corrcoef(y_test, y_pred)) for name, y_pred in predictions.items()
    }
    best = max(observed, key=observed.__getitem__)

    bootstrap = {
        name: {
            "mcc": round(observed[name], 4),
            "ci_95": _interval(samples[name]),
            "gap_to_best_ci_95": None if name == best else _interval(samples[best] - samples[name]),
        }
        for name in observed
    }

    # Tuned hyper-parameters are kept rather than re-searched, so this is a quick sensitivity
    # check and understates what a model re-tuned without the feature could achieve. A
    # constant column removes the feature's information without changing the input schema.
    def without_feature(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(**{_NEUTRALISED_FEATURE: 0.0})

    x_train_reduced, x_test_reduced = without_feature(x_train), without_feature(x_test)
    ablation = {}
    for name, model in models.items():
        refitted = clone(model).fit(x_train_reduced, y_train)
        metrics, _ = evaluate_model(refitted, x_test_reduced, y_test)
        ablation[name] = {key: round(value, 4) for key, value in metrics.as_dict().items()}

    report = {
        "bootstrap_rounds": _BOOTSTRAP_ROUNDS,
        "best_model_by_test_mcc": best,
        "bootstrap_mcc": bootstrap,
        f"metrics_without_{_NEUTRALISED_FEATURE}": ablation,
    }
    ROBUSTNESS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
