import json

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from shopper_intent.config import METRICS_PATH, TEST_DATA_PATH
from shopper_intent.data import SchemaError
from shopper_intent.inference import EvaluationRun, evaluate_models, load_models, parse_csv


@pytest.fixture(scope="module")
def models() -> dict[str, Pipeline]:
    return load_models()


@pytest.fixture(scope="module")
def run(models: dict[str, Pipeline]) -> EvaluationRun:
    return evaluate_models(models, pd.read_csv(TEST_DATA_PATH))


def test_saved_models_reproduce_the_metrics_recorded_at_training_time(run: EvaluationRun) -> None:
    """Proves test_data.csv, the artefacts and metrics.json all belong to the same run."""
    recorded = json.loads(METRICS_PATH.read_text())["models"]
    assert len(recorded) == len(run.results) == 5
    for entry in recorded:
        actual = run.results[entry["name"]].metrics.as_dict()
        assert actual == pytest.approx(entry["test_metrics"])


def test_comparison_table_uses_the_required_columns(run: EvaluationRun) -> None:
    table = run.comparison_table
    assert list(table.columns) == ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    assert len(table) == 5


def test_evaluation_never_changes_a_fitted_model(models: dict[str, Pipeline]) -> None:
    frame = pd.read_csv(TEST_DATA_PATH).head(50)
    flipped = frame.assign(Revenue=~frame["Revenue"])
    first = evaluate_models(models, frame).results["KNN"].predictions
    evaluate_models(models, flipped)
    second = evaluate_models(models, frame).results["KNN"].predictions
    assert (first == second).all()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "could not be read"),
        (b"\xff\xfe\x00bad", "could not be read"),
        (b"a,b\n", "no data rows"),
    ],
)
def test_malformed_csv_gives_a_friendly_error(content: bytes, message: str) -> None:
    with pytest.raises(SchemaError, match=message):
        parse_csv(content)
