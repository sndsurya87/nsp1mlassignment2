import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid

from shopper_intent.config import CATEGORICAL_FEATURES
from shopper_intent.data import load_raw_dataset, prepare_features, prepare_target
from shopper_intent.evaluation import positive_class_scores
from shopper_intent.models import MODEL_SPECS, ModelSpec

_SAMPLE_SIZE = 600


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    frame = load_raw_dataset().sample(_SAMPLE_SIZE, random_state=0)
    return prepare_features(frame), prepare_target(frame)


def test_registry_covers_the_five_required_models() -> None:
    assert {spec.name for spec in MODEL_SPECS} == {
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Support Vector Machine",
    }


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec.slug)
def test_every_grid_candidate_is_a_valid_configuration(spec: ModelSpec) -> None:
    pipeline = spec.build_pipeline()
    for candidate in ParameterGrid(spec.pipeline_param_grid):
        pipeline.set_params(**candidate)


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec.slug)
def test_pipeline_fits_scores_and_tolerates_unseen_categories(
    spec: ModelSpec, sample: tuple[pd.DataFrame, pd.Series]
) -> None:
    features, target = sample
    model = spec.build_pipeline().fit(features, target)

    unseen = features.head(5).copy()
    unseen[list(CATEGORICAL_FEATURES)] = "never-seen-in-training"

    assert len(model.predict(unseen)) == 5
    assert len(positive_class_scores(model, unseen)) == 5


@pytest.mark.parametrize("spec", MODEL_SPECS, ids=lambda spec: spec.slug)
def test_pipeline_imputes_missing_values(
    spec: ModelSpec, sample: tuple[pd.DataFrame, pd.Series]
) -> None:
    features, target = sample
    model = spec.build_pipeline().fit(features, target)

    with_gaps = features.head(5).copy()
    with_gaps.iloc[0, :] = float("nan")

    assert len(model.predict(with_gaps)) == 5
