"""Registry of the five classifiers and their hyper-parameter search spaces."""

from dataclasses import dataclass
from typing import Any, Final

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from shopper_intent.config import RANDOM_STATE
from shopper_intent.preprocessing import build_preprocessor

_CLASSIFIER_STEP: Final = "classifier"


@dataclass(frozen=True)
class ModelSpec:
    """A classifier, how to preprocess for it and what to tune."""

    name: str
    slug: str
    estimator: ClassifierMixin
    param_grid: dict[str, list[Any]]
    transform_numeric: bool = True

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("preprocess", build_preprocessor(transform_numeric=self.transform_numeric)),
                (_CLASSIFIER_STEP, self.estimator),
            ]
        )

    @property
    def pipeline_param_grid(self) -> dict[str, list[Any]]:
        """The search space, with keys addressed to the pipeline's classifier step."""
        return {f"{_CLASSIFIER_STEP}__{key}": values for key, values in self.param_grid.items()}


# class_weight is searched, not assumed: reweighting trades precision for recall, and on a
# 16% positive target the cross-validated score should decide if that trade is worthwhile.
MODEL_SPECS: Final[tuple[ModelSpec, ...]] = (
    ModelSpec(
        name="Logistic Regression",
        slug="logistic_regression",
        estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        param_grid={
            "C": [0.001, 0.003, 0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
    ),
    ModelSpec(
        name="Decision Tree",
        slug="decision_tree",
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [4, 6, 8, 12, None],
            "min_samples_leaf": [1, 10, 25, 50, 100, 200],
            "class_weight": [None, "balanced"],
        },
        transform_numeric=False,
    ),
    ModelSpec(
        name="KNN",
        slug="knn",
        estimator=KNeighborsClassifier(),
        param_grid={
            "n_neighbors": [5, 11, 21, 41, 81, 161],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    ),
    ModelSpec(
        name="Naive Bayes",
        slug="naive_bayes",
        estimator=GaussianNB(),
        param_grid={"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]},
    ),
    # probability=False is deliberate. AUC needs only a ranking, which decision_function
    # provides, whereas Platt scaling would multiply training time by an internal 5-fold CV.
    ModelSpec(
        name="Support Vector Machine",
        slug="svm",
        estimator=SVC(kernel="rbf", random_state=RANDOM_STATE),
        param_grid={
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", 0.01],
            "class_weight": [None, "balanced"],
        },
    ),
)
