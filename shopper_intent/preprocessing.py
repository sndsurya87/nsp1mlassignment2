"""Feature preprocessing, fitted inside each model pipeline.

Keeping preprocessing in the pipeline means it is re-fitted on the training portion of
every cross-validation fold (no leakage) and is serialised along with the model, so the
app needs no separate preprocessing artefact.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from shopper_intent.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Categories rarer than this are pooled, which avoids near-constant one-hot columns.
# These destabilise GaussianNB's variance estimates and add noise dimensions to KNN.
_MIN_CATEGORY_FREQUENCY = 10


def build_preprocessor(*, transform_numeric: bool) -> ColumnTransformer:
    """Build the column transformer.

    Args:
        transform_numeric: Apply a Yeo-Johnson power transform with standardisation. The
            page-count and duration features are heavily right-skewed, which distorts
            distance- and variance-based models. Trees only use feature order, so they
            should pass ``False``.
    """
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if transform_numeric:
        numeric_steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=True)))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        (
            "encode",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=_MIN_CATEGORY_FREQUENCY,
                sparse_output=False,
            ),
        ),
    ]

    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("categorical", Pipeline(categorical_steps), list(CATEGORICAL_FEATURES)),
        ]
    )
