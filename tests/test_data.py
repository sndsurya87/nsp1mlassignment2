import pandas as pd
import pytest

from shopper_intent.config import (
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
)
from shopper_intent.data import (
    SchemaError,
    load_raw_dataset,
    prepare_features,
    prepare_target,
    split_train_test,
)


@pytest.fixture(scope="module")
def raw_frame() -> pd.DataFrame:
    return load_raw_dataset()


def test_dataset_meets_assignment_minimums() -> None:
    raw = pd.read_csv(RAW_DATA_PATH)
    assert len(FEATURE_COLUMNS) >= 12
    assert len(raw) >= 500


def test_split_has_no_overlap_and_preserves_class_balance(raw_frame: pd.DataFrame) -> None:
    train, test = split_train_test(raw_frame)
    assert len(train) + len(test) == len(raw_frame)
    assert train.merge(test, how="inner").empty
    assert train[TARGET_COLUMN].mean() == pytest.approx(test[TARGET_COLUMN].mean(), abs=0.002)


def test_committed_test_data_matches_the_reproducible_split(raw_frame: pd.DataFrame) -> None:
    _, test = split_train_test(raw_frame)
    pd.testing.assert_frame_equal(pd.read_csv(TEST_DATA_PATH), test, check_dtype=False)


def test_missing_feature_column_is_reported_by_name(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="PageValues"):
        prepare_features(raw_frame.drop(columns="PageValues"))


def test_non_numeric_value_in_numeric_column_is_rejected(raw_frame: pd.DataFrame) -> None:
    corrupted = raw_frame.head(3).astype({"BounceRates": object})
    corrupted.loc[0, "BounceRates"] = "not-a-number"
    with pytest.raises(SchemaError, match="BounceRates"):
        prepare_features(corrupted)


def test_integer_categories_survive_being_read_as_float(raw_frame: pd.DataFrame) -> None:
    sample = raw_frame.head(3).astype({"Browser": float})
    assert prepare_features(sample)["Browser"].tolist() == [
        str(value) for value in raw_frame.head(3)["Browser"]
    ]


@pytest.mark.parametrize(
    ("raw_values", "expected"),
    [
        ([True, False], [1, 0]),
        (["TRUE", "false"], [1, 0]),
        ([1, 0], [1, 0]),
        ([1.0, 0.0], [1, 0]),
    ],
)
def test_target_spellings_are_normalised(raw_values: list, expected: list[int]) -> None:
    frame = pd.DataFrame({TARGET_COLUMN: raw_values})
    assert prepare_target(frame).tolist() == expected


def test_invalid_target_is_rejected() -> None:
    with pytest.raises(SchemaError, match="invalid values"):
        prepare_target(pd.DataFrame({TARGET_COLUMN: ["maybe"]}))


def test_missing_target_is_rejected(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match=TARGET_COLUMN):
        prepare_target(raw_frame.drop(columns=TARGET_COLUMN))
