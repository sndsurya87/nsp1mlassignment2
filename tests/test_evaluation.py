import math

import pytest

from shopper_intent.evaluation import (
    classification_report_frame,
    compute_metrics,
    confusion_matrix_frame,
)


def test_metrics_match_values_worked_out_by_hand() -> None:
    # TP=2, FN=1, FP=1, TN=4
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.3, 0.7, 0.2, 0.2, 0.1, 0.1]

    metrics = compute_metrics(y_true, y_pred, y_score)

    assert metrics.accuracy == pytest.approx(6 / 8)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    # (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = 7 / 15
    assert metrics.mcc == pytest.approx(7 / 15)
    # 14 of 15 positive/negative pairs are ranked correctly; only (0.3, 0.7) is inverted.
    assert metrics.auc == pytest.approx(14 / 15)


def test_auc_is_nan_rather_than_an_exception_for_a_single_class() -> None:
    metrics = compute_metrics([0, 0, 0], [0, 0, 1], [0.1, 0.2, 0.9])
    assert math.isnan(metrics.auc)


def test_confusion_matrix_keeps_both_classes_when_one_is_absent() -> None:
    frame = confusion_matrix_frame([0, 0], [0, 0])
    assert frame.shape == (2, 2)
    assert frame.loc["No purchase", "No purchase"] == 2
    assert frame.loc["Purchase"].sum() == 0


def test_classification_report_has_a_row_per_class() -> None:
    frame = classification_report_frame([0, 1, 1], [0, 1, 0])
    assert {"No purchase", "Purchase", "macro avg", "weighted avg"} <= set(frame.index)


def test_classification_report_accuracy_row_is_not_broadcast_across_columns() -> None:
    frame = classification_report_frame([0, 1, 1, 1], [0, 1, 0, 1])
    accuracy_row = frame.loc["accuracy"]
    assert accuracy_row["f1-score"] == pytest.approx(0.75)
    assert accuracy_row["support"] == 4
    assert math.isnan(accuracy_row["precision"])
    assert math.isnan(accuracy_row["recall"])
