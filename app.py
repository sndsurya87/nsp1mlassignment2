"""Streamlit front end: evaluate the saved classifiers on an uploaded test CSV.

Run locally with ``streamlit run app.py``. Models are only ever loaded and used for
prediction here; all training happens in ``shopper_intent.train``.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from shopper_intent.config import FEATURE_COLUMNS, TARGET_COLUMN, TEST_DATA_PATH
from shopper_intent.data import SchemaError
from shopper_intent.evaluation import classification_report_frame, confusion_matrix_frame
from shopper_intent.inference import EvaluationRun, evaluate_models, load_models, parse_csv

_METRIC_FORMAT = "{:.4f}"
_MISSING_VALUE_TEXT = "n/a"
_CONFUSION_MATRIX_WIDTH_PX = 420


@st.cache_resource(show_spinner="Loading saved models...")
def _cached_models() -> dict[str, Pipeline]:
    return load_models()


@st.cache_data(show_spinner="Scoring all models...", max_entries=4)
def _cached_evaluation(content: bytes) -> EvaluationRun:
    """Score the CSV once per distinct file, so changing the dropdown is instant."""
    return evaluate_models(_cached_models(), parse_csv(content))


def _confusion_matrix_figure(matrix: pd.DataFrame) -> Figure:
    """Plot counts, colouring by share of each actual class.

    Raw counts would make the large "No purchase" row dominate the colour scale and
    hide how well the rarer "Purchase" class is being recognised.
    """
    row_share = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    labels = matrix.astype(str) + "\n" + row_share.map("{:.1%}".format)

    figure, axes = plt.subplots(figsize=(4.6, 3.6))
    sns.heatmap(
        row_share,
        annot=labels,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=2,
        linecolor="white",
        square=True,
        ax=axes,
    )
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    axes.tick_params(length=0)
    figure.tight_layout()
    return figure


def _render_sidebar() -> tuple[bytes, str, str]:
    """Draw the controls, returning the CSV content, its description and the chosen model."""
    with st.sidebar:
        st.header("Test data")
        uploaded = st.file_uploader("Upload a test CSV", type="csv")
        st.download_button(
            "Download sample test_data.csv",
            data=TEST_DATA_PATH.read_bytes(),
            file_name=TEST_DATA_PATH.name,
            mime="text/csv",
        )
        with st.expander("Required columns"):
            st.write(f"**Target:** `{TARGET_COLUMN}`")
            st.write(", ".join(f"`{column}`" for column in FEATURE_COLUMNS))

        st.header("Model")
        selected_model = st.selectbox("Model for detailed results", _cached_models())

    if uploaded is None:
        return (
            TEST_DATA_PATH.read_bytes(),
            "the bundled hold-out set (test_data.csv)",
            selected_model,
        )
    return uploaded.getvalue(), f"your upload ({uploaded.name})", selected_model


def _render_comparison(run: EvaluationRun) -> None:
    st.subheader("All models")
    table = run.comparison_table
    styled = table.style.format(_METRIC_FORMAT, na_rep=_MISSING_VALUE_TEXT).highlight_max(
        axis=0, props="font-weight: 700"
    )
    st.table(styled)
    st.caption("Bold marks the best value in each column.")


def _render_model_detail(run: EvaluationRun, model_name: str) -> None:
    result = run.results[model_name]
    st.subheader(model_name)

    columns = st.columns(len(run.comparison_table.columns))
    for column, (label, value) in zip(
        columns, run.comparison_table.loc[model_name].items(), strict=True
    ):
        text = _MISSING_VALUE_TEXT if pd.isna(value) else _METRIC_FORMAT.format(value)
        column.metric(label, text)

    matrix_column, report_column = st.columns(2)
    with matrix_column:
        st.markdown("**Confusion matrix**")
        figure = _confusion_matrix_figure(confusion_matrix_frame(run.target, result.predictions))
        st.pyplot(figure, width=_CONFUSION_MATRIX_WIDTH_PX, alt="Confusion matrix heatmap")
        plt.close(figure)
    with report_column:
        st.markdown("**Classification report**")
        report = classification_report_frame(run.target, result.predictions)
        st.table(report.style.format({"support": "{:.0f}"}, precision=4, na_rep=""))


def main() -> None:
    st.set_page_config(page_title="Online Shopper Purchase Prediction", layout="wide")
    st.title("Online Shopper Purchase Prediction")
    st.write(
        "Compares five saved classifiers on predicting whether a browsing session ends in a "
        "purchase. Models are not retrained on the data you upload."
    )

    content, source_description, selected_model = _render_sidebar()

    try:
        run = _cached_evaluation(content)
    except SchemaError as error:
        st.error(f"This file can't be evaluated. {error}")
        st.stop()

    positives = int(run.target.sum())
    st.info(
        f"Showing results for {source_description}: {len(run.target):,} rows, "
        f"{positives:,} purchases ({positives / len(run.target):.1%})."
    )
    _render_comparison(run)
    _render_model_detail(run, selected_model)


if __name__ == "__main__":
    main()
