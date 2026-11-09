from streamlit.testing.v1 import AppTest

from shopper_intent.config import PROJECT_ROOT
from shopper_intent.models import MODEL_SPECS

_TIMEOUT_SECONDS = 60


def _run_app() -> AppTest:
    script = str(PROJECT_ROOT / "app.py")
    return AppTest.from_file(script, default_timeout=_TIMEOUT_SECONDS).run()


def test_app_loads_and_shows_every_required_section() -> None:
    app = _run_app()

    assert not app.exception
    assert not app.error
    assert [metric.label for metric in app.metric] == [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]
    # The all-models comparison and the classification report.
    assert len(app.table) == 2


def test_model_dropdown_offers_all_models_and_switches_the_detail_view() -> None:
    app = _run_app()
    dropdown = app.sidebar.selectbox[0]
    assert list(dropdown.options) == [spec.name for spec in MODEL_SPECS]

    first_accuracy = app.metric[0].value
    app = dropdown.select("Naive Bayes").run()

    assert not app.exception
    assert "Naive Bayes" in [heading.value for heading in app.subheader]
    assert app.metric[0].value != first_accuracy
