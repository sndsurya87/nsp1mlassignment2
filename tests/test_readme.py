import json

from shopper_intent.config import METRICS_PATH, PROJECT_ROOT

_METRIC_ORDER = ("accuracy", "auc", "precision", "recall", "f1", "mcc")


def test_readme_comparison_table_matches_the_recorded_metrics() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text()
    for model in json.loads(METRICS_PATH.read_text())["models"]:
        values = " | ".join(f"{model['test_metrics'][key]:.4f}" for key in _METRIC_ORDER)
        assert f"| {model['name']} | {values} |" in readme
