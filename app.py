from __future__ import annotations

import sys
from pathlib import Path

import joblib
from flask import Flask, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import FEATURE_COLUMNS, predict_species  # noqa: E402


app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
MODEL_PATH = PROJECT_ROOT / "models" / "iris_classifier.joblib"


FIELD_LABELS = {
    "sepal_length_cm": "Sepal length (cm)",
    "sepal_width_cm": "Sepal width (cm)",
    "petal_length_cm": "Petal length (cm)",
    "petal_width_cm": "Petal width (cm)",
}

DEFAULT_VALUES = {
    "sepal_length_cm": 5.8,
    "sepal_width_cm": 3.0,
    "petal_length_cm": 4.35,
    "petal_width_cm": 1.3,
}


def load_model_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The model artifact was not found. Run `python src/app.py` before starting Flask."
        )
    return joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = DEFAULT_VALUES.copy()

    if request.method == "POST":
        try:
            form_values = {
                column: float(request.form.get(column, DEFAULT_VALUES[column]))
                for column in FEATURE_COLUMNS
            }
            artifact = load_model_artifact()
            result = predict_species(artifact, form_values)
        except ValueError:
            error = "Please enter valid numeric values for every measurement."
        except Exception as exc:  # Keep UI helpful on deployment misconfiguration.
            error = str(exc)

    return render_template(
        "index.html",
        fields=FIELD_LABELS,
        values=form_values,
        result=result,
        error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
