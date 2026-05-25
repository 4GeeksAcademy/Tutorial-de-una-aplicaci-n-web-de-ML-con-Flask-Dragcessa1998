from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET_COLUMN = "species"


def load_iris_dataframe() -> pd.DataFrame:
    """Load the UCI Iris dataset bundled with scikit-learn."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(dict(enumerate(iris.target_names)))
    return df


def build_model(random_state: int = 42, **classifier_params: Any) -> Pipeline:
    """Build a small, production-friendly classification pipeline."""
    classifier = RandomForestClassifier(random_state=random_state, **classifier_params)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def predict_species(artifact: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
    """Return predicted species and per-class probabilities for a single flower."""
    model: Pipeline = artifact["model"]
    class_names: list[str] = artifact["class_names"]
    row = pd.DataFrame([[features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(row)[0]
    prediction = model.predict(row)[0]

    return {
        "prediction": str(prediction),
        "probabilities": {
            class_name: round(float(probability), 4)
            for class_name, probability in zip(class_names, probabilities)
        },
    }


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
