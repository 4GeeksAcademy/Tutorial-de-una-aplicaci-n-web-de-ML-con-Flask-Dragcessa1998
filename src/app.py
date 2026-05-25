from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split

from utils import FEATURE_COLUMNS, TARGET_COLUMN, build_model, evaluate_model, load_iris_dataframe, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "iris.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_PATH = MODEL_DIR / "iris_classifier.joblib"
METRICS_PATH = MODEL_DIR / "iris_metrics.json"
RANDOM_STATE = 42


def train_and_save_model() -> dict:
    df = load_iris_dataframe()

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_DIR / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_DIR / "test.csv", index=False)

    baseline_model = build_model(random_state=RANDOM_STATE)
    baseline_model.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)

    search = GridSearchCV(
        estimator=build_model(random_state=RANDOM_STATE),
        param_grid={
            "classifier__n_estimators": [80, 120, 180],
            "classifier__max_depth": [2, 3, None],
            "classifier__min_samples_leaf": [1, 2],
        },
        cv=5,
        scoring="accuracy",
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    optimized_model = search.best_estimator_
    optimized_metrics = evaluate_model(optimized_model, X_test, y_test)
    feature_importance = _feature_importance(optimized_model)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": optimized_model,
        "feature_columns": FEATURE_COLUMNS,
        "class_names": sorted(y.unique().tolist()),
        "dataset_source": "UCI Iris Dataset via scikit-learn",
    }
    joblib.dump(artifact, MODEL_PATH)

    results = {
        "dataset": {
            "source": "UCI Iris Dataset, loaded with sklearn.datasets.load_iris",
            "rows": int(df.shape[0]),
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "classes": sorted(y.unique().tolist()),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
        "baseline_model": baseline_metrics,
        "optimized_model": {
            "best_params": search.best_params_,
            "best_cv_accuracy": search.best_score_,
            **optimized_metrics,
        },
        "feature_importance": feature_importance,
        "artifacts": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "raw_data_path": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)),
            "train_path": str((PROCESSED_DIR / "train.csv").relative_to(PROJECT_ROOT)),
            "test_path": str((PROCESSED_DIR / "test.csv").relative_to(PROJECT_ROOT)),
            "figures_dir": str(REPORTS_DIR.relative_to(PROJECT_ROOT)),
        },
    }
    write_json(results, METRICS_PATH)
    create_visualizations(df, feature_importance)
    return results


def _feature_importance(model) -> list[dict]:
    classifier = model.named_steps["classifier"]
    return [
        {"feature": feature, "importance": round(float(importance), 4)}
        for feature, importance in sorted(
            zip(FEATURE_COLUMNS, classifier.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def create_visualizations(df: pd.DataFrame, feature_importance: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set2", legend=False)
    plt.title("Distribucion de especies")
    plt.xlabel("Especie")
    plt.ylabel("Registros")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "species_distribution.png", dpi=160)
    plt.close()

    importance_df = pd.DataFrame(feature_importance)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", color="#4b6f9f")
    plt.title("Importancia de variables")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="petal_length_cm",
        y="petal_width_cm",
        hue=TARGET_COLUMN,
        palette="Set2",
        s=80,
    )
    plt.title("Separacion por medidas del petalo")
    plt.xlabel("Longitud del petalo (cm)")
    plt.ylabel("Ancho del petalo (cm)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "petal_scatter.png", dpi=160)
    plt.close()


def main() -> None:
    results = train_and_save_model()
    print("Iris ML Flask project training complete")
    print(f"Rows: {results['dataset']['rows']}")
    print(
        "Baseline - "
        f"accuracy: {results['baseline_model']['accuracy']:.3f}, "
        f"macro F1: {results['baseline_model']['macro_f1']:.3f}"
    )
    print(
        "Optimized - "
        f"accuracy: {results['optimized_model']['accuracy']:.3f}, "
        f"macro F1: {results['optimized_model']['macro_f1']:.3f}"
    )
    print(f"Best params: {results['optimized_model']['best_params']}")
    print(f"Saved model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
