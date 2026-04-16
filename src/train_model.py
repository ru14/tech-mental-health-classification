"""
Train and serialize the mental health treatment prediction pipeline.

Run once before starting the app (or called automatically on first launch):
    python src/train_model.py
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "..", "data", "survey.csv")
MODEL_PATH = os.path.join(_HERE, "..", "models", "rf_pipeline.pkl")

FEATURE_COLS = [
    "Age",
    "Gender",
    "self_employed",
    "family_history",
    "work_interfere",
    "no_employees",
    "remote_work",
    "tech_company",
    "benefits",
    "care_options",
    "wellness_program",
    "seek_help",
    "anonymity",
    "mental_health_consequence",
    "coworkers",
    "supervisor",
    "mental_vs_physical",
    "obs_consequence",
]

CATEGORICAL_COLS = [c for c in FEATURE_COLS if c != "Age"]
NUMERIC_COLS = ["Age"]
TARGET = "treatment"
MIN_VALID_AGE = 15
MAX_VALID_AGE = 100


def standardize_gender(g: str) -> str:
    g = str(g).strip().lower()
    if g in {"male", "m", "man", "cis male", "cis man"}:
        return "Male"
    if g in {"female", "f", "woman", "cis female", "cis woman"}:
        return "Female"
    return "Other"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "comments" in df.columns:
        df.drop(columns=["comments"], inplace=True)
    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].fillna("Unknown")
    df = df[(df["Age"] > MIN_VALID_AGE) & (df["Age"] < MAX_VALID_AGE)].copy()
    df.drop_duplicates(inplace=True)
    df["Gender"] = df["Gender"].apply(standardize_gender)
    return df


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> Pipeline:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    df = load_and_clean(data_path)
    X = df[FEATURE_COLS].copy()
    y = (df[TARGET] == "Yes").astype(int)

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                CATEGORICAL_COLS,
            )
        ],
        remainder="passthrough",  # Age passes through as-is
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")

    joblib.dump(pipeline, model_path)
    print(f"  Model saved → {model_path}")
    return pipeline


if __name__ == "__main__":
    print("Training mental-health prediction pipeline …")
    train()
    print("Done.")
