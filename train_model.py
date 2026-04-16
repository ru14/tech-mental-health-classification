"""
train_model.py – Reproducible model training script.

Trains the best-performing classification pipeline (preprocessing + model)
on the mental health survey data and saves it to model.pkl.

Usage:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── 1. Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/survey.csv")

# ── 2. Clean Data (mirrors notebook steps) ───────────────────────────────────
# Drop free-text column
if "comments" in df.columns:
    df.drop(columns=["comments"], inplace=True)

# Fill missing categoricals with 'Unknown'
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("Unknown")

# Remove unrealistic ages
df = df[(df["Age"] > 15) & (df["Age"] < 100)]

# Remove duplicates
df.drop_duplicates(inplace=True)


# Standardize gender
def standardize_gender(g):
    g = str(g).strip().lower()
    if g in ["male", "m", "man", "cis male", "cis man"]:
        return "Male"
    elif g in ["female", "f", "woman", "cis female", "cis woman"]:
        return "Female"
    else:
        return "Other"


df["Gender"] = df["Gender"].apply(standardize_gender)

# ── 3. Feature / Target Selection ────────────────────────────────────────────
feature_cols = [
    "Age", "Gender", "self_employed", "family_history",
    "work_interfere", "no_employees", "remote_work", "tech_company",
    "benefits", "care_options", "wellness_program", "seek_help",
    "anonymity", "mental_health_consequence", "coworkers",
    "supervisor", "mental_vs_physical", "obs_consequence",
]
target_col = "treatment"

df_model = df[feature_cols + [target_col]].copy()

# Encode target: Yes → 1, No → 0
df_model[target_col] = (df_model[target_col] == "Yes").astype(int)

X = df_model.drop(columns=[target_col])
y = df_model[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 4. Preprocessing: ColumnTransformer (encoding + scaling) ─────────────────
numeric_cols = ["Age"]
categorical_cols = [c for c in feature_cols if c != "Age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_cols,
        ),
    ]
)

# ── 5. Build Pipelines ────────────────────────────────────────────────────────
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

rf_param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring="f1", n_jobs=-1)

dt_param_grid = {
    "classifier__max_depth": [5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__criterion": ["gini", "entropy"],
}
dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42)),
])
dt_grid = GridSearchCV(dt_pipeline, dt_param_grid, cv=5, scoring="f1", n_jobs=-1)

# ── 6. Train All Models ───────────────────────────────────────────────────────
print("Training Logistic Regression …")
lr_pipeline.fit(X_train, y_train)

print("Training Random Forest (GridSearchCV) …")
rf_grid.fit(X_train, y_train)

print("Training Decision Tree (GridSearchCV) …")
dt_grid.fit(X_train, y_train)

best_rf_pipeline = rf_grid.best_estimator_
best_dt_pipeline = dt_grid.best_estimator_

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
model_scores = {}
for name, pipeline in [
    ("Logistic Regression", lr_pipeline),
    ("Random Forest", best_rf_pipeline),
    ("Decision Tree", best_dt_pipeline),
]:
    y_pred = pipeline.predict(X_test)
    model_scores[name] = {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

print("\nModel Evaluation:")
print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
for name, scores in model_scores.items():
    print(
        f"{name:<25} {scores['accuracy']:>10.4f} {scores['f1']:>10.4f}"
        f" {scores['precision']:>10.4f} {scores['recall']:>10.4f}"
    )

# ── 8. Select Best Model by F1 ────────────────────────────────────────────────
best_model_name = max(model_scores, key=lambda k: model_scores[k]["f1"])
best_model_pipeline = model_scores[best_model_name]["pipeline"]
best_f1 = model_scores[best_model_name]["f1"]

print(f"\n✅ Best model: {best_model_name} (F1 = {best_f1:.4f})")

# ── 9. Save Pipeline ─────────────────────────────────────────────────────────
joblib.dump(best_model_pipeline, "model.pkl")
print("✅ Saved best_model_pipeline to 'model.pkl'")
print(f"   Pipeline steps: {[(n, type(s).__name__) for n, s in best_model_pipeline.steps]}")
