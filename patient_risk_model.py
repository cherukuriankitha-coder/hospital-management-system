"""Patient readmission-risk modeling example for a data-science portfolio.

Uses synthetic/sample features only. This is an educational analytics example,
not a clinical decision tool.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_and_evaluate(df: pd.DataFrame, target: str = "readmitted") -> Pipeline:
    features = ["age", "length_of_stay", "prior_visits", "department"]
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline(
        numeric_features=["age", "length_of_stay", "prior_visits"],
        categorical_features=["department"],
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 3))
    print(classification_report(y_test, predictions, zero_division=0))
    return pipeline


if __name__ == "__main__":
    # Small synthetic dataset for demonstration; replace with an approved dataset.
    data = pd.DataFrame(
        {
            "age": [24, 67, 51, 73, 45, 62, 36, 80, 58, 29, 69, 41],
            "length_of_stay": [1, 8, 4, 10, 3, 7, 2, 12, 6, 1, 9, 3],
            "prior_visits": [0, 4, 2, 5, 1, 3, 0, 6, 2, 0, 4, 1],
            "department": [
                "General", "Cardiology", "Emergency", "Cardiology",
                "General", "Emergency", "General", "Cardiology",
                "Emergency", "General", "Cardiology", "Emergency",
            ],
            "readmitted": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    train_and_evaluate(data)
