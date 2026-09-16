"""Data validation helpers for hospital analytics datasets."""
from typing import Iterable
import pandas as pd


def validate_patient_data(df: pd.DataFrame, required: Iterable[str]) -> dict:
    missing = sorted(set(required) - set(df.columns))
    duplicate_rows = int(df.duplicated().sum())
    null_counts = df.isna().sum().sort_values(ascending=False).to_dict()
    return {
        "valid_schema": len(missing) == 0,
        "missing_columns": missing,
        "duplicate_rows": duplicate_rows,
        "null_counts": {k: int(v) for k, v in null_counts.items() if v > 0},
    }
