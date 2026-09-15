"""Exploratory data analysis helpers for hospital operations data."""

import pandas as pd


def summarize_dataset(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def wait_time_by_department(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("department", dropna=False)["wait_minutes"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .round(2)
        .sort_values("mean", ascending=False)
    )


def high_wait_cases(df: pd.DataFrame, threshold: int = 45) -> pd.DataFrame:
    return df[df["wait_minutes"] > threshold].sort_values("wait_minutes", ascending=False)


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5, 6],
            "department": ["Emergency", "Cardiology", "Emergency", "General", "Cardiology", "Emergency"],
            "wait_minutes": [52, 21, 64, 17, 29, 48],
        }
    )
    print(summarize_dataset(sample))
    print(wait_time_by_department(sample))
    print(high_wait_cases(sample))
