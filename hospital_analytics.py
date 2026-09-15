"""Simple hospital operations analytics project using pandas."""

from __future__ import annotations

import pandas as pd


def clean_patient_visits(visits: pd.DataFrame) -> pd.DataFrame:
    """Standardize visit records and remove invalid/duplicate rows."""
    df = visits.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates()

    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")

    if "wait_minutes" in df.columns:
        df["wait_minutes"] = pd.to_numeric(df["wait_minutes"], errors="coerce")
        df.loc[df["wait_minutes"] < 0, "wait_minutes"] = pd.NA

    return df


def department_summary(visits: pd.DataFrame) -> pd.DataFrame:
    """Create a department-level KPI table."""
    required = {"department", "patient_id", "wait_minutes"}
    missing = required.difference(visits.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        visits.groupby("department", dropna=False)
        .agg(
            total_visits=("patient_id", "count"),
            unique_patients=("patient_id", "nunique"),
            avg_wait_minutes=("wait_minutes", "mean"),
            max_wait_minutes=("wait_minutes", "max"),
        )
        .reset_index()
        .round({"avg_wait_minutes": 1})
        .sort_values("total_visits", ascending=False)
    )


def daily_visit_trend(visits: pd.DataFrame) -> pd.DataFrame:
    """Return daily patient-visit volume."""
    if "visit_date" not in visits.columns:
        raise ValueError("visit_date column is required")
    return (
        visits.dropna(subset=["visit_date"])
        .assign(date=lambda x: x["visit_date"].dt.date)
        .groupby("date")
        .size()
        .rename("visits")
        .reset_index()
    )


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "patient_id": [1001, 1002, 1003, 1001, 1004],
            "department": ["Emergency", "Cardiology", "Emergency", "Cardiology", "Emergency"],
            "visit_date": ["2026-09-01", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-03"],
            "wait_minutes": [42, 18, 55, 25, 31],
        }
    )

    cleaned = clean_patient_visits(sample)
    print("Department KPIs:\n", department_summary(cleaned))
    print("\nDaily trend:\n", daily_visit_trend(cleaned))
