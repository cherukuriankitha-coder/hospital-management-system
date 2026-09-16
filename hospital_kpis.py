"""Hospital operations KPI calculations."""
import pandas as pd


def calculate_kpis(df: pd.DataFrame) -> dict:
    result = {"patient_records": int(len(df))}
    if "length_of_stay" in df:
        result["avg_length_of_stay"] = float(pd.to_numeric(df["length_of_stay"], errors="coerce").mean())
    if "readmitted" in df:
        readmitted = df["readmitted"].astype(str).str.lower().isin(["1", "true", "yes"])
        result["readmission_rate_pct"] = float(readmitted.mean() * 100)
    if "department" in df:
        result["patients_by_department"] = df["department"].value_counts().to_dict()
    return result
