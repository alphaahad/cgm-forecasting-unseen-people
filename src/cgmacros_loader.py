"""Load CGMacros into one tidy table (one row per minute per participant)."""
import glob
import os
import re

import pandas as pd

DATA = os.environ.get("MBZUAI_DATA", r"D:\MBZUAI_data")
ROOT = os.path.join(DATA, "cgmacros", "extracted", "CGMacros")


def hba1c_group(a1c: float) -> str:
    """ADA cutoffs on HbA1c (%): <5.7 healthy, 5.7-6.4 pre-diabetes, >=6.5 T2D.
    NOTE: the dictionary says mmol/mol but the values (4.6-8.5) are clearly %.
    This is a proxy label derived by us; verify against the paper's group counts."""
    if a1c < 5.7:
        return "healthy"
    if a1c < 6.5:
        return "prediabetes"
    return "t2d"


def load_bio() -> pd.DataFrame:
    bio = pd.read_csv(os.path.join(ROOT, "bio.csv"))
    bio.columns = [c.strip() for c in bio.columns]
    bio = bio.rename(columns={"A1c PDL (Lab)": "a1c", "Fasting GLU - PDL (Lab)": "fasting_glucose"})
    bio["group"] = bio["a1c"].map(hba1c_group)
    return bio[["subject", "Age", "Gender", "BMI", "a1c", "fasting_glucose", "Insulin", "group"]]


def load_all() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(ROOT, "CGMacros-*", "CGMacros-*.csv"))):
        sid = int(re.search(r"CGMacros-(\d+)\.csv$", f).group(1))
        d = pd.read_csv(f)
        d = d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")])
        d.columns = [c.strip() for c in d.columns]
        d = d.rename(columns={
            "Libre GL": "libre", "Dexcom GL": "dexcom", "Calories (Activity)": "cal_activity",
            "Meal Type": "meal_type", "Amount Consumed": "pct_consumed", "Image path": "image_path",
            "Calories": "meal_kcal", "Carbs": "carbs", "Protein": "protein", "Fat": "fat", "Fiber": "fiber",
        })
        d["Timestamp"] = pd.to_datetime(d["Timestamp"])
        d.insert(0, "subject", sid)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df.merge(load_bio(), on="subject", how="left")


if __name__ == "__main__":
    df = load_all()
    out = os.path.join(DATA, "processed")
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, "cgmacros_all.parquet"), index=False)
    print(df.shape, df["subject"].nunique(), "participants")
    print(load_bio()["group"].value_counts())
