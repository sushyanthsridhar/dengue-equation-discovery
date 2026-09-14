import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root: this script now lives one level down, in src/
EXTENDED_CSV = os.path.join(HERE, "extended_input.csv")
OUTPUT_CSV = os.path.join(HERE, "extended_input_normalized.csv")

ID_COLS = ["province", "year", "month", "week"]


def main():
    df = pd.read_csv(EXTENDED_CSV)
    df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)

    df["incidence"] = df["total_cases"] / df["population"] * 1e5

    df["dI_dt"] = df.groupby("province")["incidence"].diff()
    df["dI_dt"] = df.groupby("province")["dI_dt"].transform(lambda s: s.fillna(s.mean()))

    df["dI_dt_norm"] = df.groupby("province")["dI_dt"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-8)
    )

    feature_cols = [c for c in df.columns if c not in ID_COLS]

    scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
