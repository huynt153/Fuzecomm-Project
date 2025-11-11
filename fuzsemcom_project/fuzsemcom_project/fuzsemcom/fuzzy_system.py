from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class TomatoFuzzySystem:
    cfg: dict

    class_names = [
        "optimal",
        "nutrient_deficiency",
        "fungal_risk",
        "water_deficit_acidic",
        "water_deficit_alkaline",
        "acidic_soil",
        "alkaline_soil",
        "heat_stress",
    ]

    def predict_row(self, row: pd.Series) -> int:
        moisture = row["Moisture"]
        ph = row["pH"]
        n = row["N"]
        temp = row["Temperature"]
        humidity = row["Humidity"]
        ndi = row["NDI_Label"]
        pdi = row["PDI_Label"]

        if (30 <= moisture <= 60 and 6.0 <= ph <= 6.8 and 50 <= n <= 100 and 22 <= temp <= 26 and 60 <= humidity <= 70):
            return 0
        elif str(ndi).strip().lower() == "high":
            return 1
        elif (str(pdi).strip().lower() == "high") and (humidity > 80) and (temp < 22):
            return 2
        elif (moisture < 30) and (ph < 5.8):
            return 3
        elif (moisture < 30) and (ph > 7.5):
            return 4
        elif (ph < 5.8) and (moisture >= 30):
            return 5
        elif (ph > 7.5) and (moisture >= 30):
            return 6
        elif (temp > 30) and (humidity < 60):
            return 7
        else:
            return -1

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_row(r) for _, r in df.iterrows()], dtype=int)
