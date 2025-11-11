from __future__ import annotations
import os
import pandas as pd
from dataclasses import dataclass

@dataclass
class GroundTruthGenerator:
    cfg: dict

    def _paths(self):
        p = self.cfg["paths"]
        return p["data_raw"], p["semantic_dataset"]

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.astype(str).str.strip()
        return df

    @staticmethod
    def _required_columns():
        return ["Moisture", "pH", "N", "Temperature", "Humidity", "NDI_Label", "PDI_Label"]

    @staticmethod
    def create_ground_truth_labels(df: pd.DataFrame) -> list[int]:
        labels = []
        for _, row in df.iterrows():
            moisture = row["Moisture"]
            ph = row["pH"]
            n = row["N"]
            temp = row["Temperature"]
            humidity = row["Humidity"]
            ndi = row["NDI_Label"]
            pdi = row["PDI_Label"]

            if (30 <= moisture <= 60 and 6.0 <= ph <= 6.8 and 50 <= n <= 100 and 22 <= temp <= 26 and 60 <= humidity <= 70):
                labels.append(0)
            elif str(ndi).strip().lower() == "high":
                labels.append(1)
            elif (str(pdi).strip().lower() == "high") and (humidity > 80) and (temp < 22):
                labels.append(2)
            elif (moisture < 30) and (ph < 5.8):
                labels.append(3)
            elif (moisture < 30) and (ph > 7.5):
                labels.append(4)
            elif (ph < 5.8) and (moisture >= 30):
                labels.append(5)
            elif (ph > 7.5) and (moisture >= 30):
                labels.append(6)
            elif (temp > 30) and (humidity < 60):
                labels.append(7)
            else:
                labels.append(-1)
        return labels

    def create_labels(self) -> str:
        data_raw, semantic_path = self._paths()
        if not os.path.exists(data_raw):
            raise FileNotFoundError(f"Missing dataset at {data_raw}. Please place the CSV in data/.")
        df = pd.read_csv(data_raw)
        df = self._normalize_columns(df).dropna(subset=["Moisture","pH","N","Temperature","Humidity"])

        missing = set(self._required_columns()) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns in dataset: {missing}")

        df["ground_truth"] = self.create_ground_truth_labels(df)
        df = df[df["ground_truth"] != -1].reset_index(drop=True)

        os.makedirs(os.path.dirname(semantic_path), exist_ok=True)
        df.to_csv(semantic_path, index=False)
        return semantic_path
