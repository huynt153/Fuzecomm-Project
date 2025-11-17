"""
Baseline 2: Hard Threshold / If-Then Rules (instead of Fuzzy Logic)

Câu trả lời: Tại sao dùng Fuzzy Logic?
- Hard thresholds tạo ra decision boundaries cứng nhắc (hard boundaries)
- Dữ liệu gần ngưỡng có thể bị phân loại sai do noise
- Fuzzy logic cho phép "soft transitions" giữa các classes
- Fuzzy xử lý uncertainty tốt hơn (ví dụ: moisture = 59 vs 60)
- Trong IoT/sensor networks, noise là vấn đề thường xuyên
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

@dataclass
class HardThresholdSystem:
    """
    Hard threshold system using strict if-then rules
    Same logic as TomatoFuzzySystem but with hard boundaries
    """
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
        """
        Predict using HARD THRESHOLDS (strict inequalities)
        No fuzzy transitions - crisp boundaries only
        """
        moisture = row["Moisture"]
        ph = row["pH"]
        n = row["N"]
        temp = row["Temperature"]
        humidity = row["Humidity"]
        ndi = row["NDI_Label"]
        pdi = row["PDI_Label"]
        
        # HARD THRESHOLDS - strict inequalities
        # Class 0: Optimal - ALL conditions must be satisfied EXACTLY
        if (moisture >= 30 and moisture <= 60 and 
            ph >= 6.0 and ph <= 6.8 and 
            n >= 50 and n <= 100 and 
            temp >= 22 and temp <= 26 and 
            humidity >= 60 and humidity <= 70):
            return 0
        
        # Class 1: Nutrient deficiency - STRICT equality check
        elif str(ndi).strip().lower() == "high":
            return 1
        
        # Class 2: Fungal risk - ALL conditions STRICTLY required
        elif (str(pdi).strip().lower() == "high" and humidity > 80 and temp < 22):
            return 2
        
        # Class 3: Water deficit acidic - STRICT boundary
        elif moisture < 30 and ph < 5.8:
            return 3
        
        # Class 4: Water deficit alkaline - STRICT boundary
        elif moisture < 30 and ph > 7.5:
            return 4
        
        # Class 5: Acidic soil - STRICT boundary
        elif ph < 5.8 and moisture >= 30:
            return 5
        
        # Class 6: Alkaline soil - STRICT boundary
        elif ph > 7.5 and moisture >= 30:
            return 6
        
        # Class 7: Heat stress - STRICT boundary
        elif temp > 30 and humidity < 60:
            return 7
        
        else:
            # No class matched - ambiguous case
            return -1
    
    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_row(r) for _, r in df.iterrows()], dtype=int)
    
    @staticmethod
    def compare_with_fuzzy_explanation() -> dict:
        """
        Explain why Fuzzy Logic is better than Hard Thresholds
        """
        return {
            'hard_threshold_problems': [
                '1. Abrupt transitions: moisture=59.9 vs 60.0 treated completely differently',
                '2. No tolerance for noise: sensor noise can cause wrong classification',
                '3. Binary decisions: sample is either in class or not, no "partial membership"',
                '4. Poor handling of boundary cases: ambiguous samples get -1 (unclassified)',
                '5. Not robust: small sensor drift causes large errors'
            ],
            'fuzzy_logic_advantages': [
                '1. Smooth transitions: gradual membership functions (e.g., trimf, trapmf)',
                '2. Handles uncertainty: can say "70% optimal, 30% water_deficit"',
                '3. Noise tolerant: small variations do not cause classification flip',
                '4. Domain expert knowledge: membership functions reflect expert intuition',
                '5. Better for IoT: real-world sensors always have noise and uncertainty'
            ],
            'example_case': {
                'scenario': 'Moisture = 60.5, pH = 6.0, N = 50, Temp = 22, Humidity = 60',
                'hard_threshold_result': 'Class -1 (unclassified) because moisture > 60',
                'fuzzy_logic_result': 'Class 0 (optimal) with 0.95 membership - close enough!',
                'real_world_interpretation': 'Sensor reading likely has ±1% error, should still be optimal'
            },
            'answer': (
                "Fuzzy Logic is superior for IoT sensor applications because:\n"
                "- Real-world sensors have inherent noise and uncertainty\n"
                "- Hard thresholds create artificial boundaries that do not reflect reality\n"
                "- Fuzzy membership functions allow smooth transitions between classes\n"
                "- Better classification accuracy near decision boundaries\n"
                "- Aligns with how human experts reason about sensor data (gradual, not binary)"
            )
        }

@dataclass
class HardThresholdEvaluator:
    """Evaluate Hard Threshold Baseline"""
    cfg: dict
    
    def run(self, df: pd.DataFrame) -> dict:
        """Run evaluation and compare with Fuzzy"""
        # Predict using hard thresholds
        ht_system = HardThresholdSystem(self.cfg)
        y_pred = ht_system.predict_dataframe(df)
        y_true = df["ground_truth"].values
        
        # Filter valid predictions
        mask = (y_pred >= 0) & (y_pred <= 7)
        y_true_v = y_true[mask]
        y_pred_v = y_pred[mask]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true_v, y_pred_v)),
            'f1_macro': float(f1_score(y_true_v, y_pred_v, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true_v, y_pred_v, average='weighted', zero_division=0)),
            'unclassified_samples': int((y_pred == -1).sum()),
            'unclassified_percent': float((y_pred == -1).sum() / len(y_pred) * 100),
            'total_samples': len(y_pred),
            'valid_samples': int(mask.sum())
        }
        
        # Comparison explanation
        comparison = ht_system.compare_with_fuzzy_explanation()
        
        return {
            'metrics': metrics,
            'comparison': comparison,
            'classification_report': classification_report(
                y_true_v, y_pred_v, 
                target_names=ht_system.class_names,
                zero_division=0,
                output_dict=True
            )
        }
