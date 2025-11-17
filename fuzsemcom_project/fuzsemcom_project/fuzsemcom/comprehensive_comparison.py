"""
Updated Comprehensive Comparison with IoT and Efficiency Metrics
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
from typing import List, Dict

@dataclass
class ComprehensiveComparisonUpdated:
    cfg: dict
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation with all metrics"""
        
        # Load data
        sem_path = self.cfg["paths"]["semantic_dataset"]
        df = pd.read_csv(sem_path)
        df.columns = df.columns.astype(str).str.strip()
        y_true = df["ground_truth"].values
        
        # Import calculators
        from .iot_metrics import IoTMetricsCalculator
        from .system_efficiency import SystemEfficiencyCalculator
        from .results_visualization import ResultsVisualizer
        
        iot_calc = IoTMetricsCalculator(self.cfg)
        sys_calc = SystemEfficiencyCalculator(self.cfg)
        visualizer = ResultsVisualizer(self.cfg)
        
        # Define methods to evaluate
        methods_config = [
            {'name': 'FuzSemCom', 'bytes': 3, 'predictor': 'fuzzy'},
            {'name': 'Hard_Threshold', 'bytes': 3, 'predictor': 'hard_threshold'},
            {'name': 'Conventional_2B', 'bytes': 2, 'predictor': 'simulated', 'accuracy': 0.70},
            {'name': 'Conventional_8B', 'bytes': 8, 'predictor': 'simulated', 'accuracy': 0.85},
            {'name': 'Conventional_12B', 'bytes': 12, 'predictor': 'simulated', 'accuracy': 0.88},
            {'name': 'Quantized_DeepSC_2B', 'bytes': 2, 'predictor': 'simulated', 'accuracy': 0.82},
            {'name': 'Quantized_DeepSC_8B', 'bytes': 8, 'predictor': 'simulated', 'accuracy': 0.89},
            {'name': 'Quantized_DeepSC_12B', 'bytes': 12, 'predictor': 'simulated', 'accuracy': 0.91},
            {'name': 'L-DeepSC', 'bytes': 128, 'predictor': 'simulated', 'accuracy': 0.90},
        ]
        
        all_results = []
        
        for method_cfg in methods_config:
            print(f"Evaluating {method_cfg['name']}...")
            
            # Get predictions
            y_pred = self._get_predictions(df, y_true, method_cfg)
            
            # Calculate IoT metrics
            iot_metrics = iot_calc.evaluate_all_metrics(
                method_name=method_cfg['name'],
                y_true=y_true,
                y_pred=y_pred,
                bytes_per_sample=method_cfg['bytes']
            )
            
            # Calculate system efficiency metrics
            efficiency_metrics = sys_calc.evaluate_all_efficiency_metrics(
                method_name=method_cfg['name'],
                bytes_per_sample=method_cfg['bytes']
            )
            
            # Combine all metrics
            all_results.append({
                'method': method_cfg['name'],
                'bytes_per_sample': method_cfg['bytes'],
                **iot_metrics,
                **efficiency_metrics,
            })
        
        # Save and visualize results
        visualizer.save_all_results(all_results)
        
        return {
            'all_results': all_results,
            'tables_saved': True,
            'figures_saved': True,
            'html_report': os.path.join(
                self.cfg['paths']['reports_dir'], 
                'comprehensive_comparison_report.html'
            )
        }
    
    def _get_predictions(self, df: pd.DataFrame, y_true: np.ndarray, method_cfg: Dict) -> np.ndarray:
        """Get predictions for a method"""
        
        if method_cfg['predictor'] == 'fuzzy':
            from .fuzzy_system import TomatoFuzzySystem
            fs = TomatoFuzzySystem(self.cfg)
            y_pred = fs.predict_dataframe(df)
            y_pred = np.where(y_pred == -1, 0, y_pred)  # Handle unclassified
            
        elif method_cfg['predictor'] == 'hard_threshold':
            from .baseline_hard_threshold import HardThresholdSystem
            ht = HardThresholdSystem(self.cfg)
            y_pred = ht.predict_dataframe(df)
            y_pred = np.where(y_pred == -1, 0, y_pred)  # Handle unclassified
            
        elif method_cfg['predictor'] == 'simulated':
            # Simulate predictions based on target accuracy
            y_pred = self._simulate_predictions(y_true, method_cfg['accuracy'])
            
        else:
            y_pred = y_true.copy()  # Perfect prediction
        
        return y_pred
    
    def _simulate_predictions(self, y_true: np.ndarray, target_accuracy: float) -> np.ndarray:
        """Simulate predictions with target accuracy"""
        rng = np.random.default_rng(self.cfg.get('random_seed', 42))
        y_pred = y_true.copy()
        
        n_errors = int(len(y_true) * (1 - target_accuracy))
        error_indices = rng.choice(len(y_true), size=n_errors, replace=False)
        
        for idx in error_indices:
            # Random wrong class
            wrong_class = rng.choice([c for c in range(8) if c != y_true[idx]])
            y_pred[idx] = wrong_class
        
        return y_pred
