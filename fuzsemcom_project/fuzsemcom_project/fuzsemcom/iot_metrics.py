"""
IoT Application Metrics Module
Calculate: Water Saved, False-Action Rate, Age-of-Information, Energy per Correct Action
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

@dataclass
class IoTMetricsCalculator:
    """
    Calculate IoT-specific metrics for smart agriculture
    
    Domain assumptions:
    - Optimal class (0): No action needed → Water saved
    - Wrong classification → Wrong action → Water wasted
    - Each action costs water and energy
    """
    
    cfg: dict
    
    # Water consumption per action (liters per day per plant)
    WATER_PER_ACTION = {
        0: 0,      # optimal: no extra water needed
        1: 2.5,    # nutrient_deficiency: fertilizer water
        2: 3.0,    # fungal_risk: fungicide water
        3: 5.0,    # water_deficit_acidic: irrigation + pH correction
        4: 5.0,    # water_deficit_alkaline: irrigation + pH correction
        5: 2.0,    # acidic_soil: pH correction water
        6: 2.0,    # alkaline_soil: pH correction water
        7: 4.0,    # heat_stress: cooling water
    }
    
    # Energy cost per action (Joules per action)
    ENERGY_PER_ACTION = {
        0: 0,      # optimal: no action
        1: 50,     # nutrient_deficiency: pump + mixer
        2: 60,     # fungal_risk: sprayer
        3: 100,    # water_deficit_acidic: pump + pH adjuster
        4: 100,    # water_deficit_alkaline: pump + pH adjuster
        5: 40,     # acidic_soil: pH adjuster
        6: 40,     # alkaline_soil: pH adjuster
        7: 80,     # heat_stress: cooling system
    }
    
    def calculate_water_saved(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        baseline_system: str = "always_water"
    ) -> Dict:
        """
        Calculate water saved compared to baseline
        
        Baseline options:
        - "always_water": Traditional system waters all plants daily (worst case)
        - "schedule": Scheduled watering (moderate)
        - "optimal": Perfect classification (best case, for reference)
        """
        n_samples = len(y_true)
        
        # Calculate water used by predicted system
        water_used_pred = sum(self.WATER_PER_ACTION[int(p)] for p in y_pred)
        
        # Calculate water used by baseline systems
        if baseline_system == "always_water":
            # Assume average watering action (class 3)
            baseline_water = n_samples * self.WATER_PER_ACTION[3]
        elif baseline_system == "schedule":
            # Scheduled watering: 50% of samples get water
            baseline_water = n_samples * 0.5 * self.WATER_PER_ACTION[3]
        elif baseline_system == "optimal":
            # Perfect classification
            baseline_water = sum(self.WATER_PER_ACTION[int(t)] for t in y_true)
        else:
            baseline_water = n_samples * self.WATER_PER_ACTION[3]
        
        # Calculate optimal water (ground truth)
        water_optimal = sum(self.WATER_PER_ACTION[int(t)] for t in y_true)
        
        # Water saved compared to baseline
        water_saved_liters = baseline_water - water_used_pred
        water_saved_percent = (water_saved_liters / baseline_water * 100) if baseline_water > 0 else 0
        
        # Water efficiency (how close to optimal)
        water_efficiency = (1 - abs(water_used_pred - water_optimal) / baseline_water * 100) if baseline_water > 0 else 0
        
        return {
            'water_used_liters': float(water_used_pred),
            'baseline_water_liters': float(baseline_water),
            'optimal_water_liters': float(water_optimal),
            'water_saved_liters': float(water_saved_liters),
            'water_saved_percent': float(water_saved_percent),
            'water_efficiency_percent': float(water_efficiency),
            'baseline_system': baseline_system
        }
    
    def calculate_false_action_rate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict:
        """
        Calculate false action rate
        
        False actions:
        1. False positive action: Predicted action when no action needed (y_true=0, y_pred≠0)
        2. Wrong action: Wrong action type (y_true≠0, y_pred≠0, y_true≠y_pred)
        3. Missed action: No action when action needed (y_true≠0, y_pred=0)
        """
        n_samples = len(y_true)
        
        # Type 1: False positive actions (over-watering)
        false_positive = np.sum((y_true == 0) & (y_pred != 0))
        
        # Type 2: Wrong action type
        wrong_action = np.sum((y_true != 0) & (y_pred != 0) & (y_true != y_pred))
        
        # Type 3: Missed action (under-watering)
        missed_action = np.sum((y_true != 0) & (y_pred == 0))
        
        # Total false actions
        total_false_actions = false_positive + wrong_action + missed_action
        false_action_rate = total_false_actions / n_samples * 100
        
        # Calculate cost of false actions
        false_action_cost_water = 0.0
        false_action_cost_energy = 0.0
        
        for t, p in zip(y_true, y_pred):
            if t != p:
                # Water wasted = |water_used - water_needed|
                false_action_cost_water += abs(
                    self.WATER_PER_ACTION[int(p)] - self.WATER_PER_ACTION[int(t)]
                )
                # Energy wasted = energy_used (if wrong action)
                if p != 0:
                    false_action_cost_energy += self.ENERGY_PER_ACTION[int(p)]
        
        return {
            'false_positive_actions': int(false_positive),
            'wrong_action_type': int(wrong_action),
            'missed_actions': int(missed_action),
            'total_false_actions': int(total_false_actions),
            'false_action_rate_percent': float(false_action_rate),
            'false_action_water_cost_liters': float(false_action_cost_water),
            'false_action_energy_cost_joules': float(false_action_cost_energy),
        }
    
    def calculate_age_of_information(
        self,
        method_name: str,
        bytes_per_sample: int,
        transmission_rate_kbps: float = 250  # LoRa/ZigBee typical rate
    ) -> Dict:
        """
        Calculate Age of Information (AoI)
        
        AoI = sensing_time + processing_time + transmission_time + decode_time
        """
        # Sensing time (sensor reading): ~10-50ms
        sensing_time_ms = 30
        
        # Processing time (depends on method)
        processing_times = {
            'fuzsemcom': 2.0,        # Fuzzy rules: very fast
            'hard_threshold': 1.5,   # Hard rules: fastest
            'conventional': 5.0,     # Delta + Huffman: moderate
            'quantized_deepsc': 25.0, # Neural network inference: slower
            'l_deepsc': 50.0,        # Full DeepSC: slowest
        }
        processing_time_ms = processing_times.get(method_name.lower(), 10.0)
        
        # Transmission time
        bits = bytes_per_sample * 8
        transmission_time_ms = bits / (transmission_rate_kbps * 1000) * 1000  # Convert to ms
        
        # Decode time (at receiver)
        decode_times = {
            'fuzsemcom': 1.0,        # Symbol decode + Hamming
            'hard_threshold': 0.5,   # Direct decode
            'conventional': 3.0,     # Huffman decode
            'quantized_deepsc': 15.0, # Dequantize + decode
            'l_deepsc': 30.0,        # Full decode
        }
        decode_time_ms = decode_times.get(method_name.lower(), 5.0)
        
        # Total AoI
        total_aoi_ms = sensing_time_ms + processing_time_ms + transmission_time_ms + decode_time_ms
        
        return {
            'sensing_time_ms': float(sensing_time_ms),
            'processing_time_ms': float(processing_time_ms),
            'transmission_time_ms': float(transmission_time_ms),
            'decode_time_ms': float(decode_time_ms),
            'total_aoi_ms': float(total_aoi_ms),
            'transmission_rate_kbps': float(transmission_rate_kbps),
        }
    
    def calculate_energy_per_correct_action(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        communication_energy_uj: float
    ) -> Dict:
        """
        Calculate energy per correct action
        
        Energy = Communication energy + Action energy
        Only count energy for CORRECT actions
        """
        n_samples = len(y_true)
        correct_mask = (y_true == y_pred)
        n_correct = correct_mask.sum()
        
        # Communication energy (for all samples)
        total_comm_energy = communication_energy_uj * n_samples
        
        # Action energy (only for predicted actions)
        total_action_energy = sum(
            self.ENERGY_PER_ACTION[int(p)] * 1000  # Convert J to µJ
            for p in y_pred
        )
        
        # Total energy
        total_energy_uj = total_comm_energy + total_action_energy
        
        # Energy per correct action
        if n_correct > 0:
            energy_per_correct_uj = total_energy_uj / n_correct
        else:
            energy_per_correct_uj = float('inf')
        
        # Energy efficiency (correct actions / total energy)
        energy_efficiency = n_correct / (total_energy_uj / 1000)  # correct per mJ
        
        return {
            'total_samples': int(n_samples),
            'correct_actions': int(n_correct),
            'accuracy': float(n_correct / n_samples * 100),
            'communication_energy_uj': float(total_comm_energy),
            'action_energy_uj': float(total_action_energy),
            'total_energy_uj': float(total_energy_uj),
            'energy_per_correct_action_uj': float(energy_per_correct_uj),
            'energy_efficiency_correct_per_mJ': float(energy_efficiency),
        }
    
    def evaluate_all_metrics(
        self,
        method_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bytes_per_sample: int,
        energy_per_bit_uj: float = 0.5
    ) -> Dict:
        """Evaluate all IoT metrics for a method"""
        
        # Communication energy
        comm_energy_uj = bytes_per_sample * 8 * energy_per_bit_uj
        
        # Calculate all metrics
        water_metrics = self.calculate_water_saved(y_true, y_pred, baseline_system="always_water")
        false_action_metrics = self.calculate_false_action_rate(y_true, y_pred)
        aoi_metrics = self.calculate_age_of_information(method_name, bytes_per_sample)
        energy_metrics = self.calculate_energy_per_correct_action(y_true, y_pred, comm_energy_uj)
        
        return {
            'method': method_name,
            'bytes_per_sample': bytes_per_sample,
            'water_metrics': water_metrics,
            'false_action_metrics': false_action_metrics,
            'aoi_metrics': aoi_metrics,
            'energy_metrics': energy_metrics,
        }
