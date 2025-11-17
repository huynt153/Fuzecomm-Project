"""
Updated pipeline with IoT and Efficiency metrics
"""

import argparse
import yaml
import os
import logging

def main():
    parser = argparse.ArgumentParser(description="Run Comprehensive FuzSemCom Pipeline")
    parser.add_argument("--config", default="config.yml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cfg["paths"]["logs_dir"], "comprehensive_pipeline.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    print("="*60)
    print("🚀 FuzSemCom Comprehensive Evaluation Pipeline")
    print("="*60)
    
    # Step 1: Generate ground truth
    logging.info("Step 1: Generating ground truth labels...")
    print("\n[1/4] Generating ground truth labels...")
    from fuzsemcom.ground_truth import GroundTruthGenerator
    gt = GroundTruthGenerator(cfg)
    sem_path = gt.create_labels()
    logging.info(f"Semantic dataset saved to {sem_path}")
    print(f"✅ Semantic dataset: {sem_path}")
    
    # Step 2: Evaluate FuzSemCom
    logging.info("Step 2: Evaluating FuzSemCom...")
    print("\n[2/4] Evaluating FuzSemCom...")
