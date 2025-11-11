import argparse, yaml, os, logging

def main():
    parser = argparse.ArgumentParser(description="Run FuzSemCom pipeline")
    parser.add_argument("--config", default="config.yml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cfg["paths"]["logs_dir"], "pipeline.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Pipeline started.")

    from fuzsemcom.ground_truth import GroundTruthGenerator
    gt = GroundTruthGenerator(cfg)
    sem_path = gt.create_labels()
    logging.info(f"Semantic dataset saved to {sem_path}")

    from fuzsemcom.evaluate import FSEEvaluator
    ev = FSEEvaluator(cfg)
    ev_out = ev.run()
    logging.info(f"FSE evaluation saved to {ev_out['results_json']}")

    from fuzsemcom.compare import ComparisonAnalyzer
    cmp = ComparisonAnalyzer(cfg)
    cmp_out = cmp.run()
    logging.info(f"Comparison report saved to {cmp_out['report_html']}")

    print("Done. See results/ for reports and figures.")

if __name__ == "__main__":
    main()
