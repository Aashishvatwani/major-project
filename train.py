"""
Training Script for Satellite Telemetry HITL Pipeline
Executes end-to-end dataset generation, physics feature extraction,
ensemble ML training (Random Forest + XGBoost + Extra Trees),
probability calibration, and Reinforcement Learning mitigation policy training.
"""

import argparse
import sys
from src.models.model_trainer import ModelTrainer
from src.utils.logger import setup_aerospace_logger


def main():
    parser = argparse.ArgumentParser(description="Train Satellite Anomaly Detection Ensemble & RL Agent")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--duration-minutes", type=float, default=360.0, help="Simulated orbital telemetry duration in minutes")
    parser.add_argument("--output-dir", type=str, default="saved_models", help="Directory to save trained models")
    args = parser.parse_args()

    logger = setup_aerospace_logger("TrainPipeline")
    logger.info("================================================================")
    logger.info(" SATELLITE TELEMETRY HITL PIPELINE - MODEL & RL POLICY TRAINING ")
    logger.info("================================================================")

    trainer = ModelTrainer(config_path=args.config)
    results = trainer.train_full_pipeline(
        duration_minutes=args.duration_minutes,
        output_dir=args.output_dir
    )

    logger.info("Training pipeline finished successfully.")
    logger.info(f"Artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
