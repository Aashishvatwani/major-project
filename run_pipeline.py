"""
Master CLI Runner for Satellite Telemetry HITL Pipeline
Supports:
1. Real-time streaming simulation with live terminal telemetry rendering and PySerial/Virtual Arduino Pin 13 actuation
2. Batch evaluation over NASA or Synthetic telemetry datasets with full classification metrics
3. Interactive fault injection testing
"""

import argparse
import time
import sys
import os
import numpy as np
import pandas as pd
from typing import Optional

from src.pipeline.hitl_pipeline import SatelliteHITLPipeline
from src.ingestion.data_loader import TelemetryDataLoader
from src.utils.logger import setup_aerospace_logger
from src.utils.metrics import calculate_telemetry_metrics

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def run_streaming_mode(
    models_dir: str = "saved_models",
    port: str = "AUTO",
    duration_sec: float = 30.0,
    speed: float = 1.0,
    inject_fault: Optional[str] = None
):
    """Runs interactive real-time telemetry streaming"""
    logger = setup_aerospace_logger("PipelineStream")
    console = Console() if HAS_RICH else None

    # Verify models exist
    if not os.path.exists(os.path.join(models_dir, "ensemble_model.joblib")):
        logger.info("[!] Saved models not found. Initiating quick training run...")
        from src.models.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_full_pipeline(duration_minutes=180.0, output_dir=models_dir)

    pipeline = SatelliteHITLPipeline(models_dir=models_dir, serial_port=port, enable_hardware=True)
    loader = TelemetryDataLoader("data/raw/synthetic_telemetry.csv")
    df = loader.load_or_generate_dataset(duration_minutes=120.0)
    pipeline.stream.set_dataset(df)

    if inject_fault:
        pipeline.stream.inject_fault(inject_fault, duration_sec=15.0)
        logger.info(f"[*] Injected fault scenario: '{inject_fault}' for 15s")

    logger.info("================================================================")
    logger.info(" STARTING SATELLITE TELEMETRY HITL STREAMING SIMULATION ")
    logger.info("================================================================")

    start_time = time.time()
    count = 0
    y_trues, y_preds, y_probs, latencies = [], [], [], []

    try:
        while time.time() - start_time < duration_sec:
            sample = pipeline.stream.get_next_sample()
            if not sample:
                break

            record = pipeline.process_single_sample(sample)
            count += 1

            y_trues.append(record["ground_truth"])
            y_preds.append(record["final_decision"])
            y_probs.append(record["p_ensemble"])
            latencies.append(record["latency_ms"])

            # Console status line
            v = record["voltage"]
            i = record["current"]
            t = record["temperature"]
            p_ens = record["p_ensemble"]
            dec = record["final_decision"]
            pin13 = record["hardware_pin13_led"]
            rl_act = record["rl_action_name"]
            status = record["final_status"]

            led_symbol = "[ON]" if pin13 == 1 else "[OFF]"
            alert_tag = f"[bold red]ANOMALY ({p_ens:.2f})[/bold red]" if dec == 1 else f"[green]NOMINAL ({p_ens:.2f})[/green]"

            if HAS_RICH and console:
                console.print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"V: [cyan]{v:.3f}V[/cyan] | I: [yellow]{i:.3f}A[/yellow] | T: [magenta]{t:.1f}°C[/magenta] | "
                    f"RL: [blue]{rl_act}[/blue] | Status: {alert_tag} | Pin 13: {led_symbol} ({record['latency_ms']:.1f}ms)"
                )
            else:
                print(f"[{time.strftime('%H:%M:%S')}] V:{v:.3f}V I:{i:.3f}A T:{t:.1f}C | P_ens:{p_ens:.2f} | Dec:{dec} | Pin13:{pin13} | RL:{rl_act}")

            time.sleep(max(0.01, 0.4 / speed))

    except KeyboardInterrupt:
        logger.info("\nStreaming interrupted by user.")
    finally:
        pipeline.close()

    metrics = calculate_telemetry_metrics(np.array(y_trues), np.array(y_preds), np.array(y_probs), latencies)
    print("\n" + "=" * 60)
    print(f"STREAMING SUMMARY: Processed {count} frames")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} | F1-Score: {metrics['f1_score']:.4f}")
    print(f"Mean Inference Latency: {metrics['latency_ms']['mean']:.2f} ms")
    print("=" * 60)


def run_batch_evaluation(models_dir: str = "saved_models", data_path: Optional[str] = None):
    """Runs batch evaluation over a dataset"""
    logger = setup_aerospace_logger("BatchEvaluation")

    if not os.path.exists(os.path.join(models_dir, "ensemble_model.joblib")):
        logger.error("Saved models not found. Please run `python train.py` first.")
        sys.exit(1)

    pipeline = SatelliteHITLPipeline(models_dir=models_dir, enable_hardware=False)
    loader = TelemetryDataLoader(data_path or "data/raw/synthetic_telemetry.csv")
    df = loader.load_or_generate_dataset(duration_minutes=180.0)

    logger.info(f"Evaluating dataset with {len(df)} samples...")
    y_trues, y_preds, y_probs, latencies = [], [], [], []

    for _, row in df.iterrows():
        sample = row.to_dict()
        res = pipeline.process_single_sample(sample)
        y_trues.append(res["ground_truth"])
        y_preds.append(res["final_decision"])
        y_probs.append(res["p_ensemble"])
        latencies.append(res["latency_ms"])

    metrics = calculate_telemetry_metrics(np.array(y_trues), np.array(y_preds), np.array(y_probs), latencies)

    print("\n" + "=" * 60)
    print(" BATCH EVALUATION BENCHMARK RESULTS ")
    print("=" * 60)
    print(f"Total Samples Evaluated: {len(df)}")
    print(f"Ground Truth Anomalies:  {sum(y_trues)} ({sum(y_trues)/len(df):.2%})")
    print(f"ROC-AUC Score:           {metrics['roc_auc']:.4f}")
    print(f"PR-AUC Score:            {metrics['pr_auc']:.4f}")
    print(f"F1-Score:                {metrics['f1_score']:.4f}")
    print(f"Precision:               {metrics['precision']:.4f}")
    print(f"Recall:                  {metrics['recall']:.4f}")
    print(f"False Alarm Rate (FAR):  {metrics['false_alarm_rate']:.4f}")
    print(f"Average Latency:         {metrics['latency_ms']['mean']:.2f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Satellite Telemetry HITL Pipeline CLI")
    parser.add_argument("--mode", type=str, choices=["stream", "eval"], default="stream", help="Execution mode")
    parser.add_argument("--models-dir", type=str, default="saved_models", help="Directory with trained models")
    parser.add_argument("--port", type=str, default="AUTO", help="Serial port for Arduino ('AUTO', 'COM3', 'VIRTUAL')")
    parser.add_argument("--duration", type=float, default=25.0, help="Duration in seconds for streaming simulation")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--inject-fault", type=str, default=None, choices=["thermal_runaway", "short_circuit", "undervoltage", "high_impedance", "sensor_drift"], help="Fault to inject")
    parser.add_argument("--data-path", type=str, default=None, help="Custom CSV telemetry dataset path")

    args = parser.parse_args()

    if args.mode == "stream":
        run_streaming_mode(
            models_dir=args.models_dir,
            port=args.port,
            duration_sec=args.duration,
            speed=args.speed,
            inject_fault=args.inject_fault
        )
    elif args.mode == "eval":
        run_batch_evaluation(models_dir=args.models_dir, data_path=args.data_path)


if __name__ == "__main__":
    main()
