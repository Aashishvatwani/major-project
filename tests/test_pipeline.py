"""
Integration Tests for End-to-End Satellite Telemetry HITL Pipeline
"""

import pytest
import os
import numpy as np
from src.models.model_trainer import ModelTrainer
from src.pipeline.hitl_pipeline import SatelliteHITLPipeline


@pytest.fixture(scope="session")
def trained_models(tmp_path_factory):
    """Fixture ensuring model artifacts are trained once for integration testing"""
    models_dir = str(tmp_path_factory.mktemp("test_models"))
    trainer = ModelTrainer()
    trainer.ensemble.n_estimators = 20
    trainer.train_full_pipeline(duration_minutes=15.0, output_dir=models_dir, rl_episodes=3)
    return models_dir


def test_end_to_end_pipeline_flow(trained_models):
    """Verify complete single-sample streaming flow through features, ML, RL, Safety, and Virtual Arduino"""
    pipeline = SatelliteHITLPipeline(
        models_dir=trained_models,
        serial_port="VIRTUAL",
        enable_hardware=True
    )

    # 1. Normal Sample
    sample_normal = {
        "timestamp": 100.0,
        "voltage": 3.72,
        "current": 2.45,
        "temperature": 23.5,
        "soc": 0.85,
        "is_eclipse": 0,
        "anomaly_label": 0
    }

    res_normal = pipeline.process_single_sample(sample_normal)
    assert "p_ensemble" in res_normal
    assert "rl_action_name" in res_normal
    assert "hardware_pin13_led" in res_normal
    assert res_normal["final_decision"] == 0
    assert res_normal["hardware_pin13_led"] == 0

    # 2. Critical Safety Violation Sample (e.g. Extreme Over-temperature)
    sample_critical = {
        "timestamp": 100.5,
        "voltage": 3.65,
        "current": 3.80,
        "temperature": 75.0,  # Violates 65°C rule
        "soc": 0.70,
        "is_eclipse": 0,
        "anomaly_label": 1
    }

    res_critical = pipeline.process_single_sample(sample_critical)
    assert res_critical["safety_override_active"] is True
    assert res_critical["final_decision"] == 1
    assert res_critical["hardware_pin13_led"] == 1

    pipeline.close()
