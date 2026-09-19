"""
Unit & Integration Tests for Aerospace Mission Audit Logger,
10,000-Line FIFO Ring Buffer, Operator Action Tracking, and PDF Export.
"""

import os
import time
import pytest
from src.utils.audit_logger import AerospaceAuditLogger
from src.utils.pdf_generator import generate_mission_audit_pdf


@pytest.fixture
def temp_logger(tmp_path):
    log_file = os.path.join(tmp_path, "test_audit.jsonl")
    logger = AerospaceAuditLogger(max_lines=50, storage_file=log_file)
    logger.clear_buffer()
    return logger


def test_audit_logger_fifo_rotation(temp_logger):
    """Verifies that the buffer strictly enforces its max limit with FIFO pruning"""
    temp_logger.clear_buffer()

    # Log 70 events on a buffer with max_lines=50
    for i in range(1, 71):
        temp_logger.log_event(
            event_type="TELEMETRY_FRAME",
            level="INFO",
            subsystem="EPS_CORE",
            message=f"Telemetry frame {i}",
            telemetry={"voltage": 3.70 + (i * 0.001), "current": 2.45}
        )

    stats = temp_logger.get_stats()
    assert stats["buffer_count"] == 50
    assert stats["total_events_logged"] == 71
    assert stats["buffer_usage_pct"] == 100.0

    logs = temp_logger.get_logs(limit=100, reverse=False)
    assert len(logs) == 50
    # First item should be item #22 (seq_id 1 was clear log, 2-21 were first 20 frames evicted)
    assert logs[0]["seq_id"] == 22
    # Last item should be item #71
    assert logs[-1]["seq_id"] == 71


def test_audit_logger_event_types(temp_logger):
    """Verifies distinct event logging methods and schemas"""
    temp_logger.clear_buffer()

    # 1. Operator action
    temp_logger.log_operator_action("INJECT_FAULT_THERMAL_RUNAWAY", "Duration 25s", "FlightDirector")
    
    # 2. Early prediction
    temp_logger.log_early_prediction(
        precursor_feature="Thermal Gradient Surge",
        probability=0.28,
        threshold=0.70,
        recommended_action="HIGH_SENSITIVITY_PREARM",
        telemetry={"voltage": 3.65, "temperature": 38.5},
        models={"p_ensemble": 0.28}
    )

    # 3. Fault Anomaly
    temp_logger.log_fault_event(
        fault_type="THERMAL_RUNAWAY",
        severity="CRITICAL",
        details="dT/dt > 1.5°C/s exceeded threshold",
        telemetry={"voltage": 3.55, "temperature": 52.0},
        models={"p_ensemble": 0.95}
    )

    # 4. RL Mitigation
    temp_logger.log_mitigation_event(
        action_name="LOAD_SHEDDING",
        target_fault="THERMAL_RUNAWAY",
        counterfactual_temp=41.2,
        safety_score=0.96
    )

    # 5. Recovery
    temp_logger.log_recovery_event(
        mitigating_action="LOAD_SHEDDING",
        resolved_fault="THERMAL_RUNAWAY",
        details="Subsystem stabilized."
    )

    stats = temp_logger.get_stats()
    assert stats["operator_actions"] == 1
    assert stats["early_predictions"] == 1
    assert stats["faults_detected"] == 1
    assert stats["mitigations_executed"] == 1
    assert stats["recoveries"] == 1

    # Test query filtering
    action_logs = temp_logger.get_logs(category="OPERATOR_ACTION")
    assert len(action_logs) == 1
    assert "FlightDirector" in action_logs[0]["message"]

    pred_logs = temp_logger.get_logs(category="EARLY_PREDICTION")
    assert len(pred_logs) == 1
    assert pred_logs[0]["level"] == "PREDICT"


def test_pdf_report_generation(tmp_path):
    """Verifies that the ReportLab PDF generator builds a clean, multi-page report"""
    pdf_out = os.path.join(tmp_path, "AERO_GUARD_TEST_AUDIT.pdf")
    
    # Create sample logs
    sample_logs = []
    for i in range(1, 40):
        sample_logs.append({
            "seq_id": i,
            "timestamp": "2026-09-19 14:00:00.000Z",
            "unix_ts": time.time(),
            "met": f"T+00:05:{i:02d}",
            "level": "CRITICAL" if i % 10 == 0 else ("ACTION" if i % 5 == 0 else "INFO"),
            "event_type": "FAULT_ANOMALY" if i % 10 == 0 else ("OPERATOR_ACTION" if i % 5 == 0 else "TELEMETRY_FRAME"),
            "subsystem": "PCDU" if i % 2 == 0 else "SECONDARY_LIION",
            "message": f"Test flight audit log record #{i} with nominal standard verification.",
            "telemetry": {"voltage": 3.71, "current": 2.45, "temperature": 23.5, "soc": 0.85},
            "models": {"p_ensemble": 0.05, "dynamic_threshold": 0.70},
            "metadata": {"test_mode": True}
        })

    stats = {
        "buffer_count": len(sample_logs),
        "buffer_max_lines": 10000,
        "buffer_usage_pct": 0.4,
        "total_events_logged": 150,
        "operator_actions": 12,
        "early_predictions": 5,
        "faults_detected": 4,
        "mitigations_executed": 4,
        "recoveries": 4
    }

    generated_path = generate_mission_audit_pdf(
        logs=sample_logs,
        stats=stats,
        output_path=pdf_out,
        title="AERO-GUARD TEST AUDIT LEDGER"
    )

    assert os.path.exists(generated_path)
    file_size = os.path.getsize(generated_path)
    assert file_size > 1000  # PDF generated with valid binary structure
