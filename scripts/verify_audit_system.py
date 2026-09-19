"""
Verification script for Aerospace Audit Logger, 10,000-line FIFO buffer, and PDF generator.
"""
import os
import sys
import time
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.audit_logger import AerospaceAuditLogger
from src.utils.pdf_generator import generate_mission_audit_pdf


def test_audit_system():
    test_dir = "data/test_audit_tmp"
    os.makedirs(test_dir, exist_ok=True)
    log_file = os.path.join(test_dir, "test.jsonl")
    
    print("--- 1. Testing 10,000-Line FIFO Buffer & Ring Buffer Eviction ---")
    logger = AerospaceAuditLogger(max_lines=50, storage_file=log_file)
    logger.clear_buffer()

    # Log 70 events on a buffer with max 50
    for i in range(1, 71):
        logger.log_event(
            event_type="TELEMETRY_FRAME",
            level="INFO",
            subsystem="EPS_CORE",
            message=f"Telemetry frame #{i}",
            telemetry={"voltage": 3.70 + (i * 0.001), "current": 2.45, "temperature": 22.0 + (i * 0.1)}
        )

    stats = logger.get_stats()
    print("Stats after 70 logs (max 50):", stats)
    assert stats["buffer_count"] == 50, f"Expected 50, got {stats['buffer_count']}"
    assert stats["total_events_logged"] == 71, f"Expected 71, got {stats['total_events_logged']}"
    assert stats["buffer_usage_pct"] == 100.0

    logs = logger.get_logs(limit=100, reverse=False)
    assert len(logs) == 50
    # First item should be item #22 (seq_id 1 was clear log, 2-21 were first 20 frames evicted)
    assert logs[0]["seq_id"] == 22, f"Expected first seq_id=22, got {logs[0]['seq_id']}"
    assert logs[-1]["seq_id"] == 71, f"Expected last seq_id=71, got {logs[-1]['seq_id']}"
    print("[PASS] FIFO buffer strictly limited to 50 entries and evicted first 20 in order!")

    print("\n--- 2. Testing Event Categories & Schemas ---")
    logger.log_operator_action("INJECT_FAULT_THERMAL_RUNAWAY", "Duration 25s", "FlightDirector")
    logger.log_early_prediction("Thermal Gradient Rising", 0.38, 0.70, "HIGH_SENSITIVITY_PREARM", {"voltage": 3.65}, {"p_ensemble": 0.38})
    logger.log_fault_event("THERMAL_RUNAWAY", "CRITICAL", "Temp > 50C", {"voltage": 3.55}, {"p_ensemble": 0.95})
    logger.log_mitigation_event("LOAD_SHEDDING", "THERMAL_RUNAWAY", 38.5, 0.97)
    logger.log_recovery_event("LOAD_SHEDDING", "THERMAL_RUNAWAY", "Nominal limits restored.")

    stats = logger.get_stats()
    print("Stats after specialized events:", stats)
    assert stats["operator_actions"] == 1
    assert stats["early_predictions"] == 1
    assert stats["faults_detected"] == 1
    assert stats["mitigations_executed"] == 1
    assert stats["recoveries"] == 1
    print("[PASS] All event types and statistics tracked accurately!")

    print("\n--- 3. Testing PDF Audit Report Generation ---")
    pdf_path = os.path.join(test_dir, "AERO_GUARD_TEST_AUDIT.pdf")
    all_logs = logger.get_logs(limit=100, reverse=True)
    generated = generate_mission_audit_pdf(
        logs=all_logs,
        stats=stats,
        output_path=pdf_path,
        title="AERO-GUARD MISSION AUDIT LEDGER"
    )
    assert os.path.exists(generated), "PDF file was not created"
    file_sz = os.path.getsize(generated)
    print(f"[PASS] PDF generated successfully at '{generated}' (Size: {file_sz:,} bytes)!")

    shutil.rmtree(test_dir, ignore_errors=True)
    print("\n==========================================")
    print(" ALL AUDIT & PDF TESTS PASSED WITH 100% SUCCESS!")
    print("==========================================")


if __name__ == "__main__":
    test_audit_system()
