"""
Unit Tests for Spacecraft Deterministic Safety Override Engine
"""

import pytest
from src.safety.safety_override import SafetyOverrideEngine


def test_safety_override_nominal():
    """Verify nominal conditions do not trigger safety override"""
    engine = SafetyOverrideEngine()
    nominal_telem = {
        "voltage": 3.70,
        "current": 2.20,
        "temperature": 24.0,
        "thermal_gradient": 0.05,
        "power_watts": 8.14,
        "impedance_proxy": 0.05
    }

    res = engine.evaluate_telemetry(nominal_telem, ml_prediction=0, ml_prob=0.08)
    assert not res["override_active"]
    assert res["final_decision"] == 0
    assert res["final_status"] == "NOMINAL"
    assert len(res["violations"]) == 0


def test_safety_override_critical_temperature():
    """Verify critical over-temperature forces immediate override even if ML says normal"""
    engine = SafetyOverrideEngine()
    critical_temp_telem = {
        "voltage": 3.70,
        "current": 2.20,
        "temperature": 72.5,  # Exceeds 65°C limit
        "thermal_gradient": 0.1,
        "power_watts": 8.14,
        "impedance_proxy": 0.05
    }

    res = engine.evaluate_telemetry(critical_temp_telem, ml_prediction=0, ml_prob=0.15)
    assert res["override_active"]
    assert res["final_decision"] == 1
    assert res["final_status"] == "CRITICAL_SAFETY_OVERRIDE"
    assert any("OVER-TEMPERATURE" in v for v in res["violations"])


def test_safety_override_deep_undervoltage():
    """Verify critical undervoltage below 2.70V triggers safety override"""
    engine = SafetyOverrideEngine()
    critical_volt_telem = {
        "voltage": 2.45,  # Deep discharge below 2.70V
        "current": 2.20,
        "temperature": 20.0,
        "thermal_gradient": 0.0,
        "power_watts": 5.39,
        "impedance_proxy": 0.05
    }

    res = engine.evaluate_telemetry(critical_volt_telem, ml_prediction=0, ml_prob=0.20)
    assert res["override_active"]
    assert res["final_decision"] == 1
    assert any("UNDER-VOLTAGE" in v for v in res["violations"])
