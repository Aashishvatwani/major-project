"""
Unit Tests for Digital Twin Counterfactual Simulation and AI Agent RAG Reasoning
"""

import pytest
import numpy as np
from src.digital_twin.counterfactual import DigitalTwinCounterfactualSimulator
from src.reasoning.ai_agent_rag import AIAgentRAGReasoner
from src.models.fault_diagnosis import FaultDiagnosisClassifier, SeverityEstimator


def test_digital_twin_counterfactual_simulation():
    simulator = DigitalTwinCounterfactualSimulator(horizon_seconds=30.0, dt=1.0)
    nominal_telemetry = {
        "voltage": 3.72,
        "current": 2.45,
        "temperature": 23.0,
        "soc": 0.85,
        "impedance_proxy": 0.045,
        "is_eclipse": 0
    }

    outcomes = simulator.simulate_all_actions(
        current_telemetry=nominal_telemetry,
        fault_type="NOMINAL",
        anomaly_prob=0.02
    )

    assert len(outcomes) == 4
    # Action names present
    act_names = [o["action_name"] for o in outcomes]
    assert "NOMINAL_CONTINUE" in act_names
    assert "LOAD_SHEDDING" in act_names
    assert "SAFE_MODE_ISOLATION" in act_names

    # Exactly one recommended action
    rec_count = sum(1 for o in outcomes if o["is_recommended"])
    assert rec_count == 1


def test_severity_estimator_and_risk_scoring():
    # 1. Nominal telemetry
    nom_telemetry = {
        "temperature": 22.0,
        "voltage": 3.70,
        "current": 2.50,
        "impedance_proxy": 0.045,
        "thermal_gradient": 0.01
    }
    nom_eval = SeverityEstimator.estimate_severity(nom_telemetry, anomaly_prob=0.05, fault_type="NOMINAL")
    assert nom_eval["severity"] == "NOMINAL"
    assert nom_eval["risk_score"] < 0.35

    # 2. Critical Thermal Runaway
    crit_telemetry = {
        "temperature": 68.0,
        "voltage": 3.40,
        "current": 5.50,
        "impedance_proxy": 0.25,
        "thermal_gradient": 2.2
    }
    crit_eval = SeverityEstimator.estimate_severity(crit_telemetry, anomaly_prob=0.95, fault_type="THERMAL_RUNAWAY", safety_override=True)
    assert crit_eval["severity"] == "EMERGENCY"
    assert crit_eval["risk_score"] >= 0.85


def test_ai_agent_rag_reasoning():
    reasoner = AIAgentRAGReasoner()
    telemetry = {
        "temperature": 56.0,
        "voltage": 3.50,
        "current": 3.50,
        "impedance_proxy": 0.08,
        "thermal_gradient": 1.2
    }
    ml_eval = {
        "p_ensemble": 0.88,
        "p_rf": 0.85,
        "p_xgboost": 0.90,
        "p_extra_trees": 0.89
    }
    fault_diag = {
        "primary_fault": "THERMAL_RUNAWAY",
        "diagnosis_confidence": 0.94
    }
    counterfactuals = [
        {"action_name": "LOAD_SHEDDING", "is_recommended": True, "summary": "Sheds 35% load.", "safety_score": 0.95}
    ]
    safety_eval = {"override_active": False, "violations": []}
    rl_rec = {"rl_action_name": "LOAD_SHEDDING"}

    analysis = reasoner.analyze(
        telemetry=telemetry,
        ml_eval=ml_eval,
        fault_diagnosis=fault_diag,
        counterfactuals=counterfactuals,
        safety_eval=safety_eval,
        rl_recommendation=rl_rec
    )

    assert "agent_message" in analysis
    assert "NASA-HDBK-4008" in [r["doc_id"] for r in analysis["matched_regulations"]]
    assert analysis["risk_level"] in ["HIGH", "CRITICAL"]
