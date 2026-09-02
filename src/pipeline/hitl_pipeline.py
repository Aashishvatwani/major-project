"""
Satellite Telemetry Anomaly Detection — Master Hardware-in-the-Loop Pipeline
Full Digital Twin + AI Multi-Model Ensemble + Fault Diagnosis + Counterfactuals + RAG Reasoning + HITL Bridge.
"""

import time
import os
from typing import Dict, Any, Optional, List, Generator
import pandas as pd
import numpy as np

from src.ingestion.telemetry_stream import TelemetryStream
from src.features.feature_engineering import TelemetryFeatureExtractor
from src.features.preprocessor import TelemetryPreprocessor
from src.models.ensemble_classifier import SatelliteEnsembleClassifier
from src.models.fault_diagnosis import FaultDiagnosisClassifier, SeverityEstimator
from src.models.rl_agent import RLAgent
from src.models.model_registry import ModelRegistry
from src.digital_twin.counterfactual import DigitalTwinCounterfactualSimulator
from src.reasoning.ai_agent_rag import AIAgentRAGReasoner
from src.safety.safety_override import SafetyOverrideEngine
from src.hitl.serial_bridge import HITLSerialBridge
from src.utils.logger import setup_aerospace_logger


class SatelliteHITLPipeline:
    """
    Complete Aerospace Health Management (FDIR) Pipeline.
    Orchestrates:
    Telemetry Stream -> Feature Extraction -> Scaler -> ML Ensemble -> Multi-class Diagnosis ->
    Severity Estimator -> RL Policy -> Counterfactual Sim -> AI Agent RAG Reasoning ->
    Deterministic Safety -> Dual Risk Gateway -> PySerial Arduino Bridge.
    """

    def __init__(
        self,
        models_dir: str = "saved_models",
        serial_port: str = "AUTO",
        baud_rate: int = 115200,
        enable_hardware: bool = True
    ):
        self.logger = setup_aerospace_logger("SatellitePipeline")
        self.models_dir = models_dir

        # Load models from registry
        self.preprocessor, self.ensemble, self.rl_agent, self.fault_classifier = ModelRegistry.load_pipeline_artifacts(models_dir)
        self.feature_extractor = TelemetryFeatureExtractor()
        self.safety_engine = SafetyOverrideEngine()
        self.counterfactual_sim = DigitalTwinCounterfactualSimulator(horizon_seconds=60.0)
        self.ai_agent = AIAgentRAGReasoner()

        self.enable_hardware = enable_hardware
        self.serial_bridge = HITLSerialBridge(port=serial_port, baud_rate=baud_rate) if enable_hardware else None

        self.stream = TelemetryStream()
        self.recent_alerts = []
        self.human_approval_override = False

    def process_single_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes complete inference & reasoning cycle for a single incoming telemetry point.
        """
        t_start = time.perf_counter()

        # 1. Physics-Informed Feature Extraction
        features = self.feature_extractor.extract_streaming_features(raw_sample)

        # 2. Preprocessing & Scaling
        x_scaled = self.preprocessor.transform_single(features)

        # 3. Supervised Multi-Model Ensemble (RF + XGBoost + Extra Trees)
        ml_eval = self.ensemble.predict_single_detailed(x_scaled)

        # 4. Multi-Class Fault Diagnosis Classifier
        fault_eval = self.fault_classifier.predict_diagnosis(x_scaled, anomaly_prob=ml_eval["p_ensemble"])
        primary_fault = fault_eval["primary_fault"]

        # 5. Reinforcement Learning Adaptive Policy & Dynamic Threshold
        recent_rate = float(np.mean(self.recent_alerts[-20:])) if self.recent_alerts else 0.0
        rl_recommendation = self.rl_agent.get_action_recommendation(
            p_ensemble=ml_eval["p_ensemble"],
            thermal_gradient=features.get("thermal_gradient", 0.0),
            impedance_proxy=features.get("impedance_proxy", 0.05),
            soc=float(raw_sample.get("soc", 0.8)),
            is_eclipse=int(raw_sample.get("is_eclipse", 0)),
            recent_alert_rate=recent_rate
        )

        dyn_threshold = rl_recommendation["dynamic_threshold"]
        rl_action_id = rl_recommendation["rl_action_id"]

        # Re-evaluate binary decision with dynamic threshold
        ml_decision = int(ml_eval["p_ensemble"] >= dyn_threshold)
        if rl_action_id in [2, 3]:
            # Load shedding or safe mode forces anomaly flag
            ml_decision = 1

        # 6. Spacecraft Digital Twin Counterfactual 60s Simulation
        counterfactuals = self.counterfactual_sim.simulate_all_actions(
            current_telemetry=features,
            fault_type=primary_fault,
            anomaly_prob=ml_eval["p_ensemble"]
        )

        # 7. Deterministic Safety Override Guardrails
        safety_eval = self.safety_engine.evaluate_telemetry(
            telemetry=features,
            ml_prediction=ml_decision,
            ml_prob=ml_eval["p_ensemble"]
        )

        # 8. Severity & Risk Scoring
        severity_eval = SeverityEstimator.estimate_severity(
            telemetry=features,
            anomaly_prob=ml_eval["p_ensemble"],
            fault_type=primary_fault,
            safety_override=safety_eval["override_active"]
        )

        final_alert_signal = safety_eval["final_decision"]
        final_severity = severity_eval["severity"]

        # 9. AI Agent Diagnostic Reasoning & RAG Standard Verification
        ai_reasoning = self.ai_agent.analyze(
            telemetry=features,
            ml_eval=ml_eval,
            fault_diagnosis=fault_eval,
            counterfactuals=counterfactuals,
            safety_eval=safety_eval,
            rl_recommendation=rl_recommendation
        )

        # 10. Action Gateway & Risk Gate
        requires_human = ai_reasoning["requires_human_approval"] and not self.human_approval_override
        gateway_mode = "HUMAN_REVIEW_GATE" if requires_human else "AUTO_APPROVE_GATE"

        # Track recent alert history
        self.recent_alerts.append(final_alert_signal)
        if len(self.recent_alerts) > 100:
            self.recent_alerts.pop(0)

        # 11. Hardware Actuation via PySerial / Virtual Arduino
        hw_status = {}
        if self.serial_bridge and self.enable_hardware:
            hw_status = self.serial_bridge.send_anomaly_alert(
                alert_state=final_alert_signal,
                severity=final_severity,
                rl_action=rl_action_id
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Construct Master Pipeline Record
        result = {
            "timestamp": raw_sample.get("timestamp", time.time()),
            "voltage": float(raw_sample["voltage"]),
            "current": float(raw_sample["current"]),
            "temperature": float(raw_sample["temperature"]),
            "power_watts": features["power_watts"],
            "impedance_proxy": features["impedance_proxy"],
            "thermal_gradient": features["thermal_gradient"],
            "soc": float(raw_sample.get("soc", 0.0)),
            "is_eclipse": int(raw_sample.get("is_eclipse", 0)),
            "ground_truth": int(raw_sample.get("anomaly_label", 0)),
            "injected_fault": raw_sample.get("injected_fault", "none"),
            
            # Supervised Ensemble Predictions
            "p_rf": ml_eval["p_rf"],
            "p_xgboost": ml_eval["p_xgboost"],
            "p_extra_trees": ml_eval["p_extra_trees"],
            "p_ensemble": ml_eval["p_ensemble"],
            
            # Multi-Class Fault Diagnosis
            "primary_fault": primary_fault,
            "diagnosis_confidence": fault_eval["diagnosis_confidence"],
            "fault_probabilities": fault_eval["fault_probabilities"],
            
            # Severity & Risk
            "risk_score": severity_eval["risk_score"],
            "final_severity": final_severity,
            "final_status": safety_eval["final_status"],
            "final_decision": final_alert_signal,
            
            # RL Adaptive State
            "rl_action_id": rl_action_id,
            "rl_action_name": rl_recommendation["rl_action_name"],
            "dynamic_threshold": dyn_threshold,
            "policy_confidence": rl_recommendation["policy_confidence"],
            
            # Digital Twin Counterfactual Projections
            "counterfactuals": counterfactuals,
            
            # AI Agent Reasoning & RAG
            "ai_reasoning": ai_reasoning,
            "risk_level": ai_reasoning["risk_level"],
            "decision_gateway": gateway_mode,
            "requires_human_approval": requires_human,
            "rag_citation": ai_reasoning["matched_regulations"][0]["citation"],
            
            # Deterministic Safety Override
            "safety_override_active": safety_eval["override_active"],
            "safety_violations": safety_eval["violations"],
            
            # Hardware Status
            "hardware_pin13_led": hw_status.get("pin13_led_state", final_alert_signal),
            "hardware_mode": hw_status.get("pin13_mode", "SOLID_ON" if final_alert_signal == 1 else "OFF"),
            "hardware_device": hw_status.get("device", "Virtual Arduino Uno" if not self.serial_bridge else "Serial Bridge"),
            
            "latency_ms": latency_ms
        }

        return result

    def run_stream_generator(self, delay_sec: float = 0.5) -> Generator[Dict[str, Any], None, None]:
        """Generator yielding processed telemetry frames in real-time"""
        while True:
            sample = self.stream.get_next_sample()
            if sample is None:
                break
            result = self.process_single_sample(sample)
            yield result
            if delay_sec > 0:
                time.sleep(delay_sec)

    def close(self):
        if self.serial_bridge:
            self.serial_bridge.close()
