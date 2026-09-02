"""
Aerospace AI Agent Diagnostic Reasoning & RAG Safety Engine
Synthesizes multi-model ML inferences, fault taxonomy, digital twin counterfactual simulations,
and aerospace engineering guidelines (NASA Ames / ESA ECSS) into explainable diagnostic reports.
"""

from typing import Dict, Any, List, Optional
import numpy as np


RAG_KNOWLEDGE_BASE = [
    {
        "id": "NASA-HDBK-4008",
        "title": "NASA Handbook for Spacecraft Power System Fault Management",
        "rule_trigger": lambda f, t, v, c, r: f == "THERMAL_RUNAWAY" or t > 50.0,
        "citation": "NASA-HDBK-4008 §4.2.1: Secondary cell thermal escalation above 50°C requires aggressive load shedding to halt exothermic self-heating.",
        "action_priority": 2
    },
    {
        "id": "ESA-ECSS-E-ST-20C-SHORT",
        "title": "ECSS Electrical & Power Subsystems Standard",
        "rule_trigger": lambda f, t, v, c, r: f == "INTERNAL_SHORT" or c > 6.0 or r < 0.02,
        "citation": "ECSS-E-ST-20C §7.3: Bus overcurrent or sudden micro-short signature demands autonomous circuit isolation within 2.0s.",
        "action_priority": 3
    },
    {
        "id": "AIAA-S-136-2023-DISCHARGE",
        "title": "Battery Safety Standard for Space Applications",
        "rule_trigger": lambda f, t, v, c, r: f == "UNDERVOLTAGE" or v < 2.90,
        "citation": "AIAA-S-136 §5.1: Deep discharge below 2.90V risks copper dissolution and permanent cell short-circuiting.",
        "action_priority": 2
    },
    {
        "id": "NASA-SP-20205003605-IMPEDANCE",
        "title": "NASA Li-Ion Battery State of Health Monitoring Guidelines",
        "rule_trigger": lambda f, t, v, c, r: f == "HIGH_IMPEDANCE" or r > 0.15,
        "citation": "NASA-SP-20205003605 §8.4: Internal impedance doubling indicates electrolyte dryout; derate peak payload discharge by 40%.",
        "action_priority": 2
    },
    {
        "id": "CCSDS-502.0-B-3-SENSOR",
        "title": "CCSDS Spacecraft Telemetry Quality & Consistency Standards",
        "rule_trigger": lambda f, t, v, c, r: f == "SENSOR_FAULT",
        "citation": "CCSDS-502.0-B-3: Sensor drift exceeding 3σ requires switching to secondary telemetry channel or estimator fallback.",
        "action_priority": 1
    }
]


class AIAgentRAGReasoner:
    """
    Autonomous Aerospace Diagnostic Reasoner.
    Generates explainable diagnostic logs, checks RAG flight rule compliance,
    and structures decision justification.
    """

    def __init__(self):
        self.knowledge_base = RAG_KNOWLEDGE_BASE

    def analyze(
        self,
        telemetry: Dict[str, Any],
        ml_eval: Dict[str, Any],
        fault_diagnosis: Dict[str, Any],
        counterfactuals: List[Dict[str, Any]],
        safety_eval: Dict[str, Any],
        rl_recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesizes complete telemetry state into structured AI agent analysis.
        """
        temp = float(telemetry.get("temperature", 22.0))
        volt = float(telemetry.get("voltage", 3.7))
        curr = float(telemetry.get("current", 2.5))
        r_int = float(telemetry.get("impedance_proxy", 0.045))
        p_ens = float(ml_eval.get("p_ensemble", 0.0))
        primary_fault = fault_diagnosis.get("primary_fault", "NOMINAL")
        fault_conf = float(fault_diagnosis.get("diagnosis_confidence", 0.0))
        is_override = bool(safety_eval.get("override_active", False))

        # Query RAG Knowledge Base for matching aerospace flight regulations
        matched_regulations = []
        for doc in self.knowledge_base:
            try:
                if doc["rule_trigger"](primary_fault, temp, volt, curr, r_int):
                    matched_regulations.append({
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "citation": doc["citation"]
                    })
            except Exception:
                pass

        if not matched_regulations:
            matched_regulations.append({
                "doc_id": "NOMINAL-OPS-STD",
                "title": "Standard Spacecraft Telemetry Monitoring Protocols",
                "citation": "ECSS-E-ST-20C: Nominal power subsystem parameters observed within designated operational limits."
            })

        # Counterfactual recommendation
        best_cf = next((cf for cf in counterfactuals if cf.get("is_recommended", False)), counterfactuals[0] if counterfactuals else {})
        recommended_action_name = best_cf.get("action_name", "NOMINAL_CONTINUE")

        # Root Cause & Physics Signature Summary
        if is_override:
            root_cause = f"HARD CRITICAL SAFETY LIMIT EXCEEDED: {safety_eval.get('violations', ['Limit Trip'])[0]}"
            signature = f"Dangerous physical excursion: T={temp:.1f}°C, V={volt:.3f}V, I={curr:.2f}A, R={r_int:.3f}Ω."
            risk_level = "CRITICAL"
            escalate_human = True
        elif primary_fault != "NOMINAL" and p_ens >= 0.40:
            root_cause = f"Detected {primary_fault.replace('_', ' ')} with {fault_conf*100:.1f}% confidence."
            signature = f"Anomalous telemetry signature: P_ens={p_ens:.3f}, T={temp:.1f}°C (dT/dt={telemetry.get('thermal_gradient', 0.0):+.2f}°C/s), R_int={r_int:.3f}Ω."
            risk_level = "HIGH" if p_ens >= 0.70 else "MODERATE"
            escalate_human = p_ens >= 0.75 or primary_fault in ["THERMAL_RUNAWAY", "INTERNAL_SHORT"]
        else:
            root_cause = "Subsystem operating within nominal aerospace flight envelope."
            signature = f"Nominal baseline telemetry: V={volt:.3f}V, I={curr:.2f}A, T={temp:.1f}°C, SOC={telemetry.get('soc', 0.85)*100:.1f}%."
            risk_level = "LOW"
            escalate_human = False

        # Format Agent Typewriter Message
        agent_message = (
            f"[AI DIAGNOSTIC REPORT]\n"
            f"• Subsystem State: {risk_level} RISK | Primary Diagnosis: {primary_fault}\n"
            f"• ML Ensemble Consensus: {p_ens:.2%} (RF: {ml_eval.get('p_rf', 0):.2%}, XGB: {ml_eval.get('p_xgboost', 0):.2%}, ET: {ml_eval.get('p_extra_trees', 0):.2%})\n"
            f"• Digital Twin Projection: {best_cf.get('summary', 'Nominal forward projection.')}\n"
            f"• Compliance Citation: {matched_regulations[0]['citation']}\n"
            f"• Autonomous Recommendation: {recommended_action_name} (Safety Index: {best_cf.get('safety_score', 1.0):.2f})"
        )

        return {
            "risk_level": risk_level,
            "root_cause": root_cause,
            "signature_evidence": signature,
            "matched_regulations": matched_regulations,
            "recommended_action": recommended_action_name,
            "counterfactual_summary": best_cf.get("summary", ""),
            "agent_message": agent_message,
            "requires_human_approval": escalate_human,
            "decision_gateway": "HUMAN_REVIEW_GATE" if escalate_human else "AUTO_APPROVE_GATE"
        }
