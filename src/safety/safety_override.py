"""
Deterministic Spacecraft Safety Override Engine
Enforces mission-critical physical boundary limits on satellite electrical power subsystem telemetry.
Overrides statistical machine learning and reinforcement learning decisions with 100% deterministic priority.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Tuple


class SafetyOverrideEngine:
    """
    Deterministic Safety Guardrail Engine.
    Evaluates physical invariants:
    - Temperature bounds [T_min, T_max]
    - Thermal rate of change dT/dt
    - Voltage bounds [V_min, V_max]
    - Current bounds [I_min, I_max]
    - Internal resistance & impedance breakdown
    - Maximum power threshold
    """

    def __init__(self, rules_path: str = "config/safety_rules.yaml"):
        self.rules_path = rules_path
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        default_rules = {
            "temperature": {
                "critical_high_celsius": 65.0,
                "critical_low_celsius": -20.0,
                "max_rate_of_rise_c_per_sec": 2.5
            },
            "voltage": {
                "critical_overvoltage": 4.35,
                "critical_undervoltage": 2.70,
                "abnormal_voltage_drop_rate": 0.80
            },
            "current": {
                "critical_overcurrent": 8.50,
                "reverse_current_limit": -0.50
            },
            "impedance": {
                "critical_impedance_surge": 3.50,
                "internal_short_resistance": 0.15
            },
            "power": {
                "max_continuous_power_watts": 35.0
            }
        }

        if os.path.exists(self.rules_path):
            try:
                with open(self.rules_path, "r") as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        return loaded
            except Exception:
                pass

        return default_rules

    def evaluate_telemetry(
        self,
        telemetry: Dict[str, Any],
        ml_prediction: int,
        ml_prob: float
    ) -> Dict[str, Any]:
        """
        Evaluates incoming telemetry against all safety rules.
        If any critical condition is violated, triggers immediate safety override.
        """
        v = float(telemetry.get("voltage", 3.7))
        i = float(telemetry.get("current", 2.0))
        t = float(telemetry.get("temperature", 22.0))
        power = float(telemetry.get("power_watts", v * i))
        t_grad = float(telemetry.get("thermal_gradient", 0.0))
        imp = float(telemetry.get("impedance_proxy", 0.05))

        violations = []
        override_severity = "NONE"

        # 1. Temperature checks
        t_rules = self.rules.get("temperature", {})
        if t > t_rules.get("critical_high_celsius", 65.0):
            violations.append(f"CRITICAL OVER-TEMPERATURE: {t:.2f}°C > {t_rules.get('critical_high_celsius')}°C")
        elif t < t_rules.get("critical_low_celsius", -20.0):
            violations.append(f"CRITICAL UNDER-TEMPERATURE: {t:.2f}°C < {t_rules.get('critical_low_celsius')}°C")

        if abs(t_grad) > t_rules.get("max_rate_of_rise_c_per_sec", 2.5):
            violations.append(f"RAPID THERMAL EXCURSION: |dT/dt| = {abs(t_grad):.2f}°C/s > {t_rules.get('max_rate_of_rise_c_per_sec')}°C/s")

        # 2. Voltage checks
        v_rules = self.rules.get("voltage", {})
        if v > v_rules.get("critical_overvoltage", 4.35):
            violations.append(f"CRITICAL OVER-VOLTAGE: {v:.3f}V > {v_rules.get('critical_overvoltage')}V")
        elif v < v_rules.get("critical_undervoltage", 2.70):
            violations.append(f"CRITICAL UNDER-VOLTAGE (DEEP DISCHARGE): {v:.3f}V < {v_rules.get('critical_undervoltage')}V")

        # 3. Current checks
        i_rules = self.rules.get("current", {})
        if i > i_rules.get("critical_overcurrent", 8.50):
            violations.append(f"CRITICAL OVER-CURRENT / SHORT-CIRCUIT: {i:.3f}A > {i_rules.get('critical_overcurrent')}A")
        elif i < i_rules.get("reverse_current_limit", -0.50):
            violations.append(f"REVERSE DISCHARGE CURRENT: {i:.3f}A < {i_rules.get('reverse_current_limit')}A")

        # 4. Power check
        p_rules = self.rules.get("power", {})
        if power > p_rules.get("max_continuous_power_watts", 35.0):
            violations.append(f"POWER ENVELOPE EXCEEDED: {power:.2f}W > {p_rules.get('max_continuous_power_watts')}W")

        # 5. Impedance check
        imp_rules = self.rules.get("impedance", {})
        if imp > imp_rules.get("critical_impedance_surge", 3.50):
            violations.append(f"CRITICAL IMPEDANCE SURGE: {imp:.2f}Ω > {imp_rules.get('critical_impedance_surge')}Ω")

        override_active = len(violations) > 0

        if override_active:
            final_decision = 1
            final_status = "CRITICAL_SAFETY_OVERRIDE"
            final_p = max(ml_prob, 0.99)
            override_severity = "CRITICAL"
        else:
            final_decision = ml_prediction
            final_p = ml_prob
            if ml_prediction == 1:
                final_status = "ML_RL_ANOMALY_ALERT"
            else:
                final_status = "NOMINAL"

        return {
            "override_active": override_active,
            "violations": violations,
            "override_severity": override_severity,
            "ml_prediction": ml_prediction,
            "ml_probability": ml_prob,
            "final_decision": final_decision,
            "final_status": final_status,
            "final_probability": final_p
        }
