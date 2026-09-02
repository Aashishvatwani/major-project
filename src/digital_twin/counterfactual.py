"""
Spacecraft Digital Twin Counterfactual Simulation Engine
Simulates forward physical trajectories of battery electrochemistry and thermal dynamics
across multiple candidate mitigation actions to support autonomous and human-reviewed decisions.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class CounterfactualOutcome:
    action_id: int
    action_name: str
    projected_temp_60s: float
    projected_voltage_60s: float
    projected_soc_60s: float
    peak_temperature: float
    safety_score: float  # 0.0 to 1.0 (higher = safer)
    mission_availability: float  # 0.0 to 1.0 (payload operation fraction)
    energy_preservation: float  # 0.0 to 1.0
    composite_utility: float
    is_recommended: bool
    summary: str


class DigitalTwinCounterfactualSimulator:
    """
    Simulates high-fidelity 60-second forward electrochemical and thermal states
    of the spacecraft EPS under candidate mitigation actions.
    """

    ACTIONS = [
        {"id": 0, "name": "NOMINAL_CONTINUE", "load_factor": 1.0, "cooling_boost": 0.0, "availability": 1.0},
        {"id": 1, "name": "PRE_ARM_SENSITIVITY", "load_factor": 1.0, "cooling_boost": 0.1, "availability": 0.95},
        {"id": 2, "name": "LOAD_SHEDDING", "load_factor": 0.65, "cooling_boost": 0.35, "availability": 0.65},
        {"id": 3, "name": "SAFE_MODE_ISOLATION", "load_factor": 0.20, "cooling_boost": 0.70, "availability": 0.20}
    ]

    def __init__(self, horizon_seconds: float = 60.0, dt: float = 1.0):
        self.horizon_seconds = horizon_seconds
        self.dt = dt
        self.steps = int(horizon_seconds / dt)

    def simulate_all_actions(
        self,
        current_telemetry: Dict[str, Any],
        fault_type: str = "NOMINAL",
        anomaly_prob: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Runs counterfactual forward simulation for all 4 candidate actions.
        Returns sorted list of evaluated outcomes.
        """
        init_v = float(current_telemetry.get("voltage", 3.70))
        init_i = float(current_telemetry.get("current", 2.50))
        init_t = float(current_telemetry.get("temperature", 22.0))
        init_soc = float(current_telemetry.get("soc", 0.85))
        r_int = float(current_telemetry.get("impedance_proxy", 0.045))
        is_eclipse = int(current_telemetry.get("is_eclipse", 0))

        outcomes: List[CounterfactualOutcome] = []

        for act in self.ACTIONS:
            v = init_v
            t_cell = init_t
            soc = init_soc
            peak_t = init_t

            # Action parameters
            effective_i = init_i * act["load_factor"]
            cooling_boost = act["cooling_boost"]

            # Step through 60-second horizon
            for step in range(self.steps):
                # Fault dynamics injection
                if fault_type == "THERMAL_RUNAWAY":
                    # Internal exothermic reaction if not aggressively mitigated
                    exotherm = 0.18 * np.exp((t_cell - 25.0) / 18.0) * (1.0 - 0.75 * cooling_boost)
                elif fault_type == "INTERNAL_SHORT":
                    # Short circuit current heat
                    exotherm = 0.12 * (1.0 - 0.8 * cooling_boost)
                else:
                    exotherm = 0.0

                # Joule heating: P = I^2 * R_int
                joule_heat = (effective_i ** 2) * r_int * 0.015
                ambient_sink = 5.0 if is_eclipse else 25.0
                thermal_dissipation = 0.04 * (t_cell - ambient_sink) * (1.0 + cooling_boost)

                dT = (joule_heat + exotherm - thermal_dissipation) * self.dt
                t_cell += dT
                peak_t = max(peak_t, t_cell)

                # Electrochemical discharge / charge
                if is_eclipse:
                    soc -= (effective_i * self.dt) / (10.0 * 3600.0)
                else:
                    charge_i = max(0.0, 3.8 - effective_i)
                    soc += (charge_i * self.dt) / (10.0 * 3600.0)
                soc = np.clip(soc, 0.02, 1.0)

                # Voltage equation
                ocv = 3.30 + 0.85 * soc
                v = ocv - (effective_i * r_int)

            # Metrics Evaluation
            # Safety score (1.0 = cold & safe, 0.0 = thermal runaway > 60°C or deep discharge < 2.8V)
            temp_safety = np.clip(1.0 - max(0.0, peak_t - 35.0) / 30.0, 0.0, 1.0)
            volt_safety = np.clip((v - 2.70) / (4.20 - 2.70), 0.0, 1.0)
            safety_score = float(0.65 * temp_safety + 0.35 * volt_safety)

            mission_availability = act["availability"]
            energy_preservation = float(soc / max(init_soc, 0.01))

            # Composite utility function balancing Safety, Payload Availability, and Battery Life
            if safety_score < 0.40:
                # Extreme penalty if safety compromised
                utility = safety_score * 0.50
            else:
                utility = float(0.55 * safety_score + 0.30 * mission_availability + 0.15 * energy_preservation)

            # Rationale summary
            if act["id"] == 0:
                summary = f"Maintains 100% mission load. Projected T: {t_cell:.1f}°C, SOC: {soc*100:.1f}%."
            elif act["id"] == 1:
                summary = f"Pre-arms telemetry sensors with sensitivity threshold. Projected T: {t_cell:.1f}°C."
            elif act["id"] == 2:
                summary = f"Sheds 35% non-essential loads, reducing thermal stress. Projected T: {t_cell:.1f}°C, +{(init_t - t_cell):.1f}°C margin."
            else:
                summary = f"Emergency bus isolation. Drops payload to 20%, maximum thermal cooling. Projected T: {t_cell:.1f}°C."

            outcomes.append(CounterfactualOutcome(
                action_id=act["id"],
                action_name=act["name"],
                projected_temp_60s=float(round(t_cell, 2)),
                projected_voltage_60s=float(round(v, 3)),
                projected_soc_60s=float(round(soc, 4)),
                peak_temperature=float(round(peak_t, 2)),
                safety_score=float(round(safety_score, 3)),
                mission_availability=float(round(mission_availability, 2)),
                energy_preservation=float(round(energy_preservation, 3)),
                composite_utility=float(round(utility, 3)),
                is_recommended=False,
                summary=summary
            ))

        # Determine recommended action (highest composite utility)
        best_outcome = max(outcomes, key=lambda o: o.composite_utility)
        for o in outcomes:
            if o.action_id == best_outcome.action_id:
                o.is_recommended = True

        # Return list of serialized dicts
        return [asdict(o) for o in outcomes]
