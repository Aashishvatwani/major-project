"""
Orbital Mechanics & Satellite Battery Telemetry Synthetic Generator
Simulates realistic LEO (Low Earth Orbit) orbital thermal-electrical dynamics,
eclipse transitions, solar charging cycles, and injects verified aerospace failure modes.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class SatelliteTelemetryGenerator:
    """
    Generates high-fidelity multi-orbit satellite EPS (Electrical Power Subsystem) telemetry
    with realistic physical battery dynamics and injected orbital failure modes.
    """

    def __init__(
        self,
        orbital_period_sec: float = 5400.0,   # ~90 minute LEO orbit
        sunlight_duration_sec: float = 3300.0, # ~55 minutes in sunlight
        eclipse_duration_sec: float = 2100.0,  # ~35 minutes in Earth shadow
        sampling_rate_hz: float = 1.0,         # 1 Hz sampling
        nominal_voltage: float = 3.70,         # Nominal cell voltage (V)
        nominal_temp: float = 22.0,            # Nominal cell temperature (°C)
        random_seed: int = 42
    ):
        self.orbital_period_sec = orbital_period_sec
        self.sunlight_duration_sec = sunlight_duration_sec
        self.eclipse_duration_sec = eclipse_duration_sec
        self.sampling_rate_hz = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        self.nominal_voltage = nominal_voltage
        self.nominal_temp = nominal_temp
        self.rng = np.random.default_rng(random_seed)

    def generate_telemetry_dataset(
        self,
        duration_minutes: float = 360.0,  # 6 hours = 4 complete orbits
        anomaly_ratio: float = 0.12,
        inject_anomalies: bool = True
    ) -> pd.DataFrame:
        """
        Generates continuous telemetry time-series with realistic physics:
        - Sunlight: Solar panel array generation, battery charging (CC/CV curve), radiative solar heating.
        - Eclipse: Zero solar input, continuous payload power draw, battery discharging, radiative cooling to deep space.
        """
        total_seconds = int(duration_minutes * 60)
        total_samples = int(total_seconds * self.sampling_rate_hz)

        timestamps = np.arange(0, total_seconds, self.dt)[:total_samples]
        orbit_phases = (timestamps % self.orbital_period_sec) / self.orbital_period_sec

        # Eclipse Indicator (1 = Eclipse shadow, 0 = Direct Sunlight)
        is_eclipse = (timestamps % self.orbital_period_sec) >= self.sunlight_duration_sec

        voltage = np.zeros(total_samples)
        current = np.zeros(total_samples)
        temperature = np.zeros(total_samples)
        soc = np.zeros(total_samples)
        current_soc = 0.88  # Starting State of Charge (88%)
        anomaly_label = np.zeros(total_samples, dtype=int)
        anomaly_type = ["normal"] * total_samples

        base_r_int = 0.045  # 45 mOhm internal resistance

        for i in range(total_samples):
            t = timestamps[i]
            eclipse = is_eclipse[i]

            if eclipse:
                # Discharging: payload draws 2.2A - 3.2A
                load = 2.70 + 0.3 * np.sin(2 * np.pi * t / 300.0) + self.rng.normal(0, 0.04)
                current_soc -= (load * self.dt) / (10.0 * 3600.0)  # 10Ah pack
                i_val = load
                # Voltage curve: Open Circuit Voltage - I * R_int
                ocv = 3.30 + 0.85 * current_soc - 0.05 / (current_soc + 0.05)
                v_val = ocv - (i_val * base_r_int) + self.rng.normal(0, 0.012)
                # Radiative cooling towards cold space (-5°C)
                t_val = self.nominal_temp - 12.0 * (1 - np.exp(- (t % self.orbital_period_sec - self.sunlight_duration_sec) / 600.0)) + self.rng.normal(0, 0.15)
            else:
                # Charging: solar array provides current
                solar_gen = 4.20 + 0.4 * np.sin(np.pi * (t % self.orbital_period_sec) / self.sunlight_duration_sec)
                payload_draw = 1.80 + self.rng.normal(0, 0.03)
                net_charge_current = solar_gen - payload_draw
                current_soc += (net_charge_current * self.dt) / (10.0 * 3600.0)
                current_soc = min(current_soc, 0.98)
                i_val = payload_draw
                ocv = 3.30 + 0.85 * current_soc
                v_val = ocv + (net_charge_current * 0.02) + self.rng.normal(0, 0.012)
                # Solar heating towards 35°C
                t_val = self.nominal_temp + 14.0 * (1 - np.exp(- (t % self.orbital_period_sec) / 800.0)) + self.rng.normal(0, 0.15)

            current_soc = np.clip(current_soc, 0.05, 1.0)
            soc[i] = current_soc
            voltage[i] = v_val
            current[i] = i_val
            temperature[i] = t_val

        # Inject Fault Scenarios across entire duration regularly
        if inject_anomalies:
            fault_duration_samples = int(45 * self.sampling_rate_hz)  # 45-second anomaly window
            # Inject an anomaly every ~6-8 minutes evenly distributed
            interval_samples = int(7 * 60 * self.sampling_rate_hz)
            num_intervals = total_samples // interval_samples

            fault_types = [
                "thermal_runaway_precursor",
                "internal_short_circuit",
                "high_impedance_degradation",
                "sensor_drift_fault",
                "deep_undervoltage_collapse"
            ]

            for interval_idx in range(num_intervals):
                f_type = fault_types[interval_idx % len(fault_types)]
                offset = self.rng.integers(15, interval_samples - fault_duration_samples - 15)
                idx_start = interval_idx * interval_samples + offset
                idx_end = idx_start + fault_duration_samples

                if idx_end > total_samples:
                    continue

                anomaly_label[idx_start:idx_end] = 1

                for k in range(idx_start, idx_end):
                    anomaly_type[k] = f_type
                    step_k = k - idx_start

                    if f_type == "thermal_runaway_precursor":
                        # Fast accelerating temperature rise + current growth
                        temperature[k] += 12.0 + 1.25 * step_k + self.rng.normal(0, 0.25)
                        voltage[k] -= 0.15 + 0.015 * step_k
                        current[k] += 0.40 + 0.045 * step_k

                    elif f_type == "internal_short_circuit":
                        # Sudden massive current spike + sharp voltage collapse + heat pulse
                        voltage[k] -= 1.10 + self.rng.normal(0, 0.04)
                        current[k] += 4.80 + self.rng.normal(0, 0.15)
                        temperature[k] += 6.0 + 0.45 * step_k

                    elif f_type == "high_impedance_degradation":
                        # High internal resistance: severe voltage drop under payload load
                        voltage[k] -= (current[k] * 0.55) + 0.45 + self.rng.normal(0, 0.02)
                        temperature[k] += 4.0 + 0.22 * step_k

                    elif f_type == "sensor_drift_fault":
                        # Erratic high amplitude sensor oscillation / bias offset
                        voltage[k] += 1.25 * np.sin(step_k / 2.5) + self.rng.normal(0, 0.15)
                        current[k] += 1.80 * np.cos(step_k / 2.0)

                    elif f_type == "deep_undervoltage_collapse":
                        # Voltage dropping below critical discharge threshold
                        voltage[k] = max(1.95, 2.35 - 0.025 * step_k + self.rng.normal(0, 0.02))
                        soc[k] = max(0.02, 0.12 - 0.002 * step_k)
                        temperature[k] -= 0.05 * step_k

        df = pd.DataFrame({
            "timestamp": timestamps,
            "voltage": np.round(voltage, 4),
            "current": np.round(current, 4),
            "temperature": np.round(temperature, 2),
            "soc": np.round(soc, 4),
            "is_eclipse": is_eclipse.astype(int),
            "anomaly_label": anomaly_label,
            "anomaly_type": anomaly_type
        })

        return df


def generate_and_save_data(output_path: str = "data/raw/synthetic_telemetry.csv") -> pd.DataFrame:
    """Helper function to generate and save synthetic dataset"""
    generator = SatelliteTelemetryGenerator()
    df = generator.generate_telemetry_dataset(duration_minutes=360.0, inject_anomalies=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated and saved {len(df)} samples with anomalies to {output_path}")
    return df


if __name__ == "__main__":
    generate_and_save_data()
