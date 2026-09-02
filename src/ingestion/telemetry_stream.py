"""
Streaming Telemetry Ingestion Buffer and Real-Time Generator
Emulates live satellite telemetry downlink with runtime fault injection capabilities.
"""

import time
import asyncio
from typing import Dict, Any, Generator, Optional, List
import pandas as pd
import numpy as np


class TelemetryStream:
    """
    Simulates real-time streaming telemetry packet by packet.
    Allows dynamic interactive fault injection during simulation.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, playback_speed: float = 1.0):
        self.data = data
        self.playback_speed = playback_speed
        self.current_index = 0
        self._active_fault: Optional[str] = None
        self._fault_start_time: Optional[float] = None
        self._fault_duration_sec: float = 25.0
        self.sampling_rate_hz: float = 1.0
        self.rng = np.random.default_rng(42)

    def set_dataset(self, data: pd.DataFrame):
        self.data = data
        self.current_index = 0

    def inject_fault(self, fault_type: str, duration_sec: float = 25.0):
        """
        Injects a dynamic anomaly on-the-fly into the active stream:
        - "thermal_runaway": Rapid heating & degradation
        - "internal_short" / "short_circuit": High current spike + voltage drop
        - "sensor_fault" / "sensor_drift": Erratic sensor noise and offset
        - "undervoltage": Deep critical battery discharge
        - "high_impedance": High internal resistance under load
        """
        self._active_fault = fault_type.lower()
        self._fault_start_time = time.time()
        self._fault_duration_sec = duration_sec

    def clear_fault(self):
        self._active_fault = None
        self._fault_start_time = None

    def get_next_sample(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the next telemetry frame and applies active fault perturbations if any.
        """
        if self.data is None or len(self.data) == 0:
            return None

        if self.current_index >= len(self.data):
            self.current_index = 0  # Loop back

        row = self.data.iloc[self.current_index].to_dict()
        self.current_index += 1

        # Check if runtime injected fault is active
        if self._active_fault and self._fault_start_time:
            elapsed = time.time() - self._fault_start_time
            if elapsed > self._fault_duration_sec:
                self.clear_fault()
            else:
                row = self._apply_injected_fault(row, self._active_fault, elapsed)

        return row

    def _apply_injected_fault(self, row: Dict[str, Any], fault: str, elapsed: float) -> Dict[str, Any]:
        """Applies real-time synthetic corruption to normal telemetry matching trained physics"""
        row["injected_fault"] = fault
        row["anomaly_label"] = 1

        step_k = elapsed * self.sampling_rate_hz
        f = fault.lower()

        if "thermal" in f or "runaway" in f:
            row["temperature"] = float(row.get("temperature", 22.0)) + 12.0 + 1.25 * step_k + float(self.rng.normal(0, 0.2))
            row["voltage"] = max(1.5, float(row.get("voltage", 3.7)) - 0.15 - 0.015 * step_k)
            row["current"] = float(row.get("current", 2.5)) + 0.40 + 0.045 * step_k

        elif "short" in f:
            row["voltage"] = max(1.80, float(row.get("voltage", 3.7)) - 1.10 + float(self.rng.normal(0, 0.04)))
            row["current"] = float(row.get("current", 2.5)) + 4.80 + float(self.rng.normal(0, 0.15))
            row["temperature"] = float(row.get("temperature", 22.0)) + 6.0 + 0.45 * step_k

        elif "impedance" in f:
            curr = float(row.get("current", 2.5))
            row["voltage"] = max(1.90, float(row.get("voltage", 3.7)) - (curr * 0.55) - 0.45 + float(self.rng.normal(0, 0.02)))
            row["temperature"] = float(row.get("temperature", 22.0)) + 4.0 + 0.22 * step_k

        elif "sensor" in f or "drift" in f or "glitch" in f:
            row["voltage"] = float(row.get("voltage", 3.7)) + float(1.25 * np.sin(step_k / 2.5)) + float(self.rng.normal(0, 0.15))
            row["current"] = float(row.get("current", 2.5)) + float(1.80 * np.cos(step_k / 2.0))

        elif "undervolt" in f:
            row["voltage"] = max(1.95, 2.35 - 0.025 * step_k + float(self.rng.normal(0, 0.02)))
            row["soc"] = max(0.02, 0.12 - 0.002 * step_k)
            row["temperature"] = float(row.get("temperature", 22.0)) - 0.05 * step_k

        return row
