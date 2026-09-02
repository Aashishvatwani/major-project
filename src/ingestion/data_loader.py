"""
Telemetry Data Loader for NASA Battery Prognostics and Synthetic Telemetry
Supports loading NASA Ames Battery data (B0005, B0006, etc.), standard CSV streams,
and dynamically generating synthetic training datasets.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from data.synthetic_generator import SatelliteTelemetryGenerator


class TelemetryDataLoader:
    """
    Unified Data Loader for Satellite Battery Telemetry.
    Normalizes columns to [timestamp, voltage, current, temperature, anomaly_label].
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def load_or_generate_dataset(
        self,
        duration_minutes: float = 360.0,
        anomaly_ratio: float = 0.12,
        force_regenerate: bool = False
    ) -> pd.DataFrame:
        """
        Loads dataset from file if present, or generates synthetic telemetry.
        """
        if self.data_path and os.path.exists(self.data_path) and not force_regenerate:
            df = pd.read_csv(self.data_path)
            return self._standardize_columns(df)

        # Generate synthetic satellite telemetry
        generator = SatelliteTelemetryGenerator()
        df = generator.generate_telemetry_dataset(
            duration_minutes=duration_minutes,
            anomaly_ratio=anomaly_ratio,
            inject_anomalies=True
        )

        if self.data_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.data_path)), exist_ok=True)
            df.to_csv(self.data_path, index=False)

        return self._standardize_columns(df)

    def load_nasa_battery_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads NASA Ames Battery Prognostics CSV format (e.g. B0005.csv / B0006.csv).
        Maps Voltage_measured, Current_measured, Temperature_measured, Time.
        """
        df = pd.read_csv(file_path)
        col_map = {}
        for col in df.columns:
            low = col.lower()
            if "volt" in low:
                col_map[col] = "voltage"
            elif "curr" in low:
                col_map[col] = "current"
            elif "temp" in low:
                col_map[col] = "temperature"
            elif "time" in low:
                col_map[col] = "timestamp"

        df = df.rename(columns=col_map)
        if "timestamp" not in df.columns:
            df["timestamp"] = np.arange(len(df)) * 1.0

        if "anomaly_label" not in df.columns:
            # Synthetic thresholding or default 0 if unlabelled
            df["anomaly_label"] = 0

        return self._standardize_columns(df)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard column names and types"""
        required = ["timestamp", "voltage", "current", "temperature"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required telemetry column: '{col}' in DataFrame")

        if "anomaly_label" not in df.columns:
            df["anomaly_label"] = 0

        # Sort by timestamp
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        return df
