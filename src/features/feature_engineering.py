"""
Satellite Telemetry Physics-Informed Feature Engineering
Extracts electrochemical proxies, dynamic internal resistance, thermal gradients,
multi-horizon rolling window statistics, and EWMA Z-scores for both batch training and real-time streaming.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import deque


class TelemetryFeatureExtractor:
    """
    Feature extraction engine supporting:
    - Batch tabular processing (for dataset preparation & ML training)
    - Real-time rolling buffer processing (for online HITL inference)
    """

    def __init__(
        self,
        rolling_windows: Optional[List[int]] = None,
        ewma_alpha: float = 0.15,
        dt: float = 0.5
    ):
        self.rolling_windows = rolling_windows or [5, 20, 60]
        self.ewma_alpha = ewma_alpha
        self.dt = dt
        self.max_window = max(self.rolling_windows) if self.rolling_windows else 60

        # Streaming ring buffers for online inference
        self.v_buffer = deque(maxlen=self.max_window)
        self.i_buffer = deque(maxlen=self.max_window)
        self.t_buffer = deque(maxlen=self.max_window)
        self.time_buffer = deque(maxlen=self.max_window)

        # EWMA tracking state
        self.ewma_v = None
        self.ewma_i = None
        self.ewma_t = None
        self.ewm_var_v = None
        self.ewm_var_i = None
        self.ewm_var_t = None

    def reset_stream_buffer(self):
        """Clears streaming ring buffers"""
        self.v_buffer.clear()
        self.i_buffer.clear()
        self.t_buffer.clear()
        self.time_buffer.clear()
        self.ewma_v = None
        self.ewma_i = None
        self.ewma_t = None
        self.ewm_var_v = None
        self.ewm_var_i = None
        self.ewm_var_t = None

    def extract_batch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw time-series DataFrame [timestamp, voltage, current, temperature]
        into an enriched feature matrix.
        """
        feats = df.copy()

        # 1. Physics-Based Features
        feats["power_watts"] = feats["voltage"] * feats["current"]
        
        # Apparent resistance R = V / max(I, 1e-4)
        safe_current = np.where(np.abs(feats["current"]) < 1e-4, 1e-4, feats["current"])
        feats["apparent_resistance"] = feats["voltage"] / safe_current

        # First and second order derivatives
        feats["delta_v"] = feats["voltage"].diff().fillna(0.0)
        feats["delta_i"] = feats["current"].diff().fillna(0.0)
        feats["delta_t"] = feats["temperature"].diff().fillna(0.0)

        feats["v_acceleration"] = feats["delta_v"].diff().fillna(0.0)

        # Dynamic internal resistance proxy R_int = |delta_V| / |delta_I| (evaluated when delta_I >= 0.03A)
        valid_di_mask = np.abs(feats["delta_i"]) >= 0.03
        raw_r_int = np.abs(feats["delta_v"]) / np.maximum(np.abs(feats["delta_i"]), 1e-3)
        feats["impedance_proxy"] = np.where(valid_di_mask, np.clip(raw_r_int, 0.01, 3.0), 0.045)

        # Joule heat dissipation proxy = I^2 * R_int
        feats["joule_dissipation"] = (feats["current"] ** 2) * feats["impedance_proxy"]
        
        # Thermal gradient (dT/dt)
        delta_time = feats["timestamp"].diff().replace(0, self.dt).fillna(self.dt)
        feats["thermal_gradient"] = feats["delta_t"] / delta_time

        # 2. Multi-Horizon Rolling Statistics
        for w in self.rolling_windows:
            feats[f"v_mean_w{w}"] = feats["voltage"].rolling(w, min_periods=1).mean()
            feats[f"v_std_w{w}"] = feats["voltage"].rolling(w, min_periods=1).std().fillna(0.0)
            feats[f"v_ptp_w{w}"] = (
                feats["voltage"].rolling(w, min_periods=1).max() - feats["voltage"].rolling(w, min_periods=1).min()
            )

            feats[f"i_mean_w{w}"] = feats["current"].rolling(w, min_periods=1).mean()
            feats[f"i_std_w{w}"] = feats["current"].rolling(w, min_periods=1).std().fillna(0.0)
            feats[f"i_ptp_w{w}"] = (
                feats["current"].rolling(w, min_periods=1).max() - feats["current"].rolling(w, min_periods=1).min()
            )

            feats[f"t_mean_w{w}"] = feats["temperature"].rolling(w, min_periods=1).mean()
            feats[f"t_std_w{w}"] = feats["temperature"].rolling(w, min_periods=1).std().fillna(0.0)
            feats[f"t_ptp_w{w}"] = (
                feats["temperature"].rolling(w, min_periods=1).max() - feats["temperature"].rolling(w, min_periods=1).min()
            )

        # 3. EWMA and Dynamic Z-Scores
        v_ewma = feats["voltage"].ewm(alpha=self.ewma_alpha, min_periods=1).mean()
        v_ewmstd = feats["voltage"].ewm(alpha=self.ewma_alpha, min_periods=1).std().fillna(1e-3)
        feats["v_ewma_zscore"] = (feats["voltage"] - v_ewma) / (v_ewmstd + 1e-4)

        i_ewma = feats["current"].ewm(alpha=self.ewma_alpha, min_periods=1).mean()
        i_ewmstd = feats["current"].ewm(alpha=self.ewma_alpha, min_periods=1).std().fillna(1e-3)
        feats["i_ewma_zscore"] = (feats["current"] - i_ewma) / (i_ewmstd + 1e-4)

        t_ewma = feats["temperature"].ewm(alpha=self.ewma_alpha, min_periods=1).mean()
        t_ewmstd = feats["temperature"].ewm(alpha=self.ewma_alpha, min_periods=1).std().fillna(1e-3)
        feats["t_ewma_zscore"] = (feats["temperature"] - t_ewma) / (t_ewmstd + 1e-4)

        # Replace any remaining NaNs or Infs
        feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return feats

    def extract_streaming_features(self, raw_sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Real-time feature calculation for a single streaming sample using sliding ring buffers.
        Returns a dictionary of all extracted feature values ready for model inference.
        """
        v = float(raw_sample["voltage"])
        i = float(raw_sample["current"])
        t = float(raw_sample["temperature"])
        ts = float(raw_sample.get("timestamp", 0.0))

        # Update buffers
        self.v_buffer.append(v)
        self.i_buffer.append(i)
        self.t_buffer.append(t)
        self.time_buffer.append(ts)

        v_arr = np.array(self.v_buffer)
        i_arr = np.array(self.i_buffer)
        t_arr = np.array(self.t_buffer)

        # 1. Physics Features
        power = v * i
        safe_i = i if abs(i) >= 1e-4 else 1e-4
        apparent_res = v / safe_i

        if len(v_arr) >= 2:
            delta_v = v_arr[-1] - v_arr[-2]
            delta_i = i_arr[-1] - i_arr[-2]
            delta_t = t_arr[-1] - t_arr[-2]
            dt_step = max(self.time_buffer[-1] - self.time_buffer[-2], 1e-3)
        else:
            delta_v, delta_i, delta_t, dt_step = 0.0, 0.0, 0.0, self.dt

        if len(v_arr) >= 3:
            v_accel = (v_arr[-1] - v_arr[-2]) - (v_arr[-2] - v_arr[-3])
        else:
            v_accel = 0.0

        if abs(delta_i) >= 0.03:
            impedance_proxy = float(np.clip(abs(delta_v) / abs(delta_i), 0.01, 3.0))
        else:
            impedance_proxy = 0.045
        joule_dissipation = (i ** 2) * impedance_proxy
        thermal_gradient = delta_t / dt_step

        features: Dict[str, float] = {
            "voltage": v,
            "current": i,
            "temperature": t,
            "power_watts": power,
            "apparent_resistance": apparent_res,
            "delta_v": delta_v,
            "delta_i": delta_i,
            "delta_t": delta_t,
            "v_acceleration": v_accel,
            "impedance_proxy": impedance_proxy,
            "joule_dissipation": joule_dissipation,
            "thermal_gradient": thermal_gradient,
        }

        # 2. Multi-Horizon Rolling Windows
        for w in self.rolling_windows:
            v_slice = v_arr[-w:]
            i_slice = i_arr[-w:]
            t_slice = t_arr[-w:]

            features[f"v_mean_w{w}"] = float(np.mean(v_slice))
            features[f"v_std_w{w}"] = float(np.std(v_slice)) if len(v_slice) > 1 else 0.0
            features[f"v_ptp_w{w}"] = float(np.max(v_slice) - np.min(v_slice))

            features[f"i_mean_w{w}"] = float(np.mean(i_slice))
            features[f"i_std_w{w}"] = float(np.std(i_slice)) if len(i_slice) > 1 else 0.0
            features[f"i_ptp_w{w}"] = float(np.max(i_slice) - np.min(i_slice))

            features[f"t_mean_w{w}"] = float(np.mean(t_slice))
            features[f"t_std_w{w}"] = float(np.std(t_slice)) if len(t_slice) > 1 else 0.0
            features[f"t_ptp_w{w}"] = float(np.max(t_slice) - np.min(t_slice))

        # 3. EWMA and Z-scores
        if self.ewma_v is None:
            self.ewma_v = v
            self.ewma_i = i
            self.ewma_t = t
            self.ewm_var_v = 1e-3
            self.ewm_var_i = 1e-3
            self.ewm_var_t = 1e-3
        else:
            diff_v = v - self.ewma_v
            diff_i = i - self.ewma_i
            diff_t = t - self.ewma_t

            self.ewma_v += self.ewma_alpha * diff_v
            self.ewma_i += self.ewma_alpha * diff_i
            self.ewma_t += self.ewma_alpha * diff_t

            self.ewm_var_v = (1.0 - self.ewma_alpha) * (self.ewm_var_v + self.ewma_alpha * (diff_v ** 2))
            self.ewm_var_i = (1.0 - self.ewma_alpha) * (self.ewm_var_i + self.ewma_alpha * (diff_i ** 2))
            self.ewm_var_t = (1.0 - self.ewma_alpha) * (self.ewm_var_t + self.ewma_alpha * (diff_t ** 2))

        features["v_ewma_zscore"] = float((v - self.ewma_v) / (np.sqrt(max(self.ewm_var_v, 1e-6)) + 1e-4))
        features["i_ewma_zscore"] = float((i - self.ewma_i) / (np.sqrt(max(self.ewm_var_i, 1e-6)) + 1e-4))
        features["t_ewma_zscore"] = float((t - self.ewma_t) / (np.sqrt(max(self.ewm_var_t, 1e-6)) + 1e-4))

        return features
