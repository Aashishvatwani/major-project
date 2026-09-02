"""
Unit Tests for Telemetry Physics-Informed Feature Engineering
"""

import pytest
import numpy as np
import pandas as pd
from src.features.feature_engineering import TelemetryFeatureExtractor


def test_batch_feature_extraction():
    """Verify that batch extraction properly computes physics and rolling features"""
    extractor = TelemetryFeatureExtractor(rolling_windows=[5, 10], ewma_alpha=0.2)

    df_raw = pd.DataFrame({
        "timestamp": np.arange(30) * 0.5,
        "voltage": 3.7 + 0.1 * np.sin(np.arange(30)),
        "current": 2.5 + 0.05 * np.cos(np.arange(30)),
        "temperature": 22.0 + 0.2 * np.arange(30),
        "anomaly_label": np.zeros(30, dtype=int)
    })

    feats = extractor.extract_batch_features(df_raw)

    assert "power_watts" in feats.columns
    assert "apparent_resistance" in feats.columns
    assert "impedance_proxy" in feats.columns
    assert "thermal_gradient" in feats.columns
    assert "v_mean_w5" in feats.columns
    assert "v_ewma_zscore" in feats.columns
    assert not feats.isna().any().any()
    assert len(feats) == 30


def test_streaming_feature_extraction():
    """Verify that streaming ring-buffer extraction matches expected physics outputs"""
    extractor = TelemetryFeatureExtractor(rolling_windows=[5], ewma_alpha=0.2)
    extractor.reset_stream_buffer()

    samples = [
        {"timestamp": 0.0, "voltage": 3.70, "current": 2.50, "temperature": 20.0},
        {"timestamp": 0.5, "voltage": 3.65, "current": 3.00, "temperature": 21.0},
        {"timestamp": 1.0, "voltage": 3.60, "current": 3.50, "temperature": 22.5}
    ]

    res = None
    for s in samples:
        res = extractor.extract_streaming_features(s)

    assert res is not None
    assert res["voltage"] == 3.60
    assert res["power_watts"] == pytest.approx(3.60 * 3.50, rel=1e-3)
    # delta_v = -0.05, delta_i = 0.5 -> R_int = |-0.05/0.5| = 0.10
    assert res["impedance_proxy"] == pytest.approx(0.10, rel=1e-2)
    # thermal gradient = (22.5 - 21.0) / 0.5 = 3.0 °C/s
    assert res["thermal_gradient"] == pytest.approx(3.0, rel=1e-2)
