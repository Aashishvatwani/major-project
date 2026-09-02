import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from src.features.feature_engineering import TelemetryFeatureExtractor
from src.models.model_registry import ModelRegistry

preprocessor, ens, rl_agent, fault_cls = ModelRegistry.load_pipeline_artifacts("saved_models")

df = pd.read_csv('data/raw/synthetic_telemetry.csv')
ext = TelemetryFeatureExtractor()
feats_df = ext.extract_batch_features(df)

for f_type in ['normal', 'thermal_runaway_precursor', 'internal_short_circuit', 'high_impedance_degradation', 'sensor_drift_fault', 'deep_undervoltage_collapse']:
    subset = feats_df[feats_df['anomaly_type'] == f_type]
    if len(subset) > 0:
        sample_idx = subset.index[len(subset)//2]
        X = feats_df[preprocessor.feature_names].iloc[[sample_idx]]
        X_scaled = preprocessor.transform(X)
        p_ens = ens.predict_proba(X_scaled)[0]
        diag = fault_cls.predict_diagnosis(X_scaled)
        print(f"{f_type:30s} -> P_ens: {p_ens:.4f}, Pred Fault: {diag['primary_fault']} (Conf: {diag['diagnosis_confidence']:.2f})")
