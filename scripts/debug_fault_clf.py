import sys
import os
sys.path.insert(0, os.path.abspath("."))
import joblib
import pandas as pd
import numpy as np
from src.features.feature_engineering import TelemetryFeatureExtractor
from src.models.fault_diagnosis import FaultDiagnosisClassifier, FAULT_CLASSES

prep_data = joblib.load('saved_models/preprocessor.joblib')
scaler = prep_data['scaler']
feat_names = prep_data['feature_names']
fault_clf = FaultDiagnosisClassifier.load('saved_models/fault_diagnosis.joblib')

print("Classes in fault_clf:", fault_clf.classes_)
model = fault_clf.calibrated_model
print("Model classes_:", getattr(model, 'classes_', None))

df = pd.read_csv('data/raw/synthetic_telemetry.csv')
ext = TelemetryFeatureExtractor()
feats = ext.extract_batch_features(df)

for f_type in ['normal', 'thermal_runaway_precursor', 'internal_short_circuit', 'high_impedance_degradation', 'sensor_drift_fault', 'deep_undervoltage_collapse']:
    sub = feats[feats['anomaly_type'] == f_type]
    if len(sub) > 0:
        sample_idx = sub.index[len(sub)//2]
        X = feats[feat_names].iloc[[sample_idx]]
        X_s = scaler.transform(X)
        diag = fault_clf.predict_diagnosis(X_s, anomaly_prob=0.9)
        probs = model.predict_proba(X_s)[0]
        print(f"{f_type:30s} -> Probs: {np.round(probs, 3)} | Pred: {diag['primary_fault']} (Conf: {diag['diagnosis_confidence']:.2f})")
