"""
Satellite Telemetry Multi-Class Fault Diagnosis & Severity Estimation Engine
Classifies specific failure modes (Thermal, Internal Short, High Impedance, Sensor Fault, Undervoltage)
and evaluates risk levels and severity states (Nominal, Warning, Critical, Emergency).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib


FAULT_CLASSES = [
    "NOMINAL",
    "THERMAL_RUNAWAY",
    "INTERNAL_SHORT",
    "HIGH_IMPEDANCE",
    "SENSOR_FAULT",
    "UNDERVOLTAGE"
]

FAULT_MAP = {
    "normal": "NOMINAL",
    "nominal": "NOMINAL",
    "thermal_runaway_precursor": "THERMAL_RUNAWAY",
    "thermal_runaway": "THERMAL_RUNAWAY",
    "internal_short_circuit": "INTERNAL_SHORT",
    "internal_short": "INTERNAL_SHORT",
    "high_impedance_degradation": "HIGH_IMPEDANCE",
    "high_impedance": "HIGH_IMPEDANCE",
    "sensor_drift_fault": "SENSOR_FAULT",
    "sensor_drift": "SENSOR_FAULT",
    "sensor_fault": "SENSOR_FAULT",
    "deep_undervoltage_collapse": "UNDERVOLTAGE",
    "undervoltage": "UNDERVOLTAGE"
}

SEVERITY_LEVELS = ["NOMINAL", "WARNING", "CRITICAL", "EMERGENCY"]


class FaultDiagnosisClassifier:
    """
    Multi-class classifier predicting root-cause satellite fault taxonomy
    with calibrated posterior probabilities.
    """

    def __init__(self, random_state: int = 42, n_estimators: int = 120):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.classes_ = FAULT_CLASSES
        self.base_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_labels: List[str]) -> "FaultDiagnosisClassifier":
        """
        Fits multi-class classifier on scaled feature matrix and fault type labels.
        """
        standardized_y = [FAULT_MAP.get(str(y).lower(), "NOMINAL") for y in y_labels]
        label_indices = np.array([FAULT_CLASSES.index(lbl) if lbl in FAULT_CLASSES else 0 for lbl in standardized_y])

        unique_classes = np.unique(label_indices)
        if len(unique_classes) > 1:
            self.calibrated_model = CalibratedClassifierCV(estimator=self.base_model, method="sigmoid", cv=3)
            self.calibrated_model.fit(X, label_indices)
        else:
            self.base_model.fit(X, label_indices)
            self.calibrated_model = self.base_model

        self.is_fitted = True
        return self

    def predict_diagnosis(self, X_single: np.ndarray, anomaly_prob: Optional[float] = None) -> Dict[str, Any]:
        """
        Predicts fault class probabilities and primary diagnosis for a single sample.
        """
        if not self.is_fitted or self.calibrated_model is None:
            return {
                "primary_fault": "NOMINAL",
                "diagnosis_confidence": 1.0,
                "fault_probabilities": {fc: (1.0 if fc == "NOMINAL" else 0.0) for fc in FAULT_CLASSES}
            }

        X_2d = X_single.reshape(1, -1) if X_single.ndim == 1 else X_single
        try:
            probs = self.calibrated_model.predict_proba(X_2d)[0]
            classes_in_model = getattr(self.calibrated_model, "classes_", np.arange(len(probs)))
            
            prob_dict = {fc: 0.0 for fc in FAULT_CLASSES}
            for idx, c_idx in enumerate(classes_in_model):
                if c_idx < len(FAULT_CLASSES):
                    prob_dict[FAULT_CLASSES[c_idx]] = float(probs[idx])

            # Non-nominal candidate ranking
            non_nominal_probs = {k: v for k, v in prob_dict.items() if k != "NOMINAL"}
            best_fault = max(non_nominal_probs, key=non_nominal_probs.get)
            best_fault_prob = non_nominal_probs[best_fault]

            # If anomaly probability is elevated (>0.35) or best fault probability is high (>0.25), choose the specific fault
            if (anomaly_prob is not None and anomaly_prob >= 0.35 and best_fault_prob >= 0.15) or (best_fault_prob >= 0.30):
                top_fault = best_fault
                confidence = float(best_fault_prob)
            else:
                top_fault = max(prob_dict, key=prob_dict.get)
                confidence = float(prob_dict[top_fault])
        except Exception:
            prob_dict = {fc: (1.0 if fc == "NOMINAL" else 0.0) for fc in FAULT_CLASSES}
            top_fault = "NOMINAL"
            confidence = 1.0

        return {
            "primary_fault": top_fault,
            "diagnosis_confidence": confidence,
            "fault_probabilities": prob_dict
        }

    def save(self, filepath: str):
        joblib.dump({"model": self.calibrated_model, "classes": self.classes_}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "FaultDiagnosisClassifier":
        data = joblib.load(filepath)
        clf = cls()
        clf.calibrated_model = data["model"]
        clf.classes_ = data.get("classes", FAULT_CLASSES)
        clf.is_fitted = True
        return clf


class SeverityEstimator:
    """
    Computes continuous Risk Index (0.0 to 1.0) and categorical severity state.
    """

    @staticmethod
    def estimate_severity(
        telemetry: Dict[str, float],
        anomaly_prob: float,
        fault_type: str = "NOMINAL",
        safety_override: bool = False
    ) -> Dict[str, Any]:
        temp = telemetry.get("temperature", 22.0)
        volt = telemetry.get("voltage", 3.7)
        curr = telemetry.get("current", 2.5)
        imp = telemetry.get("impedance_proxy", 0.045)
        dtdt = abs(telemetry.get("thermal_gradient", 0.0))

        # Risk Factors
        r_temp = np.clip((temp - 40.0) / 25.0, 0.0, 1.0)
        r_volt = np.clip((3.0 - volt) / 1.0, 0.0, 1.0)
        r_curr = np.clip((curr - 4.5) / 3.5, 0.0, 1.0)
        r_imp = np.clip((imp - 0.15) / 0.5, 0.0, 1.0)
        r_dtdt = np.clip((dtdt - 1.0) / 2.0, 0.0, 1.0)

        physics_risk = float(0.35 * r_temp + 0.25 * r_volt + 0.20 * r_curr + 0.10 * r_imp + 0.10 * r_dtdt)
        risk_score = float(0.60 * anomaly_prob + 0.40 * physics_risk)

        if safety_override or temp >= 65.0 or volt <= 2.20 or curr >= 7.5 or risk_score >= 0.85:
            severity = "EMERGENCY"
        elif risk_score >= 0.60 or fault_type in ["THERMAL_RUNAWAY", "INTERNAL_SHORT"]:
            severity = "CRITICAL"
        elif risk_score >= 0.35 or fault_type != "NOMINAL":
            severity = "WARNING"
        else:
            severity = "NOMINAL"

        return {
            "severity": severity,
            "risk_score": float(np.clip(risk_score, 0.0, 1.0)),
            "physics_risk": physics_risk
        }
