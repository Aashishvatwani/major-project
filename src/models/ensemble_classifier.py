"""
Satellite Telemetry Multi-Model ML Ensemble Classifier
Implements Random Forest, XGBoost, and Extra Trees classifiers with Soft/Weighted Voting
and Probability Calibration for high-reliability aerospace anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib


class SatelliteEnsembleClassifier:
    """
    Ensemble ML Classifier combining:
    1. Random Forest (Bagging, non-linear interaction modeling)
    2. XGBoost (Gradient Boosting, sharp decision boundaries)
    3. Extra Trees (Extremely Randomized Trees, variance reduction)
    4. Soft Weighted Voting Aggregator
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        default_threshold: float = 0.50,
        calibrate: bool = True,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.weights = weights or {"rf": 0.35, "xgboost": 0.40, "extra_trees": 0.25}
        self._normalize_weights()
        self.default_threshold = default_threshold
        self.calibrate = calibrate
        self.random_state = random_state
        self.n_estimators = n_estimators

        # Base Classifiers
        self.rf_base = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )

        self.xgb_base = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )

        self.et_base = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )

        self.rf_model = None
        self.xgb_model = None
        self.et_model = None
        self.is_fitted = False

    def _normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SatelliteEnsembleClassifier":
        """
        Fits all three classifiers and applies probability calibration if enabled.
        """
        if self.calibrate:
            # Calibrated ensemble for well-aligned posterior probabilities
            self.rf_model = CalibratedClassifierCV(estimator=self.rf_base, method="sigmoid", cv=3)
            self.xgb_model = CalibratedClassifierCV(estimator=self.xgb_base, method="sigmoid", cv=3)
            self.et_model = CalibratedClassifierCV(estimator=self.et_base, method="sigmoid", cv=3)
        else:
            self.rf_model = self.rf_base
            self.xgb_model = self.xgb_base
            self.et_model = self.et_base

        self.rf_model.fit(X, y)
        self.xgb_model.fit(X, y)
        self.et_model.fit(X, y)

        self.is_fitted = True
        return self

    def predict_individual_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns anomaly probabilities [P(Anomaly | x)] for each individual model.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict")

        rf_p = self.rf_model.predict_proba(X)[:, 1]
        xgb_p = self.xgb_model.predict_proba(X)[:, 1]
        et_p = self.et_model.predict_proba(X)[:, 1]

        return {
            "rf": rf_p,
            "xgboost": xgb_p,
            "extra_trees": et_p
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Soft / Weighted Ensemble Probability:
        P_ens = w_rf * P_rf + w_xgb * P_xgb + w_et * P_et
        """
        probs = self.predict_individual_proba(X)
        p_ens = (
            self.weights["rf"] * probs["rf"] +
            self.weights["xgboost"] * probs["xgboost"] +
            self.weights["extra_trees"] * probs["extra_trees"]
        )
        return p_ens

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Binary Classification Decision (0 = Normal, 1 = Anomaly)
        """
        thresh = threshold if threshold is not None else self.default_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def predict_single_detailed(self, x_vector: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Performs detailed single-sample inference for real-time telemetry streaming.
        """
        if x_vector.ndim == 1:
            x_vector = x_vector.reshape(1, -1)

        probs = self.predict_individual_proba(x_vector)
        rf_p = float(probs["rf"][0])
        xgb_p = float(probs["xgboost"][0])
        et_p = float(probs["extra_trees"][0])

        p_ens = float(
            self.weights["rf"] * rf_p +
            self.weights["xgboost"] * xgb_p +
            self.weights["extra_trees"] * et_p
        )

        thresh = threshold if threshold is not None else self.default_threshold
        binary_decision = int(p_ens >= thresh)

        # Multi-level severity categorization
        if p_ens < 0.30:
            severity = "NORMAL"
        elif p_ens < 0.60:
            severity = "ELEVATED"
        elif p_ens < 0.85:
            severity = "WARNING"
        else:
            severity = "CRITICAL"

        return {
            "p_rf": rf_p,
            "p_xgboost": xgb_p,
            "p_extra_trees": et_p,
            "p_ensemble": p_ens,
            "threshold_used": thresh,
            "binary_decision": binary_decision,
            "severity": severity
        }

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Extracts feature importances from base estimators.
        """
        # If calibrated, access base estimators from cv
        importances = {}
        try:
            if hasattr(self.rf_model, "calibrated_classifiers_"):
                rf_imp = np.mean([clf.estimator.feature_importances_ for clf in self.rf_model.calibrated_classifiers_], axis=0)
                xgb_imp = np.mean([clf.estimator.feature_importances_ for clf in self.xgb_model.calibrated_classifiers_], axis=0)
                et_imp = np.mean([clf.estimator.feature_importances_ for clf in self.et_model.calibrated_classifiers_], axis=0)
            else:
                rf_imp = self.rf_model.feature_importances_
                xgb_imp = self.xgb_model.feature_importances_
                et_imp = self.et_model.feature_importances_

            importances["rf"] = dict(zip(feature_names, rf_imp))
            importances["xgboost"] = dict(zip(feature_names, xgb_imp))
            importances["extra_trees"] = dict(zip(feature_names, et_imp))
            
            # Weighted average importance
            ensemble_imp = (
                self.weights["rf"] * rf_imp +
                self.weights["xgboost"] * xgb_imp +
                self.weights["extra_trees"] * et_imp
            )
            importances["ensemble"] = dict(zip(feature_names, ensemble_imp))
        except Exception:
            pass

        return importances

    def save(self, filepath: str):
        """Saves entire ensemble model to disk"""
        joblib.dump({
            "rf_model": self.rf_model,
            "xgb_model": self.xgb_model,
            "et_model": self.et_model,
            "weights": self.weights,
            "default_threshold": self.default_threshold,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "SatelliteEnsembleClassifier":
        """Loads ensemble model from disk"""
        data = joblib.load(filepath)
        obj = cls(weights=data["weights"], default_threshold=data["default_threshold"])
        obj.rf_model = data["rf_model"]
        obj.xgb_model = data["xgb_model"]
        obj.et_model = data["et_model"]
        obj.is_fitted = data["is_fitted"]
        return obj
