"""
Feature Preprocessor and Scaler Pipeline
Manages column alignment, robust scaling, and persistence of preprocessing artifacts.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import RobustScaler
import joblib


class TelemetryPreprocessor:
    """
    Standardizes feature matrices for Machine Learning training and inference.
    Uses RobustScaler (median and IQR) to be resilient against extreme orbital sensor spikes.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        self.scaler = RobustScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "TelemetryPreprocessor":
        """Fits scaler on training features"""
        self.feature_names = [col for col in X.columns if col not in ["timestamp", "anomaly_label", "anomaly_type", "injected_fault"]]
        X_sub = X[self.feature_names].copy().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        self.scaler.fit(X_sub)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame to scaled numpy feature matrix"""
        if not self.is_fitted:
            raise RuntimeError("TelemetryPreprocessor must be fitted before transform")

        # Align columns
        aligned_df = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            aligned_df[col] = X[col] if col in X.columns else 0.0

        aligned_clean = aligned_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return self.scaler.transform(aligned_clean)

    def transform_single(self, feature_dict: dict) -> np.ndarray:
        """Transforms a single feature dictionary into 2D array [1, n_features]"""
        df = pd.DataFrame([feature_dict])
        return self.transform(df)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

    def save(self, filepath: str):
        """Saves fitted preprocessor to disk"""
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names, "is_fitted": self.is_fitted}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "TelemetryPreprocessor":
        """Loads fitted preprocessor from disk"""
        data = joblib.load(filepath)
        obj = cls(feature_names=data["feature_names"])
        obj.scaler = data["scaler"]
        obj.is_fitted = data["is_fitted"]
        return obj
