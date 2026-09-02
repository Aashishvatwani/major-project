"""
Unit Tests for Multi-Model Ensemble ML Classifiers
"""

import pytest
import numpy as np
import os
from src.models.ensemble_classifier import SatelliteEnsembleClassifier


def test_ensemble_classifier_fit_predict(tmp_path):
    """Verify RF + XGBoost + Extra Trees ensemble fitting, predictions, and calibration"""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Synthetic target with clear boundary
    y = ((X[:, 0] + X[:, 1] * 2) > 1.0).astype(int)

    ensemble = SatelliteEnsembleClassifier(
        weights={"rf": 0.35, "xgboost": 0.40, "extra_trees": 0.25},
        calibrate=True,
        n_estimators=20
    )

    ensemble.fit(X, y)
    assert ensemble.is_fitted

    ind_probs = ensemble.predict_individual_proba(X)
    assert "rf" in ind_probs
    assert "xgboost" in ind_probs
    assert "extra_trees" in ind_probs
    assert len(ind_probs["rf"]) == n_samples

    # Verify probability bounds [0, 1]
    ens_probs = ensemble.predict_proba(X)
    assert np.all(ens_probs >= 0.0) and np.all(ens_probs <= 1.0)

    # Test single-sample detailed inference
    detail = ensemble.predict_single_detailed(X[0])
    assert "p_ensemble" in detail
    assert "severity" in detail
    assert detail["binary_decision"] in [0, 1]

    # Test persistence
    save_file = os.path.join(tmp_path, "test_ens.joblib")
    ensemble.save(save_file)
    loaded_ens = SatelliteEnsembleClassifier.load(save_file)
    assert loaded_ens.is_fitted
    assert np.allclose(loaded_ens.predict_proba(X[:5]), ens_probs[:5])
