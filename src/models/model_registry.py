"""
Model Registry and Artifact Loader
Handles versioned loading and retrieval of trained preprocessors, ensembles, fault classifiers, and RL agents.
"""

import os
from typing import Tuple, Optional
from src.features.preprocessor import TelemetryPreprocessor
from src.models.ensemble_classifier import SatelliteEnsembleClassifier
from src.models.rl_agent import RLAgent
from src.models.fault_diagnosis import FaultDiagnosisClassifier


class ModelRegistry:
    """Convenience loader for trained model pipeline components"""

    @staticmethod
    def load_pipeline_artifacts(
        models_dir: str = "saved_models"
    ) -> Tuple[TelemetryPreprocessor, SatelliteEnsembleClassifier, RLAgent, FaultDiagnosisClassifier]:
        """
        Loads preprocessor, ensemble classifier, RL agent, and fault diagnosis classifier.
        """
        prep_path = os.path.join(models_dir, "preprocessor.joblib")
        ens_path = os.path.join(models_dir, "ensemble_model.joblib")
        rl_path = os.path.join(models_dir, "rl_agent.json")
        fault_path = os.path.join(models_dir, "fault_diagnosis.joblib")

        if not os.path.exists(prep_path) or not os.path.exists(ens_path) or not os.path.exists(rl_path):
            raise FileNotFoundError(
                f"Missing model artifacts in '{models_dir}'. Please run `python train.py` first."
            )

        preprocessor = TelemetryPreprocessor.load(prep_path)
        ensemble = SatelliteEnsembleClassifier.load(ens_path)
        rl_agent = RLAgent.load(rl_path)

        if os.path.exists(fault_path):
            fault_clf = FaultDiagnosisClassifier.load(fault_path)
        else:
            fault_clf = FaultDiagnosisClassifier()

        return preprocessor, ensemble, rl_agent, fault_clf
