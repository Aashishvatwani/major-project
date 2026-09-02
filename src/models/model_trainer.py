"""
Comprehensive Model Trainer Pipeline
Trains Preprocessor, Random Forest, XGBoost, Extra Trees, Soft Voting Ensemble,
Multi-Class Fault Diagnosis Classifier, and the Reinforcement Learning Adaptive Mitigation Agent
with rigorous aerospace validation metrics.
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss
)

from src.ingestion.data_loader import TelemetryDataLoader
from src.features.feature_engineering import TelemetryFeatureExtractor
from src.features.preprocessor import TelemetryPreprocessor
from src.models.ensemble_classifier import SatelliteEnsembleClassifier
from src.models.fault_diagnosis import FaultDiagnosisClassifier, FAULT_CLASSES
from src.models.rl_agent import SatelliteMitigationEnv, RLAgent


class ModelTrainer:
    """
    End-to-end Trainer orchestrating data ingestion, feature extraction,
    supervised multi-model ensemble fitting, multi-class fault classification,
    calibration, and RL policy training.
    """

    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

        self.feature_extractor = TelemetryFeatureExtractor(
            rolling_windows=self.config.get("features", {}).get("rolling_windows", [5, 20, 60]),
            ewma_alpha=self.config.get("features", {}).get("ewma_alpha", 0.15)
        )
        self.preprocessor = TelemetryPreprocessor()
        self.ensemble = SatelliteEnsembleClassifier(
            weights=self.config.get("models", {}).get("ensemble", {}).get("voting_weights"),
            default_threshold=self.config.get("models", {}).get("ensemble", {}).get("default_threshold", 0.50)
        )
        self.fault_classifier = FaultDiagnosisClassifier()
        self.rl_agent = RLAgent(
            learning_rate=self.config.get("reinforcement_learning", {}).get("learning_rate", 0.08),
            discount_factor=self.config.get("reinforcement_learning", {}).get("discount_factor", 0.95)
        )

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def train_full_pipeline(
        self,
        duration_minutes: float = 360.0,
        output_dir: str = "saved_models",
        rl_episodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end training and saves all components.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(">> [1/6] Ingesting & generating telemetry dataset with realistic orbital fault modes...")
        loader = TelemetryDataLoader(data_path="data/raw/synthetic_telemetry.csv")
        df_raw = loader.load_or_generate_dataset(duration_minutes=duration_minutes)

        print(">> [2/6] Extracting physics and multi-window statistical features...")
        df_feats = self.feature_extractor.extract_batch_features(df_raw)

        # Stratified Train / Test split
        strat_col = df_feats["anomaly_label"] if len(df_feats["anomaly_label"].unique()) > 1 else None
        train_df, test_df = train_test_split(df_feats, test_size=0.25, random_state=42, stratify=strat_col)

        y_train = train_df["anomaly_label"].values
        y_test = test_df["anomaly_label"].values

        y_type_train = train_df.get("anomaly_type", pd.Series(["normal"] * len(train_df))).values
        y_type_test = test_df.get("anomaly_type", pd.Series(["normal"] * len(test_df))).values

        print(">> [3/6] Fitting robust scaler and preprocessing...")
        X_train_scaled = self.preprocessor.fit_transform(train_df)
        X_test_scaled = self.preprocessor.transform(test_df)

        print(">> [4/6] Training Random Forest, XGBoost & Extra Trees Ensemble with Probability Calibration...")
        self.ensemble.fit(X_train_scaled, y_train)

        # Evaluate Supervised Binary Models
        probs_train = self.ensemble.predict_proba(X_train_scaled)
        probs_test = self.ensemble.predict_proba(X_test_scaled)
        ind_probs_test = self.ensemble.predict_individual_proba(X_test_scaled)

        metrics = self._evaluate_models(y_test, probs_test, ind_probs_test)

        print(">> [5/6] Training Multi-Class Fault Taxonomy Diagnosis Classifier...")
        self.fault_classifier.fit(X_train_scaled, y_type_train)

        print(">> [6/6] Training Reinforcement Learning Adaptive Mitigation Agent (Q-Learning)...")
        rl_env = SatelliteMitigationEnv(
            telemetry_df=train_df,
            ensemble_probs=probs_train,
            reward_weights=self.config.get("reinforcement_learning", {}).get("reward_weights"),
            max_episode_steps=1200
        )
        episodes = rl_episodes or self.config.get("reinforcement_learning", {}).get("training_episodes", 40)
        rl_rewards = self.rl_agent.train_on_environment(rl_env, episodes=episodes)

        print(">> Saving trained pipeline artifacts to disk...")
        self.preprocessor.save(os.path.join(output_dir, "preprocessor.joblib"))
        self.ensemble.save(os.path.join(output_dir, "ensemble_model.joblib"))
        self.fault_classifier.save(os.path.join(output_dir, "fault_diagnosis.joblib"))
        self.rl_agent.save(os.path.join(output_dir, "rl_agent.json"))

        feature_importances = self.ensemble.get_feature_importances(self.preprocessor.feature_names)

        results = {
            "metrics": metrics,
            "feature_names": self.preprocessor.feature_names,
            "feature_importances": feature_importances,
            "rl_final_reward": float(np.mean(rl_rewards[-10:])) if rl_rewards else 0.0,
            "test_samples": len(test_df),
            "test_anomalies": int(y_test.sum())
        }

        print("=" * 65)
        print(" SATELLITE HITL PIPELINE TRAINING COMPLETE")
        print(f" Ensemble Test ROC-AUC: {metrics['ensemble']['roc_auc']:.4f}")
        print(f" Ensemble Test PR-AUC:   {metrics['ensemble']['pr_auc']:.4f}")
        print(f" Ensemble Test F1-Score: {metrics['ensemble']['f1']:.4f}")
        print(f" Ensemble Precision:     {metrics['ensemble']['precision']:.4f}")
        print(f" Ensemble Recall:        {metrics['ensemble']['recall']:.4f}")
        print("=" * 65)

        return results

    def _evaluate_models(
        self,
        y_true: np.ndarray,
        ens_probs: np.ndarray,
        ind_probs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Calculates detailed metrics for each model and the ensemble"""
        metrics = {}
        models_to_eval = {
            "random_forest": ind_probs["rf"],
            "xgboost": ind_probs["xgboost"],
            "extra_trees": ind_probs["extra_trees"],
            "ensemble": ens_probs
        }

        for name, probs in models_to_eval.items():
            preds = (probs >= 0.50).astype(int)
            roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
            pr = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
            f1 = f1_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            brier = brier_score_loss(y_true, probs)
            cm = confusion_matrix(y_true, preds).tolist()

            metrics[name] = {
                "roc_auc": float(roc),
                "pr_auc": float(pr),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "brier_score": float(brier),
                "confusion_matrix": cm
            }

        return metrics
