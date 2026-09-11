"""
Comprehensive Model Trainer Pipeline
Trains Preprocessor, Random Forest, XGBoost, Extra Trees, Soft Voting Ensemble,
Multi-Class Fault Diagnosis Classifier, and the Reinforcement Learning Adaptive Mitigation Agent
with rigorous aerospace validation metrics and temporal anti-leakage splitting.
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
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
    calibration, and RL policy training with strict temporal anti-leakage isolation.
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
        Executes end-to-end training and saves all components using strict out-of-sample temporal splitting.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(">> [1/6] Ingesting & generating telemetry dataset with realistic orbital fault modes...")
        loader = TelemetryDataLoader(data_path="data/raw/synthetic_telemetry.csv")
        df_raw = loader.load_or_generate_dataset(duration_minutes=duration_minutes, force_regenerate=True)

        print(">> [2/6] Performing Temporal Orbit-Aware Split (preventing temporal & rolling feature leakage)...")
        # Split by orbit_id if available, otherwise chronological time split
        if "orbit_id" in df_raw.columns and len(df_raw["orbit_id"].unique()) > 1:
            orbits = sorted(df_raw["orbit_id"].unique())
            split_orbit = orbits[int(len(orbits) * 0.75)]
            train_raw = df_raw[df_raw["orbit_id"] < split_orbit].copy().reset_index(drop=True)
            test_raw = df_raw[df_raw["orbit_id"] >= split_orbit].copy().reset_index(drop=True)
            print(f"   Orbits for Training: {list(train_raw['orbit_id'].unique())} ({len(train_raw)} frames)")
            print(f"   Orbits for Testing (Unseen Out-of-Sample): {list(test_raw['orbit_id'].unique())} ({len(test_raw)} frames)")
        else:
            split_idx = int(len(df_raw) * 0.75)
            train_raw = df_raw.iloc[:split_idx].copy().reset_index(drop=True)
            test_raw = df_raw.iloc[split_idx:].copy().reset_index(drop=True)
            print(f"   Chronological Split: {len(train_raw)} train frames / {len(test_raw)} test frames")

        # Independent feature extraction on separate splits to prevent lookahead/rolling window leakage
        print(">> Extracting features independently on Train and Test partitions...")
        train_df = self.feature_extractor.extract_batch_features(train_raw)
        test_df = self.feature_extractor.extract_batch_features(test_raw)

        y_train = train_df["anomaly_label"].values
        y_test = test_df["anomaly_label"].values

        y_type_train = train_df.get("anomaly_type", pd.Series(["normal"] * len(train_df))).values
        y_type_test = test_df.get("anomaly_type", pd.Series(["normal"] * len(test_df))).values

        print(">> [3/6] Fitting robust scaler strictly on training partition...")
        X_train_scaled = self.preprocessor.fit_transform(train_df)
        X_test_scaled = self.preprocessor.transform(test_df)

        print(">> [4/6] Training Random Forest, XGBoost & Extra Trees Ensemble with Probability Calibration...")
        self.ensemble.fit(X_train_scaled, y_train)

        # Evaluate Supervised Binary Models on unseen test partition
        probs_train = self.ensemble.predict_proba(X_train_scaled)
        probs_test = self.ensemble.predict_proba(X_test_scaled)
        ind_probs_test = self.ensemble.predict_individual_proba(X_test_scaled)

        metrics = self._evaluate_models(test_df, probs_test, ind_probs_test)

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
        print(" SATELLITE HITL PIPELINE TRAINING COMPLETE (REALISTIC BENCHMARKS)")
        print(f" Ensemble Test ROC-AUC: {metrics['ensemble']['roc_auc']:.4f}")
        print(f" Ensemble Test PR-AUC:   {metrics['ensemble']['pr_auc']:.4f}")
        print(f" Ensemble Test F1-Score: {metrics['ensemble']['f1']:.4f}")
        print(f" Ensemble Precision:     {metrics['ensemble']['precision']:.4f}")
        print(f" Ensemble Recall:        {metrics['ensemble']['recall']:.4f}")
        print(f" Mean Time-to-Detect:    {metrics['ensemble']['mean_time_to_detect_sec']:.2f} s")
        print(f" False Alarms / Orbit Hr:{metrics['ensemble']['false_alarms_per_hour']:.2f}")
        print("=" * 65)

        return results

    def _evaluate_models(
        self,
        test_df: pd.DataFrame,
        ens_probs: np.ndarray,
        ind_probs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Calculates frame-level and event-level metrics for each model and the ensemble"""
        y_true = test_df["anomaly_label"].values
        events = test_df.get("event_id", pd.Series([0] * len(test_df))).values
        timestamps = test_df.get("timestamp", pd.Series(np.arange(len(test_df)))).values

        metrics = {}
        models_to_eval = {
            "random_forest": ind_probs["rf"],
            "xgboost": ind_probs["xgboost"],
            "extra_trees": ind_probs["extra_trees"],
            "ensemble": ens_probs
        }

        # Calculate duration in hours for FAR/hr
        total_duration_hours = max((timestamps[-1] - timestamps[0]) / 3600.0, 0.25) if len(timestamps) > 1 else 1.0

        for name, probs in models_to_eval.items():
            preds = (probs >= 0.50).astype(int)
            roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
            pr = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
            f1 = f1_score(y_true, preds, zero_division=0)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            brier = brier_score_loss(y_true, probs)
            cm = confusion_matrix(y_true, preds).tolist()

            # Event-level detection latency calculation
            detection_latencies = []
            unique_events = [e for e in np.unique(events) if e > 0]
            for ev in unique_events:
                ev_mask = (events == ev)
                ev_indices = np.where(ev_mask)[0]
                if len(ev_indices) == 0:
                    continue
                start_idx = ev_indices[0]
                detected_indices = np.where(ev_mask & (preds == 1))[0]
                if len(detected_indices) > 0:
                    first_detect_idx = detected_indices[0]
                    t_detect = float(timestamps[first_detect_idx] - timestamps[start_idx])
                    detection_latencies.append(max(t_detect, 0.0))

            mean_ttd = float(np.mean(detection_latencies)) if detection_latencies else 0.0

            # False alarms count outside events
            false_positives = np.sum((y_true == 0) & (preds == 1))
            far_per_hour = float(false_positives / total_duration_hours)

            metrics[name] = {
                "roc_auc": float(roc),
                "pr_auc": float(pr),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "brier_score": float(brier),
                "confusion_matrix": cm,
                "mean_time_to_detect_sec": mean_ttd,
                "false_alarms_per_hour": far_per_hour
            }

        return metrics
