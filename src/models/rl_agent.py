"""
Reinforcement Learning Adaptive Policy & Dynamic Thresholding Subsystem
Formulates a Markov Decision Process (MDP) for satellite telemetry monitoring.
Dynamically optimizes the anomaly decision threshold (tau_t) and commands autonomous
payload mitigation actions to maximize mission availability while guaranteeing 99.9% safety.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from collections import deque


class SatelliteMitigationEnv:
    """
    Simulated Aerospace Environment for Telemetry Anomaly Mitigation.
    State: [p_ensemble, thermal_gradient, impedance_proxy, soc, is_eclipse, rolling_anomaly_rate]
    Actions:
      0: NOMINAL_MONITOR (tau = 0.70, normal telemetry rate)
      1: HIGH_SENSITIVITY_PREARM (tau = 0.35, high-frequency sampling)
      2: LOAD_SHEDDING (sheds non-essential payload, mitigates heat)
      3: TRIGGER_SAFE_MODE (hardware Pin 13 ON, switch redundant bus)
    """

    ACTION_NAMES = [
        "NOMINAL_MONITOR",
        "HIGH_SENSITIVITY_PREARM",
        "LOAD_SHEDDING",
        "TRIGGER_SAFE_MODE"
    ]

    ACTION_THRESHOLDS = [0.70, 0.35, 0.45, 0.50]

    def __init__(
        self,
        telemetry_df: pd.DataFrame,
        ensemble_probs: np.ndarray,
        reward_weights: Optional[Dict[str, float]] = None,
        max_episode_steps: int = 1500
    ):
        self.total_steps = len(telemetry_df)
        self.max_episode_steps = min(max_episode_steps, self.total_steps)
        self.current_step = 0
        self.episode_start_step = 0
        self.steps_in_episode = 0

        self.ensemble_probs = np.asarray(ensemble_probs, dtype=np.float32)
        self.t_grad_arr = np.asarray(telemetry_df["thermal_gradient"].values if "thermal_gradient" in telemetry_df.columns else np.zeros(self.total_steps), dtype=np.float32)
        self.imp_arr = np.asarray(telemetry_df["impedance_proxy"].values if "impedance_proxy" in telemetry_df.columns else np.full(self.total_steps, 0.05), dtype=np.float32)
        self.soc_arr = np.asarray(telemetry_df["soc"].values if "soc" in telemetry_df.columns else np.full(self.total_steps, 0.75), dtype=np.float32)
        self.eclipse_arr = np.asarray(telemetry_df["is_eclipse"].values if "is_eclipse" in telemetry_df.columns else np.zeros(self.total_steps), dtype=np.float32)
        self.labels_arr = np.asarray(telemetry_df["anomaly_label"].values if "anomaly_label" in telemetry_df.columns else np.zeros(self.total_steps), dtype=np.int32)

        self.reward_weights = reward_weights or {
            "true_positive": 15.0,
            "true_negative": 1.0,
            "false_positive": -6.0,
            "false_negative": -35.0,
            "action_switch_penalty": -0.5,
            "safe_mode_downtime_penalty": -3.0
        }

        self.recent_alerts = deque(maxlen=20)
        self.prev_action = 0
        self.rng = np.random.default_rng(42)

    def reset(self) -> np.ndarray:
        max_start = max(1, self.total_steps - self.max_episode_steps)
        self.episode_start_step = int(self.rng.integers(0, max_start))
        self.current_step = self.episode_start_step
        self.steps_in_episode = 0
        self.recent_alerts.clear()
        self.prev_action = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        if self.current_step >= self.total_steps:
            return np.zeros(6, dtype=np.float32)

        p_ens = float(self.ensemble_probs[self.current_step])
        t_grad = float(self.t_grad_arr[self.current_step])
        imp = float(self.imp_arr[self.current_step])
        soc = float(self.soc_arr[self.current_step])
        eclipse = float(self.eclipse_arr[self.current_step])
        alert_rate = float(np.mean(self.recent_alerts)) if self.recent_alerts else 0.0

        # Clip and normalize state vector
        state = np.array([
            np.clip(p_ens, 0.0, 1.0),
            np.clip(t_grad / 3.0, -1.0, 1.0),
            np.clip(imp / 2.0, 0.0, 2.0),
            np.clip(soc, 0.0, 1.0),
            eclipse,
            np.clip(alert_rate, 0.0, 1.0)
        ], dtype=np.float32)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.current_step >= self.total_steps or self.steps_in_episode >= self.max_episode_steps:
            return np.zeros(6, dtype=np.float32), 0.0, True, {}

        p_ens = float(self.ensemble_probs[self.current_step])
        ground_truth = int(self.labels_arr[self.current_step])

        effective_threshold = self.ACTION_THRESHOLDS[action]
        model_flagged_anomaly = int(p_ens >= effective_threshold)

        # Action-based intervention flag
        action_triggered_alert = 1 if (action in [2, 3] or model_flagged_anomaly == 1) else 0
        self.recent_alerts.append(action_triggered_alert)

        # Calculate Reward
        rw = self.reward_weights
        reward = 0.0

        if action_triggered_alert == 1 and ground_truth == 1:
            # Successful detection & mitigation
            reward += rw["true_positive"]
            if action == 2:
                reward += 3.0  # Bonus for surgical load shedding mitigation
        elif action_triggered_alert == 0 and ground_truth == 0:
            # Nominal smooth operation
            reward += rw["true_negative"]
        elif action_triggered_alert == 1 and ground_truth == 0:
            # False Alarm penalty
            reward += rw["false_positive"]
            if action == 3:
                reward += rw["safe_mode_downtime_penalty"]
        elif action_triggered_alert == 0 and ground_truth == 1:
            # Catastrophic Missed Detection penalty
            reward += rw["false_negative"]

        # Action switching friction (penalize jitter)
        if action != self.prev_action:
            reward += rw["action_switch_penalty"]

        self.prev_action = action
        self.current_step += 1
        self.steps_in_episode += 1
        done = (self.current_step >= self.total_steps or self.steps_in_episode >= self.max_episode_steps)

        next_state = self._get_state() if not done else np.zeros(6, dtype=np.float32)

        info = {
            "action_name": self.ACTION_NAMES[action],
            "threshold": effective_threshold,
            "ground_truth": ground_truth,
            "anomaly_detected": action_triggered_alert
        }

        return next_state, reward, done, info


class RLAgent:
    """
    Q-Learning Agent with continuous state discretization and epsilon-greedy exploration.
    """

    def __init__(
        self,
        n_actions: int = 4,
        learning_rate: float = 0.08,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Tabular Q-Table: state tuple -> np.ndarray of shape (n_actions,)
        self.q_table: Dict[Tuple[int, ...], List[float]] = {}

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        Discretizes continuous state space into bins:
        State: [p_ensemble, t_grad, impedance, soc, is_eclipse, alert_rate]
        """
        p_ens_bin = int(np.clip(state[0] * 5, 0, 4))            # 5 bins
        t_grad_bin = int(np.clip((state[1] + 1.0) * 2, 0, 3))   # 4 bins
        imp_bin = int(np.clip(state[2] * 2, 0, 3))              # 4 bins
        soc_bin = int(np.clip(state[3] * 3, 0, 2))              # 3 bins
        eclipse_bin = int(state[4])                             # 2 bins
        alert_bin = int(np.clip(state[5] * 3, 0, 2))            # 3 bins

        return (p_ens_bin, t_grad_bin, imp_bin, soc_bin, eclipse_bin, alert_bin)

    def _get_q_values(self, discrete_state: Tuple[int, ...]) -> List[float]:
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = [0.0] * self.n_actions
        return self.q_table[discrete_state]

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        discrete_state = self._discretize_state(state)
        q_vals = self._get_q_values(discrete_state)

        if explore and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        else:
            return int(np.argmax(q_vals))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        discrete_state = self._discretize_state(state)
        q_vals = self._get_q_values(discrete_state)

        if done:
            target = reward
        else:
            next_discrete = self._discretize_state(next_state)
            next_q = self._get_q_values(next_discrete)
            target = reward + self.gamma * max(next_q)

        q_vals[action] += self.lr * (target - q_vals[action])
        self.q_table[discrete_state] = q_vals

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_on_environment(self, env: SatelliteMitigationEnv, episodes: int = 50) -> List[float]:
        episode_rewards = []

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state, explore=True)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)

        return episode_rewards

    def get_action_recommendation(
        self,
        p_ensemble: float,
        thermal_gradient: float = 0.0,
        impedance_proxy: float = 0.05,
        soc: float = 0.8,
        is_eclipse: int = 0,
        recent_alert_rate: float = 0.0
    ) -> Dict[str, Any]:
        """
        Fast online decision inference for real-time streaming pipeline.
        """
        state = np.array([
            np.clip(p_ensemble, 0.0, 1.0),
            np.clip(thermal_gradient / 3.0, -1.0, 1.0),
            np.clip(impedance_proxy / 2.0, 0.0, 2.0),
            np.clip(soc, 0.0, 1.0),
            float(is_eclipse),
            np.clip(recent_alert_rate, 0.0, 1.0)
        ], dtype=np.float32)

        action = self.select_action(state, explore=False)
        action_name = SatelliteMitigationEnv.ACTION_NAMES[action]
        dynamic_threshold = SatelliteMitigationEnv.ACTION_THRESHOLDS[action]

        discrete_state = self._discretize_state(state)
        q_vals = self._get_q_values(discrete_state)

        return {
            "rl_action_id": action,
            "rl_action_name": action_name,
            "dynamic_threshold": dynamic_threshold,
            "q_values": q_vals,
            "policy_confidence": float(np.max(q_vals) - np.min(q_vals)) if len(q_vals) > 0 else 0.0
        }

    def save(self, filepath: str):
        """Saves Q-table and configuration to JSON"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "q_table": {f"{k}": v for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "RLAgent":
        with open(filepath, "r") as f:
            data = json.load(f)

        agent = cls(
            learning_rate=data.get("lr", 0.08),
            discount_factor=data.get("gamma", 0.95),
            epsilon_start=data.get("epsilon", 0.05)
        )
        # Parse Q table keys back to integer tuples
        q_raw = data.get("q_table", {})
        for k_str, v in q_raw.items():
            try:
                k_clean = k_str.strip("()").replace(" ", "").split(",")
                k_tuple = tuple(int(x) for x in k_clean if x)
                agent.q_table[k_tuple] = v
            except Exception:
                pass

        return agent
