"""
Unit Tests for Reinforcement Learning Adaptive Policy & Dynamic Thresholding Agent
"""

import pytest
import numpy as np
import pandas as pd
import os
from src.models.rl_agent import SatelliteMitigationEnv, RLAgent


def test_rl_environment_and_agent_step():
    """Verify MDP environment transitions, rewards, and Q-learning updates"""
    n_steps = 50
    df = pd.DataFrame({
        "timestamp": np.arange(n_steps) * 0.5,
        "thermal_gradient": np.zeros(n_steps),
        "impedance_proxy": np.full(n_steps, 0.05),
        "soc": np.full(n_steps, 0.8),
        "is_eclipse": np.zeros(n_steps, dtype=int),
        "anomaly_label": np.zeros(n_steps, dtype=int)
    })
    # Step 20 is an anomaly
    df.loc[20:25, "anomaly_label"] = 1
    ensemble_probs = np.full(n_steps, 0.05)
    ensemble_probs[20:25] = 0.85

    env = SatelliteMitigationEnv(df, ensemble_probs)
    agent = RLAgent(n_actions=4, learning_rate=0.1)

    state = env.reset()
    assert state.shape == (6,)

    action = agent.select_action(state, explore=False)
    assert action in [0, 1, 2, 3]

    next_state, reward, done, info = env.step(action)
    assert not done
    assert "action_name" in info
    assert "threshold" in info

    agent.update(state, action, reward, next_state, done)


def test_rl_agent_recommendation_and_persistence(tmp_path):
    """Verify online action recommendation and JSON checkpointing"""
    agent = RLAgent(n_actions=4)

    rec = agent.get_action_recommendation(
        p_ensemble=0.88,
        thermal_gradient=1.5,
        impedance_proxy=0.2,
        soc=0.6,
        is_eclipse=1
    )

    assert "rl_action_id" in rec
    assert "rl_action_name" in rec
    assert "dynamic_threshold" in rec
    assert 0.30 <= rec["dynamic_threshold"] <= 0.80

    # Test Save & Load
    save_file = os.path.join(tmp_path, "test_rl.json")
    agent.save(save_file)
    loaded_agent = RLAgent.load(save_file)

    loaded_rec = loaded_agent.get_action_recommendation(p_ensemble=0.88)
    assert loaded_rec["rl_action_id"] == rec["rl_action_id"]
