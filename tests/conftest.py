import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
if str(METACONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(METACONTROLLER_DIR))


@pytest.fixture
def mock_meta_mlp():
    """A small MLP matching metacontroller's lazy init pattern: Linear->ReLU->Linear(4)."""
    input_dim = 10
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    )


@pytest.fixture
def mock_meta_trajectory():
    """
    Synthetic metalevel trajectory with 3 steps.
    Format matches search_tree.py lines 698-702.
    Feature dim = 10 to match mock_meta_mlp input.
    """
    return [
        {"decision": 0, "features": torch.randn(1, 10), "predicted_q": 0.1},
        {"decision": 0, "features": torch.randn(1, 10), "predicted_q": 0.15},
        {"decision": 2, "features": torch.randn(1, 10), "predicted_q": 0.2},
    ]


@pytest.fixture
def mock_rollout():
    """
    Synthetic rollout matching executor.py lines 164-172.
    5 frames with small positive rewards.
    """
    return {
        "token_id": 42,
        "duration": 5,
        "states": [{"dummy": i} for i in range(6)],
        "rewards": [0.1, 0.2, 0.15, 0.1, 0.05],
        "components": [{"speed_reward": 0.1} for _ in range(5)],
        "state_before": {"dummy": 0},
        "state_after": {"dummy": 5},
    }


@pytest.fixture
def mock_rollout_long():
    """A 20-frame rollout for testing duration normalization effects on longer tokens."""
    return {
        "token_id": 99,
        "duration": 20,
        "states": [{"dummy": i} for i in range(21)],
        "rewards": [0.05] * 20,
        "components": [{"speed_reward": 0.05} for _ in range(20)],
        "state_before": {"dummy": 0},
        "state_after": {"dummy": 20},
    }
