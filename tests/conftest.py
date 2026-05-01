import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
MAIN_MODEL_DIR = PROJECT_ROOT / "main_model"

for d in (METACONTROLLER_DIR, MAIN_MODEL_DIR, str(PROJECT_ROOT)):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from training_utils import load_training_config

try:
    from metacontroller import MetaMLP
    from main_model import create_encoder_weights
    _HAS_MODEL_IMPORTS = True
except ImportError:
    _HAS_MODEL_IMPORTS = False


@pytest.fixture
def mock_meta_mlp():
    """A small MetaMLP matching metacontroller's ARCH-01 architecture with test-size input."""
    if not _HAS_MODEL_IMPORTS:
        pytest.skip("MetaMLP not importable (missing reward_head dependency)")
    return MetaMLP(input_dim=10, output_dim=4)


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


@pytest.fixture
def mock_reward_mlp():
    """Reward head MLP matching reward_head.py lazy-init: Linear->ReLU->Linear(1)."""
    # Input: fused_dim*2 + RF_DIM*2 + 2 = 64*2 + 6*2 + 2 = 142
    input_dim = 142
    hidden_dim = 128
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


@pytest.fixture
def mock_rf_predictor():
    """RF predictor MLP matching reward_head.py lazy-init: Linear->ReLU->Linear(RF_DIM=6)."""
    # Input: fused_dim + fused_dim + RF_DIM = 64 + 64 + 6 = 134
    input_dim = 134
    hidden_dim = 64
    RF_DIM = 6
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, RF_DIM),
    )


@pytest.fixture
def mock_training_state(mock_meta_mlp, mock_reward_mlp, mock_rf_predictor):
    """TrainingState with mock modules, batch_size=4 for fast tests."""
    from trainer import TrainingState
    return TrainingState(
        meta_mlp=mock_meta_mlp,
        reward_mlp=mock_reward_mlp,
        rf_predictor=mock_rf_predictor,
        batch_size=4,
        buffer_capacity=100,
    )


def _make_trajectory_dict(input_dim=10, n_steps=3):
    """Helper to create a synthetic trajectory dict for buffer tests."""
    meta_trajectory = [
        {"decision": i % 4, "features": torch.randn(1, input_dim), "predicted_q": 0.1 * i}
        for i in range(n_steps)
    ]
    return {
        "meta_trajectory": meta_trajectory,
        "realized_return": 0.5,
        "is_fallback": False,
        "nodes_expanded": 3,
        "token_duration_frames": 5,
        "rollout": {
            "token_id": 42,
            "duration": 5,
            "states": [{"dummy": i} for i in range(6)],
            "rewards": [0.1, 0.2, 0.15, 0.1, 0.05],
            "components": [{"speed_reward": 0.1} for _ in range(5)],
            "state_before": {"dummy": 0},
            "state_after": {"dummy": 5},
        },
    }


# =========================================================================
# Phase 4: Module Training Pipelines fixtures
# =========================================================================

@pytest.fixture
def mock_encoder_weights():
    """Fresh encoder weights dict from create_encoder_weights()."""
    if not _HAS_MODEL_IMPORTS:
        pytest.skip("create_encoder_weights not importable (missing reward_head dependency)")
    return create_encoder_weights()


@pytest.fixture
def mock_state_pair():
    """
    Synthetic GTA state pair (state_t, state_t1) for encoder training tests.

    state_t1 has wp_dist=9.5 (closer to goal than state_t wp_dist=10.0),
    simulating forward progress.
    """
    base_state = {
        "near_entities": [],
        "near_vehs": [],
        "near_peds": [],
        "near_objects": [],
        "wp_dist": 10.0,
        "hp": 100.0,
        "v_engine_hp": 1000.0,
        "v_body_hp": 1000.0,
        "road_dist": 0.5,
        "dead": False,
    }
    state_t = dict(base_state)
    state_t1 = dict(base_state)
    state_t1["wp_dist"] = 9.5  # closer to goal, showing progress
    return (state_t, state_t1)


@pytest.fixture
def mock_training_config():
    """Training config dict loaded from training_config.json."""
    return load_training_config()


# =========================================================================
# Phase 5: Dashboard fixtures
# =========================================================================

@pytest_asyncio.fixture
async def test_db(tmp_path):
    """SQLite database with schema initialized in a temp directory."""
    from dashboard.database import init_database
    db = await init_database(db_path=tmp_path / "test.db")
    yield db
    await db.close()


@pytest.fixture
def test_client(tmp_path):
    """FastAPI TestClient with temp database and temp config."""
    import shutil
    from fastapi.testclient import TestClient
    from dashboard.server import app
    import dashboard.database as db_mod
    import dashboard.server as srv_mod
    # Copy training_config.json to temp dir so params tests work
    src_config = PROJECT_ROOT / "training_config.json"
    dst_config = tmp_path / "training_config.json"
    if src_config.exists():
        shutil.copy(src_config, dst_config)
    original_db = db_mod._DEFAULT_DB_PATH
    db_mod._DEFAULT_DB_PATH = tmp_path / "test.db"
    original_root = srv_mod._PROJECT_ROOT
    srv_mod._PROJECT_ROOT = tmp_path
    with TestClient(app) as client:
        yield client
    db_mod._DEFAULT_DB_PATH = original_db
    srv_mod._PROJECT_ROOT = original_root


@pytest.fixture
def test_client_with_auth(tmp_path):
    """FastAPI TestClient with DASHBOARD_PASSWORD set."""
    import shutil
    from fastapi.testclient import TestClient
    from dashboard.server import app
    import dashboard.database as db_mod
    import dashboard.server as srv_mod
    src_config = PROJECT_ROOT / "training_config.json"
    dst_config = tmp_path / "training_config.json"
    if src_config.exists():
        shutil.copy(src_config, dst_config)
    original_db = db_mod._DEFAULT_DB_PATH
    db_mod._DEFAULT_DB_PATH = tmp_path / "test.db"
    original_root = srv_mod._PROJECT_ROOT
    srv_mod._PROJECT_ROOT = tmp_path
    with patch.dict(os.environ, {"DASHBOARD_PASSWORD": "test_secret_123"}):
        with TestClient(app) as client:
            yield client
    db_mod._DEFAULT_DB_PATH = original_db
    srv_mod._PROJECT_ROOT = original_root
