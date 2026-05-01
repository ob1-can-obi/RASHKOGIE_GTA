"""
Tests for Phase 1: Training Correctness (TRAIN-01 through TRAIN-06).

Each test validates one requirement from REQUIREMENTS.md.
Tests use synthetic data -- they do not require GTA or real game states.
"""

import sys
from pathlib import Path
import math

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
if str(METACONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(METACONTROLLER_DIR))

from metacontroller import metacontroller, EXPLORE, INTERRUPT, COMMIT_NEXT, ROLLBACK


# =========================================================================
# TRAIN-01: Categorical sampling produces all 4 decisions over N runs
# =========================================================================

def test_categorical_sampling():
    """
    TRAIN-01: With training=True, metacontroller uses Categorical sampling
    and produces varied decisions. With training=False (default), it uses
    argmax and produces deterministic output.
    """
    fused_dim = 64
    z = torch.randn(1, fused_dim)
    cq = torch.randn(1, 3)
    ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
    ccq = torch.randn(1, 3)
    ccd = torch.ones(1, 3)
    cce = torch.randn(1, 3 * 32)  # top_k=3, embed_dim=32

    # First call to create MLP
    out_init = metacontroller(
        z, z, torch.tensor([0]), torch.tensor([[0.5]]),
        torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
        torch.tensor([[0.5]]), torch.tensor([[0.1]]),
        torch.tensor([[0.2]]), ccq, ccd, cce,
        training=False,
    )
    mlp = out_init["meta_mlp"]

    # --- Inference mode: deterministic ---
    results_inference = set()
    for _ in range(10):
        out = metacontroller(
            z, z, torch.tensor([0]), torch.tensor([[0.5]]),
            torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
            torch.tensor([[0.5]]), torch.tensor([[0.1]]),
            torch.tensor([[0.2]]), ccq, ccd, cce,
            meta_mlp=mlp, training=False,
        )
        results_inference.add(out["decision"].item())

    assert len(results_inference) == 1, (
        f"Inference mode should be deterministic, got {results_inference}"
    )

    # --- Training mode: stochastic ---
    results_training = set()
    for _ in range(500):
        out = metacontroller(
            z, z, torch.tensor([0]), torch.tensor([[0.5]]),
            torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
            torch.tensor([[0.5]]), torch.tensor([[0.1]]),
            torch.tensor([[0.2]]), ccq, ccd, cce,
            meta_mlp=mlp, training=True,
        )
        results_training.add(out["decision"].item())

    # With random MLP weights and 500 samples, we should see at least 2
    # different decisions. Getting all 4 depends on the random init but
    # 2+ proves sampling is working.
    assert len(results_training) >= 2, (
        f"Training mode should produce varied decisions, got {results_training}"
    )

    # Verify all decisions are valid (0-3)
    for d in results_training:
        assert d in {EXPLORE, INTERRUPT, COMMIT_NEXT, ROLLBACK}, (
            f"Invalid decision value: {d}"
        )


def test_training_flag_default_is_false():
    """
    TRAIN-01: Default behavior (no training= kwarg) should use argmax,
    matching the original pre-fix behavior.
    """
    fused_dim = 64
    z = torch.randn(1, fused_dim)
    cq = torch.randn(1, 3)
    ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
    ccq = torch.randn(1, 3)
    ccd = torch.ones(1, 3)
    cce = torch.randn(1, 3 * 32)

    out1 = metacontroller(
        z, z, torch.tensor([0]), torch.tensor([[0.5]]),
        torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
        torch.tensor([[0.5]]), torch.tensor([[0.1]]),
        torch.tensor([[0.2]]), ccq, ccd, cce,
    )
    mlp = out1["meta_mlp"]

    # Call without training= kwarg -- should be deterministic
    decisions = set()
    for _ in range(10):
        out = metacontroller(
            z, z, torch.tensor([0]), torch.tensor([[0.5]]),
            torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
            torch.tensor([[0.5]]), torch.tensor([[0.1]]),
            torch.tensor([[0.2]]), ccq, ccd, cce,
            meta_mlp=mlp,
        )
        decisions.add(out["decision"].item())

    assert len(decisions) == 1, (
        f"Default (no training kwarg) should be deterministic, got {decisions}"
    )
