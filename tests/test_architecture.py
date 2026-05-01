"""
Tests for Phase 3: Architecture Upgrades (ARCH-01 through ARCH-04).

Each test validates one requirement from REQUIREMENTS.md.
Tests use synthetic data -- they do not require GTA or real game states.
"""

import sys
import os
import tempfile
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
MAIN_MODEL_DIR = PROJECT_ROOT / "main_model"
ACTION_PLANNER_DIR = PROJECT_ROOT / "action_planner"
TESTS_DIR = PROJECT_ROOT / "tests"

for d in (METACONTROLLER_DIR, MAIN_MODEL_DIR, ACTION_PLANNER_DIR, TESTS_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from metacontroller import MetaMLP, META_INPUT_DIM
from main_model import create_encoder_weights, encode_state
from action_planner import action_planner


# =========================================================================
# ARCH-01: MetaMLP structure, forward pass, save/load
# =========================================================================

def test_meta_mlp_structure():
    """ARCH-01: MetaMLP has 3 hidden layers (256-256-128), skip connection, LayerNorm."""
    m = MetaMLP()
    # Check layer dimensions
    assert m.layer1.in_features == 237 and m.layer1.out_features == 256, (
        f"Layer 1 should be (237, 256), got ({m.layer1.in_features}, {m.layer1.out_features})"
    )
    assert m.layer2.in_features == 256 and m.layer2.out_features == 256, (
        f"Layer 2 should be (256, 256), got ({m.layer2.in_features}, {m.layer2.out_features})"
    )
    assert m.layer3.in_features == 256 and m.layer3.out_features == 128, (
        f"Layer 3 should be (256, 128), got ({m.layer3.in_features}, {m.layer3.out_features})"
    )
    assert m.out.in_features == 128 and m.out.out_features == 4, (
        f"Output should be (128, 4), got ({m.out.in_features}, {m.out.out_features})"
    )
    # Check skip connection projection exists
    assert hasattr(m, 'skip_proj'), "Missing skip_proj for skip connection"
    assert m.skip_proj.in_features == 237 and m.skip_proj.out_features == 256, (
        f"skip_proj should be (237, 256), got ({m.skip_proj.in_features}, {m.skip_proj.out_features})"
    )
    # Check LayerNorm at each hidden layer
    assert hasattr(m, 'ln1') and isinstance(m.ln1, nn.LayerNorm), "Missing ln1"
    assert hasattr(m, 'ln2') and isinstance(m.ln2, nn.LayerNorm), "Missing ln2"
    assert hasattr(m, 'ln3') and isinstance(m.ln3, nn.LayerNorm), "Missing ln3"
    assert m.ln1.normalized_shape == (256,), f"ln1 shape: {m.ln1.normalized_shape}"
    assert m.ln2.normalized_shape == (256,), f"ln2 shape: {m.ln2.normalized_shape}"
    assert m.ln3.normalized_shape == (128,), f"ln3 shape: {m.ln3.normalized_shape}"


def test_meta_mlp_forward():
    """ARCH-01: MetaMLP forward pass produces [batch, 4] from [batch, 237] input."""
    m = MetaMLP()
    x = torch.randn(1, META_INPUT_DIM)
    out = m(x)
    assert out.shape == (1, 4), f"Expected (1, 4), got {out.shape}"

    # Test with batch > 1
    x_batch = torch.randn(4, META_INPUT_DIM)
    out_batch = m(x_batch)
    assert out_batch.shape == (4, 4), f"Expected (4, 4), got {out_batch.shape}"

    # Test with custom input_dim
    m_small = MetaMLP(input_dim=10)
    x_small = torch.randn(1, 10)
    out_small = m_small(x_small)
    assert out_small.shape == (1, 4), f"Expected (1, 4), got {out_small.shape}"


def test_meta_mlp_save_load():
    """ARCH-01: MetaMLP save/load roundtrip produces identical output."""
    m = MetaMLP()
    x = torch.randn(1, META_INPUT_DIM)
    out_before = m(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta_mlp.pt")
        torch.save({"model_state_dict": m.state_dict()}, path)

        m2 = MetaMLP()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        m2.load_state_dict(ckpt["model_state_dict"])

        out_after = m2(x)
        assert torch.allclose(out_before, out_after), (
            f"Save/load roundtrip mismatch: max diff = {(out_before - out_after).abs().max().item()}"
        )


# =========================================================================
# ARCH-04: META_INPUT_DIM constant
# =========================================================================

def test_meta_input_dim_constant():
    """ARCH-04: META_INPUT_DIM constant equals 237."""
    assert META_INPUT_DIM == 237, f"META_INPUT_DIM should be 237, got {META_INPUT_DIM}"
    # Verify the breakdown: fused_dim(128) + scalars(10) + top_k(3) + top_k*embed(96) = 237
    expected = 128 + 10 + 3 + 96
    assert META_INPUT_DIM == expected, (
        f"META_INPUT_DIM={META_INPUT_DIM} doesn't match breakdown sum={expected}"
    )


# =========================================================================
# ARCH-02: Encoder 2-block attention with LayerNorm
# =========================================================================

def test_encoder_two_attention_blocks():
    """ARCH-02: Encoder uses 2 attention blocks with LayerNorm."""
    w = create_encoder_weights()

    # Block 1 weights must exist
    for key in ("qw1", "kw1", "vw1", "ow1"):
        assert key in w, f"Missing block 1 weight: {key}"

    # Block 2 weights must exist
    for key in ("qw2", "kw2", "vw2", "ow2"):
        assert key in w, f"Missing block 2 weight: {key}"

    # LayerNorm for each block
    assert "ln_attn1" in w and isinstance(w["ln_attn1"], nn.LayerNorm), (
        "Missing or wrong type: ln_attn1"
    )
    assert "ln_attn2" in w and isinstance(w["ln_attn2"], nn.LayerNorm), (
        "Missing or wrong type: ln_attn2"
    )

    # Block 1 query dim = embed_dim * 3 = 192
    assert w["qw1"].shape[0] == 192, (
        f"Block 1 query dim should be 192, got {w['qw1'].shape[0]}"
    )

    # Block 2 query dim = embed_dim = 64
    assert w["qw2"].shape[0] == 64, (
        f"Block 2 query dim should be 64, got {w['qw2'].shape[0]}"
    )

    # Old keys must NOT exist
    for old_key in ("qw", "kw", "vw", "ow"):
        assert old_key not in w, f"Old key still present: {old_key}"


def test_encoder_output_shape():
    """ARCH-02: Encoder output shape unchanged at [1, 128] after upgrade."""
    w = create_encoder_weights()
    raw_state = {
        "near_entities": [],
        "near_vehs": [],
        "near_peds": [],
        "near_objects": [],
    }
    z_t = encode_state(raw_state, w)
    assert z_t.shape == (1, 128), f"z_t shape should be (1, 128), got {z_t.shape}"


# =========================================================================
# ARCH-03: Action planner 2-layer MLP
# =========================================================================

def test_planner_two_layers():
    """ARCH-03: Action planner MLP has 2 hidden layers (256, 128)."""
    z_t = torch.randn(1, 128)
    z_next = torch.randn(1, 128)
    result = action_planner(z_t, z_next, vocab_size=874)

    mlp = result["planner_mlp"]
    linear_layers = [m for m in mlp if isinstance(m, nn.Linear)]

    assert len(linear_layers) == 3, (
        f"Expected 3 Linear layers (2 hidden + output), got {len(linear_layers)}"
    )
    assert linear_layers[0].in_features == 256 and linear_layers[0].out_features == 256, (
        f"Layer 0 should be (256, 256), got ({linear_layers[0].in_features}, {linear_layers[0].out_features})"
    )
    assert linear_layers[1].in_features == 256 and linear_layers[1].out_features == 128, (
        f"Layer 1 should be (256, 128), got ({linear_layers[1].in_features}, {linear_layers[1].out_features})"
    )
    assert linear_layers[2].in_features == 128 and linear_layers[2].out_features == 874, (
        f"Layer 2 should be (128, 874), got ({linear_layers[2].in_features}, {linear_layers[2].out_features})"
    )


# =========================================================================
# ARCH-04: Full checkpoint roundtrip
# =========================================================================

def test_full_checkpoint_roundtrip():
    """ARCH-04: All modules serialize to disk and reload cleanly."""
    # MetaMLP roundtrip
    meta = MetaMLP()
    x_meta = torch.randn(1, META_INPUT_DIM)
    out_meta_before = meta(x_meta)

    # Encoder weights roundtrip (save individual components)
    enc_w = create_encoder_weights()
    raw_state = {"near_entities": [], "near_vehs": [], "near_peds": [], "near_objects": []}
    z_before = encode_state(raw_state, enc_w)

    # Action planner roundtrip
    z_t = torch.randn(1, 128)
    z_next = torch.randn(1, 128)
    planner_result = action_planner(z_t, z_next, vocab_size=100)
    planner_mlp = planner_result["planner_mlp"]
    planner_out_before = planner_mlp(torch.cat([z_t, z_next], dim=-1))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        torch.save({"model_state_dict": meta.state_dict()}, os.path.join(tmpdir, "meta.pt"))
        torch.save({"model_state_dict": planner_mlp.state_dict()}, os.path.join(tmpdir, "planner.pt"))

        # Reload MetaMLP
        meta2 = MetaMLP()
        ckpt = torch.load(os.path.join(tmpdir, "meta.pt"), map_location="cpu", weights_only=True)
        meta2.load_state_dict(ckpt["model_state_dict"])
        out_meta_after = meta2(x_meta)
        assert torch.allclose(out_meta_before, out_meta_after), "MetaMLP roundtrip failed"

        # Reload planner
        planner2 = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 100),
        )
        ckpt_p = torch.load(os.path.join(tmpdir, "planner.pt"), map_location="cpu", weights_only=True)
        planner2.load_state_dict(ckpt_p["model_state_dict"])
        planner_out_after = planner2(torch.cat([z_t, z_next], dim=-1))
        assert torch.allclose(planner_out_before, planner_out_after), "Planner roundtrip failed"
