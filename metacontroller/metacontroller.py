"""
Metacontroller — decides whether to EXPLORE, INTERRUPT, COMMIT_NEXT, or ROLLBACK
given the current real state, the running action token, and search tree results.

Inputs:
- z_t: latest real fused embedding [batch, fused_dim]
- z_running: predicted state when current token was committed [batch, fused_dim]
- running_token_id: token currently being executed [batch]
- elapsed_ratio: how much of the running token's duration has passed [batch, 1]
  (0.0 = just started, 1.0 = fully elapsed)
- candidate_q: root-level Q-values from the search tree [batch, k]
- candidate_ids: root-level token ids [batch, k]
- urgency: time-pressure scalar [batch, 1]  (0 = no rush, 1 = critical)
- parent_unexplored: fraction of unopened siblings at the parent node [batch, 1]
  (0.0 = all explored, 1.0 = none explored yet)
- current_path_value: cumulative r_edge sum from root to current node [batch, 1]
  (how good is the branch we're currently on?)
- best_path_value: best cumulative r_edge found so far in this search [batch, 1]
  (how good is the best plan we've found?)
- current_candidate_q: Q-values of the candidates at the current node [batch, top_k]
  (0.0 for candidates not yet expanded)
- current_candidate_durations: execution duration of each current candidate [batch, top_k]
  (frames — longer tokens carry more risk of the world changing mid-execution)
- current_candidate_emb: token embeddings of the current candidates, flattened
  [batch, top_k * token_embed_dim]
  (what kind of actions are available here?)

Output:
- decision: 0 = EXPLORE, 1 = INTERRUPT, 2 = COMMIT_NEXT, 3 = ROLLBACK  [batch]
- decision_logits: raw scores for the four choices [batch, 4]
- selected_token_id: which token to switch to (only meaningful when
  decision is INTERRUPT or COMMIT_NEXT) [batch]
- meta_mlp: the trainable MLP (pass back in for reuse)
"""

import torch
from torch import nn
from torch.distributions import Categorical

EXPLORE = 0       # expand next child and descend into it — go deeper on this branch
INTERRUPT = 1     # stop current token now, switch immediately
COMMIT_NEXT = 2   # current token finishes, then switch to best found
ROLLBACK = 3      # this branch is done, go back up to parent and try a sibling

# fused_dim(128) + scalars(10) + top_k_durations(3) + top_k_embeddings(96) = 237
META_INPUT_DIM = 237


class MetaMLP(nn.Module):
    """Metacontroller decision MLP with skip connection and LayerNorm.

    Architecture: input_dim -> 256 -> 256 -> 128 -> output_dim
    Skip connection: input projected to 256 and added at layer 2
    LayerNorm: applied at every hidden layer (before activation)
    """
    def __init__(self, input_dim=META_INPUT_DIM, output_dim=4):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)

        self.layer2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.skip_proj = nn.Linear(input_dim, 256)

        self.layer3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)

        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.ln1(self.layer1(x)))
        h2 = self.relu(self.ln2(self.layer2(h1) + self.skip_proj(x)))
        h3 = self.relu(self.ln3(self.layer3(h2)))
        return self.out(h3)


def metacontroller(
    z_t,
    z_running,
    running_token_id,
    elapsed_ratio,
    token_frames_left,
    candidate_q,
    candidate_ids,
    urgency,
    parent_unexplored,
    current_path_value,
    best_path_value,
    current_candidate_q,
    current_candidate_durations,
    current_candidate_emb,
    meta_mlp=None,
    training=False,
):
    """
    Decide whether to keep, interrupt, commit next, or rollback.

    Input:
    - z_t: current real fused embedding, shape [batch, fused_dim]
    - z_running: predicted state when current token was committed,
      shape [batch, fused_dim]
    - running_token_id: currently executing token, shape [batch] (long)
      (unused by MLP — kept for caller context / logging)
    - elapsed_ratio: fraction of token duration elapsed, shape [batch, 1]
    - token_frames_left: raw frames remaining before token ends, shape [batch, 1]
    - candidate_q: Q-values from search tree, shape [batch, k]
    - candidate_ids: token ids from search tree, shape [batch, k] (long)
    - urgency: time-pressure scalar, shape [batch, 1]
    - parent_unexplored: fraction of unopened siblings at parent node,
      shape [batch, 1]  (0.0 = all explored, 1.0 = none explored)
    - current_path_value: cumulative r_edge sum root→current_node, shape [batch, 1]
    - best_path_value: best cumulative r_edge found so far, shape [batch, 1]
    - current_candidate_q: Q-values of current node's candidates, shape [batch, top_k]
      (0.0 for candidates not yet expanded into children)
    - current_candidate_durations: execution duration of each candidate in frames,
      shape [batch, top_k]
    - current_candidate_emb: token embeddings of current candidates flattened,
      shape [batch, top_k * token_embed_dim]

    Output dict:
    - decision: KEEP(0) / INTERRUPT(1) / COMMIT_NEXT(2) / ROLLBACK(3), shape [batch]
    - decision_logits: raw scores, shape [batch, 4]
    - selected_token_id: best candidate token id, shape [batch]
    - meta_mlp: the trainable MLP (pass back in for reuse)
    """

    fused_dim = z_t.shape[-1]
    k = candidate_q.shape[-1]

    # -------------------------------------------------------------------------
    # Step 1: compute drift — how much real state diverged from prediction
    # -------------------------------------------------------------------------

    drift = z_t - z_running  # [batch, fused_dim]

    # -------------------------------------------------------------------------
    # Step 2: summarize candidate quality from the search tree
    # -------------------------------------------------------------------------

    # root-level summary (what can be committed to GTA)
    best_q, best_idx = candidate_q.max(dim=-1, keepdim=True)  # [batch, 1]
    mean_q = candidate_q.mean(dim=-1, keepdim=True)            # [batch, 1]

    # current-node summary (what is available at the node we're exploring)
    best_current_q = current_candidate_q.max(dim=-1, keepdim=True).values  # [batch, 1]
    mean_current_q = current_candidate_q.mean(dim=-1, keepdim=True)        # [batch, 1]

    # -------------------------------------------------------------------------
    # Step 3: assemble the feature vector for the decision MLP
    # -------------------------------------------------------------------------

    # drift:                       [batch, fused_dim]        how wrong the prediction was
    # elapsed_ratio:               [batch, 1]                how far into the current token
    # token_frames_left:           [batch, 1]                frames left before token ends
    # best_q / mean_q:             [batch, 1] each           root-level candidate quality
    # urgency:                     [batch, 1]                time pressure from deadline
    # parent_unexplored:           [batch, 1]                unexplored siblings at current node
    # current_path_value:          [batch, 1]                cumulative reward on current branch
    # best_path_value:             [batch, 1]                best cumulative reward found so far
    # best_current_q / mean_current_q: [batch, 1] each      current node candidate quality
    # current_candidate_durations: [batch, top_k]            execution cost of each candidate
    # current_candidate_emb:       [batch, top_k * embed_dim] what tokens are on the table

    features = torch.cat(
        [drift, elapsed_ratio, token_frames_left,
         best_q, mean_q,
         urgency, parent_unexplored,
         current_path_value, best_path_value,
         best_current_q, mean_current_q,
         current_candidate_durations,
         current_candidate_emb],
        dim=-1,
    )

    # -------------------------------------------------------------------------
    # Step 4: create the decision MLP if one was not passed in
    # -------------------------------------------------------------------------

    if meta_mlp is None:
        meta_mlp = MetaMLP(input_dim=features.shape[-1])
    else:
        # Validate dimension consistency on subsequent calls
        expected_dim = meta_mlp.layer1.in_features
        assert features.shape[-1] == expected_dim, (
            f"Feature dim {features.shape[-1]} != MLP input_dim {expected_dim}"
        )

    # -------------------------------------------------------------------------
    # Step 5: produce logits over the three decisions
    # -------------------------------------------------------------------------

    decision_logits = meta_mlp(features)  # [batch, 4]

    # -------------------------------------------------------------------------
    # Step 6: pick the decision
    # -------------------------------------------------------------------------

    if training:
        dist = Categorical(logits=decision_logits)
        decision = dist.sample()  # [batch] -- stochastic exploration
    else:
        decision = decision_logits.argmax(dim=-1)  # [batch] -- deterministic

    # -------------------------------------------------------------------------
    # Step 7: select the best candidate token from the search tree
    # -------------------------------------------------------------------------

    best_idx_flat = best_idx.squeeze(-1)  # [batch]
    selected_token_id = candidate_ids.gather(
        1, best_idx_flat.unsqueeze(-1)
    ).squeeze(-1)  # [batch]

    return {
        "decision":          decision,
        "decision_logits":   decision_logits,
        "features":          features,          # stored in trajectory for training
        "selected_token_id": selected_token_id,
        "meta_mlp":          meta_mlp,
    }
