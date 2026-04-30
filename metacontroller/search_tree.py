"""
Search tree — frame-local tree structure and step-by-step orchestrator.

Each token duration builds a fresh tree.  The action planner proposes root
candidates, and then one search step runs per frame (interleaved with the
executor playing the current token).  Each step expands one node via the
intuition head, scores it, and asks the metacontroller what to do.

The tree exposes an init + step interface so the frame loop can call one
search step per frame alongside one executor frame.
"""

import sys
from pathlib import Path

import torch

INTUITION_HEAD_DIR = Path(__file__).resolve().parent.parent / "intuition_head"
if str(INTUITION_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(INTUITION_HEAD_DIR))

from intuition_head import intuition_head
from metacontroller import KEEP, INTERRUPT, COMMIT_NEXT, ROLLBACK, metacontroller


# =========================================================================
# Tree node
# =========================================================================

class TreeNode:
    """Single node in the frame-local search tree."""

    def __init__(self, z, candidate_ids, candidate_priors, depth, parent=None):
        """
        z:                embedding at this node, shape [1, fused_dim]
        candidate_ids:    token ids to try from this node, shape [k] (long)
        candidate_priors: soft prior scores for each candidate, shape [k]
        depth:            depth in the tree (root = 0)
        parent:           parent TreeNode or None
        """
        self.z = z
        self.candidate_ids = candidate_ids
        self.candidate_priors = candidate_priors
        self.depth = depth
        self.parent = parent

        self.children = []
        self.next_unopened = 0

        self.token_id = None       # token that led TO this node (None for root)
        self.r_edge = 0.0          # reward on the edge from parent to here

        self.n = 0                 # visit count
        self.w = 0.0               # total accumulated value
        self.q = 0.0               # mean value  (w / n)

    @property
    def fully_expanded(self):
        return self.next_unopened >= len(self.candidate_ids)

    @property
    def unexplored_fraction(self):
        total = len(self.candidate_ids)
        if total == 0:
            return 0.0
        return (total - self.next_unopened) / total


# =========================================================================
# Search state (carries everything between steps)
# =========================================================================

class SearchState:
    """Mutable state for an in-progress search, passed between steps."""

    def __init__(self, root, z_running, running_token_id, vocab_size,
                 reward_fn, unmerge_fn, token_embed, intuition_mlp,
                 meta_mlp, embed_dim, hidden_dim):
        self.root = root
        self.current_node = root
        self.z_running = z_running
        self.running_token_id = running_token_id
        self.vocab_size = vocab_size
        self.reward_fn = reward_fn
        self.unmerge_fn = unmerge_fn
        self.token_embed = token_embed
        self.intuition_mlp = intuition_mlp
        self.meta_mlp = meta_mlp
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.nodes_expanded = 0
        self.meta_trajectory = []
        self.done = False
        self.final_decision = KEEP
        self.chosen_token_id = None


# =========================================================================
# Tree helpers
# =========================================================================

def create_root(z_t, candidate_ids, candidate_priors):
    """
    Build the root node from the action planner output.

    Input:
    - z_t: current fused embedding, shape [1, fused_dim]
    - candidate_ids: top-k token ids, shape [k] (long tensor)
    - candidate_priors: top-k probabilities, shape [k] (float tensor)

    Output:
    - root TreeNode
    """
    return TreeNode(
        z=z_t,
        candidate_ids=candidate_ids,
        candidate_priors=candidate_priors,
        depth=0,
        parent=None,
    )


def expand_next_child(
    node,
    vocab_size,
    reward_fn,
    unmerge_fn,
    token_embed=None,
    intuition_mlp=None,
    embed_dim=32,
    hidden_dim=128,
):
    """
    Open the next unopened candidate at `node`.

    Merged BPE tokens are unpacked into their base token sequence via
    unmerge_fn.  The intuition head runs once per base token in order,
    so the final z_child reflects the full action sequence.

    Input:
    - node: TreeNode to expand from
    - vocab_size: token vocabulary size
    - reward_fn: callable(z_parent, z_child) -> scalar reward
    - unmerge_fn: callable(token_id) -> list of base token ids
      e.g. merged token 500 -> [12, 7, 3]
    - token_embed, intuition_mlp: shared intuition head weights

    Output:
    - child: new TreeNode (already appended to node.children)
    - token_embed: updated embedding table
    - intuition_mlp: updated MLP
    """

    idx = node.next_unopened
    token_id = node.candidate_ids[idx]

    # -----------------------------------------------------------------
    # unpack merged token into base token sequence
    # -----------------------------------------------------------------

    base_tokens = unmerge_fn(int(token_id))

    # -----------------------------------------------------------------
    # roll the intuition head through each base token
    # -----------------------------------------------------------------

    z_current = node.z
    for base_id in base_tokens:
        base_id_tensor = torch.tensor(
            [base_id], dtype=torch.long, device=node.z.device
        )
        z_current, _, token_embed, intuition_mlp = intuition_head(
            z_current,
            base_id_tensor,
            vocab_size=vocab_size,
            token_embed=token_embed,
            intuition_mlp=intuition_mlp,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )

    z_child = z_current

    # -----------------------------------------------------------------
    # score the transition
    # -----------------------------------------------------------------

    r_edge = reward_fn(node.z, z_child)

    # -----------------------------------------------------------------
    # create child node (no candidates yet — leaf until planner expands)
    # -----------------------------------------------------------------

    child = TreeNode(
        z=z_child.detach(),
        candidate_ids=torch.tensor([], dtype=torch.long),
        candidate_priors=torch.tensor([]),
        depth=node.depth + 1,
        parent=node,
    )
    child.token_id = token_id
    child.r_edge = float(r_edge)

    node.children.append(child)
    node.next_unopened = idx + 1

    # -----------------------------------------------------------------
    # backprop value up the tree
    # -----------------------------------------------------------------

    _backprop(child, float(r_edge))

    return child, token_embed, intuition_mlp


def _backprop(node, value):
    """Propagate a value from `node` up to the root, updating N/W/Q."""
    current = node
    while current is not None:
        current.n += 1
        current.w += value
        current.q = current.w / current.n
        current = current.parent


def collect_root_q(root):
    """
    Gather Q-values and token ids for all expanded root children.

    Output:
    - candidate_q: shape [1, num_children]
    - candidate_ids: shape [1, num_children]
    """
    if not root.children:
        return (
            torch.tensor([[0.0]]),
            torch.tensor([[0]], dtype=torch.long),
        )

    q_vals = [c.q for c in root.children]
    t_ids = [c.token_id for c in root.children]

    candidate_q = torch.tensor([q_vals], dtype=torch.float32)
    candidate_ids = torch.tensor([t_ids], dtype=torch.long)
    return candidate_q, candidate_ids


def best_root_child(root):
    """Return the token_id of the root child with highest Q, or None."""
    if not root.children:
        return None
    best = max(root.children, key=lambda c: c.q)
    return best.token_id


# =========================================================================
# Init + Step interface
# =========================================================================

def search_init(
    z_t,
    planner_candidate_ids,
    planner_candidate_priors,
    running_token_id,
    z_running,
    vocab_size,
    reward_fn,
    unmerge_fn,
    token_embed=None,
    intuition_mlp=None,
    meta_mlp=None,
    embed_dim=32,
    hidden_dim=128,
):
    """
    Initialize a new search.  Call once at the start of a token's duration.

    Returns a SearchState that gets passed to search_step each frame.

    Input:
    - z_t: current real fused embedding, shape [1, fused_dim]
    - planner_candidate_ids: top-k token ids from action planner, shape [k]
    - planner_candidate_priors: top-k probs from action planner, shape [k]
    - running_token_id: token currently being executed, shape [1] (long)
    - z_running: state prediction when running token was committed, shape [1, fused_dim]
    - vocab_size, reward_fn, unmerge_fn: as in expand_next_child
    - token_embed, intuition_mlp, meta_mlp: shared weights
    - embed_dim, hidden_dim: network dimensions

    Output:
    - state: SearchState
    """
    root = create_root(z_t, planner_candidate_ids, planner_candidate_priors)

    return SearchState(
        root=root,
        z_running=z_running,
        running_token_id=running_token_id,
        vocab_size=vocab_size,
        reward_fn=reward_fn,
        unmerge_fn=unmerge_fn,
        token_embed=token_embed,
        intuition_mlp=intuition_mlp,
        meta_mlp=meta_mlp,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )


def search_step(state, z_t, time_ctx):
    """
    Run one search step.  Call once per frame while the current token plays.

    Expands one node, asks the metacontroller, applies the decision.
    Mutates `state` in place and returns the decision for this step.

    Input:
    - state: SearchState from search_init (or previous search_step)
    - z_t: latest real fused embedding (updated every frame), shape [1, fused_dim]
    - time_ctx: dict from time_context() with fresh elapsed_ratio, urgency, etc.

    Output dict:
    - decision: KEEP / INTERRUPT / COMMIT_NEXT / ROLLBACK for this step
    - chosen_token_id: int or None (only set on INTERRUPT / COMMIT_NEXT)
    - done: bool, True if search has concluded
    - nodes_expanded: int, total nodes expanded so far
    """

    if state.done:
        return {
            "decision": state.final_decision,
            "chosen_token_id": state.chosen_token_id,
            "done": True,
            "nodes_expanded": state.nodes_expanded,
        }

    # -----------------------------------------------------------------
    # check budget
    # -----------------------------------------------------------------

    if time_ctx["budget_remaining"] <= 0:
        state.done = True
        state.chosen_token_id = best_root_child(state.root)
        if state.chosen_token_id is not None:
            state.final_decision = COMMIT_NEXT
        return {
            "decision": state.final_decision,
            "chosen_token_id": state.chosen_token_id,
            "done": True,
            "nodes_expanded": state.nodes_expanded,
        }

    # -----------------------------------------------------------------
    # handle fully expanded node (auto-rollback)
    # -----------------------------------------------------------------

    if state.current_node.fully_expanded:
        if state.current_node.parent is not None:
            state.current_node = state.current_node.parent
        else:
            # root fully expanded, search is over
            state.done = True
            state.chosen_token_id = best_root_child(state.root)
            if state.chosen_token_id is not None:
                state.final_decision = COMMIT_NEXT
            return {
                "decision": state.final_decision,
                "chosen_token_id": state.chosen_token_id,
                "done": True,
                "nodes_expanded": state.nodes_expanded,
            }

    # -----------------------------------------------------------------
    # expand one node
    # -----------------------------------------------------------------

    _, state.token_embed, state.intuition_mlp = expand_next_child(
        state.current_node,
        vocab_size=state.vocab_size,
        reward_fn=state.reward_fn,
        unmerge_fn=state.unmerge_fn,
        token_embed=state.token_embed,
        intuition_mlp=state.intuition_mlp,
        embed_dim=state.embed_dim,
        hidden_dim=state.hidden_dim,
    )
    state.nodes_expanded += 1

    # -----------------------------------------------------------------
    # gather root-level Q values
    # -----------------------------------------------------------------

    candidate_q, candidate_ids = collect_root_q(state.root)

    # -----------------------------------------------------------------
    # parent_unexplored
    # -----------------------------------------------------------------

    parent_unexplored = torch.tensor(
        [[state.current_node.unexplored_fraction]], dtype=torch.float32
    )

    # -----------------------------------------------------------------
    # ask metacontroller
    # -----------------------------------------------------------------

    meta_out = metacontroller(
        z_t=z_t,
        z_running=state.z_running,
        running_token_id=state.running_token_id,
        elapsed_ratio=time_ctx["elapsed_ratio"],
        candidate_q=candidate_q,
        candidate_ids=candidate_ids,
        urgency=time_ctx["urgency"],
        parent_unexplored=parent_unexplored,
        meta_mlp=state.meta_mlp,
        hidden_dim=state.hidden_dim,
    )

    state.meta_mlp = meta_out["meta_mlp"]
    decision = meta_out["decision"].item()

    # -----------------------------------------------------------------
    # record metalevel step
    # -----------------------------------------------------------------

    best_q_val = candidate_q.max().item() if candidate_q.numel() > 0 else 0.0
    state.meta_trajectory.append({
        "decision": decision,
        "decision_logits": meta_out["decision_logits"],
        "predicted_q": best_q_val,
    })

    # -----------------------------------------------------------------
    # apply decision
    # -----------------------------------------------------------------

    chosen = None

    if decision == KEEP:
        pass

    elif decision == ROLLBACK:
        if state.current_node.parent is not None:
            state.current_node = state.current_node.parent

    elif decision in (INTERRUPT, COMMIT_NEXT):
        state.done = True
        state.final_decision = decision
        state.chosen_token_id = meta_out["selected_token_id"].item()
        chosen = state.chosen_token_id

    return {
        "decision": decision,
        "chosen_token_id": chosen,
        "done": state.done,
        "nodes_expanded": state.nodes_expanded,
    }
