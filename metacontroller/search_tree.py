"""
Search tree — frame-local MCTS tree and step-by-step orchestrator.

Each token duration builds a fresh tree.  The action planner proposes root
candidates, and then one search step runs per frame (interleaved with the
executor playing the current token).

Each search step:
1. Expands one node: intuition head predicts z_child, action planner gives
   the child's candidate set, reward head scores the edge.
2. Asks the metacontroller what to do.
3. Applies the decision:
   - EXPLORE:      descend into the just-expanded child
   - ROLLBACK:     go up to parent (try a sibling there next step)
   - INTERRUPT:    stop current GTA token, switch to best path immediately
   - COMMIT_NEXT:  finish current token, then switch to best path

Tree grows by EXPLORE (going deeper) and shrinks by ROLLBACK (going up).
The committed token is always the best root child.
"""

import sys
from pathlib import Path

import torch

INTUITION_HEAD_DIR = Path(__file__).resolve().parent.parent / "intuition_head"
if str(INTUITION_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(INTUITION_HEAD_DIR))

REWARD_HEAD_DIR = Path(__file__).resolve().parent.parent / "reward_head"
if str(REWARD_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_HEAD_DIR))

ACTION_PLANNER_DIR = Path(__file__).resolve().parent.parent / "action_planner"
if str(ACTION_PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(ACTION_PLANNER_DIR))

from intuition_head import intuition_head
from reward_head import reward_head, predict_reward_features
from action_planner import action_planner
from metacontroller import EXPLORE, INTERRUPT, COMMIT_NEXT, ROLLBACK, metacontroller


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
        self.r_edge   = 0.0        # reward predicted by the reward head NN for this edge
        self.duration = 0          # duration of the token that led to this node (frames)
        self.rf       = None       # reward features at this node [1, RF_DIM]
                                   # real for root, predicted for all children

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

    def __init__(
        self,
        root,
        z_running,
        running_token_id,
        vocab_size,
        unmerge_fn,
        duration_fn,
        token_embed,
        intuition_mlp,
        reward_mlp,
        rf_predictor,
        planner_mlp,
        meta_mlp,
        embed_dim,
        hidden_dim,
        top_k=3,
        training=False,
    ):
        self.root             = root
        self.current_node     = root
        self.z_running        = z_running
        self.running_token_id = running_token_id
        self.vocab_size       = vocab_size
        self.unmerge_fn       = unmerge_fn   # int -> list[int]  (unmerge BPE token)
        self.duration_fn      = duration_fn  # int -> int        (token_id -> frames)
        self.token_embed      = token_embed
        self.intuition_mlp    = intuition_mlp
        self.reward_mlp       = reward_mlp       # reward head NN weights
        self.rf_predictor     = rf_predictor     # predicts child reward features
        self.planner_mlp      = planner_mlp      # action planner MLP
        self.meta_mlp         = meta_mlp
        self.embed_dim        = embed_dim
        self.hidden_dim       = hidden_dim
        self.top_k            = top_k            # candidates per node
        self.training         = training

        self.nodes_expanded   = 0
        self.meta_trajectory  = []
        self.done             = False
        self.final_decision   = EXPLORE
        self.chosen_token_id  = None

        # best path found so far: full sequence from root to deepest best leaf
        self.best_path        = []          # list of token_ids e.g. [A, A3, A32]
        self.best_path_value  = float("-inf")  # cumulative r_edge sum along that path


# =========================================================================
# Tree helpers
# =========================================================================

def create_root(z_t, candidate_ids, candidate_priors, rf):
    """
    Build the root node from the action planner output.

    Input:
    - z_t: current fused embedding, shape [1, fused_dim]
    - candidate_ids: top-k token ids, shape [k] (long tensor)
    - candidate_priors: top-k probabilities, shape [k] (float tensor)
    - rf: reward features at the current real state, shape [1, RF_DIM]

    Output:
    - root TreeNode
    """
    root = TreeNode(
        z=z_t,
        candidate_ids=candidate_ids,
        candidate_priors=candidate_priors,
        depth=0,
        parent=None,
    )
    root.rf = rf
    return root


def expand_next_child(
    node,
    vocab_size,
    unmerge_fn,
    duration_fn,
    time_left,
    token_embed=None,
    intuition_mlp=None,
    reward_mlp=None,
    rf_predictor=None,
    planner_mlp=None,
    embed_dim=32,
    hidden_dim=128,
    top_k=3,
):
    """
    Open the next unopened candidate at `node`.

    1. Intuition head rolls through each base token to predict z_child.
    2. Reward head NN scores the (parent → child) transition.
    3. Action planner runs on z_child to populate child.candidate_ids,
       so this child can itself be explored further (enabling depth > 1).
    4. Value is backpropped up the tree.

    Input:
    - node:          TreeNode to expand from
    - vocab_size:    token vocabulary size
    - unmerge_fn:    callable(int) -> list[int]   base token ids for a merged token
    - duration_fn:   callable(int) -> int         token duration in frames
    - time_left:     int, frames remaining until deadline (from time_context)
    - token_embed, intuition_mlp: shared intuition head weights
    - reward_mlp:    reward head NN weights (None = create fresh)
    - rf_predictor:  reward feature predictor MLP (None = create fresh)
    - planner_mlp:   action planner MLP (None = create fresh)
    - embed_dim, hidden_dim: network dimensions
    - top_k:         how many candidates to generate for the child node

    Output:
    - child:          new TreeNode (already appended to node.children)
    - token_embed:    updated embedding table
    - intuition_mlp:  updated MLP
    - reward_mlp:     updated reward head MLP
    - rf_predictor:   updated reward feature predictor MLP
    - planner_mlp:    updated action planner MLP
    """

    idx      = node.next_unopened
    token_id = node.candidate_ids[idx]

    # -----------------------------------------------------------------
    # unpack merged token into base token sequence
    # -----------------------------------------------------------------

    base_tokens = unmerge_fn(int(token_id))

    # -----------------------------------------------------------------
    # roll the intuition head through each base token to get z_child
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
    # look up token duration and build input tensors for reward head
    # -----------------------------------------------------------------

    duration_frames = duration_fn(int(token_id))

    duration_tensor  = torch.tensor(
        [[float(duration_frames)]], dtype=torch.float32, device=node.z.device
    )
    time_left_tensor = torch.tensor(
        [[float(time_left)]], dtype=torch.float32, device=node.z.device
    )

    # -----------------------------------------------------------------
    # predict reward features at the child state
    # parent rf is real (from GTA); child rf must be predicted
    # -----------------------------------------------------------------

    delta_z = z_child - node.z

    rf_child, rf_predictor = predict_reward_features(
        z_parent     = node.z,
        delta_z      = delta_z,
        rf_parent    = node.rf,
        rf_predictor = rf_predictor,
        fused_dim    = node.z.shape[-1],
        hidden_dim   = hidden_dim // 2,
    )

    # -----------------------------------------------------------------
    # score the transition with the reward head NN
    # -----------------------------------------------------------------

    fused_dim = node.z.shape[-1]
    r_edge_tensor, reward_mlp = reward_head(
        z_parent   = node.z,
        z_child    = z_child,
        rf_parent  = node.rf,
        rf_child   = rf_child,
        duration   = duration_tensor,
        time_left  = time_left_tensor,
        reward_mlp = reward_mlp,
        fused_dim  = fused_dim,
        hidden_dim = hidden_dim,
    )

    r_edge = float(r_edge_tensor.item())

    # -----------------------------------------------------------------
    # ask action planner what tokens are good after this child
    # this populates the child's candidate set so EXPLORE can go deeper
    # -----------------------------------------------------------------

    # run intuition head one step from z_child to get z_next_pred
    # the action planner needs this to decide which tokens are promising next
    token_id_tensor = torch.tensor(
        [int(token_id)], dtype=torch.long, device=node.z.device
    )
    z_next_pred, _, token_embed, intuition_mlp = intuition_head(
        z_child,
        token_id_tensor,
        vocab_size    = vocab_size,
        token_embed   = token_embed,
        intuition_mlp = intuition_mlp,
        embed_dim     = embed_dim,
        hidden_dim    = hidden_dim,
    )

    planner_out = action_planner(
        z_t         = z_child.detach(),
        z_next_pred = z_next_pred,
        vocab_size  = vocab_size,
        planner_mlp = planner_mlp,
        hidden_dim  = hidden_dim,
        top_k       = top_k,
    )

    child_candidate_ids    = planner_out["top_k_ids"][0].detach()    # [top_k]
    child_candidate_priors = planner_out["top_k_probs"][0].detach()  # [top_k]
    planner_mlp = planner_out["planner_mlp"]

    # -----------------------------------------------------------------
    # create child node — fully populated so it can be expanded further
    # -----------------------------------------------------------------

    child = TreeNode(
        z=z_child.detach(),
        candidate_ids=child_candidate_ids,
        candidate_priors=child_candidate_priors,
        depth=node.depth + 1,
        parent=node,
    )
    child.token_id = token_id
    child.r_edge   = r_edge
    child.duration = duration_frames
    child.rf       = rf_child.detach()

    node.children.append(child)
    node.next_unopened = idx + 1

    # -----------------------------------------------------------------
    # backprop value up the tree
    # -----------------------------------------------------------------

    _backprop(child, r_edge)

    return child, token_embed, intuition_mlp, reward_mlp, rf_predictor, planner_mlp


def _backprop(node, value):
    """Propagate a value from `node` up to the root, updating N/W/Q."""
    current = node
    while current is not None:
        current.n += 1
        current.w += value
        current.q  = current.w / current.n
        current = current.parent


def cumulative_path_value(node):
    """Sum of r_edge values from root down to `node`."""
    total = 0.0
    current = node
    while current.parent is not None:
        total += current.r_edge
        current = current.parent
    return total


def path_to_node(node):
    """
    Return the list of token_ids from root down to `node`.

    Example: root → A → A3 → A32  returns  [A, A3, A32]
    The first element is the token to commit next in GTA.
    """
    path = []
    current = node
    while current.parent is not None:
        path.append(int(current.token_id))
        current = current.parent
    return list(reversed(path))


def collect_root_q(root):
    """
    Gather Q-values and token ids for all expanded root children.

    These are the root-level commitments the metacontroller can choose from.
    Regardless of how deep the search has gone, the final commitment is always
    one of the root children — executing that token next in GTA.

    Output:
    - candidate_q:   shape [1, num_children]
    - candidate_ids: shape [1, num_children]
    """
    if not root.children:
        return (
            torch.tensor([[0.0]]),
            torch.tensor([[0]], dtype=torch.long),
        )

    q_vals = [c.q        for c in root.children]
    t_ids  = [c.token_id for c in root.children]

    candidate_q   = torch.tensor([q_vals], dtype=torch.float32)
    candidate_ids = torch.tensor([t_ids],  dtype=torch.long)
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
    rf_current,
    planner_candidate_ids,
    planner_candidate_priors,
    running_token_id,
    z_running,
    vocab_size,
    unmerge_fn,
    duration_fn,
    token_embed=None,
    intuition_mlp=None,
    reward_mlp=None,
    rf_predictor=None,
    planner_mlp=None,
    meta_mlp=None,
    embed_dim=32,
    hidden_dim=128,
    top_k=3,
    training=False,
):
    """
    Initialize a new search.  Call once at the start of a token's duration.

    Returns a SearchState that gets passed to search_step each frame.

    Input:
    - z_t:                      current real fused embedding, shape [1, fused_dim]
    - rf_current:               reward features at current state, shape [1, RF_DIM]
    - planner_candidate_ids:    top-k token ids from action planner, shape [k]
    - planner_candidate_priors: top-k probs from action planner, shape [k]
    - running_token_id:         token currently being executed, shape [1] (long)
    - z_running:                state prediction when running token was committed,
                                shape [1, fused_dim]
    - vocab_size:               token vocabulary size
    - unmerge_fn:               callable(int) -> list[int]
    - duration_fn:              callable(int) -> int  (token_id -> duration in frames)
    - token_embed, intuition_mlp, reward_mlp, rf_predictor,
      planner_mlp, meta_mlp:   shared weights (None = lazily created)
    - embed_dim, hidden_dim:    network dimensions
    - top_k:                    candidates per node

    Output:
    - state: SearchState
    """
    root = create_root(z_t, planner_candidate_ids, planner_candidate_priors, rf_current)

    return SearchState(
        root             = root,
        z_running        = z_running,
        running_token_id = running_token_id,
        vocab_size       = vocab_size,
        unmerge_fn       = unmerge_fn,
        duration_fn      = duration_fn,
        token_embed      = token_embed,
        intuition_mlp    = intuition_mlp,
        reward_mlp       = reward_mlp,
        rf_predictor     = rf_predictor,
        planner_mlp      = planner_mlp,
        meta_mlp         = meta_mlp,
        embed_dim        = embed_dim,
        hidden_dim       = hidden_dim,
        top_k            = top_k,
        training         = training,
    )


def search_step(state, z_t, time_ctx):
    """
    Run one search step.  Call once per frame while the current token plays.

    Expands one node (intuition head + reward head NN), asks the
    metacontroller, applies the decision.  Mutates `state` in place.

    Input:
    - state:    SearchState from search_init (or previous search_step)
    - z_t:      latest real fused embedding (updated every frame), shape [1, fused_dim]
    - time_ctx: dict from time_context() with fresh elapsed_ratio, urgency, etc.

    Output dict:
    - decision:         EXPLORE / INTERRUPT / COMMIT_NEXT / ROLLBACK for this step
    - chosen_token_id:  int or None (only set on INTERRUPT / COMMIT_NEXT)
    - done:             bool, True if search has concluded
    - nodes_expanded:   int, total nodes expanded so far
    """

    if state.done:
        return {
            "decision":        state.final_decision,
            "chosen_token_id": state.chosen_token_id,
            "done":            True,
            "nodes_expanded":  state.nodes_expanded,
        }

    # -----------------------------------------------------------------
    # check budget
    # -----------------------------------------------------------------

    if time_ctx["budget_remaining"] <= 0:
        state.done = True
        if state.best_path:
            state.chosen_token_id = state.best_path[0]
            state.final_decision  = COMMIT_NEXT
        else:
            state.chosen_token_id = best_root_child(state.root)
            if state.chosen_token_id is not None:
                state.final_decision = COMMIT_NEXT
        return {
            "decision":        state.final_decision,
            "chosen_token_id": state.chosen_token_id,
            "done":            True,
            "nodes_expanded":  state.nodes_expanded,
            "best_path":       state.best_path,
            "best_path_value": state.best_path_value,
        }

    # -----------------------------------------------------------------
    # auto-rollback if current node is fully expanded
    # -----------------------------------------------------------------

    if state.current_node.fully_expanded:
        if state.current_node.parent is not None:
            state.current_node = state.current_node.parent
        else:
            # root is fully expanded — commit the best path we found
            state.done = True
            if state.best_path:
                state.chosen_token_id = state.best_path[0]
                state.final_decision  = COMMIT_NEXT
            else:
                state.chosen_token_id = best_root_child(state.root)
                if state.chosen_token_id is not None:
                    state.final_decision = COMMIT_NEXT
            return {
                "decision":        state.final_decision,
                "chosen_token_id": state.chosen_token_id,
                "done":            True,
                "nodes_expanded":  state.nodes_expanded,
                "best_path":       state.best_path,
                "best_path_value": state.best_path_value,
            }

    # -----------------------------------------------------------------
    # expand one node — intuition head + reward head NN + action planner
    # -----------------------------------------------------------------

    child, state.token_embed, state.intuition_mlp, state.reward_mlp, \
        state.rf_predictor, state.planner_mlp = expand_next_child(
        node          = state.current_node,
        vocab_size    = state.vocab_size,
        unmerge_fn    = state.unmerge_fn,
        duration_fn   = state.duration_fn,
        time_left     = int(time_ctx["frames_left"]),
        token_embed   = state.token_embed,
        intuition_mlp = state.intuition_mlp,
        reward_mlp    = state.reward_mlp,
        rf_predictor  = state.rf_predictor,
        planner_mlp   = state.planner_mlp,
        embed_dim     = state.embed_dim,
        hidden_dim    = state.hidden_dim,
        top_k         = state.top_k,
    )
    state.nodes_expanded += 1

    # -----------------------------------------------------------------
    # track best path — the full sequence with highest cumulative reward
    # e.g. [A, A3, A32] if that path had the best sum of r_edge values
    # -----------------------------------------------------------------

    path_val = cumulative_path_value(child)
    if path_val > state.best_path_value:
        state.best_path_value = path_val
        state.best_path       = path_to_node(child)

    # -----------------------------------------------------------------
    # gather root-level Q values
    # (always root children — that's what we'll commit to in GTA)
    # -----------------------------------------------------------------

    candidate_q, candidate_ids = collect_root_q(state.root)

    # -----------------------------------------------------------------
    # how many siblings of current_node are still unexplored
    # (signals whether to ROLLBACK and try them, or keep going deeper)
    # -----------------------------------------------------------------

    parent_unexplored = torch.tensor(
        [[state.current_node.unexplored_fraction]], dtype=torch.float32
    )

    # -----------------------------------------------------------------
    # path quality signals — these are what let the metacontroller learn
    # to evaluate branches, not just individual nodes
    #
    # current_path_value: cumulative r_edge from root to current_node
    #   → "how good is the branch I'm currently exploring?"
    # best_path_value: best cumulative r_edge seen in this search
    #   → "how good is the best plan I've found so far?"
    #
    # If current_path_value > best_path_value: this branch is beating
    #   our best plan — good reason to EXPLORE deeper
    # If current_path_value << best_path_value: this branch is bad
    #   → ROLLBACK and try something else
    # -----------------------------------------------------------------

    cur_path_val  = cumulative_path_value(state.current_node)
    best_path_val = state.best_path_value if state.best_path_value != float("-inf") else 0.0

    current_path_value_t = torch.tensor([[cur_path_val]],  dtype=torch.float32)
    best_path_value_t    = torch.tensor([[best_path_val]], dtype=torch.float32)

    # -----------------------------------------------------------------
    # current node's candidate info — what tokens are on the table,
    # how long each takes, and their Q values after any partial expansion
    # -----------------------------------------------------------------

    # Q for each candidate: actual Q if already expanded as a child, else 0
    child_q_map = {int(c.token_id): c.q for c in state.current_node.children}
    current_q_list = [
        child_q_map.get(int(cid), 0.0)
        for cid in state.current_node.candidate_ids.tolist()
    ]
    current_dur_list = [
        float(state.duration_fn(int(cid)))
        for cid in state.current_node.candidate_ids.tolist()
    ]
    current_candidate_q_t   = torch.tensor([current_q_list],   dtype=torch.float32)
    current_candidate_dur_t = torch.tensor([current_dur_list], dtype=torch.float32)

    # embed each candidate token so the metacontroller sees what kind of
    # actions are available (not just their aggregate quality)
    cids_tensor = state.current_node.candidate_ids.unsqueeze(0)   # [1, top_k]
    if state.token_embed is not None:
        token_embs = state.token_embed(cids_tensor).detach()       # [1, top_k, embed_dim]
        current_candidate_emb_t = token_embs.reshape(1, -1)        # [1, top_k * embed_dim]
    else:
        n_dims = cids_tensor.shape[-1] * state.embed_dim
        current_candidate_emb_t = torch.zeros(1, n_dims)

    # -----------------------------------------------------------------
    # ask metacontroller
    # -----------------------------------------------------------------

    meta_out = metacontroller(
        z_t                      = z_t,
        z_running                = state.z_running,
        running_token_id         = state.running_token_id,
        elapsed_ratio            = time_ctx["elapsed_ratio"],
        token_frames_left        = time_ctx["token_frames_left"],
        candidate_q              = candidate_q,
        candidate_ids            = candidate_ids,
        urgency                  = time_ctx["urgency"],
        parent_unexplored        = parent_unexplored,
        current_path_value       = current_path_value_t,
        best_path_value          = best_path_value_t,
        current_candidate_q      = current_candidate_q_t,
        current_candidate_durations = current_candidate_dur_t,
        current_candidate_emb    = current_candidate_emb_t,
        meta_mlp                 = state.meta_mlp,
        hidden_dim               = state.hidden_dim,
        training                 = state.training,
    )

    state.meta_mlp = meta_out["meta_mlp"]
    decision       = meta_out["decision"].item()

    # -----------------------------------------------------------------
    # record metalevel step (features stored for REINFORCE training)
    # -----------------------------------------------------------------

    best_q_val = candidate_q.max().item() if candidate_q.numel() > 0 else 0.0
    state.meta_trajectory.append({
        "decision":    decision,
        "features":    meta_out["features"],
        "predicted_q": best_q_val,
    })

    # -----------------------------------------------------------------
    # apply decision
    # -----------------------------------------------------------------

    chosen = None

    if decision == EXPLORE:
        # descend into the child we just expanded — go deeper on this branch
        state.current_node = child

    elif decision == ROLLBACK:
        # this branch is not promising — go back up, try a sibling next step
        if state.current_node.parent is not None:
            state.current_node = state.current_node.parent

    elif decision in (INTERRUPT, COMMIT_NEXT):
        state.done           = True
        state.final_decision = decision
        # commit the root token from the best path the search found
        # — not argmax Q, but the path the metacontroller's own exploration led to
        if state.best_path:
            state.chosen_token_id = state.best_path[0]
        else:
            state.chosen_token_id = meta_out["selected_token_id"].item()
        chosen = state.chosen_token_id

    return {
        "decision":        decision,
        "chosen_token_id": chosen,
        "done":            state.done,
        "nodes_expanded":  state.nodes_expanded,
        "best_path":       state.best_path,        # e.g. [A, A3, A32]
        "best_path_value": state.best_path_value,  # cumulative r_edge sum
    }
