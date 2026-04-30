# Metacontroller

The metacontroller is the brain that turns search tree results into a single
token-level decision.  It sits between the action planner (which proposes
candidates) and the executor (which sends commands to GTA).

## The Big Picture

The current token is already driving the car.  While it plays, the
metacontroller uses that time to think about what to do next.

```
  GTA world
      |
      v
  main_model encodes everything into z_t [128]
      |
      v
  action_planner proposes top-k candidate tokens
      |
      v
  +-------------------------------------------------------+
  |                   FRAME LOOP                           |
  |                                                        |
  |  each frame, two things happen at the same time:       |
  |                                                        |
  |  executor plays one frame       search_tree thinks     |
  |  of the current token           one step about the     |
  |  (sends controls to GTA)        next token             |
  |                                                        |
  |  when current token ends:                              |
  |    next token ready? use it                            |
  |    not ready? use planner's top-1                      |
  +-------------------------------------------------------+
      |
      v
  trainer learns from what happened
  (realized return vs predicted, metalevel credit assignment)
```

## Files

```
metacontroller/
  metacontroller.py   the decision MLP (KEEP / INTERRUPT / COMMIT_NEXT / ROLLBACK)
  search_tree.py      tree nodes + init/step interface + metalevel trajectory recording
  time_context.py     external timing signals (urgency, budget, elapsed ratio)
  executor.py         per-frame token runner (one frame at a time, not a batch)
  frame_loop.py       the main driver — interleaves executor + search per frame
  trainer.py          computes returns, metalevel credit assignment, weight updates
```

## The Frame Loop

This is the core idea.  The current token buys time for thinking.

```
current token = merged_token_500, duration = 20 frames

frame 1:   executor plays frame 1   |   search expands one node
frame 2:   executor plays frame 2   |   search expands one node
frame 3:   executor plays frame 3   |   search expands one node
...
frame 20:  executor plays frame 20  |   search should have next token ready
```

The metacontroller is NOT choosing the current frame's action.  The current
token is already handling that.  The metacontroller is choosing: what should
I do AFTER this token finishes?

```
1. start current token
2. while current token is running:
     - send next control frame to GTA
     - read latest game state
     - encode state into z_t
     - search_step: expand one node, ask metacontroller
3. when current token ends:
     - if metacontroller found next token: use it
     - otherwise: use planner's top-1 (fallback)
4. trainer learns from the rollout
5. repeat with next token
```

If the metacontroller says INTERRUPT mid-token, the executor stops early
and switches immediately.  GTA never waits.

## How the Search Tree Works

Each token duration builds a fresh tree.  One search step per frame.

**Step 1: The action planner gives us candidates.**

The planner looks at the current state z_t and says "here are the top 3 tokens
I think are good."  These become the root's children-to-explore.

```
         [root: z_t]
        /     |      \
   token_A  token_B  token_C     <-- candidates from planner
   (unopened) (unopened) (unopened)
```

**Step 2: We explore ONE candidate at a time.**

We pick the first unopened candidate and ask the intuition head:
"If I do this token, what will the world look like?"

```
         [root: z_t]
        /     |      \
   [token_A]  token_B  token_C
   z_child_A  (unopened) (unopened)
   r = 0.7
```

**Step 3: Unpacking merged tokens.**

Our tokenizer uses BPE, so a single token might actually be a sequence of
base actions merged together.  Before the intuition head can predict, we
unpack the merged token and roll through each base token one by one:

```
merged token 500 --> unmerge --> [base_12, base_7, base_3]

z_0 = node.z (starting state)
                    |
     intuition_head(z_0, base_12)
                    |
                    v
                   z_1
                    |
     intuition_head(z_1, base_7)
                    |
                    v
                   z_2
                    |
     intuition_head(z_2, base_3)
                    |
                    v
                   z_3  <-- this is z_child, the predicted state
                            after the full merged token plays out
```

A base token (not merged) just returns itself: `unmerge(12) -> [12]`.
One pass through the intuition head, done.

**Step 4: The reward function scores the transition.**

```
reward_fn(z_parent, z_child) --> r_edge
```

This is a learned reward estimator that works on embeddings.  It is NOT the
same as the real reward_head (which needs actual GTA frames).

**Step 5: The metacontroller decides.**

After each expansion, the metacontroller looks at everything and picks one
of four decisions:

```
                  METACONTROLLER
                       |
    inputs:            |           output:
    - drift            |           one of four decisions
    - elapsed_ratio    |
    - best_q           +---------> KEEP
    - mean_q           |           INTERRUPT
    - urgency          |           COMMIT_NEXT
    - parent_unexplored|           ROLLBACK
```

## The Four Decisions

### KEEP

Keep expanding at the current node.  "I'm not done looking yet."

```
         [root]
        /   |   \
     [A]    B    C
    /   \
  [A1]  A2        <-- KEEP: open A2 next
```

Use when:
- The current token still looks fine
- The tree does not show a clearly better option
- There is remaining budget to explore

### INTERRUPT

Stop the current token early and switch to the best candidate.
"The current action is going badly, switch NOW."

```
Currently executing token X in GTA...
Metacontroller says INTERRUPT
--> stop token X mid-execution
--> immediately start the best candidate from the tree
```

Use when:
- The current token looks bad compared to a new candidate
- Time pressure is high
- Continuing is not worth the risk

### COMMIT_NEXT

Select the best candidate as the next token.  "I'm confident, let's go."

```
Search tree found that token_A has Q = 0.9
Metacontroller says COMMIT_NEXT
--> when current token finishes, start token_A
```

Use when:
- One candidate is clearly the best
- Search confidence is high
- Time to move forward

### ROLLBACK

Backtrack to the parent node and try a different branch.
"This path is not great, let me think more."

```
Before ROLLBACK:                  After ROLLBACK:

     [root]                           [root]
    /   |   \                        /   |   \
  [A]   B    C                     [A]  [B]   C
  / \                              / \    \
[A1][A2]  <-- all bad            [A1][A2] [B1]  <-- trying B now
```

Use when:
- All children at the current node look bad
- The parent still has unexplored siblings
- The value of thinking more (exploring other branches) is high

If the metacontroller keeps rolling back and trying new branches, it
accumulates Q-values across ALL explored paths.  When it finally decides
COMMIT_NEXT, it picks the best one it found across everything.

## Time Context

The metacontroller needs to know about the real world while it thinks.
The time_context module computes this from raw frame counts:

```
raw signals                          tensors for metacontroller
-----------                          -------------------------
current_frame = 1042        -->      elapsed_ratio = 0.6
deadline_frame = 1060                urgency = 0.7
token_start_frame = 1030             budget_remaining = 4
token_duration = 20 frames
nodes_expanded = 6
max_budget = 10
```

Time context is refreshed at EVERY node expansion, not just once at the
start.  As the search burns frames thinking, urgency goes up and budget
goes down.  The metacontroller sees this in real time.

## The Executor

The executor is a per-frame token runner.  It plays ONE frame at a time
so the frame loop can interleave it with search.

```
execution_init(token_id, token_table)
     |
     v
execution_frame()  <-- called once per frame by the frame loop
     |
     +-- send controls to GTA
     +-- read new state
     +-- compute env reward for this one frame
     +-- record to rollout
     |
     v
get_rollout()  <-- called when token is done, returns everything
```

The executor exposes init + frame + get_rollout, not a blocking loop.
The frame loop calls execution_frame and search_step alternately.

The environment reward (env_reward_fn) is the hand-designed formula from
reward_head.py applied to real GTA frames.  It is ground truth, not a
learned predictor.  The learned reward estimator (reward_fn) is a different
thing that only runs inside the search tree on embeddings.

## The Trainer

Learning happens in trainer.py, completely separate from execution.
The trainer takes the rollout from the executor and does three things.

### 1. Compute the Realized Token Return

A committed token runs for multiple frames.  The thing we compare against
the tree's predicted value is the discounted sum of ALL per-frame rewards,
not just one instant reward.

```
token runs for k frames, producing rewards r_0, r_1, ..., r_{k-1}

R_token = r_0 + gamma * r_1 + gamma^2 * r_2 + ... + gamma^(k-1) * r_{k-1}

optionally bootstrapped:

R_token = (sum of discounted rewards) + gamma^k * V(next_real_state)
```

Why bootstrapping?  Because the token ends but the episode continues.  The
value of the state we land in matters.  Without it we are only judging the
token by what happened during it, ignoring where it left us.

### 2. Back Up the Return into the Tree

The tree predicted a Q-value for the committed branch.  Now we know the
actual return.  We update that branch's N/W/Q so the tree reflects reality.

```
before backup:                    after backup:
  child.q = 0.7 (predicted)        child.q = 0.65 (blended with real)
  child.n = 3                       child.n = 4
```

### 3. Metalevel Credit Assignment

This is the key part that makes the metacontroller actually learn.

During search, the metacontroller made a sequence of decisions:

```
step 0: expand A     --> KEEP        (cost: -think_cost)
step 1: expand A1    --> KEEP        (cost: -think_cost)
step 2: expand A2    --> ROLLBACK    (cost: -think_cost)
step 3: expand B     --> KEEP        (cost: -think_cost)
step 4: expand B1    --> COMMIT_NEXT (gets: R_token)
```

Every KEEP and ROLLBACK costs a small think_cost (time spent deliberating
instead of acting).  The final COMMIT gets the realized token return.

We compute discounted returns backward from the end:

```
step 4: return = R_token
step 3: return = -think_cost + gamma * R_token
step 2: return = -think_cost + gamma * (step 3 return)
step 1: return = -think_cost + gamma * (step 2 return)
step 0: return = -think_cost + gamma * (step 1 return)
```

Then each step's advantage = its return - the predicted Q at that step.

```
advantage > 0:  "this search decision led to a good outcome, do it more"
advantage < 0:  "this search decision wasted time or led somewhere bad"
```

This means:
- If ROLLBACK at step 2 led to finding a better branch (B) that produced
  high return, that ROLLBACK gets positive advantage.  The metacontroller
  learns that rolling back in similar situations is a good idea.
- If a series of KEEPs explored a dead end before the commit, those KEEPs
  get negative advantage (they cost think_cost and did not help).
- If the final COMMIT picked a bad token, the COMMIT itself gets blame,
  but so do the earlier decisions that failed to find something better.

### The Full Learning Flow

```
search_tree returns:
  - chosen_token_id
  - root (tree)
  - meta_trajectory (every search decision + its logits + predicted Q)
          |
          v
executor plays token in GTA
          |
          v
rollout (per-frame rewards, states)
          |
          v
trainer.train_step:
          |
          +-- compute_token_return(rollout) --> R_token
          |
          +-- backup_tree(root, R_token) --> update tree N/W/Q
          |
          +-- compute_metalevel_advantages(meta_trajectory, R_token)
          |     --> advantages for every search step
          |
          +-- update_metapolicy(meta_mlp, meta_trajectory, advantages)
                --> policy gradient over the full search trajectory
```

## What Talks to What

```
                     action_planner
                          |
                     top-k candidates
                          |
                          v
  +----------------------------------------------------------+
  |                    FRAME LOOP                             |
  |                                                           |
  |  per frame:                                               |
  |                                                           |
  |  executor.execution_frame()     search_tree.search_step() |
  |       |                              |     |              |
  |       +-> GTA (send controls)        |     +-> intuition  |
  |       +-> env_reward_fn              |     +-> reward_fn  |
  |       +-> record to rollout          |                    |
  |                                      +-> metacontroller   |
  |                                      +-> record to        |
  |                                          meta_trajectory  |
  |                                                           |
  |  time_context refreshed every frame                       |
  +----------------------------------------------------------+
       |                            |
       v                            v
   rollout                     search_state
   (per-frame rewards)         (meta_trajectory, root, chosen_token_id)
       |                            |
       +------------+---------------+
                    |
                    v
                trainer
                    |
                    +-> compute_token_return (discounted sum + bootstrap)
                    +-> backup_tree (update tree N/W/Q with real return)
                    +-> compute_metalevel_advantages (credit every search step)
                    +-> update_metapolicy (policy gradient on full trajectory)
```

## Drift: Why INTERRUPT Exists

When the metacontroller committed to a token earlier, the intuition head
predicted what the world would look like (z_running).  But the real world
keeps changing.  The drift is how wrong that prediction turned out to be:

```
drift = z_t (what actually happened) - z_running (what we predicted)

small drift --> prediction was good, KEEP is fine
large drift --> world changed unexpectedly, consider INTERRUPT
```

The metacontroller sees this drift as part of its input and learns when
a large drift means it should switch actions.

## Parent Unexplored: Why ROLLBACK Exists

The metacontroller also sees how many siblings at the current node are
still unexplored:

```
parent_unexplored = 0.0   all siblings tried, ROLLBACK is pointless
parent_unexplored = 0.66  2 out of 3 siblings still untried, ROLLBACK has options
parent_unexplored = 1.0   nothing explored yet at this level
```

High parent_unexplored + bad Q-values at current node = ROLLBACK makes sense.
Low parent_unexplored = already tried everything here, just commit the best.
