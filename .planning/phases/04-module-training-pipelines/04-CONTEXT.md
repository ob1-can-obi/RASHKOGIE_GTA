# Phase 4: Module Training Pipelines - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the strict staged training chain: encoder + intuition head (joint) → reward head → freeze → action planner (imitation) → metacontroller (RL). Each module gets a standalone training script, a data collection mechanism, and convergence detection with hard freeze. A lightweight coordinator tracks pipeline state and guides the user through stages.

</domain>

<decisions>
## Implementation Decisions

### Encoder pretraining objective
- **D-01:** Next-state prediction objective — encoder learns representations that make z_{t+1} prediction easier (not reconstruction, not contrastive)
- **D-02:** Joint training with intuition head in the same loop — shared MSE loss on z_{t+1} prediction, gradients flow from intuition head back through encoder via autograd
- **D-03:** This means the training order is: (encoder + intuition head jointly) → reward head → freeze both → action planner → metacontroller RL

### Data collection strategy
- **D-04:** JSONL format for all training data files — consistent with existing tokenizer/training_data/captures/ pattern. One JSON object per line, session-timestamped filenames
- **D-05:** Both live capture + offline training — record raw state sequences during gameplay sessions to disk, then train offline from saved files. Decouples data collection speed from training speed
- **D-06:** Dedicated capture mode for action planner demonstrations — saves raw states + player actions during human driving. Encoder processes them later when trained. No dependency on encoder being ready at capture time
- **D-07:** Per-module training_data/ subfolders — each module gets its own training_data/ directory matching the existing reward_head/training_data/ and tokenizer/training_data/ pattern

### Training orchestration
- **D-08:** Per-module training scripts + lightweight coordinator — separate train.py inside each module directory (e.g., intuition_head/train.py, reward_head/train.py)
- **D-09:** Training scripts live inside their module directory, not in a central training/ folder — clear ownership, close to the code they train
- **D-10:** Coordinator uses a training_status.json file + CLI interface — stateless between runs, reads/writes status, tells user what to run next. Not a long-running daemon.

### Convergence and freeze mechanism
- **D-11:** Dual convergence criteria — module must pass minimum quality threshold (MSE < X, accuracy > Y) AND show no improvement for N consecutive evaluations (patience window). Belt and suspenders.
- **D-12:** Hard freeze — once a module is frozen, it stays frozen for the entire training run. If downstream training reveals bad representations, start a new run from scratch. No unfreeze/oscillation.
- **D-13:** Convergence thresholds stored in a training_config.json — easy to tune without code changes. Coordinator and training scripts both read from this config. Prepares for Phase 5 dashboard integration.

### Claude's Discretion
- Default patience window size (N evaluations before declaring convergence)
- Exact JSONL schema per module (what fields each record contains)
- Coordinator CLI interface design (commands, output format)
- How training_status.json tracks per-module state (schema)
- Evaluation frequency during training (every N batches)
- Initial convergence threshold values (MSE < 0.05/0.1, accuracy > 60% as starting points)

</decisions>

<specifics>
## Specific Ideas

- Encoder + intuition head are ONE training stage, not two — they share the same loss and optimizer. This simplifies the pipeline from 5 stages to 4: (enc+intuition) → reward → freeze → action planner → metacontroller
- Dedicated capture mode for player demonstrations means action planner training data can be collected at any time, even before the encoder is trained. Raw states are saved and processed later.
- The coordinator is a "what's next?" tool, not an automation engine — the user stays in control of when to advance stages

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

No external specs — requirements are fully captured in decisions above and in:

### Project context
- `.planning/PROJECT.md` — Architecture overview, training order, module dependencies
- `.planning/REQUIREMENTS.md` — PIPE-01 through PIPE-07 requirement definitions
- `.planning/ROADMAP.md` — Phase 4 success criteria (6 criteria that must be TRUE)

### Prior phase context (training infrastructure)
- `.planning/phases/01-training-correctness/01-CONTEXT.md` — Penalty signals, duration normalization, entropy annealing decisions
- `.planning/phases/02-batch-training-checkpointing/02-01-PLAN.md` — TrainingState class, batch update methods, Adam optimizer setup
- `.planning/phases/02-batch-training-checkpointing/02-02-PLAN.md` — Checkpoint save/load system

### Source files to build on
- `metacontroller/trainer.py` — TrainingState class (line 591+), train_step(), train_reward_head(), batch update methods
- `metacontroller/frame_loop.py` — drive_token() (line 117+), main integration point for metacontroller RL
- `main_model/main_model.py` — encode_state() (line 226), encoder architecture to train
- `intuition_head/intuition_head.py` — intuition_head() (line 17), joint training partner for encoder
- `reward_head/reward_head.py` — reward_head() (line 113), reward_mlp and rf_predictor to train
- `action_planner/action_planner.py` — action_planner() (line 22), imitation learning target
- `metacontroller/reward.py` — compute_reward() (line 20), ground truth reward formula (read-only reference)

### Existing training data structure (pattern to follow)
- `tokenizer/training_data/captures/` — JSONL session files (naming pattern to replicate)
- `reward_head/training_data/` — Stats and graphs (reference for per-module data layout)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `TrainingState` class (trainer.py:591+): trajectory buffer, Adam optimizers, batch update methods, checkpoint save/load — reuse for all module training
- `train_reward_head()` (trainer.py): reward head training already partially implemented — needs standalone wrapper
- Checkpoint system: `save_checkpoint()` / `load_checkpoint()` on TrainingState — extend for new per-module checkpoints
- Tokenizer JSONL capture pattern: session-timestamped filenames, appendable format

### Established Patterns
- Stateless weight pattern: all modules use `output, mlp = module(input, mlp=None)` — weights passed in/out each call
- Gradient blocking: `.detach()` already used in action_planner.py and search_tree.py child expansion
- Module independence: each module can be trained, checkpointed, and loaded independently
- Adam optimizer with gradient clipping (max_norm=0.5) from Phase 2

### Integration Points
- `drive_token()` is the existing integration point for metacontroller RL — this remains unchanged for the final stage
- Encoder training needs a NEW training loop (not in drive_token) since it runs offline on saved data
- Reward head training exists in train_step() but needs standalone offline loop too
- Action planner needs entirely new imitation learning loop
- Coordinator reads training_status.json and per-module checkpoint files to determine pipeline state

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-module-training-pipelines*
*Context gathered: 2026-04-30*
