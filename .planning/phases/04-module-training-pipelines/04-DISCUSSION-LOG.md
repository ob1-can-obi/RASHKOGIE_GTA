# Phase 4: Module Training Pipelines - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-01
**Phase:** 04-module-training-pipelines
**Areas discussed:** Encoder pretraining, Data collection strategy, Training orchestration, Convergence & freeze

---

## Encoder Pretraining

### Pretraining objective

| Option | Description | Selected |
|--------|-------------|----------|
| Reconstruction | Decoder MLP maps z_t back to raw state. Loss = MSE(decoded, original). | |
| Next-state prediction | Joint with intuition head — encoder learns representations for z_{t+1} prediction. Loss = MSE(predicted z_{t+1}, real z_{t+1}). | ✓ |
| Contrastive (SimCLR-style) | Contrasting same-state augmentations vs different states. Needs negative sampling. | |
| Skip pretraining | Let encoder train end-to-end through downstream gradients only. | |

**User's choice:** Next-state prediction
**Notes:** Ties encoder quality directly to what downstream modules need.

### Training flow

| Option | Description | Selected |
|--------|-------------|----------|
| Joint training | Encoder + intuition head share same loop and loss. Gradients flow back through encoder. | ✓ |
| Sequential pretrain | Encoder pretrains alone first with temporary head, freezes, then intuition head trains separately. | |

**User's choice:** Joint training
**Notes:** Simplifies pipeline — one loop, one optimizer group for encoder + intuition head.

### Data source

| Option | Description | Selected |
|--------|-------------|----------|
| Live gameplay | Train on (state_t, state_{t+1}) pairs as car drives. Real-time, no storage. | |
| Saved replays | Record raw state sequences to disk, train offline. Replayable, faster than real-time. | |
| Both | Record during gameplay, train offline from saved data. | ✓ |

**User's choice:** Both (capture during gameplay, train offline)
**Notes:** Decouples data collection speed from training speed.

---

## Data Collection Strategy

### Data format

| Option | Description | Selected |
|--------|-------------|----------|
| JSONL everywhere | All modules use JSONL. Consistent with tokenizer captures. Human-readable, appendable. | ✓ |
| PyTorch tensors (.pt) | Pre-tensorized data. Faster to load, smaller. Harder to inspect. | |
| You decide | Claude picks best format per module. | |

**User's choice:** JSONL everywhere
**Notes:** Consistency with existing tokenizer/training_data/ pattern.

### Demo capture for action planner

| Option | Description | Selected |
|--------|-------------|----------|
| Automatic during gameplay | Records encoder output z_t and token during play. Requires trained encoder. | |
| Dedicated capture mode | Separate 'record my driving' mode. Saves raw states + actions. Encoder processes later. | ✓ |
| Reuse tokenizer captures | Re-encode existing tokenizer session data through trained encoder. | |

**User's choice:** Dedicated capture mode
**Notes:** Decouples capture from encoder state — no dependency on encoder being ready.

### Data layout

| Option | Description | Selected |
|--------|-------------|----------|
| Per-module folders | Each module gets own training_data/ subfolder. Matches existing pattern. | ✓ |
| Central root folder | One training_data/ at project root with subfolders per module type. | |

**User's choice:** Per-module folders
**Notes:** Matches existing reward_head/training_data/ and tokenizer/training_data/.

---

## Training Orchestration

### Pipeline orchestration

| Option | Description | Selected |
|--------|-------------|----------|
| Single script per module | Separate Python scripts, user runs in order manually. Simple, debuggable. | |
| One orchestrator script | Single train_pipeline.py runs all stages in sequence. Fully automated. | |
| Per-module + coordinator | Separate training scripts + lightweight coordinator for status/guidance. | ✓ |

**User's choice:** Per-module + coordinator
**Notes:** Modular and debuggable with guidance on ordering.

### Script location

| Option | Description | Selected |
|--------|-------------|----------|
| Inside each module dir | e.g., intuition_head/train.py. Close to code. Coordinator at root. | ✓ |
| Central training/ folder | All scripts in new training/ directory. Keeps training separate from modules. | |
| You decide | Claude picks based on each module. | |

**User's choice:** Inside each module dir
**Notes:** Clear ownership, close to the code they train.

### Coordinator design

| Option | Description | Selected |
|--------|-------------|----------|
| Status file + CLI | Reads/writes training_status.json. Stateless between runs. | ✓ |
| Long-running daemon | Runs continuously, monitors, auto-triggers. Needs process management. | |
| You decide | Claude designs based on project setup. | |

**User's choice:** Status file + CLI
**Notes:** Lightweight, stateless. Perfect for single-machine research setup.

---

## Convergence & Freeze

### Convergence detection

| Option | Description | Selected |
|--------|-------------|----------|
| Patience-based | Stop when loss hasn't improved for N evaluations. Scale-agnostic. | |
| Fixed thresholds only | Hard MSE/accuracy cutoffs. Simple, predictable. Risk of wrong threshold. | |
| Both: threshold + patience | Must pass minimum threshold AND show no improvement for N evals. | ✓ |

**User's choice:** Both: threshold + patience
**Notes:** Belt and suspenders — ensures quality AND stable convergence.

### Freeze type

| Option | Description | Selected |
|--------|-------------|----------|
| Hard freeze | Once frozen, stays frozen. Bad representations = start new run. | ✓ |
| Soft freeze with unfreeze | Can unfreeze if downstream degrades. Risk of oscillation. | |

**User's choice:** Hard freeze
**Notes:** Clean separation, no oscillation risk.

### Threshold configurability

| Option | Description | Selected |
|--------|-------------|----------|
| Config file | Thresholds in training_config.json. Easy to tune without code changes. | ✓ |
| Hardcoded constants | Default values as Python constants. Simpler but less flexible. | |
| You decide | Claude picks based on project patterns. | |

**User's choice:** Config file
**Notes:** Easy to tune, prepares for Phase 5 dashboard integration.

---

## Claude's Discretion

- Default patience window size
- JSONL schema per module
- Coordinator CLI interface design
- training_status.json schema
- Evaluation frequency during training
- Initial convergence threshold values

## Deferred Ideas

None — discussion stayed within phase scope
