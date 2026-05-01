# Phase 1: Training Correctness - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the three compounding bugs that corrupt every gradient update — argmax-only decisions, no entropy regularization, and missing penalty signals — plus add advantage normalization and duration-normalized returns. After this phase, the metacontroller training loop produces valid exploration and a coherent reward signal.

</domain>

<decisions>
## Implementation Decisions

### "Not ready" penalty (TRAIN-03)
- **D-01:** Penalty triggers on TWO conditions: (a) token expiry without the metacontroller issuing COMMIT_NEXT or INTERRUPT, and (b) search budget exhausted without a commit even if the token hasn't ended
- **D-02:** Penalty magnitude = `-C * token_duration_frames` — proportional to the available thinking budget. Failing with a 20-frame window is worse than failing with a 5-frame window. C is a fixed constant (Claude picks)
- **D-03:** On trigger, fall back to planner's top-1 so the car keeps driving — the agent still acts, but the trajectory records this as a failure
- **D-04:** Add a boolean flag in the trajectory distinguishing fallback-commit from real commit, so "no-decision rate" can be tracked separately from "bad decision rate"

### Lazy commit penalty (TRAIN-04)
- **D-05:** Trigger: COMMIT_NEXT with zero or minimal nodes expanded
- **D-06:** Penalty also proportional to `token_duration_frames` — committing without searching on a long token wastes more thinking opportunity than on a short token. Short tokens get near-zero penalty (the metacontroller barely had time to search anyway)
- **D-07:** Same design philosophy as "not ready" — both penalties reflect wasted thinking opportunity, scaled by available budget

### Duration normalization (TRAIN-06)
- **D-08:** Normalize token returns by `token_return / sqrt(num_frames)` — dampens variable-length BPE bias without fully erasing duration signal
- **D-09:** sqrt chosen over full division specifically to preserve genuine short-burst reward signals (e.g., surprise events where a single frame yields exceptional reward)

### Entropy annealing (TRAIN-02)
- **D-10:** Linear decay from 0.05 → 0.005 over N gradient steps
- **D-11:** Claude picks a sensible default for N, exposed as a configurable parameter for later tuning

### Claude's Discretion
- Training mode flag mechanism (how to switch between categorical sampling during training and argmax at inference — TRAIN-01)
- Advantage normalization implementation (TRAIN-05 — zero-mean unit-variance across metalevel trajectory)
- Exact penalty constants C and K for not-ready and lazy-commit penalties
- Default step count N for entropy annealing schedule
- Where exactly each penalty is injected in the training code (metalevel rewards vs separate loss term)

</decisions>

<specifics>
## Specific Ideas

- Both penalty signals (not-ready and lazy-commit) follow the same principle: penalty ∝ available thinking budget that was wasted. This creates a unified incentive — the metacontroller learns to use its thinking time proportionally to what's available.
- "If we scale the not-ready penalty with search depth or Q-value, it might simply learn to not make the trees big" — penalty must be fixed relative to time budget, not search outcome.
- sqrt normalization for duration keeps short-burst surprise rewards visible — "if 1 frame token does give huge reward it can take it."

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

No external specs — requirements are fully captured in decisions above and in:

### Project context
- `.planning/PROJECT.md` — Architecture overview, training order, known code issues
- `.planning/REQUIREMENTS.md` — TRAIN-01 through TRAIN-06 requirement definitions
- `.planning/ROADMAP.md` — Phase 1 success criteria (5 criteria that must be TRUE)

### Source files to modify
- `metacontroller/metacontroller.py` — Line 166: argmax bug, needs training-mode sampling
- `metacontroller/trainer.py` — `update_metapolicy()`: add entropy term; `compute_metalevel_advantages()`: add penalty signals and advantage normalization; `compute_token_return()`: add duration normalization
- `metacontroller/frame_loop.py` — `drive_token()` lines 254-257: add fallback flag and penalty injection
- `metacontroller/reward.py` — Formula-based reward (read-only reference, not modified in this phase)
- `reward_head/reward_head.py` — NN reward head (read-only reference, not modified in this phase)
- `metacontroller/time_context.py` — Timing signals already computed correctly, consumed by metacontroller

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `time_context.py` already computes `elapsed_ratio`, `token_frames_left`, `urgency`, `budget_remaining` — all signals needed for penalty logic
- `compute_metalevel_advantages()` in `trainer.py` already has the metalevel reward assignment loop where penalty terms can be injected
- `meta_trajectory` dicts already store `features`, `decision`, `decision_logits` — just need to add the fallback flag

### Established Patterns
- Manual SGD everywhere (will be replaced by Adam in Phase 2 — don't introduce Adam here)
- MLPs created lazily with `if meta_mlp is None` pattern — reuse this for any new components
- All training happens inline in `drive_token()` after each token completes

### Integration Points
- `frame_loop.py:drive_token()` is the main entry point — penalty signals need to flow from here into `train_step()`
- `search_state.meta_trajectory` carries the trajectory from search into training
- `search_state.nodes_expanded` tracks search depth (used for lazy commit detection)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-training-correctness*
*Context gathered: 2026-04-30*
