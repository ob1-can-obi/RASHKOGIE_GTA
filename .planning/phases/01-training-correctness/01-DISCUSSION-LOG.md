# Phase 1: Training Correctness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 01-training-correctness
**Areas discussed:** "Not ready" penalty design, Lazy commit criteria, Duration normalization, Entropy annealing shape

---

## "Not ready" penalty design

### Q1: What should trigger the "not ready" penalty?

| Option | Description | Selected |
|--------|-------------|----------|
| Token expiry only | Penalty fires only when current token finishes and metacontroller never committed | |
| Token expiry + budget exhausted | Also penalize if search budget runs out without commit | ✓ |
| You decide | Claude picks based on frame loop interaction | |

**User's choice:** Token expiry + budget exhausted
**Notes:** None

### Q2: Fallback behavior when metacontroller produces no decision?

| Option | Description | Selected |
|--------|-------------|----------|
| Fallback + penalty | Use planner's top-1, apply not-ready penalty | |
| Fallback + penalty + flag | Same + boolean flag distinguishing fallback from real commit | ✓ |
| Repeat last token + penalty | Repeat current token instead of planner's top-1 | |
| You decide | Claude picks | |

**User's choice:** Initially chose fallback + penalty (option 1), then asked what flag does. After explanation, added the flag.
**Notes:** User asked "what is flag for?" — explained it tracks no-decision rate separately from bad-decision rate.

### Q3: How large should the "not ready" penalty be?

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed large negative (e.g., -10) | Comparable to collision, simple | |
| Scale with token value | Penalty = negative of best Q found | |
| Scale with time wasted | Penalty ∝ search steps taken before failing | |
| You decide | Claude picks reasonable magnitude | |

**User's choice:** Fixed — but then refined to proportional to token_duration_frames
**Notes:** User's key reasoning: "the penalty must be in accordance with the time it had to make a decision." Penalty = -C * token_duration_frames. Rejected scaling with search depth/Q-value: "if we scale, it might simply learn to not make the trees big."

---

## Lazy commit criteria

### Q4: What counts as "lazy" and how big is the penalty?

| Option | Description | Selected |
|--------|-------------|----------|
| Zero nodes expanded | Only penalize with literally no search | |
| Fewer than N nodes | Minimum threshold (e.g., 3 nodes) | |
| You decide | Claude picks threshold | |

**User's choice:** Penalty proportional to token_duration_frames (same philosophy as not-ready)
**Notes:** "This also must depend on the time it had to make a decision. If it has very very less time it might not wish to search. But even though it had time and didn't then it must have penalty." Same design principle as TRAIN-03 — both about wasted thinking opportunity.

---

## Duration normalization

### Q5: How should we normalize for variable-length BPE tokens?

| Option | Description | Selected |
|--------|-------------|----------|
| Divide by duration | token_return / num_frames, fully removes bias | |
| Divide by sqrt(duration) | Partial correction, dampens but doesn't eliminate | ✓ |
| Divide by duration with clamp | Clamp minimum frames to prevent inflation | |
| You decide | Claude picks numerically stable approach | |

**User's choice:** Divide by sqrt(duration)
**Notes:** "if 1 frame token does give huge reward it can take it. It will be useful for situations where there are surprises."

---

## Entropy annealing shape

### Q6: What should the annealing schedule look like?

| Option | Description | Selected |
|--------|-------------|----------|
| Linear decay over N updates | Simple ramp 0.05 → 0.005 | ✓ |
| Exponential decay | Fast early, slows later | |
| Step decay | Hold, drop, hold, drop | |
| You decide | Claude picks | |

**User's choice:** Linear decay
**Notes:** None

### Q7: What should N be?

| Option | Description | Selected |
|--------|-------------|----------|
| Configurable parameter | Default ~10K, expose for dashboard tuning | |
| You decide | Claude picks sensible default | ✓ |

**User's choice:** You decide
**Notes:** None

---

## Claude's Discretion

- Training mode flag mechanism (sampling vs argmax switch)
- Advantage normalization implementation (zero-mean unit-variance)
- Exact penalty constants (C, K)
- Default N for entropy annealing
- Where penalties are injected in code

## Deferred Ideas

None
