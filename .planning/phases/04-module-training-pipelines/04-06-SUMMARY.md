---
phase: 04-module-training-pipelines
plan: 06
subsystem: data-capture
tags: [jsonl, data-capture, training-data, state-pairs, demonstrations]

# Dependency graph
requires:
  - phase: 04-01
    provides: "Shared training infrastructure, training_data/ directories"
provides:
  - "capture_states.py: StateCaptureSession for encoder+intuition and reward head data"
  - "capture_demos.py: DemoCaptureSession for action planner demonstrations"
  - "write_synthetic_encoder_data, write_synthetic_reward_data, write_synthetic_demo_data helpers"
  - "JSONL schema documentation in all three training_data/ README.md files"
  - "6 tests for capture script data writing and session classes"
affects: [main_model/train.py, reward_head/train.py, action_planner/train.py]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "StateCaptureSession: dual JSONL output for encoder+intuition and reward head from single gameplay session"
    - "DemoCaptureSession: raw controls capture with token_id=None per D-06/Pitfall 4"
    - "Session-timestamped JSONL with # comment header matching tokenizer pattern"
    - "compute_reward (reward, components) tuple unpacking for frame-level reward accumulation"

key-files:
  created:
    - capture_states.py
    - capture_demos.py
    - main_model/training_data/README.md
    - action_planner/training_data/README.md
    - tests/test_capture_scripts.py
  modified:
    - reward_head/training_data/README.md

key-decisions:
  - "compute_reward returns (reward, components) tuple -- unpack correctly in record_frame"
  - "token_start_state tracked separately from prev_state for accurate reward head state_before"
  - "token_started resets token_rewards AND saves state; token_ended writes reward record"
  - "Synthetic data helpers use datetime.now() for session timestamps (no user input in paths per T-04-18)"

patterns-established:
  - "Dual JSONL capture from single session (encoder + reward data from same frames)"
  - "Raw controls capture with deferred tokenization (D-06/Pitfall 4)"
  - "Synthetic data helper functions for test seeding without GTA"

requirements-completed: [PIPE-01, PIPE-02, PIPE-03, PIPE-04]

# Metrics
duration: 4min
completed: 2026-05-01
---

# Phase 4 Plan 06: Data Capture Scripts Summary

**State and demonstration capture utilities producing session-timestamped JSONL for encoder+intuition, reward head, and action planner training pipelines**

## Performance

- **Duration:** 4 min (225s)
- **Started:** 2026-05-01T08:40:19Z
- **Completed:** 2026-05-01T08:44:04Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Created capture_states.py with StateCaptureSession class that writes dual JSONL output per gameplay session: encoder+intuition state pairs (state_t, state_t1, token_id) and reward head records (state_before, state_after, duration, realized_return)
- Created capture_demos.py with DemoCaptureSession class that writes action planner demonstration JSONL (state, player_controls with throttle/brake/steer/handbrake, token_id=None per D-06)
- Built synthetic data helper functions (write_synthetic_encoder_data, write_synthetic_reward_data, write_synthetic_demo_data) for test seeding without GTA
- Created/updated README.md in all three training_data/ directories documenting expected JSONL schemas
- Added 6 comprehensive tests covering synthetic data writers, session class output, and JSONL comment headers
- Full test suite green at 80 tests (74 existing + 6 new)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create capture_states.py and capture_demos.py** - `d2b4b94` (feat)
2. **Task 2: Create training_data README.md files and capture tests** - `e3841c1` (test)

## Files Created/Modified

- `capture_states.py` - StateCaptureSession + synthetic encoder/reward data helpers (171 lines)
- `capture_demos.py` - DemoCaptureSession + synthetic demo data helper (108 lines)
- `main_model/training_data/README.md` - Encoder+intuition JSONL schema documentation
- `reward_head/training_data/README.md` - Reward head JSONL schema documentation (updated from training log schema)
- `action_planner/training_data/README.md` - Action planner JSONL schema documentation
- `tests/test_capture_scripts.py` - 6 tests for capture script data writing

## Decisions Made

- compute_reward returns (reward, components) tuple -- adapted from plan to unpack correctly in StateCaptureSession.record_frame
- token_start_state tracked separately from prev_state to accurately record the state at token start for reward head state_before field
- token_started event resets token_rewards accumulator AND saves the start state; token_ended event computes realized_return via compute_token_return and writes the reward record
- Session timestamps generated via datetime.now().strftime() -- never from user input (T-04-18 mitigation)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed compute_reward return value handling**
- **Found during:** Task 1
- **Issue:** Plan code called `reward = compute_reward(self.prev_state, raw_state)` but compute_reward returns a `(reward, components)` tuple
- **Fix:** Changed to `reward, _components = compute_reward(self.prev_state, raw_state)`
- **Files modified:** capture_states.py
- **Commit:** d2b4b94

**2. [Rule 2 - Missing critical functionality] Added token_start_state tracking**
- **Found during:** Task 1
- **Issue:** Plan code used `self.prev_state` as state_before for reward records, which would be the state one frame before token end rather than the state at token start
- **Fix:** Added `self.token_start_state` field that records state when `token_started=True`, used as state_before in reward records
- **Files modified:** capture_states.py
- **Commit:** d2b4b94

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- capture_states.py and capture_demos.py are ready for integration into the gameplay frame loop
- Synthetic data helpers enable testing of training scripts (Plans 02-04) without GTA
- All training_data/ directories have schema documentation for reference
- Full test suite green (80 tests)

## Self-Check: PASSED

All 6 created/modified files verified on disk. Both task commits (d2b4b94, e3841c1) verified in git log.

---
*Phase: 04-module-training-pipelines*
*Completed: 2026-05-01*
