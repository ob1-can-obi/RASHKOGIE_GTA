---
phase: 4
slug: module-training-pipelines
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/ directory (existing) |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-T1 | 01 | 1 | PIPE-01, PIPE-05, PIPE-06 | — | N/A | unit | `python -c "from training_utils import ConvergenceDetector, freeze_module, load_training_config, update_training_status"` | ❌ W0 | ⬜ pending |
| 04-01-T2 | 01 | 1 | PIPE-01, PIPE-05, PIPE-06 | — | N/A | unit | `python -m pytest tests/test_training_pipeline.py -x -v` | ❌ W0 | ⬜ pending |
| 04-02-T1 | 02 | 2 | PIPE-02 | — | N/A | unit | `python -c "exec(open('main_model/train.py').read().split('if __name__')[0]); print('parse OK')"` | ❌ W0 | ⬜ pending |
| 04-02-T2 | 02 | 2 | PIPE-02 | — | N/A | unit | `python -m pytest tests/test_encoder_intuition_training.py -x -v` | ❌ W0 | ⬜ pending |
| 04-03-T1 | 03 | 2 | PIPE-03 | — | N/A | unit | `python -c "exec(open('reward_head/train.py').read().split('if __name__')[0]); print('parse OK')"` | ❌ W0 | ⬜ pending |
| 04-03-T2 | 03 | 2 | PIPE-03 | — | N/A | unit | `python -m pytest tests/test_reward_head_training.py -x -v` | ❌ W0 | ⬜ pending |
| 04-04-T1 | 04 | 2 | PIPE-04 | — | N/A | unit | `python -c "exec(open('action_planner/train.py').read().split('if __name__')[0]); print('parse OK')"` | ❌ W0 | ⬜ pending |
| 04-04-T2 | 04 | 2 | PIPE-04 | — | N/A | unit | `python -m pytest tests/test_action_planner_training.py -x -v` | ❌ W0 | ⬜ pending |
| 04-05-T1 | 05 | 3 | PIPE-07 | — | N/A | integration | `python coordinator.py status && python coordinator.py next` | ❌ W0 | ⬜ pending |
| 04-05-T2 | 05 | 3 | PIPE-07 | — | N/A | integration | `python -m pytest tests/test_coordinator.py -x -v` | ❌ W0 | ⬜ pending |
| 04-06-T1 | 06 | 2 | PIPE-01 | — | N/A | unit | `python -c "from capture_states import capture_state_pair; from capture_demos import capture_demo_frame"` | ❌ W0 | ⬜ pending |
| 04-06-T2 | 06 | 2 | PIPE-01 | — | N/A | unit | `python -m pytest tests/test_capture_scripts.py -x -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_training_pipeline.py` — shared infrastructure tests (PIPE-01, PIPE-05, PIPE-06)
- [ ] `tests/test_encoder_intuition_training.py` — encoder+intuition joint training tests (PIPE-02)
- [ ] `tests/test_reward_head_training.py` — reward head training tests (PIPE-03)
- [ ] `tests/test_action_planner_training.py` — action planner imitation learning tests (PIPE-04)
- [ ] `tests/test_coordinator.py` — coordinator CLI and pipeline orchestration tests (PIPE-07)
- [ ] `tests/test_capture_scripts.py` — data capture utility tests (PIPE-01)
- [ ] Test fixtures for mock encoder weights, mock state data, mock JSONL captures

*Existing test infrastructure from Phase 3 (tests/test_architecture.py) provides patterns to follow.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GTA live data capture | PIPE-01 | Requires running GTA V game | Start GTA, drive, verify JSONL files created in training_data/ |
| End-to-end training chain with real data | PIPE-07 | Requires live gameplay data | Run coordinator, verify it guides through all stages with real checkpoints |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-01
