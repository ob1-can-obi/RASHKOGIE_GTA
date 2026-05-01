---
phase: 2
slug: batch-training-checkpointing
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-30
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.3 |
| **Config file** | none — existing infrastructure |
| **Quick run command** | `.venv/bin/python -m pytest tests/ -x -q` |
| **Full suite command** | `.venv/bin/python -m pytest tests/ -x -v` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/python -m pytest tests/ -x -v`
- **After every plan wave:** Run `.venv/bin/python -m pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 02-01-T1 | 01 | 1 | BATCH-01, BATCH-02, BATCH-03, BATCH-04 | smoke | `.venv/bin/python -c "from trainer import TrainingState; ..."` | pending |
| 02-01-T2 | 01 | 1 | BATCH-01, BATCH-02, BATCH-03, BATCH-04 | unit | `.venv/bin/python -m pytest tests/test_batch_training.py -x -v` | pending |
| 02-02-T1 | 02 | 2 | BATCH-05, BATCH-06 | smoke | `.venv/bin/python -c "from trainer import TrainingState; ... save/load ..."` | pending |
| 02-02-T2 | 02 | 2 | BATCH-05, BATCH-06 | unit | `.venv/bin/python -m pytest tests/test_batch_training.py -k "checkpoint" -x -v` | pending |
| 02-03-T1 | 03 | 3 | BATCH-01 to BATCH-06 | regression | `.venv/bin/python -m pytest tests/test_training_correctness.py -x -v` | pending |
| 02-03-T2 | 03 | 3 | BATCH-01 to BATCH-06 | integration | `.venv/bin/python -m pytest tests/test_batch_training.py -k "train_step" -x -v` | pending |

*Status: pending*

---

## Wave 0 Requirements

- Existing test infrastructure covers phase requirements (conftest.py and test framework from Phase 1)

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 5s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
