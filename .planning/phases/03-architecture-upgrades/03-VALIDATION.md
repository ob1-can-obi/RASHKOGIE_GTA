---
phase: 3
slug: architecture-upgrades
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-30
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.3 |
| **Config file** | none — uses default pytest discovery |
| **Quick run command** | `.venv/bin/python -m pytest tests/ -x -q` |
| **Full suite command** | `.venv/bin/python -m pytest tests/ -x -v` |
| **Estimated runtime** | ~2 seconds |

---

## Sampling Rate

- **After every task commit:** Run `.venv/bin/python -m pytest tests/ -x -q`
- **After every plan wave:** Run `.venv/bin/python -m pytest tests/ -x -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | ARCH-01 | — | N/A | unit | `.venv/bin/python -m pytest tests/test_architecture.py -k "meta_mlp" -x -v` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | ARCH-04 | — | N/A | unit | `.venv/bin/python -m pytest tests/test_architecture.py -k "input_dim" -x -v` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | ARCH-02 | — | N/A | unit | `.venv/bin/python -m pytest tests/test_architecture.py -k "encoder" -x -v` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 1 | ARCH-03 | — | N/A | unit | `.venv/bin/python -m pytest tests/test_architecture.py -k "planner" -x -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_architecture.py` — stubs for ARCH-01, ARCH-02, ARCH-03, ARCH-04

*Existing conftest.py and test infrastructure covers shared fixtures.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
