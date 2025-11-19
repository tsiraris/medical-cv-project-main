# 🧭 MLOps Project Handover — Sprint 1 Summary & Sprint 2 Setup

**Project:** medical-cv-project  
**Pipeline:** MLflow + DVC + GitHub Actions  
**Status:** End of Sprint 1 — Stable foundation achieved  
**Next:** Sprint 2 — Evaluation, promotion logic, and backend integration  

---

## 🧩 Context Summary

This repository implements a complete **MLOps workflow** for a computer vision project using:

- **MLflow** for experiment tracking and model registry  
- **DVC** for data and model versioning  
- **GitHub Actions** for automated CI/CD  

Throughout Sprint 1, we debugged, refactored, and stabilized the workflow file  
`.github/workflows/ci.yml`, ensuring it runs **both locally and on GitHub Actions**.

---

## ✅ What We Achieved in Sprint 1

### 1. CI Workflow Stabilization
- Rebuilt `.github/workflows/ci.yml` from scratch.
- Fixed all YAML indentation and structure errors.
- Replaced **heredocs** (`<<'PY'`) with **safe inline Python scripts** via `printf` + `.py` files.
- Ensured `set -euo pipefail` is used in all bash runs.
- Verified consistent **left-aligned** YAML under `run:` blocks.

### 2. MLflow Local Server Reliability
- Added `psycopg2` sanity check before start.
- Configured `nohup mlflow server` on `127.0.0.1:5000`.
- Implemented retry loop to wait for readiness using both `curl` and `MlflowClient`.

### 3. Model Registration & Alias Automation
- Built a stable step for model registration and alias promotion.
- No heredocs, no inline try/except — pure clean Python file executed inside CI.
- Verified end-to-end model registration flow on main branch.

### 4. Pipeline Integration
- Confirmed **DVC**, **MLflow**, and **GitHub Actions** integrate correctly.
- Extracted `RUN_ID` dynamically from `metrics/run_info.json`.
- Reused environment variables via `$GITHUB_ENV` for downstream steps.

---

## ❌ What Failed and How We Fixed It

| Issue | Root Cause | Resolution |
|--------|-------------|------------|
| YAML “unexpected EOF” / “wanted `PY`” | Heredoc blocks broke YAML structure | Removed heredocs completely |
| MLflow `ConnectionRefusedError` | Server not ready + missing `psycopg2` | Added startup check and retry loop |
| “Invalid workflow file” | Bad indentation or inline syntax | Unified left-aligned `run:` syntax |
| “invalid syntax” in Python | Over-compressed `try/except` one-liners | Rewrote proper multi-line Python |
| `MLFLOW_TRACKING_URI` missing | Not persisted to environment | Exported to `$GITHUB_ENV` correctly |

---

## 📋 Ground Rules (Agreed Practices)

1. **🚫 No heredocs** in YAML (never use `<<'PY'` again).
2. **✅ Use `printf` to create .py scripts**, then execute them.
3. **✅ Keep all `run:` commands left-aligned**.
4. **✅ Always start shell blocks with `set -euo pipefail`**.
5. **✅ Propagate env vars only via `$GITHUB_ENV` or explicit `env:`.**
6. **✅ No inline `try/except` — use proper Python structure.**
7. **✅ Keep MLflow local at `127.0.0.1:5000` in CI.**

---

## 🧱 Current Project State

| Component | Status | Notes |
|------------|--------|-------|
| MLflow server start | ✅ Working | Uses `nohup` + wait loop |
| Psycopg2 check | ✅ Working | Ensures DB driver present |
| MLflow connection | ✅ Working | Waits for REST readiness |
| DVC pipeline | ✅ Working | `dvc repro --force` successful |
| Run ID extraction | ✅ Working | Reads from JSON |
| Model registration | ✅ Working | Creates model if missing |
| Model aliasing | ✅ Working | Final “no heredoc” version |
| YAML validation | ✅ Passed | Verified in CI |
| MLflow backend (remote DB) | ⏳ Pending | Sprint 2 |
| Model evaluation logic | ⏳ Pending | Sprint 2 |
| Notifications | ⏳ Pending | Sprint 2 |

---

## 🚧 Sprint 2 — Objectives

1. **Add Experiment Metadata and Metrics Storage**
   - Extend `register_and_promote.py` to log params, metrics, and tags.
   - Link model version to Git commit SHA and PR number.

2. **Model Evaluation & Automatic Promotion**
   - Compare current metrics to previous version.
   - Promote to `production` alias if improved.

3. **Remote MLflow Backend**
   - Switch from local SQLite to PostgreSQL.
   - Store credentials via GitHub Secrets.

4. **Artifact Management**
   - Push artifacts to remote DVC/MLflow storage (S3, Azure, or similar).

5. **Notifications**
   - Slack or PR comment when model promoted.

6. **Refactor Utility Scripts**
   - Move inline Python into `/ci/` folder as reusable modules.

7. **Create Documentation**
   - Add `README_CI.md` describing CI flow, env vars, and troubleshooting.

---

## 🧭 Handover Instructions

When starting Sprint 2 or opening this project in a new ChatGPT session:

> 🪄 Paste this entire message and attach the **latest repository ZIP**.  
> Then say:  
> **“Continue from this exact project state and proceed with Sprint 2 goals.”**

The assistant should:
1. Reconstruct context (no prior chat needed).  
2. Inspect the repo structure and workflow.  
3. Continue building from here: metadata, evaluation, promotion, and backend steps.

---

## 📌 Notes

- `.github/workflows/ci.yml` is now **fully valid YAML**.
- Every Python execution block uses `printf` to generate files instead of heredocs.
- The model registration step tested successfully on GitHub Actions main branch.
- The project is stable, reproducible, and ready for next-phase automation.

---

**Prepared by:** ChatGPT (Sprint 1 Engineering Assistant)  
**For:** Next AI Engineer / Sprint 2 Automation Phase  
**Date:** _Latest successful pipeline run date_
