# 🩺 Medical Computer Vision: End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688)
![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2)
![DVC](https://img.shields.io/badge/DVC-Versioning-945DD6)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployment-326CE5)

## 📌 Project Overview

This repository implements a complete, production-grade **MLOps workflow** designed for a medical computer vision use case. While the current iteration utilizes a placeholder dataset to validate the CI/CD pathways, the underlying architecture provides a robust, scalable foundation for continuous training, automated evaluation, and safe deployment of machine learning models.

The goal of this project is to demonstrate best practices in **reproducibility, automated quality gating, and Kubernetes-based serving**.

## 🏗️ Architecture & Tech Stack

This project maps the entire ML lifecycle using industry-standard tooling:

*   **Data & Pipeline Versioning (DVC):** Tracks data provenance and orchestrates a reproducible DAG (`preprocess` → `train` → `eval`).
*   **Experiment Tracking & Registry (MLflow):** Logs hyperparameters, performance metrics (Accuracy, F1), and artifacts. Manages model lifecycle states (e.g., automatically tagging models as `Candidate` or `Production`).
*   **Model Serving (FastAPI):** Exposes a high-performance REST API for batch predictions, instrumented with **Prometheus** to track HTTP and inference latency.
*   **Continuous Integration & Deployment (GitHub Actions):** 
    *   Automated model retraining pipelines with strict quality gates (`f1 >= 0.80`).
    *   Docker image builds pushed to GitHub Container Registry (GHCR).
    *   Nightly cron jobs for automated retraining and model card generation.
*   **Orchestration (Kubernetes):** Production deployment manifests including Horizontal Pod Autoscalers (HPA) and NGINX Ingress rules for **Canary Deployments** (10% traffic routing).

## 📁 Repository Structure

```text
.
├── app/                  # FastAPI serving logic and Prometheus metrics middleware
├── ci/                   # CI scripts (quality gating, MLflow registry promotion)
├── k8s/                  # Kubernetes manifests (Deployments, Services, HPA, Ingress)
├── mlops/                # Docker Compose files for local MLflow/MinIO tracking stack
├── scripts/              # Utility scripts (load testing, smoke tests, model cards)
├── src/                  # Core ML pipeline (data preprocessing, training, evaluation)
├── dvc.yaml              # DVC pipeline definition
├── .github/workflows/    # CI/CD and automation pipelines
└── Makefile              # Command shortcuts for local development
```

## 🚀 Quickstart

### 1. Local Development (Pipeline Simulation)

You can run the entire ML pipeline and serving infrastructure locally.

```bash
# 1. Install dependencies
pip install -e .
pip install -r requirements.txt

# 2. Start the local MLflow tracking server (requires Docker)
make mlflow-up

# 3. Run the DVC pipeline (Preprocess -> Train -> Evaluate)
make dvc-repro

# 4. Check the quality gate (simulates CI pipeline checks)
make gate

# 5. Run the FastAPI server locally
make app-run
```

### 2. Kubernetes Deployment (Minikube / Cluster)

The application is designed to be deployed to a Kubernetes `ml` namespace with Canary support.
```bash
# Apply the namespace, deployments, services, and HPA
make k8s-apply

# Port forward to test locally
kubectl -n ml port-forward svc/medical-cv-serve 8080:80

# Check health and predict
curl -s localhost:8080/health | jq
curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{"items":[{"f1":5.1,"f2":3.5,"f3":1.4,"f4":0.2}]}' | jq
```

## 🧠 Continuous Integration (CI) Logic

A key feature of this repository is the automated intelligence built into the `.github/workflows/`. When code or data changes are pushed:

1.  **DVC Repro:** The pipeline is executed, caching intermediate steps if unchanged.
2.  **Quality Gate (`ci/quality_gate.py`):** The CI runner parses `metrics/val.json`. If the model falls below the minimum F1/Accuracy thresholds, the build fails.
3.  **Registration (`ci/register_and_promote.py`):** If the gate passes, the model is registered in MLflow. The script compares the new `Candidate` against the current `Production` model. If it performs better, it automatically promotes the alias, triggering the CD pipeline to update K8s.

## 🗺️ Roadmap (Sprint 2)

*   **Remote Backend Integration:** Migrate MLflow backend from local SQLite to PostgreSQL and remote S3 storage.
*   **Computer Vision Integration:** Replace the placeholder tabular dataset with actual medical imaging data (PyTorch/Torchvision integration).
*   **Notifications:** Implement Slack/PR automated comments detailing model promotion status and metrics.

---
*Feel free to open an issue or reach out if you have questions about the architecture!*
```