# 🩺 Medical-CV: An End-to-End MLOps / CI-CD Harness

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688)
![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2)
![DVC](https://img.shields.io/badge/DVC-Versioning-945DD6)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Canary-326CE5)
![CI](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF)

A self-built, **production-grade MLOps pipeline** that takes a model from versioned data all the way to a **monitored, autoscaling, canary-deployed REST service** — and automates the whole lifecycle in GitHub Actions. It is intentionally **model-agnostic**: the serving, versioning, promotion, and rollout machinery is the deliverable, and the model plugged into it is a swappable placeholder.

---

## ⚠️ What this project is — and what it is *not*

**It IS** a complete, working operationalization harness: data + pipeline versioning (DVC), experiment tracking and a Model Registry with staged promotion (MLflow), containerized serving with Prometheus observability (FastAPI), Kubernetes deployment with autoscaling and a canary rollout, and four CI/CD workflows including a nightly retrain-and-gate job.

**It does NOT claim to be:**
- **A medical-imaging model.** The model currently wired through the pipeline is a scikit-learn **`LogisticRegression` on four tabular features** (`f1–f4`, derived from the bundled Iris dataset). The registered name `cxr-demo` ("chest X-ray demo") is a **placeholder**. "Medical CV" describes the *target domain the harness is designed for*, not an imaging model it ships.
- **A cloud production deployment.** It runs on a **local single-node cluster** (Docker Desktop / Minikube Kubernetes), not a managed GKE/EKS cluster serving live traffic. The `deploy-inference.yml` workflow is wired for a real cluster (via a kubeconfig secret) but is a manual, not-yet-exercised-at-scale path.
- **A clinically validated system.** The `mlops/gates.yaml` clinical bars (AUC ≥ 0.90, sensitivity/specificity ≥ 0.85) are **design intent**, staged for when a real imaging model lands — they are not yet enforced.

Being upfront about the placeholder is deliberate: the value here is the **infrastructure**, which is genuinely solid.

---

## 🏗️ Architecture & Tech Stack

| Layer | Tooling | Role |
|-------|---------|------|
| **Data & pipeline versioning** | DVC | Reproducible `preprocess → train → eval` DAG; dataset/artifact versioning to MinIO/S3 |
| **Experiment tracking & registry** | MLflow (Postgres + MinIO) | Logs params/metrics/artifacts; Model Registry with `Candidate` / `Production` aliases |
| **Serving** | FastAPI + Prometheus | REST API (`/health`, `/model-info`, `/predict`, `/reload`, `/metrics`); per-alias latency & request metrics |
| **Orchestration** | Kubernetes + NGINX Ingress | HPA autoscaling (CPU 60%, 1–5 replicas); dual-deployment **10% canary** |
| **Automation** | GitHub Actions | CI, image build/push to GHCR, manual deploy, nightly retrain |
| **Local stack** | Docker Compose | One-command Postgres + MinIO + MLflow + app |

---

## 📁 Repository Structure

```text
.
├── app/                  # FastAPI serving + Prometheus metrics middleware
├── ci/                   # quality_gate.py (F1/acc gate) + register_and_promote.py (registry/aliases)
├── k8s/                  # namespace, deployment, service, HPA, ingress (+ canary variants)
├── mlops/                # docker-compose (MLflow/Postgres/MinIO) + gates.yaml (future clinical bars)
├── scripts/              # smoke test, load test, model-card generator, placeholder trainer
├── src/                  # ML pipeline: data/, training/, and utils/ (incl. wait_for_mlflow guard)
├── .github/workflows/    # ci.yml, cd-docker.yml, deploy-inference.yml, nightly.yml
├── dvc.yaml / params.yaml# pipeline graph + single source of truth for hyperparameters
└── Makefile              # local workflow shortcuts
```

---

## 🔄 The Lifecycle (how it actually behaves)

1. **Push to `main`** triggers `ci.yml`: it stands up a Postgres service + a local MLflow server, runs `dvc repro`, then registers a new model version, **always** sets the `Candidate` alias, and conditionally moves `Production`.
2. **Promotion is a no-regression rule.** In CI, `register_and_promote.py` runs with `PROMOTE_F1_THRESHOLD=0.0`, so it promotes to `Production` only if the new F1 is **not worse** than the current `Production` version — not an absolute bar. Provenance (Git SHA, branch, PR number, run id) is stamped onto every version as tags.
3. **Quality gate (`ci/quality_gate.py`, F1/acc ≥ 0.80)** is enforced **locally** via `make gate` / `make ci-local`, and is re-used by the **nightly** job. It is *not* a step inside `ci.yml`.
4. **`cd-docker.yml`** builds and pushes the image to `ghcr.io/<owner>/medical-cv-project` (`latest` + commit SHA).
5. **`deploy-inference.yml`** (manual) does a `kubectl set image` rollout on prod + canary and runs a smoke test.
6. **`nightly.yml`** (cron) retrains on a file backend, runs the quality gate, and uploads a model card + artifacts.

---

## 🚀 Quickstart

### Local pipeline + serving
```bash
# 1. Install
pip install -e .
pip install -r requirements.txt

# 2. Start the MLflow + Postgres + MinIO stack (Docker)
make mlflow-up           # MLflow UI at http://localhost:5500

# 3. Run the DVC pipeline (preprocess → train → eval)
make dvc-repro

# 4. Enforce the quality gate (F1/acc >= 0.80)
make gate

# 5. Serve locally
make app-run             # http://localhost:8000/docs
```

### Kubernetes (local cluster)
```bash
make k8s-apply           # namespace, deployment, service, HPA
kubectl -n ml port-forward svc/medical-cv-serve 8080:80

curl -s localhost:8080/health | jq
curl -s -X POST localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"items":[{"f1":5.1,"f2":3.5,"f3":1.4,"f4":0.2}]}' | jq
```

---

## 🐛 Engineering decisions & failures fixed

- **CI race condition** against a not-yet-ready MLflow server → added `psycopg2-binary`, a `curl` polling loop, *and* an in-code `wait_for_mlflow()` readiness guard.
- **Registry "ghost-model" bug** (artifacts logged but never registered) → switched to low-level `MlflowClient.create_registered_model` + `create_model_version`, avoiding tracking-server endpoints that aren't implemented.
- **Heredoc-in-YAML collapse** → refactored inline Python into committed, version-controlled scripts.
- **KServe → vanilla-K8s pivot** when the cluster lacked Knative/KServe CRDs → achieved canary rollout with native Deployments + NGINX-Ingress `canary-weight`.

---

## 🗺️ Roadmap

- **Swap in a real imaging model** (PyTorch/Torchvision) behind the same harness; enforce the `gates.yaml` clinical bars (AUC/sensitivity/specificity).
- **Self-hosted runner** so the nightly job can register/promote against the DB-backed MLflow (cloud runners can't reach a localhost server).
- **PR/Slack notifications** summarizing promotion status and metric deltas.

---

*Questions about the architecture are welcome — open an issue.*
