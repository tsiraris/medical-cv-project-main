# ---- Variables --------------------------------------------------------------
GIT_SHA := $(shell git rev-parse --short HEAD)
COMPOSE := mlops/docker-compose.mlflow.yaml

# ---- Phony targets (not files) ---------------------------------------------
.PHONY: train app-run build docker-run push kind-apply k8s-apply hpa \
        mlflow-up mlflow-down dvc-repro gate ci-local open-mlflow

# ---- Your original targets (kept, with tiny polish) ------------------------
# Trains the simple local script (legacy). Kept for convenience.
train:
	python scripts/train_model.py

# Run the FastAPI app locally (without Docker/K8s)
app-run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

# Build local Docker image for the API
build:
	docker build -t medical-cv-serve:local --build-arg BUILD_REV=$(GIT_SHA) .

# Run the local Docker image (exposes 8000)
docker-run:
	docker run --rm -p 8000:8000 -e BUILD_REV=$(GIT_SHA) medical-cv-serve:local

# Apply Kubernetes manifests (namespace, deploy, service, hpa)
k8s-apply:
	kubectl apply -f k8s/k8s-namespace.yaml
	kubectl apply -f k8s/k8s-deployment.yaml
	kubectl apply -f k8s/k8s-service.yaml
	kubectl apply -f k8s/k8s-hpa.yaml

# Show current HPA status
hpa:
	kubectl get hpa -n ml

# ---- New targets (Sprint 1 workflow helpers) -------------------------------

# Bring up MLflow+Postgres+MinIO stack
mlflow-up:
	docker compose -f $(COMPOSE) up -d

# Take the MLflow stack down (remove volumes for a clean slate)
mlflow-down:
	docker compose -f $(COMPOSE) down -v

# Run the DVC pipeline end-to-end (preprocess -> train -> eval)
dvc-repro:
	dvc repro

# Local quality gate (fail if metrics regress)
gate:
	python ci/quality_gate.py --metrics metrics/val.json --min-acc 0.80 --min-f1 0.80

# A convenient combo: start stack, run pipeline, check gate
ci-local: mlflow-up dvc-repro gate

# Open MLflow UI in your browser (macOS 'open', WSL 'xdg-open', Windows fallback prints URL)
open-mlflow:
	@if command -v xdg-open >/dev/null 2>&1 ; then xdg-open http://localhost:5500 ; \
	elif command -v open >/dev/null 2>&1 ; then open http://localhost:5500 ; \
	else echo "Open http://localhost:5500" ; fi

