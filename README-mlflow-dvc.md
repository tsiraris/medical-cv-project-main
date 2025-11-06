# Sprint 1: DVC + MLflow quickstart

## 0) Install dependencies
```bash
pip install -U pip
pip install -e .
pip install dvc[azure,s3,gdrive,gs] mlflow matplotlib pyyaml
```

## 1) Start MLflow locally (docker-compose)
```bash
cd mlops
docker compose -f docker-compose.mlflow.yaml up -d
# UI: MLflow http://localhost:5000
# MinIO console: http://localhost:9001 (minio/minio123)
```

## 2) Export env vars (PowerShell examples)
```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "minio"
$env:AWS_SECRET_ACCESS_KEY = "minio123"
$env:MLFLOW_EXPERIMENT_NAME = "demo-cv"
$env:MLFLOW_REGISTERED_MODEL_NAME = "cxr-demo"
```

## 3) Initialize DVC and run the pipeline
```bash
dvc init
dvc repro   # runs preprocess -> train -> eval
```

Artifacts you’ll see:
- `models/model.joblib` (DVC-tracked model)
- `metrics/val.json`
- `artifacts/roc.png`, `artifacts/confusion.png`
- MLflow UI will show a run with params/metrics + a **registered model version**

## 4) Track data with DVC (optional remote)
```bash
# via DVC
dvc remote add -d minio s3://mlflow
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minio
dvc remote modify minio secret_access_key minio123
```

## 5) Next
- Promote a model version in MLflow UI (Staging/Production)
- Wire CI to run `dvc repro` and gate on metrics (gates.yaml)
- In Sprint 2 we’ll switch serving to KServe and canary rollouts
