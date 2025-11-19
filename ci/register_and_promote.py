"""
Register a model version for a given MLflow run and set an alias (e.g., Production).
Env:
  MLFLOW_TRACKING_URI=http://localhost:5500
  MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
  AWS_ACCESS_KEY_ID=minio
  AWS_SECRET_ACCESS_KEY=minio123
  AWS_DEFAULT_REGION=us-east-1
  MLFLOW_REGISTERED_MODEL_NAME=cxr-demo
  RUN_ID=<run id>  # if not set, will try metrics/run_info.json
  METRICS_PATH=metrics/val.json
  TARGET_ALIAS=Production
  PROMOTE_F1_THRESHOLD=0.80
  # Optional fallback:
  TARGET_STAGE=Production
"""

import os
import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "cxr-demo")
    target_alias = os.getenv("TARGET_ALIAS", "Production")
    target_stage = os.getenv("TARGET_STAGE")  # fallback path
    metrics_path = Path(os.getenv("METRICS_PATH", "metrics/val.json"))
    f1_threshold = float(os.getenv("PROMOTE_F1_THRESHOLD", "0.0"))

    run_id = os.getenv("RUN_ID")
    if not run_id:
        # fallback to metrics/run_info.json
        run_info_path = Path("metrics/run_info.json")
        if run_info_path.exists():
            with open(run_info_path, "r", encoding="utf-8") as f:
                ri = json.load(f)
            run_id = ri.get("run_id")
    if not run_id:
        print("RUN_ID not set and metrics/run_info.json missing—nothing to register.")
        return

    print(f"[registry] Using tracking URI: {tracking_uri}")
    print(f"[registry] Model name: {model_name}")
    print(f"[registry] Run id: {run_id}")

    # Make sure the registered model exists
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)
        print(f"[registry] Created registered model '{model_name}'")

    # Resolve source from the run's artifact URI (robust, no hardcoding)
    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/model"
    print(f"[registry] Source: {source}")

    # Create model version
    mv = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
        description=f"Auto-registered from run {run_id}",
    )
    print(f"[registry] Created model version v{mv.version} for '{model_name}'")

    # ------------------------------------------------------------------
    # Attach rich metadata to the model version (params, metrics, git)
    # ------------------------------------------------------------------
    # Copy params and metrics from the run so they are visible on the registered model version as tags.
    try:
        params = run.data.params or {}
        metrics = run.data.metrics or {}
        for key, value in params.items():
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key=f"param.{key}",
                value=str(value),
            )
        for key, value in metrics.items():
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key=f"metric.{key}",
                value=str(value),
            )
        print(
            f"[registry] Copied {len(params)} params and {len(metrics)} metrics to model version tags"
        )
    except Exception as e:
        print(
            f"[registry] Warning: could not copy run params/metrics to model tags: {e}"
        )

    # Attach Git / CI provenance if available
    commit_sha = os.getenv("GIT_COMMIT_SHA") or os.getenv("GITHUB_SHA")
    branch = (
        os.getenv("GIT_BRANCH")
        or os.getenv("GITHUB_REF_NAME")
        or os.getenv("GITHUB_REF")
    )
    pr_number = os.getenv("GITHUB_PR_NUMBER")

    try:
        if commit_sha:
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key="git.commit_sha",
                value=commit_sha,
            )
        if branch:
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key="git.branch",
                value=branch,
            )
        if pr_number:
            client.set_model_version_tag(
                name=model_name,
                version=mv.version,
                key="git.pr_number",
                value=str(pr_number),
            )

        # Always store the originating run id as a tag too
        client.set_model_version_tag(
            name=model_name,
            version=mv.version,
            key="mlflow.run_id",
            value=run_id,
        )
        print("[registry] Attached git / CI metadata tags to model version")
    except Exception as e:
        print(f"[registry] Warning: could not attach git/CI metadata: {e}")

    # Gate by F1 (if present)
    promote = True
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            f1 = float(m.get("f1")) if m.get("f1") is not None else None
            print(f"[registry] Metrics: f1={f1} (threshold={f1_threshold})")
            if f1 is not None and f1 < f1_threshold:
                print(f"[registry] Below threshold; not updating alias.")
                promote = False
        except Exception as e:
            print(f"[registry] Warning: could not read metrics: {e}")

    if not promote:
        return

    # Preferred: assign alias
    try:
        client.set_registered_model_alias(
            name=model_name, alias=target_alias, version=mv.version
        )
        print(f"[registry] Alias '{target_alias}' -> '{model_name}' v{mv.version}")
    except Exception as e:
        print(f"[registry] Alias not supported/failed: {e}")
        # Fallback to stage transition if requested
        if target_stage:
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage=target_stage,
                    archive_existing_versions=True,
                )
                print(f"[registry] (fallback) Transitioned to stage '{target_stage}'")
            except Exception as e2:
                print(f"[registry] Stage transition failed: {e2}")


if __name__ == "__main__":
    main()
