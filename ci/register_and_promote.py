# ci/register_and_promote.py
"""
Register a model version for a given MLflow run, always set a Candidate alias,
and conditionally promote to a target alias (e.g. Production).

Env:
  MLFLOW_TRACKING_URI=http://127.0.0.1:5000
  MLFLOW_REGISTERED_MODEL_NAME=cxr-demo

  RUN_ID=<run id>  # if not set, will try metrics/run_info.json
  METRICS_PATH=metrics/val.json

  # Aliases
  CANDIDATE_ALIAS=Candidate
  TARGET_ALIAS=Production

  # Promotion rules
  PROMOTE_F1_THRESHOLD=0.80        # minimum f1 required to promote
  TARGET_STAGE=Production          # optional: also set MLflow stage

  # Optional git / CI metadata (set by the workflow):
  GIT_COMMIT_SHA
  GIT_BRANCH
  GITHUB_PR_NUMBER
"""

import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


def _load_run_id() -> str:
    run_id = os.getenv("RUN_ID")
    if run_id:
        print(f"[registry] Using RUN_ID from env: {run_id}")
        return run_id

    info_path = Path("metrics/run_info.json")
    if info_path.exists():
        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
            run_id = data.get("run_id")
            if run_id:
                print(f"[registry] Using RUN_ID from {info_path}")
                return run_id
        except Exception as e:
            print(f"[registry] Warning: could not read {info_path}: {e}")

    raise SystemExit(
        "[registry] ERROR: RUN_ID not set and metrics/run_info.json not usable"
    )


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "cxr-demo")
    target_alias = os.getenv("TARGET_ALIAS", "Production")
    candidate_alias = os.getenv("CANDIDATE_ALIAS", "Candidate")
    target_stage = os.getenv("TARGET_STAGE")  # optional, may be empty

    metrics_path = Path(os.getenv("METRICS_PATH", "metrics/val.json"))
    try:
        f1_threshold = float(os.getenv("PROMOTE_F1_THRESHOLD", "0.0"))
    except ValueError:
        f1_threshold = 0.0

    run_id = _load_run_id()

    print(f"[registry] Tracking URI: {tracking_uri}")
    print(f"[registry] Model name: {model_name}")
    print(f"[registry] Target alias: {target_alias}")
    print(f"[registry] Candidate alias: {candidate_alias}")
    print(f"[registry] F1 threshold: {f1_threshold}")

    # Ensure registered model exists
    try:
        client.get_registered_model(model_name)
        print(f"[registry] Registered model '{model_name}' already exists")
    except MlflowException:
        client.create_registered_model(model_name)
        print(f"[registry] Created registered model '{model_name}'")

    # ------------------------------------------------------------------
    # Create a new model version from the run
    # ------------------------------------------------------------------
    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/model"
    print(f"[registry] Source: {source}")

    mv = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
        description=f"Auto-registered from run {run_id}",
    )
    print(f"[registry] Created model version v{mv.version} for '{model_name}'")

    # ------------------------------------------------------------------
    # Attach rich metadata as tags
    # ------------------------------------------------------------------
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
            f"[registry] Copied {len(params)} params and "
            f"{len(metrics)} metrics to tags"
        )
    except Exception as e:
        print(f"[registry] Warning: could not tag params/metrics: {e}")

    # Git / CI provenance
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

        client.set_model_version_tag(
            name=model_name,
            version=mv.version,
            key="mlflow.run_id",
            value=run_id,
        )
        print("[registry] Attached git/CI metadata tags")
    except Exception as e:
        print(f"[registry] Warning: could not tag git/CI metadata: {e}")

    # ------------------------------------------------------------------
    # Always set Candidate alias to this new version
    # ------------------------------------------------------------------
    if candidate_alias:
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=candidate_alias,
                version=mv.version,
            )
            print(
                f"[registry] Alias '{candidate_alias}' now points to "
                f"{model_name} v{mv.version}"
            )
        except Exception as e:
            print(f"[registry] Warning: could not set candidate alias: {e}")

    # ------------------------------------------------------------------
    # Decide whether to promote to target alias (Production)
    # ------------------------------------------------------------------
    new_f1 = None
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if m.get("f1") is not None:
                new_f1 = float(m["f1"])
            print(
                f"[registry] New model metrics: f1={new_f1} "
                f"(threshold={f1_threshold})"
            )
        except Exception as e:
            print(f"[registry] Warning: could not read metrics: {e}")
    else:
        print(f"[registry] Metrics file not found at {metrics_path}")

    current_f1 = None
    current_alias_version = None
    try:
        current_mv = client.get_model_version_by_alias(
            name=model_name,
            alias=target_alias,
        )
        current_alias_version = current_mv.version
        tag_f1 = (current_mv.tags or {}).get("metric.f1")
        if tag_f1 is not None:
            current_f1 = float(tag_f1)
        print(
            f"[registry] Current alias '{target_alias}' -> v{current_alias_version}, "
            f"metric.f1={current_f1}"
        )
    except MlflowException:
        print(
            f"[registry] No existing alias '{target_alias}' found; "
            "treating this as first candidate for that alias."
        )

    promote = True

    if new_f1 is not None and new_f1 < f1_threshold:
        print(
            "[registry] New model below F1 threshold; "
            "will NOT update production alias."
        )
        promote = False

    if (
        promote
        and current_f1 is not None
        and new_f1 is not None
        and new_f1 < current_f1
    ):
        print(
            "[registry] New model F1 is worse than current production "
            f"({new_f1} < {current_f1}); will NOT update alias."
        )
        promote = False

    if not promote:
        print(
            f"[registry] Keeping alias '{target_alias}' on "
            f"version {current_alias_version} (if it existed)."
        )
        return

    # ------------------------------------------------------------------
    # Promote: move Production alias (and optionally stage)
    # ------------------------------------------------------------------
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=target_alias,
            version=mv.version,
        )
        print(
            f"[registry] Alias '{target_alias}' now points to "
            f"{model_name} v{mv.version}"
        )
    except Exception as e:
        print(f"[registry] ERROR: could not set target alias: {e}")

    if target_stage:
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=target_stage,
                archive_existing_versions=True,
            )
            print(f"[registry] Transitioned to stage '{target_stage}'")
        except Exception as e2:
            print(f"[registry] Stage transition failed: {e2}")


if __name__ == "__main__":
    main()
