# scripts/model_card.py
"""
Generate a simple model card markdown file for the current run / version.

Env it can use:
  MLFLOW_TRACKING_URI
  MLFLOW_REGISTERED_MODEL_NAME  (e.g. "cxr-demo")
  RUN_ID                        (optional; otherwise reads metrics/run_info.json)
  MODEL_ALIAS                   (optional; e.g. "Production" or "Candidate")
"""

import json
import os
from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient


def _load_run_id() -> str:
    run_id = os.getenv("RUN_ID")
    if run_id:
        return run_id

    # Fallback: metrics/run_info.json (pattern used in the CI pipeline)
    info_path = Path("metrics/run_info.json")
    if info_path.exists():
        data = json.loads(info_path.read_text())
        return data["run_id"]

    raise RuntimeError("RUN_ID not set and metrics/run_info.json not found")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "cxr-demo")
    alias = os.getenv("MODEL_ALIAS")  # optional

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    run_id = _load_run_id()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params

    # Resolve model version
    version = None
    if alias:
        mv = client.get_model_version_by_alias(model_name, alias)
        version = mv.version
    else:
        # Pick latest version with this run_id, if any
        for mv in client.search_model_versions(f"name = '{model_name}'"):
            if mv.run_id == run_id:
                version = mv.version
                break

    if version is None:
        print("[card] WARNING: could not resolve model version; using 'unknown'")
        version = "unknown"

    # Load eval metrics from metrics/val.json if present
    val_metrics = {}
    mpath = Path("metrics/val.json")
    if mpath.exists():
        val_metrics = json.loads(mpath.read_text())

    # Prepare output dir
    out_dir = Path("model_cards")
    out_dir.mkdir(exist_ok=True)

    fname = out_dir / f"{model_name}_v{version}.md"
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    lines = []
    lines.append(f"# Model Card – {model_name} v{version}")
    lines.append("")
    lines.append(f"- Generated at: **{now}**")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Tracking URI: `{tracking_uri}`")
    lines.append(f"- Alias (if any): `{alias or ''}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    if val_metrics:
        for k, v in val_metrics.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("_No metrics/val.json found_")

    lines.append("")
    lines.append("## Run parameters")
    lines.append("")
    if params:
        for k, v in params.items():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("_No params logged_")

    fname.write_text("\n".join(lines), encoding="utf-8")
    print(f"[card] Wrote model card to {fname}")


if __name__ == "__main__":
    main()
