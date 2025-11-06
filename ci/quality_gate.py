# ci/quality_gate.py
import json
from pathlib import Path
import sys

# Thresholds (tune as you like)
THRESHOLDS = {
    "f1": 0.80,   # preferred
    "acc": 0.80,  # fallback when f1 is absent
}

mfile = Path("metrics/val.json")
if not mfile.exists():
    print("[gate] FAIL: metrics/val.json not found")
    sys.exit(1)

try:
    metrics = json.loads(mfile.read_text(encoding="utf-8"))
except Exception as e:
    print(f"[gate] FAIL: could not parse metrics/val.json: {e}")
    sys.exit(1)

metric_name = "f1" if "f1" in metrics else ("acc" if "acc" in metrics else None)
if metric_name is None:
    print("[gate] FAIL: neither 'f1' nor 'acc' present in metrics/val.json")
    print("[gate] metrics:", metrics)
    sys.exit(1)

value = float(metrics[metric_name])
thresh = THRESHOLDS[metric_name]

if value < thresh:
    print(f"[gate] FAIL: {metric_name} {value:.4f} < min {thresh:.4f}")
    sys.exit(1)

print(f"[gate] PASS: {metric_name} {value:.4f} >= {thresh:.4f}")
