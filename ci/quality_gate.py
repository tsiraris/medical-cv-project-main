# ci/quality_gate.py
import json
import sys
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="metrics/val.json")
    p.add_argument("--min-acc", type=float, default=0.80)
    p.add_argument("--min-f1", type=float, default=0.80)
    args = p.parse_args()

    mpath = Path(args.metrics)
    if not mpath.exists():
        print(f"[gate] Missing metrics file: {mpath}", file=sys.stderr)
        sys.exit(2)

    with open(mpath, "r", encoding="utf-8") as f:
        m = json.load(f)

    acc = float(m.get("acc", -1.0))
    f1  = float(m.get("val_f1_macro", -1.0))

    ok = True
    if acc < args.min_acc:
        print(f"[gate] FAIL: acc {acc:.4f} < min {args.min_acc:.4f}", file=sys.stderr)
        ok = False
    if f1 < args.min_f1:
        print(f"[gate] FAIL: f1 {f1:.4f} < min {args.min_f1:.4f}", file=sys.stderr)
        ok = False

    if ok:
        print(f"[gate] PASS: acc={acc:.4f} f1={f1:.4f}")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
