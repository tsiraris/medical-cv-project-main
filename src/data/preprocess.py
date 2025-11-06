"""
Preprocess step:
- Uses Iris dataset (bundled in scikit-learn) to avoid external downloads in CI.
- Converts it to a binary task (class 0 vs class 1) for a simple logistic baseline.
- Saves numpy arrays under data/processed: Xtr.npy, Xte.npy, ytr.npy, yte.npy
"""
import argparse
from pathlib import Path
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="inp", default="data/raw")         # not used, kept for API stability
parser.add_argument("--out", dest="out", default="data/processed")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-size", type=float, default=0.2)

if __name__ == "__main__":
    args = parser.parse_args()

    # Load iris (no network). Take only classes 0 and 1 => binary.
    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target
    mask = y < 2
    X = X[mask]
    y = y[mask]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    out_dir = ensure_dir(args.out)
    np.save(out_dir / "Xtr.npy", Xtr)
    np.save(out_dir / "Xte.npy", Xte)
    np.save(out_dir / "ytr.npy", ytr)
    np.save(out_dir / "yte.npy", yte)

    print(
        {
            "out": str(out_dir),
            "Xtr": Xtr.shape,
            "Xte": Xte.shape,
            "ytr": ytr.shape,
            "yte": yte.shape,
        }
    )
    print("Preprocessing done.")