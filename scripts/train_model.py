"""
Creates models/model.joblib using Iris-like tabular data so the service can run
and CI can build images. This is a placeholder; we’ll replace with CV training later.
"""
import os, json
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.joblib"

X, y = load_iris(return_X_y=True)
X = X[:, :4]  # f1..f4 to match the API
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, ytr)
yhat = clf.predict(Xte)
acc = accuracy_score(yte, yhat)
joblib.dump(clf, MODEL_PATH)

print(json.dumps({"model_path": str(MODEL_PATH), "accuracy": acc}))
