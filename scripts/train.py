"""
scripts/train.py — Train MLP and GBT on IEEE-CIS data and save artifacts.

Usage:
    python scripts/train.py

Outputs:
    models/mlp.npz
    models/gbt.pkl
    models/background.npy
    models/drift_ref.json
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.ieee_adapter import IEEEAdapter
from ml.mlp import MLP
from ml.gbt import GradientBoostedTrees
from ml.ensemble import EnsembleScorer
from explainability.drift import PSIDriftDetector
from explainability.registry import ModelRegistry

os.makedirs("models", exist_ok=True)


# ---------------------------------------------------------------------------
# AUC helper (no sklearn)
# ---------------------------------------------------------------------------

def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order   = np.argsort(-y_score)
    y_true  = y_true[order]
    n_pos   = y_true.sum()
    n_neg   = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp  = np.cumsum(y_true)
    fp  = np.cumsum(1 - y_true)
    tpr = np.concatenate([[0], tp / n_pos])
    fpr = np.concatenate([[0], fp / n_neg])
    return float(np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# 1. Load data via iterate()
# ---------------------------------------------------------------------------

def _txn_to_row(txn) -> np.ndarray:
    """Flatten one NormalizedTransaction into a 1-D feature vector."""
    def _vals(f):
        if not f:
            return []
        vals = f.values() if isinstance(f, dict) else f
        result = []
        for v in vals:
            if v == 'T':       result.append(1.0)
            elif v == 'F':     result.append(0.0)
            elif v == 'unknown' or v is None: result.append(0.0)
            else:
                try:    result.append(float(v))
                except: result.append(0.0)
        return result

    parts = (
        [
            float(txn.timestamp_delta),
            float(txn.amount),
            float(txn.dist_from_home) if txn.dist_from_home is not None else 0.0,
        ]
        + _vals(txn.count_features)
        + _vals(txn.timedelta_features)
        + _vals(txn.match_features)
        + _vals(txn.vesta_features)
    )
    return np.array(parts, dtype=np.float32)


print("Loading IEEE-CIS data (this takes ~1-2 min)...")
t0 = time.time()

adapter = IEEEAdapter(data_dir="data/ieee_cis").load()
rows, labels = [], []

for batch in adapter.iterate(batch_size=1000):
    for txn in batch:
        rows.append(_txn_to_row(txn))
        labels.append(float(txn.is_fraud))

# Pad rows to uniform length
max_len = max(len(r) for r in rows)
X = np.zeros((len(rows), max_len), dtype=np.float32)
for i, r in enumerate(rows):
    X[i, :len(r)] = r
y = np.array(labels, dtype=np.float32)

# Replace NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Loaded {len(X):,} rows, {X.shape[1]} features in {time.time()-t0:.1f}s")
print(f"  Fraud rate: {y.mean():.3%}  ({int(y.sum()):,} fraud / {len(y):,} total)")

# ---------------------------------------------------------------------------
# 2. Train/val split (80/20, time-ordered)
# ---------------------------------------------------------------------------

split    = int(len(X) * 0.80)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")

# ---------------------------------------------------------------------------
# 3. Normalize
# ---------------------------------------------------------------------------

print("\nFitting normalizer...")
scorer = EnsembleScorer()
scorer.fit_normalizer(X_train)
X_train_n = scorer.normalize(X_train)
X_val_n   = scorer.normalize(X_val)

# ---------------------------------------------------------------------------
# 4. Train GBT
# ---------------------------------------------------------------------------

print("\nTraining GBT (100 trees)...")
t0 = time.time()
gbt = GradientBoostedTrees(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8)
gbt.fit(X_train_n, y_train)
print(f"  Done in {time.time()-t0:.1f}s")

gbt_val = gbt.predict_proba(X_val_n)
print(f"  Val AUC: {_auc(y_val, gbt_val):.4f}")

with open("models/gbt.pkl", "wb") as f:
    pickle.dump(gbt, f)
print("  Saved models/gbt.pkl")

# ---------------------------------------------------------------------------
# 5. Train MLP
# ---------------------------------------------------------------------------

print("\nTraining MLP (20 epochs)...")
t0 = time.time()
mlp = MLP(input_dim=X_train_n.shape[1], hidden_dims=[128, 64, 32],
          dropout_p=0.3, lr=0.001, focal_gamma=2.0, focal_alpha=0.25)
mlp.fit(X_train_n, y_train, epochs=20, batch_size=512)
print(f"  Done in {time.time()-t0:.1f}s")

mlp_val = mlp.predict_proba(X_val_n)
mlp_auc = _auc(y_val, mlp_val)
print(f"  Val AUC: {mlp_auc:.4f}")

mlp.save("models/mlp.npz")
print("  Saved models/mlp.npz")

# ---------------------------------------------------------------------------
# 6. Ensemble metrics
# ---------------------------------------------------------------------------

ens_val  = 0.4 * mlp_val + 0.6 * gbt_val
ens_auc  = _auc(y_val, ens_val)
preds_bin = (ens_val >= 0.5).astype(int)
tp = int(((preds_bin == 1) & (y_val == 1)).sum())
fp = int(((preds_bin == 1) & (y_val == 0)).sum())
fn = int(((preds_bin == 0) & (y_val == 1)).sum())
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)
print(f"\nEnsemble Val AUC: {ens_auc:.4f}")
print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

# ---------------------------------------------------------------------------
# 7. Save SHAP background + PSI reference
# ---------------------------------------------------------------------------

rng    = np.random.default_rng(42)
bg_idx = rng.choice(len(X_train_n), size=100, replace=False)
np.save("models/background.npy", X_train_n[bg_idx])
print("\nSaved models/background.npy")

feature_names = [f"f{i}" for i in range(X_train_n.shape[1])]
drift = PSIDriftDetector(feature_names, n_bins=10, window_size=500)
ref_scores = 0.4 * mlp.predict_proba(X_train_n[:5000]) + 0.6 * gbt.predict_proba(X_train_n[:5000])
drift.fit_reference(X_train_n[:5000], ref_scores)
drift.save_reference("models/drift_ref.json")
print("Saved models/drift_ref.json")

# ---------------------------------------------------------------------------
# 8. Register models
# ---------------------------------------------------------------------------

registry = ModelRegistry()
gbt_rec  = registry.register("gbt", "models/gbt.pkl", "ieee-cis",
                              hyperparams={"n_estimators": 100, "lr": 0.1, "max_depth": 4},
                              train_metrics={"val_auc": round(_auc(y_val, gbt_val), 4)})
mlp_rec  = registry.register("mlp", "models/mlp.npz", "ieee-cis",
                              hyperparams={"hidden_dims": [128, 64, 32], "epochs": 20},
                              train_metrics={"val_auc": round(mlp_auc, 4)})
registry.activate(gbt_rec.version)
registry.activate(mlp_rec.version)
print(f"\nRegistered: {gbt_rec.version}  {mlp_rec.version}")
print("\nDone. Run: python benchmarks/run.py")