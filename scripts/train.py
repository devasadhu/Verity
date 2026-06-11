"""
scripts/train.py — Train MLP and GBT on IEEE-CIS data and save artifacts.

Speed fix: histogram-based binning (LightGBM-style) wraps the existing GBT.
Features are bucketed into 256 bins before training, reducing split candidates
from O(n) to O(256) per feature — makes 100-tree GBT tractable on 400K rows.

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
# Histogram binning — LightGBM-style speed fix
# ---------------------------------------------------------------------------

def bin_features(X: np.ndarray, n_bins: int = 256, bin_edges: list = None) -> tuple:
    """
    Bucket each feature into n_bins equal-frequency bins.
    Reduces GBT split search from O(n) to O(n_bins) per feature.
    Pass bin_edges from training to apply same bucketing to val/test.
    """
    n, d = X.shape
    X_binned  = np.zeros((n, d), dtype=np.float32)
    out_edges = []
    for j in range(d):
        if bin_edges is not None:
            edges = bin_edges[j]
        else:
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.unique(np.percentile(X[:, j], percentiles))
        binned = np.digitize(X[:, j], edges[1:]).astype(np.float32)
        X_binned[:, j] = binned
        out_edges.append(edges)
    return X_binned, out_edges


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def _txn_to_row(txn) -> np.ndarray:
    def _vals(f):
        if not f:
            return []
        vals = f.values() if isinstance(f, dict) else f
        result = []
        for v in vals:
            if v == 'T':                          result.append(1.0)
            elif v == 'F':                        result.append(0.0)
            elif v == 'unknown' or v is None:     result.append(0.0)
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

max_len = max(len(r) for r in rows)
X = np.zeros((len(rows), max_len), dtype=np.float32)
for i, r in enumerate(rows):
    X[i, :len(r)] = r
y = np.array(labels, dtype=np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Loaded {len(X):,} rows, {X.shape[1]} features in {time.time()-t0:.1f}s")
print(f"  Fraud rate: {y.mean():.3%}  ({int(y.sum()):,} fraud / {len(y):,} total)")

# ---------------------------------------------------------------------------
# 2. Train/val split
# ---------------------------------------------------------------------------

split = int(len(X) * 0.80)
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
# 4. Histogram binning — fit on train, apply to val
# ---------------------------------------------------------------------------

print("Binning features (256 bins) for fast GBT...")
t0 = time.time()
X_train_b, bin_edges = bin_features(X_train_n, n_bins=256)
X_val_b,   _         = bin_features(X_val_n,   bin_edges=bin_edges)
print(f"  Done in {time.time()-t0:.1f}s")

# Stratified subsample: keep all fraud, subsample legit at 20:1
rng_sub   = np.random.default_rng(42)
fraud_idx = np.where(y_train == 1)[0]
legit_idx = np.where(y_train == 0)[0]
n_legit   = min(len(legit_idx), len(fraud_idx) * 20)
sub_idx   = np.concatenate([
    fraud_idx,
    rng_sub.choice(legit_idx, n_legit, replace=False),
])
rng_sub.shuffle(sub_idx)
X_gbt = X_train_b[sub_idx]
y_gbt = y_train[sub_idx]
print(f"  GBT subset: {len(X_gbt):,} rows  fraud={y_gbt.mean():.1%}")

# ---------------------------------------------------------------------------
# 5. Train GBT
# ---------------------------------------------------------------------------

print("\nTraining GBT (100 trees)...")
t0 = time.time()
gbt = GradientBoostedTrees(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
)
gbt.fit(X_gbt, y_gbt)
print(f"  Done in {time.time()-t0:.1f}s")

gbt_val = gbt.predict_proba(X_val_b)
gbt_auc = _auc(y_val, gbt_val)
print(f"  Val AUC: {gbt_auc:.4f}")

with open("models/gbt.pkl", "wb") as f:
    pickle.dump(gbt, f)
with open("models/gbt_bin_edges.pkl", "wb") as f:
    pickle.dump(bin_edges, f)
print("  Saved models/gbt.pkl + models/gbt_bin_edges.pkl")

# ---------------------------------------------------------------------------
# 6. Train MLP — full dataset, no binning
# ---------------------------------------------------------------------------

print("\nTraining MLP (20 epochs)...")
t0 = time.time()
mlp = MLP(
    input_dim=X_train_n.shape[1],
    hidden_dims=[128, 64, 32],
    dropout_p=0.3,
    lr=0.001,
    focal_gamma=2.0,
    focal_alpha=0.25,
)
mlp.fit(X_train_n, y_train, epochs=20, batch_size=512)
print(f"  Done in {time.time()-t0:.1f}s")

mlp_val = mlp.predict_proba(X_val_n)
mlp_auc = _auc(y_val, mlp_val)
print(f"  Val AUC: {mlp_auc:.4f}")

mlp.save("models/mlp.npz")
print("  Saved models/mlp.npz")

# ---------------------------------------------------------------------------
# 7. Ensemble metrics
# ---------------------------------------------------------------------------

ens_val   = 0.4 * mlp_val + 0.6 * gbt_val
ens_auc   = _auc(y_val, ens_val)
preds_bin = (ens_val >= 0.5).astype(int)
tp = int(((preds_bin == 1) & (y_val == 1)).sum())
fp = int(((preds_bin == 1) & (y_val == 0)).sum())
fn = int(((preds_bin == 0) & (y_val == 1)).sum())
precision = tp / (tp + fp + 1e-8)
recall    = tp / (tp + fn + 1e-8)
f1        = 2 * precision * recall / (precision + recall + 1e-8)

print(f"\n{'='*40}")
print(f"Ensemble Val AUC:  {ens_auc:.4f}")
print(f"  GBT AUC:         {gbt_auc:.4f}")
print(f"  MLP AUC:         {mlp_auc:.4f}")
print(f"  Precision:       {precision:.4f}")
print(f"  Recall:          {recall:.4f}")
print(f"  F1:              {f1:.4f}")
print(f"{'='*40}")

# ---------------------------------------------------------------------------
# 8. SHAP background + PSI reference
# ---------------------------------------------------------------------------

bg_idx = rng_sub.choice(len(X_train_n), size=100, replace=False)
np.save("models/background.npy", X_train_n[bg_idx])
print("\nSaved models/background.npy")

feature_names = [f"f{i}" for i in range(X_train_n.shape[1])]
drift   = PSIDriftDetector(feature_names, n_bins=10, window_size=500)
ref_n   = min(5000, len(X_train_n))
ref_preds = (0.4 * mlp.predict_proba(X_train_n[:ref_n])
           + 0.6 * gbt.predict_proba(X_train_b[:ref_n]))
drift.fit_reference(X_train_n[:ref_n], ref_preds)
drift.save_reference("models/drift_ref.json")
print("Saved models/drift_ref.json")

# ---------------------------------------------------------------------------
# 9. Register
# ---------------------------------------------------------------------------

registry = ModelRegistry()
gbt_rec  = registry.register(
    "gbt", "models/gbt.pkl", "ieee-cis",
    hyperparams={"n_estimators": 100, "lr": 0.1, "max_depth": 4, "bins": 256},
    train_metrics={"val_auc": round(gbt_auc, 4)},
)
mlp_rec  = registry.register(
    "mlp", "models/mlp.npz", "ieee-cis",
    hyperparams={"hidden_dims": [128, 64, 32], "epochs": 20},
    train_metrics={"val_auc": round(mlp_auc, 4)},
)
registry.activate(gbt_rec.version)
registry.activate(mlp_rec.version)
print(f"\nRegistered: {gbt_rec.version}  {mlp_rec.version}")
print("\nAll done. Run: python benchmarks/run.py")