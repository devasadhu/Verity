"""
Ensemble scorer and feature store.

Feature store pattern (Uber Michelangelo, Feast):
    Separate feature COMPUTATION (stream processing layer) from
    feature SERVING (this module). The feature store:
        1. Caches precomputed features per entity
        2. Serves features with low latency to the ML scoring path
        3. Handles missing features gracefully (cold start)

Ensemble:
    Combines MLP and GBT scores using a simple weighted average.
    Weight calibration: GBT typically outperforms MLP on tabular fraud data.
    Default: 0.4 * MLP + 0.6 * GBT — can be tuned on validation set.

    Why ensemble?
        MLP and GBT make different types of errors:
        - MLP: better at capturing feature interactions, weaker on sparse data
        - GBT: better on tabular data with clear decision boundaries
        Ensembling reduces variance by averaging uncorrelated errors.
"""

import numpy as np
import json
import time
import threading
from typing import Optional
from ml.mlp import MLP
from ml.gbt import GradientBoostedTrees
from storage.hashmap import HashMap


# Feature names in the order the models expect them
FEATURE_NAMES = [
    "user_txn_count_1m",
    "user_txn_count_5m",
    "user_txn_count_1h",
    "user_txn_count_24h",
    "user_amount_mean_1h",
    "user_amount_std_1h",
    "user_amount_max_24h",
    "amount_zscore_1h",
    "user_unique_cities_1h",
    "user_unique_devices_24h",
    "device_txn_count_1h",
    "device_txn_count_24h",
    "merchant_txn_count_1h",
    "merchant_amount_mean_1h",
    "user_global_freq",
    "device_global_freq",
    "merchant_global_freq",
    # Transaction-level features (added at score time)
    "amount",
    "is_new_device",
    "device_age_hours",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

INPUT_DIM = len(FEATURE_NAMES)


class FeatureStore:
    """
    In-memory feature cache backed by the C/Python hash map.
    Stores precomputed window features per entity.

    In production, this would be backed by Redis or a purpose-built
    feature store (Feast, Tecton, Hopsworks). Here we use our custom
    hash map to demonstrate the concept.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._cache = HashMap()
        self._ttl   = ttl_seconds
        self._lock  = threading.Lock()

    def store(self, entity_id: str, features: dict):
        """Cache features for an entity with TTL."""
        record = {"features": features, "ts": time.time()}
        with self._lock:
            self._cache.set(entity_id, json.dumps(record))

    def retrieve(self, entity_id: str) -> Optional[dict]:
        """Get cached features. Returns None if missing or expired."""
        with self._lock:
            raw = self._cache.get(entity_id)
        if raw is None:
            return None
        record = json.loads(raw)
        if time.time() - record["ts"] > self._ttl:
            return None
        return record["features"]

    def invalidate(self, entity_id: str):
        with self._lock:
            self._cache.delete(entity_id)


def extract_transaction_features(txn: dict) -> dict:
    """
    Extract features from the raw transaction dict itself
    (not the window-based features — those come from SlidingWindowAggregator).
    """
    from datetime import datetime

    ts_str = txn.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts_str)
        hour       = dt.hour
        dow        = dt.weekday()
        is_weekend = int(dow >= 5)
    except (ValueError, TypeError):
        hour = 12
        dow  = 0
        is_weekend = 0

    return {
        "amount":         float(txn.get("amount", 0)),
        "is_new_device":  float(txn.get("is_new_device", False)),
        "device_age_hours": float(txn.get("device_age_hours", 9999)),
        "hour_of_day":    float(hour),
        "day_of_week":    float(dow),
        "is_weekend":     float(is_weekend),
    }


def features_to_vector(window_features: dict, txn_features: dict) -> np.ndarray:
    """
    Combine window features and transaction features into a fixed-length
    numpy vector in the order expected by the models.
    """
    all_features = {**window_features, **txn_features}
    return np.array([
        float(all_features.get(name, 0.0))
        for name in FEATURE_NAMES
    ], dtype=np.float32)


class EnsembleScorer:
    """
    Combines MLP and GBT into a single risk score.

    Score = mlp_weight * P(fraud|MLP) + gbt_weight * P(fraud|GBT)

    Also handles:
        - Feature normalization (z-score using training set stats)
        - Missing model fallback (if one model fails, use the other)
        - Score calibration (Platt scaling placeholder)
    """

    def __init__(
        self,
        mlp: Optional[MLP] = None,
        gbt: Optional[GradientBoostedTrees] = None,
        mlp_weight: float = 0.4,
        gbt_weight: float = 0.6,
    ):
        self.mlp        = mlp
        self.gbt        = gbt
        self.mlp_weight = mlp_weight
        self.gbt_weight = gbt_weight
        # Feature normalization stats (set after training)
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std:  Optional[np.ndarray] = None

    def fit_normalizer(self, X: np.ndarray):
        """Compute feature mean and std from training data."""
        self._feature_mean = X.mean(axis=0)
        self._feature_std  = X.std(axis=0) + 1e-8  # avoid division by zero

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self._feature_mean is None:
            return x
        return (x - self._feature_mean) / self._feature_std

    def score(self, feature_vector: np.ndarray) -> dict:
        """
        Score a single transaction.
        Returns dict with individual model scores and ensemble score.
        """
        x = feature_vector.reshape(1, -1)
        x_norm = self.normalize(x)

        scores = {}
        available = []

        if self.mlp is not None:
            try:
                mlp_prob = float(self.mlp.predict_proba(x_norm)[0])
                scores["mlp_score"] = round(mlp_prob, 4)
                available.append(("mlp", mlp_prob, self.mlp_weight))
            except Exception as e:
                scores["mlp_error"] = str(e)

        if self.gbt is not None:
            try:
                gbt_prob = float(self.gbt.predict_proba(x)[0])  # GBT doesn't need normalization
                scores["gbt_score"] = round(gbt_prob, 4)
                available.append(("gbt", gbt_prob, self.gbt_weight))
            except Exception as e:
                scores["gbt_error"] = str(e)

        if not available:
            scores["ensemble_score"] = 0.5  # no models available
        elif len(available) == 1:
            scores["ensemble_score"] = round(available[0][1], 4)
        else:
            total_weight = sum(w for _, _, w in available)
            ensemble = sum(p * w for _, p, w in available) / total_weight
            scores["ensemble_score"] = round(ensemble, 4)

        return scores

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score a batch. Returns ensemble scores array shape (N,)."""
        X_norm = self.normalize(X)
        scores = np.zeros(len(X))

        if self.mlp is not None:
            mlp_scores = self.mlp.predict_proba(X_norm)
            scores += self.mlp_weight * mlp_scores

        if self.gbt is not None:
            gbt_scores = self.gbt.predict_proba(X)
            scores += self.gbt_weight * gbt_scores

        total_weight = (
            (self.mlp_weight if self.mlp else 0) +
            (self.gbt_weight if self.gbt else 0)
        )
        if total_weight > 0:
            scores /= total_weight

        return scores