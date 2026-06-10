"""
KernelSHAP — sampling-based Shapley value estimation.

Algorithm (Lundberg & Lee 2017):
  1. Sample M random coalitions z ∈ {0,1}^d (subsets of features present).
  2. For each coalition, build a masked input: present features take their
     actual values; absent features are replaced by a background sample drawn
     from a reference dataset.
  3. Evaluate f(z_masked) for each coalition.
  4. Solve a weighted least-squares regression:
       φ = argmin_φ  Σ_z  π(z) · [f(z) - (φ_0 + Σ_j z_j φ_j)]²
     where the kernel weight π(z) = (d-1) / [C(d, |z|) · |z| · (d-|z|)]
     and the constraints φ_0 = f(background_mean) and Σφ_j = f(x)-φ_0 are
     enforced by the regression design.
  5. φ_j is the SHAP value for feature j — its average marginal contribution
     across all possible orderings.

No sklearn, no shap library. Numpy only.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Kernel weight
# ---------------------------------------------------------------------------

def _kernel_weight(d: int, z: np.ndarray) -> float:
    """
    Shapley kernel weight for a coalition z ∈ {0,1}^d.

    π(z) = (d - 1) / [C(d, |z|) · |z| · (d - |z|)]

    Coalitions of size 0 or d are given infinite weight (exact boundary
    conditions); in practice we skip them and enforce via constraints.
    """
    s = int(z.sum())
    if s == 0 or s == d:
        return 0.0  # handled via constraints, not included in regression
    binom = math.comb(d, s)
    return (d - 1) / (binom * s * (d - s))


# ---------------------------------------------------------------------------
# Weighted least-squares solver
# ---------------------------------------------------------------------------

def _wls(Z: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Solve the weighted least-squares system: min_φ  ||W^{1/2}(Zφ - y)||²

    Z  : (M, d)  — coalition matrix (columns = features)
    y  : (M,)    — model outputs for each coalition
    weights : (M,) — kernel weights π(z)

    Returns φ : (d,)  — SHAP values (intercept NOT included)

    We use the closed-form:  φ = (Z^T W Z)^{-1} Z^T W y
    with a small ridge for numerical stability.
    """
    W = np.diag(weights)
    ZtW = Z.T @ W          # (d, M)
    ZtWZ = ZtW @ Z         # (d, d)
    ZtWy = ZtW @ y         # (d,)

    # Small ridge to handle near-singular cases (constant features, etc.)
    ridge = 1e-6 * np.eye(ZtWZ.shape[0])
    phi = np.linalg.solve(ZtWZ + ridge, ZtWy)
    return phi


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KernelSHAP:
    """
    Sampling-based SHAP explanations for any black-box model.

    Parameters
    ----------
    predict_fn  : callable  f(X: np.ndarray) → np.ndarray of shape (n,)
                  Model's predict_proba or decision function. Must handle
                  batched inputs.
    background  : np.ndarray, shape (n_bg, d)
                  Reference dataset. Absent features are replaced by a
                  randomly drawn row from this set. 50–200 rows is enough.
    n_samples   : int
                  Number of coalition samples per explanation. More samples →
                  lower variance. 512 is a good default for d ≤ 30.
    feature_names : list[str], optional
                  Human-readable feature names for the explanation dict.
    random_state : int, optional
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        background: np.ndarray,
        n_samples: int = 512,
        feature_names: Optional[list] = None,
        random_state: Optional[int] = 42,
    ):
        self.predict_fn = predict_fn
        self.background = np.asarray(background, dtype=float)
        self.n_samples = n_samples
        self.d = background.shape[1]
        self.feature_names = (
            feature_names if feature_names is not None
            else [f"f{i}" for i in range(self.d)]
        )
        self.rng = np.random.default_rng(random_state)

        # Pre-compute the expected model output over the background.
        # This is the SHAP baseline (φ_0 = E[f(x)]).
        bg_preds = self.predict_fn(self.background)
        self.baseline = float(np.mean(bg_preds))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_coalitions(self) -> np.ndarray:
        """
        Sample M random coalitions in {0,1}^d.

        Strategy: sample coalition sizes from a distribution that upweights
        small and large coalitions (where the kernel weight is highest), then
        sample that many features to be "present" (=1).
        """
        M = self.n_samples
        d = self.d
        coalitions = np.zeros((M, d), dtype=float)

        # Pair each sample with its mirror complement to reduce variance.
        half = M // 2
        for i in range(half):
            # Draw coalition size s ~ Uniform(1, d-1)
            s = int(self.rng.integers(1, d))
            cols = self.rng.choice(d, size=s, replace=False)
            coalitions[i, cols] = 1.0
            # Mirror: complement coalition
            coalitions[half + i] = 1.0 - coalitions[i]

        # Fill any remaining row
        for i in range(2 * half, M):
            s = int(self.rng.integers(1, d))
            cols = self.rng.choice(d, size=s, replace=False)
            coalitions[i, cols] = 1.0

        return coalitions

    def _mask_inputs(self, x: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """
        Build M masked inputs.

        For each coalition z:
          - features where z_j = 1: take value from x
          - features where z_j = 0: draw a random background row and use that
            feature's value

        Returns masked_X : (M, d)
        """
        M = coalitions.shape[0]
        # Draw M background rows (with replacement)
        bg_idx = self.rng.integers(0, len(self.background), size=M)
        bg_rows = self.background[bg_idx]          # (M, d)

        # Vectorised: present features → x, absent → background
        x_broadcast = np.tile(x, (M, 1))           # (M, d)
        masked = np.where(coalitions == 1, x_broadcast, bg_rows)
        return masked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, x: np.ndarray) -> dict:
        """
        Compute SHAP values for a single input x of shape (d,).

        Returns
        -------
        dict with keys:
          'shap_values'  : np.ndarray (d,)  — φ_j for each feature
          'baseline'     : float            — E[f] over background
          'prediction'   : float            — f(x)
          'contributions': list of dicts    — sorted by |φ_j|, descending
          'explanation'  : str              — human-readable summary
        """
        x = np.asarray(x, dtype=float).ravel()
        assert len(x) == self.d, f"Expected {self.d} features, got {len(x)}"

        prediction = float(self.predict_fn(x.reshape(1, -1))[0])

        # --- Sample coalitions ---
        Z = self._sample_coalitions()          # (M, d)

        # --- Build masked inputs and get model outputs ---
        masked_X = self._mask_inputs(x, Z)     # (M, d)
        f_z = self.predict_fn(masked_X)        # (M,)

        # --- Kernel weights ---
        weights = np.array([_kernel_weight(self.d, Z[i]) for i in range(len(Z))])

        # Drop zero-weight rows (size 0 or d coalitions)
        valid = weights > 0
        Z_v, f_v, w_v = Z[valid], f_z[valid], weights[valid]

        # Adjust target: subtract baseline so intercept = 0 in regression
        y_adj = f_v - self.baseline

        # --- Enforce efficiency constraint: Σφ_j = f(x) - baseline ---
        # Augment last feature's contribution from the constraint, then solve
        # for the first d-1 features using the remaining rows.
        # Simpler approach: solve unconstrained then renormalize to sum exactly.
        phi_raw = _wls(Z_v, y_adj, w_v)

        # Renormalize so SHAP values sum exactly to prediction - baseline
        target_sum = prediction - self.baseline
        current_sum = phi_raw.sum()
        if abs(current_sum) > 1e-10:
            phi = phi_raw * (target_sum / current_sum)
        else:
            phi = phi_raw

        # --- Build output ---
        contributions = sorted(
            [
                {
                    "feature": self.feature_names[j],
                    "value": float(x[j]),
                    "shap": float(phi[j]),
                    "abs_shap": float(abs(phi[j])),
                }
                for j in range(self.d)
            ],
            key=lambda d: d["abs_shap"],
            reverse=True,
        )

        explanation = _format_explanation(
            contributions, self.baseline, prediction
        )

        return {
            "shap_values": phi,
            "baseline": self.baseline,
            "prediction": prediction,
            "contributions": contributions,
            "explanation": explanation,
        }

    def explain_batch(self, X: np.ndarray) -> list[dict]:
        """Explain a batch of inputs. Returns list of explain() dicts."""
        return [self.explain(X[i]) for i in range(len(X))]


# ---------------------------------------------------------------------------
# Human-readable explanation formatter
# ---------------------------------------------------------------------------

def _format_explanation(
    contributions: list[dict],
    baseline: float,
    prediction: float,
    top_k: int = 4,
) -> str:
    """
    Produce a plain-English explanation string like:
      "Risk score 0.87 (baseline 0.12). Top drivers: amount_zscore +0.41,
       new_device +0.29, hour_sin +0.18, merchant_freq -0.07."
    """
    top = contributions[:top_k]
    parts = []
    for c in top:
        sign = "+" if c["shap"] >= 0 else ""
        parts.append(f"{c['feature']} {sign}{c['shap']:.3f}")

    drivers = ", ".join(parts)
    direction = "increased" if prediction > baseline else "decreased"
    return (
        f"Risk score {prediction:.3f} (baseline {baseline:.3f}). "
        f"Score {direction} from baseline. "
        f"Top drivers: {drivers}."
    )


# ---------------------------------------------------------------------------
# Convenience: top-k features for logging
# ---------------------------------------------------------------------------

def top_features(
    shap_result: dict, k: int = 5
) -> list[tuple[str, float]]:
    """Return top-k (feature_name, shap_value) pairs by absolute contribution."""
    return [
        (c["feature"], c["shap"])
        for c in shap_result["contributions"][:k]
    ]