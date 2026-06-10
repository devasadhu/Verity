"""
PSI Drift Detector — Population Stability Index from scratch.

PSI is the standard metric for detecting distribution shift in credit risk
model monitoring (Basel II/III). It measures how much a feature's distribution
has shifted between a reference (training) population and a current population.

Formula:
    PSI = Σ_i  (A_i - E_i) * ln(A_i / E_i)

where E_i = expected fraction in bucket i (reference/training distribution)
      A_i = actual fraction in bucket i (current/production distribution)

Interpretation:
    PSI < 0.10   → No significant change. Model is stable.
    0.10 ≤ PSI < 0.20 → Minor shift. Monitor closely.
    PSI ≥ 0.20   → Significant shift. Trigger retraining alert.

Per-feature PSI is computed for each of the 23 input features, plus the
model's output score. A global alert fires when ANY feature's PSI ≥ 0.20
or when score PSI ≥ 0.10 (score drift is more sensitive).

No sklearn. NumPy only.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PSI_NO_CHANGE = 0.10
PSI_MINOR_SHIFT = 0.20        # triggers retraining alert
PSI_SCORE_ALERT = 0.10        # tighter threshold for model output score


def psi_label(psi: float) -> str:
    if psi < PSI_NO_CHANGE:
        return "stable"
    elif psi < PSI_MINOR_SHIFT:
        return "monitor"
    else:
        return "alert"


# ---------------------------------------------------------------------------
# Core PSI calculation
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    bins: Optional[np.ndarray] = None,
) -> tuple[float, np.ndarray]:
    """
    Compute PSI between reference and current distributions.

    Parameters
    ----------
    reference : 1-D array — reference (training) population values
    current   : 1-D array — current (production) population values
    n_bins    : number of equal-frequency bins (ignored if bins supplied)
    bins      : pre-computed bin edges from reference (pass to reuse edges)

    Returns
    -------
    psi   : float — Population Stability Index
    bins  : np.ndarray — bin edges used (reuse for consistency)
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if bins is None:
        # Use equal-frequency binning on the reference population.
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(reference, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

    # Compute bucket frequencies
    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    n_ref = len(reference)
    n_cur = len(current)

    # Fractional proportions per bucket, clipped to avoid log(0)
    eps = 1e-4
    E = np.clip(ref_counts / n_ref, eps, None)
    A = np.clip(cur_counts / n_cur, eps, None)

    # Renormalise after clipping
    E = E / E.sum()
    A = A / A.sum()

    psi = float(np.sum((A - E) * np.log(A / E)))
    return psi, bins


# ---------------------------------------------------------------------------
# Feature-level drift monitor
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftReport:
    feature_name: str
    psi: float
    status: str        # "stable" | "monitor" | "alert"
    reference_mean: float
    current_mean: float
    mean_shift: float  # (current - reference) / reference_std


@dataclass
class DriftReport:
    """Full drift report for one monitoring window."""
    window_size: int
    score_psi: float
    score_status: str
    feature_reports: list[FeatureDriftReport] = field(default_factory=list)
    alert: bool = False
    alert_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "score_psi": round(self.score_psi, 4),
            "score_status": self.score_status,
            "alert": self.alert,
            "alert_reason": self.alert_reason,
            "features": [
                {
                    "feature": r.feature_name,
                    "psi": round(r.psi, 4),
                    "status": r.status,
                    "ref_mean": round(r.reference_mean, 4),
                    "cur_mean": round(r.current_mean, 4),
                    "mean_shift_sigma": round(r.mean_shift, 3),
                }
                for r in sorted(
                    self.feature_reports, key=lambda x: x.psi, reverse=True
                )
            ],
        }


class PSIDriftDetector:
    """
    Monitors feature and score distributions for drift.

    Usage:
        detector = PSIDriftDetector(feature_names)
        detector.fit_reference(X_train, scores_train)
        # ... in production, accumulate a window of recent transactions ...
        report = detector.check(X_window, scores_window)
        if report.alert:
            trigger_retraining()

    Parameters
    ----------
    feature_names : list[str]
    n_bins        : number of PSI bins (10 is standard in credit risk)
    window_size   : minimum samples before checking drift (default 500)
    """

    def __init__(
        self,
        feature_names: list[str],
        n_bins: int = 10,
        window_size: int = 500,
    ):
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.window_size = window_size

        # Set during fit_reference
        self._ref_X: Optional[np.ndarray] = None
        self._ref_scores: Optional[np.ndarray] = None
        self._feature_bins: list[Optional[np.ndarray]] = [None] * len(feature_names)
        self._score_bins: Optional[np.ndarray] = None
        self._ref_means: Optional[np.ndarray] = None
        self._ref_stds: Optional[np.ndarray] = None

        # Rolling buffer for incoming production data
        self._buffer_X: list[np.ndarray] = []
        self._buffer_scores: list[float] = []

    # ------------------------------------------------------------------
    # Reference fitting
    # ------------------------------------------------------------------

    def fit_reference(
        self, X: np.ndarray, scores: np.ndarray
    ) -> None:
        """
        Fit the reference distribution from training/validation data.

        X      : (n, d) feature matrix
        scores : (n,) model output scores
        """
        X = np.asarray(X, dtype=float)
        scores = np.asarray(scores, dtype=float)

        self._ref_X = X
        self._ref_scores = scores
        self._ref_means = X.mean(axis=0)
        self._ref_stds = X.std(axis=0) + 1e-8

        # Precompute bins for each feature
        for j in range(X.shape[1]):
            _, bins = compute_psi(X[:, j], X[:, j], n_bins=self.n_bins)
            self._feature_bins[j] = bins

        # Bins for score
        _, self._score_bins = compute_psi(scores, scores, n_bins=self.n_bins)

    def save_reference(self, path: str) -> None:
        """Persist reference statistics to a JSON file."""
        if self._ref_X is None:
            raise RuntimeError("Call fit_reference first.")
        data = {
            "feature_names": self.feature_names,
            "n_bins": self.n_bins,
            "window_size": self.window_size,
            "ref_means": self._ref_means.tolist(),
            "ref_stds": self._ref_stds.tolist(),
            "feature_bins": [b.tolist() for b in self._feature_bins],
            "score_bins": self._score_bins.tolist(),
            # Store a sample of the reference for PSI computation
            "ref_X_sample": self._ref_X[:2000].tolist(),
            "ref_scores_sample": self._ref_scores[:2000].tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_reference(cls, path: str) -> "PSIDriftDetector":
        """Load a saved reference."""
        with open(path) as f:
            data = json.load(f)
        detector = cls(
            feature_names=data["feature_names"],
            n_bins=data["n_bins"],
            window_size=data["window_size"],
        )
        detector._ref_means = np.array(data["ref_means"])
        detector._ref_stds = np.array(data["ref_stds"])
        detector._feature_bins = [np.array(b) for b in data["feature_bins"]]
        detector._score_bins = np.array(data["score_bins"])
        detector._ref_X = np.array(data["ref_X_sample"])
        detector._ref_scores = np.array(data["ref_scores_sample"])
        return detector

    # ------------------------------------------------------------------
    # Production monitoring
    # ------------------------------------------------------------------

    def push(self, x: np.ndarray, score: float) -> None:
        """Add one transaction to the rolling buffer."""
        self._buffer_X.append(np.asarray(x, dtype=float).ravel())
        self._buffer_scores.append(float(score))

    def push_batch(self, X: np.ndarray, scores: np.ndarray) -> None:
        """Add a batch to the rolling buffer."""
        for i in range(len(X)):
            self.push(X[i], scores[i])

    def check(
        self,
        X_window: Optional[np.ndarray] = None,
        scores_window: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """
        Compute PSI between reference and current window.

        If X_window / scores_window are not provided, uses the internal buffer.
        Clears the buffer after checking.
        """
        if self._ref_X is None:
            raise RuntimeError("Call fit_reference before check().")

        if X_window is None:
            if len(self._buffer_X) < self.window_size:
                return DriftReport(
                    window_size=len(self._buffer_X),
                    score_psi=0.0,
                    score_status="stable",
                    alert=False,
                    alert_reason="Insufficient window data.",
                )
            X_window = np.array(self._buffer_X)
            scores_window = np.array(self._buffer_scores)
            self._buffer_X = []
            self._buffer_scores = []
        else:
            X_window = np.asarray(X_window, dtype=float)
            scores_window = np.asarray(scores_window, dtype=float)

        # Score PSI
        score_psi, _ = compute_psi(
            self._ref_scores,
            scores_window,
            bins=self._score_bins,
        )

        # Feature PSI
        cur_means = X_window.mean(axis=0)
        feature_reports = []
        alert_features = []

        for j, name in enumerate(self.feature_names):
            psi_val, _ = compute_psi(
                self._ref_X[:, j],
                X_window[:, j],
                bins=self._feature_bins[j],
            )
            mean_shift = (cur_means[j] - self._ref_means[j]) / self._ref_stds[j]
            status = psi_label(psi_val)
            feature_reports.append(
                FeatureDriftReport(
                    feature_name=name,
                    psi=psi_val,
                    status=status,
                    reference_mean=float(self._ref_means[j]),
                    current_mean=float(cur_means[j]),
                    mean_shift=float(mean_shift),
                )
            )
            if status == "alert":
                alert_features.append(f"{name} (PSI={psi_val:.3f})")

        # Determine overall alert
        alert = False
        alert_reason = ""
        if score_psi >= PSI_SCORE_ALERT:
            alert = True
            alert_reason = f"Score PSI={score_psi:.3f} ≥ {PSI_SCORE_ALERT}. "
        if alert_features:
            alert = True
            alert_reason += f"Feature alerts: {', '.join(alert_features[:5])}."

        return DriftReport(
            window_size=len(scores_window),
            score_psi=float(score_psi),
            score_status=psi_label(score_psi),
            feature_reports=feature_reports,
            alert=alert,
            alert_reason=alert_reason.strip(),
        )

    def top_drifted_features(
        self, report: DriftReport, k: int = 5
    ) -> list[FeatureDriftReport]:
        """Return top-k features by PSI from a report."""
        return sorted(report.feature_reports, key=lambda r: r.psi, reverse=True)[:k]

    def buffer_size(self) -> int:
        return len(self._buffer_X)