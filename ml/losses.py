"""
Loss functions for fraud detection.

Why not cross-entropy?
    Fraud datasets are typically 0.1-1% positive class.
    Cross-entropy lets the model predict "legit" for everything
    and still achieve 99%+ accuracy. The model learns nothing.

Focal Loss (Lin et al., 2017 — originally for object detection):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The term (1 - p_t)^gamma is the "focusing" factor.
    - Easy negatives (p_t close to 1): factor ≈ 0, loss ≈ 0 → model ignores them
    - Hard examples (p_t close to 0.5): factor ≈ 1, loss ≈ normal CE
    - gamma=0 → reduces to weighted cross-entropy
    - gamma=2 is the standard value (from the paper)

    alpha balances pos/neg class weight.
    For fraud: alpha ~ 0.25 (down-weight the majority legit class).
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def binary_cross_entropy(logits: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """Standard BCE from logits. Returns scalar mean loss."""
    p = np.clip(sigmoid(logits), eps, 1 - eps)
    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))


def binary_cross_entropy_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gradient of BCE w.r.t. logits. Clean form: sigmoid(logits) - labels."""
    return (sigmoid(logits) - labels) / len(labels)


def focal_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
    eps: float = 1e-7,
) -> float:
    """
    Focal loss from logits. Returns scalar mean loss.

    Args:
        logits: raw model outputs, shape (N,)
        labels: binary labels 0/1, shape (N,)
        gamma:  focusing parameter (2.0 from paper)
        alpha:  class weight for positives (0.25 from paper)
    """
    p     = np.clip(sigmoid(logits), eps, 1 - eps)
    p_t   = np.where(labels == 1, p, 1 - p)
    alpha_t = np.where(labels == 1, alpha, 1 - alpha)
    focal_weight = (1 - p_t) ** gamma
    loss  = -alpha_t * focal_weight * np.log(p_t)
    return float(np.mean(loss))


def focal_loss_grad(
    logits: np.ndarray,
    labels: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    Gradient of focal loss w.r.t. logits.

    Derived via chain rule:
        dFL/dlogit = dFL/dp * dp/dlogit
        dp/dlogit  = p * (1 - p)   [sigmoid derivative]

    For the focal term, the full derivative has two parts:
        - gradient through the log term
        - gradient through the focusing weight (1-p_t)^gamma
    """
    p       = np.clip(sigmoid(logits), eps, 1 - eps)
    p_t     = np.where(labels == 1, p, 1 - p)
    alpha_t = np.where(labels == 1, alpha, 1 - alpha)

    # Sign: +1 for positives, -1 for negatives (adjusts direction of p_t)
    sign = np.where(labels == 1, 1.0, -1.0)

    focal_weight = (1 - p_t) ** gamma
    log_term     = np.log(p_t + eps)

    # d/dlogit of -alpha_t * (1-p_t)^gamma * log(p_t)
    # = alpha_t * p*(1-p) * sign * [
    #       gamma * (1-p_t)^(gamma-1) * log(p_t)
    #     - (1-p_t)^gamma / p_t
    #   ]
    grad = alpha_t * p * (1 - p) * sign * (
        gamma * (1 - p_t) ** (gamma - 1) * log_term
        - focal_weight / (p_t + eps)
    )
    return grad / len(logits)


class FocalLoss:
    """Stateful focal loss — stores gamma/alpha, computes loss + grad together."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
        """Returns (loss_scalar, gradient_wrt_logits)."""
        loss = focal_loss(logits, labels, self.gamma, self.alpha)
        grad = focal_loss_grad(logits, labels, self.gamma, self.alpha)
        return loss, grad