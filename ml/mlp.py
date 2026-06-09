"""
Multi-layer perceptron from scratch using only numpy.

Architecture:
    Input → [Linear → BatchNorm → ReLU → Dropout] × n_hidden → Linear → sigmoid

Components implemented from scratch:
    - Linear layer (weight init: He initialization for ReLU networks)
    - Batch normalization (running mean/var for inference)
    - ReLU activation
    - Dropout (inverted dropout — scales at train time, not test time)
    - Adam optimizer (momentum + RMSprop combined)
    - Focal loss (from losses.py)

Batch normalization:
    Normalizes each feature across the batch to zero mean, unit variance,
    then applies learnable scale (gamma) and shift (beta).
    Reduces internal covariate shift — lets us use higher learning rates
    and makes training more stable.

    Train: normalize using batch statistics
    Inference: normalize using running statistics (accumulated during training)

Adam optimizer:
    Maintains first moment (momentum, m) and second moment (RMSprop, v).
    Bias-corrected update: prevents large steps early in training when
    m and v are initialized to zero (biased toward zero).

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    param -= lr * m_hat / (sqrt(v_hat) + eps)
"""

import numpy as np
from typing import Optional
from ml.losses import FocalLoss, sigmoid


class Linear:
    """Fully connected layer. y = xW + b"""

    def __init__(self, in_features: int, out_features: int):
        # He initialization: std = sqrt(2/fan_in) — optimal for ReLU
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None
        self._x: Optional[np.ndarray] = None  # cache for backprop

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW), (self.b, self.db)]


class BatchNorm:
    """
    Batch normalization layer.

    During training: normalize using batch mean/var, update running stats.
    During inference: normalize using accumulated running mean/var.
    """

    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma   = np.ones(n_features)    # learnable scale
        self.beta    = np.zeros(n_features)   # learnable shift
        self.momentum = momentum
        self.eps     = eps
        # Running statistics for inference
        self.running_mean = np.zeros(n_features)
        self.running_var  = np.ones(n_features)
        # Gradients
        self.dgamma: Optional[np.ndarray] = None
        self.dbeta:  Optional[np.ndarray] = None
        # Cache for backprop
        self._x_norm: Optional[np.ndarray] = None
        self._std:    Optional[np.ndarray] = None
        self._x_centered: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = x.mean(axis=0)
            var  = x.var(axis=0)
            self._std = np.sqrt(var + self.eps)
            self._x_centered = x - mean
            self._x_norm = self._x_centered / self._std
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            self._x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self._x_norm + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N = dout.shape[0]
        self.dgamma = (dout * self._x_norm).sum(axis=0)
        self.dbeta  = dout.sum(axis=0)

        dx_norm  = dout * self.gamma
        dvar     = (-0.5 * dx_norm * self._x_centered * self._std ** -3).sum(axis=0)
        dmean    = (-dx_norm / self._std).sum(axis=0) + dvar * (-2 * self._x_centered).mean(axis=0)
        dx       = dx_norm / self._std + dvar * 2 * self._x_centered / N + dmean / N
        return dx

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class ReLU:
    def __init__(self):
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


class Dropout:
    """
    Inverted dropout: scale activations by 1/p during training so that
    inference requires no scaling.
    """

    def __init__(self, p: float = 0.5):
        self.p = p  # probability of KEEPING a neuron
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 1.0:
            return x
        self._mask = (np.random.rand(*x.shape) < self.p) / self.p
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


class AdamOptimizer:
    """
    Adam optimizer. Maintains per-parameter m and v moments.
    Call step() after computing all gradients.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        self._m:  dict[int, np.ndarray] = {}
        self._v:  dict[int, np.ndarray] = {}

    def step(self, params: list[tuple[np.ndarray, Optional[np.ndarray]]]):
        """
        Update parameters in-place.
        params: list of (param, grad) tuples.
        """
        self.t += 1
        for i, (param, grad) in enumerate(params):
            if grad is None:
                continue
            g = grad + self.weight_decay * param  # L2 regularization

            if i not in self._m:
                self._m[i] = np.zeros_like(param)
                self._v[i] = np.zeros_like(param)

            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g ** 2

            m_hat = self._m[i] / (1 - self.beta1 ** self.t)
            v_hat = self._v[i] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLP:
    """
    3-layer MLP for binary fraud classification.

    Architecture:
        Input(d) → Linear(d, h1) → BN → ReLU → Dropout
                 → Linear(h1, h2) → BN → ReLU → Dropout
                 → Linear(h2, h3) → BN → ReLU → Dropout
                 → Linear(h3, 1)  → (sigmoid applied in loss)

    Output is a logit (raw score before sigmoid).
    Apply sigmoid to get probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64, 32],
        dropout_p: float = 0.3,
        lr: float = 1e-3,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        self.input_dim  = input_dim
        self.training   = True

        # Build layers
        dims = [input_dim] + hidden_dims
        self.linears  = []
        self.batchnorms = []
        self.relus    = []
        self.dropouts = []

        for i in range(len(dims) - 1):
            self.linears.append(Linear(dims[i], dims[i + 1]))
            self.batchnorms.append(BatchNorm(dims[i + 1]))
            self.relus.append(ReLU())
            self.dropouts.append(Dropout(1 - dropout_p))

        self.output_layer = Linear(dims[-1], 1)
        self.loss_fn      = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.optimizer    = AdamOptimizer(lr=lr)
        self.train_losses: list[float] = []

    def _all_params(self) -> list[tuple[np.ndarray, Optional[np.ndarray]]]:
        params = []
        for linear in self.linears:
            params.extend(linear.params())
        for bn in self.batchnorms:
            params.extend(bn.params())
        params.extend(self.output_layer.params())
        return params

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Returns logits shape (N,)."""
        h = x
        for linear, bn, relu, dropout in zip(
            self.linears, self.batchnorms, self.relus, self.dropouts
        ):
            h = linear.forward(h)
            h = bn.forward(h, training=self.training)
            h = relu.forward(h)
            h = dropout.forward(h, training=self.training)

        logits = self.output_layer.forward(h).squeeze(-1)
        return logits

    def backward(self, dlogits: np.ndarray):
        """Backward pass from loss gradient w.r.t. logits."""
        dh = self.output_layer.backward(dlogits.reshape(-1, 1))

        for linear, bn, relu, dropout in zip(
            reversed(self.linears),
            reversed(self.batchnorms),
            reversed(self.relus),
            reversed(self.dropouts),
        ):
            dh = dropout.backward(dh)
            dh = relu.backward(dh)
            dh = bn.backward(dh)
            dh = linear.backward(dh)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Single training step. Returns loss scalar."""
        self.training = True
        logits = self.forward(x)
        loss, grad = self.loss_fn(logits, y)
        self.backward(grad)
        self.optimizer.step(self._all_params())
        self.train_losses.append(loss)
        return loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Inference. Returns probabilities shape (N,)."""
        self.training = False
        logits = self.forward(x)
        return sigmoid(logits)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions."""
        return (self.predict_proba(x) >= threshold).astype(int)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> list[float]:
        """
        Full training loop with mini-batch SGD.
        Returns list of per-epoch mean losses.
        """
        n = len(X)
        epoch_losses = []

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            batch_losses = []

            for start in range(0, n, batch_size):
                Xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                loss = self.train_step(Xb, yb)
                batch_losses.append(loss)

            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(float(epoch_loss))

            if verbose and (epoch % max(1, epochs // 5) == 0):
                probs = self.predict_proba(X)
                preds = (probs >= 0.5).astype(int)
                acc   = (preds == y).mean()
                print(f"  Epoch {epoch+1:3d}/{epochs} | loss={epoch_loss:.4f} | acc={acc:.4f}")

        return epoch_losses

    def save(self, path: str):
        """Save weights to numpy .npz file."""
        data = {}
        for i, linear in enumerate(self.linears):
            data[f"linear_{i}_W"] = linear.W
            data[f"linear_{i}_b"] = linear.b
        for i, bn in enumerate(self.batchnorms):
            data[f"bn_{i}_gamma"]        = bn.gamma
            data[f"bn_{i}_beta"]         = bn.beta
            data[f"bn_{i}_running_mean"] = bn.running_mean
            data[f"bn_{i}_running_var"]  = bn.running_var
        data["output_W"] = self.output_layer.W
        data["output_b"] = self.output_layer.b
        np.savez(path, **data)

    def load(self, path: str):
        """Load weights from numpy .npz file."""
        data = np.load(path)
        for i, linear in enumerate(self.linears):
            linear.W = data[f"linear_{i}_W"]
            linear.b = data[f"linear_{i}_b"]
        for i, bn in enumerate(self.batchnorms):
            bn.gamma        = data[f"bn_{i}_gamma"]
            bn.beta         = data[f"bn_{i}_beta"]
            bn.running_mean = data[f"bn_{i}_running_mean"]
            bn.running_var  = data[f"bn_{i}_running_var"]
        self.output_layer.W = data["output_W"]
        self.output_layer.b = data["output_b"]