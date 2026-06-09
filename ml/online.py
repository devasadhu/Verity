"""
Online learning loop.

Fraud patterns shift over time — a model trained on last month's data
will decay in performance as fraudsters adapt. Online learning updates
the model continuously on new transactions without full retraining.

Strategy: SGD with a small learning rate on a fixed-size replay buffer.
    - New transactions are added to the buffer
    - Every N transactions, run one SGD pass over the buffer
    - Buffer uses reservoir sampling to maintain representative distribution
      (doesn't just keep the most recent transactions)

Replay buffer rationale:
    Pure online learning (update on each sample) suffers from
    "catastrophic forgetting" — the model overwrites old patterns.
    A replay buffer maintains memory of past distributions while
    still adapting to new ones. Same principle used in DQN (Atari).

Drift detection:
    Track rolling prediction accuracy. If it drops significantly,
    trigger a faster learning rate burst (lr warmup).
"""

import numpy as np
from typing import Optional
from ml.mlp import MLP, sigmoid
from ml.losses import focal_loss


class ReplayBuffer:
    """
    Fixed-size experience replay buffer with reservoir sampling.

    Reservoir sampling guarantees that after seeing N samples,
    each sample has equal probability max_size/N of being in the buffer
    — regardless of arrival order. This prevents recency bias.
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size   = max_size
        self._X:  list[np.ndarray] = []
        self._y:  list[float]      = []
        self._n_seen = 0            # total samples seen (including evicted)

    def add(self, x: np.ndarray, label: float):
        """Add a sample using reservoir sampling."""
        self._n_seen += 1

        if len(self._X) < self.max_size:
            self._X.append(x.copy())
            self._y.append(label)
        else:
            # Replace a random existing sample with probability max_size/n_seen
            j = np.random.randint(0, self._n_seen)
            if j < self.max_size:
                self._X[j] = x.copy()
                self._y[j] = label

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random batch from the buffer."""
        n = len(self._X)
        if n == 0:
            raise ValueError("Buffer is empty")
        idx = np.random.choice(n, min(batch_size, n), replace=False)
        X = np.stack([self._X[i] for i in idx])
        y = np.array([self._y[i] for i in idx])
        return X, y

    def size(self) -> int:
        return len(self._X)

    def fraud_rate(self) -> float:
        if not self._y:
            return 0.0
        return sum(self._y) / len(self._y)


class OnlineLearner:
    """
    Wraps an MLP with an online learning loop.

    After initial training (via MLP.fit), call update() on each new
    transaction to keep the model current.
    """

    def __init__(
        self,
        model: MLP,
        buffer_size: int = 10_000,
        update_every: int = 100,       # run SGD pass every N new samples
        online_lr: float = 1e-4,       # small LR for online updates
        batch_size: int = 64,
        drift_window: int = 500,       # samples to use for drift detection
        drift_threshold: float = 0.05, # loss increase that triggers fast adapt
    ):
        self.model         = model
        self.buffer        = ReplayBuffer(buffer_size)
        self.update_every  = update_every
        self.online_lr     = online_lr
        self.batch_size    = batch_size
        self.drift_window  = drift_window
        self.drift_threshold = drift_threshold
        self._n_since_update = 0
        self._recent_losses: list[float] = []
        self._baseline_loss: Optional[float] = None

    def update(self, x: np.ndarray, label: float) -> Optional[float]:
        """
        Process one new sample. Returns training loss if an update step ran,
        None otherwise.
        """
        self.buffer.add(x, label)
        self._n_since_update += 1

        # Track recent prediction loss for drift detection
        self.model.training = False
        logit = self.model.forward(x.reshape(1, -1))[0]
        inst_loss = focal_loss(np.array([logit]), np.array([label]))
        self._recent_losses.append(inst_loss)
        if len(self._recent_losses) > self.drift_window:
            self._recent_losses.pop(0)

        if self._n_since_update < self.update_every:
            return None

        # Run online SGD pass
        self._n_since_update = 0
        return self._run_update()

    def _run_update(self) -> float:
        """Run one mini-batch SGD update on replay buffer."""
        if self.buffer.size() < self.batch_size:
            return 0.0

        # Detect drift — increase LR temporarily if loss has spiked
        lr = self.online_lr
        if self._baseline_loss is not None and len(self._recent_losses) >= 50:
            recent_mean = np.mean(self._recent_losses[-50:])
            if recent_mean > self._baseline_loss * (1 + self.drift_threshold):
                lr = self.online_lr * 5   # fast adapt burst
                print(f"[OnlineLearner] Drift detected — loss {recent_mean:.4f} "
                      f"vs baseline {self._baseline_loss:.4f}. Using lr={lr:.2e}")

        # Save and override optimizer LR
        orig_lr = self.model.optimizer.lr
        self.model.optimizer.lr = lr

        Xb, yb = self.buffer.sample(self.batch_size)
        loss = self.model.train_step(Xb, yb)

        self.model.optimizer.lr = orig_lr

        # Update baseline loss
        if self._baseline_loss is None:
            self._baseline_loss = loss
        else:
            self._baseline_loss = 0.99 * self._baseline_loss + 0.01 * loss

        return loss

    def stats(self) -> dict:
        return {
            "buffer_size":    self.buffer.size(),
            "buffer_fraud_rate": round(self.buffer.fraud_rate(), 4),
            "n_since_update": self._n_since_update,
            "baseline_loss":  round(self._baseline_loss, 4) if self._baseline_loss else None,
            "recent_loss_mean": round(np.mean(self._recent_losses[-50:]), 4)
                                if len(self._recent_losses) >= 50 else None,
        }