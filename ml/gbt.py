"""
Gradient Boosted Trees from scratch.

XGBoost-style implementation with second-order gradients.

How gradient boosting works:
    1. Start with a constant prediction (log-odds of mean label)
    2. For each round t:
        a. Compute pseudo-residuals: first-order gradient g_i and
           second-order gradient h_i of the loss w.r.t. current prediction
        b. Fit a regression tree to minimize the weighted loss
        c. Add the tree's prediction to the ensemble (scaled by learning rate)

Why second-order gradients?
    Standard gradient boosting uses only g (gradient).
    XGBoost uses both g and h (Hessian) to compute the optimal leaf weight:
        w* = -sum(g_i) / (sum(h_i) + lambda)
    This gives more accurate leaf values and better regularization.

Tree building:
    For each split candidate, compute the gain:
        Gain = 0.5 * [G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - (G_L+G_R)^2/(H_L+H_R+λ)] - γ
    where G = sum of gradients, H = sum of Hessians in each child.
    Choose the split that maximizes gain.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TreeNode:
    """A node in a regression tree."""
    feature_idx: int = -1
    threshold: float = 0.0
    left:  Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    leaf_value: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf_value is not None


class RegressionTree:
    """
    Single regression tree for gradient boosting.
    Finds best splits by maximizing XGBoost-style gain.
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,   # L2 regularization on leaf weights
        reg_gamma: float = 0.0,    # minimum gain to make a split
        subsample_features: float = 0.8,
    ):
        self.max_depth          = max_depth
        self.min_child_weight   = min_child_weight
        self.reg_lambda         = reg_lambda
        self.reg_gamma          = reg_gamma
        self.subsample_features = subsample_features
        self.root: Optional[TreeNode] = None
        self._feature_subset: Optional[np.ndarray] = None

    def _leaf_weight(self, g: np.ndarray, h: np.ndarray) -> float:
        """Optimal leaf weight: -sum(g) / (sum(h) + lambda)"""
        return -g.sum() / (h.sum() + self.reg_lambda)

    def _gain(
        self,
        g_l: np.ndarray, h_l: np.ndarray,
        g_r: np.ndarray, h_r: np.ndarray,
    ) -> float:
        """XGBoost split gain formula."""
        def score(g, h):
            return g.sum() ** 2 / (h.sum() + self.reg_lambda)

        return 0.5 * (score(g_l, h_l) + score(g_r, h_r) - score(
            np.concatenate([g_l, g_r]),
            np.concatenate([h_l, h_r])
        )) - self.reg_gamma

    def _best_split(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
    ) -> tuple[int, float, float]:
        """
        Find best (feature, threshold, gain) across all candidate splits.
        Returns (-1, 0, -inf) if no beneficial split found.
        """
        best_gain    = -np.inf
        best_feature = -1
        best_thresh  = 0.0
        n_features   = X.shape[1]

        # Feature subsampling (column sampling like XGBoost's colsample_bytree)
        n_sample = max(1, int(n_features * self.subsample_features))
        features = np.random.choice(n_features, n_sample, replace=False)

        for feat in features:
            values = X[:, feat]
            # Use unique sorted values as split candidates
            thresholds = np.unique(values)
            if len(thresholds) <= 1:
                continue

            for thresh in thresholds[:-1]:
                left_mask  = values <= thresh
                right_mask = ~left_mask

                if (h[left_mask].sum()  < self.min_child_weight or
                    h[right_mask].sum() < self.min_child_weight):
                    continue

                gain = self._gain(g[left_mask], h[left_mask],
                                  g[right_mask], h[right_mask])
                if gain > best_gain:
                    best_gain    = gain
                    best_feature = feat
                    best_thresh  = thresh

        return best_feature, best_thresh, best_gain

    def _build(
        self,
        X: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """Recursively build the tree."""
        # Leaf conditions
        if (depth >= self.max_depth or
            len(g) <= 1 or
            h.sum() < self.min_child_weight):
            return TreeNode(leaf_value=self._leaf_weight(g, h))

        feat, thresh, gain = self._best_split(X, g, h)

        if feat == -1 or gain <= 0:
            return TreeNode(leaf_value=self._leaf_weight(g, h))

        left_mask  = X[:, feat] <= thresh
        right_mask = ~left_mask

        node = TreeNode(feature_idx=feat, threshold=thresh)
        node.left  = self._build(X[left_mask],  g[left_mask],  h[left_mask],  depth + 1)
        node.right = self._build(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray):
        """Fit tree to gradients g and Hessians h."""
        self.root = self._build(X, g, h, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return leaf values for each sample."""
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf:
            return node.leaf_value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)


class GradientBoostedTrees:
    """
    Gradient boosted binary classifier.
    Uses logistic loss with second-order (Newton) gradient steps.

    Logistic loss gradients:
        p    = sigmoid(F)        # current prediction
        g_i  = p_i - y_i        # first-order gradient
        h_i  = p_i * (1 - p_i)  # second-order gradient (Hessian)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        subsample: float = 0.8,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
    ):
        self.n_estimators    = n_estimators
        self.learning_rate   = learning_rate
        self.max_depth       = max_depth
        self.subsample       = subsample
        self.reg_lambda      = reg_lambda
        self.min_child_weight = min_child_weight
        self.trees:  list[RegressionTree] = []
        self.F0:     float = 0.0   # initial prediction (log-odds)
        self.train_losses: list[float] = []

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)))

    def _log_loss(self, y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "GradientBoostedTrees":
        n = len(y)
        # Initial prediction: log-odds of mean label
        mean_y  = np.clip(y.mean(), 1e-7, 1 - 1e-7)
        self.F0 = np.log(mean_y / (1 - mean_y))
        F       = np.full(n, self.F0)

        for t in range(self.n_estimators):
            p = self._sigmoid(F)
            g = p - y                  # first-order gradient
            h = p * (1 - p)            # second-order gradient

            # Row subsampling
            idx = np.random.choice(n, int(n * self.subsample), replace=False)
            Xs, gs, hs = X[idx], g[idx], h[idx]

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
            )
            tree.fit(Xs, gs, hs)
            self.trees.append(tree)

            # Update predictions for ALL samples
            F += self.learning_rate * tree.predict(X)

            loss = self._log_loss(y, self._sigmoid(F))
            self.train_losses.append(loss)

            if verbose and (t % max(1, self.n_estimators // 5) == 0):
                print(f"  Round {t+1:3d}/{self.n_estimators} | loss={loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probabilities for each sample."""
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return self._sigmoid(F)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self, n_features: int) -> np.ndarray:
        """
        Compute feature importance as split frequency across all trees.
        Returns array of shape (n_features,), normalized to sum to 1.
        """
        importance = np.zeros(n_features)

        def count_splits(node):
            if node is None or node.is_leaf:
                return
            importance[node.feature_idx] += 1
            count_splits(node.left)
            count_splits(node.right)

        for tree in self.trees:
            count_splits(tree.root)

        total = importance.sum()
        return importance / total if total > 0 else importance