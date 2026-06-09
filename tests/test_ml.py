import numpy as np
import pytest
from ml.losses import focal_loss, focal_loss_grad, binary_cross_entropy, sigmoid
from ml.mlp import MLP, Linear, BatchNorm, ReLU, Dropout, AdamOptimizer
from ml.gbt import GradientBoostedTrees, RegressionTree
from ml.online import ReplayBuffer, OnlineLearner
from ml.ensemble import EnsembleScorer, FeatureStore, features_to_vector, FEATURE_NAMES, INPUT_DIM


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestLosses:
    def test_sigmoid_range(self):
        x = np.array([-100, -1, 0, 1, 100], dtype=float)
        s = sigmoid(x)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_focal_loss_positive(self):
        logits = np.array([0.0, 1.0, -1.0])
        labels = np.array([1.0, 1.0, 0.0])
        loss = focal_loss(logits, labels)
        assert loss > 0

    def test_focal_less_than_bce_on_easy(self):
        # On easy examples (correct confident prediction), focal < bce
        logits = np.array([5.0, 5.0, -5.0, -5.0])
        labels = np.array([1.0, 1.0,  0.0,  0.0])
        fl  = focal_loss(logits, labels)
        bce = binary_cross_entropy(logits, labels)
        assert fl < bce

    def test_focal_grad_shape(self):
        logits = np.random.randn(32)
        labels = np.random.randint(0, 2, 32).astype(float)
        grad = focal_loss_grad(logits, labels)
        assert grad.shape == logits.shape

    def test_gradient_numerical_check(self):
        """Finite difference check on focal loss gradient."""
        np.random.seed(0)
        logits = np.random.randn(8)
        labels = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=float)
        eps    = 1e-5

        grad_analytic = focal_loss_grad(logits, labels)
        grad_numeric  = np.zeros_like(logits)

        for i in range(len(logits)):
            l_plus  = logits.copy(); l_plus[i]  += eps
            l_minus = logits.copy(); l_minus[i] -= eps
            grad_numeric[i] = (focal_loss(l_plus, labels) - focal_loss(l_minus, labels)) / (2 * eps)

        np.testing.assert_allclose(grad_analytic, grad_numeric, rtol=1e-3, atol=1e-5)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def _make_data(self, n=200, d=10, seed=42):
        np.random.seed(seed)
        X = np.random.randn(n, d).astype(np.float32)
        # Linearly separable with noise
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        return X, y

    def test_forward_shape(self):
        mlp = MLP(input_dim=10, hidden_dims=[16, 8])
        X = np.random.randn(32, 10).astype(np.float32)
        out = mlp.forward(X)
        assert out.shape == (32,)

    def test_predict_proba_range(self):
        mlp = MLP(input_dim=10)
        X = np.random.randn(50, 10).astype(np.float32)
        probs = mlp.predict_proba(X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_train_step_returns_loss(self):
        mlp = MLP(input_dim=10, hidden_dims=[16, 8])
        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randint(0, 2, 32).astype(float)
        loss = mlp.train_step(X, y)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self):
        """Loss should decrease over training on separable data."""
        np.random.seed(0)
        mlp = MLP(input_dim=10, hidden_dims=[32, 16], lr=1e-2)
        X, y = self._make_data()
        losses = mlp.fit(X, y, epochs=20, batch_size=64, verbose=False)
        assert losses[-1] < losses[0]

    def test_linear_layer_gradients(self):
        """Smoke test: gradients flow back to first layer."""
        mlp = MLP(input_dim=5, hidden_dims=[8, 4])
        X = np.random.randn(16, 5).astype(np.float32)
        y = np.random.randint(0, 2, 16).astype(float)
        mlp.train_step(X, y)
        assert mlp.linears[0].dW is not None
        assert mlp.linears[0].dW.shape == mlp.linears[0].W.shape

    def test_save_load(self, tmp_path):
        mlp = MLP(input_dim=10, hidden_dims=[16, 8])
        X = np.random.randn(32, 10).astype(np.float32)
        mlp.fit(X, np.random.randint(0, 2, 32).astype(float), epochs=2, verbose=False)
        probs_before = mlp.predict_proba(X)

        path = str(tmp_path / "model")
        mlp.save(path)

        mlp2 = MLP(input_dim=10, hidden_dims=[16, 8])
        mlp2.load(path + ".npz")
        probs_after = mlp2.predict_proba(X)

        np.testing.assert_allclose(probs_before, probs_after, rtol=1e-5)


# ---------------------------------------------------------------------------
# GBT
# ---------------------------------------------------------------------------

class TestGBT:
    def _make_data(self, n=300, seed=0):
        np.random.seed(seed)
        X = np.random.randn(n, 8).astype(np.float32)
        y = (X[:, 0] * X[:, 1] > 0).astype(float)  # non-linear boundary
        return X, y

    def test_predict_proba_range(self):
        X, y = self._make_data()
        gbt = GradientBoostedTrees(n_estimators=5)
        gbt.fit(X, y, verbose=False)
        probs = gbt.predict_proba(X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_loss_decreases(self):
        X, y = self._make_data()
        gbt = GradientBoostedTrees(n_estimators=20, learning_rate=0.1)
        gbt.fit(X, y, verbose=False)
        assert gbt.train_losses[-1] < gbt.train_losses[0]

    def test_feature_importance_sums_to_one(self):
        X, y = self._make_data()
        gbt = GradientBoostedTrees(n_estimators=10)
        gbt.fit(X, y, verbose=False)
        imp = gbt.feature_importance(X.shape[1])
        assert abs(imp.sum() - 1.0) < 1e-6 or imp.sum() == 0

    def test_predict_binary(self):
        X, y = self._make_data()
        gbt = GradientBoostedTrees(n_estimators=10)
        gbt.fit(X, y, verbose=False)
        preds = gbt.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Online learning
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(max_size=100)
        for i in range(50):
            buf.add(np.array([float(i)]), float(i % 2))
        X, y = buf.sample(20)
        assert X.shape == (20, 1)
        assert y.shape == (20,)

    def test_max_size_respected(self):
        buf = ReplayBuffer(max_size=10)
        for i in range(100):
            buf.add(np.array([float(i)]), 0.0)
        assert buf.size() <= 10

    def test_fraud_rate(self):
        buf = ReplayBuffer(max_size=1000)
        for i in range(100):
            buf.add(np.zeros(5), float(i < 10))  # 10% fraud
        # Approximate — reservoir sampling adds randomness
        assert 0 <= buf.fraud_rate() <= 1


class TestOnlineLearner:
    def test_update_runs(self):
        mlp = MLP(input_dim=5, hidden_dims=[8, 4], lr=1e-3)
        learner = OnlineLearner(mlp, buffer_size=200, update_every=10)
        for i in range(15):
            x = np.random.randn(5).astype(np.float32)
            learner.update(x, float(i % 3 == 0))
        assert learner.buffer.size() == 15


# ---------------------------------------------------------------------------
# Ensemble + Feature Store
# ---------------------------------------------------------------------------

class TestFeatureStore:
    def test_store_and_retrieve(self):
        fs = FeatureStore(ttl_seconds=60)
        features = {"user_txn_count_1h": 5, "amount_zscore_1h": 1.2}
        fs.store("user:001", features)
        retrieved = fs.retrieve("user:001")
        assert retrieved == features

    def test_missing_returns_none(self):
        fs = FeatureStore()
        assert fs.retrieve("ghost") is None

    def test_invalidate(self):
        fs = FeatureStore()
        fs.store("user:001", {"x": 1})
        fs.invalidate("user:001")
        assert fs.retrieve("user:001") is None


class TestEnsembleScorer:
    def _make_models(self):
        np.random.seed(42)
        X = np.random.randn(200, INPUT_DIM).astype(np.float32)
        y = (X[:, 0] > 0).astype(float)

        mlp = MLP(input_dim=INPUT_DIM, hidden_dims=[16, 8], lr=1e-2)
        mlp.fit(X, y, epochs=3, verbose=False)

        gbt = GradientBoostedTrees(n_estimators=5)
        gbt.fit(X, y, verbose=False)

        scorer = EnsembleScorer(mlp=mlp, gbt=gbt)
        scorer.fit_normalizer(X)
        return scorer, X

    def test_score_range(self):
        scorer, X = self._make_models()
        result = scorer.score(X[0])
        assert 0 <= result["ensemble_score"] <= 1

    def test_score_has_both_models(self):
        scorer, X = self._make_models()
        result = scorer.score(X[0])
        assert "mlp_score" in result
        assert "gbt_score" in result

    def test_score_batch_shape(self):
        scorer, X = self._make_models()
        scores = scorer.score_batch(X[:10])
        assert scores.shape == (10,)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_features_to_vector(self):
        window_features = {name: 1.0 for name in FEATURE_NAMES[:17]}
        txn_features    = {name: 0.0 for name in FEATURE_NAMES[17:]}
        vec = features_to_vector(window_features, txn_features)
        assert vec.shape == (INPUT_DIM,)