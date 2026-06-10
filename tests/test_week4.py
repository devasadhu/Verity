"""
Tests for Week 4 components:
  - KernelSHAP (explainability/shap.py)
  - ModelRegistry + AuditLog (explainability/registry.py)
  - PSIDriftDetector (explainability/drift.py)
  - AlertEngine (api/alerts.py)
  - AdversarialSimulator (adversarial/simulator.py)
"""

import math
import os
import sys
import time
import tempfile
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explainability.shap import KernelSHAP, top_features, _kernel_weight, _wls
from explainability.registry import ModelRegistry, AuditLog, _feature_hash
from explainability.drift import PSIDriftDetector, compute_psi, psi_label
from api.alerts import AlertEngine, AlertType, DedupCache, WebhookConfig
from adversarial.simulator import (
    AttackSuite, velocity_attack, device_farm_attack,
    low_and_slow_attack, amount_splitting_attack,
    geo_spoofing_attack, bot_generated_attack,
)


# ===========================================================================
# KernelSHAP tests (10 tests)
# ===========================================================================

class TestKernelWeight:
    def test_boundary_size_zero_returns_zero(self):
        z = np.zeros(5)
        assert _kernel_weight(5, z) == 0.0

    def test_boundary_size_full_returns_zero(self):
        z = np.ones(5)
        assert _kernel_weight(5, z) == 0.0

    def test_weight_is_positive_for_valid_coalition(self):
        z = np.array([1, 0, 1, 0, 1], dtype=float)
        assert _kernel_weight(5, z) > 0.0

    def test_weight_symmetric(self):
        """Coalition z and complement ~z should have same weight."""
        d = 6
        z = np.array([1, 1, 0, 0, 0, 0], dtype=float)
        z_comp = 1 - z
        assert abs(_kernel_weight(d, z) - _kernel_weight(d, z_comp)) < 1e-10


class TestWLS:
    def test_recovers_known_coefficients(self):
        """If y = Z @ phi_true, the solver should recover phi_true."""
        rng = np.random.default_rng(0)
        d = 5
        phi_true = rng.uniform(-1, 1, size=d)
        Z = rng.integers(0, 2, size=(50, d)).astype(float)
        y = Z @ phi_true
        w = np.ones(50)
        phi_hat = _wls(Z, y, w)
        np.testing.assert_allclose(phi_hat, phi_true, atol=1e-4)


class TestKernelSHAP:
    @pytest.fixture
    def explainer(self):
        rng = np.random.default_rng(42)
        d = 6
        background = rng.standard_normal((50, d))
        # Simple linear model: score = sum of absolute values, normalised
        def predict_fn(X):
            raw = np.abs(X).sum(axis=1)
            return raw / (raw.max() + 1e-8)
        return KernelSHAP(predict_fn, background, n_samples=128, random_state=0)

    def test_shap_values_shape(self, explainer):
        x = np.ones(6)
        result = explainer.explain(x)
        assert result["shap_values"].shape == (6,)

    def test_efficiency_constraint(self, explainer):
        """SHAP values must sum to prediction - baseline."""
        x = np.array([1.0, -2.0, 0.5, 1.5, -0.5, 2.0])
        result = explainer.explain(x)
        expected_sum = result["prediction"] - result["baseline"]
        actual_sum   = result["shap_values"].sum()
        assert abs(actual_sum - expected_sum) < 1e-4

    def test_contributions_sorted_by_abs(self, explainer):
        x = np.array([2.0, 0.1, 1.5, -1.8, 0.3, -0.5])
        result = explainer.explain(x)
        abs_shaps = [c["abs_shap"] for c in result["contributions"]]
        assert abs_shaps == sorted(abs_shaps, reverse=True)

    def test_explanation_string_present(self, explainer):
        x = np.ones(6)
        result = explainer.explain(x)
        assert isinstance(result["explanation"], str)
        assert "Risk score" in result["explanation"]

    def test_top_features_utility(self, explainer):
        x = np.ones(6)
        result = explainer.explain(x)
        tf = top_features(result, k=3)
        assert len(tf) == 3
        assert all(isinstance(name, str) for name, _ in tf)


# ===========================================================================
# ModelRegistry + AuditLog tests (10 tests)
# ===========================================================================

class TestModelRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return ModelRegistry(manifest_path=str(tmp_path / "registry.json"))

    def test_register_creates_version(self, registry, tmp_path):
        # Create a dummy artifact file so hash works
        art = tmp_path / "model.npz"
        art.write_bytes(b"fake_model")
        record = registry.register("mlp", str(art), "ieee-cis")
        assert record.version.startswith("mlp-")
        assert not record.is_active

    def test_activate_sets_active(self, registry, tmp_path):
        art = tmp_path / "model.npz"
        art.write_bytes(b"fake")
        record = registry.register("mlp", str(art), "ieee-cis")
        registry.activate(record.version)
        active = registry.get_active("mlp")
        assert active is not None
        assert active.version == record.version

    def test_activate_deactivates_previous(self, registry, tmp_path):
        art1 = tmp_path / "m1.npz"
        art2 = tmp_path / "m2.npz"
        art1.write_bytes(b"v1")
        art2.write_bytes(b"v2")
        r1 = registry.register("mlp", str(art1), "ds1")
        r2 = registry.register("mlp", str(art2), "ds2")
        registry.activate(r1.version)
        registry.activate(r2.version)
        assert registry.get(r1.version).is_active is False
        assert registry.get(r2.version).is_active is True

    def test_persist_and_reload(self, tmp_path):
        path = str(tmp_path / "reg.json")
        reg1 = ModelRegistry(manifest_path=path)
        art  = tmp_path / "m.npz"
        art.write_bytes(b"data")
        rec = reg1.register("gbt", str(art), "upi")
        reg2 = ModelRegistry(manifest_path=path)
        assert reg2.get(rec.version) is not None

    def test_update_metrics(self, registry, tmp_path):
        art = tmp_path / "m.npz"
        art.write_bytes(b"x")
        rec = registry.register("mlp", str(art), "ds")
        registry.update_metrics(rec.version, {"auc": 0.98, "f1": 0.87})
        updated = registry.get(rec.version)
        assert updated.train_metrics["auc"] == 0.98


class TestAuditLog:
    @pytest.fixture
    def audit(self, tmp_path):
        return AuditLog(log_path=str(tmp_path / "audit.jsonl"))

    def test_log_and_retrieve(self, audit):
        entry = audit.log(
            transaction_id="T123",
            model_version="mlp-20240601-ab12",
            feature_vector=[0.1] * 10,
            rule_score=0.3,
            ml_score=0.85,
            ensemble_score=0.72,
            decision="REVIEW",
            top_shap_features=[("amount_zscore", 0.4)],
            latency_ms=12.5,
        )
        entries = audit.get("T123")
        assert len(entries) == 1
        assert entries[0].decision == "REVIEW"

    def test_immutability(self, audit):
        """Verify we can't overwrite an existing log entry."""
        audit.log("T999", "v1", [0]*5, 0.1, 0.2, 0.15, "PASS", [], 5.0)
        original = audit.get("T999")[0].ensemble_score
        # Logging another entry for same txn should ADD, not replace
        audit.log("T999", "v1", [1]*5, 0.9, 0.95, 0.93, "BLOCK", [], 8.0)
        entries = audit.get("T999")
        assert len(entries) == 2
        assert entries[0].ensemble_score == original

    def test_feature_hash_deterministic(self):
        fv = [0.1, 0.2, 0.3]
        assert _feature_hash(fv) == _feature_hash(fv)

    def test_stats(self, audit):
        audit.log("T1", "v1", [0]*5, 0.1, 0.2, 0.15, "PASS",   [], 5.0)
        audit.log("T2", "v1", [0]*5, 0.8, 0.9, 0.85, "BLOCK",  [], 8.0)
        audit.log("T3", "v1", [0]*5, 0.6, 0.7, 0.65, "REVIEW", [], 6.0)
        s = audit.stats()
        assert s["total"] == 3
        assert s["by_decision"]["BLOCK"] == 1


# ===========================================================================
# PSI Drift Detector tests (8 tests)
# ===========================================================================

class TestComputePSI:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000)
        psi, _ = compute_psi(data, data)
        assert psi < 0.02

    def test_very_different_distributions_high_psi(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)   # shift by 5 std
        psi, _ = compute_psi(ref, cur)
        assert psi > 0.20

    def test_psi_label_stable(self):
        assert psi_label(0.05)  == "stable"
        assert psi_label(0.15)  == "monitor"
        assert psi_label(0.25)  == "alert"

    def test_reuses_bins(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        psi1, bins = compute_psi(ref, cur)
        psi2, _    = compute_psi(ref, cur, bins=bins)
        assert abs(psi1 - psi2) < 1e-10


class TestPSIDriftDetector:
    @pytest.fixture
    def detector(self):
        rng = np.random.default_rng(0)
        features = ["f0", "f1", "f2", "f3"]
        det = PSIDriftDetector(features, n_bins=5, window_size=50)
        X_train = rng.normal(0, 1, (200, 4))
        scores  = np.clip(rng.normal(0.3, 0.1, 200), 0, 1)
        det.fit_reference(X_train, scores)
        return det, rng

    def test_no_drift_stable(self, detector):
        det, rng = detector
        X_cur = rng.normal(0, 1, (100, 4))
        s_cur  = np.clip(rng.normal(0.3, 0.1, 100), 0, 1)
        report = det.check(X_cur, s_cur)
        # Stable distribution should not trigger alert
        assert report.score_status in ("stable", "monitor")

    def test_severe_drift_alert(self, detector):
        det, rng = detector
        # Shift distribution by 4 sigma → should alert
        X_drifted = rng.normal(4, 1, (100, 4))
        s_drifted  = np.clip(rng.normal(0.9, 0.05, 100), 0, 1)
        report = det.check(X_drifted, s_drifted)
        assert report.alert is True

    def test_save_load(self, detector, tmp_path):
        det, rng = detector
        path = str(tmp_path / "drift_ref.json")
        det.save_reference(path)
        det2 = PSIDriftDetector.load_reference(path)
        X_cur = rng.normal(0, 1, (50, 4))
        s_cur  = np.clip(rng.normal(0.3, 0.1, 50), 0, 1)
        report = det2.check(X_cur, s_cur)
        assert report.window_size == 50

    def test_buffer_push_and_check(self, detector):
        det, rng = detector
        X_cur = rng.normal(0, 1, (100, 4))
        s_cur  = np.clip(rng.normal(0.3, 0.1, 100), 0, 1)
        det.push_batch(X_cur, s_cur)
        assert det.buffer_size() == 100
        report = det.check()   # uses buffer
        assert det.buffer_size() == 0  # cleared after check


# ===========================================================================
# AlertEngine tests (8 tests)
# ===========================================================================

class TestDedupCache:
    def test_first_call_not_suppressed(self):
        cache = DedupCache(window_s=60)
        assert cache.should_suppress("u1", AlertType.AUTO_BLOCK) is False

    def test_second_call_suppressed(self):
        cache = DedupCache(window_s=60)
        cache.should_suppress("u1", AlertType.AUTO_BLOCK)
        assert cache.should_suppress("u1", AlertType.AUTO_BLOCK) is True

    def test_different_users_not_suppressed(self):
        cache = DedupCache(window_s=60)
        cache.should_suppress("u1", AlertType.AUTO_BLOCK)
        assert cache.should_suppress("u2", AlertType.AUTO_BLOCK) is False

    def test_different_alert_types_not_suppressed(self):
        cache = DedupCache(window_s=60)
        cache.should_suppress("u1", AlertType.AUTO_BLOCK)
        assert cache.should_suppress("u1", AlertType.REVIEW_FLAGGED) is False


class TestAlertEngine:
    def test_block_fires_alert(self):
        received = []
        engine = AlertEngine()
        engine.register_handler(received.append)
        engine.process("T1", "u1", 0.95, "BLOCK")
        assert len(received) == 1
        assert received[0].alert_type == AlertType.AUTO_BLOCK

    def test_review_fires_alert(self):
        received = []
        engine = AlertEngine()
        engine.register_handler(received.append)
        engine.process("T2", "u1", 0.75, "REVIEW")
        assert len(received) == 1
        assert received[0].alert_type == AlertType.REVIEW_FLAGGED

    def test_pass_does_not_fire(self):
        received = []
        engine = AlertEngine()
        engine.register_handler(received.append)
        engine.process("T3", "u1", 0.30, "PASS")
        assert len(received) == 0

    def test_dedup_suppresses_second_alert(self):
        received = []
        engine = AlertEngine(dedup_window_s=60)
        engine.register_handler(received.append)
        engine.process("T1", "u1", 0.95, "BLOCK")
        engine.process("T2", "u1", 0.96, "BLOCK")   # same user → suppressed
        assert len(received) == 1
        assert engine.stats()["suppressed"] == 1


# ===========================================================================
# Adversarial Simulator tests (10 tests)
# ===========================================================================

class TestAttackGenerators:
    def _collect(self, gen, max_n=100):
        return [t for t, _ in zip(gen, range(max_n))]

    def test_velocity_attack_generates_transactions(self):
        txns = self._collect(velocity_attack(n_txns=10))
        assert len(txns) == 10
        user_ids = {t["user_id"] for t in txns}
        assert len(user_ids) == 1   # all same user

    def test_device_farm_shared_devices(self):
        txns = self._collect(device_farm_attack(n_users=20, n_devices=2, txns_per_user=1))
        assert len(txns) == 20
        devices = {t["device_id"] for t in txns}
        assert len(devices) == 2

    def test_low_and_slow_escalates(self):
        txns = self._collect(low_and_slow_attack(n_days=4))
        amounts = [t["amount"] for t in txns]
        # Last quarter should be higher than first quarter on average
        n = len(amounts)
        assert sum(amounts[n*3//4:]) / len(amounts[n*3//4:]) > \
               sum(amounts[:n//4]) / len(amounts[:n//4])

    def test_amount_splitting_sums_to_target(self):
        target = 10_000.0
        txns = self._collect(amount_splitting_attack(target_amount=target, n_splits=5))
        total = sum(t["amount"] for t in txns)
        # Allow 1% tolerance for rounding
        assert abs(total - target) / target < 0.01

    def test_geo_spoofing_multiple_cities(self):
        txns = self._collect(geo_spoofing_attack(n_txns=10))
        cities = {t["city"] for t in txns}
        assert len(cities) >= 2

    def test_bot_generated_regular_timing(self):
        interval = 60.0
        txns = self._collect(bot_generated_attack(n_txns=20, interval_s=interval))
        timestamps = sorted(t["timestamp"] for t in txns)
        intervals  = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        std = (sum((x - interval)**2 for x in intervals) / len(intervals)) ** 0.5
        assert std < 0.01   # machine-perfect timing

    def test_all_transactions_have_required_fields(self):
        required = {"transaction_id", "user_id", "merchant_id", "amount",
                    "mcc", "city", "device_id", "timestamp"}
        for gen in [velocity_attack(5), device_farm_attack(5, 2, 1),
                    low_and_slow_attack(2), geo_spoofing_attack(5)]:
            for txn in gen:
                assert required.issubset(txn.keys())


class TestAttackSuite:
    def test_suite_runs_all_attacks(self):
        def mock_score(txn):
            # Simple rule: high amounts get blocked
            score = min(txn["amount"] / 10_000, 1.0)
            decision = "BLOCK" if score >= 0.9 else ("REVIEW" if score >= 0.7 else "PASS")
            return score, decision

        suite = AttackSuite(mock_score)
        report = suite.run()
        assert len(report.attacks) == 6
        assert 0.0 <= report.overall_detection_rate <= 1.0

    def test_report_to_dict(self):
        def mock_score(txn):
            return 0.5, "REVIEW"
        suite = AttackSuite(mock_score)
        report = suite.run()
        d = report.to_dict()
        assert "overall_detection_rate" in d
        assert len(d["attacks"]) == 6
        for attack in d["attacks"]:
            assert "detection_rate" in attack
            assert "block_rate" in attack
            assert "avg_score" in attack